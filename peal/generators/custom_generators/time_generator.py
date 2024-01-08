import os
import types
import shutil
import copy
import torch
import io
import blobfile as bf

from mpi4py import MPI
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor

from peal.generators.interfaces import EditCapableGenerator
from peal.global_utils import load_yaml_config, embed_numberstring
from peal.dependencies.time.generate_ce import main as time_main
from peal.dependencies.time.get_predictions import get_predictions
from peal.dependencies.time.training import training

class TimeAdaptor(EditCapableGenerator):
    def __init__(self, config, dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.classifier_dataset = dataset

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data_test")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals_test")

    def sample_x(self, batch_size=1):
        return self.diffusion.p_sample_loop(
            self.model, [batch_size] + self.classifier_dataset.config.input_size
        )

    def edit(
            self,
            x_in: torch.Tensor,
            target_confidence_goal: float,
            source_classes: torch.Tensor,
            target_classes: torch.Tensor,
            classifier: nn.Module,
            pbar=None,
            mode="",
    ):
        dataset = [(
            torch.zeros([len(x_in)], dtype=torch.long),
            x_in,
            [source_classes, target_classes]
        )]
        args = copy.deepcopy(self.config)
        args.dataset = dataset
        args.model_path = os.path.join(self.model_dir, "final.pt")
        args.classifier = classifier
        args.diffusion = self.diffusion
        args.model = self.model
        args.output_path = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        x_counterfactuals = time_main(args=args)
        x_counterfactuals = torch.cat(x_counterfactuals, dim=0)

        device = [p for p in classifier.parameters()][0].device
        preds = torch.nn.Softmax(dim=-1)(
            classifier(x_counterfactuals.to(device)).detach().cpu()
        )

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        return (
            list(x_counterfactuals),
            list(x_in - x_counterfactuals),
            list(y_target_end_confidence),
            list(x_in),
        )