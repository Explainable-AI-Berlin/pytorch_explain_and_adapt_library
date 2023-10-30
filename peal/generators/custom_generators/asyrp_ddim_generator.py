import torch
import torchvision
import os
import types
import shutil
import copy

from pathlib import Path
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor

from peal.generators.interfaces import EditCapableGenerator
from peal.data.datasets import Image2ClassDataset
from peal.global_utils import load_yaml_config, embed_numberstring
from run_asyrp import asyrp_main
from ace.guided_diffusion import dist_util, logger
from ace.guided_diffusion.resample import create_named_schedule_sampler
from ace.guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from ace.guided_diffusion.train_util import TrainLoop
from peal.data.dataset_factory import get_datasets
from peal.data.dataloaders import get_dataloader
from dime2.core.dist_util import (
    load_state_dict,
)


class AsyrpDDIMAdaptor(EditCapableGenerator):
    def __init__(self, config, dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.dataset = (
            dataset if not dataset is None else get_datasets(self.config.data)[0]
        )
        if not self.config.image_size is None:
            self.config.image_size = self.dataset.config.input_size[-1]

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals")

    def sample_x(self, batch_size=1):
        pass

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
        shutil.rmtree(self.data_dir, ignore_errors=True)
        shutil.rmtree(self.counterfactual_path, ignore_errors=True)
        self.dataset.serialize_dataset(
            output_dir=self.data_dir,
            x_list=x_in,
            y_list=source_classes,
            sample_names=list(
                map(lambda x: embed_numberstring(str(x)) + ".jpg", range(x_in.shape[0]))
            ),
        )

        args = copy.deepcopy(self.config)
        args.dataset = Image2ClassDataset(
            root_dir=self.data_dir,
            mode=None,
            config=copy.deepcopy(self.dataset.config),
            transform=self.dataset.transform,
        )
        args.model_path = os.path.join(self.model_dir, "final.pt")
        args.classifier = classifier
        args.exp = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        x_counterfactuals = asyrp_main(args=args)
        x_list = x_in

        x_counterfactuals = torch.stack(x_counterfactuals)
        x_counterfactuals = self.dataset.project_from_pytorch_default(x_counterfactuals)
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
            list(x_list),
        )
