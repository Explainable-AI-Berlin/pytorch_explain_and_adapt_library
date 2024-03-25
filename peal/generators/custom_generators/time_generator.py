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
from peal.dependencies.time.generate_ce import (
    generate_time_counterfactuals,
)
from peal.dependencies.time.get_predictions import get_predictions
from peal.dependencies.time.training import training


class StableDiffusion(EditCapableGenerator):
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
        self.initialized = False

    def sample_x(self, batch_size=1):
        return self.diffusion.p_sample_loop(
            self.model, [batch_size] + self.classifier_dataset.config.input_size
        )

    def initialize(self, classifier):
        prediction_args = types.SimpleNamespace(
            batch_size=32,
            dataset=self.classifier_dataset,
            classifier=classifier,
            label_path=os.path.join(self.base_path, "class_predictions"),
        )
        get_predictions(prediction_args)
        train_context_embedding_args = types.SimpleNamespace(
            output_path=os.path.join(self.base_path, "context_embedding"),
            dataset=self.classifier_dataset,
            label_query=0,
            training_label=-1,
            custom_tokens="'|<C*1>|' '|<C*2>|' '|<C*3>|'",
            custom_tokens_init="centered realistic celebrity",
            phase="context",
            mini_batch_size=1,
            enable_xformers_memory_efficient_attention=True,
        )
        training(train_context_embedding_args)
        for class_idx in range(self.classifier_dataset.config.num_classes):
            class_related_bias_embedding_args = types.SimpleNamespace(
                embedding_files=os.path.join(self.base_path, "context_embedding"),
                output_path=os.path.join(
                    self.base_path, "class_token" + str(class_idx)
                ),
                dataset=self.classifier_dataset,
                label_query=0,
                training_label=class_idx,
                custom_tokens="'|<A*"
                + str(class_idx)
                + "1>|' '|<A*"
                + str(class_idx)
                + "2>|' '|<A*"
                + str(class_idx)
                + "3>|'",
                custom_tokens_init="centered realistic celebrity",
                phase="class",
                mini_batch_size=1,
                base_prompt="A |<C*1>| |<C*2>| |<C*3>| photo",
                enable_xformers_memory_efficient_attention=True,
            )
            training(class_related_bias_embedding_args)

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        classifier: nn.Module,
        pbar=None,
        mode="",
        **kwargs
    ):
        if not self.initialized:
            self.initialize(classifier)
            self.initialized = True

        ce_generation_args = types.SimpleNamespace(
            embedding_files=[
                os.path.join(self.base_path, "context_embedding"),
                os.path.join(self.base_path, "class_token0"),
                os.path.join(self.base_path, "class_token1"),
            ],
            use_negative_guidance_denoise=True,
            use_negative_guidance_inverse=True,
            guidance_scale_denoising="4 4 4 4",
            guidance_scale_invertion="4 4 4 4",
            num_inference_steps="15 20 25 35",
            output_path=self.counterfactual_path,
            exp_name="time",
            label_target=-1,
            label_query=31,
            neg_custom_token='|<A*01>| |<A*02>| |<A*03>|',
            pos_custom_token='|<A*11>| |<A*12>| |<A*13>|',
            base_prompt="A |<C*1>| |<C*2>| |<C*3>| photo",
            chunks=1,
            chunk=0,
            enable_xformers_memory_efficient_attention=True,
            partition="val",
            dataset=self.classifier_dataset,
            classifier=classifier,
        )
        x_counterfactuals = generate_time_counterfactuals(ce_generation_args)

        """dataset = [
            (
                torch.zeros([len(x_in)], dtype=torch.long),
                x_in,
                [source_classes, target_classes],
            )
        ]
        args = copy.deepcopy(self.config)
        args.dataset = dataset
        args.model_path = os.path.join(self.model_dir, "final.pt")
        args.classifier = classifier
        args.diffusion = self.diffusion
        args.model = self.model
        args.output_path = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        x_counterfactuals = time_main(args=args)"""

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
