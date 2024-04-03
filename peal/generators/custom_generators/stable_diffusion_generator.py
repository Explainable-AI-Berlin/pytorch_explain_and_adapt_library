import os
import types
import shutil
import copy
from pathlib import Path

import torch
import io
import blobfile as bf

from mpi4py import MPI
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor

from peal.data.dataset_factory import get_datasets
from peal.data.datasets import Image2MixedDataset
from peal.dependencies.ddpm_inversion.ddpm_inversion import DDPMInversion
from peal.dependencies.time.core.utils import load_tokens_and_embeddings
from peal.generators.interfaces import EditCapableGenerator
from peal.global_utils import load_yaml_config, embed_numberstring
from peal.dependencies.time.generate_ce import (
    generate_time_counterfactuals,
)
from peal.dependencies.time.get_predictions import get_predictions
from peal.dependencies.time.training import training


class StableDiffusion(EditCapableGenerator):
    def __init__(self, config, classifier_dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.classifier_dataset = copy.deepcopy(classifier_dataset)
        self.generator_dataset = None

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

    def initialize(self, classifier, base_path, explainer_config):
        class_predictions_path = os.path.join(base_path, "explainer", "predictions.csv")
        generator_dataset = get_datasets(self.config.data)[0]
        generator_dataset.task_config = self.classifier_dataset.task_config
        Path(os.path.join(base_path, "explainer")).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(class_predictions_path):
            generator_dataset.enable_url()
            prediction_args = types.SimpleNamespace(
                batch_size=32,
                dataset=generator_dataset,
                classifier=classifier,
                label_path=class_predictions_path,
                partition="train",
                label_query=0,
            )
            get_predictions(prediction_args)

        generator_dataset_config = copy.deepcopy(self.config.data)
        generator_dataset_config.split = [1.0, 1.0]
        self.generator_dataset = get_datasets(
            config=generator_dataset_config, data_dir=class_predictions_path
        )[0]
        context_embedding_path = os.path.join(
            base_path, "explainer", "context_embedding"
        )
        if not os.path.exists(context_embedding_path):
            train_context_embedding_args = types.SimpleNamespace(
                sd_model="CompVis/stable-diffusion-v1-4",
                embedding_files=[],
                output_path=context_embedding_path,
                dataset=self.generator_dataset,
                partition="train",
                phase="context",
                batch_size=explainer_config.train_batch_size,
                **explainer_config.__dict__
            )
            training(train_context_embedding_args)

        # TODO how to extend this for multiclass??
        for class_idx in range(2):
            class_token_path = os.path.join(
                base_path, "explainer", "class_token" + str(class_idx)
            )
            if not os.path.exists(class_token_path):
                class_related_bias_embedding_args = types.SimpleNamespace(
                    embedding_files=[context_embedding_path],
                    output_path=class_token_path,
                    dataset=self.generator_dataset,
                    custom_tokens=explainer_config.class_custom_token[class_idx],
                    training_label=class_idx,
                    phase="class",
                    batch_size=explainer_config.train_batch_size,
                    **explainer_config.__dict__
                )
                training(class_related_bias_embedding_args)

        if explainer_config.editing_type == "ddpm_inversion":
            # TODO somehow the config should be possible to influence
            self.editor = DDPMInversion()
            embedding_files=[
                os.path.join(base_path, "explainer", "context_embedding"),
                os.path.join(base_path, "explainer", "class_token0"),
                os.path.join(base_path, "explainer", "class_token1"),
            ]
            load_tokens_and_embeddings(sd_model=self.editor.pipe, files=embedding_files)

        else:
            self.editor = None

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        classifier: nn.Module,
        explainer_config,
        classifier_dataset,
        pbar=None,
        mode="",
        base_path="",
    ):
        if self.generator_dataset is None:
            self.initialize(classifier, base_path, explainer_config)

        classifier_to_generator = (
            lambda x: self.generator_dataset.project_from_pytorch_default(
                self.classifier_dataset.project_to_pytorch_default(x)
            )
        )
        generator_to_classifier = (
            lambda x: self.classifier_dataset.project_from_pytorch_default(
                self.generator_dataset.project_to_pytorch_default(x)
            )
        )
        dataset = [
            (
                torch.zeros([len(x_in)], dtype=torch.long),
                classifier_to_generator(x_in),
                [source_classes, target_classes],
            )
        ]
        print("[x_in.min(), x_in.max()]")
        print([x_in.min(), x_in.max()])
        print([x_in.min(), x_in.max()])
        print([x_in.min(), x_in.max()])
        ce_generation_args = types.SimpleNamespace(
            embedding_files=[
                os.path.join(base_path, "explainer", "context_embedding"),
                os.path.join(base_path, "explainer", "class_token0"),
                os.path.join(base_path, "explainer", "class_token1"),
            ],
            postprocess=lambda x, size: self.generator_dataset.project_to_pytorch_default(
                x
            ),
            dataset=dataset,
            classifier=classifier,
            output_path=os.path.join(base_path, "explainer", "outputs"),
            partition="val",
            batch_size=explainer_config.inference_batch_size,
            neg_custom_token=explainer_config.class_custom_token[0],
            pos_custom_token=explainer_config.class_custom_token[1],
            editor=self.editor,
            **explainer_config.__dict__
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

        x_counterfactuals = generator_to_classifier(torch.cat(x_counterfactuals, dim=0))
        print("[x_counterfactuals.min(), x_counterfactuals.max()]")
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        device = [p for p in classifier.parameters()][0].device
        preds = torch.nn.Softmax(dim=-1)(
            classifier(x_counterfactuals.to(device)).detach().cpu()
        )

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        return (
            list(x_counterfactuals.cpu()),
            list(x_in - x_counterfactuals.cpu()),
            list(y_target_end_confidence),
            list(x_in),
        )
