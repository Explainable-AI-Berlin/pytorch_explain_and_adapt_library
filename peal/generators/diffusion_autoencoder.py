import os
import types
import copy
import torch

from pathlib import Path
from torch import nn
from typing import Union

from peal.dependencies.diffusion_regression_counterfactuals.src.related_work.diffae.experiment import LitModel
from peal.dependencies.diffusion_regression_counterfactuals.src.related_work.diffae.templates import square64_autoenc
from peal.data.dataset_factory import get_datasets
from peal.generators.interfaces import EditCapableGenerator, InvertibleGenerator
from peal.global_utils import load_yaml_config, save_yaml_config
from peal.generators.interfaces import GeneratorConfig
from peal.data.interfaces import DataConfig
from peal.architectures.interfaces import TaskConfig
from peal.training.trainers import distill_predictor


class DiffusionAutoencoderConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "DiffusionAutoencoder"
    base_path: str = "/home/space/datasets/peal/peal_runs/diffusion_autoencoder"


class DiffusionAutoencoder(InvertibleGenerator, EditCapableGenerator):
    def __init__(self, config, predictor_dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.predictor_dataset = copy.deepcopy(predictor_dataset)
        # TODO something is wrong here!!!
        self.train_dataset = get_datasets(self.config.data)[0]
        if not self.config.task_config is None:
            self.train_dataset.task_config = self.config.task_config

        elif not self.predictor_dataset is None:
            self.train_dataset.task_config = self.predictor_dataset.task_config

        self.generator_dataset = None

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data_test")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals_test")
        conf = square64_autoenc()
        if os.path.exists(self.config.checkpoint_path):
            self.model = LitModel.load_from_checkpoint(
                checkpoint_path=self.config.checkpoint_path, conf=conf, map_location="cpu"
            )

        else:
            self.model = LitModel(conf)

    def sample_z(self, batch_size=1):
        # TODO this has to be done properly!!!
        z_sem = torch.randn(batch_size, 4, 8, 8)
        xT = torch.randn(batch_size, 3, 64, 64)
        return z_sem, xT

    def encode(self, x, t=1.0):
        z_sem: torch.Tensor = self.model.encode(x)
        # TODO why is t not used here???
        xT: torch.Tensor = self.model.encode_stochastic(x, z_sem)
        return z_sem, xT

    def decode(self, z, t=1.0):
        z_sem, xT = z
        # return self.model.render(xT, z_sem, T=self.backward_t, grads=True)
        return self.model.render(xT, z_sem, T=t, grads=True)

    def train_model(
        self,
    ):
        # write the yaml config on disk
        if not os.path.exists(self.config.base_path):
            Path(self.config.base_path).mkdir(parents=True, exist_ok=True)

        save_yaml_config(self.config, os.path.join(self.config.base_path, "config.yaml"))
        finetune_args = types.SimpleNamespace(**self.config.__dict__)
        finetune_args.train_dataset = self.train_dataset
        finetune_args.pipeline = self.pipeline
        finetune_args.resume_from_checkpoint = "latest"
        finetune_args.img_semantic_encoder = self.img_semantic_encoder
        # TODO add actual training here

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        predictor: nn.Module,
        explainer_config: dict,
        predictor_datasets: list,
        pbar=None,
        base_path: str = "",
        mode: str = "",
    ):
        if not explainer_config.distilled_predictor is None:
            distilled_path = os.path.join(base_path, "explainer", "distilled_predictor", "model.cpl")
            if not os.path.exists(distilled_path):
                self.gradient_predictor = distill_predictor(
                    explainer_config.distilled_predictor,
                    base_path,
                    predictor,
                    predictor_datasets,
                    predictor_distilled=nn.Sequential(
                        [
                            self.model.encoder,
                            nn.Linear(self.model.encoder.output_dimensions, self.predictor_dataset.output_size),
                        ]
                    ),  # TODO fix this!
                    only_last_layer=True,
                    continue_training=True,
                )

            else:
                self.gradient_predictor = torch.load(distilled_path, map_location=self.device)

        else:
            self.gradient_predictor = predictor

        classifier_to_generator = lambda x: self.generator_dataset.project_from_pytorch_default(
            self.predictor_dataset.project_to_pytorch_default(x)
        )
        generator_to_classifier = lambda x: self.predictor_dataset.project_from_pytorch_default(
            self.generator_dataset.project_to_pytorch_default(x)
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
        z_sem, xT = self.encode(x_in.to(self.device))
        z_sem2 = self._calculate_z_counterfactuals(z_sem)
        x_counterfactuals = self.decode((z_sem2, xT))
        print("[x_counterfactuals.min(), x_counterfactuals.max()]")
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        device = [p for p in predictor.parameters()][0].device
        preds = torch.nn.Softmax(dim=-1)(predictor(x_counterfactuals.to(device)).detach().cpu())

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        return (
            list(x_counterfactuals.cpu()),
            list(x_in - x_counterfactuals.cpu()),
            list(y_target_end_confidence),
            list(x_in),
        )

    def _calculate_z_counterfactuals(self, z_sem: torch.Tensor) -> torch.Tensor:
        # TODO this has to be implemented properly!!!
        return z_sem
