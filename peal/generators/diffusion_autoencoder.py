import os
import types
import copy
import torch

from pathlib import Path
from torch import nn
from typing import Union

from peal.dependencies.diffusion_regression_counterfactuals.src.related_work.diffae.experiment import LitModel
from peal.dependencies.diffusion_regression_counterfactuals.src.related_work.diffae.templates_latent import (
    square64_autoenc,
    train,
    square64_autoenc_latent,
)
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
    """
    The config of the data.
    """
    data: DataConfig = DataConfig()
    """
    The task config for the diffusion autoencoder.
    """
    task_config: Union[TaskConfig, None] = None
    checkpoint_path: str = "peal_runs/diffusion_autoencoder/final.ckpt"
    encoder_dimensions: int = 512
    save_every_samples: int = 20000
    total_samples: int = 40000000
    batch_size: int = 20
    encoder_path: Union[str, None] = None
    is_torchvision_resnet: bool = False
    is_loaded: bool = False


class DiffusionAutoencoder(InvertibleGenerator, EditCapableGenerator):
    def __init__(self, config, predictor_dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        # check if cuda device is available and assign to self.device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor_dataset = copy.deepcopy(predictor_dataset)
        # TODO something is wrong here!!!
        self.generator_datasets = get_datasets(self.config.data)
        if not self.config.task_config is None:
            self.generator_datasets[0].task_config = self.config.task_config
            self.generator_datasets[1].task_config = self.config.task_config

        elif not self.predictor_dataset is None:
            self.generator_datasets[0].task_config = self.predictor_dataset.task_config
            self.generator_datasets[1].task_config = self.predictor_dataset.task_config

        self.generator_dataset = self.generator_datasets[0]

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data_test")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals_test")
        conf = square64_autoenc()
        self.adjust_config(conf)
        self.checkpoint_path = os.path.join(self.config.base_path, "square64_ddim", "last.ckpt")
        if os.path.exists(self.checkpoint_path) and self.config.is_loaded:
            self.model = LitModel.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path, conf=conf, map_location="cpu"
            )
            self.model.to(self.device)

        else:
            self.model = None  # LitModel(conf)

    def sample_z(self, batch_size=1):
        # TODO this has to be done properly with the learned prior!!!
        z_sem = torch.randn(batch_size, self.config.encoder_dimensions).to(self.device)
        xT = torch.randn([batch_size] + self.config.data.input_size).to(self.device)
        return z_sem, xT

    def encode(self, x, t=1.0):
        z_sem: torch.Tensor = self.model.encode(x)
        # TODO why is t not used here???
        xT: torch.Tensor = self.model.encode_stochastic(x, z_sem)
        return z_sem, xT

    def decode(self, z, t=1.0):
        z_sem, xT = z
        # return self.model.render(xT, z_sem, T=self.backward_t, grads=True)
        # return self.model.render(xT, z_sem, T=t, grads=True)
        return self.model.render(xT, z_sem, grads=True)

    def adjust_config(self, conf):
        if not self.config.encoder_path is None:
            encoder = torch.load(self.config.encoder_path, map_location="cpu")
            if self.config.is_torchvision_resnet:
                # remove the head
                encoder.model.fc = nn.Identity()

            # nn module that normalizes with self.generator_dataset.config.normalization
            class NormalizationModule(nn.Module):
                def __init__(self, mean, std):
                    super().__init__()
                    self.mean = torch.tensor(mean)
                    self.std = torch.tensor(std)

                def forward(self, x):
                    return (x - self.mean.to(x.device)) / self.std.to(x.device)

            normalization = NormalizationModule(
                self.generator_dataset.config.normalization[0], self.generator_dataset.config.normalization[1]
            )
            # include the normalization directly into the encoder
            encoder = torch.nn.Sequential(normalization, encoder)

        else:
            encoder = None


        conf.base_dir = self.config.base_path
        conf.dataset = self.generator_dataset
        conf.encoder = encoder
        conf.img_size = self.generator_dataset.config.input_size[-1]
        conf.model_conf.image_size = self.generator_dataset.config.input_size[-1]
        conf.batch_size = self.config.batch_size
        conf.save_every_samples = self.config.save_every_samples
        conf.total_samples = self.config.total_samples
        self.generator_dataset.return_dict = True
        self.generator_dataset.idx_enabled = True

    def train_model(
        self,
    ):
        # write the yaml config on disk
        if not os.path.exists(self.config.base_path):
            Path(self.config.base_path).mkdir(parents=True, exist_ok=True)

        save_yaml_config(self.config, os.path.join(self.config.base_path, "config.yaml"))
        # finetune_args = types.SimpleNamespace(**self.config.__dict__)
        #
        conf = square64_autoenc()
        self.adjust_config(conf)
        train(conf)
        conf.eval_programs = ["infer"]
        # DHA: Assume pretrained. The model is loaded in eval mode.
        train(conf, mode="eval")

        # NOTE: a lot of gpus can speed up this process
        conf = square64_autoenc_latent(self.config.base_path)
        self.adjust_config(conf)
        train(conf)

        self.model = LitModel.load_from_checkpoint(checkpoint_path=self.checkpoint_path, conf=conf, map_location="cpu")
        self.model.to(self.device)

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
        device = [p for p in predictor.parameters()][0].device
        if not explainer_config.distilled_predictor is None:
            # assert explainer_config.distilled_predictor.task.output_channels == 1
            distilled_path = os.path.join(base_path, "explainer", "distilled_predictor", "model.cpl")
            if not os.path.exists(distilled_path):
                self.gradient_predictor = distill_predictor(
                    explainer_config.distilled_predictor,
                    os.path.join(base_path, "explainer"),
                    predictor,
                    self.generator_datasets,
                    predictor_distilled=nn.Sequential(
                        *[
                            self.model.ema_model.encoder,
                            nn.Linear(self.config.encoder_dimensions, 1, bias=False),
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
        print("[x_in.min(), x_in.max()]")
        print([x_in.min(), x_in.max()])
        print([x_in.min(), x_in.max()])
        print([x_in.min(), x_in.max()])

        preds = torch.nn.Softmax(dim=-1)(predictor(x_in.to(device)).detach().cpu())
        print("preds_before: " + str(preds.argmax(dim=-1)))

        x_generator = classifier_to_generator(x_in)
        x_generator = 2 * x_generator
        # x_generator = x_in
        print("[x_generator.min(), x_generator.max()]")
        print([x_generator.min(), x_generator.max()])
        print([x_generator.min(), x_generator.max()])
        print([x_generator.min(), x_generator.max()])
        z_sem, xT = self.encode(x_generator.to(self.device))
        w = list(self.gradient_predictor.children())[-1].weight[0]
        dot_ab = torch.tensor(torch.sum(z_sem * w, dim=-1, keepdim=True) > 0, dtype=torch.uint8)
        print("dot_ab before editing:", dot_ab.squeeze().detach().cpu().numpy())
        z_sem2 = self._calculate_z_counterfactuals(z_sem)
        dot_ab2 = torch.tensor(torch.sum(z_sem2 * w, dim=-1, keepdim=True) > 0, dtype=torch.uint8)
        print("dot_ab after editing:", dot_ab2.squeeze().detach().cpu().numpy())
        x_counterfactuals_generator = self.decode((z_sem2, xT))
        z_sem3, _ = self.encode(x_counterfactuals_generator.to(self.device))
        dot_ab3 = torch.tensor(torch.sum(z_sem3 * w, dim=-1, keepdim=True) > 0, dtype=torch.uint8)
        print("dot_ab3:", dot_ab3.squeeze().detach().cpu().numpy())
        # x_counterfactuals_generator = x_generator
        print("[x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]")
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        # x_counterfactuals = generator_to_classifier(x_counterfactuals_generator.cpu())
        x_counterfactuals = x_counterfactuals_generator
        print("[x_counterfactuals.min(), x_counterfactuals.max()]")
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        print([x_counterfactuals.min(), x_counterfactuals.max()])

        preds = torch.nn.Softmax(dim=-1)(predictor(x_counterfactuals.to(device)).detach().cpu())
        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        print("preds_after:", preds.argmax(dim=-1))
        import pdb

        pdb.set_trace()

        return (
            list(x_counterfactuals.cpu()),
            list(x_in - x_counterfactuals.cpu()),
            list(y_target_end_confidence),
            list(x_in),
            [],
        )

    def _calculate_z_counterfactuals(self, z_sem: torch.Tensor) -> torch.Tensor:
        # get the last module from the nn.Sequential self.gradient_predictor
        last_layer = list(self.gradient_predictor.children())[-1]
        # get weight from last layer
        b = last_layer.weight[0]
        a = z_sem
        #
        dot_ab = torch.sum(a * b, dim=-1, keepdim=True)  # shape (batch, 1)
        dot_bb = torch.sum(b * b)  # scalar

        # projection and reflection
        proj = dot_ab / dot_bb * b  # shape (batch, n)
        # reflected = 2 * proj - a
        reflected = a - 2 * proj
        # import pdb; pdb.set_trace()
        return reflected
