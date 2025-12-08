import math
import os
import shutil
import types
import copy
from datetime import time, datetime

import torch

from pathlib import Path

import torchvision
from torch import nn
from typing import Union

from transformers import AutoModel, AutoImageProcessor

from peal.data.dataloaders import WeightedDataloaderList
from peal.dependencies.diffusion_regression_counterfactuals.src.related_work.diffae.experiment import (
    LitModel,
)
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
from peal.sparse_dictionaries.singular_value_decomposition import (
    SVDDictionary,
    SVDDictionaryConfig,
)
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
    is_loaded: bool = True
    model_type: Union[str, None] = None
    sparse_dictionary: Union[str, SVDDictionaryConfig, None] = SVDDictionaryConfig()


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

        self.set_encoder()
        self.checkpoint_path = os.path.join(
            self.config.base_path, "square64_ddim", "last.ckpt"
        )
        self.checkpoint_path_latent = os.path.join(
            self.config.base_path, "square64_autoenc_latent", "last.ckpt"
        )
        self.load_models()

        if not self.config.sparse_dictionary is None:
            self.sparse_dictionary_path = os.path.join(
                self.config.base_path,
                self.config.sparse_dictionary.sparse_dictionary_type
                + self.config.sparse_dictionary.ending,
            )
            if not os.path.exists(self.sparse_dictionary_path):
                self.sparse_dictionary = None

            else:
                self.sparse_dictionary = SVDDictionary(
                    self.config.sparse_dictionary.sparse_dictionary_type
                )
                self.sparse_dictionary.load_from_disk(self.sparse_dictionary_path)

    def sample_z(self, batch_size=1):
        # TODO this has to be done properly with the learned prior!!!
        z_sem = torch.randn(batch_size, self.config.encoder_dimensions).to(self.device)
        xT = torch.randn([batch_size] + self.config.data.input_size).to(self.device)
        return z_sem, xT

    def sample_x(self, batch_size=1):
        return self.latent_model.sample(N=batch_size, device=self.device)

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

    def set_encoder(self):
        if not self.config.model_type is None:
            if self.config.model_type[: len("dino_v2")] == "dino_v2":
                if self.config.model_type == "dino_v2_small":
                    model = AutoModel.from_pretrained("facebook/dinov2-small")
                    processor = AutoImageProcessor.from_pretrained(
                        "facebook/dinov2-small"
                    )

                elif self.config.model_type == "dino_v2_base":
                    model = AutoModel.from_pretrained("facebook/dinov2-base")
                    processor = AutoImageProcessor.from_pretrained(
                        "facebook/dinov2-base"
                    )

                elif self.config.model_type == "dino_v2":
                    model = AutoModel.from_pretrained("facebook/dinov2-large")
                    processor = AutoImageProcessor.from_pretrained(
                        "facebook/dinov2-large"
                    )

                class DinoV2(nn.Module):
                    def __init__(self, model, processor):
                        super().__init__()
                        self.model = model
                        self.processor = processor

                    def forward(self, x):
                        cs = self.processor.crop_size
                        x_resized = torchvision.transforms.Resize(
                            [cs["height"], cs["width"]]
                        )(x)

                        def pv(v):
                            v = torch.tensor(v).to(x_resized)[:, None, None]
                            return torch.tile(v, [1, cs["height"], cs["width"]])

                        x_processed = (x_resized - pv(self.processor.image_mean)) / pv(
                            self.processor.image_std
                        )
                        latent_code = self.model(x_processed)["last_hidden_state"][:, 0]
                        return latent_code

                encoder = DinoV2(model, processor)

            elif self.config.model_type == "UNI":
                import timm
                from timm.data import resolve_data_config
                from timm.data.transforms_factory import create_transform
                from huggingface_hub import login

                login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

                # pretrained=True needed to load UNI weights (and download weights for the first time)
                # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
                model = timm.create_model(
                    "hf-hub:MahmoodLab/uni",
                    pretrained=True,
                    init_values=1e-5,
                    dynamic_img_size=True,
                )
                transform = create_transform(
                    **resolve_data_config(model.pretrained_cfg, model=model)
                )

                class UNI(nn.Module):
                    def __init__(self, model, transform):
                        super().__init__()
                        self.model = model
                        self.transform = transform

                    def forward(self, x):
                        x_processed = self.transform(x)
                        latent_code = self.model(x_processed)
                        return latent_code

                encoder = UNI(model, transform)

        elif not self.config.encoder_path is None:
            encoder = torch.load(self.config.encoder_path, map_location="cpu")
            if self.config.is_torchvision_resnet:
                # remove the head
                encoder.model.fc = nn.Identity()

            else:
                encoder.fc = nn.Identity()

        else:
            encoder = None

        if not encoder is None:
            # nn module that normalizes with self.generator_dataset.config.normalization

            normalization = NormalizationModule(
                self.generator_dataset.config.normalization[0],
                self.generator_dataset.config.normalization[1],
            )
            # include the normalization directly into the encoder
            self.encoder = torch.nn.Sequential(normalization, encoder)

        else:
            self.encoder = None

    def adjust_config(self, conf):
        conf.base_dir = self.config.base_path
        conf.dataset = self.generator_dataset
        conf.img_size = self.generator_dataset.config.input_size[-1]
        conf.model_conf.image_size = self.generator_dataset.config.input_size[-1]
        conf.batch_size = self.config.batch_size
        conf.save_every_samples = self.config.save_every_samples
        conf.total_samples = self.config.total_samples
        conf.style_ch = self.config.encoder_dimensions
        conf.net_beatgans_embed_channels = self.config.encoder_dimensions
        conf.embed_channels = self.config.encoder_dimensions
        conf.enc_out_channels = self.config.encoder_dimensions
        conf.encoder = self.encoder

    def train_model(
        self,
    ):
        return_dict_buffer = self.generator_dataset.return_dict
        idx_enabled_buffer = self.generator_dataset.idx_enabled
        self.generator_dataset.return_dict = True
        self.generator_dataset.idx_enabled = True
        # write the yaml config on disk
        if not os.path.exists(self.config.base_path):
            Path(self.config.base_path).mkdir(parents=True, exist_ok=True)

        self.config.is_loaded = True
        save_yaml_config(
            self.config, os.path.join(self.config.base_path, "config.yaml")
        )
        # finetune_args = types.SimpleNamespace(**self.config.__dict__)
        #
        conf = square64_autoenc()
        self.adjust_config(conf)
        train(conf)
        conf.eval_programs = ["infer"]
        # DHA: Assume pretrained. The model is loaded in eval mode.
        train(conf, mode="eval")

        # NOTE: a lot of gpus can speed up this process
        latent_conf = square64_autoenc_latent(os.path.join(self.config.base_path, "square64_ddim"))
        self.adjust_config(latent_conf)
        train(latent_conf)

        self.generator_dataset.return_dict = return_dict_buffer
        self.generator_dataset.idx_enabled = idx_enabled_buffer
        self.load_models()

        # analyze the encoder components
        if not self.config.sparse_dictionary is None:
            # TODO we should measure average activation when active to have reference point in SAEs
            self.fit_sparse_dictionary()

    def load_models(self):
        conf = square64_autoenc()
        self.adjust_config(conf)
        if os.path.exists(self.checkpoint_path):
            if self.config.is_loaded:
                self.model = LitModel.load_from_checkpoint(
                    checkpoint_path=self.checkpoint_path, conf=conf, map_location="cpu"
                )
                self.model.to(self.device)

            else:
                shutil.move(
                    self.config.base_path,
                    self.config.base_path
                    + "_old_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S"),
                )

        else:
            self.model = None  # LitModel(conf)

        latent_conf = square64_autoenc_latent(os.path.join(self.config.base_path, "square64_ddim"))
        self.adjust_config(latent_conf)
        if os.path.exists(self.checkpoint_path_latent):
            self.latent_model = LitModel.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path_latent, conf=latent_conf, map_location="cpu"
            )
            self.latent_model.to(self.device)

        else:
            self.latent_model = None

    def fit_sparse_dictionary(self):
        self.sparse_dictionary = SVDDictionary()
        self.sparse_dictionary.fit_from_dataloaders(
            [torch.utils.data.DataLoader(self.generator_datasets[1], batch_size=10)], self.model.ema_model.encoder
        )
        self.sparse_dictionary.save_on_disk(self.sparse_dictionary_path)

    def explain_all_components(self):
        if self.sparse_dictionary is None:
            self.fit_sparse_dictionary()

        if not self.latent_model is None:
            explanation_path = os.path.join(
                self.config.base_path,
                self.config.sparse_dictionary.sparse_dictionary_type
            )
            Path(explanation_path).mkdir(parents=True, exist_ok=True)
            sampled_x = self.sample_x(self, self.config.batch_size).detach().cpu()
            torchvision.utils.save_image(sampled_x, n_rows=int(math.sqrt(self.config.batch_size)))

        result_list = []
        for component_idx in range(self.config.sparse_dictionary.n_components):
            result_list.append(
                self.explain_sparse_component(
                    torch.utils.data.DataLoader(self.generator_datasets[1], batch_size=10),
                    component_idx,
                )
            )

    def explain_sparse_component(self, dataloader, component_idx):
        x_factual_list = []
        x_counterfactual_list = []
        start_idx = 0
        current_base_path = os.path.join(
            self.config.base_path,
            self.config.sparse_dictionary.sparse_dictionary_type,
            str(component_idx),
        )
        Path(current_base_path).mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(dataloader):
            x_factual_list.extend(list(batch[0]))
            x_factual = batch[0].to(self.device)
            x_counterfactual, (
                dot_before,
                dot_after,
            ) = self.explain_sparse_component_batch(x_factual, component_idx)
            x_counterfactual_list.extend(list(x_counterfactual.cpu()))
            (
                x_attribution_list,
                collage_path_list,
            ) = self.generator_datasets[1].generate_contrastive_collage(
                x_list=list((batch[0])),
                x_counterfactual_list=list(x_counterfactual.cpu()),
                y_target_list=list(map(lambda x: -x, list(dot_before.cpu()))),
                y_source_list=list(dot_before.cpu()),
                y_list=list(dot_before.cpu()),
                y_target_start_confidence_list=list(dot_before.cpu()),
                y_target_end_confidence_list=list(dot_after.cpu()),
                base_path=current_base_path,
                start_idx=start_idx,
            )
            start_idx += len(x_factual)

        return x_factual_list, x_counterfactual_list

    def explain_sparse_component_batch(self, x_generator, component_idx):
        print("[x_generator.min(), x_generator.max()]")
        print([x_generator.min(), x_generator.max()])
        print([x_generator.min(), x_generator.max()])
        print([x_generator.min(), x_generator.max()])
        z_sem, xT = self.encode(x_generator.to(self.device))
        w = self.sparse_dictionary.get_components()[component_idx].to(self.device)
        #z_sem2 = self._calculate_z_counterfactuals(z_sem, w)
        proj_factors = (z_sem - self.sparse_dictionary.mu.to(self.device)) @ w
        print("proj_factors:" + str(list(proj_factors.cpu().numpy())))
        z_sem_after = z_sem - 2 * proj_factors.unsqueeze(1) * w
        #z_sem_after = z_sem - proj_factors.unsqueeze(1) * w
        proj_factors_after = (z_sem_after - self.sparse_dictionary.mu.to(self.device)) @ w
        print("proj_factors_after:" + str(list(proj_factors_after.cpu().numpy())))
        x_counterfactuals_generator = self.decode((z_sem_after, xT))
        print("[x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]")
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        return x_counterfactuals_generator.cpu(), (proj_factors.cpu(), proj_factors_after.cpu())

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

        classifier_to_generator = (
            lambda x: self.generator_dataset.project_from_pytorch_default(
                self.predictor_dataset.project_to_pytorch_default(x)
            )
        )
        generator_to_classifier = (
            lambda x: self.predictor_dataset.project_from_pytorch_default(
                self.generator_dataset.project_to_pytorch_default(x)
            )
        )

        distilled_datasources = []
        for idx, predictor_dataset in enumerate(predictor_datasets):
            distilled_datasource = copy.deepcopy(predictor_dataset)
            if isinstance(distilled_datasource, torch.utils.data.DataLoader):
                distilled_datasource.dataset.normalization = self.generator_datasets[
                    idx
                ].normalization
                distilled_datasource.dataset.transform = self.generator_datasets[
                    idx
                ].transform
                distilled_datasource.dataset.config.normalization = (
                    self.generator_datasets[idx].config.normalization
                )

            elif isinstance(distilled_datasource, WeightedDataloaderList):
                for j in range(len(distilled_datasource.dataloaders)):
                    distilled_datasource.dataloaders[
                        j
                    ].dataset.normalization = self.generator_datasets[idx].normalization
                    distilled_datasource.dataloaders[
                        j
                    ].dataset.transform = self.generator_datasets[idx].transform
                    distilled_datasource.dataloaders[
                        j
                    ].dataset.config.normalization = self.generator_datasets[
                        idx
                    ].config.normalization

            else:
                distilled_datasource.normalization = self.generator_datasets[
                    idx
                ].normalization
                distilled_datasource.transform = self.generator_datasets[idx].transform
                distilled_datasource.config.normalization = self.generator_datasets[
                    idx
                ].config.normalization

            distilled_datasources.append(distilled_datasource)

        if not explainer_config.distilled_predictor is None:
            # assert explainer_config.distilled_predictor.task.output_channels == 1
            distilled_path = os.path.join(
                base_path, "explainer", "distilled_predictor", "model.cpl"
            )
            if not os.path.exists(distilled_path):
                self.gradient_predictor = distill_predictor(
                    predictor_distillation=explainer_config.distilled_predictor,
                    base_path=os.path.join(base_path, "explainer"),
                    predictor=lambda x: predictor(generator_to_classifier(x)),
                    predictor_datasource=distilled_datasources,
                    predictor_distilled=nn.Sequential(
                        *[
                            self.model.ema_model.encoder,
                            nn.Linear(self.config.encoder_dimensions, 1, bias=False),
                        ]
                    ),
                    only_last_layer=True,
                    continue_training=True,
                    task_config=TaskConfig(
                        **explainer_config.distilled_predictor["task"]
                    ),
                )

            else:
                self.gradient_predictor = torch.load(
                    distilled_path, map_location=self.device
                )

        else:
            self.gradient_predictor = predictor

        print("[x_in.min(), x_in.max()]")
        print([x_in.min(), x_in.max()])
        print([x_in.min(), x_in.max()])
        print([x_in.min(), x_in.max()])

        preds = torch.nn.Softmax(dim=-1)(predictor(x_in.to(device)).detach().cpu())
        print("preds_before: " + str(preds.argmax(dim=-1)))

        x_generator = classifier_to_generator(x_in)
        # x_generator = 2 * x_generator
        # x_generator = x_in
        print("[x_generator.min(), x_generator.max()]")
        print([x_generator.min(), x_generator.max()])
        print([x_generator.min(), x_generator.max()])
        print([x_generator.min(), x_generator.max()])
        z_sem, xT = self.encode(x_generator.to(self.device))
        w = list(self.gradient_predictor.children())[-1].weight[0]
        dot_ab = torch.tensor(
            torch.sum(z_sem * w, dim=-1, keepdim=True) > 0.5, dtype=torch.uint8
        )
        print("dot_ab before editing:", dot_ab.squeeze().detach().cpu().numpy())
        z_sem2 = self._calculate_z_counterfactuals(z_sem, w)
        dot_ab2 = torch.tensor(
            torch.sum(z_sem2 * w, dim=-1, keepdim=True) > 0, dtype=torch.uint8
        )
        print("dot_ab after editing:", dot_ab2.squeeze().detach().cpu().numpy())
        x_counterfactuals_generator = self.decode((z_sem2, xT))
        z_sem3, _ = self.encode(
            classifier_to_generator(x_counterfactuals_generator.to(self.device))
        )
        dot_ab3 = torch.tensor(
            torch.sum(z_sem3 * w, dim=-1, keepdim=True) > 0, dtype=torch.uint8
        )
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

        preds = torch.nn.Softmax(dim=-1)(
            predictor(x_counterfactuals.to(device)).detach().cpu()
        )
        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        print("preds_after:", preds.argmax(dim=-1))

        return (
            list(x_counterfactuals.cpu()),
            list(x_in - x_counterfactuals.cpu()),
            list(y_target_end_confidence),
            list(x_in),
            [],
        )

    def _calculate_z_counterfactuals(self, z_sem: torch.Tensor, w) -> torch.Tensor:
        b = w
        a = z_sem
        #
        dot_ab = torch.sum(a * b, dim=-1, keepdim=True)  # shape (batch, 1)
        dot_bb = torch.sum(b * b)  # scalar

        # projection and reflection
        proj = dot_ab / dot_bb * b  # shape (batch, n)
        # reflected = 2 * proj - a
        reflected = a - 2 * proj
        return reflected


class NormalizationModule(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
