import math
import os
import shutil
import types
import copy
from datetime import time, datetime

import torch

from pathlib import Path

import torchvision
import yaml
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
from peal.sparse_dictionaries.interfaces import SparseDictionaryConfig, SparseDictionary
from peal.sparse_dictionaries.singular_value_decomposition import (
    SVDDictionaryConfig,
)
from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary
from peal.sparse_dictionaries.utils import plot_component_ground_truth_correlations
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
    sparse_dictionary: Union[str, SparseDictionaryConfig, None] = SVDDictionaryConfig()
    visualizations_per_component: Union[int, None] = 100


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
        self.checkpoint_path = os.path.join(self.config.base_path, "square64_ddim", "last.ckpt")
        self.checkpoint_path_latent = os.path.join(self.config.base_path, "square64_autoenc_latent", "last.ckpt")
        self.load_models()
        self.load_sparse_dictionary()

    def load_sparse_dictionary(self):
        if not self.config.sparse_dictionary is None:
            self.config.sparse_dictionary.base_path = os.path.join(
                self.config.base_path,
                self.config.sparse_dictionary.sparse_dictionaries_type,
            )
            self.config.sparse_dictionary.weights_path = os.path.join(
                self.config.sparse_dictionary.base_path, self.config.sparse_dictionary.weights_name
            )
            if not os.path.exists(self.config.sparse_dictionary.weights_path):
                self.sparse_dictionary = None

            else:
                self.sparse_dictionary = get_sparse_dictionary(self.config.sparse_dictionary)
                self.sparse_dictionary.load_from_disk(self.config.sparse_dictionary.weights_path)

    def sample_z(self, batch_size=1):
        # TODO this has to be done properly with the learned prior!!!
        z_sem = torch.randn(batch_size, self.config.encoder_dimensions).to(self.device)
        xT = torch.randn([batch_size] + self.config.data.input_size).to(self.device)
        return z_sem, xT

    def sample_x(self, batch_size=1):
        if not self.latent_model is None:
            return self.latent_model.sample(N=batch_size, device=self.device)

        else:
            return torch.randn([batch_size] + self.config.data.input_size).to(self.device)

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
            # --- Existing DINOv2 Logic ---
            if self.config.model_type[: len("dino_v2")] == "dino_v2":
                # ... (your existing DINO code) ...
                pass

            # --- New OpenCLIP Logic ---
            elif "open_clip" in self.config.model_type:
                import open_clip

                # Expecting config.model_type format: "open_clip:ViT-B-32:laion2b_s34b_b79k"
                # Defaulting to ViT-B-32 if only "open_clip" is provided
                parts = self.config.model_type.split(":")
                model_name = parts[1] if len(parts) > 1 else "ViT-B-32"
                pretrained = parts[2] if len(parts) > 2 else "laion2b_s34b_b79k"

                model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

                class OpenCLIPEncoder(nn.Module):
                    def __init__(self, model, preprocess):
                        super().__init__()
                        self.model = model
                        # Convert the Compose transform to an nn.Module-like flow if possible,
                        # or keep as is if input is PIL. If input is Tensor, use torchvision.
                        self.preprocess = preprocess

                    def forward(self, x):
                        # OpenCLIP preprocess usually expects PIL or Tensors.
                        # If x is already a batch of Tensors, we apply the transforms.
                        # Note: OpenCLIP's preprocess often includes ToTensor() and Normalize().
                        # If your input x is already a normalized tensor, you might need to
                        # bypass parts of self.preprocess.
                        x_processed = self.preprocess(x)

                        # Ensure batch dimension if single image
                        if x_processed.ndim == 3:
                            x_processed = x_processed.unsqueeze(0)

                        # Extract visual features
                        latent_code = self.model.encode_image(x_processed)
                        return latent_code

                encoder = OpenCLIPEncoder(model, preprocess)

            # --- Existing UNI Logic ---
            elif self.config.model_type == "UNI":
                # ... (your existing UNI code) ...
                pass

        # ... (rest of your existing logic for encoder_path and normalization) ...

    def set_encoder(self):
        if not self.config.model_type is None:
            if self.config.model_type[: len("dino_v2")] == "dino_v2":
                if self.config.model_type == "dino_v2_small":
                    model = AutoModel.from_pretrained("facebook/dinov2-small")
                    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

                elif self.config.model_type == "dino_v2_base":
                    model = AutoModel.from_pretrained("facebook/dinov2-base")
                    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

                elif self.config.model_type == "dino_v2":
                    model = AutoModel.from_pretrained("facebook/dinov2-large")
                    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

                class DinoV2(nn.Module):
                    def __init__(self, model, processor):
                        super().__init__()
                        self.model = model
                        self.processor = processor

                    def forward(self, x):
                        cs = self.processor.crop_size
                        x_resized = torchvision.transforms.Resize([cs["height"], cs["width"]])(x)

                        def pv(v):
                            v = torch.tensor(v).to(x_resized)[:, None, None]
                            return torch.tile(v, [1, cs["height"], cs["width"]])

                        x_processed = (x_resized - pv(self.processor.image_mean)) / pv(self.processor.image_std)
                        latent_code = self.model(x_processed)["last_hidden_state"][:, 0]
                        return latent_code

                encoder = DinoV2(model, processor)

            elif "open_clip" in self.config.model_type:
                print("use open clip!!!")
                print("use open clip!!!")
                print("use open clip!!!")
                print("use open clip!!!")
                print("use open clip!!!")
                import pdb

                pdb.set_trace()
                # --- New OpenCLIP Logic ---
                import open_clip

                # Expecting config.model_type format: "open_clip:ViT-B-32:laion2b_s34b_b79k"
                # Defaulting to ViT-B-32 if only "open_clip" is provided
                parts = self.config.model_type.split(":")
                model_name = parts[1] if len(parts) > 1 else "ViT-B-32"
                pretrained = parts[2] if len(parts) > 2 else "laion2b_s34b_b79k"

                model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

                class OpenCLIPEncoder(nn.Module):
                    def __init__(self, model, preprocess):
                        super().__init__()
                        self.model = model
                        # Convert the Compose transform to an nn.Module-like flow if possible,
                        # or keep as is if input is PIL. If input is Tensor, use torchvision.
                        self.preprocess = preprocess

                    def forward(self, x):
                        # OpenCLIP preprocess usually expects PIL or Tensors.
                        # If x is already a batch of Tensors, we apply the transforms.
                        # Note: OpenCLIP's preprocess often includes ToTensor() and Normalize().
                        # If your input x is already a normalized tensor, you might need to
                        # bypass parts of self.preprocess.
                        x_processed = self.preprocess(x)

                        # Ensure batch dimension if single image
                        if x_processed.ndim == 3:
                            x_processed = x_processed.unsqueeze(0)

                        # Extract visual features
                        latent_code = self.model.encode_image(x_processed)
                        return latent_code

                encoder = OpenCLIPEncoder(model, preprocess)

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
                transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

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
                    self.config.base_path + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                )

        else:
            self.model = None  # LitModel(conf)

        latent_conf = square64_autoenc_latent(os.path.join(self.config.base_path, "square64_ddim"))
        self.adjust_config(latent_conf)
        if os.path.exists(self.checkpoint_path_latent):
            self.latent_model = LitModel.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path_latent,
                conf=latent_conf,
                map_location="cpu",
            )
            self.latent_model.to(self.device)

        else:
            self.latent_model = None

    def fit_sparse_dictionary(self):
        self.sparse_dictionary = get_sparse_dictionary(self.config.sparse_dictionary)
        self.sparse_dictionary.fit_from_dataloaders(
            [torch.utils.data.DataLoader(self.generator_datasets[1], batch_size=10)],
            self.model.ema_model.encoder,
        )
        Path(self.config.sparse_dictionary.base_path).mkdir(parents=True, exist_ok=True)
        self.sparse_dictionary.save_on_disk(self.config.sparse_dictionary.weights_path)
        save_yaml_config(
            self.config.sparse_dictionary, os.path.join(self.config.sparse_dictionary.base_path, "config.yaml")
        )

    def explain_all_components(self, sparse_dictionary=None):
        if self.sparse_dictionary is None or not sparse_dictionary is None:
            if isinstance(sparse_dictionary, SparseDictionary):
                self.sparse_dictionary = sparse_dictionary
                self.config.sparse_dictionary = copy.deepcopy(sparse_dictionary.config)

            else:
                if isinstance(sparse_dictionary, SparseDictionaryConfig):
                    self.config.sparse_dictionary = sparse_dictionary

                self.config.sparse_dictionary.act_size = self.config.encoder_dimensions
                self.load_sparse_dictionary()
                if self.sparse_dictionary is None:
                    self.fit_sparse_dictionary()

            # save_yaml_config(self.config, os.path.join(self.config.base_path, "config.yaml"))

        explanation_path = os.path.join(
            self.config.base_path,
            self.config.sparse_dictionary.sparse_dictionaries_type,
        )
        Path(explanation_path).mkdir(parents=True, exist_ok=True)

        task_config_buffer = (
            self.generator_datasets[1].task_config if hasattr(self.generator_datasets[1], "task_config") else None
        )
        self.generator_datasets[1].task_config = None
        y_list = []
        c_list = []
        z_list = []
        for idx, batch in enumerate(torch.utils.data.DataLoader(self.generator_datasets[1], batch_size=10)):
            print(str(10 * idx) + "/" + str(len(self.generator_datasets[1])))
            x, y = batch
            z, _ = self.encode(x.to(self.device))
            c = z @ self.sparse_dictionary.get_components().to(self.device)
            y_list.append(y)
            c_list.append(c.detach().cpu())
            z_list.append(z.detach().cpu())

        y_stack = torch.cat(y_list)
        c_stack = torch.cat(c_list)
        z_stack = torch.cat(z_list)
        plot_component_ground_truth_correlations(
            filename=os.path.join(
                self.config.base_path,
                self.config.sparse_dictionary.sparse_dictionaries_type,
                "correlations.png",
            ),
            components=c_stack,
            ground_truth_attributes=y_stack,
            data=z_stack,
        )
        self.generator_datasets[1].task_config = task_config_buffer

        if not self.latent_model is None:
            sampled_x = self.sample_x(self.config.batch_size).detach().cpu()
            torchvision.utils.save_image(
                sampled_x,
                os.path.join(explanation_path, "samples.png"),
                nrow=int(math.sqrt(self.config.batch_size)),
            )

        result_list = []
        for component_idx in range(self.config.sparse_dictionary.n_components):
            result_list.append(
                self.explain_sparse_component(
                    torch.utils.data.DataLoader(
                        self.generator_datasets[1],
                        batch_size=self.config.sparse_dictionary.visualizations_per_component,
                    ),
                    component_idx,
                )
            )

    def explain_sparse_component(self, dataloader, component_idx):
        x_factual_list = []
        x_counterfactual_list = []
        start_idx = 0
        current_base_path = os.path.join(
            self.config.base_path,
            self.config.sparse_dictionary.sparse_dictionaries_type,
            str(component_idx),
        )
        Path(current_base_path).mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(dataloader):
            if (
                not self.config.visualizations_per_component is None
                and start_idx >= self.config.visualizations_per_component
            ):
                break

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
        # w = self.sparse_dictionary.get_components()[:, component_idx].to(self.device)
        w_raw = self.sparse_dictionary.get_components()[:, component_idx].to(self.device)
        # Normalize w to ensure it is a unit vector
        w = w_raw / torch.norm(w_raw, p=2)
        # z_sem2 = self._calculate_z_counterfactuals(z_sem, w)
        proj_factors = (z_sem - self.sparse_dictionary.mu.to(self.device)) @ w

        print("proj_factors:" + str(list(proj_factors.detach().cpu().numpy())))
        z_sem_after = z_sem - 2 * proj_factors.unsqueeze(1) * w
        # z_sem_after = z_sem - proj_factors.unsqueeze(1) * w
        proj_factors_after = (z_sem_after - self.sparse_dictionary.mu.to(self.device)) @ w
        print("proj_factors_after:" + str(list(proj_factors_after.detach().cpu().numpy())))
        x_counterfactuals_generator = self.decode((z_sem_after, xT))
        print("[x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]")
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        x_counterfactuals_generator = self.generator_datasets[1].project_from_pytorch_default(
            x_counterfactuals_generator
        )
        print("[x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]")
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        return x_counterfactuals_generator.cpu(), (
            proj_factors.cpu(),
            proj_factors_after.cpu(),
        )

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
        param_list = [p for p in predictor.parameters()]
        device = param_list[0].device
        pred_original = torch.nn.functional.softmax(predictor(x_in.to(self.device))).detach().cpu()
        target_confidences = [pred_original[i][target_classes[i]] for i in range(len(target_classes))]
        target_confidence_goal = 1 - torch.tensor(target_confidences)

        if isinstance(predictor_datasets[1], WeightedDataloaderList):
            validation_dataset = predictor_datasets[1].dataloaders[0].dataset

        else:
            validation_dataset = predictor_datasets[1]

        classifier_to_generator = lambda x: self.generator_dataset.project_from_pytorch_default(
            self.predictor_dataset.project_to_pytorch_default(x)
        )
        generator_to_classifier = lambda x: self.predictor_dataset.project_from_pytorch_default(
            self.generator_dataset.project_to_pytorch_default(x)
        )

        distilled_datasources = []
        for idx, predictor_dataset in enumerate(predictor_datasets):
            distilled_datasource = copy.deepcopy(predictor_dataset)
            if isinstance(distilled_datasource, torch.utils.data.DataLoader):
                distilled_datasource.dataset.normalization = self.generator_datasets[idx].normalization
                distilled_datasource.dataset.transform = self.generator_datasets[idx].transform
                distilled_datasource.dataset.config.normalization = self.generator_datasets[idx].config.normalization

            elif isinstance(distilled_datasource, WeightedDataloaderList):
                for j in range(len(distilled_datasource.dataloaders)):
                    distilled_datasource.dataloaders[j].dataset.normalization = self.generator_datasets[
                        idx
                    ].normalization
                    distilled_datasource.dataloaders[j].dataset.transform = self.generator_datasets[idx].transform
                    distilled_datasource.dataloaders[j].dataset.config.normalization = self.generator_datasets[
                        idx
                    ].config.normalization

            else:
                distilled_datasource.normalization = self.generator_datasets[idx].normalization
                distilled_datasource.transform = self.generator_datasets[idx].transform
                distilled_datasource.config.normalization = self.generator_datasets[idx].config.normalization

            distilled_datasources.append(distilled_datasource)

        if not explainer_config.distilled_predictor is None:
            # assert explainer_config.distilled_predictor.task.output_channels == 1
            distilled_path = os.path.join(base_path, "explainer", "distilled_predictor", "model.cpl")
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
                    task_config=TaskConfig(**explainer_config.distilled_predictor["task"]),
                )

            else:
                self.gradient_predictor = torch.load(distilled_path, map_location=self.device)

            decision_boundary_path = os.path.join(
                base_path, "explainer", "distilled_predictor", "decision_boundary.png"
            )
            if hasattr(
                self.generator_datasets[0],
                "visualize_decision_boundary",
            ) and not os.path.exists(decision_boundary_path):
                self.generator_datasets[1].visualize_decision_boundary(
                    self.gradient_predictor,
                    100,
                    self.device,
                    decision_boundary_path,
                )

        else:
            self.gradient_predictor = predictor

        x_generator = classifier_to_generator(x_in)
        # x_generator = 2 * x_generator
        # x_generator = x_in
        z_sem, xT = self.encode(x_generator.to(self.device))
        w = list(self.gradient_predictor.children())[-1].weight[0]
        z_sem_before, indices, distances = self._calculate_z_counterfactuals(
            z_sem, w, explainer_config, explainer_config.num_attempts
        )
        z_sem2 = z_sem_before.reshape([-1, z_sem_before.shape[-1]])
        xT_decoding = xT.unsqueeze(1).unsqueeze(1)
        xT_decoding = xT_decoding.tile(
            1, explainer_config.num_attempts, len(explainer_config.linesearch_factors), 1, 1, 1
        )
        xT_decoding = xT_decoding.reshape([-1] + list(xT.shape[1:]))
        x_counterfactuals_generator = self.decode((z_sem2, xT_decoding))
        x_counterfactuals = x_counterfactuals_generator.detach()

        preds = torch.nn.Softmax(dim=-1)(predictor(x_counterfactuals.to(device)).detach().cpu())
        y_target_end_confidence = torch.zeros([preds.shape[0]])
        for i in range(preds.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i % target_classes.shape[0]]]

        x_counterfactuals = torch.reshape(
            x_counterfactuals, list(z_sem_before.shape[:3]) + list(x_counterfactuals.shape[1:])
        )
        y_target_end_confidence = torch.reshape(y_target_end_confidence, z_sem_before.shape[:3])
        x_counterfactuals_out_list = []
        y_target_end_confidence_list = []
        x_out_list = []
        indices_list = []
        print("x_counterfactuals: " + str(x_counterfactuals.min()) + " to " + str(x_counterfactuals.max()))
        print("x_counterfactuals: " + str(x_counterfactuals.min()) + " to " + str(x_counterfactuals.max()))
        print("x_counterfactuals: " + str(x_counterfactuals.min()) + " to " + str(x_counterfactuals.max()))
        for i in range(explainer_config.num_attempts):
            if x_counterfactuals.shape[2] >= 2:
                y_target_diff = torch.clone(y_target_end_confidence)
                for b in range(y_target_end_confidence.shape[0]):
                    outlier_scores = validation_dataset.calculate_outlier_score(x_counterfactuals[b, i])["relative"].cpu()
                    mask = torch.logical_and(outlier_scores < 1.3, outlier_scores > 0.1)
                    masked_difference = y_target_diff[b, i] * mask
                    y_target_diff[b, i] = torch.abs(masked_difference - target_confidence_goal[b])

                # j = torch.argmax(y_target_end_confidence[:, i, :], dim=-1)
                j = torch.argmin(y_target_diff[:, i, :], dim=-1)

            else:
                j = torch.zeros([x_counterfactuals.shape[0]], dtype=torch.long)

            for k in range(j.shape[0]):
                x_counterfactuals_out_list.append(x_counterfactuals[k, i, j[k], :])
                y_target_end_confidence_list.append(float(y_target_end_confidence[k, i, j[k]]))
                indices_list.append(indices[k, i])

            x_out_list.append(x_in)

        x_counterfactuals = torch.stack(x_counterfactuals_out_list, dim=0)
        x_out = torch.cat(x_out_list, dim=0)
        y_target_end_confidence = torch.tensor(y_target_end_confidence_list)
        indices = torch.tensor(indices_list) #, dim=0)
        x_difference = x_out - x_counterfactuals.cpu()

        return (
            list(x_counterfactuals.cpu()),
            list(x_difference),
            list(y_target_end_confidence),
            list(x_in),
            [],
            list(indices.cpu()),
        )

    def _calculate_z_counterfactuals(self, z_sem: torch.Tensor, w, explainer_config=None, num_attempts=1):
        if explainer_config is None or explainer_config.num_attempts == 1:
            b = w
            a = z_sem
            #
            dot_ab = torch.sum(a * b, dim=-1, keepdim=True)  # shape (batch, 1)
            dot_bb = torch.sum(b * b)  # scalar

            # projection and reflection
            proj = dot_ab / dot_bb * b  # shape (batch, n)
            # reflected = 2 * proj - a
            reflected = a - 2 * proj
            return reflected, None, torch.norm(reflected - z_sem, p=2, dim=-1, keepdim=False)

        else:
            component_indices = range(num_attempts)
            W_all = self.sparse_dictionary.get_components()[:, :num_attempts]

            if component_indices is not None:
                W = W_all[:, component_indices].to(z_sem.device)
            else:
                W = W_all.to(z_sem.device)

            # --- CHANGED: Calculate "Cross-Projection" to target Classifier Flip ---

            # 1. Alignment of current Z with Classifier (z . w)
            # z_sem: [Batch, Dim] | w: [Dim] -> Result: [Batch, 1]
            dot_zw = torch.sum(z_sem * w, dim=-1, keepdim=True)

            # 2. Alignment of Components with Classifier (u . w)
            # w: [Dim] | W: [Dim, K] -> Result: [K]
            # This tells us how much moving along a component 'u' affects the class score
            dot_uw = torch.matmul(w, W)

            # Safety: If a component is orthogonal to the classifier (dot_uw ~ 0),
            # moving along it won't change the class. We prevent division by zero.
            eps = 1e-6
            dot_uw_safe = dot_uw.clone()
            dot_uw_safe[torch.abs(dot_uw_safe) < eps] = eps

            # 3. Calculate Projection Factor
            # We want a step `s` such that (z - s*u).w = -z.w
            # This requires s = 2 * (z.w) / (u.w)
            # The '2' is applied later by linesearch_factors (assuming it contains 2.0)
            # Shape: [Batch, K]
            proj_factors = dot_zw / dot_uw_safe.unsqueeze(0)

            # 4. Create Projection Vectors
            # Expand factors to: [Batch, K, 1]
            # Expand W to: [1, K, Dim]
            # Result: [Batch, K, Dim]
            projections = proj_factors.unsqueeze(-1) * W.permute(1, 0).unsqueeze(0)

            # --- END CHANGES ---

            # 5. Compute Reflections using Linesearch
            line_search_factors = torch.tensor(explainer_config.linesearch_factors).to(z_sem.device)
            z_base = z_sem.unsqueeze(1).unsqueeze(1)
            proj_expanded = projections.unsqueeze(2)
            factors_expanded = line_search_factors.view(1, 1, -1, 1)

            # Apply the update
            z_reflected = z_base - factors_expanded * proj_expanded

            # 6. Calculate Distances & Sort
            distances = torch.norm(z_base - z_reflected, p=2, dim=-1)
            sorted_indices = torch.argsort(distances, dim=1)

            component_indices = torch.arange(sorted_indices.shape[1]).unsqueeze(0).tile([sorted_indices.shape[0], 1])

            return z_reflected, component_indices, distances


class NormalizationModule(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


"""z_sem2_list = []
        xT_list = []
        x_out = []
        indices_list = []
        for i in range(explainer_config.num_attempts):
            for j in range(z_sem2.shape[2]):
                z_sem2_list.append(z_sem2[:, i, j, :])
                xT_list.append(xT)

            x_out.append(x_in)
            indices_list.append(indices)

        z_sem2 = torch.cat(z_sem2_list, dim=0)
        xT = torch.cat(xT_list, dim=0)
        x_out = torch.cat(x_out, dim=0)
        indices = torch.cat(indices_list, dim=0)
        dot_ab2 = torch.tensor(torch.sum(z_sem2 * w, dim=-1, keepdim=True) > 0, dtype=torch.uint8)
        print("dot_ab after editing:", dot_ab2.squeeze().detach().cpu().numpy())
        x_counterfactuals_generator = self.decode((z_sem2, xT))
        # x_counterfactuals_generator = x_generator
        print("[x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]")
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        print([x_counterfactuals_generator.min(), x_counterfactuals_generator.max()])
        # x_counterfactuals = generator_to_classifier(x_counterfactuals_generator.cpu())
        x_counterfactuals = x_counterfactuals_generator.detach()
        print("[x_counterfactuals.min(), x_counterfactuals.max()]")
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        print([x_counterfactuals.min(), x_counterfactuals.max()])
        print([x_counterfactuals.min(), x_counterfactuals.max()])

        preds = torch.nn.Softmax(dim=-1)(predictor(x_counterfactuals.to(device)).detach().cpu())
        y_target_end_confidence = torch.zeros([preds.shape[0]])
        for i in range(preds.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i % target_classes.shape[0]]]

        # TODO here i want you to select the best counterfactual of the line search

        # keep only the linesearch attempt with highest target confidence
        print("preds_after:", preds.argmax(dim=-1))
        x_out_final = x_out - x_counterfactuals.cpu()
        """

"""z_sem2_list = []
        xT_list = []
        x_out_list = []
        # New list to track which component index corresponds to which candidate
        component_indices_list = []

        batch_size = x_in.shape[0]

        # Flatten logic: We iterate K and S, creating a long list of Batch-sized tensors
        # Total candidates = K * S * Batch
        for i in range(num_attempts): # i is Component Index
            for j in range(z_sem2.shape[2]):           # j is Line Search Step
                z_sem2_list.append(z_sem2[:, i, j, :])
                xT_list.append(xT)
                x_out_list.append(x_in)

                # Create a tensor [Batch] filled with the current component index 'i'
                component_indices_list.append(torch.full((batch_size,), i, dtype=torch.long, device=device))

        z_sem2_cat = torch.cat(z_sem2_list, dim=0)
        xT_cat = torch.cat(xT_list, dim=0)
        x_out_cat = torch.cat(x_out_list, dim=0)
        indices_cat = torch.cat(component_indices_list, dim=0) # Tracked components

        # Decode EVERYTHING (expensive but supports your structure)
        x_counterfactuals_generator = self.decode((z_sem2_cat, xT_cat))
        x_counterfactuals = x_counterfactuals_generator.detach()

        # Predict on EVERYTHING
        preds_all = torch.nn.Softmax(dim=-1)(predictor(x_counterfactuals.to(device)).detach().cpu())

        # --- SELECTION LOGIC ---
        # We need to select 1 best candidate per original batch item
        # Structure of preds_all: [Total_Candidates * Batch, Num_Classes]
        # The order is: [Cand0_Batch0, Cand0_Batch1, ... Cand1_Batch0, Cand1_Batch1, ...]

        total_candidates = num_attempts * z_sem2.shape[2]

        # 1. Reshape to [Total_Candidates, Batch, Num_Classes]
        preds_reshaped = preds_all.view(total_candidates, batch_size, -1)

        # 2. Permute to [Batch, Total_Candidates, Num_Classes]
        preds_batch = preds_reshaped.permute(1, 0, 2)

        # 3. Get Target Confidence for all candidates
        # target_classes: [Batch] -> expand to [Batch, Total_Candidates, 1]
        target_cls_expanded = target_classes.unsqueeze(1).unsqueeze(2).expand(-1, total_candidates, 1)

        # Gather confidence: [Batch, Total_Candidates]
        candidate_confs = torch.gather(preds_batch.cpu(), 2, target_cls_expanded.cpu()).squeeze(-1)

        # 4. Find Best Candidate (Minimize distance to goal)
        # [Batch, Total_Candidates]
        diff = torch.abs(candidate_confs - target_confidence_goal)
        best_candidate_indices = torch.argmin(diff, dim=1) # [Batch] (indices 0 to Total_Candidates-1)

        # 5. Select the Data corresponding to best indices
        # We need to compute the flat indices in the 'cat' tensors
        # Flat Index = (Best_Candidate_Index * Batch_Size) + Batch_Index
        flat_indices = best_candidate_indices * batch_size + torch.arange(batch_size)

        # Gather the winners
        x_counterfactuals_final = x_counterfactuals[flat_indices]
        y_target_end_confidence = candidate_confs[torch.arange(batch_size), best_candidate_indices]
        final_component_indices = indices_cat[flat_indices]

        # Recalculate x_out diff (x_in - chosen_counterfactual)
        x_out_final = x_in - x_counterfactuals_final.cpu()

        print("preds_after:", preds_all[flat_indices].argmax(dim=-1))"""
