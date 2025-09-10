import os
import types
import shutil
import copy
from pathlib import Path

import torch
import io
import blobfile as bf
import torchvision
from diffusers import StableDiffusionPipeline

from mpi4py import MPI
from torch import nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from transformers import AutoModel, AutoImageProcessor

from peal.dependencies.ddpm_inversion.ddm_inversion.inversion_utils import inversion_forward_process, \
    inversion_reverse_process
from peal.dependencies.diffusion_regression_counterfactuals.src.related_work.diffae.experiment import LitModel
from peal.dependencies.diffusion_regression_counterfactuals.src.related_work.diffae.templates import square64_autoenc
from peal.editors.ddpm_inversion import DDPMInversionConfig
from peal.data.dataloaders import get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.data.datasets import Image2MixedDataset
from peal.dependencies.ddpm_inversion.ddpm_inversion import DDPMInversion
from peal.dependencies.lora.train_text_to_image_lora import lora_finetune
from peal.dependencies.time.core.utils import load_tokens_and_embeddings
from peal.generators.interfaces import EditCapableGenerator, InvertibleGenerator
from peal.global_utils import load_yaml_config, embed_numberstring, save_yaml_config
from peal.dependencies.time.generate_ce import (
    generate_time_counterfactuals,
)
from peal.dependencies.time.get_predictions import get_predictions
from peal.dependencies.time.training import textual_inversion_training

from typing import Union

from peal.generators.interfaces import GeneratorConfig
from peal.data.interfaces import DataConfig
from peal.architectures.interfaces import TaskConfig


class DiffusionAutoencoderConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "StableDiffusion"
    base_path: str = "/home/space/datasets/peal/peal_runs/stable_diffusion"
    #full_args: Union[None, dict] = None
    """
    The config of the data.
    """
    data: DataConfig = DataConfig()
    sd_model: str = "CompVis/stable-diffusion-v1-4"
    #
    revision: Union[str, type(None)] = None
    variant: Union[str, type(None)] = None
    dataset_name: Union[str, type(None)] = None
    dataset_config_name: Union[str, type(None)] = None
    train_data_dir: Union[str, type(None)] = None
    image_column: Union[str, type(None)] = "image"
    caption_column: Union[str, type(None)] = "text"
    validation_prompt: Union[str, type(None)] = None
    num_validation_images: int = 4
    validation_epochs: int = 1
    max_train_samples: Union[int, type(None)] = None
    cache_dir: Union[str, type(None)] = None
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    train_batch_size: int = 16
    num_train_epochs: int = 100
    max_train_steps: Union[int, type(None)] = 100000 # None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    snr_gamma: Union[float, type(None)] = None
    use_8bit_adam: bool = False
    allow_tf32: bool = False
    dataloader_num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    push_to_hub: bool = False
    hub_token: Union[str, type(None)] = None
    prediction_type: Union[str, type(None)] = None
    hub_model_id: Union[str, type(None)] = None
    logging_dir: Union[str, type(None)] = "logs"
    mixed_precision: Union[str, type(None)] = None
    report_to: Union[str, type(None)] = "tensorboard"
    local_rank: int = 1
    checkpointing_steps: int = 500
    checkpoints_total_limit: Union[int, type(None)] = None
    resume_from_checkpoint: Union[str, type(None)] = None
    enable_xformers_memory_efficient_attention: bool = False
    noise_offset: float = 0.0
    rank: int = 10
    task_config: Union[TaskConfig, type(None)] = None
    encoder : str = "facebook/dinov2-small"
    use_lora: bool = False


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

        if self.config.encoder[:len("facebook/dinov2")] == "facebook/dinov2":
            sem_encoder = AutoModel.from_pretrained(self.config.encoder).to("cuda")
            sem_encoder_processor = AutoImageProcessor.from_pretrained(self.config.encoder)
            cs = sem_encoder_processor.crop_size
            def img_semantic_encoder(x):
                x_resized = torchvision.transforms.Resize([cs['height'],cs['width']])(x)
                def pv(v):
                    v = torch.tensor(v).to(x_resized)[:, None, None]
                    return torch.tile(v, [1, cs['height'],cs['width']])

                x_processed = (x_resized - pv(sem_encoder_processor.image_mean)) / pv(sem_encoder_processor.image_std)
                latent_code = sem_encoder(x_processed.to(('cuda')))['last_hidden_state'][:,0]

                return latent_code

            self.img_semantic_encoder = img_semantic_encoder

    def sample_x(self, batch_size=1):
        images = self.pipeline(batch_size * [""]).images
        images_torch = torch.stack([ToTensor()(image) for image in images])
        return images_torch

    def encode(self, x, t=1.0):
        z = None
        # TODO implement encoding properly
        return z

    def decode(self, z, t=1.0):
        x = None
        # TODO implement decoding properly
        return x

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
        finetune_args.resume_from_checkpoint = 'latest'
        finetune_args.img_semantic_encoder = self.img_semantic_encoder
        # TODO add actual training here

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        classifier: nn.Module,
        explainer_config,
        predictor_dataset,
        pbar=None,
        mode="",
        base_path="",
    ):
        if self.generator_dataset is None:
            self.initialize(classifier, base_path, explainer_config)

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
