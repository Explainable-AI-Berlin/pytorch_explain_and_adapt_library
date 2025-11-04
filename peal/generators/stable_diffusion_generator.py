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

from peal.dependencies.ddpm_inversion.ddm_inversion.inversion_utils import (
    inversion_forward_process,
    inversion_reverse_process,
)
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


class StableDiffusionConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
    """

    """
    The type of generator that shall be used.
    """
    generator_type: str = "StableDiffusion"
    base_path: str = "/home/space/datasets/peal/peal_runs/stable_diffusion"
    # full_args: Union[None, dict] = None
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
    seed: Union[int, type(None)] = None
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    train_batch_size: int = 16
    num_train_epochs: int = 100
    max_train_steps: Union[int, type(None)] = 100000  # None
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


class StableDiffusion(InvertibleGenerator, EditCapableGenerator):
    def __init__(self, config, classifier_dataset=None, model_dir=None, device="cpu"):
        super().__init__()
        self.config = load_yaml_config(config)
        self.classifier_dataset = copy.deepcopy(classifier_dataset)
        # TODO something is wrong here!!!
        self.train_dataset = get_datasets(self.config.data)[0]
        if not self.config.task_config is None:
            self.train_dataset.task_config = self.config.task_config

        elif not self.classifier_dataset is None:
            self.train_dataset.task_config = self.classifier_dataset.task_config

        self.generator_dataset = None

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data_test")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals_test")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.sd_model,
        )
        self.pipeline.to(device)
        # self.pipeline.run_safety_checker = lambda image, device, dtype: image, False
        self.pipeline.safety_checker = None
        """lora_dir = os.path.join(self.model_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(lora_dir):
            self.pipeline.unet.load_attn_procs(lora_dir)"""

    def sample_x(self, batch_size=1):
        images = self.pipeline(batch_size * [""]).images
        images_torch = torch.stack([ToTensor()(image) for image in images])
        return images_torch

    def encode(self, x, t=1.0):
        """
        respaced_steps = int(t * int(self.config.timestep_respacing))
        timesteps = list(range(respaced_steps))[::-1]
        def local_forward(x, t, idx, noise, steps, diffusion, model):
            out = diffusion.p_mean_variance(model, x, t, clip_denoised=True)

            x = out["mean"]

            if idx != (steps - 1):
                x += torch.exp(0.5 * out["log_variance"]) * noise

            return x

        for idx, t in enumerate(timesteps):
            t = torch.tensor([t] * x.size(0), device=x.device)

            if idx == 0:
                x = self.diffusion.q_sample(x, t, noise=self.noise_fn(x))

            if hasattr(self, "fix_noise") and self.fix_noise:
                noise = self.noise[idx + 1, ...].unsqueeze(dim=0)

            elif self.config.stochastic:
                noise = torch.randn_like(x)

            else:
                noise = torch.zeros_like(x)

            x = local_forward(x, t, idx, noise, respaced_steps, self.diffusion, self.model)
        """

        # TODO why are gradients in ACE scaled???
        # t = torch.tensor([self.steps - 1] * x.size(0), device=x.device)
        x0 = torchvision.transforms.Resize([512, 512])(
            torch.clone(x).to(self.device)
        )  # load_512(image_path, *offsets, device)

        # vae encode image
        w0 = (self.pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # find Zs and wts - forward process
        wt, zs, wts = inversion_forward_process(
            self.pipe,
            w0,
            etas=self.config.eta,
            prompt=x.shape[0] * [""],
            cfg_scale=self.config.cfg_scale_src,
            prog_bar=True,
            num_inference_steps=self.config.num_diffusion_steps,
        )
        x = wt, zs, wts
        return x

    def decode(self, z, t=1.0):
        # TODO test decode function via sampling function
        from peal.dependencies.ddpm_inversion.prompt_to_prompt.ptp_classes import (
            AttentionStore,
        )

        controller = AttentionStore()
        from peal.dependencies.ddpm_inversion.prompt_to_prompt.ptp_utils import (
            register_attention_control,
        )

        register_attention_control(self.pipe, controller)
        wt, zs, wts = z
        w0, _ = inversion_reverse_process(
            self.pipe,
            xT=wts[self.config.num_diffusion_steps - self.config.skip],
            etas=self.config.eta,
            prompts=z.shape[0] * [""],
            cfg_scales=[self.config.cfg_scale_tar],
            prog_bar=True,
            zs=zs[: (self.config.num_diffusion_steps - self.config.skip)],
            controller=controller,
            # classifier=classifier_loss,
        )

        # vae decode image
        x = self.pipe.vae.decode(1 / 0.18215 * w0).sample
        """
        respaced_steps = int(t * int(self.config.timestep_respacing))
        timesteps = list(range(respaced_steps))[::-1]
        for idx, t in enumerate(timesteps):
            t = torch.tensor([t] * x.size(0), device=x.device)

            if idx == 0:
                x = self.diffusion.q_sample(x, t, noise=self.noise_fn(x))

            out = self.diffusion.p_mean_variance(self.model, x, t, clip_denoised=True)

            x = out["mean"]

            if idx != (respaced_steps - 1):
                if self.config.stochastic:
                    x += torch.exp(0.5 * out["log_variance"]) * self.noise_fn(x)"""

        return x

    def train_model(
        self,
    ):
        # write the yaml config on disk
        if not os.path.exists(self.config.base_path):
            Path(self.config.base_path).mkdir(parents=True, exist_ok=True)

        save_yaml_config(
            self.config, os.path.join(self.config.base_path, "config.yaml")
        )
        finetune_args = types.SimpleNamespace(**self.config.__dict__)
        finetune_args.train_dataset = self.train_dataset
        finetune_args.pipeline = self.pipeline
        finetune_args.resume_from_checkpoint = "latest"

        """train_dataloader = get_dataloader(
            self.train_dataset, mode="train", batch_size=self.config.batch_size
        )

        val_dataloader = get_dataloader(
            self.val_dataset, mode="train", batch_size=self.config.batch_size
        )
        finetune_args.train_dataloader = train_dataloader
        finetune_args.val_dataloader = val_dataloader"""
        print("Start LORA finetuning")
        print("Start LORA finetuning")
        print("Start LORA finetuning")
        self.pipeline = lora_finetune(finetune_args)
        print("Finished LORA finetuning")
        print("Finished LORA finetuning")
        print("Finished LORA finetuning")

    def initialize(self, classifier, base_path, explainer_config):
        if explainer_config.use_lora:
            self.train_model()

        class_predictions_path = os.path.join(base_path, "explainer", "predictions.csv")
        Path(os.path.join(base_path, "explainer")).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(class_predictions_path):
            self.classifier_dataset.enable_url()
            prediction_args = types.SimpleNamespace(
                batch_size=32,
                dataset=self.classifier_dataset,
                classifier=classifier,
                label_path=class_predictions_path,
                partition="train",
                label_query=0,
                max_samples=explainer_config.max_samples,
            )
            get_predictions(prediction_args)
            self.classifier_dataset.disable_url()

        writer = SummaryWriter(os.path.join(base_path, "explainer", "logs"))
        generator_dataset_config = copy.deepcopy(self.config.data)
        generator_dataset_config.split = [0.9, 1.0]
        self.generator_dataset, self.generator_dataset_val, _ = get_datasets(
            config=generator_dataset_config, data_dir=class_predictions_path
        )
        if self.generator_dataset.task_config is None:
            self.generator_dataset.task_config = TaskConfig()
            self.generator_dataset.task_config.y_selection = ["prediction"]

        else:
            self.generator_dataset.task_config.y_selection = ["prediction"]

        self.generator_dataset_val.task_config = self.generator_dataset.task_config
        if explainer_config.learn_dataset_embedding:
            context_embedding_path = os.path.join(
                base_path, "explainer", "context", "context_embedding"
            )
            if not os.path.exists(context_embedding_path):
                os.makedirs(
                    os.path.join(base_path, "explainer", "context"), exist_ok=True
                )
                train_context_embedding_args = types.SimpleNamespace(
                    embedding_files=[],
                    output_path=context_embedding_path,
                    dataset=self.generator_dataset,
                    partition="train",
                    phase="context",
                    batch_size=explainer_config.train_batch_size,
                    training_label=-1,
                    custom_tokens=explainer_config.custom_tokens_context,
                    prompt=explainer_config.base_prompt,
                    pipeline=self.pipeline,
                    generator_dataset_val=self.generator_dataset_val,
                    writer=writer,
                    **explainer_config.__dict__
                )
                textual_inversion_training(train_context_embedding_args)

            embedding_files = [context_embedding_path]

        else:
            embedding_files = []

        # TODO how to extend this for multiclass??
        for class_idx in range(self.generator_dataset.config.output_size[0]):
            class_token_path = os.path.join(
                base_path,
                "explainer",
                "class" + str(class_idx),
                "class_token" + str(class_idx),
            )
            if not os.path.exists(class_token_path):
                os.makedirs(
                    os.path.join(base_path, "explainer", "class" + str(class_idx)),
                    exist_ok=True,
                )
                class_related_bias_embedding_args = types.SimpleNamespace(
                    embedding_files=embedding_files,
                    output_path=class_token_path,
                    dataset=self.generator_dataset,
                    custom_tokens=explainer_config.class_custom_token[class_idx].split(
                        " "
                    ),
                    training_label=class_idx,
                    phase="class",
                    batch_size=explainer_config.train_batch_size,
                    generator_dataset_val=self.generator_dataset_val,
                    writer=writer,
                    pipeline=self.pipeline,
                    prompt=explainer_config.base_prompt
                    + explainer_config.prompt_connector
                    + explainer_config.class_custom_token[class_idx],
                    **explainer_config.__dict__
                )
                textual_inversion_training(class_related_bias_embedding_args)

        if explainer_config.editing_type == "ddpm_inversion":
            # TODO somehow the config should be possible to influence
            ddpm_inversion_config = DDPMInversionConfig()
            ddpm_inversion_config.cfg_scale_src = (
                explainer_config.guidance_scale_invertion[0]
            )
            ddpm_inversion_config.cfg_scale_tar = (
                explainer_config.guidance_scale_denoising[0]
            )
            self.editor = DDPMInversion(ddpm_inversion_config)
            embedding_files = []
            for class_idx in range(self.generator_dataset.config.output_size[0]):
                embedding_files.append(
                    os.path.join(
                        base_path, "explainer", "class0", "class_token" + str(class_idx)
                    )
                )

            if explainer_config.learn_dataset_embedding:
                embedding_files = [
                    os.path.join(base_path, "explainer", "context", "context_embedding")
                ] + embedding_files

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
