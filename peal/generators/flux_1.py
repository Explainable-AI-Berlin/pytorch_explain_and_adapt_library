"""Rectified-flow (FLUX / Stable Diffusion 3) generator with FastDiME editing.

:class:`FluxGenerator` wraps a diffusers ``StableDiffusion3Pipeline`` loaded
from ``pretrained_model_name_path`` (FLUX.1-dev by default) as a PEAL
``EditCapableGenerator``. Images live in the pipeline's VAE latent space and
are moved along the flow-matching schedule ``x_sigma = (1 - sigma) x_0 +
sigma * noise``. Counterfactual edits are produced with FastDiME-style
classifier guidance in latent space (:meth:`FluxGenerator.FastDiME` and the
edit-friendly-inversion variant :meth:`FluxGenerator.FastDiME_DDPM`), with
optional self-optimised or fixed masks that keep unchanged regions equal to the
input. Many methods dump intermediate images below ``$PEAL_RUNS/debug`` (or
``utils/delete_me``) for inspection; :meth:`FluxGenerator.train_model` hooks
into the vendored LoRA fine-tuning script.
"""

import copy
import os
import types
from pathlib import Path
from huggingface_hub.hf_api import HfFolder
from torchvision.transforms import transforms, ToPILImage
from peal.dependencies.lora.train_text_to_image_lora import lora_finetune
from peal.global_utils import load_yaml_config, save_yaml_config
from diffusers import StableDiffusion3Pipeline
import torch
from peal.architectures.interfaces import TaskConfig
from peal.generators.interfaces import EditCapableGenerator
from peal.global_utils import generate_smooth_mask
from peal.data.dataset_factory import get_datasets
from torchvision.transforms import ToTensor
from torch import nn
from typing import Union, Tuple
from peal.generators.interfaces import GeneratorConfig, ExplainerConfig
from peal.data.interfaces import DataConfig
import gc

from torch.nn import functional as F
from peal.dependencies.FastDiME_CelebA.core.sample_utils import (
    clean_class_cond_fn,
    clean_multiclass_cond_fn,
)
from peal.global_utils import high_contrast_heatmap
from peal.log import get_logger

_log = get_logger(__name__)


# Debug image dumps. Previously hardcoded to a co-author's home directory; now under
# $PEAL_RUNS so the module is portable.
_DEBUG_DIR = os.path.join(os.environ.get("PEAL_RUNS", "peal_runs"), "debug")


class FluxGeneratorConfig(GeneratorConfig):
    """Config of :class:`FluxGenerator`.

    Most fields mirror the diffusers LoRA fine-tuning script arguments and are
    only consumed by :meth:`FluxGenerator.train_model`. The fields that drive
    inference and editing are listed below.

    Parameters
    ----------
    data : DataConfig
        Dataset the generator is built for; ``data.input_size`` fixes the image
        resolution used for the VAE latents.
    base_path : str
        Directory for ``config.yaml`` and fine-tuning outputs.
    pretrained_model_name_path : str
        HuggingFace id or local path passed to
        ``StableDiffusion3Pipeline.from_pretrained``.
    HUGGING_FACE_TOKEN : str or None
        Optional token saved with ``HfFolder.save_token`` before the download;
        None (default) keeps the user's existing Hugging Face login.
    prompt : str
        Text prompt encoded once per call and fed to the transformer.
    steps_number : int, default 15
        Number of flow-matching integration steps.
    shift : float, default 3.0
        Timestep shift of the sigma schedule (see ``prepare_time_steps``).
    guidance_scale : float, default 0.0
        Classifier-free guidance weight; ``> 0`` enables CFG.
    strength : float, default 1.0
        Img2img strength (only used by ``adjust_time_steps_img_2_img``).
    classifier_scale, use_logits, l1_loss, l2_loss
        Classifier-guidance temperature, logit-vs-log-softmax switch and the
        distance-loss weights used by the FastDiME guidance functions.
    self_optimized_masking : bool, default True
        Enable FastDiME's per-step mask that resets unchanged regions.
    method : str, default "FastDime"
        Edit strategy selected in :meth:`FluxGenerator.edit`: ``fastdime``,
        ``fastdime2``, ``fastdime2+``, ``fastdime_ddpm`` or ``fastdime_ddpm2``.
    guided_iterations : int
        Passed through to the FastDiME methods (currently unused there).
    offload_cpu : bool, default False
        Call ``enable_model_cpu_offload`` on the pipeline.
    task_config : TaskConfig, optional
        Overrides the task config attached to the training dataset.
    """

    generator_type: str = "DiffusionGenerator"
    """
    The type of generator that shall be used.
    """
    data: DataConfig = DataConfig()
    """
    The config of the data.
    """
    revision: Union[str, type(None)] = None
    base_path: str = "$PEAL_RUNS/stable_diffusion3"
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
    seed: Union[type(None), int] = None
    center_crop: bool = False
    random_flip: bool = False
    train_batch_size: int = 1
    num_train_epochs: int = 10
    max_train_steps: Union[int, type(None)] = 100  # None
    gradient_accumulation_steps: int = (
        1  # to manage memory and computation during training process.
    )
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    scale_lr: bool = False
    push_to_hub: bool = False
    hub_token: Union[str, type(None)] = None
    prediction_type: Union[str, type(None)] = None
    hub_model_id: Union[str, type(None)] = None
    logging_dir: Union[str, type(None)] = "logs"
    mixed_precision: Union[str, type(None)] = None
    task_config: Union[TaskConfig, type(None)] = None
    offload_cpu: bool = False
    guidance_scale: float = 0.0
    joint_attention_kwargs: Union[dict, type(None)] = None
    prompt: str = """ """
    steps_number: int = 15
    strength: float = 1.0
    guided_iterations: int = 9999999
    classifier_scale: float = 5.0
    l1_loss: float = 0.0
    l2_loss: float = 0.0
    pretrained_model_name_path: Union[str, type(None)] = "black-forest-labs/FLUX.1-dev"
    # None: use whatever Hugging Face login the user already has ($HF_TOKEN or
    # `huggingface-cli login`). Set it only to store a token of your own.
    HUGGING_FACE_TOKEN: Union[str, type(None)] = None
    enable_grade: bool = True
    self_optimized_masking: bool = True
    use_logits: bool = True
    method: str = "FastDime"
    shift: float = 3.0


class FluxGenerator(EditCapableGenerator):  # InvertibleGenerator
    """FLUX / SD3 rectified-flow generator usable as a PEAL edit generator.

    Parameters
    ----------
    config : FluxGeneratorConfig or str
        Config object or yaml path (resolved with ``load_yaml_config``).
    model_dir : str, optional
        Working directory; defaults to ``config.base_path``. ``data_test`` and
        ``counterfactuals_test`` sub-paths are derived from it.
    device : str, default "cuda"
        Device the pipeline is moved to.
    dtype : torch.dtype, default torch.float32
        Weight and latent dtype.
    classifier_dataset : PealDataset, optional
        Deep-copied; its ``task_config`` is attached to the train dataset when
        ``config.task_config`` is None.
    predictor_dataset, train
        Accepted for interface compatibility and ignored.

    Attributes
    ----------
    pipe : StableDiffusion3Pipeline
        Loaded without the T5 text encoder (``text_encoder_3=None``).
    train_dataset, val_dataset, dataset
        Datasets from ``get_datasets(config.data)``; ``dataset`` is the
        validation split and supplies ``project_to_pytorch_default``.
    classifier_free_guidance : bool
        True when ``config.guidance_scale > 0``.
    loss
        Perceptual loss handle passed as ``l_perc``; None unless set later.
    """

    def __init__(
        self,
        config,
        model_dir=None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        classifier_dataset=None,
        predictor_dataset=None,
        train=False,
    ):
        """Save the HF token, load the datasets and the diffusers pipeline."""
        super().__init__()
        self.config = load_yaml_config(config)
        _log.info("%s", "initialize generator")
        # Only an explicitly configured token is stored; the default leaves the
        # user's own Hugging Face login untouched.
        if getattr(config, "HUGGING_FACE_TOKEN", None):
            HfFolder.save_token(config.HUGGING_FACE_TOKEN)
        self.train_dataset, self.val_dataset, _ = get_datasets(self.config.data)
        self.dataset = self.val_dataset
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.guided_iterations = self.config.guided_iterations
        self.classifier_dataset = copy.deepcopy(classifier_dataset)
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

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.config.pretrained_model_name_path,
            torch_dtype=self.dtype,
            text_encoder_3=None,
            tokenizer_3=None,
        ).to(device)
        self.prompt = self.config.prompt
        self.self_optamized_masking = self.config.self_optimized_masking
        self.steps_number = self.config.steps_number
        self.guidance_scale = self.config.guidance_scale
        self.classifier_free_guidance = True if self.guidance_scale > 0.0 else False
        self.strength = self.config.strength
        if self.config.offload_cpu:
            self.pipe.enable_model_cpu_offload()
        self.to_idle = lambda x: x.to("cpu")
        self.to_cuda = lambda x: x.to("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_grade = self.config.enable_grade
        self.classifier_scale = self.config.classifier_scale
        self.use_logits = self.config.use_logits
        self.l1_loss = self.config.l1_loss
        self.l2_loss = self.config.l2_loss
        self.loss = None
        self.method = self.config.method
        self.shift = self.config.shift
        _log.info("%s", "initializing generator done!")

    def rescale_time_step(self, timestep):
        """Map a sigma in ``[0, 1]`` to the transformer's ``[0, 1000]`` timestep."""
        x_min = 0.0
        x_max = 1.0
        y_min = 0.0
        y_max = 1000.0
        scaled = (timestep - x_min) * (y_max - y_min) / (x_max - x_min) + y_min
        return scaled

    def prepare_time_steps(self, num_inference_steps=10, end_time=1000, shift=3):
        """Build the shifted, descending sigma schedule of the flow.

        Parameters
        ----------
        num_inference_steps : int, default 10
            Number of integration steps.
        end_time : float, default 1000
            Largest training timestep (out of 1000) to start from; ``t * 1000``
            is used to start the edit at a partial noise level.
        shift : float, default 3
            SD3 timestep shift ``sigma' = shift * sigma / (1 + (shift - 1) *
            sigma)``, applied twice (once to derive the range, once to the
            resampled steps).

        Returns
        -------
        torch.Tensor
            Shape ``(num_inference_steps + 1,)``: sigmas from the largest down
            to the smallest, followed by a trailing ``0``.
        """
        num_training_steps = 1000
        timesteps = torch.linspace(1, end_time, num_training_steps).flip(0)
        sigmas = timesteps / num_training_steps

        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()
        timesteps = torch.linspace(
            num_training_steps * sigma_max,
            num_training_steps * sigma_min,
            num_inference_steps,
        )
        sigmas = timesteps / num_training_steps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        return sigmas

    # @torch.no_grad()
    def latent_encoder(
        self,
        img: torch.Tensor,
    ):
        """Encode a ``[0, 1]`` image batch into scaled VAE latents.

        Parameters
        ----------
        img : torch.Tensor
            Shape ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Sampled latent ``(z - shift_factor) * scaling_factor`` of shape
            ``(B, C, H / vae_scale_factor, W / vae_scale_factor)``.
        """

        im_tensor = 2.0 * img - 1.0
        # print(im_tensor.dtype)
        latent_image = self.pipe.vae.encode(im_tensor.to(self.dtype))
        latent_model_input = latent_image.latent_dist.sample()
        latent_model_input = (
            latent_model_input - self.pipe.vae.config.shift_factor
        ) * self.pipe.vae.config.scaling_factor
        return latent_model_input

    def latent_decoder(
        self,
        latent: torch.tensor,
        img: bool = False,
    ):
        """Decode scaled VAE latents back to an image in ``[0, 1]``.

        Parameters
        ----------
        latent : torch.Tensor
            Shape ``(B, C, h, w)``; only the first element of the batch is
            returned.
        img : bool, default False
            Return a clamped ``PIL.Image`` instead of a tensor.

        Returns
        -------
        torch.Tensor or PIL.Image.Image
            Tensor of shape ``(1, 3, H, W)`` (not clamped), or the PIL image.

        Notes
        -----
        Enables gradient checkpointing on the VAE and empties the CUDA cache.
        """

        self.pipe.vae.enable_gradient_checkpointing()
        latent = (
            latent / self.pipe.vae.config.scaling_factor
        ) + self.pipe.vae.config.shift_factor
        decoded = self.pipe.vae.decode(latent)
        decoded = (decoded.sample / 2.0 + 0.5)[0]
        # print(
        #     f"memory summary after decoding, empty_cache and delete unused tensors: \n",
        #     torch.cuda.memory_summary(device=None, abbreviated=False),
        # )
        if img:
            to_pil = ToPILImage()
            img = to_pil(decoded.squeeze(0).clamp(0, 1))
            return img
        torch.cuda.empty_cache()
        return decoded.unsqueeze(0)

    def adjust_time_steps_img_2_img(
        self, num_inference_steps, strength, timesteps, encoding: bool
    ):
        """Truncate a schedule to the last ``strength`` fraction of its steps.

        Parameters
        ----------
        num_inference_steps : int
            Length of the full schedule.
        strength : float
            Fraction in ``[0, 1]`` of the steps to keep (img2img strength).
        timesteps : torch.Tensor
            Full schedule as returned by :meth:`prepare_time_steps`.
        encoding : bool
            If True the trailing ``0`` entry is dropped as well.

        Returns
        -------
        (torch.Tensor, int)
            The truncated schedule and the number of remaining steps.
        """

        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        time_index = int(max(num_inference_steps - init_timestep, 0))
        if encoding:
            return timesteps[time_index:-1], num_inference_steps - time_index
        return timesteps[time_index:], num_inference_steps - time_index

    @torch.no_grad()
    def forward_process(self, image, num_inference_steps):
        """Noise the image latent independently to every sigma of the schedule.

        Parameters
        ----------
        image : torch.Tensor
            Shape ``(1, 3, H, W)`` in ``[0, 1]``.
        num_inference_steps : int
            Number of sigmas (schedule uses the default shift of 3).

        Returns
        -------
        torch.Tensor
            Shape ``(num_inference_steps + 1, 1, C, h, w)``: the noised latents
            ``(1 - sigma_i) x_0 + sigma_i eps_i`` from the noisiest to the least
            noisy, followed by the clean latent ``x_0``.
        """
        # if isinstance(image, torch.Tensor):
        #     x_0 = image
        #     if x_0.dim() < 4:
        #         x_0 = x_0.unsqueeze(0)
        # else:
        x_0 = self.latent_encoder(image).to(self.dtype)
        x_ts = x_0.expand(num_inference_steps, -1, -1, -1).to(x_0.device)
        timesteps = (
            self.prepare_time_steps(num_inference_steps)[:-1]
            .to(x_0.device)
            .to(x_0.dtype)
        )
        timesteps = timesteps.view(num_inference_steps, 1, 1, 1)
        x_ts = (1 - timesteps) * x_ts + timesteps * torch.randn_like(x_ts).to(
            x_0.device
        ).to(x_0.dtype)
        return torch.cat([x_ts, x_0], dim=0).unsqueeze(1)  # xT, .................. X_0

    @torch.no_grad()
    def encode_image(self, image):
        """Edit-friendly inversion: extract per-step residual noises.

        Runs the transformer along the schedule on the independently noised
        latents from :meth:`forward_process` and stores, for every step, the
        residual ``z_t = x_{t-1} - mu(x_t)`` that :meth:`decode_image` has to add
        to reproduce the input exactly.

        Parameters
        ----------
        image : torch.Tensor
            Shape ``(1, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        x_T : torch.Tensor
            The noisiest latent, shape ``(1, C, h, w)``.
        zs : torch.Tensor
            Shape ``(steps_number, 1, C, h, w)`` residual noises; the last entry
            is zero.
        """
        latents = self.forward_process(image, self.steps_number)
        timesteps = (
            self.prepare_time_steps(
                num_inference_steps=self.steps_number, shift=self.shift
            )[:-1]
            .to(latents.device)
            .to(self.dtype)
        )
        eta = 1.0
        do_classifier_free_guidance = True if self.guidance_scale > 0.0 else False
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=self.prompt,
            prompt_2=self.prompt,
            prompt_3=self.prompt,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        xts = latents.clone()
        x_t = latents[0]
        zs = torch.zeros(
            (self.steps_number, *x_t.shape), device=x_t.device, dtype=self.dtype
        )
        x_t = torch.cat([x_t] * 2) if do_classifier_free_guidance else x_t
        for i, t in enumerate(timesteps):  # 1,............, 0
            x_t = latents[i]
            x_t = torch.cat([x_t] * 2) if do_classifier_free_guidance else x_t
            timestep = self.rescale_time_step(t).expand(x_t.shape[0]).to(self.dtype)
            try:
                noise_pred = self.pipe.transformer(
                    hidden_states=x_t,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            except Exception as error:
                _log.info("%s", error)
                raise
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            prev_sigma = timesteps[i + 1] if i < self.steps_number - 1 else 0  # t-1
            sigma = t

            dt = prev_sigma - sigma
            # predicted_x0 = (1 - dt/(1-sigma)) * (x_t - noise_pred) if sigma < 1 else (1 - dt)*(x_t - noise_pred)
            predicted_x0 = x_t - sigma * noise_pred
            # st_dev = -dt / (1 - sigma) * (prev_sigma / sigma)
            # std = eta * st_dev
            # direction_pointing_to_xt = (prev_sigma + std) * noise_pred
            direction_pointing_to_xt = prev_sigma * noise_pred
            mu_xt = predicted_x0 + direction_pointing_to_xt
            prev_xt = (
                torch.cat([latents[i + 1]] * 2)
                if do_classifier_free_guidance
                else latents[i + 1]
            )
            z_t = (
                prev_xt - mu_xt
            )  # /(std) if prev_sigma > 0 else torch.zeros_like(mu_xt, dtype=mu_xt.dtype)
            # print("z_t dtype", z_t.dtype)
            x_t = mu_xt + ((z_t))
            z_in = z_t
            x_in = x_t
            if do_classifier_free_guidance:
                z_in, _ = z_t.chunk(2)
                x_in, _ = x_t.chunk(2)
            # latent_decoder(x_in, img=True).resize((256,256)).show()
            # latent_decoder(z_in, img=True).resize((256,256)).show()
            # latent_decoder(z_in - x_in, img=True).resize((256,256)).show()
            zs[i] = z_in
            xts[i + 1] = x_in
            # print("sigma", t)
            # # print("std", std)
            # print("prev_sigma", prev_sigma)
            # print("dt", dt)
        zs[-1] = torch.zeros_like(zs[-1])
        return xts[0], zs

    def decode_image(self, xts, zts, eta: float = 1.0):
        """Re-generate an image from ``x_T`` and the stored residual noises.

        Inverse of :meth:`encode_image`.

        Parameters
        ----------
        xts : torch.Tensor
            Starting latent ``x_T`` of shape ``(1, C, h, w)``.
        zts : torch.Tensor
            Residual noises ``(steps_number, 1, C, h, w)`` added after every
            Euler step.
        eta : float, default 1.0
            Unused; kept for DDIM-style signature compatibility.

        Returns
        -------
        torch.Tensor
            Decoded image of shape ``(1, 3, H, W)``.
        """
        timesteps = (
            self.prepare_time_steps(
                num_inference_steps=self.steps_number, shift=self.shift
            )
            .to(xts.device)
            .to(self.dtype)
        )
        do_classifier_free_guidance = True if self.guidance_scale > 0.0 else False
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=self.prompt,
            prompt_2=self.prompt,
            prompt_3=self.prompt,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        variance_noises = zts.to(self.dtype)
        x_t = xts
        x_t = torch.cat([x_t] * 2) if do_classifier_free_guidance else x_t
        self.pipe.transformer.enable_gradient_checkpointing()
        for i, t in enumerate(timesteps):  # 1,....., , 0
            # print(i, t)
            timestep = self.rescale_time_step(t).expand(x_t.shape[0])
            # try:
            noise_pred = self.pipe.transformer(
                hidden_states=x_t,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            # except:
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            prev_sigma = timesteps[i + 1] if i < self.steps_number - 1 else 0  # t-1
            sigma = t

            dt = prev_sigma - sigma
            dt = prev_sigma - sigma
            # predicted_x0 = (1 - dt/(1-t)) * (x_t - noise_pred) if sigma < 1 else (x_t - noise_pred)
            predicted_x0 = x_t - sigma * noise_pred
            # st_dev = dt / (1 - prev_sigma) * (prev_sigma / sigma)
            # std = eta * st_dev
            # direction_pointing_to_xt = (prev_sigma + std) * noise_pred
            direction_pointing_to_xt = prev_sigma * noise_pred
            x_t = predicted_x0 + direction_pointing_to_xt
            variance_noise = (
                variance_noises[i]
                if i < self.steps_number - 1
                else torch.zeros_like(x_t).to(self.dtype)
            )
            sigma_z = variance_noise
            # latent_decoder(sigma_z, img=True).resize((256,256)).show()
            x_t = x_t + sigma_z.to(x_t.device)
            torch.cuda.empty_cache()
            # self.latent_decoder(x_t, img=True).resize((256,256)).show()
        if do_classifier_free_guidance:
            x_t, _ = x_t.chunk(2, dim=0)
        return self.latent_decoder(x_t)

    def normalize_tensor(self, tensor):
        """Standardise a tensor to zero mean and unit standard deviation."""
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        return (tensor - mean) / std

    def encode(
        self, x: torch.tensor, t: float = 1.0, stochastic=None, num_steps=None
    ) -> torch.tensor:
        """Encode an image to a noisy latent at noise level ``t``.

        The VAE latent is mixed with fresh Gaussian noise at the largest sigma
        of the schedule ending at ``t * 1000`` (the DDIM-style inversion loop is
        commented out).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 3, H, W)`` in ``[0, 1]``.
        t : float, default 1.0
            Fraction of the diffusion time; ``1.0`` yields (almost) pure noise.
        stochastic, num_steps
            Unused.

        Returns
        -------
        torch.Tensor
            Latent of shape ``(B, C, h, w)`` with ``requires_grad`` enabled.
        """
        sigmas = self.prepare_time_steps(
            num_inference_steps=self.steps_number, end_time=t * 1000, shift=self.shift
        )
        num_inference_steps = self.steps_number
        sigmas = sigmas.to(x.device).to(x.dtype)
        # in case of using strength do decide the final time step.
        sigmas = sigmas.flip(0)
        z = self.latent_encoder(x)

        noise = torch.randn(
            z.shape,
            dtype=z.dtype,
            device=z.device,
            generator=None,
            layout=torch.strided,
        )
        sigma = sigmas[-1]
        z_t = noise * sigma + (1.0 - sigma) * z

        # in case of using ddim inversion"""

        # torch.cuda.empty_cache()

        return z_t.requires_grad_()

    def decode(
        self,
        z: torch.tensor,
        t: float = 1.0,
        stochastic=None,
        num_steps=None,
    ) -> torch.tensor:
        """Integrate the flow from a noisy latent back to an image.

        Euler steps ``z <- z + (sigma_{i+1} - sigma_i) * v(z, sigma_i)`` over
        ``steps_number`` sigmas of the schedule ending at ``t * 1000``, with
        classifier-free guidance when enabled.

        Parameters
        ----------
        z : torch.Tensor or sequence
            ``z[0]`` is the starting latent ``(C, h, w)`` or ``(1, C, h, w)``.
        t : float, default 1.0
            Fraction of the diffusion time the latent was encoded to.
        stochastic, num_steps
            Unused.

        Returns
        -------
        torch.Tensor
            Decoded image of shape ``(1, 3, H, W)``.
        """
        torch.cuda.empty_cache()
        sigmas = self.prepare_time_steps(
            self.steps_number, end_time=t * 1000, shift=self.shift
        )
        # sigmas, num_inference_steps = self.adjust_time_steps_img_2_img(
        #     self.steps_number, self.strength, sigmas, encoding=False
        # )
        num_inference_steps = self.steps_number
        sigmas = sigmas.to(z[0].device)
        prompt = self.prompt
        # prompt = " "
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(prompt, prompt, prompt, device=z[0].device)
        z_t = z[0]
        if z_t.dim() < 4:
            z_t = z_t.unsqueeze(0)
        if self.classifier_free_guidance:
            z_t = torch.cat([z_t] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        # del negative_prompt_embeds, negative_pooled_prompt_embeds
        gc.collect()
        self.pipe.transformer.to(z[0].device)
        torch.cuda.empty_cache()
        self.pipe.transformer.enable_gradient_checkpointing()
        for i in range(num_inference_steps):
            t = self.rescale_time_step(sigmas[i])
            timestep = t.expand(z_t.shape[0])
            # try:
            noise_pred = self.pipe.transformer(
                hidden_states=z_t,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            if self.classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            dt = sigmas[i + 1] - sigmas[i]

            z_t = z_t + (dt) * noise_pred

            torch.cuda.empty_cache()

        return self.latent_decoder(z_t)

    def rbg_gray(self, rbg):
        """Convert an RGB tensor to one luminance channel.

        Accepts ``(3, H, W)`` or ``(B, 3, H, W)`` and keeps the channel axis.
        """
        rbg_grey_tensor = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
        if rbg.dim() > 3:
            return (rbg_grey_tensor * rbg).sum(dim=1, keepdim=True)
        return (rbg_grey_tensor * rbg).sum(dim=0, keepdim=True)

    @torch.no_grad()
    def repaint(
        self,
        x,
        pe,
        inpaint,
        dilation,
        t,
        stochastic,
        old_mask,
        mask_momentum,
        boolmask_in,
    ):
        """RePaint-style inpainting of the regions where ``pe`` differs from ``x``.

        Starts from the noised latent of ``pe`` and, after every one of
        ``2 * steps_number`` Euler steps, overwrites the unmasked region with a
        freshly noised copy of the original latent of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Original image ``(1, 3, H, W)`` in dataset normalisation.
        pe : torch.Tensor
            Edited image whose latent initialises the sampling.
        inpaint : float
            Threshold on the dilated difference mask; pixels with a value
            ``>= inpaint`` are regenerated.
        dilation : int
            Odd kernel size of the mask dilation.
        t : float
            Fraction of the diffusion time to start from.
        stochastic
            Unused.
        old_mask : torch.Tensor or None
            Previous mask subtracted (scaled by ``inpaint * mask_momentum``)
            from the current dilated mask.
        mask_momentum : float
            Weight of ``old_mask``.
        boolmask_in : torch.Tensor or None
            Extra RGB mask whose zero regions are forced into the inpainted set.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Decoded result ``(1, 3, H, W)`` and the binary mask on CPU.

        Notes
        -----
        Writes ``utils/delete_me/masked_image.png`` and ``latent{i}.png``
        relative to the working directory.
        """
        do_classifier_free_guidance = True if self.guidance_scale > 0.0 else False
        # timesteps
        num_inference_steps = int(self.steps_number * 2)
        strength = self.strength
        sigmas = self.prepare_time_steps(
            num_inference_steps, t * 1000, shift=self.shift
        )
        # sigmas, num_inference_steps = adjust_time_steps_img_2_img(num_inference_steps, strength, sigmas, False)
        # encoding the prompt
        prompt = self.prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        # x = transform(x)
        # pe = transform(pe)
        # prepare image latent
        x = self.dataset.project_to_pytorch_default(x)
        pe = self.dataset.project_to_pytorch_default(pe)
        x_latents = self.latent_encoder(x)
        pe_latent = self.latent_encoder(pe)
        noise = torch.randn_like(x_latents)
        t = sigmas[:1].item()
        if t == 1.0:
            latent = noise
        else:
            # latent = (1 - t) * x_latents + t * noise
            latent = (1 - t) * pe_latent + t * noise
        # preparing the mask and masked image
        mask, dil_mask = generate_smooth_mask(x, pe, dilation)

        # mask = transform(mask)
        # mask_grey = rbg_gray(mask)

        if old_mask is not None:
            old_mask = self.rbg_gray(old_mask)
            dil_mask = dil_mask - inpaint * old_mask.to(dil_mask) * mask_momentum

        boolmask = (dil_mask >= inpaint).float()
        if boolmask_in is not None:
            boolmask_in = self.rbg_gray(boolmask_in)
            boolmask = torch.minimum(
                torch.ones_like(boolmask), boolmask + 1 - boolmask_in.to(boolmask)
            )
        # to_pil = ToPILImage()
        # to_pil(boolmask.squeeze(0)).resize((256, 256)).save(
        #     "utils/delete_me/boolmask.png"
        # )
        masked_x = x * (boolmask < inpaint)
        height, width = self.config.data.input_size[1:]
        vae_scale_factor = self.pipe.vae_scale_factor

        boolmask_latent = torch.nn.functional.interpolate(
            boolmask, size=(height // vae_scale_factor, width // vae_scale_factor)
        )
        # print(latent_mask.shape)
        masked_image_latent = self.latent_encoder(masked_x)
        self.latent_decoder(masked_image_latent, img=True).resize((256, 256)).save(
            "utils/delete_me/masked_image.png"
        )
        boolmask_latent = (
            torch.cat([boolmask_latent] * 2)
            if do_classifier_free_guidance
            else boolmask_latent
        )

        # denosing loop
        for i in range(num_inference_steps):
            latent_model_input = (
                torch.cat([latent] * 2) if do_classifier_free_guidance else latent
            )
            t = 1000 * sigmas[i]
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep.to(self.device),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            # compute the previous noise sample
            dt = sigmas[i + 1] - sigmas[i]
            latent = latent + dt * noise_pred
            init_mask = boolmask_latent
            init_latents_proper = x_latents
            # print("init latent proper")
            # print(init_latents_proper.shape)
            # latent_decoder(init_latents_proper, img=True).resize((256,256)).show()
            if do_classifier_free_guidance:
                init_mask, _ = boolmask_latent.chunk(2)
            if i < len(range(num_inference_steps - 1)):
                noise_timestep = sigmas[i + 1]
                # print("init latent proper noised")
                init_latents_proper = (
                    1 - noise_timestep
                ) * init_latents_proper + noise_timestep * torch.randn_like(
                    init_latents_proper, device=self.device
                )
                # latent_decoder(init_latents_proper, img=True).resize((256,256)).show()
            try:
                latent = (
                    1 - init_mask.to(self.device)
                ) * init_latents_proper + init_mask.to(self.device) * latent
                self.latent_decoder(latent, img=True).resize((256, 256)).save(
                    f"utils/delete_me/latent{i}.png"
                )
            except:
                raise
            # print("latent")
            # latent_decoder(latent, img=True).resize((256,256)).show()
            z_updated = self.latent_decoder(latent)
        return z_updated, boolmask.cpu()

    def sample_x(self, batch_size=1):
        """Sample ``batch_size`` images from the prompt; returns ``(B, 3, H, W)``."""
        images = self.pipe(batch_size * [self.prompt]).images
        images_torch = torch.stack([ToTensor()(image) for image in images])
        return images_torch

    def sample_z(self, n="auto"):
        """Draw Gaussian latents scaled by ``config.temp`` for each z shape.

        Not functional for this generator: :meth:`calc_z_shapes` returns None,
        so the loop over shapes raises ``TypeError``.
        """
        if isinstance(n, str) and n == "auto":
            n_sample = self.config.batch

        else:
            n_sample = n

        z_sample = []
        z_shapes = self.calc_z_shapes()
        for z in z_shapes:
            z_new = torch.randn(n_sample, *z) * self.config.temp
            z_sample.append(z_new.to(self.device))

        return z_sample

    @torch.enable_grad()
    def clean_multiclass_cond_fn(
        self,
        x_t,
        y,
        resize,
        classifier,
        s,
        use_logits,
        predictor_img_size,
        lr,
        momentum,
        optimizer,
    ):
        """Classifier-guidance gradient towards the target classes.

        Decodes the latent (unless ``resize``), resizes to the predictor input
        size and differentiates ``-s * score[y]`` w.r.t. the latent, where
        ``score`` is the raw logit or the log-softmax.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, C, h, w)``, or an image when ``resize`` is True.
        y : torch.Tensor
            Target class indices, shape ``(B,)``.
        resize : bool
            True when ``x_t`` is already an image and needs no VAE decoding.
        classifier : nn.Module
            Predictor evaluated in eval mode.
        s : float
            Scale (temperature) of the loss.
        use_logits : bool
            Use logits instead of log-softmax scores.
        predictor_img_size : tuple of int
            ``(H, W)`` the decoded image is resized to.
        lr, momentum, optimizer
            An SGD / Adam optimizer is instantiated and zeroed but never
            stepped; the values otherwise have no effect.

        Returns
        -------
        torch.Tensor
            Gradient with the shape of ``x_t``.
        """
        classifier.eval()
        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        x_in = [x_in]
        if optimizer == "SGD":
            optimizer = torch.optim.SGD(
                x_in,
                lr=lr,
                momentum=momentum,
            )
        elif optimizer == "Adam":
            optimizer = torch.optim.Adam(x_in, lr=lr)
        optimizer.zero_grad()
        if resize:
            x_img = x_in[0]
        else:
            x_img = self.latent_decoder(x_in[0])
        # test_transform = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )

        # train_transform = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(224),  # Random crop and resize
        #         transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406],  # ImageNet normalization mean
        #             std=[0.229, 0.224, 0.225],
        #         ),  # ImageNet normalization std
        #     ]
        # )
        # x_img = test_transform(x_img)
        x_img = transforms.Resize(predictor_img_size)(x_img)
        classifier.eval()
        # classifier.train()
        selected = classifier(x_img)
        # classifier.eval()
        # # # Select the target logits
        if not use_logits:
            selected = F.log_softmax(selected, dim=1)
        selected = -selected[range(len(y)), y]
        selected = selected * s
        grads = torch.autograd.grad(selected.sum(), x_in)[0]
        # loss = torch.nn.CrossEntropyLoss()
        # logits_gradient = classifier(x_img) / s
        # print("target label", y)
        # losses = loss(logits_gradient, y.to("cuda"))
        # losses.backward()
        # optimizer.step()
        # grads = x_in[0].grad
        # grads = torch.autograd.grad(losses, x_in)[0]
        # self.latent_decoder(x_in[0], img=True).save(
        #     f"{_DEBUG_DIR}/decoded_parameters.png"
        # )
        # self.latent_decoder(grads, img=True).save(
        #     f"{_DEBUG_DIR}/grads_parameters.png"
        # )
        return grads

    @torch.enable_grad()
    def dist_cond_fn(
        self,
        x_tau,
        z_t,
        x_t,
        alpha_t,
        l1_loss,
        l2_loss,
        l_perc,
        scale_grads,
        lr,
        momentum,
    ):
        """Gradient of the FastDiME distance losses w.r.t. the noisy latent.

        Parameters
        ----------
        x_tau : torch.Tensor
            Reference (original) image the decoded latents are compared to.
        z_t : torch.Tensor
            Current noisy latent; L1 / L2 losses are taken on its decoding.
        x_t : torch.Tensor
            Current clean-estimate latent; the perceptual loss is taken on it.
        alpha_t : float
            Time-dependent constant dividing the perceptual gradient when
            ``scale_grads`` is set.
        l1_loss, l2_loss : float
            Weights of the L1 / L2 pixel distances (0 disables the term).
        l_perc : callable or None
            Perceptual loss ``l_perc(x_in, x_tau)``; None disables it.
        scale_grads : bool
            Divide the perceptual gradient by ``alpha_t``.
        lr, momentum
            Unused.

        Returns
        -------
        torch.Tensor or int
            Gradient w.r.t. ``z_t`` (plus the perceptual gradient w.r.t.
            ``x_t``), or ``0`` when every term is disabled.
        """

        z_in = nn.Parameter(z_t.detach().requires_grad_(True))
        x_in = nn.Parameter(x_t.detach().requires_grad_(True))

        z_img = self.latent_decoder(z_in)
        x_img = self.latent_decoder(x_in)
        try:
            m1 = (
                l1_loss * torch.norm(z_img - x_tau.to(z_in.device), p=1, dim=1).sum()
                if l1_loss != 0
                else 0
            )
            m2 = (
                l2_loss * torch.norm(z_img - x_tau.to(z_in.device), p=2, dim=1).sum()
                if l2_loss != 0
                else 0
            )
        except:
            raise
        mv = l_perc(x_in, x_tau) if l_perc is not None else 0

        if isinstance(m1 + m2 + mv, int):
            return 0

        if isinstance(m1 + m2, int):
            grads = 0
        else:
            grads = torch.autograd.grad(m1 + m2, z_in)[0]

        if isinstance(mv, int):
            return grads
        else:
            if scale_grads:
                return grads + torch.autograd.grad(mv, x_in)[0] / alpha_t
            else:
                return grads + torch.autograd.grad(mv, x_in)[0]

    def FastDiME(
        self,
        img,
        inpaint: float,
        dilation: float,
        t: float,
        guided_iterations: int,
        class_grad_kwargs: dict,
        dist_grad_kargs: dict,
        explainer_config,
        scale_grads: bool = False,
        boolmask_in=None,
    ):
        """FastDiME counterfactual generation in the flow latent space.

        The input latent is noised to level ``t`` and integrated back over
        ``steps_number`` steps. At every step the noisy latent is pushed by
        classifier guidance (:meth:`clean_multiclass_cond_fn`) plus the
        distance gradient (:meth:`dist_cond_fn`), clipped to
        ``explainer_config.gradient_clipping`` and scaled by
        ``class_grad_kwargs["lr"]``. With self-optimised masking, regions whose
        current clean estimate differs little from the input (dilated mask
        below ``explainer_config.inpaint``) are reset to the input latent.

        Parameters
        ----------
        img : torch.Tensor
            Input ``(1, 3, H, W)`` in ``[0, 1]`` at ``config.data.input_size``.
        inpaint, dilation : float
            Accepted for signature compatibility; the values actually used are
            ``explainer_config.inpaint`` / ``explainer_config.dilation``.
        t : float
            Fraction of the diffusion time the input is noised to.
        guided_iterations : int
            Unused.
        class_grad_kwargs : dict
            Keyword arguments of :meth:`clean_multiclass_cond_fn` except
            ``x_t`` / ``resize`` (``y``, ``classifier``, ``s``, ``use_logits``,
            ``lr``, ``momentum``, ``predictor_img_size``, ``optimizer``).
        dist_grad_kargs : dict
            ``l1_loss``, ``l2_loss``, ``l_perc``, ``lr``, ``momentum`` for
            :meth:`dist_cond_fn`.
        explainer_config : ExplainerConfig
            Supplies ``gradient_clipping``, ``learning_rate``, ``dilation`` and
            ``inpaint``.
        scale_grads : bool, default False
            Divide the class gradient by ``t + 5e-3``.
        boolmask_in : torch.Tensor, optional
            Fixed mask ``(1, 1, H, W)``; ``1`` marks pixels kept from the input.

        Returns
        -------
        final_image : torch.Tensor
            Decoded counterfactual ``(1, 3, H, W)``.
        x_t_steps, z_t_steps : list of torch.Tensor
            CPU copies of the clean-estimate and noisy latents before each step.

        Notes
        -----
        Saves debug PNGs (``original_img``, ``noised_image``, ``z_t{i}``,
        ``x_t{i}``, ``grads_class{i}``, ...) into ``$PEAL_RUNS/debug``.
        """
        boolmask = None
        to_pil = ToPILImage()
        class_grad_fn = (
            clean_class_cond_fn if "Multiclass" else clean_multiclass_cond_fn
        )

        class_grad_fn = self.clean_multiclass_cond_fn
        dist_fn = self.dist_cond_fn
        guidance_scale = self.guidance_scale
        num_inference_steps = self.steps_number
        do_classifier_free_guidance = self.classifier_free_guidance
        self_optimized_masking = self.self_optamized_masking
        # Prepare timesteps.
        sigmas = self.prepare_time_steps(
            num_inference_steps=num_inference_steps, end_time=t * 1000, shift=self.shift
        ).to(self.device)
        to_pil(img.squeeze(0)).save(f"{_DEBUG_DIR}/original_img.png")
        # Encode image into latent space with the VAE
        height, width = self.config.data.input_size[1:]
        vae_scale_factor = self.pipe.vae_scale_factor
        with torch.no_grad():
            latents = self.latent_encoder(img.to(self.device))
        # Initialize x_t as the current latent (will be updated iteratively)
        x_t = latents.clone()
        noise = torch.randn_like(x_t, device=self.device)
        # self.latent_decoder(noise, img=True).save(
        #     f"{_DEBUG_DIR}/noise.png"
        # )
        z_t = t * noise + (1 - t) * x_t
        self.latent_decoder(z_t, img=True).save(f"{_DEBUG_DIR}/noised_image.png")
        x_t_steps = []
        z_t_steps = []
        if do_classifier_free_guidance:
            x_t = torch.cat([x_t] * 2)
            z_t = torch.cat([z_t] * 2)

        # encoding prompts
        prompt = self.prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        # Main iterative denoising loop in latent space
        for i in range(num_inference_steps):
            x_t_steps.append(x_t.detach().cpu().clone())
            z_t_steps.append(z_t.detach().cpu().clone())
            t = self.rescale_time_step(sigmas[i])
            timestep = t.expand(x_t.shape[0])
            # Use the Transformer to predict the noise residual.
            self.pipe.transformer.enable_gradient_checkpointing()
            with torch.no_grad():
                noise_pred = self.pipe.transformer(
                    hidden_states=z_t,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # noise_pred = -guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

            # Compute guidance gradients:
            grads = 0
            if class_grad_fn is not None:
                xt_in = torch.clone(x_t.detach())
                # visualzing the original image gradient
                if i == 0:

                    grads_org = class_grad_fn(
                        x_t=img.to("cuda").requires_grad_(True),
                        resize=True,
                        **class_grad_kwargs,
                    )
                    ref = torch.zeros_like(grads_org[0])
                    grad_img = high_contrast_heatmap(ref, -grads_org)
                    to_pil(grad_img[0][0]).save(f"{_DEBUG_DIR}/grads_class_org_img.png")
                    del grads_org
                class_grad = class_grad_fn(
                    x_t=xt_in,
                    resize=False,
                    **class_grad_kwargs,
                ) / (t + 5.0e-3 if scale_grads else 1)
                grads += class_grad
                # visualize the gradient
                _log.info("%s %s", "grads norm", grads.norm(p=float("inf")))
                # self.latent_decoder(grads[0])
                ref = torch.zeros_like(grads[0]).to("cpu")
                grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))
                self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                    f"{_DEBUG_DIR}/grads_class{i}.png"
                )

                if dist_fn is not None:
                    x_t_in = torch.clone(x_t.detach())
                    z_t_in = torch.clone(z_t.detach())
                    dist_grad = self.dist_cond_fn(
                        x_tau=img,
                        z_t=z_t_in,
                        x_t=x_t_in,
                        alpha_t=(1 - t + 5.0e-3),
                        scale_grads=False,
                        **dist_grad_kargs,
                    )
                    ref = torch.zeros_like(grads).to("cpu")
                    grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))

                    self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                        f"{_DEBUG_DIR}/grads_class_dist{i}.png"
                    )
                    grads = grads + dist_grad
                    # visualize the gradient
                    _log.info("%s %s", "grads1 norm", grads.norm(p=float("inf")))

                    # grads_pixels = self.latent_decoder(grads).squeeze(0)
                    ref = torch.zeros_like(grads).to("cpu")
                    grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))

                    self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                        f"{_DEBUG_DIR}/grads_class_dist_plus{i}.png"
                    )
            # scaling the gradeint
            dt = sigmas[i + 1] - sigmas[i]
            z_t = z_t + dt * noise_pred
            x_t = z_t - (sigmas[i]) * noise_pred  # / (1 - sigmas[i])
            # visualizing the results
            self.latent_decoder(z_t, img=True).save(f"{_DEBUG_DIR}/z_t{i}.png")
            self.latent_decoder(noise_pred, img=True).save(
                f"{_DEBUG_DIR}/noise_pred{i}.png"
            )
            self.latent_decoder(x_t, img=True).save(f"{_DEBUG_DIR}/x_t{i}.png")

            norm = grads.norm(p=float("inf"))
            _log.info("%s %s", "norm", norm)
            if norm > explainer_config.gradient_clipping:
                _log.info("%s", "gradient clipping")
                rescale_factor = (
                    explainer_config.gradient_clipping
                    / norm
                    / explainer_config.learning_rate
                )
                grads = grads * rescale_factor

            # different way to calculate the grad in pixel space and move it to the VAE space, does not work

            z_t = z_t - class_grad_kwargs["lr"] * grads
            self.latent_decoder(z_t, img=True).save(
                f"{_DEBUG_DIR}/z_t_after_grad{i}.png"
            )
            # Apply FASTDiME self-optimized masking if enabled:
            if self_optimized_masking:
                _log.info("%s", "self optimized mask")
                # extract time-depedent mask (Eq. 6)
                with torch.no_grad():
                    x_0_denoised = self.latent_decoder(x_t.detach())
                mask_t, dil_mask = generate_smooth_mask(
                    img.to(self.device), x_0_denoised, explainer_config.dilation
                )
                mask_t, dil_mask = self.generate_mask(
                    img.to(self.device), x_0_denoised, explainer_config.dilation
                )

                boolmask = (dil_mask < explainer_config.inpaint).float()
                height, width = x_0_denoised.shape[2:]

                boolmask_latent = torch.nn.functional.interpolate(
                    boolmask,
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                )

                # masking denoised and sampled images (Eq. 7 & 8)
                with torch.no_grad():
                    x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents
                    noise = (
                        torch.randn_like(z_t) * sigmas[i] + (1 - sigmas[i]) * latents
                    )
                    self.latent_decoder(noise, img=True).save(
                        f"{_DEBUG_DIR}/noise_boolmask{i}.png"
                    )
                    z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise
                    self.latent_decoder(z_t, img=True).save(
                        f"{_DEBUG_DIR}/zt_after_boolmask{i}.png"
                    )
            # apply masking with fixed mask when available with our 2-step approaches
            if boolmask_in is not None:
                # fixed mask

                height, width = img.shape[2:]
                boolmask = boolmask_in
                boolmask_latent = torch.nn.functional.interpolate(
                    boolmask,
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                )
                # masking denoised and sampled images (Eq. 7 & 8)
                with torch.no_grad():
                    x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents
                    noise = (
                        torch.randn_like(z_t) * sigmas[i] + (1 - sigmas[i]) * latents
                    )
                    self.latent_decoder(noise, img=True).save(
                        f"{_DEBUG_DIR}/noise_boolmask{i}.png"
                    )
                    self.latent_decoder(boolmask_latent * noise, img=True).save(
                        f"{_DEBUG_DIR}/noise_x_boolmask{i}.png"
                    )
                    self.latent_decoder(z_t * (1 - boolmask_latent), img=True).save(
                        f"{_DEBUG_DIR}/zt_x_boolmask{i}.png"
                    )
                    z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise
                    self.latent_decoder(z_t, img=True).save(
                        f"{_DEBUG_DIR}/zt_after_boolmask{i}.png"
                    )

                # allowing to set a threshold to stop after reaching oin target confidence
                """with torch.no_grad():
                    predictor = class_grad_kwargs["classifier"]
                    counterfactual = self.latent_decoder(z_t)
                    preds = torch.nn.Softmax(dim=-1)(
                        predictor(counterfactual.to(self.device)).detach().cpu()
                    )
                    print("pred", preds)
                    y_target_end_confidence = torch.zeros([img.shape[0]])
                    target_classes = class_grad_kwargs["y"]
                    for i in range(img.shape[0]):

                        y_target_end_confidence[i] = preds[i, target_classes[i]]
                        if y_target_end_confidence[i] > 0.5:"""
            # clean up some variables if needed
            """del (x_t_in, z_t_in, noise, boolmask, boolmask_latent, x_0_denoised)"""
        with torch.no_grad():
            if boolmask is not None:

                z_t = z_t * (1 - boolmask_latent) + boolmask_latent * latents
                self.latent_decoder(z_t, img=True).save(f"{_DEBUG_DIR}/zt_final{i}.png")
            final_image = self.latent_decoder(z_t)

        return final_image, x_t_steps, z_t_steps

    def FastDiME_DDPM(
        self,
        img,
        inpaint: float,
        dilation: float,
        t: float,
        guided_iterations: int,
        class_grad_kwargs: dict,
        dist_grad_kargs: dict,
        explainer_config,
        scale_grads: bool = False,
        boolmask_in=None,
    ):
        """FastDiME variant that starts from the edit-friendly inversion.

        Same guidance and masking scheme as :meth:`FastDiME`, but the noisy
        latent is initialised with ``x_T`` from :meth:`encode_image` and the
        stored residual noises are added after every Euler step so that an
        unguided run reproduces the input. Classifier-free guidance here uses
        ``-guidance_scale * (v_text - v_uncond)`` as the velocity, and the
        distance loss compares against the input latent rather than the image.
        Parameters and return values are those of :meth:`FastDiME`.
        """
        boolmask = None
        to_pil = ToPILImage()

        class_grad_fn = self.clean_multiclass_cond_fn
        dist_fn = self.dist_cond_fn
        guidance_scale = self.guidance_scale
        num_inference_steps = self.steps_number
        do_classifier_free_guidance = self.classifier_free_guidance
        self_optimized_masking = self.self_optamized_masking
        # Prepare timesteps.
        sigmas = self.prepare_time_steps(
            num_inference_steps=num_inference_steps, end_time=t * 1000, shift=self.shift
        ).to(self.device)
        # Encode image into latent space with the VAE

        height, width = self.config.data.input_size[1:]
        vae_scale_factor = self.pipe.vae_scale_factor
        with torch.no_grad():
            xts, variance_noises = self.encode_image(img.to(self.device))
            # for idx, var in enumerate(variance_noises):
            #     self.latent_decoder(var, img=True).save(
            #         f"{_DEBUG_DIR}/variance{idx}.png"
            #     )
            latents = self.latent_encoder(img.to(self.device))
        # Initialize x_t as the current latent (will be updated iteratively)
        x_t = latents.clone()
        noise = torch.randn_like(x_t, device=self.device)

        # z_t = t * noise + (1 - t) * x_t
        z_t = xts
        x_t_steps = []
        z_t_steps = []
        if do_classifier_free_guidance:
            x_t = torch.cat([x_t] * 2)
            z_t = torch.cat([z_t] * 2)

        # encoding prompts
        prompt = self.prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        # Main iterative denoising loop in latent space
        for i in range(num_inference_steps):
            x_t_steps.append(x_t.detach().cpu().clone())
            z_t_steps.append(z_t.detach().cpu().clone())
            t = self.rescale_time_step(sigmas[i])
            timestep = t.expand(x_t.shape[0])
            # Use the Transformer to predict the noise residual.
            self.pipe.transformer.enable_gradient_checkpointing()
            with torch.no_grad():
                noise_pred = self.pipe.transformer(
                    hidden_states=z_t,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = -guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute guidance gradients:
            grads = 0
            if class_grad_fn is not None:
                xt_in = torch.clone(x_t.detach())
                # visualzing the original image gradient
                class_grad = class_grad_fn(
                    x_t=xt_in,
                    resize=False,
                    **class_grad_kwargs,
                ) / (t + 5.0e-3 if scale_grads else 1)
                grads += class_grad
                # visualize the gradient
                with torch.no_grad():
                    _log.info("%s %s", "grads norm", grads.norm(p=float("inf")))

                    ref = torch.zeros_like(grads[0]).to("cpu")
                    grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))
                    self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                        f"{_DEBUG_DIR}/grads_class{i}.png"
                    )

                if dist_fn is not None:
                    x_t_in = torch.clone(x_t.detach())
                    z_t_in = torch.clone(z_t.detach())
                    grads = grads + self.dist_cond_fn(
                        x_tau=latents,
                        z_t=z_t_in,
                        x_t=x_t_in,
                        alpha_t=t,
                        scale_grads=False,
                        **dist_grad_kargs,
                    )
                    # visualize the gradient

            dt = sigmas[i + 1] - sigmas[i]
            z_t = z_t + dt * noise_pred
            x_t = z_t - (sigmas[i]) * noise_pred  # / (1 - sigmas[i])

            variance_noise = (
                variance_noises[i]
                if i < self.steps_number - 1
                else torch.zeros_like(x_t).to(self.dtype)
            )
            sigma_z = variance_noise
            z_t = z_t + sigma_z
            # visualizing the results
            self.latent_decoder(z_t, img=True).save(f"{_DEBUG_DIR}/z_t{i}.png")
            self.latent_decoder(noise_pred, img=True).save(
                f"{_DEBUG_DIR}/noise_pred{i}.png"
            )
            self.latent_decoder(x_t, img=True).save(f"{_DEBUG_DIR}/x_t{i}.png")
            # scaling the gradeint
            norm = grads.norm(p=float("inf"))
            if norm > explainer_config.gradient_clipping:
                _log.info("%s", "gradient clipping")
                rescale_factor = (
                    explainer_config.gradient_clipping
                    / norm
                    / explainer_config.learning_rate
                )
                grads = grads * rescale_factor
                _log.info("%s %s", "after scaling norm", grads.norm(p=float("inf")))

            # different way to calculate the grad in pixel space and move it to the VAE space, does not work

            z_t = z_t - class_grad_kwargs["lr"] * grads
            # Apply FASTDiME self-optimized masking if enabled:
            if self_optimized_masking:
                # extract time-depedent mask (Eq. 6)
                with torch.no_grad():
                    x_0_denoised = self.latent_decoder(x_t.detach())
                mask_t, dil_mask = self.generate_mask(
                    img.to(self.device), x_0_denoised, explainer_config.dilation
                )
                boolmask = (dil_mask < explainer_config.inpaint).float()
                height, width = x_0_denoised.shape[2:]

                boolmask_latent = torch.nn.functional.interpolate(
                    boolmask,
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                )

                # masking denoised and sampled images (Eq. 7 & 8)
                x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents
                noise = torch.randn_like(z_t) * sigmas[i] + (1 - sigmas[i]) * latents
                z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise

            # apply masking with fixed mask when available with our 2-step approaches
            if boolmask_in is not None:
                # fixed mask

                height, width = img.shape[2:]
                boolmask = boolmask_in
                boolmask_latent = torch.nn.functional.interpolate(
                    boolmask,
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                )
                # masking denoised and sampled images (Eq. 7 & 8)
                x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents
                noise = torch.randn_like(z_t) * sigmas[i] + (1 - sigmas[i]) * latents
                z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise

                # allowing to set a threshold to stop after reaching oin target confidence
                """with torch.no_grad():
                    predictor = class_grad_kwargs["classifier"]
                    counterfactual = self.latent_decoder(z_t)
                    preds = torch.nn.Softmax(dim=-1)(
                        predictor(counterfactual.to(self.device)).detach().cpu()
                    )
                    print("pred", preds)
                    y_target_end_confidence = torch.zeros([img.shape[0]])
                    target_classes = class_grad_kwargs["y"]
                    for i in range(img.shape[0]):

                        y_target_end_confidence[i] = preds[i, target_classes[i]]
                        if y_target_end_confidence[i] > 0.5:"""
            # clean up some variables if needed
            """del (x_t_in, z_t_in, noise, boolmask, boolmask_latent, x_0_denoised)"""
        with torch.no_grad():
            if boolmask is not None:
                z_t = z_t * (1 - boolmask_latent) + boolmask_latent * latents
            final_image = self.latent_decoder(z_t)

        return final_image, x_t_steps, z_t_steps

    @torch.no_grad()
    def generate_mask(self, x1, x2, dilation):
        """ACE-style difference mask between two images.

        Both inputs are mapped with ``(x + 1) / 2`` (i.e. assumed in
        ``[-1, 1]``), the per-pixel absolute difference is summed over channels
        and normalised by its maximum per sample, then dilated with max pooling.

        Parameters
        ----------
        x1, x2 : torch.Tensor
            Images of shape ``(B, 3, H, W)`` (callers pass the input and the
            current denoised estimate).
        dilation : int
            Odd max-pool kernel size.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            ``mask`` and ``dil_mask``, both ``(B, 1, H, W)`` in ``[0, 1]``.
        """
        assert (dilation % 2) == 1, "dilation must be an odd number"
        x1 = (x1 + 1) / 2
        x2 = (x2 + 1) / 2
        mask = (x1 - x2).abs().sum(dim=1, keepdim=True)
        mask = mask / mask.view(mask.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        dil_mask = F.max_pool2d(mask, dilation, stride=1, padding=(dilation - 1) // 2)
        return mask, dil_mask

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        predictor: nn.Module,
        explainer_config: ExplainerConfig,
        predictor_datasets,
        pbar: object = None,
        mode: object = "",
        base_path: object = "",
    ) -> Tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """
        This function edits the input to match the target confidence goal and target classes
        Args:
            predictor_datasets:
            explainer_config:
            base_path:
            x_in: The input
            target_confidence_goal: The target confidence goal
            source_classes: The source classes
            target_classes: The target classes
            predictor: The predictor according to which the confidence is measured
            pbar: A progress bar
            mode: The mode of the edit. This is used to determine the edit method

        Returns:
            list[torch.Tensor]: List of the counterfactuals
            list[torch.Tensor]: List of the differences in latent codes. In the simplest case just x_in - x_counterfactual
            list[torch.Tensor]: List of the achieved target confidences of the counterfactuals
            list[torch.Tensor]: List of x_in. This is necessary since the counterfactuals might be in a different order
        If not implemented, it will throw a NotImplementedError.
        """
        inpaint = explainer_config.inpaint
        dilation = explainer_config.dilation
        t = explainer_config.sampling_time_fraction
        predictor_data_size = predictor_datasets[0].dataset.config.input_size[1:]
        # t = 0.5
        to_pil = ToPILImage()
        transformed = torch.zeros_like(source_classes).bool()
        class_grad_kwargs = {
            "y": target_classes,
            "classifier": predictor,
            "s": explainer_config.temperature,
            "use_logits": self.use_logits,
            "lr": explainer_config.learning_rate,
            "momentum": explainer_config.momentum,
            "predictor_img_size": predictor_data_size,
            "optimizer": explainer_config.optimizer,
        }
        dist_grad_kargs = {
            "l1_loss": self.l1_loss,
            "l2_loss": self.l2_loss,
            "l_perc": self.loss,
            "lr": explainer_config.learning_rate,
            "momentum": explainer_config.momentum,
        }
        x_in = transforms.Resize(self.config.data.input_size[1:])(x_in)

        if self.method.lower() == "fastdime2":
            self.self_optamized_masking = False
            # no masking included in first step
            x_counterfactuals, x_steps, z_steps = self.FastDiME(
                img=x_in,
                inpaint=inpaint,
                dilation=dilation,
                t=t,
                guided_iterations=self.guided_iterations,
                class_grad_kwargs=class_grad_kwargs,
                dist_grad_kargs=dist_grad_kargs,
                explainer_config=explainer_config,
                scale_grads=True,
                boolmask_in=None,
            )

            mask, dil_mask = generate_smooth_mask(
                x_in.to("cuda"), x_counterfactuals, dilation
            )
            mask, dil_mask = self.generate_mask(
                x_in.to("cuda"), x_counterfactuals, dilation
            )
            _log.info("%s", "using fixed mask")
            boolmask = (dil_mask < inpaint).float()
            to_pil(boolmask.squeeze(0).clamp(0, 1)).save(
                f"{_DEBUG_DIR}/boolmask_{mode}.png"
            )
            to_pil(x_counterfactuals.squeeze(0).clamp(0, 1)).save(
                f"{_DEBUG_DIR}/x_counterfactuals_{mode}.png"
            )
            class_grad_kwargs["lr"] = class_grad_kwargs["lr"] * 10
            x_counterfactuals, x_steps, z_steps = self.FastDiME(
                img=x_in,
                inpaint=inpaint,
                dilation=dilation,
                t=t,
                guided_iterations=self.guided_iterations,
                class_grad_kwargs=class_grad_kwargs,
                dist_grad_kargs=dist_grad_kargs,
                explainer_config=explainer_config,
                scale_grads=False,
                boolmask_in=boolmask,
            )

        elif self.method.lower() == "fastdime2+":
            # masks are used in first step
            self.self_optamized_masking = True
            x_counterfactuals, x_steps, z_steps = self.FastDiME(
                img=x_in,
                inpaint=inpaint,
                dilation=dilation,
                t=t,
                guided_iterations=self.guided_iterations,
                class_grad_kwargs=class_grad_kwargs,
                dist_grad_kargs=dist_grad_kargs,
                explainer_config=explainer_config,
                scale_grads=False,
                boolmask_in=None,
            )
            mask, dil_mask = generate_smooth_mask(
                x_in.to("cuda"), x_counterfactuals, explainer_config.dilation
            )
            boolmask = (dil_mask < explainer_config.inpaint).float()
            _log.info("%s", "using fixed mask")
            to_pil(boolmask.squeeze(0).clamp(0, 1)).save(
                f"{_DEBUG_DIR}/boolmask_{mode}.png"
            )
            to_pil(x_counterfactuals.squeeze(0).clamp(0, 1)).save(
                f"{_DEBUG_DIR}/x_counterfactuals_{mode}.png"
            )
            class_grad_kwargs["lr"] = class_grad_kwargs["lr"] * 10
            self.self_optamized_masking = False
            x_counterfactuals, x_steps, z_steps = self.FastDiME(
                img=x_in,
                inpaint=inpaint,
                dilation=dilation,
                t=t,
                guided_iterations=self.guided_iterations,
                class_grad_kwargs=class_grad_kwargs,
                dist_grad_kargs=dist_grad_kargs,
                explainer_config=explainer_config,
                scale_grads=False,
                boolmask_in=boolmask,
            )
        elif self.method.lower() == "fastdime":
            # FastDime one iteration
            x_counterfactuals, x_steps, z_steps = self.FastDiME(
                img=x_in,
                inpaint=inpaint,
                dilation=dilation,
                t=t,
                guided_iterations=self.guided_iterations,
                class_grad_kwargs=class_grad_kwargs,
                dist_grad_kargs=dist_grad_kargs,
                explainer_config=explainer_config,
                scale_grads=False,
                boolmask_in=None,
            )
        elif self.method.lower() == "fastdime_ddpm":
            x_counterfactuals, x_steps, z_steps = self.FastDiME_DDPM(
                img=x_in,
                inpaint=inpaint,
                dilation=dilation,
                t=t,
                guided_iterations=self.guided_iterations,
                class_grad_kwargs=class_grad_kwargs,
                dist_grad_kargs=dist_grad_kargs,
                explainer_config=explainer_config,
                scale_grads=False,
                boolmask_in=None,
            )
        elif self.method.lower() == "fastdime_ddpm2":
            self.self_optamized_masking = False
            x_counterfactuals, x_steps, z_steps = self.FastDiME_DDPM(
                img=x_in,
                inpaint=inpaint,
                dilation=dilation,
                t=t,
                guided_iterations=self.guided_iterations,
                class_grad_kwargs=class_grad_kwargs,
                dist_grad_kargs=dist_grad_kargs,
                explainer_config=explainer_config,
                scale_grads=False,
                boolmask_in=None,
            )
            mask, dil_mask = generate_smooth_mask(
                x_in.to("cuda"), x_counterfactuals, explainer_config.dilation
            )

            boolmask = (dil_mask < explainer_config.inpaint).float()
            to_pil(boolmask.squeeze(0).clamp(0, 1)).save(f"{_DEBUG_DIR}/boolmask.png")
            self.self_optamized_masking = False
            x_counterfactuals, x_steps, z_steps = self.FastDiME_DDPM(
                img=x_in,
                inpaint=inpaint,
                dilation=dilation,
                t=t,
                guided_iterations=self.guided_iterations,
                class_grad_kwargs=class_grad_kwargs,
                dist_grad_kargs=dist_grad_kargs,
                explainer_config=explainer_config,
                scale_grads=True,
                boolmask_in=boolmask,
            )
        to_pil = ToPILImage()
        # final counterfactual viz
        x_counterfactuals = transforms.Resize(predictor_data_size)(x_counterfactuals)
        x_in = transforms.Resize(predictor_data_size)(x_in)
        preds = torch.nn.Softmax(dim=-1)(
            predictor(x_counterfactuals.to(self.device)).detach().cpu()
        )

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        _log.info("%s", "edit done")
        return (
            list(x_counterfactuals.cpu()),
            list(x_in - x_counterfactuals.cpu()),
            list(y_target_end_confidence),
            list(x_in),
            None,
        )

    def log_prob_z(self, z):
        """Not available for this generator; returns None."""
        return None

    def calc_z_shapes(self):
        """Not available for this generator; returns None."""
        return None

    def train_model(
        self,
    ):
        """LoRA fine-tune the pipeline on the training dataset.

        Writes ``config.yaml`` into ``config.base_path`` and calls the vendored
        ``lora_finetune`` with the config fields, ``train_dataset`` and
        ``resume_from_checkpoint="latest"``. Note that it reads
        ``self.pipeline``, which the constructor never sets (the pipeline is
        stored as ``self.pipe``), so it fails with ``AttributeError`` as is.
        """
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

        _log.info("%s", "Start LORA finetuning")
        _log.info("%s", "Start LORA finetuning")
        _log.info("%s", "Start LORA finetuning")
        self.pipeline = lora_finetune(finetune_args)
        _log.info("%s", "Finished LORA finetuning")
        _log.info("%s", "Finished LORA finetuning")
        _log.info("%s", "Finished LORA finetuning")
