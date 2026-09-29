"""Stable Diffusion 3 generator for guided counterfactual edits.

This module wraps the diffusers ``StableDiffusion3Pipeline`` as a PEAL
``EditCapableGenerator``. Images are encoded into the SD3 VAE latent space, the
rectified-flow transformer is driven with an explicit Euler sigma schedule, and
the denoising is steered by classifier gradients (FastDiME-style guidance with
self-optimised masks, an edit-friendly DDPM inversion variant and direct latent
optimisation). Optional gradient-smoothing attributions (zennit LRP, captum
SmoothGrad/DeepLift/IG, SmoothDiff, manifold-aware smoothing) can replace the raw
classifier gradient. ``train_model`` delegates to the LoRA fine-tuning script.
"""

import torch
import torchvision
from torchvision.transforms import ToTensor, Normalize
from torch import nn
from zennit.attribution import Gradient
from zennit.core import Stabilizer, Composite
from zennit.rules import Epsilon
from zennit.types import Convolution
from zennit.torchvision import ResNetCanonizer
import copy
import math
import os
import types
from pathlib import Path
from huggingface_hub.hf_api import HfFolder
from zennit.composites import (
    NameLayerMapComposite,
    layer_map_base,
)
from zennit.rules import Gamma
from zennit.rules import NoMod
from torchvision.transforms import transforms, ToPILImage
from peal.dependencies.lora.train_text_to_image_lora import lora_finetune
from peal.global_utils import load_yaml_config, save_yaml_config
from zennit.core import BasicHook
from peal.training.trainers import distill_predictor
from diffusers import StableDiffusion3Pipeline
from peal.architectures.interfaces import TaskConfig
from peal.generators.interfaces import EditCapableGenerator
from peal.global_utils import generate_smooth_mask
from peal.data.dataset_factory import get_datasets
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
from peal.dependencies.smoothdiff_experiments.python.smoothdiff import *
from captum.attr import DeepLift, NoiseTunnel
from peal.generators.deeplift_resnet import *
from captum.attr import Saliency
from peal.log import get_logger

_log = get_logger(__name__)


# Debug image dumps. Previously hardcoded to a co-author's home directory; now under
# $PEAL_RUNS so the module is portable.
_DEBUG_DIR = os.path.join(os.environ.get("PEAL_RUNS", "peal_runs"), "debug")

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)


class DiffusionGeneratorConfig(GeneratorConfig):
    """Configuration of the Stable Diffusion 3 based ``DiffusionGenerator``.

    The fields fall into four groups. The LoRA fine-tuning arguments
    (``pretrained_model_name_path``, ``revision``, ``train_batch_size``,
    ``num_train_epochs``, ``learning_rate``, ``mixed_precision`` ...) mirror the
    command line of the diffusers text-to-image LoRA script and are forwarded to
    ``lora_finetune`` by ``DiffusionGenerator.train_model``. The sampling fields
    (``prompt``, ``steps_number``, ``guidance_scale``, ``shift``, ``strength``,
    ``offload_cpu``) control the SD3 pipeline. The counterfactual fields select and
    tune the edit loop: ``method`` (``FastDime``, ``fastdime2``, ``fastdime2+``,
    ``fastdime_ddpm``, ``fastdime_ddpm2`` or ``ddpm_inversion``),
    ``gradient_smoothing`` (``distilled``, ``vanilla``, ``lrp``, ``smoothdiff``,
    ``smoothgrad``, ``manifold_aware``, ``integrated_gradient``,
    ``smooth_integrated_gradient``, ``smoothgrad_integrated_gradient`` or
    ``deep_lift``), the guidance weights ``classifier_scale``, ``l1_loss``,
    ``l2_loss``, ``use_logits``, ``grad_threshold``, ``warmup_steps``,
    ``use_momentum`` and the masking switches ``self_optimized_masking`` and
    ``use_gussian_blur_masking``. The ``parameters_*`` dictionaries hold the
    keyword arguments of the individual smoothing methods and
    ``parameters_rewight`` the multipliers applied on retry attempts when
    ``adjust_parameters`` is set.

    Attributes
    ----------
    generator_type : str
        Always ``"DiffusionGenerator"``; used by the generator factory.
    data : DataConfig
        Dataset config from which the train/val datasets and the input size
        (``data.input_size``) are taken.
    base_path : str
        Run directory (``$PEAL_RUNS/stable_diffusion3`` by default) that holds the
        written ``config.yaml``, the cached IG baseline and the LoRA outputs.
    task_config : TaskConfig or None
        If given, overrides the task config of the training dataset.
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
    pretrained_model_name_path: Union[str, type(None)] = (
        "stabilityai/stable-diffusion-3-medium-diffusers"
    )
    # None: use whatever Hugging Face login the user already has ($HF_TOKEN or
    # `huggingface-cli login`). Set it only to store a token of your own.
    HUGGING_FACE_TOKEN: Union[str, type(None)] = None
    enable_grade: bool = True
    self_optimized_masking: bool = True
    use_logits: bool = True
    method: str = "FastDime"
    shift: float = 3.0
    grad_threshold: float = 0.1
    warmup_steps: int = 0
    use_gussian_blur_masking: bool = True
    adjust_parameters: bool = False
    parameters_rewight: dict = {"lr": 0.5, "l1": 10e02, "l2": 10e-2, "temperature": 1.5}
    paramters_smoothdiff: dict = {"n_samples": 20, "std": 0.5, "use_noise_level": False}
    parameters_manifold_smoothing: dict = {
        "n_samples": 20,
        "strength": 0.3,
        "num_inference_steps": None,
        "prompt": None,
        "guidance_scale": None,
    }
    parameters_integrated_grad: dict = {
        "n_steps": 50,
        "baseline": None,
        "batch_size": 9,
        "height": 128,
        "width": 128,
    }
    gradient_smoothing: str = "distilled"
    use_momentum: bool = False


class NameLayerMapComposite1(Composite):
    """A Composite for which hooks are specified by both a mapping from
    module names and module types to hooks.

    The layer-name mapping will be matched before the layer-type mapping.

    Parameters
    ----------
    name_map: `list[tuple[tuple[str, ...], Hook]]`
        A mapping as a list of tuples, with a tuple of applicable module names and a Hook.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]], optional
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    """

    def __init__(self, name_map=None, layer_map=None, canonizers=None):
        """Store both maps and register :meth:`mapping` as the module map."""
        self.name_map = name_map if name_map is not None else []
        self.layer_map = layer_map if layer_map is not None else []
        super().__init__(module_map=self.mapping, canonizers=canonizers)

    # pylint: disable=unused-argument
    def mapping(self, ctx, name, module):
        """Get the appropriate hook given mappings from module names and types to hooks.

        The name mapping is checked first, followed by the layer type mapping.

        Parameters
        ----------
        ctx: dict
            A context dictionary to keep track of previously registered hooks.
        name: str
            Name of the module.
        module: obj:`torch.nn.Module`
            Instance of the module to find a hook for.

        Returns
        -------
        obj:`Hook` or None
            The hook found with the module name or type in the given maps,
            or None if no applicable hook was found.
        """
        # Check name_map first (early exit if found)
        for names, hook in self.name_map:
            if name in names:
                return hook

        # Check layer_map second (early exit if found)
        for layer_types, hook in self.layer_map:
            if isinstance(module, layer_types):
                return hook

        # No match found
        return None


class EpsilonGradientOnly(BasicHook):
    """Modified Epsilon rule that returns only the gradient without input multiplication.

    This rule computes: R_input = R_output / (output + ε)
    Instead of the standard: R_input = input * (R_output / (output + ε))

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter. If ``epsilon`` is a float, it will be added to the denominator with the same sign as
        each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected, of
        which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    """

    def __init__(self, epsilon=1e-6, zero_params=None):
        """Build the BasicHook modifiers of the gradient-only epsilon rule.

        The reducer keeps the stabilized gradient only, i.e. it omits the
        usual multiplication with the layer input.
        """
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(zero_params=zero_params)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(
                lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])
            ),
            reducer=(
                lambda inputs, gradients: gradients[0]
            ),  # Only gradient, no input multiplication
        )


class LatentToClassification(nn.Module):
    """Chain the SD3 VAE decoder with an image classifier.

    Used to obtain a classifier that operates directly on scaled SD3 latents
    (``(z - shift) * scale``), for example as the model handed to attribution
    libraries.

    Parameters
    ----------
    vae_decoder : nn.Module
        Decoder of the SD3 VAE (``pipe.vae.decoder``).
    classifier : nn.Module
        Classifier that expects images in ``[0, 1]``.
    vae_config : object
        VAE config providing ``scaling_factor`` and ``shift_factor``.
    """

    def __init__(self, vae_decoder, classifier, vae_config):
        """Keep references to the VAE decoder, the classifier and the config."""
        super().__init__()
        self.vae_decoder = vae_decoder
        self.classifier = classifier
        self.vae_config = vae_config

    def forward(self, latent):
        """Unscale the latent, decode it, map to ``[0, 1]`` and classify.

        Parameters
        ----------
        latent : torch.Tensor
            Scaled SD3 latent of shape ``(B, 16, H / 8, W / 8)``.

        Returns
        -------
        torch.Tensor
            Classifier output for the decoded images.
        """
        latent = (
            latent / self.vae_config.scaling_factor
        ) + self.vae_config.shift_factor

        decoder_output = self.vae_decoder(latent)

        if hasattr(decoder_output, "sample"):
            decoded = decoder_output.sample
        elif isinstance(decoder_output, tuple):
            decoded = decoder_output[0]
        else:
            decoded = decoder_output

        # Normalize to [0, 1] range
        decoded = decoded / 2.0 + 0.5

        # Apply classifier
        classification = self.classifier(decoded)

        return classification


class DiffusionGenerator(EditCapableGenerator):  # InvertibleGenerator
    """Stable Diffusion 3 generator that edits images by guided denoising.

    The class holds a diffusers ``StableDiffusion3Pipeline`` (without the T5 text
    encoder) and implements the ``EditCapableGenerator`` interface on top of it.
    All work happens in the VAE latent space: ``latent_encoder`` /
    ``latent_decoder`` convert between ``[0, 1]`` images and scaled latents,
    ``prepare_time_steps`` builds the shifted rectified-flow sigma schedule and
    ``encode`` / ``decode`` implement the plain img2img forward and Euler reverse
    processes. ``edit`` is the entry point used by the counterfactual explainer
    and dispatches on ``config.method`` to ``FastDiME``, ``FastDiME_DDPM`` or
    ``ddpm_inversion``. The ``gradient_smoothing`` setting decides which
    attribution method (raw gradient, LRP, SmoothGrad, SmoothDiff, DeepLift,
    integrated gradients or manifold-aware smoothing) supplies the classifier
    gradient inside those loops.

    Parameters
    ----------
    config : DiffusionGeneratorConfig
        Generator config. It is passed through ``load_yaml_config`` but its
        ``HUGGING_FACE_TOKEN`` attribute is read from the object directly, so an
        already constructed config object is expected.
    model_dir : str, optional
        Directory of the generator; defaults to ``config.base_path``.
    device : str, optional
        Device the pipeline is moved to. The attribute ``self.device`` is set
        independently from CUDA availability.
    dtype : torch.dtype, optional
        Weight and activation dtype of the pipeline.
    classifier_dataset : Dataset, optional
        Dataset of the classifier; its ``task_config`` is copied onto the
        training dataset when ``config.task_config`` is unset.
    predictor_dataset : Dataset, optional
        Stored as ``self.predictor_dataset``; not used by the edit loops.
    train : bool, optional
        Accepted for interface compatibility; unused.

    Notes
    -----
    Construction registers the Hugging Face token via ``HfFolder.save_token``,
    loads the train/val datasets of ``config.data`` and downloads/loads the SD3
    weights, so it is slow and needs network access on the first call. Several
    methods write debug images below ``_DEBUG_DIR`` (``$PEAL_RUNS/debug``).
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
        """Load the datasets and the SD3 pipeline and copy the config fields.

        See the class docstring for the meaning of the parameters.
        """
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
        # from torch.cuda.amp import autocast
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.config.pretrained_model_name_path,
            torch_dtype=dtype,
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
        self.grad_threshold = self.config.grad_threshold
        self.predictor_dataset = predictor_dataset
        self.warmup_steps = self.config.warmup_steps
        self.use_gussian_blur_masking = self.config.use_gussian_blur_masking
        self.adjust_parameters = self.config.adjust_parameters
        self.parameters_rewight = self.config.parameters_rewight
        self.gradient_smoothing = self.config.gradient_smoothing
        self._lrp_composite_cache = {}
        self._sample_id = 0
        self.smoothdiff_parameters = self.config.paramters_smoothdiff
        self.manifold_smoothing_parameters = self.config.parameters_manifold_smoothing
        self.parameters_integrated_grad = self.config.parameters_integrated_grad
        self.base_path = None
        self.baseline = None
        self.use_momentum = self.config.use_momentum
        _log.info("%s", "initializing generator done!")

    def rescale_time_step(self, timestep):
        """Map a sigma in ``[0, 1]`` to the transformer timestep range ``[0, 1000]``.

        Parameters
        ----------
        timestep : float or torch.Tensor
            Flow-matching sigma.

        Returns
        -------
        float or torch.Tensor
            ``timestep * 1000``.
        """
        x_min = 0.0
        x_max = 1.0
        y_min = 0.0
        y_max = 1000.0
        scaled = (timestep - x_min) * (y_max - y_min) / (x_max - x_min) + y_min
        return scaled

    def prepare_time_steps(self, num_inference_steps=10, end_time=1000, shift=3):
        """Build the shifted rectified-flow sigma schedule used by SD3.

        The 1000 training timesteps up to ``end_time`` are turned into sigmas, the
        SD3 time shift ``shift * s / (1 + (shift - 1) * s)`` is applied, and the
        range between the resulting maximum and minimum sigma is resampled with
        ``num_inference_steps`` points (shifted again) and terminated with a zero.

        Parameters
        ----------
        num_inference_steps : int, optional
            Number of denoising steps.
        end_time : float, optional
            Largest training timestep to start from (``t * 1000`` for a partial
            img2img edit at strength ``t``).
        shift : float, optional
            SD3 timestep shift.

        Returns
        -------
        torch.Tensor
            Descending sigmas of shape ``(num_inference_steps + 1,)`` whose last
            entry is ``0``.
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
        """Encode ``[0, 1]`` images into scaled SD3 VAE latents.

        Parameters
        ----------
        img : torch.Tensor
            Images of shape ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Sampled posterior latents of shape ``(B, 16, H / 8, W / 8)`` after
            ``(z - shift_factor) * scaling_factor``.
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
        """Decode scaled SD3 latents back to ``[0, 1]`` images.

        Parameters
        ----------
        latent : torch.Tensor
            Scaled latents of shape ``(B, 16, h, w)``.
        img : bool, optional
            If ``True`` return the first decoded image as a ``PIL.Image`` instead
            of the tensor.

        Returns
        -------
        torch.Tensor or PIL.Image.Image
            Decoded images of shape ``(B, 3, 8h, 8w)`` in ``[0, 1]`` (not clamped),
            or a PIL image of the first element.

        Notes
        -----
        Enables gradient checkpointing on the VAE and empties the CUDA cache on
        the tensor path.
        """
        # if latent.dim() < 4:
        #     latent = latent.unsqueeze(0)
        self.pipe.vae.enable_gradient_checkpointing()
        latent = (
            latent / self.pipe.vae.config.scaling_factor
        ) + self.pipe.vae.config.shift_factor
        decoded = self.pipe.vae.decode(latent)
        decoded = decoded.sample / 2.0 + 0.5
        # print(
        #     f"memory summary after decoding, empty_cache and delete unused tensors: \n",
        #     torch.cuda.memory_summary(device=None, abbreviated=False),
        # )
        if img:
            to_pil = ToPILImage()
            img = to_pil(decoded[0].squeeze(0).clamp(0, 1))
            return img
        torch.cuda.empty_cache()
        return decoded

    def adjust_time_steps_img_2_img(
        self, num_inference_steps, strength, timesteps, encoding: bool
    ):
        """Truncate a sigma schedule for an img2img edit of the given strength.

        Parameters
        ----------
        num_inference_steps : int
            Length of the full schedule.
        strength : float
            Fraction of the schedule to keep (``1.0`` keeps everything).
        timesteps : torch.Tensor
            Full schedule as returned by ``prepare_time_steps``.
        encoding : bool
            If ``True`` also drop the trailing zero sigma.

        Returns
        -------
        tuple[torch.Tensor, int]
            The truncated schedule and the number of remaining steps.
        """

        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        time_index = int(max(num_inference_steps - init_timestep, 0))
        if encoding:
            return timesteps[time_index:-1], num_inference_steps - time_index
        return timesteps[time_index:], num_inference_steps - time_index

    def adjust_timestep_img2img(
        self, num_inference_steps, strength, timesteps, encoding: bool
    ):
        """Alias of ``adjust_time_steps_img_2_img``."""
        return self.adjust_time_steps_img_2_img(
            num_inference_steps, strength, timesteps, encoding
        )

    @torch.no_grad()
    def forward_process(self, image, num_inference_steps):
        """Noise the latent of an image to every sigma of the schedule.

        Parameters
        ----------
        image : torch.Tensor
            Image of shape ``(1, 3, H, W)`` in ``[0, 1]``.
        num_inference_steps : int
            Number of sigmas (schedule built with ``self.shift``).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(num_inference_steps + 1, 1, 16, h, w)`` holding
            ``(1 - sigma_i) * x_0 + sigma_i * eps_i`` for the descending sigmas
            followed by the clean latent ``x_0`` as last entry. Independent noise
            is drawn for every step.
        """
        # if isinstance(image, torch.Tensor):
        #     x_0 = image
        #     if x_0.dim() < 4:
        #         x_0 = x_0.unsqueeze(0)
        # else:
        x_0 = self.latent_encoder(image).to(self.dtype)
        x_ts = x_0.expand(num_inference_steps, -1, -1, -1).to(x_0.device)
        timesteps = (
            self.prepare_time_steps(num_inference_steps, shift=self.shift)[:-1]
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
        """Edit-friendly DDPM inversion of an image in the SD3 flow model.

        Runs the forward process, then for every step predicts the velocity from
        the noised latent ``x_{sigma_i}`` and stores the residual
        ``z_i = x_{sigma_{i+1}} - (x_0_pred + sigma_{i+1} * v)`` needed to reach the
        next (independently noised) latent. Feeding ``zs`` into ``decode_image``
        reproduces the input image, so the ``zs`` act as a per-step noise map that
        can be kept fixed while the starting latent is optimised.

        Parameters
        ----------
        image : torch.Tensor
            Image of shape ``(1, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(x_T, zs)``: the latent at the largest sigma with shape
            ``(1, 16, h, w)`` and the noise maps of shape
            ``(steps_number, 1, 16, h, w)`` whose last entry is zero.
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
        """Run the reverse process from ``xts`` while injecting the noise maps ``zts``.

        Parameters
        ----------
        xts : torch.Tensor
            Starting latent of shape ``(1, 16, h, w)`` (typically the ``x_T`` from
            ``encode_image``).
        zts : torch.Tensor
            Per-step noise maps of shape ``(steps_number, 1, 16, h, w)``.
        eta : float, optional
            Accepted for API symmetry; not used.

        Returns
        -------
        torch.Tensor
            Decoded image of shape ``(1, 3, H, W)`` in ``[0, 1]``.
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
        """Encode an image and noise its latent to the sigma reached at time ``t``.

        This is the forward half of an SD3 img2img edit: the VAE latent is mixed
        with Gaussian noise as ``sigma * eps + (1 - sigma) * z`` where ``sigma`` is
        the largest value of the schedule ending at ``t * 1000``. A DDIM-style
        deterministic inversion exists in the source only as a commented block.

        Parameters
        ----------
        x : torch.Tensor
            Images of shape ``(B, 3, H, W)`` in ``[0, 1]``.
        t : float, optional
            Edit strength in ``[0, 1]``; ``1.0`` noises to pure noise.
        stochastic : object, optional
            Unused.
        num_steps : int, optional
            Unused; the schedule length is ``self.steps_number``.

        Returns
        -------
        torch.Tensor
            Noised latent of shape ``(B, 16, h, w)`` with ``requires_grad`` set.
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
        guidance_scale,
        z: torch.tensor,
        t: float = 1.0,
        stochastic=None,
        num_steps=None,
    ) -> torch.tensor:
        """Denoise a latent with an explicit Euler rectified-flow integration.

        Parameters
        ----------
        guidance_scale : float
            Classifier-free guidance scale; ``0`` or ``None`` disables it.
        z : torch.Tensor
            Latents of shape ``(B, 16, h, w)``. Only ``z[0]`` is decoded.
        t : float, optional
            Edit strength defining the start sigma (schedule ends at ``t * 1000``).
        stochastic : object, optional
            Unused.
        num_steps : int
            Number of Euler steps taken along the schedule.

        Returns
        -------
        torch.Tensor
            Decoded image of shape ``(1, 3, H, W)`` in ``[0, 1]``.
        """
        torch.cuda.empty_cache()
        sigmas = self.prepare_time_steps(
            self.steps_number, end_time=t * 1000, shift=self.shift
        )
        # sigmas, num_inference_steps = self.adjust_time_steps_img_2_img(
        #     self.steps_number, self.strength, sigmas, encoding=False
        # )
        num_inference_steps = num_steps
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
        if guidance_scale:
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
            if guidance_scale > 0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            dt = sigmas[i + 1] - sigmas[i]

            z_t = z_t + (dt) * noise_pred

            torch.cuda.empty_cache()

        return self.latent_decoder(z_t)

    def rbg_gray(self, rbg):
        """Convert an RGB tensor to single-channel luminance.

        Parameters
        ----------
        rbg : torch.Tensor
            ``(3, H, W)`` or ``(B, 3, H, W)`` tensor.

        Returns
        -------
        torch.Tensor
            Luminance with the channel dimension kept (``1`` channel).
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
        """Inpaint the regions where ``pe`` differs from ``x`` with SD3 denoising.

        The denoising starts from the noised latent of ``pe`` and, after every
        Euler step, resets the latent outside the mask to the (re-noised) latent of
        ``x`` so that only the masked region is regenerated. The mask is the dilated
        difference between ``x`` and ``pe`` thresholded at ``inpaint``; it can be
        shrunk by a previous mask (``old_mask`` weighted by ``mask_momentum``) and
        extended by ``boolmask_in``. Twice ``self.steps_number`` steps are used.

        Parameters
        ----------
        x : torch.Tensor
            Original image in the dataset's value range, shape ``(1, 3, H, W)``.
        pe : torch.Tensor
            Edited image (same shape) providing the inpainting target.
        inpaint : float
            Threshold on the dilated difference mask.
        dilation : int
            Dilation kernel size passed to ``generate_smooth_mask``.
        t : float
            Edit strength; defines the start sigma.
        stochastic : object
            Unused.
        old_mask : torch.Tensor or None
            Mask of a previous attempt that is subtracted from the new mask.
        mask_momentum : float
            Weight of ``old_mask`` in that subtraction.
        boolmask_in : torch.Tensor or None
            Additional binary mask that is united with the computed mask.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The decoded result of shape ``(1, 3, H, W)`` and the binary mask on
            CPU.

        Notes
        -----
        Writes debug images to the relative path ``utils/delete_me/``.
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
        """Sample images from the SD3 pipeline with the configured prompt.

        Parameters
        ----------
        batch_size : int, optional
            Number of images.

        Returns
        -------
        torch.Tensor
            Images of shape ``(batch_size, 3, H, W)`` in ``[0, 1]``.
        """
        images = self.pipe(batch_size * [self.prompt]).images
        images_torch = torch.stack([ToTensor()(image) for image in images])
        return images_torch

    def sample_z(self, n="auto"):
        """Draw Gaussian latents according to ``calc_z_shapes``.

        Kept from the ``InvertibleGenerator`` interface. ``calc_z_shapes`` returns
        ``None`` for this generator, so calling this method raises ``TypeError``;
        it also reads ``config.batch`` and ``config.temp`` which
        ``DiffusionGeneratorConfig`` does not define.

        Parameters
        ----------
        n : int or str, optional
            Number of samples, or ``"auto"`` to use ``config.batch``.

        Returns
        -------
        list[torch.Tensor]
            One tensor per latent shape.
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
        s=0.3,
        use_logits=True,
        predictor_img_size=None,
        lr=None,
        momentum=None,
        optimizer=None,
        threshold: int = None,
        **kwargs,
    ):
        """Classifier-guidance gradient of the target-class score w.r.t. a latent.

        The latent is decoded with the VAE (unless ``resize`` is ``True``, in which
        case ``x_t`` is already an image), mapped into the classifier's input
        domain with the ``generator_to_classifier`` callable from ``kwargs``,
        resized to ``predictor_img_size`` and classified. The loss is
        ``-s * logit[y]`` (or ``-s * log_softmax[y]`` when ``use_logits`` is
        ``False``) summed over the batch.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, 16, h, w)`` or image ``(B, 3, H, W)``.
        y : torch.Tensor
            Target class indices of shape ``(B,)``.
        resize : bool
            If ``True`` skip the VAE decoding.
        classifier : nn.Module
            Classifier whose score is differentiated (set to eval mode).
        s : float, optional
            Temperature/scale of the loss.
        use_logits : bool, optional
            Differentiate raw logits instead of log-probabilities.
        predictor_img_size : tuple[int, int], optional
            Spatial size expected by the classifier.
        lr, momentum, optimizer : object, optional
            Accepted because ``class_grad_kwargs`` is splatted in; unused here.
        threshold : float, optional
            If truthy, per-sample entries below ``threshold * max`` or above
            ``(1 - threshold) * max`` of the latent gradient are set to zero
            (signed comparison, ``max`` over the absolute gradient).
        **kwargs
            ``generator_to_classifier`` callable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Gradient w.r.t. ``x_t`` and gradient w.r.t. the classifier input image.
        """
        classifier.eval()

        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        x_in = x_in
        generator_to_classifier = kwargs.get("generator_to_classifier", None)
        if resize:
            x_img = x_in
        else:
            x_img = self.latent_decoder(x_in)
            x_img.retain_grad()
        x_img = generator_to_classifier(x_img) if generator_to_classifier else x_img
        x_img = transforms.Resize(predictor_img_size)(x_img)

        classifier.eval()
        selected = classifier(x_img)
        if not use_logits:
            selected = F.log_softmax(selected, dim=1)
        selected = -selected[range(len(y)), y]
        selected = selected * s
        # selected.sum().backward()
        grads, pixel_grad = torch.autograd.grad(
            selected.sum(), [x_in, x_img], retain_graph=False
        )
        # grads = x_in.grad

        # grads = torch.autograd.grad(selected.sum(), x_in)[0]
        # pixel_grad = x_img.grad
        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            # max_values = grads.abs().max()
            grads[grads < threshold * max_values] = 0.0
            grads[grads > (1 - threshold) * max_values] = 0.0
        return grads, pixel_grad

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
        """
        Computes the distance loss between x_t, z_t and x_tau
        :x_tau: initial image
        :z_t: current noisy instance
        :x_t: current clean instance
        :alpha_t: time dependant constant
        :scale_grads: scale grads based on time dependant constant
        """

        z_in = nn.Parameter(z_t.detach().requires_grad_(True))
        x_in = nn.Parameter(x_t.detach().requires_grad_(True))
        try:
            m1 = (
                l1_loss * torch.norm(z_in - x_tau.to(z_in.device), p=1, dim=1).sum()
                if l1_loss != 0
                else 0
            )
            m2 = (
                l2_loss * torch.norm(z_in - x_tau.to(z_in.device), p=2, dim=1).sum()
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

    def create_latent_classifier(self, classifier):
        """Wrap ``classifier`` into a ``LatentToClassification`` on the SD3 decoder.

        Parameters
        ----------
        classifier : nn.Module
            Image classifier.

        Returns
        -------
        LatentToClassification
            Module mapping scaled latents to class scores.
        """
        Latent_classifeir = LatentToClassification(
            self.pipe.vae.decoder, classifier, self.pipe.vae.config
        )

        return Latent_classifeir

    def visualize_lrp_relevance(
        self,
        original_img,
        relevance,
        save_dir,
        index=0,
        cmap="seismic",
    ):
        """
        Visualize LRP relevance and save alongside original image

        Args:
            original_img: Original input image tensor (C, H, W) or (B, C, H, W)
            relevance: LRP relevance tensor in latent space (C, H, W) or (B, C, H, W)
            save_dir: Directory path to save visualizations
            index: Index for batch processing
            cmap: Colormap for relevance ('seismic', 'hot', 'viridis')
        """
        from PIL import Image

        os.makedirs(save_dir, exist_ok=True)
        to_pil = ToPILImage()

        # Handle batch dimension
        if original_img.dim() == 4:
            original_img = original_img[0]
        if relevance.dim() == 4:
            relevance = relevance[0]

        # Move to CPU and detach
        original_img = original_img.detach().cpu()
        relevance = relevance.detach().cpu()

        # Decode latent relevance to pixel space
        with torch.no_grad():
            if relevance.shape[-1] < 64:  # Likely in latent space
                relevance_decoded = (
                    self.latent_decoder(relevance.unsqueeze(0).to(self.device))
                    .squeeze(0)
                    .cpu()
                )
            else:
                relevance_decoded = relevance

        # Save original image
        original_pil = to_pil(original_img.clamp(0, 1))
        original_pil.save(os.path.join(save_dir, f"original_{index}.png"))

        # Create heatmap using existing high_contrast_heatmap function
        ref = torch.zeros_like(relevance_decoded[0:1])
        heatmap = high_contrast_heatmap(ref, -relevance_decoded)[0]
        heatmap = torch.stack(heatmap)
        heatmap_pil = to_pil(heatmap)
        heatmap_pil.save(os.path.join(save_dir, f"lrp_heatmap_{index}.png"))

        # Create side-by-side comparison
        width, height = original_pil.size
        if original_pil.size != heatmap_pil.size:
            heatmap_pil = heatmap_pil.resize(original_pil.size, Image.LANCZOS)

        comparison = Image.new("RGB", (width * 2, height))
        comparison.paste(original_pil, (0, 0))
        comparison.paste(heatmap_pil, (width, 0))
        comparison.save(os.path.join(save_dir, f"lrp_comparison_{index}.png"))

        _log.info("%s", f"LRP visualizations saved to {save_dir}")

        return comparison

    def visualize_lrp(self, image_tensor, relevance_tensor, save_dir, index=0):
        """
        Plots the original image and the LRP heatmap side-by-side and saves them.

        Args:
            image_tensor: Original input image tensor (B, C, H, W) or (C, H, W)
            relevance_tensor: LRP relevance tensor (B, C, H, W) or (C, H, W)
            save_dir: Directory path to save visualizations
            index: Index for batch processing
        """
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(save_dir, exist_ok=True)

        # Handle batch dimension
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        if relevance_tensor.dim() == 4:
            relevance_tensor = relevance_tensor[0]

        # 1. Prepare the Original Image
        # Convert from PyTorch (Channel, Height, Width) -> Numpy (H, W, C)
        img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)

        # Normalize image for display (mapping values to 0-1 range)
        img = (img - img.min()) / (img.max() - img.min())

        # 2. Prepare the Heatmap
        # Sum relevance across the 3 color channels (R, G, B)
        heatmap = relevance_tensor.sum(0)  # Changed from sum(1) to sum(0) for (C,H,W)
        heatmap = heatmap.detach().cpu().numpy()

        # 3. Set up the plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Original Image
        axs[0].imshow(img)
        axs[0].set_title("Original Input", fontsize=14)
        axs[0].axis("off")

        # Plot LRP Heatmap
        # We calculate the max absolute value to ensure 0 is centered at white
        vmax = np.max(np.abs(heatmap))

        im = axs[1].imshow(heatmap, cmap="seismic", vmin=-vmax, vmax=vmax)
        axs[1].set_title("LRP Relevance Map", fontsize=14)
        axs[1].axis("off")

        # Add a colorbar to interpret the values
        plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(save_dir, f"lrp_visualization_{index}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # Close to free memory

        _log.info("%s", f"LRP visualization saved to {save_path}")

        return save_path

    def visualize_batch_lrp_simple(self, image_tensors, relevance_tensors, save_dir):
        """
        Visualize LRP for a batch of images using the simple matplotlib approach

        Args:
            image_tensors: Batch of original images (B, C, H, W)
            relevance_tensors: Batch of relevances (B, C, H, W)
            save_dir: Directory to save visualizations
        """
        batch_size = image_tensors.shape[0]

        for i in range(batch_size):
            self.visualize_lrp(
                image_tensor=image_tensors[i],
                relevance_tensor=relevance_tensors[i],
                save_dir=save_dir,
                index=i,
            )

    def get_dynamic_gamma_composite(
        self,
        model,
        start_gamma=0.5,
        end_gamma=0.1,
        stabilizer=1e-6,
        canoniezer=None,
    ):
        """
        Creates a Zennit Composite that:
        1. Uses Epsilon-rule for the first and last layers.
        2. Uses Gamma-rule for intermediate layers.
        3. Linearly decays Gamma from start_gamma to end_gamma for intermediate layers.

        Args:
            model: PyTorch model to analyze
            start_gamma: Gamma value for first intermediate layer (default: 0.5)
            end_gamma: Gamma value for last intermediate layer (default: 0.1)
            stabilizer: Small constant for numerical stability (default: 1e-6)

        Returns:
            NameLayerMapComposite: Zennit composite with custom rules
        """
        # Collect relevant layers
        model_type = type(model).__name__
        cache_key = f"{model_type}_custom"

        # Return cached composite if exists
        if cache_key in self._lrp_composite_cache:
            return self._lrp_composite_cache[cache_key]

        relevant_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                relevant_layers.append(name)

        total_layers = len(relevant_layers)

        if total_layers == 0:
            raise ValueError("No Conv2d or Linear layers found in the model")

        # Get base layer map for other layer types
        layer_map = []
        layer_map = layer_map_base(stabilizer=stabilizer) + [(Convolution, Gamma(0.25))]

        # Build name map for Conv2d and Linear layers
        name_map = []
        for i, layer_name in enumerate(relevant_layers):
            if i == 0:
                # First and last layers: use Epsilon rule
                name_map.append((layer_name, Epsilon(epsilon=0)))
            elif i == total_layers - 1:
                name_map.append((layer_name, EpsilonGradientOnly(epsilon=0)))
            else:
                # Intermediate layers: use Gamma with linear decay
                # progress = (i - 1) / max(total_layers - 2, 1)  # Avoid division by zero
                # current_gamma = end_gamma + (progress * (start_gamma - end_gamma))
                # name_map.append((layer_name, Gamma(gamma=current_gamma)))
                continue
        self._lrp_composite_cache[cache_key] = NameLayerMapComposite(
            layer_map=layer_map, name_map=name_map, canonizers=[canoniezer]
        )
        _log.info("%s", f"Created composite with {total_layers} layers:")
        _log.info("%s", f"  First/Last: Epsilon(epsilon={0})")
        if total_layers > 2:
            _log.info(
                "%s", f"  Intermediate: Gamma({start_gamma:.3f} → {end_gamma:.3f})"
            )

        return NameLayerMapComposite(
            layer_map=layer_map, name_map=name_map, canonizers=[canoniezer]
        )

    def LRP(
        self,
        x_t,
        y,
        source_class,
        classifier,
        resize,
        predictor_img_size,
        threshold,
        **kwargs,
    ):
        """Layer-wise relevance propagation of the source-class score into the latent.

        The decoded image is attributed with zennit (``Gradient`` attributor,
        ``ResNetCanonizer`` and the composite from ``get_dynamic_gamma_composite``)
        for a one-hot output of ``source_class``. The resulting pixel relevance is
        then propagated through the VAE decoder with ``torch.autograd.grad`` to
        obtain a latent-space "gradient" that the FastDiME loop can apply.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, 16, h, w)`` or image when ``resize`` is ``True``.
        y : torch.Tensor
            Target classes; only its length is used here.
        source_class : int or list[int]
            Class whose relevance is propagated.
        classifier : nn.Module
            ResNet-like classifier.
        resize : bool
            If ``True`` skip the VAE decoding.
        predictor_img_size : tuple[int, int]
            Input size of the classifier.
        threshold : float or None
            Same sparsification as in ``clean_multiclass_cond_fn``.
        **kwargs
            ``generator_to_classifier`` callable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Latent gradient and pixel-space relevance.

        Notes
        -----
        The one-hot target is moved to ``"cuda"`` unconditionally.
        """

        # latent_classifier = self.create_latent_classifier(classifier=classifier)
        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        generator_to_classifier = kwargs.get("generator_to_classifier", None)
        if resize:
            x_img = x_in
        else:
            x_img = self.latent_decoder(x_in)
        x_img = generator_to_classifier(x_img) if generator_to_classifier else x_img
        x_img = transforms.Resize(predictor_img_size)(x_img)
        transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        low, high = transform_norm(torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]]))
        classifier.eval()
        canonizer = ResNetCanonizer()
        composite = self.get_dynamic_gamma_composite(
            model=classifier,
            canoniezer=canonizer,
            start_gamma=0.5,
            end_gamma=0.1,
            stabilizer=10e-6,
        )
        # composite = EpsilonGammaBox(
        #     low=low, high=high, canonizers=[canonizer], epsilon=0.0, gamma=0.25
        # )
        low, high = transform_norm(torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]]))
        # composite = SpecialFirstLayerMapComposite(
        #     layer_map=[
        #         (Activation, Pass()),  # ignore activations
        #         (AvgPool, Norm()),  # normalize relevance for any AvgPool
        #         (Convolution, Gamma(0.25)),  # any convolutional layer
        #         (Linear, Epsilon(epsilon=0)),  # this is the dense Linear, not any
        #         (BatchNorm, Pass()),  # ignore BatchNorm
        #     ],
        #     first_map=[(Convolution, EpsilonGradientOnly(epsilon=0))],
        #     canonizers=[canonizer],
        # )
        # composite = EpsilonPlusFlat(canonizers=[canonizer])

        with composite.context(classifier) as modified_model:

            attributor = Gradient(model=modified_model, composite=composite)
            output = modified_model(x_img)
            num_classes = output.shape[1]
            prediction_score = torch.eye(num_classes, device=output.device)[
                [source_class]
            ].to("cuda")

            output_relevance_pixel_space, input_relevance_pixel_space = attributor(
                x_img, prediction_score
            )
            # x_img.backward(
            #     gradient=-input_relevance_pixel_space,
            #     retain_graph=True,
            # )
            # self.visualize_batch_lrp_simple(
            #     x_img,
            #     input_relevance_pixel_space,
            #     "<PEAL_BASE>/explanations/lrp",
            # )
            output = -output[range(len(y)), y]
            grads = torch.autograd.grad(
                x_img, x_in, grad_outputs=input_relevance_pixel_space
            )[0]
            if threshold:
                max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
                # max_values = grads.abs().max()
                grads[grads < threshold * max_values] = 0.0
                grads[grads > (1 - threshold) * max_values] = 0.0
            latent_relevance = x_in.grad
        return grads, input_relevance_pixel_space

    @torch.no_grad()
    def generate_baseline(self, height: int, width: int, batch_size: int):
        """Create (or load) the mean SD3 sample used as attribution baseline.

        ``batch_size`` images are sampled from the pipeline with the configured
        prompt, resized to 256x256 if larger, averaged into one image and cached
        at ``<base_path>/baselines/baseline.pt`` next to a collage PNG.

        Parameters
        ----------
        height, width : int
            Sampling resolution.
        batch_size : int
            Number of samples to average.

        Returns
        -------
        torch.Tensor
            Baseline image of shape ``(1, 3, H, W)``.
        """
        baseline_dir = os.path.join(self.base_path, "baselines")
        baseline_path = os.path.join(baseline_dir, "baseline.pt")
        if os.path.exists(baseline_path):
            _log.info("%s", f"Loading existing baseline from {baseline_path}")
            baseline = torch.load(baseline_path, map_location=self.device)
            return baseline

        images = self.pipe(
            prompt=self.prompt,
            height=height,
            width=width,
            num_images_per_prompt=batch_size,
            output_type="pt",
        ).images
        if height > 256 or width > 256:
            images = transforms.Resize((256, 256))(images)

        os.makedirs(baseline_dir, exist_ok=True)
        filename = os.path.join(baseline_dir, "baseline_collages.png")
        baseline = images.mean(dim=0, keepdim=True)
        torch.save(baseline, baseline_path)
        collages = torch.cat([images, baseline], dim=0)
        nrows = int(math.ceil(math.sqrt(batch_size + 1)))
        torchvision.utils.save_image(collages, fp=filename, nrow=nrows)
        return baseline

    def _create_deeplift_compatible_resnet(self, original_model):
        """
        Create a new ResNet model with unique ReLU instances for each usage,
        then load weights from the original model.

        This avoids the DeepLift issue of shared ReLU modules.
        """

        # Detect model type based on structure
        model_name = type(original_model).__name__
        num_classes = original_model.model.fc.out_features

        # Determine which ResNet variant we're dealing with
        if hasattr(original_model.model, "layer4"):
            # Count blocks to determine variant
            num_blocks = [
                len(original_model.model.layer1),
                len(original_model.model.layer2),
                len(original_model.model.layer3),
                len(original_model.model.layer4),
            ]

            # Check block type
            first_block = original_model.model.layer1[0]
            is_bottleneck = hasattr(first_block, "conv3")

            if is_bottleneck:
                if num_blocks == [3, 4, 6, 3]:
                    base_model = "resnet50"
                elif num_blocks == [3, 4, 23, 3]:
                    base_model = "resnet101"
                elif num_blocks == [3, 8, 36, 3]:
                    base_model = "resnet152"
                else:
                    base_model = "resnet50"  # Default
            else:
                if num_blocks == [2, 2, 2, 2]:
                    base_model = "resnet18"
                elif num_blocks == [3, 4, 6, 3]:
                    base_model = "resnet34"
                else:
                    base_model = "resnet18"  # Default
        else:
            base_model = "resnet18"  # Default fallback

        _log.info(
            "%s", f"Detected model type: {base_model}, num_classes: {num_classes}"
        )

        # Create the DeepLift-compatible model
        if base_model == "resnet18":
            new_model = DeepLiftResNet18(num_classes=num_classes)
        elif base_model == "resnet34":
            new_model = DeepLiftResNet34(num_classes=num_classes)
        elif base_model == "resnet50":
            new_model = DeepLiftResNet50(num_classes=num_classes)
        elif base_model == "resnet101":
            new_model = DeepLiftResNet101(num_classes=num_classes)
        elif base_model == "resnet152":
            new_model = DeepLiftResNet152(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {base_model}")

        # Load weights from original model
        new_model = self._load_weights_to_deeplift_model(original_model, new_model)

        return new_model.eval().to(self.device)

    def _load_weights_to_deeplift_model(self, source_model, target_model):
        """
        Load weights from source model to target DeepLift-compatible model.
        Handles the mapping between 'relu' -> 'relu1', 'relu2', etc.
        """
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()

        # Create mapping for weights
        new_state = {}

        for key in target_state.keys():
            if key in source_state:
                # Direct match
                new_state[key] = source_state[key]
            else:
                # Handle renamed layers (relu1, relu2, relu3 don't have weights anyway)
                # ReLU modules don't have learnable parameters, so we can skip them
                _log.info(
                    "%s",
                    f"Warning: Key {key} not found in source model (likely a ReLU parameter)",
                )

        # Load the matched weights
        target_model.load_state_dict(new_state, strict=False)

        # Verify the loading
        _log.info(
            "%s",
            f"Loaded {len(new_state)} weight tensors into DeepLift-compatible model",
        )

        return target_model

    def deeplift(
        self,
        x_t,
        y,
        source_class,
        classifier,
        resize,
        baseline,
        predictor_img_size,
        height,
        width,
        batch_size,
        threshold,
        **kwargs,
    ):
        """DeepLift attribution (captum) propagated into the latent.

        A DeepLift-compatible copy of the ResNet classifier with unshared ReLUs is
        built, the decoded image is attributed against ``baseline`` (generated with
        ``generate_baseline`` when missing) and the attribution is pulled back
        through the VAE decoder.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, 16, h, w)`` or image when ``resize`` is ``True``.
        y : torch.Tensor
            Target classes.
        source_class : object
            Unused.
        classifier : nn.Module
            ResNet-like PEAL classifier wrapping ``.model``.
        resize : bool
            If ``True`` skip the VAE decoding.
        baseline : torch.Tensor or None
            Baseline image; ``None`` uses/creates ``self.baseline``.
        predictor_img_size : tuple[int, int]
            Input size of the classifier.
        height, width, batch_size : int
            Passed to ``generate_baseline``.
        threshold : float or None
            Zero latent-gradient entries with magnitude below ``threshold * max``.
        **kwargs
            ``generator_to_classifier`` callable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Latent gradient and pixel-space attribution.
        """
        model = self._create_deeplift_compatible_resnet(classifier)

        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        generator_to_classifier = kwargs.get("generator_to_classifier", None)
        if resize:
            x_img = x_in
        else:
            x_img = self.latent_decoder(x_in)
        x_img = generator_to_classifier(x_img) if generator_to_classifier else x_img
        x_img = transforms.Resize(predictor_img_size)(x_img)

        if baseline is None and self.baseline is None:
            baseline = self.generate_baseline(height, width, batch_size)
            self.baseline = baseline
        elif baseline is None:
            baseline = self.baseline
        if baseline.shape[1:] != x_img.shape[1:]:
            baseline = transforms.Resize((x_img.shape[2:]))(baseline)
            baseline = baseline.to(x_img.device)

        # Expand baseline to match batch size
        if baseline.shape[0] != x_img.shape[0]:
            baseline = baseline.expand(x_img.shape[0], -1, -1, -1).clone()

        deep_lift = DeepLift(model)
        attribution = deep_lift.attribute(
            inputs=x_img, baselines=baseline, target=y.to(x_img.device)
        )

        grads = torch.autograd.grad(x_img, x_in, grad_outputs=attribution)[0]
        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            grads[grads.abs() < threshold * max_values] = 0.0

        return grads, attribution

    def IG(
        self,
        inputs: torch.Tensor,
        baseline: torch.Tensor,
        n_steps: int,
        y: list,
        s: float,
        height: int,
        width: int,
        batch_size: int,
        model: nn.Module,
        Noise_tunnel=None,
        **kwargs,
    ):
        """Integrated gradients of ``-s * logit[y]`` along a straight path.

        ``n_steps`` interpolations between ``baseline`` and ``inputs`` are
        evaluated; the average of their (negated) input gradients is multiplied
        with ``inputs - baseline``. With ``Noise_tunnel`` the gradient at each path
        point is replaced by a captum ``NoiseTunnel`` attribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Classifier inputs ``(B, 3, H, W)``.
        baseline : torch.Tensor or None
            ``None`` uses/creates ``self.baseline``; a given tensor is replaced by
            zeros (this is what the code does).
        n_steps : int
            Number of path points.
        y : torch.Tensor
            Target classes.
        s : float
            Loss temperature.
        height, width, batch_size : int
            Passed to ``generate_baseline``.
        model : nn.Module
            Classifier (set to eval mode).
        Noise_tunnel : captum.attr.NoiseTunnel, optional
            Smoothing wrapper; ``nt_samples``, ``stdevs`` and ``target`` are read
            from ``kwargs``.

        Returns
        -------
        torch.Tensor
            Attribution of the same shape as ``inputs``.
        """
        model.eval()
        if baseline is None and self.baseline is None:
            baseline = self.generate_baseline(height, width, batch_size)
            self.baseline = baseline
        elif baseline is None and self.baseline is not None:
            baseline = self.baseline
        else:
            baseline = torch.zeros_like(inputs)
        if baseline.shape[1:] != inputs.shape[1:]:
            baseline = transforms.Resize((inputs.shape[2:]))(baseline)
        scaled_inputs = [
            baseline + (float(i) / n_steps) * (inputs - baseline)
            for i in range(1, n_steps + 1)
        ]
        grads = []
        for i in scaled_inputs:
            x = i.clone().detach().requires_grad_(True)
            if Noise_tunnel is None:
                model.zero_grad()
                output = model(x)
                output = -output[range(len(y)), y]
                output = output * s
                output.sum().backward()
                grad = x.grad
                grads.append(-grad)
            else:
                nt_samples = kwargs.get("nt_samples", 10)
                stdevs = kwargs.get("stdevs", 0.1)
                target = kwargs.get("target", y)
                attribution = Noise_tunnel.attribute(
                    inputs=x,
                    nt_samples=nt_samples,
                    stdevs=stdevs,
                    target=target.to(x.device),
                    abs=False,
                )
                grads.append(attribution)

        avg_grads = torch.mean(torch.stack(grads), dim=0)

        ig = avg_grads * (inputs - baseline)
        return ig

    def smooth_grad_captum(
        self,
        x_t,
        y,
        source_class,
        classifier,
        resize,
        n_samples,
        std,
        use_noise_level,
        predictor_img_size,
        threshold,
        **kwargs,
    ):
        """SmoothGrad (captum ``Saliency`` + ``NoiseTunnel``) pulled into the latent.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, 16, h, w)`` or image when ``resize`` is ``True``.
        y : torch.Tensor
            Target classes.
        source_class : object
            Unused.
        classifier : nn.Module
            Classifier (set to eval mode).
        resize : bool
            If ``True`` skip the VAE decoding.
        n_samples : int
            Number of noisy samples.
        std : float
            Noise standard deviation, or fraction of the input range when
            ``use_noise_level`` is ``True``.
        use_noise_level : bool
            Scale ``std`` by ``max - min`` of the image.
        predictor_img_size : tuple[int, int]
            Input size of the classifier.
        threshold : float or None
            Zero latent-gradient entries with magnitude below ``threshold * max``.
        **kwargs
            ``generator_to_classifier`` callable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Latent gradient (of the negated attribution) and pixel attribution.
        """

        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        generator_to_classifier = kwargs.get("generator_to_classifier", None)

        if resize:
            x_img = x_in
        else:
            x_img = self.latent_decoder(x_in)

        x_img = generator_to_classifier(x_img) if generator_to_classifier else x_img
        x_img = transforms.Resize(predictor_img_size)(x_img)

        classifier.eval()

        saliency = Saliency(classifier)

        noise_tunnel = NoiseTunnel(saliency)
        input_range = x_img.max() - x_img.min()
        noise_level = std * input_range
        attribution = noise_tunnel.attribute(
            inputs=x_img,
            nt_type="smoothgrad",
            nt_samples=n_samples,
            target=y.to(x_img.device),
            stdevs=noise_level.item() if use_noise_level else std,
            abs=False,
        )

        grads = torch.autograd.grad(x_img, x_in, grad_outputs=-attribution)[0]

        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            grads[grads.abs() < threshold * max_values] = 0.0

        return grads, attribution

    def smooth_grad_ig(
        self,
        x_t: torch.Tensor,
        y: list,
        source_class: list,
        s: float,
        classifier: nn.Module,
        n_samples: int,
        n_steps: int,
        std: float,
        use_noise_level: bool,
        resize: bool,
        threshold: float,
        baseline: torch.Tensor,
        height: int,
        width: int,
        batch_size: int,
        predictor_img_size: tuple,
        **kwargs,
    ):
        """Integrated gradients whose path gradients are SmoothGrad attributions.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, 16, h, w)`` or image when ``resize`` is ``True``.
        y : torch.Tensor
            Target classes.
        source_class : object
            Unused.
        s : float
            Loss temperature.
        classifier : nn.Module
            Classifier (set to eval mode).
        n_samples : int
            Noisy samples per path point.
        n_steps : int
            Number of IG path points.
        std : float
            Noise standard deviation (or fraction of the input range).
        use_noise_level : bool
            Scale ``std`` by ``max - min`` of the image.
        resize : bool
            If ``True`` skip the VAE decoding.
        threshold : float or None
            Zero latent-gradient entries with magnitude below ``threshold * max``.
        baseline : torch.Tensor or None
            IG baseline (see ``IG``).
        height, width, batch_size : int
            Passed to ``generate_baseline``.
        predictor_img_size : tuple[int, int]
            Input size of the classifier.
        **kwargs
            ``generator_to_classifier`` callable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Latent gradient and pixel-space IG attribution.
        """
        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        generator_to_classifier = kwargs.get("generator_to_classifier", None)
        if resize:
            x_img = x_in
        else:
            x_img = self.latent_decoder(x_in)
        x_img = generator_to_classifier(x_img) if generator_to_classifier else x_img
        x_img = transforms.Resize(predictor_img_size)(x_img)
        classifier.eval()
        inputs = x_img.clone().detach().requires_grad_(True)
        saliency = Saliency(classifier)

        # Wrap with NoiseTunnel for SmoothGrad
        input_range = x_img.max() - x_img.min()
        noise_level = std * input_range
        noise_grad_args = {
            "nt_type": "smoothgrad",
            "nt_samples": n_samples,
            "stdevs": noise_level.item() if use_noise_level else std,
            "target": y,
        }

        noise_tunnel = NoiseTunnel(saliency)
        ig_grads = self.IG(
            inputs=inputs,
            baseline=baseline,
            n_steps=n_steps,
            y=y,
            s=s,
            model=classifier,
            height=height,
            width=width,
            batch_size=batch_size,
            Noise_tunnel=noise_tunnel,
            **noise_grad_args,
        )

        grads = torch.autograd.grad(x_img, x_in, grad_outputs=ig_grads)[0]
        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            # max_values = grads.abs().max()
            grads[grads.abs() < threshold * max_values] = 0.0
            # grads[grads > (1 - threshold) * max_values] = 0.0

        return grads, ig_grads

    def integrated_gradient(
        self,
        x_t,
        y,
        source_class,
        s,
        n_steps,
        baseline,
        height,
        width,
        resize,
        classifier,
        batch_size,
        threshold,
        predictor_img_size,
        **kwargs,
    ):
        """Plain integrated gradients pulled back into the latent.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, 16, h, w)`` or image when ``resize`` is ``True``.
        y : torch.Tensor
            Target classes.
        source_class : object
            Unused.
        s : float
            Loss temperature.
        n_steps : int
            Number of IG path points.
        baseline : torch.Tensor or None
            IG baseline (see ``IG``).
        height, width, batch_size : int
            Passed to ``generate_baseline``.
        resize : bool
            If ``True`` skip the VAE decoding.
        classifier : nn.Module
            Classifier.
        threshold : float or None
            Zero latent-gradient entries with magnitude below ``threshold * max``.
        predictor_img_size : tuple[int, int]
            Input size of the classifier.
        **kwargs
            ``generator_to_classifier`` callable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Latent gradient and pixel-space IG attribution.
        """

        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        generator_to_classifier = kwargs.get("generator_to_classifier", None)
        if resize:
            x_img = x_in
        else:
            x_img = self.latent_decoder(x_in)
        x_img = generator_to_classifier(x_img) if generator_to_classifier else x_img
        x_img = transforms.Resize(predictor_img_size)(x_img)
        classifier.eval()
        inputs = x_img.clone().detach().requires_grad_(True)
        ig_grads = self.IG(
            inputs=inputs,
            baseline=baseline,
            n_steps=n_steps,
            y=y,
            s=s,
            model=classifier,
            height=height,
            width=width,
            batch_size=batch_size,
        )
        grads = torch.autograd.grad(x_img, x_in, grad_outputs=ig_grads)[0]
        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            # max_values = grads.abs().max()
            grads[grads.abs() < threshold * max_values] = 0.0
            # grads[grads > (1 - threshold) * max_values] = 0.0

        return grads, ig_grads

    def smooth_diff_ig(
        self,
        x_t: torch.Tensor,
        y: list,
        source_class: list,
        s: float,
        classifier: nn.Module,
        n_samples: int,
        use_noise_level: bool,
        n_steps: int,
        std: float,
        resize: bool,
        threshold: float,
        baseline: torch.Tensor,
        height: int,
        width: int,
        batch_size: int,
        predictor_img_size: tuple,
        **kwargs,
    ):
        """Integrated gradients through a SmoothDiff-smoothed classifier.

        The classifier's nonlinearities are replaced by SmoothDiff layers whose
        statistics are collected over ``n_samples`` noisy forward passes; the
        backward pass then uses the smoothed derivatives. ``reset_statisticcs``
        restores the model afterwards.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, 16, h, w)`` or image when ``resize`` is ``True``.
        y : torch.Tensor
            Target classes.
        source_class : object
            Unused.
        s : float
            Loss temperature.
        classifier : nn.Module
            Classifier to smooth.
        n_samples : int
            Noisy forward passes for the statistics.
        use_noise_level : bool
            Scale ``std`` by ``max - min`` of the image.
        n_steps : int
            Number of IG path points.
        std : float
            Noise standard deviation (or fraction of the input range).
        resize : bool
            If ``True`` skip the VAE decoding.
        threshold : float or None
            Zero latent-gradient entries with magnitude below ``threshold * max``.
        baseline : torch.Tensor or None
            IG baseline (see ``IG``).
        height, width, batch_size : int
            Passed to ``generate_baseline``.
        predictor_img_size : tuple[int, int]
            Input size of the classifier.
        **kwargs
            ``generator_to_classifier`` callable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Latent gradient and pixel-space IG attribution.
        """
        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        generator_to_classifier = kwargs.get("generator_to_classifier", None)
        if resize:
            x_img = x_in
        else:
            x_img = self.latent_decoder(x_in)
        x_img = generator_to_classifier(x_img) if generator_to_classifier else x_img
        x_img = transforms.Resize(predictor_img_size)(x_img)
        classifier.eval()
        input_range = x_img.max() - x_img.min()
        noise_level = std * input_range
        model = replace_nonlinear_layers(classifier)
        set_smoothdiff_layer_mode(model, collect_stats=True, smooth_backward=False)
        with torch.no_grad():
            for _ in range(n_samples):
                noise = torch.randn_like(x_img)
                model(x_img + noise * (noise_level.item() if use_noise_level else std))
        set_smoothdiff_layer_mode(model, collect_stats=False, smooth_backward=True)
        inputs = x_img.clone().detach().requires_grad_(True)
        ig_grads = self.IG(
            inputs=inputs,
            baseline=baseline,
            n_steps=n_steps,
            y=y,
            s=s,
            model=model,
            height=height,
            width=width,
            batch_size=batch_size,
        )
        grads = torch.autograd.grad(x_img, x_in, grad_outputs=ig_grads)[0]
        reset_statisticcs(model)
        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            # max_values = grads.abs().max()
            grads[grads.abs() < threshold * max_values] = 0.0
            # grads[grads > (1 - threshold) * max_values] = 0.0

        return grads, ig_grads

    def smooth_diff_grad(
        self,
        x_t,
        y,
        source_class,
        s,
        classifier,
        n_samples,
        use_noise_level,
        std,
        resize,
        threshold,
        predictor_img_size,
        **kwargs,
    ):
        """SmoothDiff gradient of ``-s * logit[y]`` pulled back into the latent.

        Same smoothing setup as ``smooth_diff_ig`` but with a single backward pass
        instead of integrated gradients.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, 16, h, w)`` or image when ``resize`` is ``True``.
        y : torch.Tensor
            Target classes.
        source_class : object
            Unused.
        s : float
            Loss temperature.
        classifier : nn.Module
            Classifier to smooth.
        n_samples : int
            Noisy forward passes for the statistics.
        use_noise_level : bool
            Scale ``std`` by ``max - min`` of the image.
        std : float
            Noise standard deviation (or fraction of the input range).
        resize : bool
            If ``True`` skip the VAE decoding.
        threshold : float or None
            Signed sparsification as in ``clean_multiclass_cond_fn``.
        predictor_img_size : tuple[int, int]
            Input size of the classifier.
        **kwargs
            ``generator_to_classifier`` callable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Latent gradient and pixel-space gradient.
        """
        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        generator_to_classifier = kwargs.get("generator_to_classifier", None)
        if resize:
            x_img = x_in
        else:
            x_img = self.latent_decoder(x_in)
        x_img = generator_to_classifier(x_img) if generator_to_classifier else x_img
        x_img = transforms.Resize(predictor_img_size)(x_img)
        classifier.eval()
        input_range = x_img.max() - x_img.min()
        noise_level = std * input_range
        model = replace_nonlinear_layers(classifier)
        set_smoothdiff_layer_mode(model, collect_stats=True, smooth_backward=False)
        with torch.no_grad():
            for _ in range(n_samples):
                noise = torch.randn_like(x_img)
                model(x_img + noise * (noise_level if use_noise_level else std))
        set_smoothdiff_layer_mode(model, collect_stats=False, smooth_backward=True)
        inputs = x_img.clone().detach().requires_grad_(True)
        output = model(inputs)
        output = -output[range(len(y)), y]
        output = output * s
        output.sum().backward()
        attribute = inputs.grad
        grads = torch.autograd.grad(x_img, x_in, grad_outputs=attribute)[0]
        reset_statisticcs(model)
        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            # max_values = grads.abs().max()
            grads[grads < threshold * max_values] = 0.0
            # grads[grads > (1 - threshold) * max_values] = 0.0
        return grads, attribute

    def manifold_aware_smoothing(
        self,
        x_t,
        y,
        source_class,
        s,
        classifier,
        n_samples=20,
        strength=0.3,
        use_gfm=True,
        prompt=None,
        guidance_scale=None,
        num_inference_steps=None,
        resize=False,
        threshold=None,
        predictor_img_size=None,
        height=None,
        use_logits=True,
        **kwargs,
    ):
        """Average classifier gradients over SD3 img2img resamples of the image.

        For each of ``n_samples`` draws the decoded image is re-noised to strength
        ``strength`` with ``encode`` and denoised again with ``decode``; the
        gradient of ``-s * logit[y]`` at the resampled image is collected. The
        mean gradient is propagated through the VAE decoder into the latent.

        Parameters
        ----------
        x_t : torch.Tensor
            Latent ``(B, 16, h, w)`` or image when ``resize`` is ``True``.
        y : torch.Tensor
            Target classes.
        source_class : object
            Unused.
        s : float
            Loss temperature.
        classifier : nn.Module
            Classifier (set to eval mode).
        n_samples : int, optional
            Number of resamples.
        strength : float, optional
            img2img strength of each resample.
        use_gfm : bool, optional
            Unused.
        prompt : str, optional
            Unused; ``decode`` always uses ``self.prompt``.
        guidance_scale : float, optional
            Classifier-free guidance scale passed to ``decode``.
        num_inference_steps : int, optional
            Euler steps per resample; defaults to ``self.steps_number``.
        resize : bool, optional
            If ``True`` skip the VAE decoding.
        threshold : float, optional
            Zero latent-gradient entries with magnitude below ``threshold * max``.
        predictor_img_size : tuple[int, int], optional
            Input size of the classifier.
        height : int, optional
            Unused.
        use_logits : bool, optional
            Differentiate raw logits instead of log-probabilities.
        **kwargs
            ``generator_to_classifier`` callable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Latent gradient and the averaged pixel-space gradient.

        Notes
        -----
        ``decode`` only decodes the first element of its input, so the resampled
        batch has size one.
        """
        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        generator_to_classifier = kwargs.get("generator_to_classifier", None)
        num_inference_steps = num_inference_steps or self.steps_number

        if resize:
            x_img = x_in
        else:
            x_img = self.latent_decoder(x_in)

        classifier.eval()

        grads_list = []
        for _ in range(n_samples):
            with torch.no_grad():
                sampled_latent = self.encode(
                    x_img.detach(),
                    t=strength,
                    num_steps=num_inference_steps,
                )
                sampled = self.decode(
                    z=sampled_latent,
                    t=strength,
                    num_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

            sampled = sampled.detach().requires_grad_(True).to(self.device)
            sampled_to_cls = (
                generator_to_classifier(sampled) if generator_to_classifier else sampled
            )
            sampled_to_cls = transforms.Resize(predictor_img_size)(sampled_to_cls)

            y_pred = classifier(sampled_to_cls)
            if not use_logits:
                y_pred = F.log_softmax(y_pred, dim=1)
            y_pred_target = -y_pred[range(len(y)), y] * s
            sample_grad = torch.autograd.grad(
                y_pred_target.sum(), sampled, retain_graph=False
            )[0]

            grads_list.append(sample_grad)

            attribute = torch.stack(grads_list).mean(dim=0)

        grads = torch.autograd.grad(x_img, x_in, grad_outputs=attribute)[0]

        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            grads[grads.abs() < threshold * max_values] = 0.0

        return grads, attribute

    def GMD(
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
        """Guided-diffusion draft with a binary (``y * logit``) guidance loss.

        Encodes ``img``, noises it to time ``t`` and steps through the sigma
        schedule while computing classifier and distance gradients w.r.t. ``z_t``.
        The gradients are accumulated but never applied, and the Euler update is
        performed once after the loop with the last step index. The method
        also references an undefined ``to_pil`` and is not reachable from
        ``edit``; it is kept as an experimental stub.

        Parameters
        ----------
        img : torch.Tensor
            Input image ``(1, 3, H, W)``.
        inpaint, dilation : float
            Unused.
        t : float
            Edit strength.
        guided_iterations : int
            Unused.
        class_grad_kwargs : dict
            Must contain ``classifier``, ``y``, ``use_logits`` and ``s``.
        dist_grad_kargs : dict
            Must contain ``l1_loss``, ``l2_loss`` and ``l_perc``.
        explainer_config : ExplainerConfig
            Unused.
        scale_grads : bool, optional
            Unused.
        boolmask_in : torch.Tensor, optional
            Mask that is interpolated to latent size (unused afterwards).

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
            Final noisy latent and the per-step ``x_t`` / ``z_t`` histories.
        """

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
        _log.info("%s %s", "time: ", t)
        _log.info("%s %s", "shift", self.shift)
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

        for i in range(num_inference_steps):
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

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

            # extract sqrtalphacum
            height, width = img.shape[2:]
            boolmask = boolmask_in
            boolmask_latent = torch.nn.functional.interpolate(
                boolmask,
                size=(height // vae_scale_factor, width // vae_scale_factor),
            )
            # nonzero_mask = (
            #     (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            # )  # no noise when t == 0

            grads = 0

            x_t_0_denoise = x_t - sigmas[i] * noise_pred
            x_t_0_denoise_clone = x_t_0_denoise
            classifier = class_grad_kwargs["classifier"]
            y = class_grad_kwargs["y"]
            use_logits = class_grad_kwargs["use_logits"]
            s = class_grad_kwargs["s"]
            logits = classifier(x_t_0_denoise_clone)
            y = y.to(logits.device).float()
            selected = y * logits - (1 - y) * logits
            if use_logits:
                selected = -selected
            else:
                selected = -F.logsigmoid(selected)
            selected = selected * s

            grads = grads + torch.autograd.grad(selected.sum(), z_t)[0]

            m1 = (
                dist_grad_kargs["l1_loss"] * torch.norm(z_t - img, p=1, dim=1).sum()
                if dist_grad_kargs["l1_loss"] != 0
                else 0
            )
            m2 = (
                dist_grad_kargs["l2_loss"] * torch.norm(z_t - img, p=2, dim=1).sum()
                if dist_grad_kargs["l2_loss"] != 0
                else 0
            )
            mv = (
                dist_grad_kargs["l_perc"](x_t_0_denoise_clone, img)
                if dist_grad_kargs["l_perc"] is not None
                else 0
            )
            dist_grads = torch.autograd.grad(m1 + m2, z_t)[0]
            dist_grads = dist_grads + torch.autograd.grad(mv, x_t_0_denoise_clone)[0]
            grads = grads + dist_grads

        #     out["mean"] = out["mean"].float() - out["variance"] * grads
        dt = sigmas[i + 1] - sigmas[i]
        z_t = z_t + dt * noise_pred
        x_t = z_t - (sigmas[i]) * noise_pred

        return z_t, x_t_steps, z_t_steps

    @torch.no_grad()
    def _visualize_gradient(
        self,
        x_original,
        z_t,
        x_t,
        clean_img_old,
        grads,
        pixel_grad,
        boolmask,
        boolmask_in,
        best_z,
        best_mask,
        step_idx,
        save_dir,
    ):
        """
        Visualize a single FastDiME step similar to visualize_step in counterfactual_explainer.py

        Args:
            x_original: Original input image (B, C, H, W)
            z_t: Current noisy latent (B, C, H, W)
            x_t: Current denoised latent (B, C, H, W)
            clean_img_old: Previous iteration's decoded image (B, C, H, W)
            grads: Gradient in latent space (B, C, H, W)
            pixel_grad: Gradient/relevance in pixel space (B, C, H, W) or None
            boolmask: Current mask from self-optimized masking (B, 1, H, W)
            boolmask_in: Input mask if using fixed masking (B, C, H, W) or None
            best_z: Best latent found so far (B, C, H, W)
            best_mask: Best mask found so far (B, C, H, W)
            step_idx: Current step index
            save_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(save_dir, exist_ok=True)
        to_pil = ToPILImage()

        with torch.no_grad():
            # Decode latents to pixel space
            z_decoded = self.latent_decoder(z_t.to("cuda")).cpu()
            x_decoded = self.latent_decoder(x_t.to("cuda")).cpu()
            best_z_decoded = self.latent_decoder(best_z.to("cuda")).cpu()

            # # Decode gradients to pixel space for visualization
            grads_decoded = self.latent_decoder(grads.to("cuda").detach()).cpu()

            batch_size = x_original.shape[0]

            for b in range(batch_size):
                # Create figure with subplots
                fig, axes = plt.subplots(3, 4, figsize=(20, 15))
                fig.suptitle(f"FastDiME Step {step_idx}", fontsize=16)

                # Row 1: Original, Current z_t decoded, Current x_t decoded, Previous state
                # Original image
                img_orig = x_original[b].clamp(0, 1).permute(1, 2, 0).numpy()
                axes[0, 0].imshow(img_orig)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis("off")

                # Current z_t decoded
                img_z = z_decoded[b].clamp(0, 1).permute(1, 2, 0).numpy()
                axes[0, 1].imshow(img_z)
                axes[0, 1].set_title("z_t Decoded (Noisy)")
                axes[0, 1].axis("off")

                # Current x_t decoded
                img_x = x_decoded[b].clamp(0, 1).permute(1, 2, 0).numpy()
                axes[0, 2].imshow(img_x)
                axes[0, 2].set_title("x_0 Predicteed Decoded (Denoised)")
                axes[0, 2].axis("off")

                # Previous state
                if isinstance(clean_img_old, torch.Tensor):
                    img_old = clean_img_old[b].clamp(0, 1).permute(1, 2, 0).numpy()
                else:
                    img_old = np.zeros_like(img_orig)
                axes[0, 3].imshow(img_old)
                axes[0, 3].set_title("Previous State")
                axes[0, 3].axis("off")

                # Row 2: Gradients and masks
                # Latent gradient (decoded)
                grad_img = grads_decoded[b].permute(1, 2, 0).clamp(0, 1)
                ref = torch.zeros_like(grad_img)
                # heatmap = high_contrast_heatmap(ref, -grad_img)[0]
                grad_magnitude = np.abs(grad_img).sum(axis=-1)
                vmax = np.abs(grad_magnitude).max() + 1e-8
                im1 = axes[1, 0].imshow(grad_magnitude, cmap="hot", vmin=0, vmax=vmax)
                axes[1, 0].set_title("Latent Gradient (Decoded)")
                axes[1, 0].axis("off")
                # plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

                # Pixel gradient/relevance
                if pixel_grad is not None:
                    pixel_grad_np = pixel_grad[b].detach().cpu()
                    ref = torch.zeros_like(pixel_grad_np)
                    heatmap = high_contrast_heatmap(ref, -pixel_grad_np)[0]
                    # if pixel_grad_np.dim() == 3:
                    #     pixel_grad_magnitude = pixel_grad_np.abs().sum(dim=0).numpy()
                    # else:
                    #     pixel_grad_magnitude = pixel_grad_np.abs().numpy()
                    # vmax_pixel = np.abs(pixel_grad_magnitude).max()
                    im2 = axes[1, 1].imshow(heatmap.permute(1, 2, 0))
                    axes[1, 1].set_title("Pixel Gradient/Relevance")
                    axes[1, 1].axis("off")
                    # plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
                else:
                    axes[1, 1].text(
                        0.5, 0.5, "No pixel gradient", ha="center", va="center"
                    )
                    axes[1, 1].axis("off")

                # Current boolmask
                if boolmask is not None:
                    mask_np = boolmask[b, 0].detach().cpu().numpy()
                    axes[1, 2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
                    axes[1, 2].set_title("Current Boolmask")
                    axes[1, 2].axis("off")
                else:
                    axes[1, 2].text(0.5, 0.5, "No mask", ha="center", va="center")
                    axes[1, 2].axis("off")

                # Input boolmask
                if boolmask_in is not None:
                    mask_in_np = boolmask_in[b, 0].detach().cpu().numpy()
                    axes[1, 3].imshow(mask_in_np, cmap="gray", vmin=0, vmax=1)
                    axes[1, 3].set_title("Input Boolmask")
                    axes[1, 3].axis("off")
                else:
                    axes[1, 3].text(0.5, 0.5, "No input mask", ha="center", va="center")
                    axes[1, 3].axis("off")

                # Row 3: Best results and overlays
                # Best z decoded
                img_best = best_z_decoded[b].clamp(0, 1).permute(1, 2, 0).numpy()
                axes[2, 0].imshow(img_best)
                axes[2, 0].set_title("Best z Decoded")
                axes[2, 0].axis("off")

                # Best mask
                if best_mask is not None:
                    best_mask_np = (
                        best_mask[b, 0].detach().cpu().numpy()
                        if best_mask.shape[1] == 1
                        else best_mask[b].mean(dim=0).detach().cpu().numpy()
                    )
                    axes[2, 1].imshow(best_mask_np, cmap="gray", vmin=0, vmax=1)
                    axes[2, 1].set_title("Best Mask")
                    axes[2, 1].axis("off")
                else:
                    axes[2, 1].text(0.5, 0.5, "No best mask", ha="center", va="center")
                    axes[2, 1].axis("off")

                # Difference heatmap (current - original)
                heatmap = high_contrast_heatmap(x_original[b], x_decoded[b])[0]
                # diff = np.abs(img_x - img_orig).sum(axis=-1)
                # diff_normalized = diff / (diff.max() + 1e-8)
                im3 = axes[2, 2].imshow(heatmap.permute(1, 2, 0))
                axes[2, 2].set_title("Difference (Current - Original)")
                axes[2, 2].axis("off")

                # Overlay: gradient on original
                if pixel_grad is not None:
                    ref = torch.zeros_like(pixel_grad[b])
                    pixel_grad_np = high_contrast_heatmap(ref, -pixel_grad[b].cpu())[0]
                    # pixel_grad_np = pixel_grad[b].detach().cpu()
                    if pixel_grad_np.dim() == 3:
                        pixel_grad_magnitude = pixel_grad_np.abs().sum(dim=0).numpy()
                    else:
                        pixel_grad_magnitude = pixel_grad_np.abs().numpy()
                    pixel_grad_normalized = pixel_grad_magnitude / (
                        pixel_grad_magnitude.max() + 1e-8
                    )
                    overlay = (
                        img_orig * 0.4
                        + plt.cm.jet(pixel_grad_normalized)[:, :, :3] * 0.6
                    )
                    axes[2, 3].imshow(overlay)
                    axes[2, 3].set_title("Gradient Overlay")
                    axes[2, 3].axis("off")
                else:
                    axes[2, 3].imshow(img_orig)
                    axes[2, 3].set_title("Original (No Overlay)")
                    axes[2, 3].axis("off")

                plt.tight_layout()

                # Save figure
                save_path = os.path.join(
                    save_dir, f"step_{self._sample_id}_{step_idx:04d}_batch_{b}.png"
                )
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Also save individual images for easier inspection
                individual_dir = os.path.join(save_dir, "individual", f"batch_{b}")
                os.makedirs(individual_dir, exist_ok=True)

                to_pil(x_original[b].clamp(0, 1)).save(
                    os.path.join(
                        individual_dir,
                        f"step_{self._sample_id}_{step_idx:04d}_original.png",
                    )
                )
                to_pil(z_decoded[b].clamp(0, 1)).save(
                    os.path.join(
                        individual_dir,
                        f"step_{self._sample_id}_{step_idx:04d}_z_decoded.png",
                    )
                )
                to_pil(x_decoded[b].clamp(0, 1)).save(
                    os.path.join(
                        individual_dir,
                        f"step_{self._sample_id}_{step_idx:04d}_x_decoded.png",
                    )
                )

                if boolmask is not None:
                    to_pil(boolmask[b].clamp(0, 1)).save(
                        os.path.join(
                            individual_dir,
                            f"step_{self._sample_id}_{step_idx:04d}_mask.png",
                        )
                    )

                # Save gradient heatmap
                if pixel_grad is not None:
                    ref = torch.zeros_like(pixel_grad[b])
                    heatmap = high_contrast_heatmap(ref, -pixel_grad[b].cpu())[0]
                    to_pil(heatmap).save(
                        os.path.join(
                            individual_dir,
                            f"step_{self._sample_id}_{step_idx:04d}_gradient_heatmap.png",
                        )
                    )

        _log.info("%s", f"Visualization saved for step {step_idx}")

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
        base_path=None,
        mode: str = "",
        visualize_gradients: bool = False,
    ):
        """FastDiME counterfactual generation in the SD3 latent space.

        The image latent is noised to strength ``t`` and denoised with Euler steps.
        After every transformer call the classifier gradient (selected by
        ``self.gradient_smoothing``; see the ``*_grad``/``LRP``/``IG`` methods) and
        the ``dist_cond_fn`` distance gradient towards the original latent are
        computed on the current prediction of ``x_0``, clipped to
        ``explainer_config.gradient_clipping`` in the inf-norm and subtracted from
        ``z_t`` with step size ``class_grad_kwargs["lr"]`` (optionally with
        momentum). If ``self_optamized_masking`` is set and ``i > warmup_steps`` a
        mask of changed pixels (Gaussian-blur mask or ACE-style ``generate_mask``,
        threshold ``explainer_config.inpaint``) freezes the unchanged region by
        copying the (re-noised) original latent into it; a given ``boolmask_in``
        does the same with a fixed mask.

        Parameters
        ----------
        img : torch.Tensor
            Input images ``(B, 3, H, W)`` in ``[0, 1]`` at the generator size.
        inpaint, dilation : float
            Unused; the values from ``explainer_config`` are used instead.
        t : float
            Edit strength (``sampling_time_fraction``).
        guided_iterations : int
            Unused.
        class_grad_kwargs : dict
            Keyword arguments of the classifier-gradient method (``y``,
            ``classifier``, ``s``, ``lr``, ``predictor_img_size``,
            ``generator_to_classifier`` ...).
        dist_grad_kargs : dict
            ``l1_loss``, ``l2_loss``, ``l_perc``, ``lr``, ``momentum`` for
            ``dist_cond_fn``.
        explainer_config : ExplainerConfig
            Provides ``gradient_clipping``, ``dilation``, ``inpaint`` and
            optionally ``momentum``.
        scale_grads : bool, optional
            Unused.
        boolmask_in : torch.Tensor, optional
            Fixed binary mask ``(B, 1, H, W)`` (1 = frozen) from a previous pass.
        base_path : str, optional
            Run directory; with ``visualize_gradients`` step figures are written to
            ``<base_path>/<mode>_fastdime``.
        mode : str, optional
            Name suffix of the visualisation directory.
        visualize_gradients : bool, optional
            Dump a figure per step via ``_visualize_gradient``.

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
            Final decoded images ``(B, 3, H, W)`` and the CPU histories of ``x_t``
            and ``z_t`` before every step.

        Notes
        -----
        The prompt embeddings are computed for empty strings regardless of
        ``self.prompt``, and ``torch.manual_seed`` is reseeded randomly.
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
        height, width = self.config.data.input_size[1:]
        vae_scale_factor = self.pipe.vae_scale_factor
        with torch.no_grad():
            latents = self.latent_encoder(img.to(self.device))
        # Initialize x_t as the current latent (will be updated iteratively)
        import random

        x_t = latents.clone()
        velocity = torch.zeros_like(latents)
        momentum = (
            explainer_config.momentum if hasattr(explainer_config, "momentum") else 0.9
        )
        torch.manual_seed(random.randint(1, 100))
        noise = torch.randn_like(x_t, device=self.device)
        z_t = t * noise + (1 - t) * x_t
        x_t_steps = []
        z_t_steps = []
        # encoding prompts
        # vislulize the gradients
        if visualize_gradients and base_path is not None:
            x_orginal = img.clone().detach().cpu()
            gradients_path = os.path.join(base_path, f"{mode}_fastdime")
            Path(gradients_path).mkdir(parents=True, exist_ok=True)

        prompt = [self.prompt] * img.shape[0]
        prompt = [""] * img.shape[0]
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

        best_z = z_t.clone()
        best_mask = torch.ones_like(img)
        lr = class_grad_kwargs["lr"]
        # Main iterative denoising loop in latent space
        # z_t = nn.Parameter(z_t.clone().detach(), requires_grad=True)
        # optimizer = torch.optim.SGD([z_t], lr=lr, momentum=momentum)

        for i in range(num_inference_steps):
            x_t_steps.append(x_t.detach().cpu().clone())
            z_t_steps.append(z_t.detach().cpu().clone())
            clean_img_old = (
                self.latent_decoder(z_t.detach()).cpu() if i > 0 else img.clone()
            )
            # Use the Transformer to predict the noise residual.
            self.pipe.transformer.enable_gradient_checkpointing()
            if do_classifier_free_guidance:
                x_t = torch.cat([x_t] * 2)
                z_t = torch.cat([z_t] * 2)

            t = self.rescale_time_step(sigmas[i])
            timestep = t.expand(x_t.shape[0])
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
                    x_t = x_t.chunk(2)[0]
                    z_t = z_t.chunk(2)[0]
            # Compute guidance gradients:
            grads = 0
            pixel_grads = None
            relevance = None

            xt_in = torch.clone(x_t.detach())
            if latents.shape != xt_in.shape:
                xt_in = xt_in.chunk(2)[0]

            if (
                self.gradient_smoothing == "distilled"
                or self.gradient_smoothing == "vanilla"
            ):
                class_grad, pixel_grads = self.clean_multiclass_cond_fn(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                )
                # class_grad = class_grad / (t + 5.0e-3 if scale_grads else 1)
                grads += class_grad
            if self.gradient_smoothing == "lrp":
                grad, relevance = self.LRP(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                )
                grads += grad
            if self.gradient_smoothing == "smoothdiff":
                smooth_grad, pixel_grads = self.smooth_diff_grad(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                    **self.smoothdiff_parameters,
                )
                grads += smooth_grad

            if self.gradient_smoothing == "smoothgrad":
                smooth_grad, pixel_grads = self.smooth_grad_captum(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                    **self.smoothdiff_parameters,
                )
                grads += smooth_grad
            if self.gradient_smoothing == "manifold_aware":
                manifold_grad, pixel_grads = self.manifold_aware_smoothing(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                    **self.manifold_smoothing_parameters,
                )
                grads += manifold_grad
            if self.gradient_smoothing == "integrated_gradient":
                ig_grad, pixel_grads = self.integrated_gradient(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                    **self.parameters_integrated_grad,
                )
                grads += ig_grad
            if self.gradient_smoothing == "smooth_integrated_gradient":
                ig_grad, pixel_grads = self.smooth_diff_ig(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                    **self.parameters_integrated_grad,
                    **self.smoothdiff_parameters,
                )

                grads += ig_grad

            if self.gradient_smoothing == "smoothgrad_integrated_gradient":
                ig_grad, pixel_grads = self.smooth_grad_ig(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                    **self.parameters_integrated_grad,
                    **self.smoothdiff_parameters,
                )

                grads += ig_grad
            if self.gradient_smoothing == "deep_lift":
                deep_lift, pixel_grads = self.deeplift(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                    **self.parameters_integrated_grad,
                )
                grads += deep_lift

            x_t_in = torch.clone(x_t.detach())
            z_t_in = torch.clone(z_t.detach())
            if latents.shape != x_t_in.shape:
                x_t_in = x_t_in.chunk(2)[0]
                z_t_in = z_t_in.chunk(2)[0]
            dist_grad = self.dist_cond_fn(
                x_tau=latents,
                z_t=z_t_in,
                x_t=x_t_in,
                alpha_t=(1 - t + 5.0e-3),
                scale_grads=False,
                **dist_grad_kargs,
            )
            grads = grads + dist_grad

            dt = sigmas[i + 1] - sigmas[i]
            z_t = z_t + dt * noise_pred
            x_t = z_t - (sigmas[i]) * noise_pred

            norm = grads.norm(p=float("inf"))
            # print("norm", norm)
            if norm > explainer_config.gradient_clipping:
                _log.info("%s", "gradient clipping")
                rescale_factor = explainer_config.gradient_clipping / norm
                grads = grads * rescale_factor
            if boolmask_in is None:
                boolmask_latent = torch.ones_like(z_t)
            else:
                boolmask_latent = torch.nn.functional.interpolate(
                    (1 - boolmask_in),
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                )

            # lr = class_grad_kwargs["lr"]
            if self.use_momentum:
                velocity_prev = velocity.clone()
                velocity = momentum * velocity_prev - lr * grads
                z_t = z_t + velocity
            else:
                z_t = z_t - lr * grads
            # optimizer.zero_grad()
            # z_t.grad = grads
            # optimizer.step()

            torch.cuda.empty_cache()
            # Apply FASTDiME self-optimized masking if enabled:
            with torch.no_grad():
                x_0_denoised = self.latent_decoder(x_t.detach())
                if self.use_gussian_blur_masking:
                    mask_t, dil_mask = generate_smooth_mask(
                        img.to(self.device),
                        x_0_denoised,
                        explainer_config.dilation,
                    )
                else:
                    mask_t, dil_mask = self.generate_mask(
                        img.to(self.device),
                        x_0_denoised,
                        explainer_config.dilation,
                    )

                boolmask = (dil_mask < explainer_config.inpaint).float()
                boolmask_latent = None
                if boolmask_in is not None:
                    if boolmask.shape != boolmask_in.shape:
                        boolmask_in = transforms.Resize(boolmask.shape[2:])(boolmask_in)
                    added_term = torch.ones_like(boolmask) - boolmask_in[:, 0:1].to(
                        boolmask
                    )
                    new_candidate = boolmask + added_term
                    boolmask = torch.minimum(torch.ones_like(boolmask), new_candidate)

            if visualize_gradients and base_path is not None:
                self._visualize_gradient(
                    x_original=x_orginal,
                    z_t=z_t.clone().detach().cpu(),
                    x_t=x_t.clone().detach().cpu(),
                    clean_img_old=clean_img_old.clone().detach().cpu(),
                    grads=grads.clone().detach().cpu(),
                    pixel_grad=(
                        pixel_grads.clone().detach().cpu()
                        if pixel_grads is not None
                        else relevance.clone().detach().cpu()
                    ),
                    boolmask=boolmask.clone().detach().cpu(),
                    boolmask_in=(
                        boolmask_in.clone().detach().cpu()
                        if boolmask_in is not None
                        else boolmask_in
                    ),
                    best_z=best_z.clone().detach().cpu(),
                    best_mask=best_mask.clone().detach().cpu(),
                    step_idx=i,
                    save_dir=gradients_path,
                )

            if boolmask_in is not None and i > self.warmup_steps:
                # fixed mask
                # print("fixed_mask")

                height, width = img.shape[2:]
                # boolmask = boolmask_in
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
                    z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise

            if self_optimized_masking and i > self.warmup_steps and boolmask_in is None:
                # print("self optimized mask")
                # extract time-depedent mask (Eq. 6)

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
                    z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise
        with torch.no_grad():
            if boolmask_latent is not None:

                z_t = z_t * (1 - boolmask_latent) + boolmask_latent * latents
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
        """FastDiME variant on top of the edit-friendly DDPM inversion.

        Instead of noising the latent with fresh Gaussian noise, ``encode_image``
        supplies the starting latent ``x_T`` and the per-step noise maps that are
        added back after every Euler step, so the unguided trajectory reproduces
        the input. Guidance, gradient clipping, self-optimised and fixed masking
        follow ``FastDiME`` (without warm-up, momentum or gradient smoothing).

        Parameters
        ----------
        img : torch.Tensor
            Input image ``(1, 3, H, W)`` in ``[0, 1]``.
        inpaint, dilation : float
            Unused; values from ``explainer_config`` are used.
        t : float
            Edit strength.
        guided_iterations : int
            Unused.
        class_grad_kwargs : dict
            Arguments of ``clean_multiclass_cond_fn`` plus ``lr``.
        dist_grad_kargs : dict
            Arguments of ``dist_cond_fn``.
        explainer_config : ExplainerConfig
            Provides ``gradient_clipping``, ``dilation`` and ``inpaint``.
        scale_grads : bool, optional
            Divide the classifier gradient by ``t + 5e-3``.
        boolmask_in : torch.Tensor, optional
            Fixed binary mask applied after every step.

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
            Final decoded image and the ``x_t`` / ``z_t`` histories.

        Notes
        -----
        ``clean_multiclass_cond_fn`` returns a tuple, which this method divides by
        a scalar; the classifier-gradient step therefore fails at runtime as
        written. Debug images are written to ``_DEBUG_DIR``.
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
        _log.info("%s %s", "sigmas", sigmas)
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
        with torch.no_grad():
            self.latent_decoder(z_t.to("cuda"), img=True).save(
                f"{_DEBUG_DIR}/start_zt.png"
            )
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
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

            # Compute guidance gradients:
            grads = 0
            if class_grad_fn is not None:
                xt_in = torch.clone(x_t.detach())
                # visualzing the original image gradient
                class_grad = class_grad_fn(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                ) / (t + 5.0e-3 if scale_grads else 1)
                grads += class_grad
                # visualize the gradient
                with torch.no_grad():
                    _log.info("%s %s", "grads norm", grads.norm(p=float("inf")))

                    ref = torch.zeros_like(grads[0]).to("cpu")
                    grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))
                    # self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                    #     f"{_DEBUG_DIR}/grads_class{i}.png"
                    # )

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
                    # print("grads1 norm", grads.norm(p=float("inf")))

                    # grads_pixels = self.latent_decoder(grads).squeeze(0)
                    # ref = torch.zeros_like(grads_pixels).to("cpu")
                    # grad_img = high_contrast_heatmap(ref, grads_pixels.to("cpu"))

                    # self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                    #     f"{_DEBUG_DIR}/grads_class_dist{i}.png"
                    # )
                    # to_pil(grad_img[0]).save(
                    #     f"{_DEBUG_DIR}/grads_class_dist{i}.png"
                    # )
            dt = sigmas[i + 1] - sigmas[i]
            z_t = z_t + dt * noise_pred
            variance_noise = (
                variance_noises[i]
                if i < self.steps_number - 1
                else torch.zeros_like(x_t).to(self.dtype)
            )
            sigma_z = variance_noise
            x_t = z_t - (sigmas[i]) * noise_pred  # + sigma_z

            z_t = z_t + sigma_z
            # visualizing the results
            # self.latent_decoder(z_t, img=True).save(
            #     f"{_DEBUG_DIR}/z_t{i}.png"
            # )
            # self.latent_decoder(noise_pred, img=True).save(
            #     f"{_DEBUG_DIR}/noise_pred{i}.png"
            # )
            # self.latent_decoder(x_t, img=True).save(
            #     f"{_DEBUG_DIR}/x_t{i}.png"
            # )
            # scaling the gradeint
            norm = grads.norm(p=float("inf"))
            if norm > explainer_config.gradient_clipping:
                _log.info("%s", "gradient clipping")
                rescale_factor = explainer_config.gradient_clipping / norm
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
        """
        Extracts a mask by binarizing the difference between
        denoised image at time-step t and original input.
        We generate the mask similar to ACE.

        :x1: denoised image at time-step t
        :x2: original input image
        :dilation: dilation parameters
        """
        assert (dilation % 2) == 1, "dilation must be an odd number"
        x1 = (x1 + 1) / 2
        x2 = (x2 + 1) / 2
        mask = (x1 - x2).abs().sum(dim=1, keepdim=True)
        mask = mask / mask.view(mask.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        dil_mask = F.max_pool2d(mask, dilation, stride=1, padding=(dilation - 1) // 2)
        return mask, dil_mask

    def visualize_grads(
        self, original_img_latent, perturbed_latent, learning_rate, index
    ):
        """Save a heatmap of ``(perturbed - original) / learning_rate`` in pixel space.

        Both latents are decoded, their difference is rendered with
        ``high_contrast_heatmap`` and written to ``_DEBUG_DIR/grads_<index>.png``.

        Parameters
        ----------
        original_img_latent, perturbed_latent : torch.Tensor
            Latents before and after an optimiser step.
        learning_rate : float
            Step size used to normalise the difference.
        index : int
            Suffix of the output file.
        """
        original_img = self.latent_decoder(original_img_latent.to(self.device)).to(
            "cpu"
        )
        perturbed_img = self.latent_decoder(perturbed_latent.to(self.device)).to("cpu")
        diff = (perturbed_img - original_img).detach().cpu() / learning_rate
        ref = torch.zeros_like(diff[0]).to("cpu")
        grad_img = high_contrast_heatmap(ref, diff[0])
        to_pil = ToPILImage()
        to_pil(grad_img[0]).save(f"{_DEBUG_DIR}/grads_{index}.png")

        return None

    def ddpm_inversion(self, x_in, explainer_config, predictor, target_classes, t):
        """Optimise the inverted start latent directly against the predictor.

        ``encode_image`` yields ``x_T`` and the noise maps of ``x_in``; ``x_T`` is
        made a parameter and updated with SGD or Adam (``explainer_config.optimizer``,
        ``learning_rate``, ``momentum``) for ``explainer_config.gradient_steps``
        steps on the cross-entropy of ``predictor(decode_image(z_t)) / temperature``
        towards ``target_classes``. Gradients are sparsified with
        ``self.grad_threshold`` and clipped to ``gradient_clipping``.

        Parameters
        ----------
        x_in : torch.Tensor
            Input image ``(1, 3, H, W)``.
        explainer_config : ExplainerConfig
            Optimiser settings as listed above.
        predictor : nn.Module
            Classifier; it is cast to half precision and set to eval mode.
        target_classes : torch.Tensor
            Target labels.
        t : float
            Unused.

        Returns
        -------
        torch.Tensor
            Final decoded image ``(1, 3, H, W)``.

        Notes
        -----
        Writes many debug PNGs (gradients, intermediate outputs) to ``_DEBUG_DIR``.
        """
        input_size = self.config.data.input_size[1:]
        # Prepare timesteps.
        with torch.no_grad():
            xts, variance_noises = self.encode_image(x_in.to(self.device))
            # for idx, var in enumerate(variance_noises):
            #     self.latent_decoder(var, img=True).save(
            #         f"{_DEBUG_DIR}/variance{idx}.png"
            #     )
            latents = self.latent_encoder(x_in.to(self.device))
            to_pil = ToPILImage()
            vae_scale_factor = self.pipe.vae_scale_factor

            to_pil(
                self.decode_image(xts.to(self.device), variance_noises).squeeze(0)
            ).save(f"{_DEBUG_DIR}/original_img.png")
        x_tt = self.decode_image(xts.to(self.device), variance_noises)
        # x_t = latents.clone()
        z_t = nn.Parameter(torch.clone(xts.detach()), requires_grad=True)
        _log.info("%s %s", "z_t", z_t.shape)
        lr = explainer_config.learning_rate
        momentum = explainer_config.momentum
        optimizer = explainer_config.optimizer
        predictor.half()
        predictor.eval()
        grads = self.clean_multiclass_cond_fn(
            x_t=x_in.to(self.device).to(self.dtype).requires_grad_(True),
            resize=True,
            predictor_img_size=input_size,
            threshold=0.0,
            classifier=predictor,
            y=target_classes,
        )
        to_pil(grads[0]).save(f"{_DEBUG_DIR}/grads_class_org_img.png")
        ref = torch.zeros_like(grads[0]).to("cpu")
        grad_img = high_contrast_heatmap(ref, -grads[0].to("cpu"))
        to_pil(grad_img[0]).save(f"{_DEBUG_DIR}/grads_class_org_img_high_contrast.png")
        encoded_grads = self.latent_encoder(grads.to(self.device))
        self.latent_decoder(encoded_grads, img=True).save(
            f"{_DEBUG_DIR}/grads_class_org_img_latent.png"
        )

        if optimizer == "SGD":
            optimizer = torch.optim.SGD([z_t], lr=lr, momentum=momentum)
        elif optimizer == "Adam":
            optimizer = torch.optim.Adam([z_t], lr=lr)
        loss = nn.CrossEntropyLoss()
        for i in range(explainer_config.gradient_steps):
            # x_original = self.decode_image(xts.to(self.device), variance_noises)

            # mask_t, dil_mask = self.generate_mask(x_in.to(self.device), x_original, 5)
            # boolmask = (dil_mask < 0.1).float()
            # height, width = x_in.shape[2:]
            # to_pil(boolmask.squeeze(0).clamp(0, 1)).save(
            #     f"{_DEBUG_DIR}/boolmask_{i}.png"
            # )
            # # boolmask_latent = torch.nn.functional.interpolate(
            # #     boolmask,
            # #     size=(height // vae_scale_factor, width // vae_scale_factor),
            # # )

            # x_original = x_original * (1 - boolmask) + boolmask * x_in
            x_out = self.decode_image(z_t.to(self.device), variance_noises)
            to_pil(x_out.squeeze(0).clamp(0, 1)).save(f"{_DEBUG_DIR}/x_out_{i}.png")
            logits = predictor(x_out) / explainer_config.temperature
            losses = loss(
                logits, target_classes.to(self.device)
            )  # + self.l1_loss * torch.norm(x_out - x_in.to(self.device), p=1)
            losses.backward()
            grads = z_t.grad
            if self.grad_threshold > 0.0:
                max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
                # max_values = grads.abs().max()
                grads[grads < self.grad_threshold * max_values] = 0.0
                z_t.grad = grads
            _log.info("%s %s", "grads norm", grads.norm(p=float("inf")))
            if explainer_config.gradient_clipping > 0:
                norm = grads.norm(p=float("inf"))
                if norm > explainer_config.gradient_clipping:
                    _log.info("%s", "gradient clipping")
                    rescale_factor = explainer_config.gradient_clipping / norm
                    grads = grads * rescale_factor
                    z_t.grad = grads
            z_prev = z_t.data.clone().detach()
            optimizer.step()
            with torch.no_grad():
                self.visualize_grads(
                    z_prev,
                    z_t.data.clone().detach(),
                    learning_rate=lr,
                    index=i,
                )
                ref = torch.zeros_like(grads[0]).to("cpu")
                grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))
                self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                    f"{_DEBUG_DIR}/grads_class{i}.png"
                )

            optimizer.zero_grad()

        _log.info("%s", "final")
        with torch.no_grad():
            final_x = self.decode_image(z_t.to(self.device), variance_noises)
        to_pil(final_x.squeeze(0).clamp(0, 1)).save(f"{_DEBUG_DIR}/final_img.png")
        # x_original = self.decode_image(x_original.to(self.device), variance_noises)
        return final_x

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        predictor: nn.Module,
        explainer_config: ExplainerConfig,
        predictor_datasets,
        boolmask_in: torch.Tensor,
        attempt_number: int,
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
        """Generate counterfactuals for ``x_in`` towards ``target_classes``.

        Entry point of the ``EditCapableGenerator`` interface. The input is mapped
        from the predictor's value range into the generator's, resized to
        ``config.data.input_size`` and edited with the loop chosen by
        ``config.method``:

        * ``fastdime`` - one ``FastDiME`` pass (parameters are multiplied by
          ``parameters_rewight`` on retries when ``adjust_parameters`` is set);
        * ``fastdime2`` / ``fastdime2+`` - two passes, the second with the change
          mask of the first fixed and a larger step size;
        * ``fastdime_ddpm`` / ``fastdime_ddpm2`` - the same with ``FastDiME_DDPM``;
        * ``ddpm_inversion`` - direct latent optimisation.

        When ``gradient_smoothing == "distilled"`` and
        ``explainer_config.distilled_predictor`` is set, a distilled predictor
        (loaded from ``<model_path>/distilled_predictor/model.cpl`` or trained
        with ``distill_predictor``) supplies the guidance gradients. Afterwards the
        result is resized and mapped back to the predictor domain, the target
        confidence is measured and the final change masks and 2x2 collages are
        written to ``<base_path>/masks``.

        Parameters
        ----------
        x_in : torch.Tensor
            Inputs ``(B, C, H, W)`` in the predictor's value range.
        target_confidence_goal : float
            Unused; the confidence is only reported.
        source_classes, target_classes : torch.Tensor
            Source and target labels of shape ``(B,)``.
        predictor : nn.Module
            Classifier used for the final confidence (and guidance if not
            distilled).
        explainer_config : ExplainerConfig
            Provides ``inpaint``, ``dilation``, ``sampling_time_fraction``,
            ``temperature``, ``learning_rate``, ``momentum``, ``optimizer``,
            ``gradient_clipping``, ``distilled_predictor`` and optionally
            ``visualize_gradients``.
        predictor_datasets : list
            Datasets of the predictor; ``[0].dataset`` supplies the value-range
            projections and input size.
        boolmask_in : torch.Tensor or None
            Fixed mask forwarded to ``FastDiME`` in ``fastdime`` mode.
        attempt_number : int
            Retry counter; also used in the output file names.
        pbar : object, optional
            Unused.
        mode : str, optional
            Unused.
        base_path : str, optional
            Run directory for the distilled predictor and mask images.

        Returns
        -------
        tuple
            ``(counterfactuals, x_in - counterfactuals, target_confidences, x_in,
            None, boolmask)`` where the first four entries are lists over the batch
            and ``boolmask`` is the final binary change mask ``(B, 1, H, W)``.
        """

        classifier_to_generator = lambda x: self.dataset.project_from_pytorch_default(
            predictor_datasets[0].dataset.project_to_pytorch_default(x)
        )
        # breakpoint()clear
        generator_to_classifier = lambda x: predictor_datasets[
            0
        ].dataset.project_from_pytorch_default(
            self.dataset.project_to_pytorch_default(x)
        )
        if (
            not explainer_config.distilled_predictor is None
            and self.gradient_smoothing == "distilled"
        ):
            model_path = "peal_runs/predictor1"
            if "model_path" in explainer_config.distilled_predictor:
                model_path = explainer_config.distilled_predictor["model_path"]

            peal_base = ["/"]
            splited_path = model_path.split("/")
            if splited_path[0] == "<PEAL_BASE>":
                _log.info("%s", "using PEAL_BASE distilled predictor")
                for i in base_path.split("/"):
                    if i == splited_path[1]:

                        break
                    else:
                        peal_base.append(i)
                model_path = os.path.join(*peal_base, *splited_path[1:])
            if os.path.exists(model_path):
                distilled_path = os.path.join(
                    model_path, "distilled_predictor", "model.cpl"
                )
            else:
                distilled_path = os.path.join(
                    base_path, "distilled_predictor", "model.cpl"
                )
            # if "1" in distilled_path:
            #     distilled_path = distilled_path.replace("1", "0")
            _log.info("%s %s", "distilled_path", distilled_path)
            # distilled_path = "$PEAL_RUNS/sce_cfkd/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned100_vit16/0/distilled_predictor/model.cpl"
            if not os.path.exists(distilled_path):
                gradient_predictor = distill_predictor(
                    explainer_config.distilled_predictor,
                    base_path,
                    predictor,
                    predictor_datasets,
                    replace_with_activation="leakysoftplus",
                )

            else:
                # distilled_path = "$PEAL_RUNS/sce_cfkd/celeba_blond_hair_natural/0/distilled_predictor/distilled_predictor/model.cpl"
                gradient_predictor = torch.load(
                    distilled_path, map_location=self.device
                )

        else:
            gradient_predictor = predictor
        inpaint = explainer_config.inpaint
        dilation = explainer_config.dilation
        t = explainer_config.sampling_time_fraction
        predictor_data_size = predictor_datasets[0].dataset.config.input_size[1:]
        to_pil = ToPILImage()
        transformed = torch.zeros_like(source_classes).bool()

        class_grad_kwargs = {
            "y": target_classes,
            "source_class": source_classes,
            "classifier": gradient_predictor,
            "s": explainer_config.temperature,
            "use_logits": self.use_logits,
            "lr": explainer_config.learning_rate,
            "momentum": explainer_config.momentum,
            "predictor_img_size": predictor_data_size,
            "optimizer": explainer_config.optimizer,
            "generator_to_classifier": generator_to_classifier,
        }
        dist_grad_kargs = {
            "l1_loss": self.l1_loss,
            "l2_loss": self.l2_loss,
            "l_perc": self.loss,
            "lr": explainer_config.learning_rate,
            "momentum": explainer_config.momentum,
        }
        x_in = classifier_to_generator(x_in)
        x_in = transforms.Resize(self.config.data.input_size[1:])(x_in)
        self.base_path = base_path
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
                scale_grads=False,
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
            # to_pil(boolmask.squeeze(0).clamp(0, 1)).save(
            #     f"{_DEBUG_DIR}/boolmask_{mode}.png"
            # )
            # to_pil(x_counterfactuals.squeeze(0).clamp(0, 1)).save(
            #     f"{_DEBUG_DIR}/x_counterfactuals_{mode}.png"
            # )
            class_grad_kwargs["lr"] = class_grad_kwargs["lr"] * 2
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
            # to_pil(boolmask.squeeze(0).clamp(0, 1)).save(
            #     f"{_DEBUG_DIR}/boolmask_{mode}.png"
            # )
            # to_pil(x_counterfactuals.squeeze(0).clamp(0, 1)).save(
            #     f"{_DEBUG_DIR}/x_counterfactuals_{mode}.png"
            # )
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
            _log.info("%s", f"attempt {attempt_number}")
            if attempt_number > 0 and self.adjust_parameters:
                class_grad_kwargs["lr"] = (
                    class_grad_kwargs["lr"] * self.parameters_rewight["lr"]
                )
                dist_grad_kargs["l1_loss"] = (
                    dist_grad_kargs["l1_loss"] * self.parameters_rewight["l1"]
                )
                dist_grad_kargs["l2_loss"] = (
                    dist_grad_kargs["l2_loss"] * self.parameters_rewight["l2"]
                )
                class_grad_kwargs["s"] = (
                    class_grad_kwargs["s"] * self.parameters_rewight["temperature"]
                )
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
                boolmask_in=boolmask_in if boolmask_in is not None else None,
                base_path=base_path,
                mode=f"attempt_{attempt_number}",
                visualize_gradients=(
                    explainer_config.visualize_gradients
                    if hasattr(explainer_config, "visualize_gradients")
                    else False
                ),
            )
            if attempt_number > 0:
                self._sample_id += 1

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
            # to_pil(boolmask.squeeze(0).clamp(0, 1)).save(
            #     f"{_DEBUG_DIR}/boolmask.png"
            # )
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
        elif self.method.lower() == "ddpm_inversion":
            x_counterfactuals = self.ddpm_inversion(
                x_in=x_in,
                explainer_config=explainer_config,
                predictor=predictor,
                target_classes=target_classes,
                t=t,
            )
        to_pil = ToPILImage()
        # final counterfactual viz
        x_counterfactuals = transforms.Resize(predictor_data_size)(x_counterfactuals)
        x_counterfactuals = x_counterfactuals.clamp(0, 1)
        x_counterfactuals = generator_to_classifier(x_counterfactuals)
        x_in = transforms.Resize(predictor_data_size)(x_in)
        x_in = generator_to_classifier(x_in)
        preds = (
            torch.nn.Softmax(dim=-1)(predictor(x_counterfactuals.to(self.device)))
            .detach()
            .cpu()
        )

        y_target_end_confidence = torch.zeros([x_in.shape[0]])
        for i in range(x_in.shape[0]):
            y_target_end_confidence[i] = preds[i, target_classes[i]]

        mask, dil_mask = generate_smooth_mask(
            x_in.to("cuda"), x_counterfactuals, explainer_config.dilation
        )

        boolmask = (dil_mask < explainer_config.inpaint).float()

        mask_path = os.path.join(base_path, "masks")
        os.makedirs(mask_path, exist_ok=True)
        for i, mask in enumerate(boolmask):
            to_pil(mask).save(os.path.join(mask_path, f"{attempt_number}_{i}.png"))
        if boolmask_in is not None:
            img = x_in * (1 - boolmask_in.to("cpu")) + x_counterfactuals.to(
                "cpu"
            ) * boolmask_in.to("cpu")
        else:
            img = x_in * (1 - boolmask.to("cpu")) + x_counterfactuals.to(
                "cpu"
            ) * boolmask.to("cpu")

        from PIL import Image

        for i in range(x_in.shape[0]):
            # Convert tensors to PIL images
            original = to_pil(x_in[i].clamp(0, 1))
            counterfactual = to_pil(x_counterfactuals[i].clamp(0, 1))
            masked = to_pil(img[i].clamp(0, 1))
            mask_img = to_pil(boolmask[i])

            # Get dimensions
            w, h = original.size

            # Create collage (2x2 grid)
            collage = Image.new("RGB", (w * 2, h * 2))
            collage.paste(original, (0, 0))
            collage.paste(counterfactual, (w, 0))
            collage.paste(mask_img.convert("RGB"), (0, h))
            collage.paste(masked, (w, h))

            # Save collage
            collage.save(os.path.join(mask_path, f"collage_{attempt_number}_{i}.png"))

        _log.info("%s", "edit done")
        return (
            list(x_counterfactuals.detach().cpu()),
            list(x_in - x_counterfactuals.detach().cpu()),
            list(y_target_end_confidence),
            list(x_in),
            None,
            boolmask,
        )

    def log_prob_z(self, z):
        """Not available for this generator; returns ``None``."""
        return None

    def calc_z_shapes(self):
        """Not available for this generator; returns ``None``."""
        return None

    def train_model(
        self,
    ):
        """LoRA-finetune the SD3 pipeline on the configured training dataset.

        Writes ``config.yaml`` into ``config.base_path`` and calls ``lora_finetune``
        with the config fields, the training dataset and ``self.pipeline`` as a
        ``SimpleNamespace`` (resuming from the latest checkpoint).

        Notes
        -----
        ``self.pipeline`` is never assigned in ``__init__`` (the pipeline lives in
        ``self.pipe``), so this method raises ``AttributeError`` as written.
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
