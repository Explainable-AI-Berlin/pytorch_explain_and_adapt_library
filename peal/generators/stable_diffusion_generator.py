"""Stable Diffusion v1.x as a PEAL counterfactual generator.

Wraps a diffusers ``StableDiffusionPipeline`` in the ``EditCapableGenerator``
protocol. The default ``edit`` path is a re-implementation of FastDiME
(classifier-guided latent denoising with self-optimised inpainting masks) on
the SD U-Net; the alternative path delegates to the vendored TIME code
(textual inversion of class tokens + edit-friendly DDPM inversion, see
``peal.dependencies.time`` and ``peal.editors.ddpm_inversion``).
``train_model`` LoRA-finetunes the U-Net on the generator dataset. Debug
images go to ``$PEAL_RUNS/debug``.
"""

import os
import types
import copy
from pathlib import Path

import torch
import torchvision
from diffusers import StableDiffusionPipeline
from torch.nn import functional as F
from torch import nn
from peal.training.trainers import distill_predictor
from peal.global_utils import load_yaml_config, generate_smooth_mask
from torchvision.transforms import ToTensor, transforms, ToPILImage

from peal.dependencies.ddpm_inversion.ddm_inversion.inversion_utils import (
    inversion_forward_process,
    inversion_reverse_process,
)
from peal.editors.ddpm_inversion import DDPMInversionConfig
from peal.data.dataset_factory import get_datasets
from peal.dependencies.ddpm_inversion.ddpm_inversion import DDPMInversion
from peal.dependencies.lora.train_text_to_image_lora import lora_finetune
from peal.dependencies.time.core.utils import load_tokens_and_embeddings
from peal.generators.interfaces import EditCapableGenerator
from peal.global_utils import save_yaml_config
from peal.dependencies.time.generate_ce import (
    generate_time_counterfactuals,
)
from peal.dependencies.time.get_predictions import get_predictions
from peal.dependencies.time.training import textual_inversion_training

from typing import Union

from peal.generators.interfaces import GeneratorConfig
from peal.data.interfaces import DataConfig
from peal.architectures.interfaces import TaskConfig
from peal.log import get_logger

_log = get_logger(__name__)


# Debug image dumps, under $PEAL_RUNS so the module is portable.
_DEBUG_DIR = os.path.join(os.environ.get("PEAL_RUNS", "peal_runs"), "debug")


class StableDiffusionConfig(GeneratorConfig):
    """Config of :class:`StableDiffusion`.

    Most fields mirror the argument namespace of diffusers'
    ``train_text_to_image_lora.py`` and are passed verbatim to
    ``lora_finetune`` by :meth:`StableDiffusion.train_model` (``resolution``,
    ``train_batch_size``, ``learning_rate``, ``lr_scheduler``,
    ``mixed_precision``, ``checkpointing_steps``, ``rank`` ...). The fields
    below are the ones the generator itself reads.

    Parameters
    ----------
    generator_type : str
        Registry name, ``"StableDiffusion"``.
    base_path : str
        Run directory (``$PEAL_RUNS/stable_diffusion``); ``config.yaml`` and
        the LoRA weights are written here.
    data : DataConfig
        Dataset used for finetuning and for the generator-space projections.
    sd_model : str
        Hugging Face id of the pipeline (``CompVis/stable-diffusion-v1-4``).
    steps_number : int
        Number of denoising steps used by :meth:`StableDiffusion.FastDiME`.
    guided_iterations : int
        Number of leading steps that receive classifier/distance guidance.
    classifier_scale, use_logits, grad_threshold : float, bool, float
        Guidance strength, logits-vs-log-softmax objective and relative
        gradient sparsification threshold of the classifier term.
    l1_loss, l2_loss : float
        Weights of the latent L1/L2 distance terms.
    self_optimized_masking, use_gussian_blur_masking, warmup_steps : bool, bool, int
        FastDiME mask handling: re-inpaint unchanged regions after
        ``warmup_steps``, and build the mask with the blurred difference from
        ``peal.global_utils.generate_smooth_mask`` or with
        :meth:`StableDiffusion.generate_mask`.
    method : str
        ``"FastDime"`` (case-insensitive) selects the FastDiME path in
        ``edit``; anything else selects the TIME path.
    prompt, guidance_scale : str, float
        Text prompt and classifier-free guidance scale for FastDiME
        (guidance is only applied when ``guidance_scale > 1``).
    task_config : TaskConfig or None
        Overrides the training dataset's task config when given.
    """

    generator_type: str = "StableDiffusion"
    """
    The type of generator that shall be used.
    """
    base_path: str = "$PEAL_RUNS/stable_diffusion"
    data: DataConfig = DataConfig()
    """
    The config of the data.
    """
    sd_model: str = "CompVis/stable-diffusion-v1-4"
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
    steps_number: int = 15
    strength: float = 1.0
    guided_iterations: int = 9999999
    classifier_scale: float = 5.0
    l1_loss: float = 0.0
    l2_loss: float = 0.0
    self_optimized_masking: bool = True
    use_logits: bool = True
    method: str = "FastDime"
    grad_threshold: float = 0.0
    task_config: Union[TaskConfig, type(None)] = None
    prompt: str = " "
    offload_cpu: bool = False
    guidance_scale: float = 0.0
    joint_attention_kwargs: Union[dict, type(None)] = None
    warmup_steps: int = 20
    use_gussian_blur_masking: bool = True


class StableDiffusion(EditCapableGenerator):
    """Stable Diffusion pipeline exposed as an editable PEAL generator.

    Parameters
    ----------
    config : str or StableDiffusionConfig
        Config object or YAML path (resolved with ``load_yaml_config``).
    classifier_dataset : Dataset, optional
        Dataset of the classifier under explanation; deep-copied and used by
        :meth:`initialize` to write the predictions CSV the TIME path needs.
    predictor_dataset : Dataset, optional
        Unused.
    model_dir : str, optional
        Overrides ``config.base_path`` as the run directory.
    device : str
        Device the pipeline is moved to.

    Attributes
    ----------
    pipeline : StableDiffusionPipeline
        The diffusers pipeline (safety checker disabled).
    train_dataset, val_dataset, dataset : Dataset
        Generator-space datasets from ``config.data``; ``dataset`` is the
        validation split and supplies the ``project_*_pytorch_default``
        conversions used in ``edit``.
    editor : DDPMInversion or None
        Set by :meth:`initialize` for the TIME path.
    loss : callable or None
        Optional perceptual loss for :meth:`dist_cond_fn`; ``None`` here.
    """

    def __init__(
        self,
        config,
        classifier_dataset=None,
        predictor_dataset=None,
        model_dir=None,
        device="cuda",
    ):
        """Load the config, the datasets and the pretrained pipeline."""
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

        self.classifier_scale = self.config.classifier_scale
        self.use_logits = self.config.use_logits
        self.l1_loss = self.config.l1_loss
        self.l2_loss = self.config.l2_loss
        self.method = self.config.method
        self.grad_threshold = self.config.grad_threshold
        self.steps_number = self.config.steps_number
        self.pipeline.to(device)
        self.loss = None
        self.device = device
        self.train_dataset, self.val_dataset, _ = get_datasets(self.config.data)
        self.dataset = self.val_dataset
        self.guided_iterations = self.config.guided_iterations
        self.prompt = self.config.prompt
        self.guidance_scale = self.config.guidance_scale
        self.self_optamized_masking = self.config.self_optimized_masking
        self.use_gussian_blur_masking = self.config.use_gussian_blur_masking
        self.warmup_steps = self.config.warmup_steps
        # self.pipeline.run_safety_checker = lambda image, device, dtype: image, False
        self.pipeline.safety_checker = None

    def sample_x(self, batch_size=1):
        """Sample ``batch_size`` images with the empty prompt.

        Parameters
        ----------
        batch_size : int
            Number of images.

        Returns
        -------
        torch.Tensor
            ``[batch_size, 3, H, W]`` images in ``[0, 1]`` (``ToTensor`` of the
            pipeline's PIL outputs).
        """
        images = self.pipeline(batch_size * [""]).images
        images_torch = torch.stack([ToTensor()(image) for image in images])
        return images_torch

    def encode(self, x, t=1.0):
        """Edit-friendly DDPM inversion of ``x`` into ``(wt, zs, wts)``.

        Resizes ``x`` to 512x512, VAE-encodes it (scaled by 0.18215) and runs
        ``inversion_forward_process`` with the empty prompt.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, 3, H, W]`` images.
        t : float
            Unused.

        Returns
        -------
        tuple
            ``(wt, zs, wts)``: terminal latent, per-step noise maps and the
            latent trajectory.

        Notes
        -----
        This method reads ``self.pipe`` and ``self.config.eta`` /
        ``cfg_scale_src`` / ``num_diffusion_steps``, none of which the
        constructor or :class:`StableDiffusionConfig` define; it is not
        reachable from the current ``edit`` path.
        """

        # TODO why are gradients in ACE scaled???
        # t = torch.tensor([self.steps - 1] * x.size(0), device=x.device)
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
        """Reverse an :meth:`encode` result back into an image with the target scale.

        Parameters
        ----------
        z : tuple
            ``(wt, zs, wts)`` from :meth:`encode`.
        t : float
            Unused.

        Returns
        -------
        torch.Tensor
            VAE-decoded image batch in the pipeline's ``[-1, 1]`` range.

        Notes
        -----
        Same caveat as :meth:`encode`: depends on ``self.pipe`` and DDPM
        inversion config fields that are not set up by this class, and calls
        ``z.shape`` on a tuple.
        """
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
            # classifier=classifier_loss,
        )

        # vae decode image
        x = self.pipe.vae.decode(1 / 0.18215 * w0).sample

        return x

    def train_model(
        self,
    ):
        """LoRA-finetune the U-Net on ``train_dataset`` via ``lora_finetune``.

        Writes ``<base_path>/config.yaml``, builds a ``SimpleNamespace`` from
        the config fields plus ``train_dataset``/``pipeline`` and
        ``resume_from_checkpoint="latest"``, and replaces ``self.pipeline``
        with the finetuned one.
        """
        # write the yaml config on disk
        if not os.path.exists(self.config.base_path):
            Path(self.config.base_path).mkdir(parents=True, exist_ok=True)

        save_yaml_config(
            self.config, os.path.join(self.config.base_path, "config.yaml")
        )
        save_yaml_config(
            self.config, os.path.join(self.config.base_path, "config.yaml")
        )
        finetune_args = types.SimpleNamespace(**self.config.__dict__)
        finetune_args.train_dataset = self.train_dataset
        finetune_args.pipeline = self.pipeline
        finetune_args.resume_from_checkpoint = "latest"
        finetune_args.resume_from_checkpoint = "latest"

        _log.info("%s", "Start LORA finetuning")
        _log.info("%s", "Start LORA finetuning")
        _log.info("%s", "Start LORA finetuning")
        self.pipeline = lora_finetune(finetune_args)
        _log.info("%s", "Finished LORA finetuning")
        _log.info("%s", "Finished LORA finetuning")
        _log.info("%s", "Finished LORA finetuning")

    def initialize(self, classifier, base_path, explainer_config):
        """Prepare the TIME editing path for a given classifier.

        Parameters
        ----------
        classifier : nn.Module
            Predictor whose train-split predictions are written to
            ``<base_path>/explainer/predictions.csv`` (once).
        base_path : str
            Explainer run directory.
        explainer_config
            Explainer config with ``use_lora``, ``max_samples``,
            ``learn_dataset_embedding``, ``custom_tokens_context``,
            ``base_prompt``, ``prompt_connector``, ``class_custom_token``,
            ``train_batch_size``, ``editing_type`` and
            ``guidance_scale_invertion`` / ``guidance_scale_denoising``.

        Notes
        -----
        Side effects, all under ``<base_path>/explainer``: optional LoRA
        finetuning, a ``generator_dataset`` built from the predictions CSV
        with ``y_selection=["prediction"]``, textual-inversion training of a
        context embedding (``context/context_embedding``) and one class token
        per class (``class<i>/class_token<i>``), TensorBoard logs, and when
        ``editing_type == "ddpm_inversion"`` a ``DDPMInversion`` editor in
        ``self.editor`` with the learned tokens loaded into its pipeline.
        """
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

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(base_path, "explainer", "logs"))
        generator_dataset_config = copy.deepcopy(self.config.data)
        generator_dataset_config.split = [0.9, 1.0]
        self.generator_dataset, self.generator_dataset_val, _ = get_datasets(
            config=generator_dataset_config, data_dir=class_predictions_path
        )
        if self.generator_dataset.task_config is None:
            self.generator_dataset.task_config = TaskConfig()
            self.generator_dataset.task_config.y_selection = ["prediction"]
            self.generator_dataset.task_config.y_selection = ["prediction"]

        else:
            self.generator_dataset.task_config.y_selection = ["prediction"]
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
                    **explainer_config.__dict__,
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
                    **explainer_config.__dict__,
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
                embedding_files.append(
                    os.path.join(
                        base_path, "explainer", "class0", "class_token" + str(class_idx)
                    )
                )

            if explainer_config.learn_dataset_embedding:
                embedding_files = [
                    os.path.join(base_path, "explainer", "context", "context_embedding")
                ] + embedding_files
                embedding_files = [
                    os.path.join(base_path, "explainer", "context", "context_embedding")
                ] + embedding_files

            load_tokens_and_embeddings(sd_model=self.editor.pipe, files=embedding_files)

        else:
            self.editor = None

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
        """Gradient of the (scaled) classifier objective with respect to ``x_t``.

        Parameters
        ----------
        x_t : torch.Tensor
            Current clean estimate: an image when ``resize`` is true, a VAE
            latent otherwise (then decoded with :meth:`vae_latent_decoder`).
        y : torch.Tensor
            ``[B]`` target classes.
        resize : bool
            Whether ``x_t`` is already in pixel space.
        classifier : nn.Module
            Predictor evaluated after ``generator_to_classifier`` (from
            ``kwargs``) and a resize to ``predictor_img_size``.
        s : float
            Temperature multiplying the objective.
        use_logits : bool
            Use raw logits; otherwise log-softmax.
        predictor_img_size : tuple
            Spatial size expected by the classifier.
        lr, momentum, optimizer : optional
            Unused.
        threshold : float, optional
            When set, gradient entries below ``threshold`` times the per-image
            max are zeroed.

        Returns
        -------
        torch.Tensor
            Gradient of ``-s * score[y]`` summed over the batch, same shape as
            ``x_t``.
        """
        classifier.eval()
        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        x_in = [x_in]
        generator_to_classifier = kwargs.get("generator_to_classifier", None)
        if resize:
            x_img = x_in[0]
        else:
            x_img = self.vae_latent_decoder(x_in[0])
        x_img = generator_to_classifier(x_img) if generator_to_classifier else x_img
        x_img = transforms.Resize(predictor_img_size)(x_img)

        classifier.eval()
        selected = classifier(x_img)
        if not use_logits:
            selected = F.log_softmax(selected, dim=1)
        selected = -selected[range(len(y)), y]
        selected = selected * s
        grads = torch.autograd.grad(selected.sum(), x_in)[0]
        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            # max_values = grads.abs().max()
            grads[grads < threshold * max_values] = 0.0
        return grads

    def vae_latent_encoder(self, x):
        """Map images in ``[0, 1]`` to scaled SD latents (mode of the VAE posterior).

        Parameters
        ----------
        x : torch.Tensor
            ``[B, 3, H, W]`` images in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            ``[B, 4, H/8, W/8]`` latents multiplied by ``0.18215``.
        """

        x = 2.0 * x - 1.0
        VAE_SCALE = 0.18215

        x = (self.pipeline.vae.encode(x).latent_dist.mode() * VAE_SCALE).float()

        return x

    def vae_latent_decoder(self, z):
        """Decode scaled SD latents back to images clamped to ``[0, 1]``.

        Parameters
        ----------
        z : torch.Tensor
            ``[B, 4, h, w]`` latents as produced by :meth:`vae_latent_encoder`.

        Returns
        -------
        torch.Tensor
            ``[B, 3, 8h, 8w]`` images in ``[0, 1]``.
        """
        VAE_SCALE = 0.18215
        z = self.pipeline.vae.decode(z * (1 / VAE_SCALE)).sample

        z = (z / 2 + 0.5).clamp(0, 1)

        return z

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

        # z_img = self.vae_latent_decoder(z_in)
        # x_img = self.vae_latent_decoder(x_in)
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
        """
        FastDiME for Stable Diffusion v1.4 (uses U-Net instead of transformer)

        Encodes ``img`` to latents, noises them to a fraction ``t`` of the
        1000-step schedule, and denoises over ``config.steps_number`` steps.
        During the first ``guided_iterations`` steps the latent is pushed by
        the classifier gradient (:meth:`clean_multiclass_cond_fn`) and the
        distance gradient (:meth:`dist_cond_fn`), clipped to
        ``explainer_config.gradient_clipping``. After ``warmup_steps`` the
        regions where the decoded estimate hardly differs from ``img``
        (dilated difference below ``explainer_config.inpaint``) are reset to
        the re-noised original latents (self-optimised masking); a fixed
        ``boolmask_in`` restricts edits the same way.

        Parameters
        ----------
        img : torch.Tensor
            ``[B, 3, H, W]`` images in ``[0, 1]`` (generator space).
        inpaint, dilation : float
            Unused here; the values are read from ``explainer_config``.
        t : float
            Fraction of the noise schedule to start from.
        guided_iterations : int
            Number of guided denoising steps.
        class_grad_kwargs : dict
            Keyword arguments for :meth:`clean_multiclass_cond_fn`; ``"lr"``
            scales the latent update.
        dist_grad_kargs : dict
            Keyword arguments for :meth:`dist_cond_fn`.
        explainer_config
            Provides ``gradient_clipping``, ``dilation`` and ``inpaint``.
        scale_grads : bool
            Divide the classifier gradient by the normalised timestep.
        boolmask_in : torch.Tensor, optional
            ``[B, 1, H, W]`` mask of editable pixels (1 = editable).

        Returns
        -------
        tuple
            ``(final_image, x_t_steps, z_t_steps)``: the decoded result in
            ``[0, 1]`` and the CPU copies of the clean estimate and the noisy
            latent before every step.
        """
        boolmask = boolmask_in
        to_pil = ToPILImage()
        class_grad_fn = self.clean_multiclass_cond_fn
        dist_fn = self.dist_cond_fn
        self_optimized_masking = self.self_optamized_masking

        # Encode image into latent space with the VAE
        height, width = self.config.data.input_size[1:]
        _log.info("%s", f"Encoding image of size: {height}x{width}")

        with torch.no_grad():
            latents = self.vae_latent_encoder(img.to(self.device))

        # Initialize latents
        x_t = latents.clone()
        batch_size = x_t.shape[0]
        vae_scale_factor = (
            self.pipeline.vae_scale_factor
        )  # SD v1.4 uses 8x downsampling

        # Prepare noise scheduler timesteps
        num_inference_steps = 1000
        self.pipeline.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipeline.scheduler.timesteps  # Tensor of scheduler timesteps
        max_idx = int(t * len(timesteps))
        if max_idx < 1:
            max_idx = 1
        prefix = timesteps.flip((0,))[0:max_idx]  # actual scheduler timesteps
        idxs = (
            torch.linspace(0, len(prefix) - 1, steps=self.steps_number)
            .long()
            .flip(dims=(0,))
        )  # reversed
        timesteps = prefix[idxs].to(self.device)  # device-match

        # Add noise for img2img strength
        # init_timestep = min(int(self.steps_number * t), self.steps_number)
        # t_start = max(self.steps_number - init_timestep, 0)
        # timesteps = timesteps[t_start:]

        # Add noise to latents
        noise = torch.randn_like(latents)
        z_t = self.pipeline.scheduler.add_noise(latents, noise, timesteps[0:1])

        # Prepare text embeddings
        do_classifier_free_guidance = self.guidance_scale > 1.0
        prompt = [self.prompt] * img.shape[0]
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            lora_scale=None,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        x_t_steps = []
        z_t_steps = []

        # Main denoising loop
        for i, timestep in enumerate(timesteps):
            x_t_steps.append(x_t.detach().cpu().clone())
            z_t_steps.append(z_t.detach().cpu().clone())

            # Expand latents for classifier free guidance
            latent_model_input = (
                torch.cat([z_t] * 2) if self.guidance_scale > 1.0 else z_t
            )
            latent_model_input = self.pipeline.scheduler.scale_model_input(
                latent_model_input, timestep
            )

            # Predict noise residual using U-Net
            with torch.no_grad():
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                ).sample

            # Perform guidance
            if self.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                x_t = x_t.chunk(2)[0]
                z_t = z_t.chunk(2)[0]

            # Compute predicted x0
            alpha_prod_t = self.pipeline.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t

            # Compute gradients for guidance
            grads = 0
            if class_grad_fn is not None and i < guided_iterations:
                class_grad = class_grad_fn(
                    x_t=x_t,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                )
                if scale_grads:
                    class_grad = class_grad / (timestep.float() / 1000.0 + 1e-5)
                grads += class_grad

            if dist_fn is not None and i < guided_iterations:
                dist_grad = dist_fn(
                    x_tau=latents,
                    z_t=z_t.clone(),
                    x_t=x_t.clone(),
                    alpha_t=alpha_prod_t,
                    scale_grads=scale_grads,
                    **dist_grad_kargs,
                )
                grads = grads + dist_grad

            with torch.no_grad():
                to_pil = ToPILImage()
                z_t = self.pipeline.scheduler.step(
                    noise_pred, timestep, z_t
                ).prev_sample

                x_t = (z_t - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5

            # Apply gradient clipping
            if isinstance(grads, torch.Tensor):
                norm = grads.norm(p=float("inf"))
                if norm > explainer_config.gradient_clipping:
                    rescale_factor = explainer_config.gradient_clipping / norm
                    grads = grads * rescale_factor

                # Update with gradients
                if boolmask_in is None:
                    boolmask_latent = torch.ones_like(z_t)
                else:
                    boolmask_latent = torch.nn.functional.interpolate(
                        (1 - boolmask_in),
                        size=(height // vae_scale_factor, width // vae_scale_factor),
                    )

                z_t = z_t - boolmask_latent * class_grad_kwargs["lr"] * grads
            else:
                z_t = z_t
            x_0_denoised = self.vae_latent_decoder(x_t.detach())
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

            # Apply self-optimized masking
            if self_optimized_masking and i > self.warmup_steps and boolmask_in is None:
                with torch.no_grad():

                    boolmask_latent = torch.nn.functional.interpolate(
                        boolmask,
                        size=(height // vae_scale_factor, width // vae_scale_factor),
                        mode="nearest",
                    )

                    # Mask the latents
                    x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents

                # Add noise to masked regions
                if i < len(timesteps) - 1:
                    noise = torch.randn_like(z_t)
                    z_t_noisy = self.pipeline.scheduler.add_noise(
                        latents, noise, timestep
                    )
                    z_t = z_t * (1 - boolmask_latent) + boolmask_latent * z_t_noisy

            # Apply fixed mask if provided
            if boolmask_in is not None:
                boolmask_latent = torch.nn.functional.interpolate(
                    boolmask_in,
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                    mode="nearest",
                )
                noise = torch.randn_like(z_t)
                z_t_noisy = self.pipeline.scheduler.add_noise(latents, noise, timestep)
                x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents
                z_t = z_t * (1 - boolmask_latent) + boolmask_latent * z_t_noisy

        # Final decoding
        with torch.no_grad():
            if boolmask_latent is not None:
                boolmask_latent = torch.nn.functional.interpolate(
                    boolmask,
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                    mode="nearest",
                )
                z_t = z_t * (1 - boolmask_latent) + boolmask_latent * latents

            final_image = self.vae_latent_decoder(z_t)

        return final_image, x_t_steps, z_t_steps

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        predictor: nn.Module,
        explainer_config,
        predictor_datasets,
        boolmask_in: torch.Tensor,
        attempt_number: int,
        pbar=None,
        mode="",
        base_path="",
    ):
        """Generate counterfactuals for ``x_in`` (``EditCapableGenerator`` protocol).

        With ``config.method == "FastDime"`` the inputs are projected from
        the predictor's dataset space into the generator's space, resized to
        ``config.data.input_size`` and edited with :meth:`FastDiME`, using
        either ``predictor`` or a distilled predictor
        (``explainer_config.distilled_predictor``; loaded from
        ``<model_path>/distilled_predictor/model.cpl`` or trained with
        ``distill_predictor``) for the guidance gradient. Otherwise the TIME
        path calls ``generate_time_counterfactuals``.

        Parameters
        ----------
        x_in : torch.Tensor
            ``[B, 3, H, W]`` inputs in the predictor's dataset space.
        target_confidence_goal : float
            Unused.
        source_classes, target_classes : torch.Tensor
            ``[B]`` source and target labels.
        predictor : nn.Module
            Classifier under explanation.
        explainer_config
            Provides ``distilled_predictor``, ``inpaint``, ``dilation``,
            ``sampling_time_fraction``, ``temperature``, ``learning_rate``,
            ``momentum``, ``optimizer``, ``gradient_clipping`` (FastDiME) or
            the TIME arguments.
        predictor_datasets : list
            ``predictor_datasets[0].dataset`` supplies the space conversions
            and ``config.input_size``.
        boolmask_in : torch.Tensor or None
            Optional mask of editable pixels.
        attempt_number : int
            Used in the filenames of the saved masks and collages.
        pbar, mode : optional
            Unused.
        base_path : str
            Explainer directory; masks are written to ``<base_path>/masks``.

        Returns
        -------
        tuple
            FastDiME path: ``(counterfactuals, x_in - counterfactuals,
            target confidences, inputs, None, boolmask)`` with the first four
            as lists over the batch (predictor space, ``[0, 1]``) and
            ``boolmask`` the ``[B, 1, H, W]`` change mask. TIME path: the
            first four entries only.

        Notes
        -----
        Side effects (FastDiME): ``<base_path>/masks/<attempt>_<i>.png`` and
        ``collage_<attempt>_<i>.png`` (original, counterfactual, mask,
        masked blend). The TIME branch references ``dataset`` and
        ``classifier``, which are not defined in this method, so it fails
        with ``NameError`` as written.
        """
        # if self.generator_dataset is None:
        #     self.initialize(predictor, base_path, explainer_config)

        # classifier_to_generator = (
        #     lambda x: self.generator_dataset.project_from_pytorch_default(
        #         self.classifier_dataset.project_to_pytorch_default(x)
        #     )
        # )
        # generator_to_classifier = (
        #     lambda x: self.classifier_dataset.project_from_pytorch_default(
        #         self.generator_dataset.project_to_pytorch_default(x)
        #     )
        # )
        # dataset = [
        #     (
        #         torch.zeros([len(x_in)], dtype=torch.long),
        #         classifier_to_generator(x_in),
        #         [source_classes, target_classes],
        #     )
        # ]

        classifier_to_generator = lambda x: self.dataset.project_from_pytorch_default(
            predictor_datasets[0].dataset.project_to_pytorch_default(x)
        )
        # breakpoint()clear
        generator_to_classifier = lambda x: predictor_datasets[
            0
        ].dataset.project_from_pytorch_default(
            self.dataset.project_to_pytorch_default(x)
        )

        if self.method.lower() == "fastdime":
            if not explainer_config.distilled_predictor is None:
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
            )
            # to_pil = ToPILImage()
            # final counterfactual viz
            x_counterfactuals = transforms.Resize(predictor_data_size)(
                x_counterfactuals
            )
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
                collage.save(
                    os.path.join(mask_path, f"collage_{attempt_number}_{i}.png")
                )

            _log.info("%s", "edit done")
            return (
                list(x_counterfactuals.detach().cpu()),
                list(x_in - x_counterfactuals.detach().cpu()),
                list(y_target_end_confidence),
                list(x_in),
                None,
                boolmask,
            )

        else:

            _log.info("%s", "[x_in.min(), x_in.max()]")
            _log.info("%s", [x_in.min(), x_in.max()])
            _log.info("%s", [x_in.min(), x_in.max()])
            _log.info("%s", [x_in.min(), x_in.max()])
            # The vendored TIME loop accepts a list of ready-made batches in
            # place of a dataset name: (indices, images, [labels, targets]).
            dataset = [
                (
                    torch.zeros([len(x_in)], dtype=torch.long),
                    x_in,
                    [source_classes, target_classes],
                )
            ]
            classifier = predictor
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
                **explainer_config.__dict__,
            )
            x_counterfactuals = generate_time_counterfactuals(ce_generation_args)

            x_counterfactuals = generator_to_classifier(
                torch.cat(x_counterfactuals, dim=0)
            )
            _log.info("%s", "[x_counterfactuals.min(), x_counterfactuals.max()]")
            _log.info("%s", [x_counterfactuals.min(), x_counterfactuals.max()])
            _log.info("%s", [x_counterfactuals.min(), x_counterfactuals.max()])
            _log.info("%s", [x_counterfactuals.min(), x_counterfactuals.max()])
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
