"""DiffAE-based diffusion autoencoder, PEAL's main counterfactual generator.

Wraps the DiffAE ``LitModel`` (semantic encoder plus a conditional DDIM/DDPM
UNet) as an invertible, edit-capable PEAL generator: images are encoded to a
semantic code ``z_sem`` and a diffusion state, edited in ``z_sem``, and
decoded back. The semantic encoder may be the jointly trained one or a
pretrained backbone (DINOv2/v3, open_clip, OpenAI CLIP, UNI), and a sparse
dictionary fitted on that space turns edit directions into named concepts --
the latent machinery the DiDAE adaptor is built on.
"""

import json
import re
import math
import os
import shutil
import copy
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset

from pathlib import Path

import torchvision
from torch import nn
import collections

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
from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary
from peal.sparse_dictionaries.utils import plot_component_ground_truth_correlations
from peal.training.trainers import distill_predictor, load_first_loadable
from peal.log import get_logger

_log = get_logger(__name__)


def _safe_name(name):
    """Filesystem-safe version of a direction/pair name: only letters, digits,
    '.', '-' and '_' (e.g. "OFF homme [GT: Male, F1=0.98] (SAE #4898)  ->  ON x"
    -> "OFF_homme_GT_Male_F1_0.98_SAE_4898_TO_ON_x"; a ('Male', 0.978) tuple
    -> "Male_0.98"). Spaces, brackets, quotes and arrows in folder names made
    the run directories painful to navigate (2026-09-15)."""
    if (
        isinstance(name, (tuple, list))
        and len(name) == 2
        and isinstance(name[1], (int, float))
    ):
        name = f"{name[0]}_{float(name[1]):.2f}"
    name = str(name).replace("->", " TO ")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "unnamed"


def _zs_map(zs, fn):
    """Apply fn to every per-step noise map. The DiffAE's edit-friendly inversion
    returns zs as {t: [B, ...]}, the RAE's as a tensor [steps, B, ...]; both are
    indexed/tiled along their batch axis here so the shared edit and sweep code
    works for either generator (2026-09-15, found by the RAE smoke test)."""
    if zs is None:
        return None
    if isinstance(zs, dict):
        return {key: fn(value) for key, value in zs.items()}
    return torch.stack([fn(zs[i]) for i in range(zs.shape[0])], dim=0)


def _zs_cat(parts):
    """Concatenate per-chunk noise maps along the batch axis (dict or tensor)."""
    if isinstance(parts[0], dict):
        return {
            key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]
        }
    return torch.cat(parts, dim=1)


class DiffusionAutoencoderConfig(GeneratorConfig):
    """
    This class defines the config of a DDPM / Diffusion Autoencoder.
    """

    generator_type: str = "DiffusionAutoencoder"
    """
    The type of generator that shall be used.
    """
    base_path: str = "$PEAL_RUNS/diffusion_autoencoder"
    data: DataConfig = DataConfig()
    """
    The config of the data.
    """
    task_config: Union[TaskConfig, None] = None
    """
    The task config for the diffusion autoencoder.
    """
    checkpoint_path: str = "peal_runs/diffusion_autoencoder/final.ckpt"
    encoder_dimensions: int = 512
    save_every_samples: int = 20000
    total_samples: int = 40000000
    batch_size: int = 20
    accum_batches: int = 1
    continue_training: bool = False
    eval_fid: bool = True
    eval_lpips: bool = True
    net_ch: Union[int, None] = None
    """
    Optional decoder-UNet architecture overrides, applied on top of diffae's
    square64_autoenc template in adjust_config(). None keeps the template value
    (net_ch 64, net_ch_mult (1, 2, 4, 8), net_attn (16,)), which is a 64px
    design: at 256px it has only four levels, bottlenecks at 32x32 and never
    reaches the attention resolution. DiffAE's own 256px recipe is net_ch 128,
    net_ch_mult [1, 1, 2, 2, 4, 4], net_attn [16], net_enc_channel_mult
    [1, 1, 2, 2, 4, 4, 4]. Changing these starts a NEW model; checkpoints of
    the old architecture do not load.
    """
    net_ch_mult: Union[list, None] = None
    net_attn: Union[list, None] = None
    net_enc_channel_mult: Union[list, None] = None
    precision: Union[str, None] = None
    """
    Lightning precision string ("bf16-mixed", "16-mixed", "32-true") and
    wall-clock budget ("DD:HH:MM:SS") per train() call; None keeps diffae's
    defaults (fp16 with loss scaling, no time limit).
    """
    max_time: Union[str, None] = None
    encoder_path: Union[str, None] = None
    sampler: Union[dict, None] = None
    render_noise: str = "inverted"
    is_torchvision_resnet: bool = False
    is_loaded: bool = True
    model_type: Union[str, None] = None
    encoder: Union[str, None] = None
    """
    Alias for model_type. The generator configs under configs/didae_experiments
    spell this key `encoder` (e.g. "open_clip:ViT-L-14:laion2b_s32b_b82k"), but
    it was never declared here, so pydantic dropped it without a word and the
    run silently fell back to the diffusion autoencoder's own semantic encoder.
    That is why $PEAL_RUNS/celeba/diffusion_autoencoder_openclip_vit_l14 has
    model_type: null and has never actually seen an OpenCLIP feature.
    """
    encoder_input_space: str = "legacy"
    sparse_dictionary: Union[str, SparseDictionaryConfig, None] = None
    visualizations_per_component: Union[int, None] = 1


class DiffusionAutoencoder(InvertibleGenerator, EditCapableGenerator):
    """Diffusion autoencoder generator built on the DiffAE ``LitModel``.

    An image is encoded to a semantic code ``z_sem`` plus a diffusion state (the
    DDIM-inverted ``xT``, and for DDPM inversion the per-step noise maps ``zs``),
    edited in ``z_sem`` and decoded back with the conditional UNet, so the edit
    changes semantics while the state preserves the input's details. The semantic
    encoder is either the one DiffAE trains jointly or a configured pretrained
    backbone (see :meth:`set_encoder`), and an optional sparse dictionary fitted
    on that space names the directions an edit can follow.

    The generator is used at three levels: ``encode``/``decode`` alone (the latent
    half of DiDAE, which works without a decoder checkpoint),
    :meth:`sweep_all_directions` for finding which dictionary atoms flip a
    classifier, and :meth:`edit` as the ``EditCapableGenerator`` entry point of
    the counterfactual explainer.

    Parameters
    ----------
    config : dict, str, Path or DiffusionAutoencoderConfig
        See :class:`DiffusionAutoencoderConfig`.
    predictor_dataset : Dataset, optional
        The classifier's dataset, used for the normalisation conversions in
        :meth:`edit` and to fill in a missing ``data.dataset_path``.
    model_dir : str, optional
        Overrides ``config.base_path`` as the model directory.
    device : str
        Ignored; CUDA is used when available.

    Attributes
    ----------
    model : LitModel or None
        The trained autoencoder; ``None`` when no checkpoint exists yet.
    latent_model : LitModel or None
        The latent DDIM prior used by :meth:`sample_x`.
    encoder : torch.nn.Module or None
        The configured semantic encoder, with its input-space conversion.
    sparse_dictionary : SparseDictionary or None
        The fitted dictionary over the encoder's feature space.
    sample_type : {"ddim_inv", "ddpm_inv"}
        Which inversion ``encode``/``decode`` use by default.
    """

    def __init__(self, config, predictor_dataset=None, model_dir=None, device="cpu"):
        """Build the generator: datasets, encoder, checkpoints and dictionary.

        Resolves the data config (falling back to the predictor dataset's
        ``dataset_path`` when the generator config has none), builds the three
        generator dataset splits, installs the semantic encoder
        (:meth:`set_encoder`), loads the autoencoder and latent-DDIM checkpoints
        from ``<base_path>/square64_ddim`` and
        ``<base_path>/square64_autoenc_latent`` (:meth:`load_models`), loads the
        sparse dictionary if its weights are on disk, and applies
        ``config.sampler``.

        Parameters
        ----------
        config : dict, str, Path or DiffusionAutoencoderConfig
            Parsed by ``load_yaml_config`` into a
            :class:`DiffusionAutoencoderConfig`.
        predictor_dataset : Dataset, optional
            The classifier's dataset; deep-copied.
        model_dir : str, optional
            Directory the model is looked up in. Defaults to
            ``config.base_path``.
        device : str
            Ignored; the device is CUDA when available and CPU otherwise.

        Notes
        -----
        ``self.model`` stays ``None`` when no decoder checkpoint exists, which
        still allows ``encode(..., only_semantic=True)``.
        """
        super().__init__()
        self.config = load_yaml_config(config)
        # check if cuda device is available and assign to self.device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor_dataset = copy.deepcopy(predictor_dataset)
        # TODO something is wrong here!!!
        if (
            getattr(self.config.data, "dataset_path", None) is None
            and self.predictor_dataset is not None
        ):
            self.config.data.dataset_path = self.predictor_dataset.config.dataset_path
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
        self.load_sparse_dictionary()
        self.set_sampler(self.config.sampler)

    def feature_extractor_tag(self) -> str:
        """Names the space get_feature_extractor() produces activations in."""
        if getattr(self, "encoder", None) is not None:
            return str(
                self.config.model_type
                or self.config.encoder
                or self.config.encoder_path
            )
        return "diffusion_autoencoder_semantic"

    def get_feature_extractor(self):
        """The module whose outputs the sparse dictionary is fitted on.

        A configured encoder wins over the diffusion autoencoder's own semantic
        encoder: callers used to do `try: self.model.ema_model.encoder except
        AttributeError: self.encoder`, which meant that setting model_type
        changed nothing at all for a DiffAE that had its own encoder — the
        configured one was built and then never used.
        """
        if getattr(self, "encoder", None) is not None:
            return self.encoder
        try:
            return self.model.ema_model.encoder
        except AttributeError:
            return self.encoder

    def load_sparse_dictionary(self):
        # Generators configured without a dictionary (the original DAE baseline)
        # otherwise never get the attribute at all, and anything reading
        # generator.sparse_dictionary raises AttributeError. None is already a
        # valid value here -- it is what the missing-weights branch below sets.
        """Load the configured sparse dictionary from disk, if it was fitted.

        Fills in ``sparse_dictionary.name`` (defaulting to the dictionary type),
        derives its ``base_path`` and ``weights_path`` under the generator's
        ``base_path`` and sets ``self.sparse_dictionary`` to the loaded
        dictionary -- or to ``None`` when none is configured or the weights file
        is missing, so the attribute can always be read.
        """
        self.sparse_dictionary = None
        if not self.config.sparse_dictionary is None:
            if self.config.sparse_dictionary.name is None:
                self.config.sparse_dictionary.name = (
                    self.config.sparse_dictionary.sparse_dictionaries_type
                )

            self.config.sparse_dictionary.base_path = os.path.join(
                self.config.base_path,
                self.config.sparse_dictionary.name,
            )
            self.config.sparse_dictionary.weights_path = os.path.join(
                self.config.sparse_dictionary.base_path,
                self.config.sparse_dictionary.weights_name,
            )
            if not os.path.exists(self.config.sparse_dictionary.weights_path):
                self.sparse_dictionary = None

            else:
                self.sparse_dictionary = get_sparse_dictionary(
                    self.config.sparse_dictionary
                )
                self.sparse_dictionary.load_from_disk(
                    self.config.sparse_dictionary.weights_path
                )

    def sample_z(self, batch_size=1):
        # TODO this has to be done properly with the learned prior!!!
        """Draw a prior sample ``(z_sem, xT)`` of standard Gaussians.

        Returns
        -------
        tuple of torch.Tensor
            ``z_sem`` of shape ``(batch_size, encoder_dimensions)`` and ``xT`` of
            shape ``(batch_size, *data.input_size)``.

        Notes
        -----
        This ignores the trained latent DDIM prior, so ``z_sem`` is not actually
        distributed like the encoder's codes.
        """
        z_sem = torch.randn(batch_size, self.config.encoder_dimensions).to(self.device)
        xT = torch.randn([batch_size] + self.config.data.input_size).to(self.device)
        return z_sem, xT

    def sample_x(self, batch_size=1):
        """Sample images from the latent DDIM prior, or noise if it is untrained.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.

        Returns
        -------
        torch.Tensor
            ``(batch_size, *data.input_size)``: real samples when
            ``self.latent_model`` is loaded, plain Gaussian noise otherwise.
        """
        if not self.latent_model is None:
            return self.latent_model.sample(N=batch_size, device=self.device)

        else:
            return torch.randn([batch_size] + self.config.data.input_size).to(
                self.device
            )

    def encode(self, x, t=1.0, only_semantic=False, sample_type=None):
        """Encode images to a semantic code plus the state they decode from.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, H, W)`` in generator normalisation.
        t : float
            Unused; kept for the ``InvertibleGenerator`` signature.
        only_semantic : bool
            Return just ``z_sem`` and skip the expensive inversion.
        sample_type : {"ddim_inv", "ddpm_inv"}, optional
            Which inversion to run. Defaults to the sampler chosen by
            :meth:`set_sampler`, and to ``"ddim_inv"`` if none was configured.

        Returns
        -------
        torch.Tensor or tuple
            ``z_sem`` of shape ``(B, encoder_dimensions)`` when ``only_semantic``;
            ``(z_sem, xT)`` for DDIM; ``(z_sem, xT, zs)`` for DDPM, where ``zs``
            maps each timestep to its per-step noise map.

        Raises
        ------
        RuntimeError
            Without a decoder checkpoint, when there is no configured encoder to
            fall back on, or when more than the semantic code is requested.
        ValueError
            On an unknown ``sample_type`` or ``config.render_noise``.

        Notes
        -----
        ``config.render_noise`` decides what a counterfactual is decoded from:
        ``"inverted"`` inverts ``x`` under its own ``z_sem``, so the edit keeps
        the input's details, while ``"fresh"`` returns fresh Gaussian noise, so
        the decode is a new sample from the edited ``z_sem``.
        """
        if self.model is None:
            # No diffusion checkpoint on disk yet. z_sem still exists whenever
            # the semantic encoder is a configured pretrained model rather than
            # a jointly-trained one: LitModel only ever deep-copies conf.encoder
            # into ema_model.encoder, so this returns the very same activations
            # the trained decoder would be conditioned on. That is what lets the
            # latent-space half of DiDAE (distillation, concept matching,
            # direction ranking) run before the decoder is trained; anything
            # that has to decode still needs the checkpoint.
            encoder = self.get_feature_extractor()
            if encoder is None:
                raise RuntimeError(
                    f"No diffusion autoencoder checkpoint under "
                    f"{self.config.base_path} and no configured encoder to fall "
                    "back on, so there is nothing to encode with. Train the "
                    "generator first, or set encoder/model_type to a pretrained "
                    "semantic encoder."
                )
            if not only_semantic:
                raise RuntimeError(
                    "Only semantic encoding is available without a diffusion "
                    "autoencoder checkpoint: DDIM/DDPM inversion needs the "
                    "trained UNet. Call encode(..., only_semantic=True), or "
                    "train the generator."
                )
            return encoder(x)

        z_sem: torch.Tensor = self.model.encode(x)
        # TODO why is t not used here???
        if only_semantic:
            return z_sem
        if sample_type is None:
            # Follow the sampler configured via set_sampler(); DDIM only as the
            # fallback when no sampler was ever configured.
            sample_type = getattr(self, "sample_type", "ddim_inv")
        if sample_type not in ("ddim_inv", "ddpm_inv"):
            raise ValueError(f"Unknown sample_type: {sample_type}")
        render_noise = getattr(self.config, "render_noise", "inverted") or "inverted"
        if render_noise == "fresh":
            # No inversion: the diffusion state has the input's shape (pixel-space
            # DiffAE), so fresh noise of that shape replaces the inverted x_T, and
            # the edit-friendly sampler gets fresh per-step noise maps {t: z_t}.
            xT = torch.randn_like(x)
            if sample_type == "ddim_inv":
                return z_sem, xT
            T = int(self.model.ddpm_sampler.T)
            return z_sem, xT, {t: torch.randn_like(x) for t in range(1, T + 1)}
        if render_noise != "inverted":
            raise ValueError(
                f"Unknown render_noise: {render_noise!r} (inverted | fresh)"
            )
        if sample_type == "ddim_inv":
            return z_sem, self.model.encode_stochastic(x, z_sem)
        return z_sem, *self.model.encode_ddpm_inversion(x, z_sem)

    def decode(self, z, t=1.0, sample_type=None):
        """
        Inverse of encode: returns images in generator normalization.

        render() rescales the raw model output to [0, 1] while
        decode_ddpm_inversion() does not, so the ddim branch is mapped back to
        generator normalization to keep both samplers interchangeable.
        """
        if sample_type is None:
            # Follow the sampler configured via set_sampler(); DDIM only as the
            # fallback when no sampler was ever configured.
            sample_type = getattr(self, "sample_type", "ddim_inv")
        if sample_type == "ddim_inv":
            z_sem, xT = z
            return self.generator_dataset.project_from_pytorch_default(
                self.model.render(xT, z_sem, grads=True)
            )
        elif sample_type == "ddpm_inv":
            z_sem, xT, zs = z
            return self.model.decode_ddpm_inversion(xT, zs, z_sem)
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

    def set_encoder(self):
        """Build ``self.encoder``, the semantic encoder ``z_sem`` is read from.

        ``config.model_type`` (or its alias ``config.encoder``) selects a
        pretrained backbone, each wrapped in a small ``nn.Module`` that applies
        that checkpoint's own resize and mean/std to a batched tensor and returns
        one vector per image: ``dino_v2_base_reg`` (the token space the RA-SAE
        dictionaries decompose), ``dino_v2``/``dino_v2_small``/``dino_v2_base``,
        the ``dino_v3_*`` family, ``open_clip:<arch>:<pretrained>``,
        ``clip``/``openai_clip:<arch>`` (the space the MSAE checkpoints live in)
        and ``UNI``. Failing that, ``config.encoder_path`` loads a pickled
        predictor and replaces its head with ``nn.Identity``. With neither,
        ``self.encoder`` is ``None`` and DiffAE trains its own semantic encoder.

        ``config.encoder_input_space`` then decides what is prepended, because the
        dataloader already hands out images in *generator* normalisation:
        ``"pytorch_default"`` denormalises back to [0, 1] (correct for the
        pretrained backbones above), ``"dataset"`` prepends nothing, and
        ``"legacy"`` applies the dataset normalisation a second time -- wrong, but
        what the existing checkpoints were trained with, so it stays the default.

        Raises
        ------
        ValueError
            On an unknown ``dino_v3`` variant or ``encoder_input_space``.
        """
        if self.config.model_type is None and self.config.encoder is not None:
            self.config.model_type = self.config.encoder

        if not self.config.model_type is None:
            if self.config.model_type == "dino_v2_base_reg":
                # DINOv2 ViT-B/14 *with registers*, reproducing the exact token
                # space the pretrained RA-SAE dictionaries decompose
                # (huggingface.co/matybohacek/RA-SAE-DINOv2-32k, whose config
                # names torch.hub facebookresearch/dinov2 -> dinov2_vitb14_reg).
                # That SAE reads `dino.norm(forward_features(x)["x_prenorm"])`,
                # i.e. all 261 tokens after the final layernorm, and z_sem is
                # token 0 of those -- the normed CLS token, identical to
                # forward_features()["x_norm_clstoken"].
                #
                # The weights come from HF rather than torch.hub because the
                # hub repo's current main needs Python >= 3.10 (`float | None`
                # annotations) and dispatches attention through xFormers. The
                # two were checked against each other on this machine: on GPU,
                # HF facebook/dinov2-with-registers-base reproduces the hub
                # dinov2_vitb14_reg tokens bit for bit (0.0 max abs, against a
                # 0.68 spread between different images); on CPU they differ by
                # 1.9e-5 on activations of norm ~25, i.e. float32 kernel noise.
                #
                # NOTE this is NOT facebook/dinov2-base: that checkpoint has no
                # register tokens and different weights, so its CLS token is not
                # in the space the SAE was trained on.
                model = AutoModel.from_pretrained("facebook/dinov2-with-registers-base")

                class DinoV2WithRegisters(nn.Module):
                    """Expects a [0, 1] image, i.e. encoder_input_space=pytorch_default."""

                    # RA-SAE's own preprocessing: Resize(256, BICUBIC) ->
                    # CenterCrop(224) -> ImageNet mean/std.
                    RESIZE = 256
                    CROP = 224
                    MEAN = (0.485, 0.456, 0.406)
                    STD = (0.229, 0.224, 0.225)

                    def __init__(self, model):
                        """Wrap the HF DINOv2-with-registers backbone."""
                        super().__init__()
                        self.model = model

                    def forward(self, x):
                        """Resize, crop, normalise x; return the normed CLS token."""
                        if x.ndim == 3:
                            x = x.unsqueeze(0)
                        # torchvision's tensor bicubic is not bit-identical to
                        # PIL's, the same approximation the open_clip and
                        # OpenAI-CLIP branches above already make. Resize takes
                        # the shorter side to 256 keeping the aspect ratio, as
                        # transforms.Resize(256) does on a PIL image.
                        x = torchvision.transforms.functional.resize(
                            x,
                            self.RESIZE,
                            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                            antialias=True,
                        ).clamp(0, 1)
                        x = torchvision.transforms.functional.center_crop(
                            x, [self.CROP, self.CROP]
                        )
                        mean = torch.tensor(self.MEAN, device=x.device, dtype=x.dtype)
                        std = torch.tensor(self.STD, device=x.device, dtype=x.dtype)
                        x = (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
                        # last_hidden_state is post-final-layernorm, so [:, 0]
                        # is dino.norm(x_prenorm)[:, 0].
                        return self.model(pixel_values=x).last_hidden_state[:, 0]

                encoder = DinoV2WithRegisters(model)

            elif self.config.model_type[: len("dino_v2")] == "dino_v2":
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
                    """HF DINOv2 backbone returning the CLS token.

                    The processor's ``crop_size`` and mean/std are applied to a batched
                    tensor, so the input is expected in [0, 1].
                    """

                    def __init__(self, model, processor):
                        """Store the backbone and its image processor."""
                        super().__init__()
                        self.model = model
                        self.processor = processor

                    def forward(self, x):
                        """Resize and normalise x, then return the CLS token."""
                        cs = self.processor.crop_size
                        x_resized = torchvision.transforms.Resize(
                            [cs["height"], cs["width"]]
                        )(x)

                        def pv(v):
                            """Broadcast a per-channel constant to the crop size."""
                            v = torch.tensor(v).to(x_resized)[:, None, None]
                            return torch.tile(v, [1, cs["height"], cs["width"]])

                        x_processed = (x_resized - pv(self.processor.image_mean)) / pv(
                            self.processor.image_std
                        )
                        latent_code = self.model(x_processed)["last_hidden_state"][:, 0]
                        return latent_code

                encoder = DinoV2(model, processor)

            elif self.config.model_type[: len("dino_v3")] == "dino_v3":

                if self.config.model_type == "dino_v3_small":
                    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"

                elif self.config.model_type == "dino_v3_small_plus":
                    model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"

                elif self.config.model_type == "dino_v3_base":
                    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"

                elif self.config.model_type == "dino_v3_large":
                    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

                elif self.config.model_type == "dino_v3_huge":
                    model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

                else:
                    raise ValueError(f"Unknown model type: {self.config.model_type}")

                model = AutoModel.from_pretrained(model_name)
                processor = AutoImageProcessor.from_pretrained(model_name)

                class DinoV3(nn.Module):
                    """HF DINOv3 backbone returning ``pooler_output``.

                    The processor's resize and mean/std are applied to a
                    batched tensor in [0, 1]; the CLS token would be
                    ``last_hidden_state[:, 0]`` instead.
                    """

                    def __init__(self, model, processor):
                        """Store the backbone and its image processor."""
                        super().__init__()
                        self.model = model
                        self.processor = processor

                    def forward(self, x):
                        """Resize and normalise x, then return the pooled embedding."""
                        cs = self.processor.crop_size

                        x = torchvision.transforms.Resize((cs["height"], cs["width"]))(
                            x
                        )

                        mean = torch.tensor(
                            self.processor.image_mean,
                            device=x.device,
                            dtype=x.dtype,
                        ).view(1, -1, 1, 1)

                        std = torch.tensor(
                            self.processor.image_std,
                            device=x.device,
                            dtype=x.dtype,
                        ).view(1, -1, 1, 1)

                        x = (x - mean) / std

                        outputs = self.model(pixel_values=x)

                        # Preferred embedding for DINOv3
                        return outputs.pooler_output

                        # Alternatively:
                        # return outputs.last_hidden_state[:, 0]

                encoder = DinoV3(model, processor)

            elif "open_clip" in self.config.model_type:
                # --- New OpenCLIP Logic ---
                import open_clip

                # Expecting config.model_type format: "open_clip:ViT-B-32:laion2b_s34b_b79k"
                # Defaulting to ViT-B-32 if only "open_clip" is provided
                parts = self.config.model_type.split(":")
                model_name = parts[1] if len(parts) > 1 else "ViT-B-32"
                pretrained = parts[2] if len(parts) > 2 else "laion2b_s34b_b79k"

                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained
                )

                class OpenCLIPEncoder(nn.Module):
                    """open_clip visual tower producing one embedding per image.

                    ``preprocess`` is only read for the checkpoint's input
                    resolution and its normalisation constants; it is never
                    called, because it is a PIL-oriented ``Compose`` that
                    would throw on a tensor batch.
                    """

                    def __init__(self, model, preprocess):
                        """Store the open_clip model and its preprocessing Compose."""
                        super().__init__()
                        self.model = model
                        # Convert the Compose transform to an nn.Module-like flow if possible,
                        # or keep as is if input is PIL. If input is Tensor, use torchvision.
                        self.preprocess = preprocess

                    def forward(self, x):
                        # self.preprocess is a PIL-oriented Compose containing
                        # ToTensor(); calling it on an already-batched tensor
                        # throws. Apply only the parts that mean anything for a
                        # tensor batch — the resize to the checkpoint's input
                        # resolution and its normalisation constants — reading
                        # both off the checkpoint's own preprocessing rather than
                        # hard-coding 224 and the CLIP means.
                        """Resize and normalise x, then return the image embedding."""
                        size, mean, std = 224, None, None
                        for t in self.preprocess.transforms:
                            name = type(t).__name__
                            if name == "Resize":
                                size = t.size if isinstance(t.size, int) else t.size[0]
                            elif name == "Normalize":
                                mean, std = t.mean, t.std

                        # Expects a [0, 1] image, which is what
                        # encoder_input_space="pytorch_default" delivers.
                        x_processed = x
                        if x_processed.ndim == 3:
                            x_processed = x_processed.unsqueeze(0)
                        x_processed = torch.nn.functional.interpolate(
                            x_processed,
                            size=(size, size),
                            mode="bicubic",
                            align_corners=False,
                            antialias=True,
                        ).clamp(0, 1)
                        if mean is not None:
                            m = torch.tensor(mean, device=x_processed.device).view(
                                1, -1, 1, 1
                            )
                            s_ = torch.tensor(std, device=x_processed.device).view(
                                1, -1, 1, 1
                            )
                            x_processed = (x_processed - m) / s_

                        # Extract visual features
                        latent_code = self.model.encode_image(x_processed)
                        return latent_code

                encoder = OpenCLIPEncoder(model, preprocess)

            elif self.config.model_type.split(":")[0] in ("clip", "openai_clip"):
                # OpenAI CLIP, which is NOT open_clip: same ViT-L/14 architecture
                # and the same 768 dimensions, different weights, different
                # embedding space. The MSAE checkpoints (arXiv:2502.20578) are
                # trained on this one — its precompute_activations.py does
                # `import clip; clip.load(...)` — so a dictionary from there only
                # means anything on top of these features.
                import clip as openai_clip

                parts = self.config.model_type.split(":")
                model_name = parts[1] if len(parts) > 1 else "ViT-L/14"
                # clip.load() defaults its download_root to ~/.cache/clip, which
                # the GPU nodes mount read-only. Put the weights wherever torch
                # already caches its own (TORCH_HOME), so one writable location
                # covers torchvision, torch.hub and CLIP alike.
                clip_cache = os.path.join(torch.hub.get_dir(), "clip")
                os.makedirs(clip_cache, exist_ok=True)
                clip_model, _ = openai_clip.load(
                    model_name, device="cpu", download_root=clip_cache
                )

                class OpenAICLIPEncoder(nn.Module):
                    """Expects a [0, 1] image, i.e. encoder_input_space=pytorch_default."""

                    # OpenAI CLIP's own preprocessing constants.
                    MEAN = (0.48145466, 0.4578275, 0.40821073)
                    STD = (0.26862954, 0.26130258, 0.27577711)

                    def __init__(self, model):
                        """Store the CLIP model and read its input resolution."""
                        super().__init__()
                        self.model = model
                        self.size = model.visual.input_resolution

                    def forward(self, x):
                        """Resize and normalise x, then return float image features."""
                        if x.ndim == 3:
                            x = x.unsqueeze(0)
                        x = torch.nn.functional.interpolate(
                            x,
                            size=(self.size, self.size),
                            mode="bicubic",
                            align_corners=False,
                            antialias=True,
                        ).clamp(0, 1)
                        m = torch.tensor(self.MEAN, device=x.device).view(1, -1, 1, 1)
                        s = torch.tensor(self.STD, device=x.device).view(1, -1, 1, 1)
                        return self.model.encode_image((x - m) / s).float()

                encoder = OpenAICLIPEncoder(clip_model)

            elif self.config.model_type == "UNI":
                import timm
                from timm.data import resolve_data_config
                from timm.data.transforms_factory import create_transform
                import sys

                import huggingface_hub

                # The UNI weights are gated. An unconditional login() here used
                # to open an interactive prompt inside generator construction,
                # which hangs or crashes every unattended run. Use the cached
                # token or $HF_TOKEN; only prompt when a person is at a terminal.
                _get_token = getattr(huggingface_hub, "get_token", None) or (
                    huggingface_hub.HfFolder.get_token
                )
                if _get_token() is None:
                    if sys.stdin.isatty():
                        huggingface_hub.login()
                    else:
                        raise RuntimeError(
                            "hf-hub:MahmoodLab/uni is a gated model and no Hugging "
                            "Face token is available. Set HF_TOKEN or run "
                            "`huggingface-cli login` before starting a "
                            "non-interactive run."
                        )

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
                    """UNI pathology ViT (``hf-hub:MahmoodLab/uni``) as an encoder.

                    Applies timm's transform for that checkpoint, so this wrapper only
                    accepts inputs that transform accepts. Building it calls
                    ``huggingface_hub.login()``, since the weights are gated.
                    """

                    def __init__(self, model, transform):
                        """Store the timm model and its timm data transform."""
                        super().__init__()
                        self.model = model
                        self.transform = transform

                    def forward(self, x):
                        """Transform x and return the model's pooled features."""
                        x_processed = self.transform(x)
                        latent_code = self.model(x_processed)
                        return latent_code

                encoder = UNI(model, transform)

        elif not self.config.encoder_path is None:
            try:
                encoder = torch.load(self.config.encoder_path, map_location="cpu")
            except Exception:
                encoder = torch.load(
                    self.config.encoder_path, map_location="cpu", weights_only=False
                )
            if self.config.is_torchvision_resnet:
                # remove the head
                encoder.model.fc = nn.Identity()

            else:
                encoder.fc = nn.Identity()

        else:
            encoder = None

        if encoder is None:
            self.encoder = None
            return

        # The dataloader already hands out images in *generator normalization*:
        # Normalization(mean, std) is part of the dataset transform, so
        # normalization = [0.5, 0.5] means the encoder is fed [-1, 1], not [0, 1].
        # encoder_input_space says what the encoder itself expects, and therefore
        # which conversion is baked in front of it:
        #
        #   "pytorch_default"  undo the dataset normalization -> [0, 1]. Correct
        #                      for the pretrained foundation models above
        #                      (open_clip, DINOv2/v3, UNI): each applies its own
        #                      mean/std on top of a [0, 1] image.
        #   "dataset"          identity. Correct for an encoder_path predictor
        #                      that was trained through this same data pipeline.
        #   "legacy"           apply the dataset normalization a *second* time.
        #                      Wrong, but it is what the existing checkpoints were
        #                      trained with, so it stays the default: changing it
        #                      changes z_sem and invalidates a trained decoder.
        normalization = self.generator_dataset.config.normalization
        input_space = getattr(self.config, "encoder_input_space", "legacy")

        if normalization is None or input_space == "dataset":
            self.encoder = encoder

        elif input_space == "pytorch_default":
            self.encoder = torch.nn.Sequential(
                DenormalizationModule(normalization[0], normalization[1]), encoder
            )

        elif input_space == "legacy":
            self.encoder = torch.nn.Sequential(
                NormalizationModule(normalization[0], normalization[1]), encoder
            )

        else:
            raise ValueError(
                f"Unknown encoder_input_space: {input_space!r} "
                '(expected "pytorch_default", "dataset" or "legacy")'
            )

    def set_sampler(self, sampler):
        """Select DDIM or DDPM inversion and optionally rebuild the sampler.

        Parameters
        ----------
        sampler : dict or None
            ``{"type": "ddim"|"ddpm", "num_steps": int, "spacing": str}``. The
            type sets ``self.sample_type`` to ``"<type>_inv"``, which
            :meth:`encode` and :meth:`decode` follow; for ``"ddpm"``,
            ``num_steps`` and ``spacing`` rebuild ``model.ddpm_sampler``.
            ``None`` leaves an already chosen sampler alone and otherwise falls
            back to ``"ddim_inv"``.
        """
        sampler_type = sampler.get("type") if sampler else None
        if sampler_type is not None:
            self.sample_type = f"{sampler_type}_inv"
            if self.model is None:
                return
            if sampler_type == "ddpm":
                kwargs = {}
                if sampler.get("num_steps") is not None:
                    kwargs["T"] = sampler["num_steps"]
                if sampler.get("spacing") is not None:
                    kwargs["spacing"] = sampler["spacing"]
                self.model.ddpm_sampler = self.model.create_ddpm_sampler(**kwargs)

        elif not hasattr(self, "sample_type"):
            self.sample_type = "ddim_inv"

    def adjust_config(self, conf):
        """Write this generator's settings into a diffae ``TrainConfig``.

        Copies the run directory, dataset, image size, batch and accumulation
        sizes, checkpoint cadence and ``encoder_dimensions`` (which diffae spells
        ``style_ch``, ``embed_channels``, ``net_beatgans_embed_channels`` and
        ``enc_out_channels``) onto ``conf``, hands it ``self.encoder`` as the
        semantic encoder, applies the optional UNet overrides ``net_ch``,
        ``net_ch_mult``, ``net_attn`` and ``net_enc_channel_mult`` plus the
        ``precision``/``max_time`` trainer settings, and calls
        ``conf.make_model_conf()`` so ``conf.model_conf`` stays consistent.
        ``$PEAL_NUM_WORKERS`` overrides the dataloader worker count, which is the
        binding constraint on memory-capped nodes.

        Parameters
        ----------
        conf : TrainConfig
            A diffae config from ``square64_autoenc()`` or
            ``square64_autoenc_latent()``; modified in place.
        """
        conf.base_dir = self.config.base_path
        conf.dataset = self.generator_dataset
        conf.img_size = self.generator_dataset.config.input_size[-1]
        conf.model_conf.image_size = self.generator_dataset.config.input_size[-1]
        conf.batch_size = self.config.batch_size
        # Host RAM is the binding constraint on the Slurm nodes (16 GiB cgroup
        # cap); each dataloader worker is a fork of a multi-GB parent. Allow
        # trading throughput for memory without touching the config schema.
        conf.num_workers = int(os.environ.get("PEAL_NUM_WORKERS", conf.num_workers))
        conf.accum_batches = getattr(self.config, "accum_batches", 16)
        conf.continue_training = getattr(self.config, "continue_training", False)
        conf.eval_fid = getattr(self.config, "eval_fid", True)
        conf.eval_lpips = getattr(self.config, "eval_lpips", True)
        conf.save_every_samples = self.config.save_every_samples
        conf.total_samples = self.config.total_samples
        conf.style_ch = self.config.encoder_dimensions
        conf.net_beatgans_embed_channels = self.config.encoder_dimensions
        conf.embed_channels = self.config.encoder_dimensions
        conf.enc_out_channels = self.config.encoder_dimensions
        conf.encoder = self.encoder
        # Optional architecture / trainer overrides (see the config docstring).
        # LitModel.__init__ rebuilds model_conf from these TrainConfig fields
        # via make_model_conf(), so setting them here is what takes effect; the
        # make_model_conf() call keeps conf.model_conf consistent for anything
        # that reads it before LitModel is built.
        for key in ("net_ch", "net_ch_mult", "net_attn", "net_enc_channel_mult"):
            val = getattr(self.config, key, None)
            if val is not None:
                setattr(
                    conf, key, tuple(val) if isinstance(val, (list, tuple)) else val
                )
        conf.precision = getattr(self.config, "precision", None)
        conf.max_time = getattr(self.config, "max_time", None)
        conf.make_model_conf()

    def train_model(
        self,
    ):
        """Train the autoencoder, then the latent DDIM, then the dictionary.

        Writes ``config.yaml`` into ``base_path`` and runs diffae's ``train()``
        three times: the autoencoder itself, an ``eval_programs=["infer"]`` pass
        that produces the latent inference file, and the latent DDIM prior under
        ``<base_path>/square64_autoenc_latent``. The checkpoints are then
        reloaded and, when a sparse dictionary is configured,
        :meth:`fit_sparse_dictionary` runs on the trained encoder.

        Notes
        -----
        The generator dataset is temporarily switched into ``return_dict`` and
        ``idx_enabled`` mode, which diffae's training loop expects, and restored
        afterwards.
        """
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
        latent_conf = square64_autoenc_latent(
            os.path.join(self.config.base_path, "square64_ddim")
        )
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
        """Load the autoencoder and latent DDIM checkpoints, if they exist.

        Prefers ``final.ckpt`` over ``last.ckpt`` in ``<base_path>/square64_ddim``
        and ``<base_path>/square64_autoenc_latent``, falling back to the
        configured ``checkpoint_path``. Sets ``self.model`` and
        ``self.latent_model`` to the loaded ``LitModel``, or to ``None`` when
        nothing is there, so the latent-only code paths still work.

        Notes
        -----
        When a checkpoint exists but ``config.is_loaded`` is false and
        ``continue_training`` is off, the whole ``base_path`` is moved aside to
        ``<base_path>_old_<timestamp>`` instead of being overwritten.
        """
        conf = square64_autoenc()
        self.adjust_config(conf)
        final_path = os.path.join(self.config.base_path, "square64_ddim", "final.ckpt")
        last_path = os.path.join(self.config.base_path, "square64_ddim", "last.ckpt")
        ckpt_to_load = (
            final_path
            if os.path.exists(final_path)
            else (last_path if os.path.exists(last_path) else self.checkpoint_path)
        )

        if os.path.exists(ckpt_to_load):
            if self.config.is_loaded or getattr(
                self.config, "continue_training", False
            ):
                self.model = LitModel.load_from_checkpoint(
                    checkpoint_path=ckpt_to_load,
                    conf=conf,
                    map_location="cpu",
                    strict=False,
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

        latent_conf = square64_autoenc_latent(
            os.path.join(self.config.base_path, "square64_ddim")
        )
        self.adjust_config(latent_conf)
        final_latent_path = os.path.join(
            self.config.base_path, "square64_autoenc_latent", "final.ckpt"
        )
        last_latent_path = os.path.join(
            self.config.base_path, "square64_autoenc_latent", "last.ckpt"
        )
        latent_ckpt_to_load = (
            final_latent_path
            if os.path.exists(final_latent_path)
            else (
                last_latent_path
                if os.path.exists(last_latent_path)
                else self.checkpoint_path_latent
            )
        )

        if os.path.exists(latent_ckpt_to_load):
            self.latent_model = LitModel.load_from_checkpoint(
                checkpoint_path=latent_ckpt_to_load,
                conf=latent_conf,
                map_location="cpu",
            )
            self.latent_model.to(self.device)

        else:
            self.latent_model = None

    def fit_sparse_dictionary(self):
        """Fit the configured sparse dictionary on the encoder's activations.

        Runs the feature extractor over the dictionary's own ``data`` config (or
        the generator's datasets) and fits on the train and validation splits,
        either from a cached activation file
        (:mod:`peal.sparse_dictionaries.activation_cache`, which caches all three
        splits so the later evaluation reads the same file) or straight from
        dataloaders. The weights are written to
        ``<base_path>/<dictionary name>/<weights_name>`` and the dictionary's
        ``config.yaml`` records ``fitted_on_encoder``, the tag of the space the
        components live in, so a consumer can check it matches its ``z_sem``.
        """
        self.sparse_dictionary = get_sparse_dictionary(self.config.sparse_dictionary)
        feature_extractor = self.get_feature_extractor()

        if getattr(self.config.sparse_dictionary, "data", None) is not None:
            datasets = get_datasets(self.config.sparse_dictionary.data)
        else:
            datasets = self.generator_datasets
        from peal.sparse_dictionaries import activation_cache

        data_cfg = (
            getattr(self.config.sparse_dictionary, "data", None) or self.config.data
        )
        if activation_cache.enabled() and hasattr(
            self.sparse_dictionary, "fit_from_activations"
        ):
            # Cache every split, not just the two used for fitting: the
            # evaluation afterwards wants the test split from the same file, and
            # a cache keyed on a different number of splits would invalidate
            # itself and re-extract on every run.
            acts = activation_cache.load_or_extract(
                self, list(datasets), data_cfg, feature_extractor, self.device
            )
            fit_on = acts[:2]
            self.sparse_dictionary.fit_from_activations(
                torch.cat([a[0] for a in fit_on], dim=0),
                (
                    torch.cat([a[1] for a in fit_on], dim=0)
                    if all(a[1] is not None for a in fit_on)
                    else None
                ),
            )
        else:
            # batch_size=10 with a single worker left the GPU at ~15% utilisation:
            # extracting the 182k CelebA activations was disk/decode bound and took
            # longer than fitting the dictionary on them. Same activations, same
            # order, just fetched in usable batches.
            self.sparse_dictionary.fit_from_dataloaders(
                [
                    torch.utils.data.DataLoader(
                        datasets[0], batch_size=256, num_workers=4, pin_memory=True
                    ),
                    torch.utils.data.DataLoader(
                        datasets[1], batch_size=256, num_workers=4, pin_memory=True
                    ),
                ],
                feature_extractor,
            )
        Path(self.config.sparse_dictionary.base_path).mkdir(parents=True, exist_ok=True)
        self.sparse_dictionary.save_on_disk(self.config.sparse_dictionary.weights_path)
        # Record the space these components live in, so a consumer that edits
        # z_sem can tell whether they belong to it.
        self.config.sparse_dictionary.fitted_on_encoder = self.feature_extractor_tag()
        save_yaml_config(
            self.config.sparse_dictionary,
            os.path.join(self.config.sparse_dictionary.base_path, "config.yaml"),
        )

    def explain_sae(self, sparse_dictionary=None):
        """Resolve the sparse dictionary and search counterfactuals for it.

        Parameters
        ----------
        sparse_dictionary : SparseDictionary, SparseDictionaryConfig or None
            A fitted dictionary to adopt, a config to load or fit, or ``None`` to
            keep the one already on the generator.

        Notes
        -----
        Runs :meth:`find_counterfactual` over all three dataset splits with batch
        size 1, which writes its images under
        ``<base_path>/counterfactuals_tests``.
        """
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

        if (
            hasattr(self.sparse_dictionary, "sae")
            and self.sparse_dictionary.sae is not None
        ):
            self.sparse_dictionary.sae.eval()

        explanation_path = os.path.join(
            self.config.base_path,
            self.config.sparse_dictionary.sparse_dictionaries_type,
        )
        Path(explanation_path).mkdir(parents=True, exist_ok=True)

        self.find_counterfactual(
            torch.utils.data.DataLoader(
                ConcatDataset(
                    [
                        self.generator_datasets[0],
                        self.generator_datasets[1],
                        self.generator_datasets[2],
                    ]
                ),
                batch_size=1,
            )
        )
        # self.add_sae_labels_to_image(torch.utils.data.DataLoader(ConcatDataset([self.generator_datasets[0], self.generator_datasets[1], self.generator_datasets[2]]), batch_size=1))

    def explain_all_components(self, sparse_dictionary=None):
        """Characterise every dictionary component and render its edits.

        Encodes the validation split, computes the component coefficients
        ``c = z_sem @ components`` and writes, under
        ``<base_path>/<dictionary name>/``: ``c_min_and_maxes.txt`` (the observed
        range per component), ``correlations.png`` (component vs ground-truth
        attribute correlations) and, when the latent DDIM is trained,
        ``samples.png``. It then calls :meth:`explain_sparse_component` for every
        component in ``[comp_min, comp_max)`` -- a wide dictionary cannot be swept
        exhaustively, since each component costs a full encode plus a line search
        of decodes.

        Parameters
        ----------
        sparse_dictionary : SparseDictionary, SparseDictionaryConfig or None
            As in :meth:`explain_sae`.

        Returns
        -------
        None
            The per-component results are written to disk, not returned.
        """
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

        if self.config.sparse_dictionary.name is None:
            self.config.sparse_dictionary.name = (
                self.config.sparse_dictionary.sparse_dictionaries_type
            )

        if (
            hasattr(self.sparse_dictionary, "sae")
            and self.sparse_dictionary.sae is not None
        ):
            self.sparse_dictionary.sae.eval()
        explanation_path = os.path.join(
            self.config.base_path, self.config.sparse_dictionary.name
        )
        Path(explanation_path).mkdir(parents=True, exist_ok=True)

        task_config_buffer = (
            self.generator_datasets[1].task_config
            if hasattr(self.generator_datasets[1], "task_config")
            else None
        )
        self.generator_datasets[1].task_config = None
        y_list = []
        c_list = []
        z_list = []
        for idx, batch in enumerate(
            torch.utils.data.DataLoader(self.generator_datasets[1], batch_size=10)
        ):
            _log.info("%s", str(10 * idx) + "/" + str(len(self.generator_datasets[1])))
            x, y = batch
            z = self.encode(x.to(self.device), only_semantic=True)
            c = z @ self.sparse_dictionary.get_components().to(self.device)
            y_list.append(y)
            c_list.append(c.detach().cpu())
            z_list.append(z.detach().cpu())

        y_stack = torch.cat(y_list)
        c_stack = torch.cat(c_list)
        z_stack = torch.cat(z_list)

        c_min = c_stack.min(dim=0).values
        c_max = c_stack.max(dim=0).values
        with open(os.path.join(explanation_path, "c_min_and_maxes.txt"), "w") as f:
            for i in range(c_min.shape[0]):
                f.write(
                    f"Component {i}: min={c_min[i].item():.4f}, max={c_max[i].item():.4f}\n"
                )

        plot_component_ground_truth_correlations(
            filename=os.path.join(
                self.config.base_path,
                self.config.sparse_dictionary.name,
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

        # A wide dictionary cannot be swept exhaustively: every component costs a
        # full encode + line-search of decodes. comp_min/comp_max bound the sweep
        # (the same fields find_counterfactual already honours).
        comp_min = (
            self.config.sparse_dictionary.comp_min
            if getattr(self.config.sparse_dictionary, "comp_min", -1) != -1
            else 0
        )
        comp_max = (
            self.config.sparse_dictionary.comp_max
            if getattr(self.config.sparse_dictionary, "comp_max", -1) != -1
            else self.config.sparse_dictionary.n_components
        )

        result_list = []
        for component_idx in range(comp_min, comp_max):
            result_list.append(
                self.explain_sparse_component(
                    torch.utils.data.DataLoader(
                        self.generator_datasets[1],
                        batch_size=self.config.sparse_dictionary.visualizations_per_component,
                    ),
                    component_idx,
                    c_min[component_idx].item(),
                    c_max[component_idx].item(),
                )
            )

    def find_counterfactual(self, dataloader):
        """Remove each labelled dictionary component from a matching sample.

        For every component in ``[comp_min, comp_max)`` that the SAE evaluation
        mapped to a ground-truth attribute, this walks ``dataloader`` for the
        first sample that carries the attribute and in which the component is
        active, projects the component out of ``z_sem`` and decodes. The edit is
        accepted only when re-encoding the counterfactual gives exactly the
        original active set minus that component; the pair is then written by
        :meth:`save_counterfactual` and the search moves on to the next
        component.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Yields ``(x, y)`` with ``x`` in generator normalisation and ``y`` the
            attribute vector; a batch size of 1 is assumed.

        Notes
        -----
        Needs ``sae_evaluation_results.pkl`` in the dictionary's ``base_path``.
        If it is missing, the SAE evaluation is run on the validation split
        first; if it still cannot be produced, the search is skipped with a
        warning.
        """
        import pickle

        eval_path = os.path.join(
            self.config.sparse_dictionary.base_path, "sae_evaluation_results.pkl"
        )
        if not os.path.exists(eval_path):
            _log.info(
                "%s",
                f"[find_counterfactual] Evaluation file '{eval_path}' not found. Computing SAE evaluation on validation dataset...",
            )
            from peal.sparse_dictionaries.sae_evaluation import run_sae_eval

            val_dataset = (
                self.generator_datasets[1]
                if hasattr(self, "generator_datasets")
                and len(self.generator_datasets) > 1
                else None
            )
            if val_dataset is not None:
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=128, shuffle=False
                )
                X_list, Y_list = [], []
                feature_extractor = self.get_feature_extractor()

                with torch.no_grad():
                    for batch in val_loader:
                        x_b = batch[0] if isinstance(batch, (list, tuple)) else batch
                        z_b = feature_extractor(x_b.to(self.device))
                        X_list.append(z_b.cpu())
                        if isinstance(batch, (list, tuple)) and len(batch) > 1:
                            Y_list.append(batch[1].cpu())

                X = torch.cat(X_list, dim=0)
                if len(Y_list) > 0:
                    Y = torch.cat(Y_list, dim=0)
                    attribute_names = getattr(val_dataset, "attributes", None)
                    if attribute_names is None:
                        attribute_names = [f"Attr_{i}" for i in range(Y.shape[1])]
                    run_sae_eval(
                        sae=self.sparse_dictionary,
                        x=X,
                        y=Y,
                        base_path=self.config.sparse_dictionary.base_path,
                        label_names=attribute_names,
                    )

        if not os.path.exists(eval_path):
            _log.info(
                "%s",
                f"[find_counterfactual] Warning: Evaluation file '{eval_path}' is still not available. Skipping counterfactual search.",
            )
            return

        with open(eval_path, "rb") as f:
            results = pickle.load(f)
        latent_to_label = results["latent_to_label"]

        comp_min = (
            self.config.sparse_dictionary.comp_min
            if hasattr(self.config.sparse_dictionary, "comp_min")
            and self.config.sparse_dictionary.comp_min not in (-1, None)
            else 0
        )

        n_comps = getattr(self.config.sparse_dictionary, "n_components", None)
        if n_comps is None and hasattr(self.sparse_dictionary, "get_components"):
            try:
                n_comps = self.sparse_dictionary.get_components().shape[1]
            except Exception:
                n_comps = 0
        if n_comps is None or n_comps == 0:
            n_comps = len(latent_to_label) if latent_to_label is not None else 0

        comp_max = (
            self.config.sparse_dictionary.comp_max
            if hasattr(self.config.sparse_dictionary, "comp_max")
            and self.config.sparse_dictionary.comp_max not in (-1, None)
            else n_comps
        )

        for component_idx in range(comp_min, comp_max):
            _log.info("%s %s", "Processing component ", component_idx)
            next_component = False
            for data_idx, batch in enumerate(dataloader):
                x, y = batch
                label = latent_to_label[component_idx]
                if label == -1:
                    break
                if y[0, label] == 0:
                    continue

                if self.sample_type == "ddim_inv":
                    z_sem, xT = self.encode(
                        x.to(self.device), sample_type=self.sample_type
                    )
                elif self.sample_type == "ddpm_inv":
                    z_sem, xT, zs = self.encode(
                        x.to(self.device), sample_type=self.sample_type
                    )
                z = self.sparse_dictionary.encode(z_sem)
                indizes_original = (z != 0).nonzero(as_tuple=True)[1].tolist()
                if component_idx not in indizes_original:
                    continue

                w_raw = self.sparse_dictionary.get_components()[:, component_idx].to(
                    self.device
                )
                w = w_raw / torch.norm(w_raw, p=2)
                proj_factors = (z_sem - self.sparse_dictionary.mu.to(self.device)) @ w
                factor = 1.0
                z_sem_after = z_sem - factor * proj_factors.unsqueeze(1) * w
                if self.sample_type == "ddim_inv":
                    x_counterfactuals = self.decode(
                        (z_sem_after, xT), sample_type=self.sample_type
                    )
                elif self.sample_type == "ddpm_inv":
                    x_counterfactuals = self.decode(
                        (z_sem_after, xT, zs), sample_type=self.sample_type
                    )

                z_sem_after_decoded = self.encode(
                    x_counterfactuals.to(self.device), only_semantic=True
                )
                z_after = self.sparse_dictionary.encode(z_sem_after_decoded)
                indizes_after = (z_after != 0).nonzero(as_tuple=True)[1].tolist()
                indizes = indizes_original.copy()
                indizes.remove(component_idx)
                if indizes_after != indizes:
                    continue
                else:
                    self.save_counterfactual(
                        x, x_counterfactuals, component_idx, data_idx, label
                    )
                    _log.info(
                        "%s",
                        f"Found counterfactual for component {component_idx} at data index {data_idx}",
                    )
                    next_component = True
                    break

            if not next_component:
                _log.info(
                    "%s %s", "No counterfactual found for component ", component_idx
                )

    def save_counterfactual(
        self, x, x_counterfactual, component_idx, data_idx, label, trial=0
    ):
        """Write a factual/counterfactual pair under ``counterfactuals_tests``.

        Both images are projected back to [0, 1] and saved as
        ``data_<data_idx>_trial_<trial>_original.png`` and the matching
        ``_counterfactual.png`` inside
        ``<base_path>/counterfactuals_tests/comp<component_idx>_label<label>/``.
        """
        c_path = os.path.join(
            self.config.base_path,
            "counterfactuals_tests",
            f"comp{component_idx}_label{label}",
        )
        x_path = os.path.join(
            c_path,
            f"data_{data_idx}_trial_{trial}_original.png",
        )
        x_counterfactual_path = os.path.join(
            c_path,
            f"data_{data_idx}_trial_{trial}_counterfactual.png",
        )
        Path(os.path.dirname(x_path)).mkdir(parents=True, exist_ok=True)

        x = self.generator_datasets[1].project_to_pytorch_default(x)
        x_counterfactual = self.generator_datasets[1].project_to_pytorch_default(
            x_counterfactual
        )
        torchvision.utils.save_image(x, x_path)
        torchvision.utils.save_image(x_counterfactual, x_counterfactual_path)

    def add_sae_labels_to_image(self, dataloader):
        """Add an inactive dictionary component and keep the clean edits.

        The mirror image of :meth:`find_counterfactual`: for data indices 30 to
        40 and every component in ``[comp_min, comp_max)`` that is *not* active
        in the sample, ``z_sem`` is pushed along the component by 1, 2, 3 and 5
        times its projection until re-encoding the decode gives exactly the
        original active set plus that component. Successful pairs are written by
        :meth:`save_counterfactual_2` under ``added_SAE_dimensions``.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Yields ``(x, y)`` in generator normalisation; a batch size of 1 is
            assumed.

        Notes
        -----
        Requires ``sae_evaluation_results.pkl`` in the dictionary's
        ``base_path`` and raises if it is absent.
        """
        import pickle

        with open(
            self.config.sparse_dictionary.base_path + "/sae_evaluation_results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
        latent_to_label = results["latent_to_label"]

        comp_min = (
            self.config.sparse_dictionary.comp_min
            if hasattr(self.config.sparse_dictionary, "comp_min")
            and self.config.sparse_dictionary.comp_min != -1
            else 0
        )
        comp_max = (
            self.config.sparse_dictionary.comp_max
            if hasattr(self.config.sparse_dictionary, "comp_max")
            and self.config.sparse_dictionary.comp_max != -1
            else self.config.sparse_dictionary.n_components
        )

        for data_idx, batch in enumerate(dataloader):

            if data_idx < 30:
                continue

            for component_idx in range(comp_min, comp_max):
                _log.info(
                    "%s",
                    f"Processing component {component_idx} at data index {data_idx}",
                )
                x, y = batch
                if self.sample_type == "ddim_inv":
                    z_sem, xT = self.encode(
                        x.to(self.device), sample_type=self.sample_type
                    )
                elif self.sample_type == "ddpm_inv":
                    z_sem, xT, zs = self.encode(
                        x.to(self.device), sample_type=self.sample_type
                    )
                z = self.sparse_dictionary.encode(z_sem)
                indizes_original = (z != 0).nonzero(as_tuple=True)[1].tolist()
                if component_idx in indizes_original:
                    continue

                for factor in [1.0, 2.0, 3.0, 5.0]:

                    w_raw = self.sparse_dictionary.get_components()[
                        :, component_idx
                    ].to(self.device)
                    w = w_raw / torch.norm(w_raw, p=2)
                    proj_factors = (
                        z_sem - self.sparse_dictionary.mu.to(self.device)
                    ) @ w
                    z_sem_after = z_sem + factor * proj_factors.unsqueeze(1) * w
                    if self.sample_type == "ddim_inv":
                        x_counterfactuals = self.decode(
                            (z_sem_after, xT), sample_type=self.sample_type
                        )
                    elif self.sample_type == "ddpm_inv":
                        x_counterfactuals = self.decode(
                            (z_sem_after, xT, zs), sample_type=self.sample_type
                        )

                    z_sem_after_decoded = self.encode(
                        x_counterfactuals.to(self.device), only_semantic=True
                    )
                    z_after = self.sparse_dictionary.encode(z_sem_after_decoded)
                    indizes_after = (z_after != 0).nonzero(as_tuple=True)[1].tolist()
                    indizes = indizes_original.copy()
                    indizes.append(component_idx)
                    if indizes_after != indizes:
                        continue
                    else:
                        self.save_counterfactual_2(
                            x,
                            x_counterfactuals,
                            component_idx,
                            data_idx,
                            latent_to_label[component_idx],
                        )
                        # self.save_counterfactual(x, xT, component_idx, data_idx, 0)
                        _log.info(
                            "%s",
                            f"Created counterfactual for component {component_idx} at data index {data_idx}",
                        )
                        next_component = True
                        break

            if data_idx >= 40:
                break

    def save_counterfactual_2(
        self, x, x_counterfactual, component_idx, data_idx, label, trial=0
    ):
        """Write a factual/counterfactual pair under ``added_SAE_dimensions``.

        Same layout as :meth:`save_counterfactual`, but for the edits that *add*
        a component:
        ``<base_path>/added_SAE_dimensions/comp<component_idx>_label<label>/
        data_<data_idx>_trial_<trial>_{original,counterfactual}.png``.
        """
        c_path = os.path.join(
            self.config.base_path,
            "added_SAE_dimensions",
            f"comp{component_idx}_label{label}",
        )
        x_path = os.path.join(
            c_path,
            f"data_{data_idx}_trial_{trial}_original.png",
        )
        x_counterfactual_path = os.path.join(
            c_path,
            f"data_{data_idx}_trial_{trial}_counterfactual.png",
        )
        Path(os.path.dirname(x_path)).mkdir(parents=True, exist_ok=True)

        x = self.generator_datasets[1].project_to_pytorch_default(x)
        x_counterfactual = self.generator_datasets[1].project_to_pytorch_default(
            x_counterfactual
        )
        torchvision.utils.save_image(x, x_path)
        torchvision.utils.save_image(x_counterfactual, x_counterfactual_path)

    def compute_additional_info(
        self, z_sem, z_sem_after, latent_to_label, index, x_counterfactual
    ) -> str:
        """Report which named dictionary labels are active around an edit.

        Builds the caption text of the contrastive collages: the sorted labels
        active in ``z_sem``, the component the edit tried to remove, the labels
        active in ``z_sem_after``, and the labels active in the counterfactual
        once it has been decoded and re-encoded -- the last being the one that
        can differ from the intended edit.

        Parameters
        ----------
        z_sem, z_sem_after : torch.Tensor
            Semantic codes before and after the edit, ``(1, D)``.
        latent_to_label : sequence
            Maps a component index to its ground-truth label index.
        index : torch.Tensor
            Scalar tensor holding the edited component index.
        x_counterfactual : torch.Tensor
            The decoded counterfactual, in generator normalisation.

        Returns
        -------
        str
            A multi-line report.
        """
        result = ""
        z = self.sparse_dictionary.encode(z_sem)
        indizes = (z != 0).nonzero(as_tuple=True)[1].tolist()
        labels = [latent_to_label[i] for i in indizes]
        labels.sort()
        result += f"SAE labels of z_sem:                  {labels}"

        result += f"\n\ntry to remove:   {index.item()}"

        z = self.sparse_dictionary.encode(z_sem_after)
        indizes = (z != 0).nonzero(as_tuple=True)[1].tolist()
        labels = [latent_to_label[i] for i in indizes]
        labels.sort()
        result += f"\n\nSAE labels of z_sem_after:         {labels}"

        z_counter = self.encode(x_counterfactual.to(self.device), only_semantic=True)
        z = self.sparse_dictionary.encode(z_counter)
        indizes = (z != 0).nonzero(as_tuple=True)[1].tolist()
        labels = [latent_to_label[i] for i in indizes]
        labels.sort()
        result += f"\n\n\nSAE labels of x_counterfactual:  {labels}"

        return result

    def explain_sparse_component(
        self, dataloader, component_idx, c_min_val=None, c_max_val=None
    ):
        """Render the line-search visualisations for one component.

        Walks ``dataloader`` until ``config.visualizations_per_component``
        samples have been handled, calls
        :meth:`explain_sparse_component_batch` per batch and writes, under
        ``<base_path>/<dictionary name>/<component_idx>/``, one
        ``<idx>_linearsearch.png`` strip per sample (the original followed by the
        decode at every line-search factor) plus the dataset's own contrastive
        collages.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Source of factual samples in generator normalisation.
        component_idx : int
            Index of the dictionary component to suppress.
        c_min_val, c_max_val : float, optional
            Observed coefficient range of that component, used by the "dynamic"
            line-search step.

        Returns
        -------
        tuple of list
            The factual images and the chosen counterfactuals.
        """
        x_factual_list = []
        x_counterfactual_list = []
        start_idx = 0
        current_base_path = os.path.join(
            self.config.base_path,
            self.config.sparse_dictionary.name,
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
            (
                x_counterfactual,
                (
                    dot_before,
                    dot_after,
                ),
                x_counterfactuals_all,
            ) = self.explain_sparse_component_batch(
                x_factual, component_idx, c_min_val, c_max_val
            )
            x_counterfactual_list.extend(list(x_counterfactual.cpu()))

            for b_idx in range(x_factual.shape[0]):
                global_idx = start_idx + b_idx
                orig_img = batch[0][b_idx].cpu()
                cf_imgs = x_counterfactuals_all[b_idx]
                grid_imgs = torch.cat([orig_img.unsqueeze(0), cf_imgs], dim=0)
                grid_imgs = self.generator_datasets[1].project_to_pytorch_default(
                    grid_imgs
                )
                save_path = os.path.join(
                    current_base_path, f"{global_idx:07d}_linearsearch.png"
                )
                torchvision.utils.save_image(grid_imgs, save_path, nrow=len(grid_imgs))

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
                # additional_info=self.compute_additional_info(z_sem, z_sem_after, latent_to_label, index, x_counterfactual)
            )
            start_idx += len(x_factual)

        return x_factual_list, x_counterfactual_list

    def explain_sparse_component_batch(
        self, x_generator, component_idx, c_min_val=None, c_max_val=None
    ):
        """Project one component out of a batch and decode the line search.

        Encodes ``x_generator``, normalises the component to a unit direction
        ``w`` and, for every factor in ``[0.5, 1, 1.5, 2, 3, 5, 10, "dynamic"]``,
        steps ``z_sem`` by ``-factor * ((z_sem - mu) @ w) * w`` and decodes. The
        "dynamic" step instead aims the raw coefficient at ``c_min_val`` or
        ``c_max_val``, whichever lies on the far side of the current value.

        Parameters
        ----------
        x_generator : torch.Tensor
            ``(B, C, H, W)`` in generator normalisation.
        component_idx : int
            Dictionary component to suppress.
        c_min_val, c_max_val : float, optional
            Observed coefficient range; without them the dynamic step falls back
            to a plain factor of 2.

        Returns
        -------
        tuple
            ``(x_cf, (proj_before, proj_after), x_cf_all)``: the decode at factor
            2 in generator normalisation, the projection of ``z_sem`` on ``w``
            before and after that step, and ``(B, n_factors, C, H, W)`` holding
            every decode of the line search.
        """
        _log.info("%s", "[x_generator.min(), x_generator.max()]")
        _log.info("%s", [x_generator.min(), x_generator.max()])
        if self.sample_type == "ddim_inv":
            z_sem, xT = self.encode(
                x_generator.to(self.device), sample_type=self.sample_type
            )
        elif self.sample_type == "ddpm_inv":
            z_sem, xT, zs = self.encode(
                x_generator.to(self.device), sample_type=self.sample_type
            )
        w_raw = self.sparse_dictionary.get_components()[:, component_idx].to(
            self.device
        )
        w = w_raw / torch.norm(w_raw, p=2)
        proj_factors = (z_sem - self.sparse_dictionary.mu.to(self.device)) @ w
        linesearch_factors = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, "dynamic"]
        x_counterfactuals_all = []
        x_counterfactuals_generator = None
        proj_factors_after_2 = None

        for factor in linesearch_factors:
            if factor == "dynamic":
                if c_min_val is not None and c_max_val is not None:
                    c_current = z_sem @ w_raw
                    c_target = torch.where(
                        proj_factors > 0,
                        torch.tensor(c_min_val, device=self.device),
                        torch.tensor(c_max_val, device=self.device),
                    )
                    z_sem_after = (
                        z_sem
                        - ((c_current - c_target).unsqueeze(1) / torch.norm(w_raw, p=2))
                        * w
                    )
                else:
                    z_sem_after = z_sem - 2.0 * proj_factors.unsqueeze(1) * w
            else:
                z_sem_after = z_sem - factor * proj_factors.unsqueeze(1) * w

            proj_factors_after = (
                z_sem_after - self.sparse_dictionary.mu.to(self.device)
            ) @ w
            if self.sample_type == "ddim_inv":
                x_cf_gen = self.decode((z_sem_after, xT), sample_type=self.sample_type)
            elif self.sample_type == "ddpm_inv":
                x_cf_gen = self.decode(
                    (z_sem_after, xT, zs), sample_type=self.sample_type
                )
            # decode() already returns generator normalization
            x_counterfactuals_all.append(x_cf_gen.cpu())
            if factor == 2.0:
                x_counterfactuals_generator = x_cf_gen.cpu()
                proj_factors_after_2 = proj_factors_after.cpu()

        if x_counterfactuals_generator is None:
            x_counterfactuals_generator = x_counterfactuals_all[3]
            proj_factors_after_2 = (
                z_sem
                - 2.0 * proj_factors.unsqueeze(1) * w
                - self.sparse_dictionary.mu.to(self.device)
            ) @ w
            proj_factors_after_2 = proj_factors_after_2.cpu()

        x_counterfactuals_all = torch.stack(x_counterfactuals_all, dim=1)

        return (
            x_counterfactuals_generator,
            (
                proj_factors.cpu(),
                proj_factors_after_2,
            ),
            x_counterfactuals_all,
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
        """Produce counterfactuals for a batch by editing the semantic code.

        The ``EditCapableGenerator`` entry point used by the counterfactual
        explainer. ``x_in`` is converted from predictor to generator
        normalisation and encoded; the direction the edit follows is read off the
        last linear layer of a *distilled* predictor -- a copy of the classifier
        refitted as ``encoder -> Linear(D, 1)`` on ``z_sem``, loaded from
        ``<base_path>/explainer/distilled_predictor/`` when a checkpoint is there
        and distilled otherwise. ``_calculate_z_counterfactuals`` proposes
        ``num_attempts * len(linesearch_factors)`` edited codes per sample, each
        is decoded against the sample's own noise state and scored by the
        predictor, and per sample and attempt the decode whose target confidence
        comes closest to the goal is kept -- restricted to decodes whose relative
        outlier score lies in ``(0.1, 1.3)``.

        Parameters
        ----------
        x_in : torch.Tensor
            ``(B, C, H, W)`` in predictor normalisation.
        target_confidence_goal : float
            Overwritten: the goal is recomputed per sample as one minus the
            predictor's current confidence in the target class.
        source_classes, target_classes : torch.Tensor
            Class index per sample.
        predictor : torch.nn.Module
            The classifier the counterfactuals are scored against.
        explainer_config : ExplainerConfig or dict
            Reads ``distilled_predictor``, ``num_attempts``,
            ``linesearch_factors`` and ``sampler``.
        predictor_datasets : list
            Train/validation/test datasets or loaders in predictor
            normalisation. Copies carrying the generator's normalisation are used
            for the distillation, and the validation one supplies
            ``calculate_outlier_score``.
        pbar : object, optional
            Unused.
        base_path : str
            Run directory the distilled predictor is cached in.
        mode : str
            Unused.

        Returns
        -------
        tuple
            ``(counterfactuals, differences, end_confidences, originals, history,
            component_indices)``, each a list of length ``B * num_attempts``. The
            counterfactuals are in predictor normalisation, ``differences`` is
            ``x_in - x_cf``, and ``history`` holds
            ``"<component>:<c_int>:<c_factual>-><c_after>"`` strings when the edit
            went through a sparse dictionary (empty strings otherwise).
        """
        param_list = [p for p in predictor.parameters()]
        device = param_list[0].device
        pred_original = (
            torch.nn.functional.softmax(predictor(x_in.to(self.device))).detach().cpu()
        )
        target_confidences = [
            pred_original[i][target_classes[i]] for i in range(len(target_classes))
        ]
        target_confidence_goal = 1 - torch.tensor(target_confidences)

        if isinstance(predictor_datasets[1], WeightedDataloaderList):
            validation_dataset = predictor_datasets[1].dataloaders[0].dataset

        else:
            validation_dataset = predictor_datasets[1]

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
                    distilled_datasource.dataloaders[j].dataset.normalization = (
                        self.generator_datasets[idx].normalization
                    )
                    distilled_datasource.dataloaders[j].dataset.transform = (
                        self.generator_datasets[idx].transform
                    )
                    distilled_datasource.dataloaders[j].dataset.config.normalization = (
                        self.generator_datasets[idx].config.normalization
                    )

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
            # model.cpl holds the whole module and can be a stub left by a failed
            # save; checkpoints/final.cpl holds the state_dict of the same
            # nn.Sequential and is written separately. Take whichever loads --
            # the state_dict branch below rebuilds the architecture -- and only
            # re-distill when neither does.
            loaded_predictor = load_first_loadable(
                [
                    distilled_path,
                    os.path.join(
                        os.path.dirname(distilled_path), "checkpoints", "final.cpl"
                    ),
                ],
                map_location=self.device,
            )
            if loaded_predictor is None:
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
                if isinstance(loaded_predictor, (dict, collections.OrderedDict)):
                    self.gradient_predictor = nn.Sequential(
                        *[
                            self.model.ema_model.encoder,
                            nn.Linear(self.config.encoder_dimensions, 1, bias=False),
                        ]
                    ).to(self.device)
                    self.gradient_predictor.load_state_dict(loaded_predictor)
                else:
                    self.gradient_predictor = loaded_predictor

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

        sampler = (
            explainer_config.get("sampler")
            if isinstance(explainer_config, dict)
            else getattr(explainer_config, "sampler", None)
        )

        self.set_sampler(sampler)

        x_generator = classifier_to_generator(x_in)
        # x_generator = 2 * x_generator
        # x_generator = x_in

        if self.sample_type == "ddim_inv":
            z_sem, xT = self.encode(
                x_generator.to(self.device), sample_type=self.sample_type
            )
        elif self.sample_type == "ddpm_inv":
            z_sem, xT, zs = self.encode(
                x_generator.to(self.device), sample_type=self.sample_type
            )

        if hasattr(self.gradient_predictor, "get_last_layer"):
            w = self.gradient_predictor.get_last_layer().weight[0]
        else:
            w = list(self.gradient_predictor.children())[-1].weight[0]

        z_sem_before, indices, distances, c_info = self._calculate_z_counterfactuals(
            z_sem, w, explainer_config, explainer_config.num_attempts
        )
        z_sem2 = z_sem_before.reshape([-1, z_sem_before.shape[-1]])
        xT_decoding = xT.unsqueeze(1).unsqueeze(1)
        xT_decoding = xT_decoding.tile(
            1,
            explainer_config.num_attempts,
            len(explainer_config.linesearch_factors),
            1,
            1,
            1,
        )
        xT_decoding = xT_decoding.reshape([-1] + list(xT.shape[1:]))
        zs_new = (
            _zs_map(
                zs,
                lambda value: value.unsqueeze(1)
                .unsqueeze(1)
                .tile(
                    1,
                    explainer_config.num_attempts,
                    len(explainer_config.linesearch_factors),
                    1,
                    1,
                    1,
                )
                .reshape(-1, *value.shape[1:]),
            )
            if self.sample_type == "ddpm_inv"
            else None
        )

        # Rendering only: the counterfactual search happened in latent space
        # above and every consumer below detaches the images. decode() keeps
        # gradients on because the SCE explainer backpropagates through it, but
        # here that only kept the whole T-step autograd graph alive - the
        # reason decode batches had to be so small.
        with torch.no_grad():
            if self.sample_type == "ddim_inv":
                x_counterfactuals_generator = self.decode(
                    (z_sem2, xT_decoding), sample_type=self.sample_type
                )
            elif self.sample_type == "ddpm_inv":
                x_counterfactuals_generator = self.decode(
                    (z_sem2, xT_decoding, zs_new), sample_type=self.sample_type
                )
        # The decoder returns images in generator normalization, while the
        # predictor, the validation dataset and every downstream consumer of the
        # returned counterfactuals (collages, serialized finetuning datasets)
        # expect predictor normalization.
        x_counterfactuals = generator_to_classifier(
            x_counterfactuals_generator.detach()
        )

        preds = torch.nn.Softmax(dim=-1)(
            predictor(x_counterfactuals.to(device)).detach().cpu()
        )
        y_target_end_confidence = torch.zeros([preds.shape[0]])
        items_per_batch = preds.shape[0] // target_classes.shape[0]
        for i in range(preds.shape[0]):
            correct_batch_idx = i // items_per_batch
            y_target_end_confidence[i] = preds[i, target_classes[correct_batch_idx]]

        x_counterfactuals = torch.reshape(
            x_counterfactuals,
            list(z_sem_before.shape[:3]) + list(x_counterfactuals.shape[1:]),
        )
        y_target_end_confidence = torch.reshape(
            y_target_end_confidence, z_sem_before.shape[:3]
        )
        x_counterfactuals_out_list = []
        y_target_end_confidence_list = []
        x_out_list = []
        indices_list = []
        history_list = []
        _log.info(
            "%s",
            "x_counterfactuals: "
            + str(x_counterfactuals.min())
            + " to "
            + str(x_counterfactuals.max()),
        )
        _log.info(
            "%s",
            "x_counterfactuals: "
            + str(x_counterfactuals.min())
            + " to "
            + str(x_counterfactuals.max()),
        )
        _log.info(
            "%s",
            "x_counterfactuals: "
            + str(x_counterfactuals.min())
            + " to "
            + str(x_counterfactuals.max()),
        )
        for i in range(explainer_config.num_attempts):
            if x_counterfactuals.shape[2] >= 2:
                y_target_diff = torch.clone(y_target_end_confidence)
                for b in range(y_target_end_confidence.shape[0]):
                    outlier_scores = validation_dataset.calculate_outlier_score(
                        x_counterfactuals[b, i]
                    )["relative"].cpu()
                    mask = torch.logical_and(outlier_scores < 1.3, outlier_scores > 0.1)
                    masked_difference = y_target_diff[b, i] * mask
                    y_target_diff[b, i] = torch.abs(
                        masked_difference - target_confidence_goal[b]
                    )

                # j = torch.argmax(y_target_end_confidence[:, i, :], dim=-1)
                j = torch.argmin(y_target_diff[:, i, :], dim=-1)

            else:
                j = torch.zeros([x_counterfactuals.shape[0]], dtype=torch.long)

            for k in range(j.shape[0]):
                x_counterfactuals_out_list.append(x_counterfactuals[k, i, j[k], :])
                y_target_end_confidence_list.append(
                    float(y_target_end_confidence[k, i, j[k]])
                )
                indices_list.append(indices[k, i])
                if c_info is not None:
                    comp_idx = indices[k, i].item()
                    c_f = c_info["c_factual"][k, i].item()
                    c_i = c_info["c_int"][k, i].item()
                    c_a = c_info["c_after"][k, i, j[k]].item()
                    history_list.append(f"{comp_idx}:{c_i:.3f}:{c_f:.3f}->{c_a:.3f}")
                else:
                    history_list.append("")

            x_out_list.append(x_in)

        x_counterfactuals = torch.stack(x_counterfactuals_out_list, dim=0)
        x_out = torch.cat(x_out_list, dim=0)
        y_target_end_confidence = torch.tensor(y_target_end_confidence_list)
        indices = torch.tensor(indices_list)  # , dim=0)
        x_difference = x_out - x_counterfactuals.cpu()

        return (
            list(x_counterfactuals.cpu()),
            list(x_difference),
            list(y_target_end_confidence),
            list(x_in),
            history_list,
            list(indices.cpu()),
        )

    @staticmethod
    def compute_bipartite_gt_sae_matching(
        y_samples,
        c_sae,
        attribute_names=None,
        vocabulary=None,
        output_dir=None,
    ):
        """
        Computes optimal bipartite matching between ground-truth labels y_samples
        and SAE activations c_sae based on maximum F1 score. Writes full K_sae row mapping to matches.txt.
        """
        y_tensor = (
            y_samples.detach().cpu()
            if torch.is_tensor(y_samples)
            else torch.tensor(y_samples)
        )
        c_tensor = (
            c_sae.detach().cpu() if torch.is_tensor(c_sae) else torch.tensor(c_sae)
        )

        if y_tensor.ndim == 1:
            unique_vals = torch.unique(y_tensor)
            if len(unique_vals) <= 10:
                y_matrix = torch.nn.functional.one_hot(
                    y_tensor.long(), num_classes=len(unique_vals)
                ).float()
                if attribute_names is None:
                    attribute_names = [f"Class_{val.item()}" for val in unique_vals]
            else:
                y_matrix = y_tensor.unsqueeze(1).float()
                if attribute_names is None:
                    attribute_names = ["Target_Label"]
        else:
            y_matrix = y_tensor.float()

        N, K_gt = y_matrix.shape
        _, K_sae = c_tensor.shape

        if attribute_names is None or len(attribute_names) != K_gt:
            attribute_names = [
                f"Num{j}" if K_gt == 1000 else f"Attr_{j}" for j in range(K_gt)
            ]

        y_bin = (y_matrix > 0.5).float()  # [N, K_gt]
        c_bin = (c_tensor > 0.0).float()  # [N, K_sae]

        # Vectorized TP, FP, FN [K_gt, K_sae]
        tp = torch.matmul(y_bin.T, c_bin)  # [K_gt, K_sae]
        fp = torch.matmul((1.0 - y_bin).T, c_bin)  # [K_gt, K_sae]
        fn = torch.matmul(y_bin.T, (1.0 - c_bin))  # [K_gt, K_sae]

        prec_matrix = tp / (tp + fp + 1e-8)
        rec_matrix = tp / (tp + fn + 1e-8)
        f1_matrix = (2.0 * prec_matrix * rec_matrix) / (prec_matrix + rec_matrix + 1e-8)

        # Vectorized Pearson Correlation [K_gt, K_sae]
        y_mean = y_matrix.mean(dim=0, keepdim=True)  # [1, K_gt]
        c_mean = c_tensor.mean(dim=0, keepdim=True)  # [1, K_sae]

        y_cent = y_matrix - y_mean  # [N, K_gt]
        c_cent = c_tensor - c_mean  # [N, K_sae]

        y_std = torch.sqrt(torch.sum(y_cent**2, dim=0, keepdim=True)).T  # [K_gt, 1]
        c_std = torch.sqrt(torch.sum(c_cent**2, dim=0, keepdim=True))  # [1, K_sae]

        denom = y_std * c_std  # [K_gt, K_sae]
        cov = torch.matmul(y_cent.T, c_cent)  # [K_gt, K_sae]

        corr_matrix = torch.where(
            denom > 1e-6, cov / (denom + 1e-8), torch.zeros_like(cov)
        )

        matched_results = []
        assigned_gt = set()
        assigned_sae = set()

        score_matrix = f1_matrix * torch.relu(corr_matrix)
        flat_indices = torch.argsort(score_matrix.flatten(), descending=True)

        for flat_idx in flat_indices:
            j = (flat_idx // K_sae).item()
            k = (flat_idx % K_sae).item()

            if j in assigned_gt or k in assigned_sae:
                continue

            if score_matrix[j, k].item() == 0.0 and f1_matrix[j, k].item() == 0.0:
                break

            assigned_gt.add(j)
            assigned_sae.add(k)

            attr_name = attribute_names[j]
            dim_name = (
                vocabulary[k] if (vocabulary and k < len(vocabulary)) else f"dim_{k}"
            )

            matched_results.append(
                {
                    "attribute_idx": j,
                    "attribute_name": attr_name,
                    "sae_idx": k,
                    "sae_dim_name": dim_name,
                    "precision": prec_matrix[j, k].item(),
                    "recall": rec_matrix[j, k].item(),
                    "f1_score": f1_matrix[j, k].item(),
                    "correlation": corr_matrix[j, k].item(),
                }
            )

            if len(assigned_gt) == K_gt:
                break

        # Save to matches.txt if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            matches_path = os.path.join(output_dir, "matches.txt")
            lines = []
            for j in range(K_gt):
                attr_name = attribute_names[j]
                gt_count = int(y_bin[:, j].sum().item())
                max_f1 = f1_matrix[j].max().item()

                if max_f1 > 0.0:
                    best_k = torch.argmax(f1_matrix[j]).item()
                    dim_name = (
                        vocabulary[best_k]
                        if (vocabulary and best_k < len(vocabulary))
                        else f"dim_{best_k}"
                    )
                    sae_str = f"Matched SAE #{best_k} ({dim_name})"
                    f1 = f1_matrix[j, best_k].item()
                    prec = prec_matrix[j, best_k].item()
                    rec = rec_matrix[j, best_k].item()
                else:
                    sae_str = "No Matched SAE (-)"
                    f1 = 0.0
                    prec = 0.0
                    rec = 0.0

                lines.append(
                    f"Component {j} ({attr_name}): {sae_str} | Occurrences: {gt_count}/{N} samples | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}"
                )

            with open(matches_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            _log.info(
                "%s",
                f"[DiDAE] Saved ground-truth component matching ({K_gt} lines, sorted by GT component idx) to {matches_path}",
            )

        return matched_results

    def _load_component_bounds(
        self, K, device, dtype, component_bounds_scale=1.0, verbose=True
    ):
        """Empirical [c_min, c_max] per dictionary atom.

        These are the bounds DiDAE._ensure_component_bounds wrote, computed as
        raw `z @ W` projections -- the same coordinate `c_factual` uses here, so
        a target expressed in one is directly comparable to the other.

        Returns ([K], [K]) tensors, or (None, None) if the file is not there yet.
        """
        sd_base_path = getattr(
            getattr(self.sparse_dictionary, "config", None), "base_path", None
        )
        c_min_max_path = (
            os.path.join(sd_base_path, "c_min_and_maxes.txt") if sd_base_path else None
        )
        if not c_min_max_path or not os.path.exists(c_min_max_path):
            return None, None

        c_mins_raw, c_maxs_raw = [], []
        with open(c_min_max_path, "r") as file:
            for line in file:
                parts = line.strip().split("min=")
                if len(parts) > 1:
                    c_mins_raw.append(float(parts[1].split(",")[0]))
                    c_maxs_raw.append(float(line.strip().split("max=")[1]))

        c_mins = torch.tensor(c_mins_raw, device=device, dtype=dtype)
        c_maxs = torch.tensor(c_maxs_raw, device=device, dtype=dtype)

        if component_bounds_scale != 1.0:
            mid = 0.5 * (c_mins + c_maxs)
            half = 0.5 * (c_maxs - c_mins) * float(component_bounds_scale)
            c_mins = mid - half
            c_maxs = mid + half
            if verbose:
                _log.info(
                    "%s",
                    f"[DiDAE Sweep] Widened the empirical component bounds "
                    f"by x{component_bounds_scale} about their midpoint.",
                )
        return c_mins, c_maxs

    def _dictionary_active_mask(self, z_sem, c_factual):
        """Which concepts are ON for each sample, as [N, K] bool.

        Uses the dictionary's own sparse code where it has one -- MSAE is
        TopKReLU-64, RA-SAE a thresholded ReLU -- so "active" means what the SAE
        itself means by it, not merely a positive projection onto a decoder atom.
        Falls back to `c_factual > 0` for dictionaries with no encoder.
        """
        sd = self.sparse_dictionary
        if hasattr(sd, "encode"):
            try:
                with torch.no_grad():
                    codes = sd.encode(z_sem)
                if codes is not None and codes.shape[-1] == c_factual.shape[-1]:
                    return codes.to(device=c_factual.device) > 0
                _log.info(
                    "%s",
                    f"[DiDAE Replacement] sparse code has {tuple(codes.shape)} "
                    f"columns, dictionary has {c_factual.shape[-1]}; falling back "
                    "to c_factual > 0 for the active set.",
                )
            except Exception as exc:  # noqa: BLE001 - fall back, never abort
                _log.info(
                    "%s",
                    f"[DiDAE Replacement] sparse encode unavailable ({exc}); "
                    "falling back to c_factual > 0 for the active set.",
                )
        return c_factual > 0

    def sweep_concept_replacement(
        self,
        z_sem,
        w,
        bias,
        margins,
        original_signs,
        W,
        get_dim_name,
        n_candidates=64,
        component_bounds_scale=1.0,
        output_dir=None,
        top_k_report=10,
        unique_assignment=False,
        unique_strategy="greedy",
        return_state=False,
    ):
        """Rank *pairs* of concepts (turn one off, turn another on).

        The single-direction sweep asks one atom to carry the whole margin, and
        then clamps its activation to the empirical [c_min, c_max] range. On
        NICO/MSAE that clamp bound 100% of the steps and left a median 2.5% of
        the requested step, so the sweep measured the clamp more than it measured
        the dictionary. This mode changes the question: rather than "how far can
        I push atom k before the clamp stops me", it asks "which concept, removed
        entirely, and which other concept, added at full strength, together flip
        the class". Both targets are the bounds themselves, so the edit is
        maximal-but-in-distribution by construction and the clamp cannot cut it
        short -- the budget is two atoms' full range instead of a fraction of one.

        Off means the atom's projection is driven to c_min, on means c_max, and a
        pair is only considered for a sample where the off-concept is currently
        active and the on-concept currently inactive (the dictionary's own sparse
        code decides that, see _dictionary_active_mask).

        The two atoms are not orthogonal, so moving along one changes the other's
        projection. The pair is therefore solved jointly: find a, b with

            z_cf = z - a*u_off - b*u_on,   z_cf . u_off = c_min,  z_cf . u_on = c_max

        which is a 2x2 system per (sample, pair) whose matrix depends only on the
        pair. The margin then moves by exactly -a*(u_off.w) - b*(u_on.w).

        With unique_assignment, the ranking is restricted so that each concept
        may be deactivated at most once and activated at most once. Without it
        the ranking repeats its strongest concepts -- on NICO/MSAE seven of the
        top ten all deactivate #2145 crocodile -- which reads as ten findings but
        is closer to two.

        unique_strategy picks how:
          "greedy"   walk the ranking by flip count and keep a pair whenever
                     neither of its concepts is spoken for. The best pair is
                     always kept, so the head of the list is the head of the
                     unconstrained ranking with repeats dropped, and each entry
                     is the best remaining claim given everything above it.
          "matching" the maximum-weight one-to-one matching over latent flips
                     (scipy.linear_sum_assignment, the routine the GT matching
                     uses). Maximises the total, which can cost the single
                     strongest pair: on NICO/MSAE it drops #1 (deactivate lizard
                     -> activate crocodile, 221/228) because spending those two
                     slots elsewhere adds more in total, 718 flips against 698.

        Greedy answers "what are the top replacements, deduplicated"; matching
        answers "what disjoint set explains the most flips". They are different
        questions and greedy is the default because it is usually the one being
        asked of a ranking.

        Returns the same result-dict shape sweep_all_directions returns, with
        direction_idx set to the activated concept and replacement_off_idx to the
        deactivated one.
        """
        device = z_sem.device
        dtype = z_sem.dtype
        N = z_sem.shape[0]
        K = W.shape[1]
        eps = 1e-6

        c_mins, c_maxs = self._load_component_bounds(
            K, device, dtype, component_bounds_scale
        )
        if c_mins is None:
            raise RuntimeError(
                "concept_replacement needs the empirical component bounds: 'off' "
                "is c_min and 'on' is c_max. c_min_and_maxes.txt has not been "
                "written for this dictionary yet."
            )
        c_mins, c_maxs = c_mins[:K], c_maxs[:K]

        c_factual = torch.matmul(z_sem, W)  # [N, K]
        u_norm_sq = torch.sum(W * W, dim=0)  # [K]
        u_norm_sq = torch.where(
            u_norm_sq.abs() < eps, torch.full_like(u_norm_sq, eps), u_norm_sq
        )
        dot_uw = torch.matmul(w, W)  # [K]

        active = self._dictionary_active_mask(z_sem, c_factual)  # [N, K]
        _log.info(
            "%s",
            f"[DiDAE Replacement] Sparse code: {active.sum(dim=1).float().mean():.1f} "
            f"of {K} concepts active per sample on average "
            f"({active.any(dim=0).sum().item()} distinct concepts ever active).",
        )

        # Single-atom step sizes to each bound, and the margin progress they buy.
        # progress = -sign(margin) * delta_margin, so a flip needs progress > |margin|.
        scale_off = (c_factual - c_mins.unsqueeze(0)) / u_norm_sq.unsqueeze(0)
        scale_on = (c_factual - c_maxs.unsqueeze(0)) / u_norm_sq.unsqueeze(0)
        sgn = original_signs.unsqueeze(1)
        prog_off = sgn * scale_off * dot_uw.unsqueeze(0)  # [N, K]
        prog_on = sgn * scale_on * dot_uw.unsqueeze(0)  # [N, K]

        abs_margin = margins.abs()
        neg_inf = torch.finfo(dtype).min

        # How many samples a *single* atom driven all the way to its bound can
        # flip. Same maximal-step rule as the pairs, so the comparison below
        # isolates what pairing adds rather than what the changed target adds.
        single_best = torch.maximum(
            torch.where(active, prog_off, torch.full_like(prog_off, neg_inf))
            .max(dim=1)
            .values,
            torch.where(~active, prog_on, torch.full_like(prog_on, neg_inf))
            .max(dim=1)
            .values,
        )
        single_cov = int((single_best > abs_margin).sum())

        # Ceiling on what pairing can reach over the *whole* dictionary, not just
        # the M x M grid. Under the orthogonal approximation the two terms are
        # independent, so the best pair for a sample is the best off-atom plus
        # the best on-atom and this is an O(N*K) max. It ignores the u_off . u_on
        # cross-term the grid solves exactly, so treat it as approximate -- it
        # exists to say how much the candidate shortlist is leaving on the table.
        best_off_any = (
            torch.where(active, prog_off, torch.full_like(prog_off, neg_inf))
            .max(dim=1)
            .values
        )
        best_on_any = (
            torch.where(~active, prog_on, torch.full_like(prog_on, neg_inf))
            .max(dim=1)
            .values
        )
        pair_ceiling_cov = int(((best_off_any + best_on_any) > abs_margin).sum())

        # --- Candidate atoms -------------------------------------------------
        # The best pair for a sample separates (the two terms are independent
        # under the orthogonal approximation), so mean progress over the samples
        # where an atom is eligible is a sound way to shortlist. The grid below
        # then scores the shortlisted pairs exactly.
        off_elig = active.sum(dim=0)  # [K]
        on_elig = (~active).sum(dim=0)  # [K]
        off_score = torch.where(active, prog_off, torch.zeros_like(prog_off)).sum(
            dim=0
        ) / off_elig.clamp(min=1)
        on_score = torch.where(~active, prog_on, torch.zeros_like(prog_on)).sum(
            dim=0
        ) / on_elig.clamp(min=1)
        off_score = torch.where(
            off_elig > 0, off_score, torch.full_like(off_score, neg_inf)
        )
        on_score = torch.where(
            on_elig > 0, on_score, torch.full_like(on_score, neg_inf)
        )

        M = int(min(int(n_candidates), K))
        off_idx = torch.topk(off_score, M).indices  # [M]
        on_idx = torch.topk(on_score, M).indices  # [M]

        # --- Exact joint solve over the M x M grid ---------------------------
        U_off = W[:, off_idx]  # [D, M]
        U_on = W[:, on_idx]  # [D, M]
        a_ii = u_norm_sq[off_idx]  # [M]
        b_jj = u_norm_sq[on_idx]  # [M]
        G = torch.matmul(U_off.transpose(0, 1), U_on)  # [M, M]
        det = a_ii.unsqueeze(1) * b_jj.unsqueeze(0) - G * G  # [M, M]
        # Near-parallel atoms: the pair cannot hit both targets, drop it.
        solvable = det.abs() > eps
        det_safe = torch.where(solvable, det, torch.ones_like(det))

        r_off = (c_factual[:, off_idx] - c_mins[off_idx].unsqueeze(0)).unsqueeze(
            2
        )  # [N,M,1]
        r_on = (c_factual[:, on_idx] - c_maxs[on_idx].unsqueeze(0)).unsqueeze(
            1
        )  # [N,1,M]

        a = (r_off * b_jj.view(1, 1, M) - r_on * G.unsqueeze(0)) / det_safe.unsqueeze(0)
        b = (r_on * a_ii.view(1, M, 1) - r_off * G.unsqueeze(0)) / det_safe.unsqueeze(0)

        new_margin = (
            margins.view(N, 1, 1)
            - a * dot_uw[off_idx].view(1, M, 1)
            - b * dot_uw[on_idx].view(1, 1, M)
        )  # [N, M, M]
        # How far past the boundary each pair landed, the quantity
        # decode_selection="deepest" ranks on.
        pair_depth = -original_signs.view(N, 1, 1) * new_margin  # [N, M, M]

        eligible = active[:, off_idx].unsqueeze(2) & (~active[:, on_idx]).unsqueeze(1)
        eligible = eligible & solvable.unsqueeze(0)
        flips = (torch.sign(new_margin) != original_signs.view(N, 1, 1)) & eligible

        pair_flips = flips.sum(dim=0)  # [M, M]
        pair_attempts = eligible.sum(dim=0)  # [M, M]
        total_pair_flips = int(pair_flips.sum())
        pair_cov = int(flips.any(dim=2).any(dim=1).sum())

        _log.info(
            "%s",
            f"[DiDAE Replacement] {M} off-candidates x {M} on-candidates = "
            f"{int(solvable.sum())} solvable pairs over {N} samples.",
        )
        _log.info(
            "%s",
            f"[DiDAE Replacement] {total_pair_flips} total latent flips across "
            f"(sample, pair); {int((pair_flips > 0).sum())} pairs flipped at least one.",
        )
        _log.info(
            "%s",
            f"[DiDAE Replacement] Sample coverage: {pair_cov}/{N} samples flippable "
            f"by some replacement in the {M}x{M} grid (exact); {single_cov}/{N} by "
            f"the best single atom over all {K} driven to its own bound (exact); "
            f"{pair_ceiling_cov}/{N} by the best pair over all {K}x{K} "
            "(orthogonal approximation, ignores the cross-term).",
        )

        if total_pair_flips == 0:
            _log.info(
                "%s",
                "[DiDAE Replacement] No pair flipped. Raise "
                "concept_replacement_candidates, or widen the bounds with "
                "component_bounds_scale.",
            )
            return ([], None) if return_state else []

        # --- Rank and report --------------------------------------------------
        flat = torch.argsort(pair_flips.flatten(), descending=True)
        ranked = []
        for pos in flat.tolist():
            i, j = divmod(pos, M)
            if int(pair_flips[i, j]) <= 0:
                break
            ranked.append((i, j))

        if not unique_assignment:
            selected = ranked
        elif str(unique_strategy).lower() == "matching":
            # Maximum-weight one-to-one matching: the disjoint set that explains
            # the most flips, which need not contain the single best pair.
            from scipy.optimize import linear_sum_assignment

            weights = pair_flips.detach().cpu().numpy().astype(float)
            rows, cols = linear_sum_assignment(-weights)
            selected = [
                (int(i), int(j)) for i, j in zip(rows, cols) if weights[i, j] > 0
            ]
            selected.sort(key=lambda ij: -weights[ij[0], ij[1]])
        else:
            # Greedy: the unconstrained ranking with repeats dropped. Keeps the
            # best pair, and every entry is the best remaining claim given the
            # concepts already spent above it.
            used_off, used_on = set(), set()
            selected = []
            for i, j in ranked:
                if i in used_off or j in used_on:
                    continue
                used_off.add(i)
                used_on.add(j)
                selected.append((i, j))

        if unique_assignment:
            covered = int(sum(int(pair_flips[i, j]) for i, j in selected))
            _log.info(
                "%s",
                f"[DiDAE Replacement] Unique assignment ({unique_strategy}): "
                f"{len(selected)} disjoint replacements (each concept deactivated "
                f"at most once and activated at most once), covering {covered} of "
                f"{total_pair_flips} latent flips.",
            )

        if selected:
            sel_i = torch.tensor([i for i, _ in selected], device=device)
            sel_j = torch.tensor([j for _, j in selected], device=device)
            selected_cov = int(flips[:, sel_i, sel_j].any(dim=1).sum())
            _log.info(
                "%s",
                f"[DiDAE Replacement] The reported replacements flip "
                f"{selected_cov}/{N} samples between them.",
            )

        results = []
        lines = []
        for i, j in selected:
            count = int(pair_flips[i, j])
            k_off = int(off_idx[i])
            k_on = int(on_idx[j])
            name_off = get_dim_name(k_off)
            name_on = get_dim_name(k_on)
            results.append(
                {
                    "direction_idx": k_on,
                    "replacement_off_idx": k_off,
                    "dimension_name": f"OFF {name_off}  ->  ON {name_on}",
                    "replacement_off_name": name_off,
                    "replacement_on_name": name_on,
                    "success_count": None,
                    "ambient_flip_count": None,
                    "latent_flip_count": count,
                    "total_attempted": int(pair_attempts[i, j]),
                    "increase_flips": None,
                    "increase_attempts": None,
                    "decrease_flips": None,
                    "decrease_attempts": None,
                    "pairs": [],
                }
            )
            lines.append(
                f"OFF #{k_off} ({name_off})\t->\tON #{k_on} ({name_on})\t"
                f"{count}/{int(pair_attempts[i, j])} latent flips"
            )

        _log.info(
            "%s",
            f"[DiDAE Replacement] Top {min(top_k_report, len(results))} concept "
            "replacements ranked by LATENT-SPACE flips:",
        )
        for rank, r in enumerate(results[:top_k_report], start=1):
            _log.info(
                "%s",
                f"  #{rank}: deactivate {r['replacement_off_name']}  ->  "
                f"activate {r['replacement_on_name']} — "
                f"{r['latent_flip_count']}/{r['total_attempted']} eligible samples flipped",
            )

        if return_state:
            # Columns for the decode half: one per reported replacement, in the
            # same order as `results`. The decode block indexes everything by
            # column, so a pair is a drop-in for a direction there -- except for
            # z_cf, which needs both atoms and both coefficients, and the
            # specificity check, which has two target concepts instead of one.
            sel_i = torch.tensor([i for i, _ in selected], device=device)
            sel_j = torch.tensor([j for _, j in selected], device=device)
            state = {
                "flips": flips[:, sel_i, sel_j],  # [N, P]
                "depth": pair_depth[:, sel_i, sel_j],  # [N, P]
                "counts": pair_flips[sel_i, sel_j].cpu(),  # [P]
                "a": a[:, sel_i, sel_j],  # [N, P]
                "b": b[:, sel_i, sel_j],  # [N, P]
                "col_off": off_idx[sel_i],  # [P] dictionary atom indices
                "col_on": on_idx[sel_j],  # [P]
                "names": [r["dimension_name"] for r in results],
            }
        else:
            state = None

        if output_dir:
            out_path = os.path.join(output_dir, "concept_replacement_pairs.txt")
            with open(out_path, "w") as fh:
                fh.write(
                    f"# concept replacement sweep: {N} samples, {K} concepts, "
                    f"{M}x{M} candidate grid\n"
                )
                fh.write(
                    f"# sample coverage: {pair_cov}/{N} grid pairs (exact), "
                    f"{single_cov}/{N} best single atom (exact), "
                    f"{pair_ceiling_cov}/{N} best pair over all concepts "
                    "(orthogonal approximation)\n"
                )
                fh.write("\n".join(lines) + "\n")
            _log.info("%s", f"[DiDAE Replacement] Wrote the pair ranking to {out_path}")

        if return_state:
            return results, state
        return results

    def _invert_for_decoding(self, x_samples, sample_idx, batch_size):
        """Invert only the samples that will be rendered.

        Parameters
        ----------
        x_samples : torch.Tensor
            The whole sweep pool, ``[N, C, H, W]``.
        sample_idx : torch.Tensor
            Pool index of every flip to decode, ``[M]`` (repeats allowed).
        batch_size : int
            Inversion chunk size.

        Returns
        -------
        tuple
            ``(xT, zs, row)``: x_T and the noise maps (``None`` for DDIM) of the
            unique samples, and ``row[i]``, the row of pool sample ``i`` in them
            (``-1`` for samples that are not decoded).
        """
        device = sample_idx.device
        unique = torch.unique(sample_idx)
        row = torch.full((x_samples.shape[0],), -1, dtype=torch.long, device=device)
        row[unique] = torch.arange(unique.numel(), device=device)
        if unique.numel() == 0:
            return None, None, row
        xT_parts, zs_parts = [], []
        base_seed = int(getattr(self.config, "seed", 0) or 0)
        unique_cpu = unique.cpu()
        with torch.no_grad():
            for start in range(0, unique_cpu.numel(), batch_size):
                x_chunk = x_samples[unique_cpu[start : start + batch_size]].to(
                    self.device
                )
                # The DDPM inversion draws noise; seed it per chunk so a run is
                # reproducible whatever else consumed the global RNG before.
                cuda_devices = [self.device] if self.device.type == "cuda" else []
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(base_seed + start)
                    if self.sample_type == "ddpm_inv":
                        _, xT_chunk, zs_chunk = self.encode(
                            x_chunk, sample_type=self.sample_type
                        )
                        zs_parts.append(zs_chunk)
                    else:
                        _, xT_chunk = self.encode(x_chunk, sample_type=self.sample_type)
                xT_parts.append(xT_chunk)
        xT = torch.cat(xT_parts, dim=0)
        zs = _zs_cat(zs_parts) if zs_parts else None
        return xT, zs, row

    def sweep_all_directions(
        self,
        x_samples: torch.Tensor,
        w: torch.Tensor,
        predictor,
        linesearch_factors=None,
        decode_batch_size: int = 32,
        max_cf_per_direction: int = 5,
        y_samples=None,
        attribute_names=None,
        output_dir=None,
        precomputed_matched_results=None,
        explainer_config=None,
        latent_only=False,
        bias: float = 0.0,
        max_decode_directions=None,
        max_decode_per_direction=None,
        decode_selection: str = "random",
        max_export_per_direction=20,
        component_bounds_scale: float = 1.0,
        concept_replacement: bool = False,
        concept_replacement_candidates: int = 64,
        concept_replacement_unique: bool = False,
        concept_replacement_unique_strategy: str = "greedy",
        edit_depth_factor=None,
        sweep_atom_subset=None,
    ):
        """
        Sweep ALL sparse dictionary directions for a batch of samples.

        Steps:
          2) Compute latent counterfactuals for every (sample, direction) pair.
          3) Filter out pairs that don't flip the linear classifier in latent space.
          4) Decode surviving latent counterfactuals to images.
          5) Filter out decoded images that don't flip the original predictor.
          6) Rank directions by number of successful ambient-space flips.

        Args:
            x_samples: [N, C, H, W] input images in generator normalization.
            w: [Dim] classifier weight vector (from distilled linear probe).
            predictor: the user's original classifier (callable).
            bias: intercept of the distilled probe. The probe is fitted as
                w^T z + bias ~ margin, so its decision boundary is
                w^T z = -bias, not w^T z = 0. Leaving it out tests the wrong
                boundary: on CelebA/ViT-L14 the fitted bias is -34 while every
                sample's w^T z is positive, so `sign(w^T z)` is +1 for all of
                them and no reachable step along any atom flips it -- the sweep
                reported 0 latent flips out of 1000 x 1536. Default 0.0 keeps
                the old behaviour for callers that pass a bias-free probe.
            linesearch_factors: list of factors, defaults to ["dynamic"].
            decode_batch_size: how many latent CFs to decode at once.
            max_cf_per_direction: max representative CFs to store per direction.
            precomputed_matched_results: precomputed bipartite GT matching results list.
            latent_only: stop after step 3 and rank by latent flips alone. Steps
                4-6 need the trained decoder (DDIM/DDPM inversion to get xT, then
                rendering), so this is the half of the sweep that is meaningful
                before the diffusion autoencoder has been trained. The returned
                dicts carry latent_flip_count but no pairs, and success_count /
                ambient_flip_count are reported as None rather than 0, so a
                consumer cannot mistake "not measured" for "measured zero".
            max_decode_directions: decode only the top-N directions by latent
                flips. None decodes every direction that flipped at all, which
                on CelebA/ViT-L14 is 1463 directions and 35841 candidates -- a
                20-step DDPM decode each, plus a re-encode. Only
                top_k_directions of them are ever shown to the teacher, so the
                rest is work whose result is discarded.
            max_decode_per_direction: decode at most this many of a direction's
                latent flips, sampled without replacement under a fixed seed.
                The verified/ambient rates are then estimated from the subsample
                and scaled back up to the direction's full latent flip count for
                ranking, so a direction is not penalised for having been
                subsampled. None decodes all of them.
            max_export_per_direction: how many verified flips per direction are
                written to ``<output_dir>/successful_flips`` (at least
                max_cf_per_direction); None writes all of them. The folder's
                ``index.json`` lists the written pairs per direction.
            decode_selection: how max_decode_per_direction picks which of a
                direction's latent flips to decode. "random" is an unbiased
                sample of that direction's ambient flip rate; "deepest" takes
                the flips that landed furthest past the boundary, which is the
                setting to use when the run is after successful counterfactuals
                rather than an unbiased rate -- the reported rates, and
                expected_verified_flips with them, are then an upper bound.
            concept_replacement: rank *pairs* of concepts -- deactivate one
                concept that is currently active and activate one that is
                currently inactive -- instead of stepping a single atom toward
                the boundary. Both targets are the empirical bounds themselves,
                so the clamp that limits the single-atom step (100% of steps on
                NICO/MSAE, median 2.5% of the requested step surviving) does not
                apply. See sweep_concept_replacement.
            concept_replacement_candidates: how many off- and on-candidates the
                pair grid considers, M. The sweep scores M x M pairs exactly;
                cost is O(N * M^2), so 64 is cheap and 256 is still small.
            concept_replacement_unique: report a one-to-one selection -- each
                concept deactivated at most once and activated at most once --
                instead of the raw ranking, which repeats its strongest concepts
                across most of the top entries.
            concept_replacement_unique_strategy: "greedy" (the ranking with
                repeats dropped, keeps the best pair) or "matching" (maximum
                total flips, may drop it). See sweep_concept_replacement.
            component_bounds_scale: widen (>1) or tighten (<1) the empirical
                [c_min, c_max] clamp about its midpoint. The clamp is what
                actually limits the edit -- on CelebA/ViT-L14 98% of steps hit
                it and a median 2% of the requested step survives -- and the
                bounds themselves are a min/max over ~960 samples, which
                understates the range. 1.0 uses the bounds as written.

        Returns:
        """
        if linesearch_factors is None:
            linesearch_factors = ["dynamic"]

        sampler = (
            explainer_config.get("sampler")
            if isinstance(explainer_config, dict)
            else getattr(explainer_config, "sampler", None)
        )

        self.set_sampler(sampler)

        device = self.device
        N = x_samples.shape[0]

        # --- Encode all samples (semantic part only) ---
        # The latent sweep needs only z_sem. The inversion (x_T and, for DDPM,
        # the per-step noise maps) feeds rendering alone, and only the samples
        # of the flips selected for decoding are ever rendered, so it runs after
        # that selection (see _invert_for_decoding). Inverting the whole pool up
        # front spent ~80 % of a web job's time on samples that were never
        # decoded and held ~100 MB of noise maps per sample.
        with torch.no_grad():
            z_sem = torch.cat(
                [
                    self.encode(
                        x_samples[start : start + decode_batch_size].to(device),
                        only_semantic=True,
                    )
                    for start in range(0, N, decode_batch_size)
                ],
                dim=0,
            )

        # Ensure w has matching device and dtype
        w = w.to(device=device, dtype=z_sem.dtype)

        # --- Get all components ---
        W_all = self.sparse_dictionary.get_components()  # [Dim, K]
        K = W_all.shape[1]
        W = W_all.to(device=device, dtype=z_sem.dtype)

        # Automatically align W dimensions to match encoder latent dimension (w.shape[0])
        D_encoder = w.shape[0]
        if W.shape[0] != D_encoder:
            if W.shape[0] < D_encoder:
                W = torch.nn.functional.pad(W, (0, 0, 0, D_encoder - W.shape[0]))
            else:
                W = W[:D_encoder, :]

        # --- Compute original classifier scores ---
        # The probe predicts the student's margin as w^T z + bias, so the class
        # is sign(w^T z + bias) and the boundary sits at w^T z = -bias.
        bias = float(bias)
        z_dot_w = torch.matmul(z_sem, w)  # [N]
        margins = z_dot_w + bias  # [N]
        original_signs = torch.sign(margins)  # [N]

        n_pos = int((margins > 0).sum())
        _log.info(
            "%s",
            f"[DiDAE Sweep] Probe margins (w^T z + bias, bias={bias:.4f}): "
            f"{n_pos}/{N} positive, "
            f"mean {margins.mean().item():+.4f}, "
            f"|margin| median {margins.abs().median().item():.4f}",
        )
        if n_pos in (0, N):
            _log.info(
                "%s",
                "[DiDAE Sweep] WARNING: the probe assigns every sweep sample to "
                "the same class. Every flip has to cross the boundary from one "
                "side, which is the hardest direction for the clamped step.",
            )

        vocabulary = None
        if hasattr(self.sparse_dictionary, "get_vocabulary"):
            try:
                vocabulary = self.sparse_dictionary.get_vocabulary()
            except Exception:
                vocabulary = None
        elif hasattr(self.sparse_dictionary, "vocab"):
            vocabulary = self.sparse_dictionary.vocab
        elif hasattr(self.sparse_dictionary, "concept_names"):
            vocabulary = self.sparse_dictionary.concept_names
        elif hasattr(self.sparse_dictionary, "component_names"):
            vocabulary = self.sparse_dictionary.component_names

        # --- Bipartite Ground-Truth vs. SAE Matching ---
        c_sae = torch.matmul(z_sem, W)  # [N, K]
        sae_to_gt_map = {}
        matched_results = precomputed_matched_results

        if matched_results is None and y_samples is not None:
            matched_results = self.compute_bipartite_gt_sae_matching(
                y_samples=y_samples,
                c_sae=c_sae,
                attribute_names=attribute_names,
                vocabulary=vocabulary,
                output_dir=output_dir,
            )

        if matched_results:
            for m in matched_results:
                sae_to_gt_map[m["sae_idx"]] = (m["attribute_name"], m["f1_score"])

            _log.info(
                "%s",
                "\n================================================================================",
            )
            _log.info(
                "%s",
                "[DiDAE Bipartite Matching] Ground-Truth Attributes vs. SAE Predictions:",
            )
            _log.info(
                "%s",
                "================================================================================",
            )
            for m in matched_results:
                _log.info(
                    "%s",
                    f"Attribute '{m['attribute_name']}'  <-->  Matched SAE Feature #{m['sae_idx']} ({m['sae_dim_name']}):",
                )
                _log.info("%s", f"  • Precision: {m['precision']:.4f}")
                _log.info("%s", f"  • Recall   : {m['recall']:.4f}")
                _log.info("%s", f"  • F1 Score : {m['f1_score']:.4f}")
                _log.info("%s", f"  • Pearson r: {m['correlation']:+.4f}")
            _log.info(
                "%s",
                "================================================================ algorithm end\n",
            )

        def get_dim_name(idx):
            """Readable name for an SAE atom: vocabulary, matched GT and index."""
            vocab_name = (
                vocabulary[idx] if (vocabulary and 0 <= idx < len(vocabulary)) else None
            )
            gt_info = sae_to_gt_map.get(idx, None)

            if vocab_name and gt_info:
                gt_attr, f1 = gt_info
                return f"{vocab_name} [GT: {gt_attr}, F1={f1:.2f}] (SAE #{idx})"
            elif vocab_name:
                return f"{vocab_name} (SAE #{idx})"
            elif gt_info:
                gt_attr, f1 = gt_info
                return f"{gt_attr} [F1={f1:.2f}] (SAE #{idx})"
            else:
                return f"dim_{idx}"

        # A replacement edit moves along two atoms at once, so it cannot be
        # expressed as a single (sample, direction) step. The decode half below
        # is shared all the same: it indexes everything by *column*, and a column
        # is a pair here instead of an atom. Only two places care which --
        # rebuilding z_cf, which needs both atoms and both coefficients, and the
        # specificity check, which has two target concepts. pair_state carries
        # what those need.
        pair_state = None
        if concept_replacement:
            replacement_results, pair_state = self.sweep_concept_replacement(
                z_sem=z_sem,
                w=w,
                bias=bias,
                margins=margins,
                original_signs=original_signs,
                W=W,
                get_dim_name=get_dim_name,
                n_candidates=concept_replacement_candidates,
                component_bounds_scale=component_bounds_scale,
                output_dir=output_dir,
                unique_assignment=concept_replacement_unique,
                unique_strategy=concept_replacement_unique_strategy,
                return_state=True,
            )
            if latent_only or not replacement_results:
                return replacement_results

            # Re-key the sweep onto pair columns and fall through to steps 4-7.
            pair_names = pair_state["names"]
            K = len(pair_names)
            best_flips_latent = pair_state["flips"]
            best_depth = pair_state["depth"]
            latent_flip_counts = pair_state["counts"]
            best_scale = None  # z_cf comes from (a, b), not a single scale
            increase_successes = increase_attempts = None
            decrease_successes = decrease_attempts = None
            total_latent_flips = int(best_flips_latent.sum())
            top_latent_indices = torch.argsort(latent_flip_counts, descending=True)

            def get_dim_name(idx, _names=pair_names):  # noqa: F811 - pair columns
                """Column name of a concept-replacement pair."""
                return _names[int(idx)]

        # --- Compute dot products for all directions ---
        dot_uw = torch.matmul(w, W)  # [K]
        eps = 1e-6
        dot_uw_safe = dot_uw.clone()
        dot_uw_safe[torch.abs(dot_uw_safe) < eps] = eps

        # proj_factors[n, k] = (z_n . w + bias) / (u_k . w)
        # Stepping z by -proj_factors * u_k takes the margin exactly to zero, so
        # a linesearch factor > 1 crosses the boundary. Using z_dot_w instead of
        # the margin here aims at w^T z = 0, which is a different point whenever
        # bias != 0.
        proj_factors = margins.unsqueeze(1) / dot_uw_safe.unsqueeze(0)  # [N, K]

        # --- Compute latent counterfactuals for each linesearch factor ---
        # We'll collect the best factor per (sample, direction)
        c_min_max_loaded = False
        all_c_mins = None
        all_c_maxs = None

        # Track the step size along each atom rather than the counterfactual it
        # produces. best_z_cf was [N, K, Dim] -- 4.5 GB at N=1000, K=1536,
        # Dim=768, and `z_sem.unsqueeze(1) - step` needs two more of those live at
        # once, so the sweep peaked near 14 GB before decoding a single image.
        # z_cf is recoverable from the scale exactly (z - scale * u_k), and only
        # the handful of pairs that actually get decoded need it.
        if pair_state is None:
            # In concept-replacement mode these three were already re-keyed
            # onto pair columns above; resetting them here zeroed the pair
            # flips and dropped best_depth, so the decode step below always
            # saw "0 counterfactual candidates after filtering".
            best_scale = None  # [N, K]
            best_depth = None  # [N, K] how far past the boundary the flip landed
            best_flips_latent = torch.zeros(N, K, dtype=torch.bool, device=device)

        for f in [] if pair_state is not None else linesearch_factors:
            if f == "dynamic" and not c_min_max_loaded:
                all_c_mins, all_c_maxs = self._load_component_bounds(
                    K, device, z_sem.dtype, component_bounds_scale
                )
                c_min_max_loaded = all_c_mins is not None

            factor = 1.2 if f == "dynamic" else float(f)
            c_factual = torch.matmul(z_sem, W)  # [N, K]
            u_norm_sq = torch.sum(W * W, dim=0).unsqueeze(0)  # [1, K]

            # Desired activation change to cross decision boundary with margin factor f
            delta_c = factor * proj_factors * u_norm_sq  # [N, K]
            c_target_desired = c_factual - delta_c  # [N, K]

            if all_c_mins is not None and all_c_maxs is not None:
                # Clamp target activation to empirical range [c_min, c_max]
                c_target = torch.clamp(
                    c_target_desired,
                    min=all_c_mins[:K].unsqueeze(0),
                    max=all_c_maxs[:K].unsqueeze(0),
                )
            else:
                c_target = c_target_desired

            # scale[n, k] is how far along atom u_k the counterfactual moves.
            scale = (c_factual - c_target) / u_norm_sq  # [N, K]

            # The clamp to the empirical [c_min, c_max] range is the second thing
            # that stops a flip, after the boundary itself: if the activation the
            # step asks for is outside anything seen in the data, the step is cut
            # short and the margin never reaches zero. Report how often that bites,
            # so "0 latent flips" can be told apart from "the clamp ate every step".
            if all_c_mins is not None and all_c_maxs is not None:
                requested = (c_factual - c_target_desired) / u_norm_sq
                denom = torch.where(
                    requested.abs() < eps,
                    torch.full_like(requested, eps),
                    requested,
                )
                fulfilment = (scale / denom).clamp(min=0.0, max=1.0)
                clamped_frac = (c_target != c_target_desired).float().mean().item()
                _log.info(
                    "%s",
                    f"[DiDAE Sweep] factor={factor}: {100 * clamped_frac:.1f}% of "
                    f"(sample, direction) steps hit the empirical [c_min, c_max] "
                    f"clamp; median fraction of the requested step that survived: "
                    f"{fulfilment.median().item():.3f}",
                )

            # The flip test is closed-form and never needs z_cf itself:
            #   z_cf[n, k] . w = z[n] . w - scale[n, k] * (u_k . w)
            # which is [N, K], the same size as everything else in the loop.
            # Materialising z_cf here cost [N, K, Dim] -- 98 GB at N=1000 with a
            # 32000-atom dictionary -- for a number this line computes directly.
            z_cf_dot_w = z_dot_w.unsqueeze(1) - scale * dot_uw.unsqueeze(0)

            cf_margins = z_cf_dot_w + bias
            cf_signs = torch.sign(cf_margins)
            flips = cf_signs != original_signs.unsqueeze(1)  # [N, K]

            # Track increase (toward c_max) vs decrease (toward c_min) attempts and successes
            is_increase = c_target_desired > c_factual
            is_decrease = c_target_desired < c_factual

            increase_attempts = is_increase.sum(dim=0).cpu()
            decrease_attempts = is_decrease.sum(dim=0).cpu()

            increase_successes = (flips & is_increase).sum(dim=0).cpu()
            decrease_successes = (flips & is_decrease).sum(dim=0).cpu()

            # How far past the boundary this step landed: positive exactly when
            # the sign flipped, and larger the more decisively it did.
            depth = -original_signs.unsqueeze(1) * cf_margins  # [N, K]

            # Keep the deepest crossing, not the first one. The clamp leaves a
            # median 2% of the requested step, so a flip found at factor 1.2 is
            # typically a margin of ~0 -- it survives the latent sign test and
            # then loses the sign again to decode/re-encode noise. A larger
            # factor, where the clamp still allows it, lands the same edit
            # further inside the other class.
            if best_scale is None:
                best_scale = scale
                best_depth = depth
                best_flips_latent = flips
            else:
                new_better = flips & (~best_flips_latent | (depth > best_depth))
                best_scale = torch.where(new_better, scale, best_scale)
                best_depth = torch.where(new_better, depth, best_depth)
                best_flips_latent = best_flips_latent | flips

        # --- Step 3: Filter by latent flips ---
        if pair_state is None:
            latent_flip_counts = best_flips_latent.sum(dim=0).cpu()  # [K]
            total_latent_flips = (best_flips_latent > 0).sum().item()

        if pair_state is None:
            _log.info(
                "%s",
                f"[DiDAE Sweep] {total_latent_flips} total latent flips across {N} samples × {K} directions",
            )
            top_latent_indices = torch.argsort(latent_flip_counts, descending=True)
        top_k_latent_show = (
            0
            if pair_state is not None
            else min(10, (latent_flip_counts > 0).sum().item())
        )
        if top_k_latent_show > 0:
            _log.info(
                "%s",
                f"[DiDAE Sweep] Top {top_k_latent_show} directions ranked by LATENT-SPACE flips:",
            )
            for i in range(top_k_latent_show):
                d_idx = int(top_latent_indices[i])
                l_count = int(latent_flip_counts[d_idx])
                dim_name = get_dim_name(d_idx)
                inc_succ = int(increase_successes[d_idx])
                inc_att = int(increase_attempts[d_idx])
                dec_succ = int(decrease_successes[d_idx])
                dec_att = int(decrease_attempts[d_idx])
                _log.info(
                    "%s",
                    f"  #{i+1}: Index {d_idx} ({dim_name}) — Total: {l_count}/{N} latent flips",
                )
                _log.info(
                    "%s",
                    f"      • Setting toward c_max (Increase): {inc_succ}/{inc_att} successful attempts",
                )
                _log.info(
                    "%s",
                    f"      • Setting toward c_min (Decrease): {dec_succ}/{dec_att} successful attempts",
                )

        if total_latent_flips == 0:
            _log.info(
                "%s", "[DiDAE Sweep] 0 latent flips recorded across all directions."
            )
            return []

        if latent_only:
            # Steps 4-6 need the decoder; rank on what the latent space alone
            # can say and hand that back.
            direction_results = [
                {
                    "direction_idx": d_idx,
                    "dimension_name": get_dim_name(d_idx),
                    "success_count": None,
                    "ambient_flip_count": None,
                    "latent_flip_count": int(latent_flip_counts[d_idx]),
                    "total_attempted": int(latent_flip_counts[d_idx]),
                    "increase_flips": int(increase_successes[d_idx]),
                    "increase_attempts": int(increase_attempts[d_idx]),
                    "decrease_flips": int(decrease_successes[d_idx]),
                    "decrease_attempts": int(decrease_attempts[d_idx]),
                    "pairs": [],
                }
                for d_idx in range(K)
                if int(latent_flip_counts[d_idx]) > 0
            ]
            direction_results.sort(key=lambda r: r["latent_flip_count"], reverse=True)
            _log.info(
                "%s",
                f"[DiDAE Sweep] latent_only: ranked {len(direction_results)} directions "
                "with at least one latent flip; decoding and ambient verification "
                "skipped.",
            )
            return direction_results

        # --- Decode budget ---
        # Every candidate below costs one 20-step DDPM decode plus a re-encode,
        # and only top_k_directions of them are ever shown to the teacher.
        # Without a budget this decodes all 35841 CelebA/ViT-L14 candidates to
        # rank 1463 directions and then throw 1453 of them away.
        candidate_dirs = [d for d in range(K) if int(latent_flip_counts[d]) > 0]
        if sweep_atom_subset is not None and pair_state is None:
            # Targeted single-atom run: render every latent flip of the listed atoms
            # only (not a decode budget -- nothing of a listed atom is skipped).
            subset = set(int(a) for a in sweep_atom_subset)
            candidate_dirs = [d for d in candidate_dirs if d in subset]
            _log.info(
                "%s",
                f"[DiDAE Sweep] sweep_atom_subset: rendering the {len(candidate_dirs)} of "
                f"{len(subset)} listed atoms that have latent flips ({sorted(candidate_dirs)}).",
            )
        if max_decode_directions and len(candidate_dirs) > int(max_decode_directions):
            candidate_dirs = [
                int(d)
                for d in top_latent_indices[: int(max_decode_directions)].tolist()
                if int(latent_flip_counts[d]) > 0
            ]
            _log.info(
                "%s",
                f"[DiDAE Sweep] Decoding the top {len(candidate_dirs)} of "
                f"{(latent_flip_counts > 0).sum().item()} flipping directions "
                f"(max_decode_directions={max_decode_directions}).",
            )

        per_dir_cap = (
            int(max_decode_per_direction)
            if max_decode_per_direction and int(max_decode_per_direction) > 0
            else None
        )
        subsample_gen = torch.Generator(device="cpu").manual_seed(0)

        filtered_flips = []
        direction_sampled = {}  # d_idx -> how many of its latent flips were decoded
        for d_idx in candidate_dirs:
            samples_for_d = (
                best_flips_latent[:, d_idx].nonzero(as_tuple=False)
            ).squeeze(-1)
            if samples_for_d.numel() == 0:
                continue
            if per_dir_cap is not None and samples_for_d.numel() > per_dir_cap:
                if decode_selection == "deepest":
                    # The flips that crossed furthest: the ones whose latent edit
                    # is largest, and so the ones most likely to still read as
                    # the other class after a decode and a re-encode.
                    order = torch.argsort(
                        best_depth[samples_for_d, d_idx], descending=True
                    )
                    samples_for_d = samples_for_d[order[:per_dir_cap]].cpu()
                else:
                    # Sample without replacement under a fixed seed: an unbiased
                    # estimate of this direction's ambient flip rate, reproducible
                    # across runs.
                    pick = torch.randperm(
                        samples_for_d.numel(), generator=subsample_gen
                    )[:per_dir_cap]
                    samples_for_d = samples_for_d.cpu()[pick]
            direction_sampled[d_idx] = int(samples_for_d.numel())
            for s_idx in samples_for_d:
                filtered_flips.append((int(s_idx), d_idx))

        if len(filtered_flips) == 0:
            _log.info(
                "%s", "[DiDAE Sweep] 0 counterfactual candidates after filtering."
            )
            return []

        _log.info(
            "%s",
            f"[DiDAE Sweep] Decoding {len(filtered_flips)} counterfactual candidates "
            f"across {len(direction_sampled)} directions "
            f"(of {total_latent_flips} latent flips in total).",
        )

        flip_indices = torch.tensor(filtered_flips, dtype=torch.long, device=device)

        # --- Steps 4-5: decode, classify and re-encode in ONE streaming pass ---
        # Every latent flip inside the empirical bounds is rendered and checked
        # for an ambient flip, exactly like the classical CFKD explainer renders
        # every counterfactual (2026-09-14). M can be 10^5 on a 6144-atom
        # dictionary, so nothing of size [M, C, H, W] or [M, K] is ever held:
        # each batch is decoded, run through the predictor, re-encoded and
        # scored, and only the images of verified flips are kept for collages.
        # Rebuild z_cf for the selected pairs only: z_cf = z - scale * u_k.
        # Edit depth. Default: the full edit (single atom driven to its bound, or the
        # joint replacement that puts the off-concept on c_min and the on-concept on
        # c_max) however far past the boundary that lands. With edit_depth_factor = s
        # every edit is shortened to s x the smallest step along the SAME direction
        # that crosses the distilled probe's boundary (the margin is linear in z), so
        # the rendered counterfactual is as small as the flip allows; s > 1 leaves
        # headroom for the decoder realising only part of the step (2026-09-16).
        s_sel = flip_indices[:, 0]
        p_sel = flip_indices[:, 1]
        depth_mult = torch.ones(flip_indices.shape[0], device=W.device)
        if edit_depth_factor is not None:
            w_dev = w.to(W.device)
            dot_uw_all = torch.matmul(w_dev, W)  # [K]
            m0 = (
                torch.matmul(z_sem[s_sel].to(W.device), w_dev) + bias
            )  # signed source margin
            if pair_state is None:
                delta = best_scale[s_sel, p_sel].to(W.device) * dot_uw_all[p_sel]
            else:
                delta = (
                    pair_state["a"][s_sel, p_sel].to(W.device)
                    * dot_uw_all[pair_state["col_off"][p_sel]]
                    + pair_state["b"][s_sel, p_sel].to(W.device)
                    * dot_uw_all[pair_state["col_on"][p_sel]]
                )
            t_min = (
                m0
                / torch.where(delta.abs() < 1e-8, torch.full_like(delta, 1e-8), delta)
            ).clamp(min=0.0, max=1.0)
            depth_mult = (float(edit_depth_factor) * t_min).clamp(max=1.0)
            _log.info(
                "%s",
                f"[DiDAE Sweep] edit_depth_factor={float(edit_depth_factor):.2f}: the boundary "
                f"sits at {t_min.mean().item():.2f} of the full edit on average; edits are "
                f"shortened to {depth_mult.mean().item():.2f} of the full edit (min "
                f"{depth_mult.min().item():.2f}, {(depth_mult < 1).float().mean().item()*100:.0f}% shortened).",
            )
        depth_mult = depth_mult.to(
            best_scale.device if pair_state is None else pair_state["a"].device
        )
        if pair_state is None:
            z_cf_surviving = (
                z_sem[s_sel]
                - (best_scale[s_sel, p_sel] * depth_mult).unsqueeze(-1) * W[:, p_sel].T
            )  # [M, Dim]
        else:
            # z_cf = z - a*u_off - b*u_on, with (a, b) the joint solve that puts
            # the off-concept on its c_min and the on-concept on its c_max.
            z_cf_surviving = (
                z_sem[s_sel]
                - (pair_state["a"][s_sel, p_sel] * depth_mult).unsqueeze(-1)
                * W[:, pair_state["col_off"][p_sel]].T
                - (pair_state["b"][s_sel, p_sel] * depth_mult).unsqueeze(-1)
                * W[:, pair_state["col_on"][p_sel]].T
            )  # [M, Dim]

        generator_to_classifier = (
            lambda x: self.predictor_dataset.project_from_pytorch_default(
                self.generator_dataset.project_to_pytorch_default(x)
            )
        )
        with torch.no_grad():
            orig_preds = torch.nn.functional.softmax(
                predictor(generator_to_classifier(x_samples).to(device)), dim=-1
            ).cpu()
        orig_classes = orig_preds.argmax(dim=-1)  # [N]
        sample_indices = flip_indices[:, 0].cpu()
        direction_indices = flip_indices[:, 1].cpu()
        # Use the SAE encoder if available for accurate feature activations
        use_sd_encoder = hasattr(self.sparse_dictionary, "encode")
        with torch.no_grad():
            c_orig_dev = (
                self.sparse_dictionary.encode(z_sem.to(device))
                if use_sd_encoder
                else c_sae.to(device)
            )  # [N, K]
        if pair_state is None:
            target_cols = flip_indices[:, 1].unsqueeze(1)  # [M, 1]
        else:
            # The intended change spans both concepts: the deactivated one
            # should fall and the activated one should rise, so both count
            # toward "the edit that was asked for" rather than toward the
            # off-target energy it is measured against.
            target_cols = torch.stack(
                [
                    pair_state["col_off"][flip_indices[:, 1]],
                    pair_state["col_on"][flip_indices[:, 1]],
                ],
                dim=1,
            ).to(
                device
            )  # [M, 2]
        M = flip_indices.shape[0]
        # Live progress for the web demo: the latent ranking is final here, the
        # ambient / verified counts fill in batch by batch.
        progress_path = (
            os.path.join(output_dir, "sweep_progress.json") if output_dir else None
        )
        progress = {
            "stage": "inverting",
            "n_candidates": int(M),
            "decoded": 0,
            "directions": sorted(
                (
                    {
                        "direction_idx": int(d),
                        "name": str(get_dim_name(d)),
                        "latent_flips": int(latent_flip_counts[d]),
                        "to_render": int(n),
                        "rendered": 0,
                        "ambient_flips": 0,
                        "verified_flips": 0,
                    }
                    for d, n in direction_sampled.items()
                ),
                key=lambda e: -e["latent_flips"],
            ),
        }
        progress_row = {e["direction_idx"]: e for e in progress["directions"]}

        def write_progress():
            if progress_path is None:
                return
            tmp = progress_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(progress, f)
            os.replace(tmp, progress_path)

        write_progress()
        xT, zs, inv_row = self._invert_for_decoding(
            x_samples, flip_indices[:, 0], decode_batch_size
        )
        progress["stage"] = "rendering"
        write_progress()
        cf_preds = torch.zeros(M, orig_preds.shape[1])
        ambient_flips = torch.zeros(M, dtype=torch.bool)
        edit_realisation = torch.zeros(M)
        target_activation_diff = torch.zeros(M)
        specificity_scores = torch.zeros(M)
        kept_images = {}  # m -> x_cf (cpu) for verified flips only
        z_sem_dev = z_sem.to(device)
        n_batches = (M + decode_batch_size - 1) // decode_batch_size
        with torch.no_grad():
            for b_i, start in enumerate(range(0, M, decode_batch_size)):
                end = min(start + decode_batch_size, M)
                s_b = flip_indices[start:end, 0]
                z_batch = z_cf_surviving[start:end]
                r_b = inv_row[s_b]
                xT_batch = xT[r_b]
                if self.sample_type == "ddim_inv":
                    x_cf_batch = self.decode(
                        (z_batch, xT_batch), sample_type=self.sample_type
                    )
                elif self.sample_type == "ddpm_inv":
                    zs_batch = _zs_map(zs, lambda value: value[r_b])
                    x_cf_batch = self.decode(
                        (z_batch, xT_batch, zs_batch), sample_type=self.sample_type
                    )
                # ambient check: the student on the rendered image
                preds = torch.nn.functional.softmax(
                    predictor(generator_to_classifier(x_cf_batch).to(device)), dim=-1
                ).cpu()
                cf_preds[start:end] = preds
                amb = preds.argmax(dim=-1) != orig_classes[s_b.cpu()]
                ambient_flips[start:end] = amb
                # re-encode: where did the edit really land?
                z_re = self.encode(x_cf_batch.to(device), only_semantic=True)
                requested = z_batch - z_sem_dev[s_b]
                achieved = z_re - z_sem_dev[s_b]
                edit_realisation[start:end] = (
                    (achieved * requested).sum(dim=-1)
                    / ((requested * requested).sum(dim=-1) + 1e-8)
                ).cpu()
                c_cf = (
                    self.sparse_dictionary.encode(z_re)
                    if use_sd_encoder
                    else torch.matmul(z_re, W)
                )  # [b, K]
                delta = (c_cf - c_orig_dev[s_b]).abs()
                d_target = delta.gather(1, target_cols[start:end].to(device)).sum(dim=1)
                d_total = delta.sum(dim=1)
                spec = d_target / (d_total + 1e-8)
                target_activation_diff[start:end] = d_target.cpu()
                specificity_scores[start:end] = spec.cpu()
                # Target feature counts as changed in the rendered image if the
                # real activation moved significantly (> 0.5) and specificity is
                # above the random noise floor (> 0.015).
                changed = ((d_target > 0.5) & (spec > 0.015)).cpu()
                for j in (amb & changed).nonzero(as_tuple=False).flatten().tolist():
                    kept_images[start + j] = x_cf_batch[j].detach().cpu().half()
                verified_b = (amb & changed).tolist()
                for j, d in enumerate(flip_indices[start:end, 1].tolist()):
                    row = progress_row[int(d)]
                    row["rendered"] += 1
                    row["ambient_flips"] += int(bool(amb[j]))
                    row["verified_flips"] += int(verified_b[j])
                progress["decoded"] = int(end)
                write_progress()
                _log.info(
                    "%s",
                    f"[DiDAE Sweep] decoded {end}/{M} candidates, "
                    f"{int(ambient_flips[:end].sum())} ambient flips so far",
                )
        # --- Did the decoder actually realise the edit? ---
        # The sweep asks for z_cf = z - scale*u_k and then decodes it. Re-encoding
        # the rendered image says where it really landed. The ratio
        #   <achieved, requested> / ||requested||^2
        # is ~1 when the decoder produced the edit that was asked for.
        #
        # What it measures on CelebA/ViT-L14: 0.10-0.25 in EVERY configuration
        # tried -- the decoder realises roughly a fifth of the requested latent
        # step, whatever the dictionary or the clamp width.
        #
        # What it does NOT do is detect an off-manifold edit: on an
        # unrecognisable image both the student and the oracle flip. There is
        # currently NO automated guard in this sweep against an edit that flips
        # the classifier by leaving the data manifold: read the collages in
        # direction_collages/ before trusting a flip rate.
        _log.info(
            "%s",
            f"[DiDAE Sweep] Edit realisation (re-encoded / requested latent step): "
            f"mean {edit_realisation.mean().item():.3f}, "
            f"median {edit_realisation.median().item():.3f}. "
            "Near 1 means the decoder produced the edit that was asked for; near 0 "
            "means the rendered image is not the requested counterfactual.",
        )
        target_feature_changed = (target_activation_diff > 0.5) & (
            specificity_scores > 0.015
        )
        verified_flips = ambient_flips & target_feature_changed
        _log.info(
            "%s",
            f"[DiDAE Sweep] {ambient_flips.sum().item()}/{M} rendered images flipped ambient predictor.",
        )
        _log.info(
            "%s",
            f"[DiDAE Sweep] Re-encoding SAE check: {verified_flips.sum().item()}/{M} rendered images flipped predictor AND specifically modified target SAE feature (Avg Specificity: {specificity_scores.mean().item():.4f}).",
        )
        # --- Step 6: Rank directions by verified success count ---
        direction_success = (
            {}
        )  # direction_idx -> list of (sample_idx, x_cf, confidence)
        direction_verified_count = {}
        direction_ambient_count = {}
        direction_total = {}

        for m in range(flip_indices.shape[0]):
            d_idx = int(direction_indices[m])
            if d_idx not in direction_total:
                direction_total[d_idx] = 0
                direction_ambient_count[d_idx] = 0
                direction_verified_count[d_idx] = 0
                direction_success[d_idx] = []
            direction_total[d_idx] += 1

            if ambient_flips[m]:
                direction_ambient_count[d_idx] += 1

            if verified_flips[m]:
                direction_verified_count[d_idx] += 1
                s_idx = int(sample_indices[m])
                orig_class = int(orig_classes[s_idx])
                target_class = (
                    1 - orig_class
                    if cf_preds.shape[1] == 2
                    else int(cf_preds[m].argmax())
                )
                confidence = float(cf_preds[m, target_class])
                direction_success[d_idx].append(
                    {
                        "sample_idx": s_idx,
                        "x_factual": x_samples[s_idx].cpu(),
                        "x_counterfactual": kept_images[m],
                        "confidence": confidence,
                        "orig_class": orig_class,
                        "target_class": target_class,
                        "target_delta": float(target_activation_diff[m]),
                        "specificity": float(specificity_scores[m]),
                    }
                )

        # Build results list
        direction_results = []
        for d_idx in direction_total:
            successes = direction_success.get(d_idx, [])
            successes.sort(
                key=lambda x: (x["confidence"], x["specificity"]), reverse=True
            )
            dim_name = get_dim_name(d_idx)
            max_pairs = (
                max_cf_per_direction
                if (max_cf_per_direction and max_cf_per_direction > 0)
                else len(successes)
            )
            # When a direction was subsampled, its raw counts are out of a
            # smaller denominator than an un-subsampled direction's. Scale the
            # measured rate back up to the direction's full latent flip count so
            # the ranking compares like with like; with no subsampling
            # attempted == latent_flip_count and these equal the raw counts.
            n_latent = int(latent_flip_counts[d_idx])
            n_attempted = max(1, direction_total[d_idx])
            direction_results.append(
                {
                    "direction_idx": d_idx,
                    "dimension_name": dim_name,
                    "success_count": direction_verified_count[d_idx],
                    "ambient_flip_count": direction_ambient_count[d_idx],
                    "latent_flip_count": n_latent,
                    "total_attempted": direction_total[d_idx],
                    "expected_verified_flips": direction_verified_count[d_idx]
                    * n_latent
                    / n_attempted,
                    "expected_ambient_flips": direction_ambient_count[d_idx]
                    * n_latent
                    / n_attempted,
                    "edit_realisation": (
                        float(edit_realisation[direction_indices == d_idx].mean())
                        if int((direction_indices == d_idx).sum()) > 0
                        else None
                    ),
                    "pairs": successes[:max_pairs],
                    # Every verified flip of the direction (confidence-sorted):
                    # a model teacher judges all of them, not the top pairs.
                    "all_pairs": successes,
                }
            )

        # Sort by expected verified flips descending, then expected ambient
        # flips, then latent flips.
        direction_results.sort(
            key=lambda x: (
                x["expected_verified_flips"],
                x["expected_ambient_flips"],
                x["latent_flip_count"],
            ),
            reverse=True,
        )

        # --- Step 7: Export all successful ambient flips into folders named after original components ---
        if output_dir and len(direction_results) > 0:
            successful_flips_dir = os.path.join(output_dir, "successful_flips")
            os.makedirs(successful_flips_dir, exist_ok=True)
            saved_count = 0
            flip_index = []

            for d_idx, successes in direction_success.items():
                if len(successes) == 0:
                    continue

                dim_name = get_dim_name(d_idx)
                comp_name = sae_to_gt_map.get(d_idx, dim_name)
                comp_dir = os.path.join(successful_flips_dir, _safe_name(comp_name))
                os.makedirs(comp_dir, exist_ok=True)
                index_entry = {
                    "direction_idx": int(d_idx),
                    "name": str(dim_name),
                    "folder": os.path.basename(comp_dir),
                    "n_verified": len(successes),
                    "pairs": [],
                }
                flip_index.append(index_entry)

                # Every flip is counted above; by default only the top pairs are
                # written to disk (thousands of PNGs per direction help nobody).
                export_cap = (
                    None
                    if max_export_per_direction is None
                    else max(
                        int(max_cf_per_direction or 0), int(max_export_per_direction)
                    )
                )
                for pair_idx, p in enumerate(successes[:export_cap]):
                    x_fac = p["x_factual"].float()
                    x_cf = p["x_counterfactual"].float()

                    if hasattr(self, "generator_dataset") and hasattr(
                        self.generator_dataset, "project_to_pytorch_default"
                    ):
                        x_fac_vis = self.generator_dataset.project_to_pytorch_default(
                            x_fac.unsqueeze(0)
                        ).squeeze(0)
                        x_cf_vis = self.generator_dataset.project_to_pytorch_default(
                            x_cf.unsqueeze(0)
                        ).squeeze(0)
                    else:
                        x_fac_vis = (
                            torch.clamp((x_fac + 1.0) / 2.0, 0.0, 1.0)
                            if (x_fac.min() < 0)
                            else x_fac
                        )
                        x_cf_vis = (
                            torch.clamp((x_cf + 1.0) / 2.0, 0.0, 1.0)
                            if (x_cf.min() < 0)
                            else x_cf
                        )

                    diff_vis = torch.abs(x_cf_vis - x_fac_vis)
                    grid = torch.stack([x_fac_vis, x_cf_vis, diff_vis], dim=0)

                    pair_path = os.path.join(
                        comp_dir,
                        f"sample{p['sample_idx']}_pair{pair_idx}_conf{p['confidence']:.2f}.png",
                    )
                    cf_path = os.path.join(
                        comp_dir,
                        f"sample{p['sample_idx']}_cf{pair_idx}_conf{p['confidence']:.2f}.png",
                    )

                    torchvision.utils.save_image(grid, pair_path, nrow=3)
                    torchvision.utils.save_image(x_cf_vis, cf_path)
                    saved_count += 1
                    index_entry["pairs"].append(
                        {
                            "file": os.path.basename(pair_path),
                            "sample_idx": int(p["sample_idx"]),
                            "orig_class": int(p.get("orig_class", -1)),
                            "target_class": int(p.get("target_class", -1)),
                            "confidence": float(p["confidence"]),
                        }
                    )

            with open(os.path.join(successful_flips_dir, "index.json"), "w") as f:
                json.dump(flip_index, f, indent=1)

            _log.info(
                "%s",
                f"[DiDAE Sweep] Exported {saved_count} successful ambient flip images into '{successful_flips_dir}/<original_component_name>/'",
            )

        return direction_results

    def _calculate_z_counterfactuals(
        self, z_sem: torch.Tensor, w, explainer_config=None, num_attempts=1
    ):
        # CFKD sets explainer.num_attempts = len(component_indices), so a
        # single-component config (e.g. component_indices [788], the confounder-only
        # ablation) lands on num_attempts == 1. The legacy fast path below ignores the
        # sparse dictionary entirely -- it reflects along the classifier direction w
        # rather than along the requested SAE component -- and returns a 2-D
        # [Batch, Dim] tensor where the caller expects [Batch, K, S, Dim], which then
        # dies in edit() as
        #   RuntimeError: shape '[200, 512, 3, 128, 128]' is invalid for input of size 9830400
        # Steer along the dictionary whenever one is configured with explicit
        # component_indices, regardless of how many components that is.
        steer_along_dictionary = (
            self.sparse_dictionary is not None
            and explainer_config is not None
            and getattr(explainer_config, "component_indices", None)
        )

        if not steer_along_dictionary and (
            explainer_config is None or explainer_config.num_attempts == 1
        ):
            b = w
            a = z_sem
            #
            dot_ab = torch.sum(a * b, dim=-1, keepdim=True)  # shape (batch, 1)
            dot_bb = torch.sum(b * b)  # scalar

            # projection and reflection
            proj = dot_ab / dot_bb * b  # shape (batch, n)
            # reflected = 2 * proj - a
            reflected = a - 2 * proj
            return (
                reflected,
                None,
                torch.norm(reflected - z_sem, p=2, dim=-1, keepdim=False),
                None,
            )

        elif self.sparse_dictionary is None:
            # reflection on w with all linesearch factors
            b = w
            a = z_sem
            dot_ab = torch.sum(a * b, dim=-1, keepdim=True)  # shape (batch, 1)
            dot_bb = torch.sum(b * b)  # scalar
            proj = dot_ab / dot_bb * b  # shape (batch, n)

            numeric_factors = [
                f for f in explainer_config.linesearch_factors if f != "dynamic"
            ]
            if not numeric_factors:
                numeric_factors = [2.0]
            else:
                numeric_factors = [float(f) for f in numeric_factors]
            line_search_factors = torch.tensor(numeric_factors).to(z_sem.device)
            # Apply the update: [Batch, 1, S, Dim]
            z_reflected = a.unsqueeze(1).unsqueeze(1) - line_search_factors.view(
                1, 1, -1, 1
            ) * proj.unsqueeze(1).unsqueeze(1)
            # Tile to [Batch, num_attempts, S, Dim]
            z_reflected = torch.tile(z_reflected, [1, num_attempts, 1, 1])

            # distances: [Batch, num_attempts, S]
            distances = torch.norm(
                a.unsqueeze(1).unsqueeze(1) - z_reflected, p=2, dim=-1
            )
            # component_indices: [Batch, num_attempts]
            component_indices = (
                torch.arange(num_attempts, device=z_sem.device)
                .unsqueeze(0)
                .tile([z_sem.shape[0], 1])
            )

            return z_reflected, component_indices, distances, None

        else:
            custom_indices = (
                getattr(explainer_config, "component_indices", None)
                if explainer_config
                else None
            )
            if custom_indices is not None and len(custom_indices) > 0:
                selected_indices = list(custom_indices)
            else:
                selected_indices = list(range(num_attempts))

            w = w.to(device=z_sem.device, dtype=z_sem.dtype)
            W_all = self.sparse_dictionary.get_components()
            W = W_all[:, selected_indices].to(device=z_sem.device, dtype=z_sem.dtype)

            D_encoder = w.shape[0]
            if W.shape[0] != D_encoder:
                if W.shape[0] < D_encoder:
                    W = torch.nn.functional.pad(W, (0, 0, 0, D_encoder - W.shape[0]))
                else:
                    W = W[:D_encoder, :]

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
            z_base = z_sem.unsqueeze(1)  # [Batch, 1, Dim]
            z_reflected_list = []

            c_factual = torch.matmul(z_sem, W)  # [Batch, K]
            u_norm_sq = torch.sum(W * W, dim=0).unsqueeze(0)  # [1, K]
            c_int = c_factual - proj_factors * u_norm_sq  # [Batch, K]
            c_after_list = []

            for f in explainer_config.linesearch_factors:
                if f == "dynamic":
                    sd_base_path = getattr(
                        getattr(self.sparse_dictionary, "config", None),
                        "base_path",
                        None,
                    )
                    c_min_max_path = (
                        os.path.join(sd_base_path, "c_min_and_maxes.txt")
                        if sd_base_path
                        else None
                    )
                    if c_min_max_path and os.path.exists(c_min_max_path):
                        c_mins_raw, c_maxs_raw = [], []
                        with open(c_min_max_path, "r") as file:
                            for line in file:
                                parts = line.strip().split("min=")
                                if len(parts) > 1:
                                    c_mins_raw.append(float(parts[1].split(",")[0]))
                                    c_maxs_raw.append(
                                        float(line.strip().split("max=")[1])
                                    )
                        all_c_mins = torch.tensor(c_mins_raw, device=z_sem.device)
                        all_c_maxs = torch.tensor(c_maxs_raw, device=z_sem.device)
                        # Same widening as _load_component_bounds() in the DiDAE
                        # sweep, so a CFKD told to edit a direction DiDAE found
                        # at component_bounds_scale s can take the same step.
                        bounds_scale = float(
                            getattr(explainer_config, "component_bounds_scale", 1.0)
                            or 1.0
                        )
                        if bounds_scale != 1.0:
                            mid = 0.5 * (all_c_mins + all_c_maxs)
                            half = 0.5 * (all_c_maxs - all_c_mins) * bounds_scale
                            all_c_mins, all_c_maxs = mid - half, mid + half
                        c_mins = all_c_mins[selected_indices]
                        c_maxs = all_c_maxs[selected_indices]

                        c_target = torch.where(
                            c_int > c_factual, c_maxs.unsqueeze(0), c_mins.unsqueeze(0)
                        )
                        step = ((c_factual - c_target) / u_norm_sq).unsqueeze(
                            -1
                        ) * W.permute(1, 0).unsqueeze(0)
                        z_reflected_list.append(z_base - step)
                        c_after_list.append(c_target)
                    else:
                        step = proj_factors.unsqueeze(-1) * W.permute(1, 0).unsqueeze(0)
                        z_reflected_list.append(z_base - 2.0 * step)
                        c_after_list.append(c_factual)
                else:
                    factor = float(f)
                    z_reflected_list.append(z_base - factor * projections)
                    c_after_list.append(c_factual - factor * proj_factors * u_norm_sq)

            z_reflected = torch.stack(z_reflected_list, dim=2)  # [Batch, K, S, Dim]
            z_base_expanded = z_base.unsqueeze(2)  # [Batch, 1, 1, Dim]

            # 6. Calculate Distances & Sort
            distances = torch.norm(z_base_expanded - z_reflected, p=2, dim=-1)
            sorted_indices = torch.argsort(distances, dim=1)

            component_indices = (
                torch.tensor(selected_indices, device=z_sem.device)
                .unsqueeze(0)
                .tile([sorted_indices.shape[0], 1])
            )

            c_info = {
                "c_factual": c_factual,
                "c_int": c_int,
                "c_after": torch.stack(c_after_list, dim=2),
            }

            return z_reflected, component_indices, distances, c_info


def _as_channel_tensor(value):
    """Shape a scalar or per-channel normalization constant for NCHW broadcast."""
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1).unsqueeze(-1)
    return tensor


class NormalizationModule(nn.Module):
    """pytorch-default [0, 1] -> generator normalization."""

    def __init__(self, mean, std):
        """Store mean and std as NCHW-broadcastable tensors."""
        super().__init__()
        # Plain attributes, not buffers: these must not show up in the module's
        # state_dict, or the encoder keys stop matching existing checkpoints.
        self.mean = _as_channel_tensor(mean)
        self.std = _as_channel_tensor(std)

    def forward(self, x):
        """Apply ``(x - mean) / std``."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class DenormalizationModule(nn.Module):
    """generator normalization -> pytorch-default [0, 1]. Inverse of the above."""

    def __init__(self, mean, std):
        """Store mean and std as NCHW-broadcastable tensors."""
        super().__init__()
        # Plain attributes, not buffers: these must not show up in the module's
        # state_dict, or the encoder keys stop matching existing checkpoints.
        self.mean = _as_channel_tensor(mean)
        self.std = _as_channel_tensor(std)

    def forward(self, x):
        """Apply ``x * std + mean``."""
        return x * self.std.to(x.device) + self.mean.to(x.device)
