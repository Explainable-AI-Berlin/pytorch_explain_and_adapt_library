"""PathLDMAutoencoder — DiDAE with PathLDM + PLIP foundation models.

Uses a PathLDM latent diffusion model as the pixel-level encoder/decoder
and a PLIP image encoder + SpLICE as the semantic encoder/decomposer.

Architecture:
    - Semantic encoder: PLIP image encoder → semantic embedding (z_sem)
    - Stochastic encoder: PathLDM forward diffusion q_sample → start latent (wT)
    - Sparse dictionary: SpLICE decomposes z_sem into concept weights
    - Editing: DiDAE Algorithms 1 & 2 (reflect/project z_sem along dictionary components)
    - Decoder: DDPM inversion reverse process conditioned on modified z_sem
"""

import types
import os
import copy
import math
from pathlib import Path
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from omegaconf import OmegaConf
from transformers import CLIPModel, CLIPProcessor

from peal.dependencies.PathLDM.ldm.util import instantiate_from_config
from peal.dependencies.PathLDM.ldm.models.diffusion.ddim import DDIMSampler

from peal.generators.interfaces import (
    GeneratorConfig,
    InvertibleGenerator,
    EditCapableGenerator,
)
from peal.data.interfaces import DataConfig
from peal.data.dataset_factory import get_datasets
from peal.architectures.interfaces import TaskConfig
from peal.sparse_dictionaries.interfaces import SparseDictionaryConfig, SparseDictionary
from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary
from peal.global_utils import load_yaml_config, save_yaml_config
from peal.dependencies.ddpm_inversion.ddm_inversion.inversion_utils import (
    inversion_forward_process_pathldm,
    inversion_reverse_process_pathldm,
)
from peal.dependencies.ddpm_inversion.prompt_to_prompt.ptp_utils import (
    register_attention_control_pathldm,
)
from peal.sparse_dictionaries.utils import (
    plot_component_ground_truth_correlations,
    read_component_bounds,
    select_dictionary_components,
)
from peal.log import get_logger

_log = get_logger(__name__)


# The PathLDM PLIP weights are not vendored (they are ~4 GB and separately
# licensed); they are downloaded into the repository checkout. Resolve them
# relative to this file so the defaults work on any clone, and let
# $PEAL_PLIP_DIR point somewhere else on machines that keep weights outside
# the repo.
_PLIP_DIR = os.environ.get(
    "PEAL_PLIP_DIR",
    os.path.join(
        str(Path(__file__).resolve().parents[2]),
        "peal",
        "dependencies",
        "plip_imagenet_finetune",
    ),
)


def load_model_from_config(config, ckpt_path, device):
    """Load a PathLDM model checkpoint and instantiate from config.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Loaded PathLDM project config; ``config.model`` is passed to
        ``instantiate_from_config``.
    ckpt_path : str
        Lightning checkpoint (``state_dict`` key) or bare state dict. Loaded
        with ``weights_only=False`` as fallback because the released
        checkpoints pickle callbacks.
    device : str or torch.device
        Device the model is moved to.

    Returns
    -------
    torch.nn.Module
        The latent diffusion model in eval mode; weights are loaded with
        ``strict=False``.

    Raises
    ------
    ValueError
        If the config does not instantiate a model.
    """
    # The released PathLDM checkpoints pickle their Lightning callbacks, which
    # torch >= 2.6 refuses under the new weights_only=True default.
    try:
        state = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = (
        state["state_dict"]
        if isinstance(state, dict) and "state_dict" in state
        else state
    )
    model = instantiate_from_config(config.model)
    if model is None:
        raise ValueError("PathLDM config did not instantiate a model.")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_pathldm_model(config_path, ckpt_path, device):
    """Instantiate PathLDM by config + checkpoint.

    Removes the nested ``ckpt_path`` entries of the first-stage (VAE) and
    U-Net configs so that only ``ckpt_path`` is loaded, then delegates to
    :func:`load_model_from_config`.

    Parameters
    ----------
    config_path : str
        YAML project config of the PathLDM run.
    ckpt_path : str
        Checkpoint with the full model weights.
    device : str or torch.device
        Target device.

    Returns
    -------
    torch.nn.Module
        The PathLDM latent diffusion model.
    """
    model_config = OmegaConf.load(config_path)
    if "ckpt_path" in model_config["model"]["params"]["first_stage_config"]["params"]:
        del model_config["model"]["params"]["first_stage_config"]["params"]["ckpt_path"]
    if "ckpt_path" in model_config["model"]["params"]["unet_config"]["params"]:
        del model_config["model"]["params"]["unet_config"]["params"]["ckpt_path"]
    return load_model_from_config(model_config, ckpt_path, device)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class PathldmAutoencoderConfig(GeneratorConfig):
    """Configuration for the PathLDM-based DiDAE generator.

    Parameters
    ----------
    generator_type : str
        Registry name, ``"PathldmAutoencoder"``.
    ckpt_path, config_path : str
        PathLDM checkpoint and project YAML; default to the PLIP-finetuned
        weights under ``$PEAL_PLIP_DIR`` (or
        ``peal/dependencies/plip_imagenet_finetune``).
    plip_model_id : str
        Hugging Face id of the PLIP/CLIP model used as semantic encoder.
    prompt : str
        Text prompt whose token embedding provides the baseline conditioning
        sequence (usually empty).
    data : str or DataConfig
        Generator dataset; overridden by the sparse dictionary's ``data`` key
        when that is set.
    task_config : TaskConfig or None
        Applied to every generator dataset split when given.
    base_path : str
        Run directory for finetuned weights, sparse dictionaries and the
        component explanations.
    encoder_dimensions : int
        Width of ``z_sem`` (512 for PLIP).
    vae_latent_shape : list of int
        ``[C, H, W]`` of the first-stage latent, used by ``sample_x``.
    num_diffusion_steps, eta, skip : int, float, int
        DDPM-inversion schedule: number of steps, DDIM eta (1.0 = stochastic)
        and how many of the noisiest steps are skipped when decoding.
    cfg_scale_src, cfg_scale_tar : float
        Classifier-free guidance scales for the forward/reconstruction pass
        and for counterfactual decoding.
    guidence_scale : float
        Guidance scale of ``sample_x`` (unconditional sampling).
    sparse_dictionary : str or SparseDictionaryConfig or None
        Dictionary (SpLICE, MSAE, Procrustes, ...) fitted on ``z_sem``; its
        ``base_path``/``weights_path`` default to
        ``<base_path>/sparse_dictionaries/<type>/``.
    visualizations_per_component : int or None
        Number of images rendered per component by
        ``explain_sparse_component``.
    debug : bool
        Run the ``debug_edit`` pipeline (PNG dumps) before every ``edit``.
    use_lora, rank, lora_alpha : bool, int, int or None
        Finetune the cross-attention projections with LoRA instead of the
        full U-Net.
    train_batch_size, dataloader_num_workers, learning_rate, adam_weight_decay, num_train_epochs, max_train_steps, gradient_accumulation_steps, max_grad_norm, checkpointing_steps, drop_last
        Finetuning hyper-parameters read by :func:`lora_finetune_pathldm`.
    train : bool
        Whether finetuning is intended (currently not consulted by ``edit``).
    trained_model_path : str or None
        Where the finetuned state dict is saved/loaded; defaults to
        ``<base_path>/pathldm_with_lora.pt`` or ``pathldm.pt``.
    """

    generator_type: str = "PathldmAutoencoder"
    ckpt_path: str = os.path.join(_PLIP_DIR, "checkpoints", "epoch_3.ckpt")
    config_path: str = os.path.join(_PLIP_DIR, "configs", "08-03T09-35-project.yaml")
    plip_model_id: str = "vinid/plip"
    prompt: str = ""
    data: Union[str, DataConfig] = DataConfig()
    task_config: Union[TaskConfig, None] = None
    base_path: str = ""
    encoder_dimensions: int = 512
    vae_latent_shape: list = [3, 32, 32]

    # DDPM inversion parameters
    num_diffusion_steps: int = 100
    cfg_scale_src: float = 3.5
    cfg_scale_tar: float = 15.0
    eta: float = 1.0
    skip: int = 36
    guidence_scale: float = 0.0
    # Optional edit guidance for counterfactual decoding (off by default). When set,
    # decoding uses uncond + cfg_scale_src * (eps(z) - uncond) + lambda * (eps(z_edit) - eps(z)):
    # the unedited part at the inversion scale (exact reconstruction) and only the
    # edit amplified by lambda. cfg_scale_tar is then not used for counterfactuals.
    edit_guidance_scale: Union[float, None] = None

    # Sparse dictionary (SpLICE)
    sparse_dictionary: Union[str, SparseDictionaryConfig, None] = None

    # Visualization
    visualizations_per_component: Union[int, None] = 10

    # Debug mode
    debug: bool = False

    # PathLDM LoRA finetuning
    use_lora: bool = True
    rank: int = 4
    lora_alpha: Union[int, None] = None
    train_batch_size: int = 1
    dataloader_num_workers: int = 0
    learning_rate: float = 1e-4
    adam_weight_decay: float = 0.0
    num_train_epochs: int = 1
    max_train_steps: Union[int, None] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    checkpointing_steps: int = 500
    drop_last: bool = False
    train: bool = True
    trained_model_path: Union[str, None] = None


# ---------------------------------------------------------------------------
# CLIP ViT-L/14 Encoder wrapper
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Small LoRA wrapper for PathLDM attention projection layers.

    Computes ``base(x) + up(down(x)) * alpha / rank`` with the base layer
    frozen, ``down`` initialised ``N(0, 1/rank)`` and ``up`` zero so the
    wrapped layer starts as the identity of the original.

    Parameters
    ----------
    base_layer : nn.Linear
        Frozen projection (``to_q``, ``to_k``, ``to_v`` or ``to_out[0]``).
    rank : int
        LoRA rank.
    alpha : int, optional
        LoRA alpha; defaults to ``rank`` (scale 1).
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: int = None):
        """Freeze ``base_layer`` and create the two low-rank factors."""
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scale = self.alpha / self.rank

        for param in self.base_layer.parameters():
            param.requires_grad_(False)

        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        """Base projection plus the scaled low-rank update."""
        return self.base_layer(x) + self.lora_up(self.lora_down(x)) * self.scale


def _inject_pathldm_lora(model: nn.Module, rank: int = 4, alpha: int = None):
    """Attach LoRA layers to PathLDM CrossAttention projections.

    Wraps ``to_q``/``to_k``/``to_v`` and ``to_out[0]`` of every module whose
    class is named ``CrossAttention`` in place and returns the list of new
    trainable LoRA parameters (empty if nothing was wrapped).
    """
    lora_parameters = []
    for module in model.modules():
        if module.__class__.__name__ != "CrossAttention":
            continue

        for name in ["to_q", "to_k", "to_v"]:
            layer = getattr(module, name, None)
            if isinstance(layer, nn.Linear) and not isinstance(layer, LoRALinear):
                wrapped = LoRALinear(layer, rank=rank, alpha=alpha)
                setattr(module, name, wrapped)
                lora_parameters.extend(wrapped.lora_down.parameters())
                lora_parameters.extend(wrapped.lora_up.parameters())

        if (
            hasattr(module, "to_out")
            and isinstance(module.to_out, nn.Sequential)
            and len(module.to_out) > 0
            and isinstance(module.to_out[0], nn.Linear)
            and not isinstance(module.to_out[0], LoRALinear)
        ):
            wrapped = LoRALinear(module.to_out[0], rank=rank, alpha=alpha)
            module.to_out[0] = wrapped
            lora_parameters.extend(wrapped.lora_down.parameters())
            lora_parameters.extend(wrapped.lora_up.parameters())

    return lora_parameters


def _pathldm_lora_state_dict(model: nn.Module):
    """Collect only the ``lora_down``/``lora_up`` weights (CPU) keyed by module path."""
    lora_state = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{module_name}.lora_down.weight"] = (
                module.lora_down.weight.detach().cpu()
            )
            lora_state[f"{module_name}.lora_up.weight"] = (
                module.lora_up.weight.detach().cpu()
            )
    return lora_state


def _extract_image_batch(batch):
    """Return the image tensor of a dataloader batch (dict key ``x``/``image``/
    ``pixel_values``, first element of a tuple, or the batch itself)."""
    if isinstance(batch, dict):
        for key in ["x", "image", "pixel_values"]:
            if key in batch:
                return batch[key]
        raise KeyError("Could not find an image tensor in batch dict.")
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def lora_finetune_pathldm(args):
    """Fine-tune PathLDM with PLIP semantic conditioning and optional LoRA.

    Trains the U-Net so that the cross-attention conditioning built from the
    frozen PLIP embedding of each image (via
    ``PathldmAutoencoder._build_balanced_conditioning``) reconstructs that
    image: standard epsilon (or x0) MSE on ``q_sample``-noised first-stage
    latents at random timesteps.

    Parameters
    ----------
    args : types.SimpleNamespace
        Must carry ``pathldm_autoencoder`` (the :class:`PathldmAutoencoder`
        whose ``model``, ``encoder`` and projections are used),
        ``train_dataset`` and ``base_path``. Optional, read with defaults:
        ``train_batch_size``/``batch_size``, ``dataloader_num_workers``,
        ``drop_last``, ``use_lora``, ``rank``, ``lora_alpha``,
        ``learning_rate``, ``adam_weight_decay``, ``max_train_steps``,
        ``num_train_epochs``, ``gradient_accumulation_steps``,
        ``max_grad_norm``, ``checkpointing_steps``, ``trained_model_path``.

    Returns
    -------
    torch.nn.Module
        The finetuned PathLDM model in eval mode (LoRA layers injected when
        ``use_lora``).

    Raises
    ------
    RuntimeError
        If ``use_lora`` is set but no ``CrossAttention`` layer was found.

    Notes
    -----
    Writes ``<base_path>/checkpoint-<step>/pathldm_lora.pt`` (or
    ``pathldm.pt``) every ``checkpointing_steps`` optimizer steps and, at the
    end, ``<base_path>/pathldm_lora.pt`` plus the full state dict at
    ``trained_model_path`` (default ``<base_path>/pathldm_with_lora.pt`` /
    ``pathldm.pt``). Attention control is reset with
    ``register_attention_control_pathldm(model, None)`` before training.
    """
    generator = args.pathldm_autoencoder
    model = generator.model
    device = torch.device(generator.device)
    model.to(device)
    model.train()

    train_dataset = args.train_dataset
    batch_size = getattr(args, "train_batch_size", getattr(args, "batch_size", 1))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=getattr(args, "dataloader_num_workers", 0),
        drop_last=getattr(args, "drop_last", False),
    )

    model.requires_grad_(False)
    use_lora = getattr(args, "use_lora", True)
    if use_lora:
        trainable_parameters = _inject_pathldm_lora(
            model,
            rank=getattr(args, "rank", 4),
            alpha=getattr(args, "lora_alpha", getattr(args, "rank", 4)),
        )
        if not trainable_parameters:
            raise RuntimeError("No PathLDM CrossAttention layers were found for LoRA.")
    else:
        diffusion_model = getattr(model, "model", model)
        diffusion_model.requires_grad_(True)
        trainable_parameters = [
            param for param in diffusion_model.parameters() if param.requires_grad
        ]
    register_attention_control_pathldm(model, None)

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=getattr(args, "learning_rate", 1e-4),
        weight_decay=getattr(args, "adam_weight_decay", 0.0),
    )

    max_train_steps = getattr(args, "max_train_steps", None)
    num_train_epochs = getattr(args, "num_train_epochs", 1)
    grad_accum = getattr(args, "gradient_accumulation_steps", 1)
    global_step = 0
    progress_total = (
        max_train_steps
        if max_train_steps is not None
        else num_train_epochs * math.ceil(len(train_dataloader) / grad_accum)
    )
    progress_bar = tqdm(range(progress_total), desc="PathLDM LoRA steps")

    generator.encoder.eval()
    for epoch in range(num_train_epochs):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_dataloader):
            x = _extract_image_batch(batch).to(device).float()
            x0 = generator._project_to_generator_space(x)
            model_dtype = getattr(model, "dtype", torch.float32)

            with torch.no_grad():
                z_sem = generator.encoder(x)
                encoder_hidden_states = generator._build_balanced_conditioning(
                    x.shape[0], z_sem
                ).to(device=device, dtype=model_dtype)
                latents = generator.vae_latent_encoder(x0).detach().to(model_dtype)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, model.num_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = model.q_sample(latents, timesteps, noise=noise)
            model_pred = model.apply_model(
                noisy_latents, timesteps, encoder_hidden_states
            )
            target = (
                latents if getattr(model, "parameterization", "eps") == "x0" else noise
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, getattr(args, "max_grad_norm", 1.0)
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=float(loss.detach().cpu()))

                if global_step % getattr(args, "checkpointing_steps", 500) == 0:
                    save_path = os.path.join(
                        args.base_path, f"checkpoint-{global_step}"
                    )
                    Path(save_path).mkdir(parents=True, exist_ok=True)
                    if use_lora:
                        torch.save(
                            _pathldm_lora_state_dict(model),
                            os.path.join(save_path, "pathldm_lora.pt"),
                        )
                    else:
                        torch.save(
                            model.state_dict(), os.path.join(save_path, "pathldm.pt")
                        )

                if max_train_steps is not None and global_step >= max_train_steps:
                    break

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    progress_bar.close()
    Path(args.base_path).mkdir(parents=True, exist_ok=True)
    if use_lora:
        full_model_path = (
            args.trained_model_path
            if getattr(args, "trained_model_path", None) is not None
            else os.path.join(args.base_path, "pathldm_with_lora.pt")
        )
        torch.save(
            _pathldm_lora_state_dict(model),
            os.path.join(args.base_path, "pathldm_lora.pt"),
        )
        torch.save(model.state_dict(), full_model_path)
    else:
        full_model_path = (
            args.trained_model_path
            if getattr(args, "trained_model_path", None) is not None
            else os.path.join(args.base_path, "pathldm.pt")
        )
        torch.save(model.state_dict(), full_model_path)
    model.eval()
    return model


class PLIPImageEncoder(nn.Module):
    """Wraps a PLIP checkpoint from transformers for image embeddings.

    Frozen ``CLIPModel``; :meth:`forward` resizes to the vision tower's input
    size, applies the CLIP mean/std normalisation and returns L2-normalised
    ``get_image_features`` outputs, which DiDAE uses as ``z_sem``.

    Parameters
    ----------
    model_id : str
        Hugging Face id of the PLIP/CLIP checkpoint.
    device : str
        Device for the model and the normalisation constants.

    Attributes
    ----------
    foundation_model : transformers.CLIPModel
        The frozen CLIP model (its ``text_projection`` is also used by the
        generator to map ``z_sem`` into text-embedding space).
    image_size : int
        Input resolution of the vision tower (224).
    """

    def __init__(self, model_id="vincentqb/PLIP", device="cpu"):
        """Load and freeze the CLIP model and register the normalisation constants."""
        super().__init__()
        # Note: "vinid/plip" is often redirected to "vincentqb/PLIP"
        self.foundation_model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.foundation_model.eval()
        # Freeze all parameters
        for p in self.foundation_model.parameters():
            p.requires_grad_(False)

        # PLIP/CLIP standard vision config
        self.image_size = self.foundation_model.config.vision_config.image_size
        self.device = device

        # Standard CLIP normalization constants
        self.mean = (
            torch.tensor([0.48145466, 0.4578275, 0.40821073])
            .view(1, 3, 1, 1)
            .to(device)
        )

        self.std = (
            torch.tensor([0.26862954, 0.26130258, 0.27577711])
            .view(1, 3, 1, 1)
            .to(device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to PLIP embeddings.
        Args:
            x: (B, 3, H, W) image tensor in [0, 1]
        Returns:
            (B, 512) L2-normalized PLIP image embeddings.
        """
        # 1. Resize to the model's expected input (224x224 for ViT-B/32)
        x_resized = torchvision.transforms.functional.resize(
            x, [self.image_size, self.image_size], antialias=True
        ).to(self.device)
        # 2. Normalize colors
        x_norm = (x_resized - self.mean) / self.std

        # 3. Extract Features
        with torch.no_grad():
            # get_image_features runs the projection layer (768 -> 512)
            features = self.foundation_model.get_image_features(pixel_values=x_norm)

        # 4. L2 Normalize (Essential for cosine similarity)
        return F.normalize(features.float(), p=2, dim=1)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`forward` matching the CLIP-model API used by SpLICE."""
        return self.forward(x)


# ---------------------------------------------------------------------------
# PathLDMAutoencoder
# ---------------------------------------------------------------------------


class PathldmAutoencoder(InvertibleGenerator, EditCapableGenerator):
    """DiDAE generator using PathLDM + PLIP + SpLICE.

    Treats PathLDM as a diffusion autoencoder where:

    - the semantic code ``z_sem`` comes from PLIP's image encoder,
    - the stochastic code is the edit-friendly DDPM inversion trajectory
      ``(wT, zs, wts)`` of the first-stage latent,
    - a sparse dictionary (SpLICE by default) decomposes ``z_sem`` into
      concept components,
    - editing moves ``z_sem`` along dictionary components so that a linear
      probe flips (DiDAE Algorithm 2), and
    - decoding runs the DDPM-inversion reverse process conditioned on the
      modified ``z_sem`` injected into the cross-attention sequence.

    Parameters
    ----------
    config : str or PathldmAutoencoderConfig
        Config or YAML path. When ``config.sparse_dictionary`` carries a
        ``data`` entry it replaces ``config.data``.
    predictor_dataset : Dataset, optional
        Dataset of the classifier under explanation; deep-copied and used to
        map predictor-space images to the generator's space.
    model_dir : str, optional
        Unused.
    device : str
        ``"cuda"`` is honoured only when CUDA is available, else ``"cpu"``.

    Attributes
    ----------
    encoder : PLIPImageEncoder
        Semantic encoder.
    model : torch.nn.Module
        PathLDM latent diffusion model; ``sampler`` is its ``DDIMSampler``.
    generator_datasets, generator_dataset : list, Dataset
        Splits from ``config.data`` (index 1 is the validation split used
        for fitting and explaining the dictionary) and the train split.
    sparse_dictionary : SparseDictionary or None
        Loaded/fitted dictionary. Assigning this attribute is intercepted by
        :meth:`__setattr__`.

    Notes
    -----
    When a sparse dictionary is configured the constructor loads it, fits it
    if its weights file does not exist yet, and runs
    :meth:`explain_all_components` once (guarded by the existence of
    ``c_min_and_maxes.txt`` in the dictionary's ``base_path``).
    """

    def __init__(self, config, predictor_dataset=None, model_dir=None, device="cpu"):
        """Load config, datasets, PLIP encoder, PathLDM and the sparse dictionary."""
        super().__init__()
        self.config = load_yaml_config(config, PathldmAutoencoderConfig)
        self.predictor_dataset = copy.deepcopy(predictor_dataset)
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.prompt = self.config.prompt
        self.guidence_scale = self.config.guidence_scale
        self.vae_shape = self.config.vae_latent_shape
        # --- Resolve data config override from sparse dictionary ---
        if self.config.sparse_dictionary is not None:
            sd_config = self.config.sparse_dictionary

            # If it's a string path, load it temporarily to check for data override
            if isinstance(sd_config, str):
                sd_config = load_yaml_config(sd_config)

            sd_data = None
            if isinstance(sd_config, dict):
                sd_data = sd_config.get("data")
            elif hasattr(sd_config, "data"):
                sd_data = sd_config.data

            if sd_data:
                _log.info(
                    "%s",
                    f"PathLDMAutoencoder: Overriding generator data with {sd_data} from sparse dictionary config.",
                )
                self.config.data = sd_data

        # Ensure self.config.data is fully loaded as a DataConfig

        self.config.data = load_yaml_config(self.config.data, DataConfig)
        if isinstance(self.config.data, types.SimpleNamespace):
            self.config.data = DataConfig(**vars(self.config.data))

        # --- Load datasets ---
        self.generator_datasets = get_datasets(self.config.data)
        if self.config.task_config is not None:
            for ds in self.generator_datasets:
                if ds is not None:
                    ds.task_config = self.config.task_config

        self.generator_dataset = (
            self.generator_datasets[0] if self.generator_datasets else None
        )

        # --- Setup PLIP semantic encoder ---
        self.encoder = PLIPImageEncoder(
            model_id=self.config.plip_model_id,
            device=self.device,
        )

        # --- Load PathLDM backend ---
        self.model = get_pathldm_model(
            config_path=self.config.config_path,
            ckpt_path=self.config.ckpt_path,
            device=self.device,
        )
        self.sampler = DDIMSampler(self.model)

        # --- Load sparse dictionary (SpLICE) if configured ---
        self.sparse_dictionary = None
        if self.config.sparse_dictionary is not None:
            loaded_from_disk = self.load_sparse_dictionary()

            if self.sparse_dictionary is not None and hasattr(
                self.sparse_dictionary, "set_clip_model"
            ):
                self.sparse_dictionary.set_clip_model(self.encoder)

            # Check if it was actually loaded from disk (has image_mean)
            # If not, it means the (default) weights file didn't exist yet, so we fit.
            if getattr(self.sparse_dictionary, "image_mean", None) is None:
                if not loaded_from_disk:
                    self.fit_sparse_dictionary()

                # explain_all_components() is expensive (full generate/decode
                # per component) and its own output is what tells us whether
                # it already ran — image_mean above is SpLICE-specific and is
                # always None for other dictionary types (e.g.
                # OrthogonalProcrustesDictionary uses `mu` instead), so relying
                # on it would re-run this every time regardless of dictionary
                # type.
                c_min_max_path = os.path.join(
                    self.config.sparse_dictionary.base_path, "c_min_and_maxes.txt"
                )
                if not os.path.exists(c_min_max_path):
                    self.explain_all_components()

        self._edit_training_completed = False

    def _trained_model_path(self) -> str:
        """Path of the finetuned state dict (``config.trained_model_path`` or the
        ``pathldm_with_lora.pt``/``pathldm.pt`` default under ``base_path``)."""
        if self.config.trained_model_path is not None:
            return self.config.trained_model_path
        filename = "pathldm_with_lora.pt" if self.config.use_lora else "pathldm.pt"
        return os.path.join(self.config.base_path, filename)

    def _load_trained_model_if_available(self) -> bool:
        """Inject LoRA layers if configured and load the finetuned state dict;
        returns False when the file does not exist."""
        trained_model_path = self._trained_model_path()
        if not os.path.exists(trained_model_path):
            return False

        state_dict = torch.load(trained_model_path, map_location=self.device)
        if self.config.use_lora:
            _inject_pathldm_lora(
                self.model,
                rank=self.config.rank,
                alpha=(
                    self.config.lora_alpha
                    if self.config.lora_alpha is not None
                    else self.config.rank
                ),
            )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        return True

    def _ensure_trained_model(self):
        """Load the finetuned model or run :meth:`train_model`, once per instance."""
        if self._edit_training_completed:
            return
        if self._load_trained_model_if_available():
            _log.info(
                "%s",
                "PathLDMAutoencoder: loaded trained PathLDM model from "
                f"{self._trained_model_path()}.",
            )
        else:
            _log.info(
                "%s",
                "PathLDMAutoencoder: trained model not found; starting PathLDM "
                "finetuning.",
            )
            self.train_model()
        self._edit_training_completed = True

    def __setattr__(self, name, value):
        """Override __setattr__ to ensure dictionary consistency.

        If a new sparse_dictionary is assigned (e.g., by CFKD), we ensure
        it inherits the correctly resolved weights_path from our config
        if one was already established.
        """
        if name == "sparse_dictionary" and value is not None:
            # If we already have a weights_path resolved in our config,
            # ensure the new dictionary uses it.
            if (
                hasattr(self, "config")
                and self.config.sparse_dictionary is not None
                and self.config.sparse_dictionary.weights_path
            ):
                if getattr(value.config, "weights_path", None) is None:
                    value.config.weights_path = (
                        self.config.sparse_dictionary.weights_path
                    )
                    # If the file exists, load it immediately to avoid re-fitting or using internet mean
                    if os.path.exists(value.config.weights_path):
                        if getattr(value, "image_mean", None) is None:
                            value.load_from_disk(value.config.weights_path)

            # Keep sparse dictionary in sync with the semantic encoder API
            if hasattr(self, "encoder") and hasattr(value, "set_clip_model"):
                value.set_clip_model(self.encoder)

        super().__setattr__(name, value)

    # -------------------------------------------------------------------
    # Debugging Utilities
    # -------------------------------------------------------------------

    def _setup_debug_dir(self, base_path: str = "debug_pathldm") -> str:
        """Create and return debug directory path."""
        debug_dir = Path(base_path)
        debug_dir.mkdir(parents=True, exist_ok=True)
        return str(debug_dir)

    def _save_debug_image(self, tensor: torch.Tensor, filename: str, debug_dir: str):
        """Save a tensor as PNG image for debugging.

        Args:
            tensor: (B, C, H, W) or (C, H, W) or (H, W) tensor in [-1, 1] or [0, 1]
            filename: output filename (without directory)
            debug_dir: directory to save to
        """
        from PIL import Image
        import numpy as np

        # Handle batch dimension
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first sample

        # Move to CPU and detach
        tensor = tensor.detach().cpu()

        # Normalize to [0, 1]
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to numpy
        if tensor.dim() == 3:
            # (C, H, W) → permute to (H, W, C)
            arr = tensor.permute(1, 2, 0).numpy()
        else:
            # (H, W) grayscale
            arr = tensor.numpy()

        # Convert to uint8
        arr = (arr * 255).astype(np.uint8)

        # Handle grayscale → RGB
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        # Save
        img = Image.fromarray(arr)
        filepath = Path(debug_dir) / filename
        img.save(filepath)
        _log.info("%s", f"[DEBUG] Saved: {filepath}")

    def _project_to_generator_space(self, x: torch.Tensor) -> torch.Tensor:
        """Convert predictor-space tensors into PathLDM's image space."""
        if self.predictor_dataset is not None and hasattr(
            self.predictor_dataset, "project_to_pytorch_default"
        ):
            x = self.predictor_dataset.project_to_pytorch_default(x)
        if self.generator_dataset is not None and hasattr(
            self.generator_dataset, "project_from_pytorch_default"
        ):
            x = self.generator_dataset.project_from_pytorch_default(x)
        return x.to(self.device)

    def debug_encode(self, x: torch.Tensor, debug_dir: str = "debug_pathldm") -> Tuple:
        """Encode with intermediate visualization.

        Saves:
        - input_image.png (original input)
        - encoder_output.png (PLIP semantic embedding norm)
        - vae_latent.png (VAE encoder output)
        - diffusion_wT.png (terminal diffusion state)
        """
        debug_dir = self._setup_debug_dir(debug_dir)
        batch_size = x.shape[0]
        x = x.to(self.device)

        # Step 1: Input
        self._save_debug_image(x, "00_input_image.png", debug_dir)
        _log.info(
            "%s",
            f"[DEBUG] Input shape: {x.shape}, dtype: {x.dtype}, range: [{x.min():.3f}, {x.max():.3f}]",
        )

        # Step 2: PLIP encoder
        z_sem = self.encoder(x)
        _log.info(
            "%s",
            f"[DEBUG] z_sem (PLIP embed) shape: {z_sem.shape}, norm: {z_sem.norm(dim=-1).mean():.3f}",
        )
        # Visualize z_sem magnitude
        z_sem_vis = z_sem.norm(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, 64)
        self._save_debug_image(
            z_sem_vis / z_sem_vis.max(), "01_z_sem_magnitude.png", debug_dir
        )

        # Step 3: Image preprocessing for PathLDM
        x0 = self._project_to_generator_space(x)
        self._save_debug_image(x0, "02_x0_preprocessed.png", debug_dir)
        _log.info(
            "%s",
            f"[DEBUG] x0 preprocessed shape: {x0.shape}, range: [{x0.min():.3f}, {x0.max():.3f}]",
        )

        # Step 4: VAE latent encoding
        w0 = self.vae_latent_encoder(x0)
        _log.info(
            "%s",
            f"[DEBUG] w0 (VAE latent) shape: {w0.shape}, range: [{w0.min():.3f}, {w0.max():.3f}]",
        )
        # Visualize decoded w0 (back to pixel space)
        w0_decoded = self.vae_latent_decoder(w0)
        self._save_debug_image(w0_decoded, "03_w0_vae_decoded.png", debug_dir)
        # Step 5: Conditioning
        z_sem_cond, _ = self._build_balanced_conditioning(batch_size, z_sem)
        _log.info("%s", f"[DEBUG] z_sem_cond shape: {z_sem_cond.shape}")

        # Step 6: Forward diffusion
        wT, zs, wts = inversion_forward_process_pathldm(
            self.model,
            w0.to(self.model.dtype),
            etas=self.config.eta,
            prompt=batch_size * [""],
            cfg_scale=self.config.cfg_scale_src,
            prog_bar=False,
            num_inference_steps=self.config.num_diffusion_steps,
            encoder_hidden_states=z_sem_cond.to(self.model.dtype),
            debug=True,
            debug_dir=debug_dir,
        )
        _log.info(
            "%s",
            f"[DEBUG] wT shape: {wT.shape}, range: [{wT.min():.3f}, {wT.max():.3f}]",
        )
        # print(f"[DEBUG] zs length: {len(zs)}, zs[0] shape: {zs[0].shape}")
        # Visualize terminal state (decode back to pixel space)
        wT_decoded = self.vae_latent_decoder(wT)
        self._save_debug_image(wT_decoded, "04_wT_terminal_decoded.png", debug_dir)
        _log.info("%s", "debug breakpoint")
        for i, latents in enumerate(wts):
            latent = self.vae_latent_decoder(latents)
            self._save_debug_image(latent, f"intermediate_latents{i}.png", debug_dir)

        return z_sem, (wT, zs, wts)

    def debug_calculate_z_counterfactuals(
        self,
        z_sem: torch.Tensor,
        w: torch.Tensor,
        explainer_config,
        num_attempts: int = 1,
        debug_dir: str = "debug_pathldm",
    ):
        """Calculate counterfactuals with intermediate visualization.

        Saves:
        - z_sem_original.png (original embedding norm)
        - z_sem_reflected_*.png (reflected embeddings per attempt)
        - distances.png (reflection distances)
        """
        debug_dir = self._setup_debug_dir(debug_dir)

        # Visualize original z_sem
        z_sem_vis = z_sem.norm(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, 64)
        self._save_debug_image(
            z_sem_vis / z_sem_vis.max(), "10_z_sem_original.png", debug_dir
        )
        _log.info(
            "%s",
            f"[DEBUG] z_sem (input) shape: {z_sem.shape}, norm range: [{z_sem.norm(dim=-1).min():.3f}, {z_sem.norm(dim=-1).max():.3f}]",
        )

        # Visualize classifier weight
        _log.info(
            "%s", f"[DEBUG] Classifier weight w shape: {w.shape}, norm: {w.norm():.3f}"
        )

        if num_attempts == 1:
            # Simple reflection
            b = w.to(z_sem.dtype)
            a = z_sem
            dot_ab = torch.sum(a * b, dim=-1, keepdim=True)
            dot_bb = torch.sum(b * b)
            proj = dot_ab / dot_bb * b
            reflected = a - 2 * proj

            _log.info(
                "%s",
                f"[DEBUG] Simple reflection: dot_ab min/max: {dot_ab.min():.3f}/{dot_ab.max():.3f}",
            )
            _log.info(
                "%s",
                f"[DEBUG] Reflected shape: {reflected.shape}, norm range: [{reflected.norm(dim=-1).min():.3f}, {reflected.norm(dim=-1).max():.3f}]",
            )

            # Visualize reflected
            reflected_vis = (
                reflected.norm(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, 64)
            )
            self._save_debug_image(
                reflected_vis / (reflected_vis.max() + 1e-8),
                "11_z_sem_reflected_simple.png",
                debug_dir,
            )

            distances = torch.norm(reflected - z_sem, p=2, dim=-1, keepdim=False)
            _log.info(
                "%s",
                f"[DEBUG] Distance range: [{distances.min():.3f}, {distances.max():.3f}]",
            )

            return reflected, None, distances

        else:
            # Sparse dictionary reflection (commented code path)
            _log.info(
                "%s", f"[DEBUG] Using sparse dictionary with {num_attempts} attempts"
            )

            # This would call the actual _calculate_z_counterfactuals
            # For now, just note that we're in complex mode
            z_sem_before, indices, distances = self._calculate_z_counterfactuals(
                z_sem, w, explainer_config, num_attempts
            )

            # Visualize first reflected
            if z_sem_before.shape[1] > 0:
                reflected_vis = (
                    z_sem_before[:, 0]
                    .norm(dim=-1, keepdim=True)
                    .unsqueeze(-1)
                    .expand(-1, -1, 64)
                )
                self._save_debug_image(
                    reflected_vis / (reflected_vis.max() + 1e-8),
                    "11_z_sem_reflected_0.png",
                    debug_dir,
                )

            return z_sem_before, indices, distances

    # -------------------------------------------------------------------
    # Encode / Decode (DiDAE Algorithm 1)
    # -------------------------------------------------------------------

    def _build_balanced_conditioning(
        self,
        batch_size: int,
        z_sem: torch.Tensor,
        proj: torch.Tensor = None,
        factor: float = None,
        z_sem_ref: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build PathLDM cross-attention conditioning from semantic embeddings.

        ``z_sem`` is mapped into CLIP text-embedding space with the inverse of
        the PLIP ``text_projection``, rescaled to the norm of the prompt's
        token 1 (the ``<eos>`` of an empty prompt) and written into every
        token position ``1:`` of the prompt's conditioning sequence; the
        final positions are restored to the prompt's own tokens.

        Parameters
        ----------
        batch_size : int
            Batch size of the conditioning sequence.
        z_sem : torch.Tensor
            ``[B, D]`` (edited or original) PLIP embeddings.
        proj : torch.Tensor, optional
            ``[B, D]`` edit direction; with ``factor`` the step
            ``factor * proj`` is applied in text space after normalising the
            unedited ``z_sem`` (line search along one direction).
        factor : float, optional
            Line-search factor for ``proj``.
        z_sem_ref : torch.Tensor, optional
            ``[B, D]`` pre-edit embeddings; when given (and
            ``PEAL_PATHLDM_ZSEM_REF != "0"``) the rescaling uses the
            reference norm so candidates derived from the same reference keep
            their relative magnitudes.

        Returns
        -------
        tuple
            ``(z_sem_cond, z_sem_text_dist)``: the ``[B, L, C]`` conditioning
            sequence and the un-normalised text-space embedding ``[B, C]``.
        """
        uncond, cond = self._prepare_conditioning(
            batch_size=batch_size, summary=self.prompt
        )
        # PathLDM uses sequence conditioning. We replace token 1 and keep remaining
        # tokens as repeated baseline values to avoid attention saturation.
        text_projection_weights = self.encoder.foundation_model.text_projection.weight
        weights_inv = torch.linalg.inv(text_projection_weights)
        z_sem_text_dist = z_sem @ weights_inv.T
        z_sem_cond = uncond.clone()
        target_norm = torch.norm(uncond[:, 1, :], p=2, dim=-1, keepdim=True)
        if proj is not None and factor is not None:
            # Applying linesearch after normalization rather than before: all linesearch
            # factors move along the same semantic direction, so normalizing each
            # reflected z independently would collapse them to the same conditioning
            # vector, losing the scale difference between steps. By normalizing z_sem
            # once and stepping in text space, the factors produce conditioning vectors
            # at different distances from the base — preserving linesearch strength.
            scale = target_norm / (
                torch.norm(z_sem_text_dist, p=2, dim=-1, keepdim=True) + 1e-8
            )
            z_sem_normalized = z_sem_text_dist * scale
            proj_text = (proj @ weights_inv.T) * scale
            z_sem_scaled = z_sem_normalized - factor * proj_text
        elif (
            z_sem_ref is not None
            and os.environ.get("PEAL_PATHLDM_ZSEM_REF", "1") == "1"
        ):
            # Same idea as the proj/factor branch, but for callers that already
            # pre-combined the edit into a single z_sem candidate (e.g. the
            # counterfactual linesearch/dynamic outputs from
            # _calculate_z_counterfactuals). Deriving the rescaling factor from
            # the *reference* (pre-edit) embedding's norm instead of each
            # candidate's own norm means multiple candidates coming from the
            # same z_sem_ref keep their relative magnitude differences instead
            # of all independently collapsing onto the same target_norm sphere.
            ref_text_dist = z_sem_ref @ weights_inv.T
            scale = target_norm / (
                torch.norm(ref_text_dist, p=2, dim=-1, keepdim=True) + 1e-8
            )
            z_sem_scaled = z_sem_text_dist * scale
        else:
            current_norm = torch.norm(z_sem_text_dist, p=2, dim=-1, keepdim=True)
            z_sem_scaled = z_sem_text_dist * (target_norm / (current_norm + 1e-8))
        # z_sem only in location 1
        # z_sem_cond[:, 1, :] = z_sem_scaled.to(z_sem_cond.dtype)
        # all z_sem
        z_sem_cond[:, 1:, :] = (
            z_sem_scaled.unsqueeze(-2)
            .expand(-1, z_sem_cond.shape[1] - 1, -1)
            .to(z_sem_cond.dtype)
        )
        z_sem_cond[:, 77:76, :] = uncond[:, 0:1, :]
        z_sem_cond[:, -1:, :] = uncond[:, 1:2, :]
        z_sem_cond[:, 78:79, :] = uncond[:, 1:2, :]
        # # Eos in all except token 1
        # z_sem_cond[:, 2:, :] = uncond[:, 1:2, :].expand(-1, z_sem_cond.shape[1] - 2, -1)
        # z_sem_cond[:, 78, :] = z_sem_scaled.to(z_sem_cond.dtype)
        # use same tokens as empty tokens except 1 is z sem
        # z_sem_cond[:, 2:, :] = uncond[:, 2:, :]
        # z_sem_cond[:, 1 + 77, :] = z_sem_scaled.to(z_sem_cond.dtype)
        return z_sem_cond, z_sem_text_dist

    def _repeat_stochastic_code_for_candidates(
        self, stochastic_code: Tuple, batch_size: int
    ) -> Tuple:
        """Repeat inversion trajectories when one input image has many candidates."""
        wT, zs, wts = stochastic_code

        def repeat_batch(tensor, batch_dim):
            if tensor is None or tensor.shape[batch_dim] == batch_size:
                return tensor
            if batch_size % tensor.shape[batch_dim] != 0:
                raise ValueError(
                    "Cannot align stochastic code batch "
                    f"{tensor.shape[batch_dim]} with target batch {batch_size}."
                )
            repeats = batch_size // tensor.shape[batch_dim]
            return tensor.repeat_interleave(repeats, dim=batch_dim)

        # wT has shape [B, C, H, W], while zs/wts have shape [T, B, C, H, W].
        return repeat_batch(wT, 0), repeat_batch(zs, 1), repeat_batch(wts, 1)

    def _get_starting_latent(self, wT, wts, batch_size: int) -> torch.Tensor:
        """Latent the reverse process starts from: ``wts[T - skip]`` when the
        trajectory is available, else ``wT`` repeated to ``batch_size``."""
        if wts is not None:
            return wts[self.config.num_diffusion_steps - self.config.skip]
        if wT.shape[0] != batch_size:
            if batch_size % wT.shape[0] != 0:
                raise ValueError(
                    f"Cannot align wT batch {wT.shape[0]} with target batch {batch_size}."
                )
            wT = wT.repeat_interleave(batch_size // wT.shape[0], dim=0)
        return wT

    def _sample_latent(
        self,
        conditioning: torch.Tensor,
        x_t: torch.Tensor,
        guidance_scale: float,
        num_steps: int,
    ) -> torch.Tensor:
        """Run DDIM reverse process from a provided starting latent."""
        uncond = self.model.get_learned_conditioning(
            x_t.shape[0] * [self.config.prompt]
        ).to(self.device)
        latent_shape = tuple(x_t.shape[1:])
        samples, _ = self.sampler.sample(
            S=num_steps,
            batch_size=x_t.shape[0],
            shape=latent_shape,
            conditioning=conditioning,
            eta=self.config.eta,
            x_T=x_t,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond,
            use_tqdm=False,
            verbose=False,
        )
        return samples

    @torch.enable_grad()
    def vae_latent_encoder(self, image_tensor):
        """Encode images with PathLDM's first stage.

        Parameters
        ----------
        image_tensor : torch.Tensor
            ``[B, 3, H, W]`` images in the generator's ``[-1, 1]`` space.

        Returns
        -------
        torch.Tensor
            Scaled first-stage latents (``get_first_stage_encoding``) with
            ``requires_grad`` enabled.
        """
        image_tensor = image_tensor
        vae_encode = self.model.encode_first_stage(image_tensor)
        vae_encode = self.model.get_first_stage_encoding(vae_encode)

        return vae_encode.requires_grad_(True)

    def vae_latent_decoder(self, latnet_tensor):
        """Decode first-stage latents to images in ``[0, 1]``.

        Parameters
        ----------
        latnet_tensor : torch.Tensor
            Scaled latents as returned by :meth:`vae_latent_encoder`.

        Returns
        -------
        torch.Tensor
            ``[B, 3, H, W]`` images mapped from ``[-1, 1]`` to ``[0, 1]`` and
            clamped.
        """
        # This method internally divides by scale_factor
        vae_decoded = self.model.decode_first_stage(latnet_tensor)
        # Convert to [0, 1] for visualization
        vae_decoded = (vae_decoded / 2.0) + 0.5
        return vae_decoded.clamp(0, 1)

    def get_unconditional_token(self, batch_size):
        """Return ``batch_size`` empty prompts for classifier-free guidance."""
        return [""] * batch_size

    def get_conditional_token(self, batch_size, summary):
        """Return ``batch_size`` copies of ``config.prompt``; ``summary`` is
        ignored."""

        # append tumor and TIL probability to the summary
        tumor = [self.prompt] * (batch_size)

        return [t for t in tumor]

    def _prepare_conditioning(self, batch_size: int, summary: str):
        """
        Prepares unconditional and conditional embeddings for the model.

        Args:
            model: The diffusion model instance.
            batch_size (int): The batch size of the input.
            summary (str): The text summary for conditional guidance.

        Returns:
            tuple: A tuple containing:
                - uc (torch.Tensor): Unconditional conditioning embedding.
                - cc (torch.Tensor or None): Conditional conditioning embedding, or None if no summary.
        """
        ut = self.get_unconditional_token(batch_size)
        uc = self.model.get_learned_conditioning(ut)

        ct = self.get_conditional_token(batch_size, summary)
        cc = self.model.get_learned_conditioning(ct)
        return uc, cc

    def encode(self, x, t=1.0, stochastic=None, num_steps=None):
        """Encode an image into semantic code + stochastic noise path.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, 3, H, W)`` input images in the predictor's dataset space.

        Returns
        -------
        z_sem : torch.Tensor
            ``(B, D)`` PLIP image embedding.
        stochastic_code : tuple
            ``(wT, zs, wts)`` from the edit-friendly DDPM inversion of the
            first-stage latent, conditioned on ``z_sem``: the terminal latent
            ``[B, C, h, w]``, the per-step noise maps and the latent trajectory
            (both ``[T, B, C, h, w]``).
        """
        debug_dir = self._setup_debug_dir()
        batch_size = x.shape[0]
        x = x.to(self.device)
        z_sem = self.encoder(x)
        x0 = self._project_to_generator_space(x)
        # VAE latent for PathLDM
        w0 = self.vae_latent_encoder(x0)
        # Forward diffuse to DDIM terminal timestep used by the sampler
        # uncond_embedding, _ = self._prepare_conditioning(
        #     batch_size=batch_size, summary=self.prompt
        # )
        # z_sem_cond = uncond_embedding.clone()

        # target_norm = torch.norm(uncond_embedding[:, 1, :], p=2, dim=-1, keepdim=True)
        # current_norm = torch.norm(z_sem, p=2, dim=-1, keepdim=True)
        # z_sem_scaled = z_sem * (target_norm / (current_norm + 1e-8))

        # z_sem_cond[:, 1, :] = z_sem_scaled
        # # Explicitly replicate the <EOS> token (at index 1 of an empty prompt) 75 times
        # z_sem_cond[:, 2:, :] = uncond_embedding[:, 1:2, :].expand(-1, 75, -1)
        z_sem_cond, z_sem_1 = self._build_balanced_conditioning(batch_size, z_sem)
        wT, zs, wts = inversion_forward_process_pathldm(
            self.model,
            w0.to(self.model.dtype),
            etas=self.config.eta,
            prompt=batch_size * [""],
            cfg_scale=self.config.cfg_scale_src,
            prog_bar=False,
            num_inference_steps=self.config.num_diffusion_steps,
            encoder_hidden_states=z_sem_cond.to(self.model.dtype),
            debug_dir=debug_dir,
            debug=False,
        )

        return z_sem, (wT, zs, wts)

    def decode(self, z, t=1.0, stochastic=None, num_steps=None):
        """Decode from semantic code + stochastic noise path.

        Parameters
        ----------
        z : tuple
            ``(z_sem, (wT, zs, wts))``: ``z_sem`` is the ``(B, D)`` PLIP
            embedding to condition on; ``wT, zs, wts`` is the stochastic code
            from :meth:`encode`. The reverse process starts at
            ``wts[num_diffusion_steps - skip]`` and re-injects the first
            ``num_diffusion_steps - skip`` noise maps ``zs``, with
            ``cfg_scale_src`` as guidance scale.

        Returns
        -------
        x_decoded : torch.Tensor
            ``(B, 3, H, W)`` reconstructed/edited images in ``[0, 1]``, detached.
        """
        z_sem, (wT, zs, wts) = z
        batch_size = z_sem.shape[0]
        z_sem_cond, _ = self._build_balanced_conditioning(batch_size, z_sem)
        # controller = AttentionStore()
        # register_attention_control_pathldm(self.model, controller)
        xT = wts[self.config.num_diffusion_steps - self.config.skip]

        w0_dec, _ = inversion_reverse_process_pathldm(
            self.model,
            xT=xT.to(self.model.dtype),
            etas=self.config.eta,
            prompts=batch_size * [""],
            cfg_scales=[self.config.cfg_scale_src],
            prog_bar=False,
            zs=zs[: (self.config.num_diffusion_steps - self.config.skip)],
            controller=None,
            encoder_hidden_states=z_sem_cond.to(self.model.dtype),
            num_inference_steps=self.config.num_diffusion_steps,
        )

        x_decoded = self.vae_latent_decoder(w0_dec)

        if x_decoded.dim() < 4:
            x_decoded = x_decoded.unsqueeze(0)

        # Detach the batch-size-specific controller so it doesn't leak into
        # later forward passes through self.model at a different batch size
        # (register_attention_control_pathldm attaches it persistently).
        # register_attention_control_pathldm(self.model, None)

        return x_decoded.detach().float()

    def decode_with_modified_embedding(
        self,
        z_sem_modified,
        stochastic_code,
        original_shape,
        prompts=None,
        z_sem_ref=None,
    ):
        """Decode using a modified semantic embedding for counterfactual generation.

        Uses the modified PLIP embedding as text conditioning for the PathLDM DDIM
        reverse process. The stochastic noise path preserves structure while
        the modified embedding changes semantics.

        Args:
            z_sem_modified: (B, D) modified PLIP embedding.
            stochastic_code: tuple (wT, zs, wts) from encode.
            original_shape: target spatial dimensions for output.
            prompts: Optional list of text prompts for benchmarking.
            z_sem_ref: Optional (B, D) pre-edit PLIP embedding that
                z_sem_modified candidates were derived from. When given, the
                conditioning norm is rescaled relative to this reference
                instead of each candidate's own norm, preserving relative
                edit-strength differences between candidates (see
                _build_balanced_conditioning).

        Returns:
            x_counterfactual: (B, 3, H, W) counterfactual images.
        """
        batch_size = z_sem_modified.shape[0]
        wT, zs, wts = self._repeat_stochastic_code_for_candidates(
            stochastic_code, batch_size
        )
        if prompts is not None:
            encoder_hidden_states = self.model.get_learned_conditioning(prompts).to(
                self.device
            )
        else:
            encoder_hidden_states, _ = self._build_balanced_conditioning(
                batch_size, z_sem_modified, z_sem_ref=z_sem_ref
            )
        debug_dir = self._setup_debug_dir()
        # controller = AttentionStore()
        # register_attention_control_pathldm(self.model, controller)
        xT = self._get_starting_latent(wT, wts, batch_size)
        # Optional edit guidance: unedited part at the inversion scale, edit amplified by
        # edit_guidance_scale. The reference conditioning equals the one encode() inverted with.
        use_edit_guidance = (
            getattr(self.config, "edit_guidance_scale", None) is not None
            and prompts is None
            and z_sem_ref is not None
        )
        ref_encoder_hidden_states = None
        if use_edit_guidance:
            ref_encoder_hidden_states, _ = self._build_balanced_conditioning(
                batch_size, z_sem_ref, z_sem_ref=z_sem_ref
            )
        w0_dec, _ = inversion_reverse_process_pathldm(
            self.model,
            xT=xT.to(self.model.dtype),
            etas=self.config.eta,
            prompts=batch_size * [""],
            cfg_scales=[
                self.config.cfg_scale_src if use_edit_guidance else self.config.cfg_scale_tar
            ],
            prog_bar=False,
            zs=(
                zs[: (self.config.num_diffusion_steps - self.config.skip)]
                if zs is not None
                else None
            ),
            controller=None,
            encoder_hidden_states=encoder_hidden_states,
            num_inference_steps=self.config.num_diffusion_steps,
            debug=False,
            debug_dir=debug_dir,
            ref_encoder_hidden_states=ref_encoder_hidden_states,
            edit_guidance_scale=(
                self.config.edit_guidance_scale if use_edit_guidance else None
            ),
        )

        x_decoded = self.vae_latent_decoder(w0_dec)
        if x_decoded.dim() < 4:
            x_decoded = x_decoded.unsqueeze(0)

        # Resize back to original spatial dimensions
        x_counterfactual = torchvision.transforms.Resize(
            original_shape[2:], antialias=True
        )(x_decoded.detach().cpu())

        return x_counterfactual.float()

    def debug_decode_with_modified_embedding(
        self,
        z_sem_modified: torch.Tensor,
        stochastic_code: Tuple,
        original_shape: Tuple,
        debug_dir: str = "debug_pathldm",
        prompts=None,
    ):
        """Decode with step-by-step visualization of reverse diffusion process.

        Saves:
        - z_sem_modified.png (modified embedding)
        - encoder_hidden_states.png (conditioning tensor)
        - xT.png (starting latent)
        - w0_dec.png (decoded latent before VAE)
        - x_decoded_vae.png (VAE decoder output)
        - x_counterfactual_final.png (final resized output)

        Parameters
        ----------
        z_sem_modified : torch.Tensor
            ``[B, D]`` edited embeddings.
        stochastic_code : tuple
            ``(wT, zs, wts)`` from :meth:`encode`.
        original_shape : tuple
            Shape whose last two entries give the output resolution.
        debug_dir : str
            Directory for the PNG dumps.
        prompts : list of str, optional
            Text prompts used as conditioning instead of ``z_sem_modified``.

        Returns
        -------
        torch.Tensor
            ``[B, 3, H, W]`` counterfactuals on CPU. Unlike
            :meth:`decode_with_modified_embedding` this uses
            ``cfg_scale_src`` and no ``z_sem_ref`` rescaling.
        """
        debug_dir = self._setup_debug_dir(debug_dir)

        batch_size = z_sem_modified.shape[0]
        wT, zs, wts = self._repeat_stochastic_code_for_candidates(
            stochastic_code, batch_size
        )

        # Step 1: Modified embedding
        z_sem_vis = (
            z_sem_modified.norm(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, 64)
        )
        self._save_debug_image(
            z_sem_vis / (z_sem_vis.max() + 1e-8), "20_z_sem_modified.png", debug_dir
        )
        _log.info(
            "%s",
            f"[DEBUG] z_sem_modified shape: {z_sem_modified.shape}, norm range: [{z_sem_modified.norm(dim=-1).min():.3f}, {z_sem_modified.norm(dim=-1).max():.3f}]",
        )

        # Step 2: Build conditioning
        if prompts is not None:
            encoder_hidden_states = self.model.get_learned_conditioning(prompts).to(
                self.device
            )
        else:
            encoder_hidden_states, _ = self._build_balanced_conditioning(
                batch_size, z_sem_modified
            )

        _log.info(
            "%s",
            f"[DEBUG] encoder_hidden_states shape: {encoder_hidden_states.shape}, dtype: {encoder_hidden_states.dtype}",
        )
        # Visualize first token
        hs_vis = (
            encoder_hidden_states[:1, :1]
            .norm(dim=-1, keepdim=True)
            .unsqueeze(-1)
            .expand(-1, -1, 64)
        )
        self._save_debug_image(
            hs_vis / (hs_vis.max() + 1e-8), "21_encoder_hidden_states.png", debug_dir
        )

        # Step 3: Setup reverse process
        register_attention_control_pathldm(self.model, None)
        xT = self._get_starting_latent(wT, wts, batch_size)

        _log.info(
            "%s",
            f"[DEBUG] xT shape: {xT.shape}, range: [{xT.min():.3f}, {xT.max():.3f}]",
        )
        xT_vis = xT[:1, :1]
        self._save_debug_image(xT_vis, "22_xT_starting.png", debug_dir)

        # Step 4: Reverse diffusion
        _log.info(
            "%s",
            f"[DEBUG] Starting reverse diffusion with {len(zs)} stored noise steps...",
        )
        w0_dec, _ = inversion_reverse_process_pathldm(
            self.model,
            xT=xT.to(self.model.dtype),
            etas=self.config.eta,
            prompts=batch_size * [""],
            cfg_scales=[self.config.cfg_scale_src],
            prog_bar=False,
            zs=(
                zs[: (self.config.num_diffusion_steps - self.config.skip)]
                if zs is not None
                else None
            ),
            controller=None,
            encoder_hidden_states=encoder_hidden_states,
            num_inference_steps=self.config.num_diffusion_steps,
        )

        _log.info(
            "%s",
            f"[DEBUG] w0_dec (reverse output) shape: {w0_dec.shape}, range: [{w0_dec.min():.3f}, {w0_dec.max():.3f}]",
        )
        w0_dec_vis = w0_dec[:1, :1]
        self._save_debug_image(w0_dec_vis, "23_w0_dec_after_diffusion.png", debug_dir)

        # Step 5: VAE decoder
        x_decoded = self.vae_latent_decoder(w0_dec)
        if x_decoded.dim() < 4:
            x_decoded = x_decoded.unsqueeze(0)

        _log.info(
            "%s",
            f"[DEBUG] x_decoded (VAE output) shape: {x_decoded.shape}, range: [{x_decoded.min():.3f}, {x_decoded.max():.3f}]",
        )
        self._save_debug_image(x_decoded, "24_x_decoded_vae.png", debug_dir)

        # Step 6: Resize to original
        x_counterfactual = torchvision.transforms.Resize(
            original_shape[2:], antialias=True
        )(x_decoded.detach().cpu())

        _log.info(
            "%s",
            f"[DEBUG] x_counterfactual (final) shape: {x_counterfactual.shape}, range: [{x_counterfactual.min():.3f}, {x_counterfactual.max():.3f}]",
        )
        self._save_debug_image(
            x_counterfactual, "25_x_counterfactual_final.png", debug_dir
        )

        return x_counterfactual.float()

    # -------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------
    def sample_x(self, batch_size=None, renormalize=True):
        """Sample images from PathLDM with the config prompt (50 DDIM steps).

        Parameters
        ----------
        batch_size : int, optional
            Defaults to ``config.batch_size``.
        renormalize : bool
            Project the ``[0, 1]`` samples into the validation dataset's
            tensor space. ``False`` leaves ``sample`` unassigned, so the
            method currently requires ``True``.

        Returns
        -------
        torch.Tensor
            ``[batch_size, 3, H, W]`` samples.
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        scale = self.guidence_scale

        with torch.no_grad():

            # unconditional token for classifier free guidance
            ut = self.get_unconditional_token(batch_size)
            uc = self.model.get_learned_conditioning(ut)

            ct = self.get_conditional_token(batch_size, self.prompt)
            cc = self.model.get_learned_conditioning(ct)
            sampler = DDIMSampler(self.model)
            samples_ddim, _ = sampler.sample(
                50,
                batch_size,
                self.vae_shape,
                cc,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=0,
            )
            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            # x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()
        if renormalize:
            sample = self.generator_datasets[1].project_to_pytorch_default(
                x_samples_ddim
            )

        return sample

    def sample_z(self, batch_size=1):
        """Draw ``[batch_size, encoder_dimensions]`` standard-normal semantic codes."""
        return torch.randn(batch_size, self.config.encoder_dimensions)

    def log_prob_z(self, z):
        """Not available for this generator; always raises ``NotImplementedError``."""
        raise NotImplementedError(
            "Log probability not available for PathLDM autoencoder."
        )

    # -------------------------------------------------------------------
    # Sparse Dictionary (SpLICE)
    # -------------------------------------------------------------------

    def load_sparse_dictionary(self):
        """Load or initialize sparse dictionary.

        Fills in ``config.sparse_dictionary.base_path`` /
        ``weights_path`` (``<base_path>/sparse_dictionaries/<type>/weights<ending>``)
        when unset, builds the dictionary with ``get_sparse_dictionary`` and
        assigns it to ``self.sparse_dictionary`` (which triggers
        :meth:`__setattr__` and loads the weights if the file exists).

        Returns
        -------
        bool
            Whether the weights file already existed on disk.
        """
        # Ensure dictionary has a base_path if not provided by the user
        # We store it relative to the generator's base_path for organization
        if (
            self.config.sparse_dictionary is not None
            and hasattr(self.config.sparse_dictionary, "base_path")
            and self.config.sparse_dictionary.base_path is None
            and self.config.base_path
        ):
            self.config.sparse_dictionary.base_path = os.path.join(
                self.config.base_path,
                "sparse_dictionaries",
                self.config.sparse_dictionary.sparse_dictionaries_type,
            )
            # weights_path is usually base_path + weights.ending
            ending = getattr(self.config.sparse_dictionary, "ending", ".pt")
            self.config.sparse_dictionary.weights_path = os.path.join(
                self.config.sparse_dictionary.base_path, "weights" + ending
            )
        _log.info(
            "%s",
            f"PathLDMAutoencoder: loading sparse dictionary "
            f"({self.config.sparse_dictionary.sparse_dictionaries_type}) "
            f"from {self.config.sparse_dictionary.weights_path} ...",
        )
        self.sparse_dictionary = get_sparse_dictionary(self.config.sparse_dictionary)
        loaded_from_disk = os.path.exists(self.config.sparse_dictionary.weights_path)
        if loaded_from_disk:
            _log.info(
                "%s",
                "PathLDMAutoencoder: found existing sparse dictionary weights at "
                f"{self.config.sparse_dictionary.weights_path}.",
            )
        else:
            _log.info(
                "%s",
                "PathLDMAutoencoder: no sparse dictionary weights found at "
                f"{self.config.sparse_dictionary.weights_path}; will fit a new one.",
            )
        return loaded_from_disk

    def fit_sparse_dictionary(self):
        """Fit the sparse dictionary on PLIP embeddings of the validation split.

        Calls ``fit_from_dataloaders`` with a batch-64 loader over
        ``generator_datasets[1]`` and ``self.encoder``; for SpLICE this is
        the dataset-specific image mean. When the dictionary config has a
        ``base_path`` the weights are saved to ``weights_path`` and the
        dictionary config to ``<base_path>/config.yaml``.
        """
        _log.info(
            "%s",
            "PathLDMAutoencoder: fitting new sparse dictionary "
            f"({self.config.sparse_dictionary.sparse_dictionaries_type}) from dataset ...",
        )
        self.sparse_dictionary.fit_from_dataloaders(
            [torch.utils.data.DataLoader(self.generator_datasets[1], batch_size=64)],
            self.encoder,
        )
        _log.info("%s", "PathLDMAutoencoder: finished fitting sparse dictionary.")
        # Save results to disk
        if self.config.sparse_dictionary.base_path:
            Path(self.config.sparse_dictionary.base_path).mkdir(
                parents=True, exist_ok=True
            )
            self.sparse_dictionary.save_on_disk(
                self.config.sparse_dictionary.weights_path
            )
            save_yaml_config(
                self.config.sparse_dictionary,
                os.path.join(self.config.sparse_dictionary.base_path, "config.yaml"),
            )
            Path(self.config.sparse_dictionary.base_path).mkdir(
                parents=True, exist_ok=True
            )
            self.sparse_dictionary.save_on_disk(
                self.config.sparse_dictionary.weights_path
            )
            save_yaml_config(
                self.config.sparse_dictionary,
                os.path.join(self.config.sparse_dictionary.base_path, "config.yaml"),
            )
            _log.info(
                "%s",
                "PathLDMAutoencoder: saved sparse dictionary weights to "
                f"{self.config.sparse_dictionary.weights_path}.",
            )

    # -------------------------------------------------------------------
    # Edit (DiDAE Algorithm 2)
    # -------------------------------------------------------------------

    def debug_edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        predictor: nn.Module,
        explainer_config: dict,
        predictor_datasets: list,
        debug_dir: str = "debug_pathldm",
        boolmask_in=None,
        attempt_number=None,
        pbar=None,
        base_path: str = "",
        mode: str = "",
    ) -> Tuple[list, list, list, list, list, list]:
        """Full debug pipeline with visualization at each stage.

        Generates visualizations for:
        1. Encoding: input -> z_sem + wT
        2. Counterfactual calculation: reflection of embeddings
        3. Decoding: modified z_sem -> counterfactual images

        Same arguments and return structure as :meth:`edit`, plus
        ``debug_dir`` for the PNG dumps. Unlike ``edit`` it always picks the
        first line-search candidate (no outlier scoring) and returns
        ``x_in`` in place of the difference list.
        """
        debug_dir = self._setup_debug_dir(debug_dir)
        _log.info(
            "%s", f"\n[DEBUG] Starting full debug edit pipeline. Output: {debug_dir}\n"
        )

        device = list(predictor.parameters())[0].device

        # ===== Step 1: Encode with debugging =====
        _log.info("%s", "[DEBUG] === STEP 1: ENCODING ===")
        z_sem, stochastic_code = self.debug_encode(
            x_in.to(self.device), debug_dir=debug_dir
        )

        # ===== Step 2: Calculate counterfactuals with debugging =====
        _log.info("%s", "[DEBUG] === STEP 2: COUNTERFACTUAL CALCULATION ===")

        # Get predictor weight
        if (
            not hasattr(explainer_config, "distilled_predictor")
            or explainer_config.distilled_predictor is None
        ):
            w = list(predictor.children())[-1].weight[0]
        else:
            from peal.adaptors.counterfactual_knowledge_distillation import (
                distill_predictor,
            )

            distilled_path = os.path.join(
                base_path, "explainer", "distilled_predictor", "model.cpl"
            )
            if not os.path.exists(distilled_path):
                self.gradient_predictor = distill_predictor(
                    predictor_distillation=explainer_config.distilled_predictor,
                    base_path=os.path.join(base_path, "explainer"),
                    predictor=lambda x: predictor(x),
                    predictor_datasource=predictor_datasets,
                    predictor_distilled=nn.Sequential(
                        self.encoder,
                        nn.Linear(self.config.encoder_dimensions, 1, bias=False),
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
            w = list(self.gradient_predictor.children())[-1].weight[0]

        z_sem_before, indices, distances = self.debug_calculate_z_counterfactuals(
            z_sem,
            w,
            explainer_config,
            explainer_config.num_attempts,
            debug_dir=debug_dir,
        )

        # ===== Step 3: Decode counterfactuals with debugging =====
        _log.info("%s", "[DEBUG] === STEP 3: DECODING ===")

        # Flatten and tile for batch decoding
        z_sem2 = z_sem_before.reshape([-1, z_sem_before.shape[-1]])
        z_sem_ref2 = z_sem
        for _ in range(z_sem_before.dim() - z_sem.dim()):
            z_sem_ref2 = z_sem_ref2.unsqueeze(1)
        z_sem_ref2 = z_sem_ref2.expand_as(z_sem_before).reshape(
            [-1, z_sem_before.shape[-1]]
        )
        wT, zs, wts = stochastic_code
        wT_decoding = wT.unsqueeze(1).unsqueeze(1)
        wT_decoding = wT_decoding.tile(
            1, z_sem_before.shape[1], len(explainer_config.linesearch_factors), 1, 1, 1
        )
        wT_decoding = wT_decoding.reshape([-1] + list(wT.shape[1:]))

        # Debug first candidate
        _log.info(
            "%s", "[DEBUG] Decoding first counterfactual candidate (debug_decode)..."
        )
        x_cf_debug = self.debug_decode_with_modified_embedding(
            z_sem2[:1],
            (wT_decoding[:1], zs, wts),
            x_in.shape,
            debug_dir=debug_dir,
        )
        _log.info(
            "%s",
            f"[DEBUG] First counterfactual shape: {x_cf_debug.shape}, range: [{x_cf_debug.min():.3f}, {x_cf_debug.max():.3f}]",
        )

        # Now decode all candidates normally
        _log.info(
            "%s", f"[DEBUG] Decoding all {z_sem2.shape[0]} candidates (normal mode)..."
        )
        x_counterfactuals_generator = self.decode_with_modified_embedding(
            z_sem2,
            (wT_decoding, zs, wts),
            x_in.shape,
            prompts=None,
            z_sem_ref=z_sem_ref2,
        )
        x_counterfactuals = x_counterfactuals_generator.detach()
        _log.info("%s", f"[DEBUG] All counterfactuals shape: {x_counterfactuals.shape}")
        self._save_debug_image(
            x_counterfactuals, "30_all_counterfactuals.png", debug_dir
        )

        # Evaluate
        _log.info("%s", "[DEBUG] === STEP 4: EVALUATION ===")
        pred_original = (
            F.softmax(predictor(x_in.to(device).float()), dim=-1).detach().cpu()
        )
        preds = (
            F.softmax(predictor(x_counterfactuals.to(device).float()), dim=-1)
            .detach()
            .cpu()
        )

        y_target_end_confidence = torch.zeros([preds.shape[0]])
        for i in range(preds.shape[0]):
            y_target_end_confidence[i] = preds[
                i, target_classes[i % target_classes.shape[0]]
            ]

        _log.info(
            "%s", f"[DEBUG] Original pred confidence: {pred_original[0].max():.3f}"
        )
        _log.info(
            "%s",
            f"[DEBUG] Counterfactual pred confidence range: [{y_target_end_confidence.min():.3f}, {y_target_end_confidence.max():.3f}]",
        )

        # Reshape and select best
        x_counterfactuals = torch.reshape(
            x_counterfactuals,
            list(z_sem_before.shape[:3]) + list(x_counterfactuals.shape[1:]),
        )
        y_target_end_confidence = torch.reshape(
            y_target_end_confidence, z_sem_before.shape[:3]
        )

        x_counterfactuals_out_list = []
        y_target_end_confidence_list = []
        indices_list = []

        for i in range(explainer_config.num_attempts):
            j = torch.zeros([x_counterfactuals.shape[0]], dtype=torch.long)
            for k in range(j.shape[0]):
                x_counterfactuals_out_list.append(x_counterfactuals[k, i, j[k], :])
                y_target_end_confidence_list.append(
                    float(y_target_end_confidence[k, i, j[k]])
                )
                indices_list.append(indices[k, i].unsqueeze(0))

        x_counterfactuals_final = torch.stack(x_counterfactuals_out_list, dim=0)
        self._save_debug_image(
            x_counterfactuals_final, "31_best_counterfactuals.png", debug_dir
        )

        _log.info(
            "%s", f"\n[DEBUG] Pipeline complete! Visualizations saved to: {debug_dir}\n"
        )

        return (
            list(x_counterfactuals_final.cpu()),
            list(x_in.cpu()),
            list(y_target_end_confidence_list),
            list(x_in.cpu()),
            [],
            list(torch.cat(indices_list).cpu()),
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
        boolmask_in=None,
        attempt_number=None,
        pbar=None,
        base_path: str = "",
        mode: str = "",
    ) -> Tuple[list, list, list, list, list, list]:
        """Generate counterfactuals using DiDAE Algorithms 1 & 2.

        1. Encode input → z_sem + stochastic noise path
        2. Reflect z_sem along sparse dictionary components
        3. Decode reflected z_sem with original noise path → counterfactual

        This follows the same flow as DiffusionAutoencoder.edit() but uses
        PathLDM + PLIP for semantic-to-latent counterfactual generation.

        If debug=True in config, runs debug pipeline before normal pipeline.

        Parameters
        ----------
        x_in : torch.Tensor
            ``[B, 3, H, W]`` inputs in the predictor's dataset space.
        target_confidence_goal : float
            Ignored; replaced by ``1 - p(target | x_in)`` per sample.
        source_classes, target_classes : torch.Tensor
            ``[B]`` source and target labels.
        predictor : nn.Module
            Classifier under explanation.
        explainer_config
            Provides ``num_attempts``, ``linesearch_factors`` (floats or
            ``"dynamic"``) and optionally ``distilled_predictor``.
        predictor_datasets : list
            ``predictor_datasets[1]`` (or its first dataloader's dataset) is
            used for ``calculate_outlier_score``.
        boolmask_in, attempt_number, pbar, mode : optional
            Unused.
        base_path : str
            Explainer directory; the distilled linear probe is cached at
            ``<base_path>/explainer/distilled_predictor/model.cpl``.

        Returns
        -------
        tuple
            ``(counterfactuals, x_in - counterfactuals, target confidences,
            inputs, [], component indices)``, each a list with
            ``B * num_attempts`` entries ordered attempt-major.

        Notes
        -----
        The edit direction ``w`` is the last-layer weight of ``predictor``
        or, when ``distilled_predictor`` is configured, of a linear probe on
        ``self.encoder`` trained with ``distill_predictor``. Candidates are
        ``z_sem_before`` of shape ``[B, K, S, D]`` (``K = num_attempts``
        components, ``S`` line-search factors); all ``B*K*S`` are decoded
        with the tiled stochastic code and scored by the predictor. Per
        (sample, attempt) the candidate whose masked target confidence is
        closest to the goal is chosen, where candidates with a relative
        outlier score outside ``(0.1, 3.5)`` are masked out.
        """
        # if self.config.train:
        #     self._ensure_trained_model()

        # ===== DEBUG: Run debug pipeline first if enabled =====
        if self.config.debug:
            debug_dir = os.path.join(base_path, "..", "debug_pathldm")
            _log.info("%s", f"\n[DEBUG] Running debug pipeline (output: {debug_dir})")
            try:
                self.debug_edit(
                    x_in=x_in,
                    target_confidence_goal=target_confidence_goal,
                    source_classes=source_classes,
                    target_classes=target_classes,
                    predictor=predictor,
                    explainer_config=explainer_config,
                    predictor_datasets=predictor_datasets,
                    debug_dir=debug_dir,
                    attempt_number=attempt_number,
                    pbar=pbar,
                    base_path=base_path,
                    mode=mode,
                )
            except Exception as e:
                _log.info("%s", f"[DEBUG] Warning: debug pipeline failed: {e}")
                _log.info("%s", "[DEBUG] Continuing with normal pipeline...")

        # ===== NORMAL EDIT PIPELINE =====
        param_list = [p for p in predictor.parameters()]
        device = param_list[0].device

        # Compute initial predictions
        pred_original = (
            F.softmax(predictor(x_in.to(device).float()), dim=-1).detach().cpu()
        )
        target_confidences = [
            pred_original[i][target_classes[i]] for i in range(len(target_classes))
        ]
        target_confidence_goal = 1 - torch.tensor(target_confidences)

        # Get validation dataset for outlier scoring
        from peal.data.dataloaders import WeightedDataloaderList

        if isinstance(predictor_datasets[1], WeightedDataloaderList):
            validation_dataset = predictor_datasets[1].dataloaders[0].dataset
        else:
            validation_dataset = predictor_datasets[1]

        # Encode
        z_sem, stochastic_code = self.encode(x_in.to(self.device))
        # Get editing direction from gradient predictor
        # Uses the same approach as DiffusionAutoencoder: distill classifier
        # into a linear probe on top of the encoder, then use its weights
        if (
            not hasattr(explainer_config, "distilled_predictor")
            or explainer_config.distilled_predictor is None
        ):
            # Direct approach: use the last linear layer weights
            w = list(predictor.children())[-1].weight[0]
        else:
            from peal.adaptors.counterfactual_knowledge_distillation import (
                distill_predictor,
            )

            distilled_path = os.path.join(
                base_path, "explainer", "distilled_predictor", "model.cpl"
            )
            if not os.path.exists(distilled_path):
                self.gradient_predictor = distill_predictor(
                    predictor_distillation=explainer_config.distilled_predictor,
                    base_path=os.path.join(base_path, "explainer"),
                    predictor=lambda x: predictor(x),
                    predictor_datasource=predictor_datasets,
                    predictor_distilled=nn.Sequential(
                        self.encoder,
                        nn.Linear(self.config.encoder_dimensions, 1, bias=False),
                    ),
                    only_last_layer=True,
                    continue_training=True,
                    task_config=TaskConfig(
                        **explainer_config.distilled_predictor["task"]
                    ),
                )
            else:
                # PEAL's own distilled predictor; torch >= 2.6 defaults
                # weights_only=True, which cannot unpickle a whole nn.Module.
                self.gradient_predictor = torch.load(
                    distilled_path, map_location=self.device, weights_only=False
                )

            w = list(self.gradient_predictor.children())[-1].weight[0]

            dot_zw = torch.sum(z_sem * w, dim=-1).detach().cpu()
        # Calculate counterfactual z_sem values (DiDAE Algorithm 2)
        z_sem_before, indices, distances = self._calculate_z_counterfactuals(
            z_sem, w, explainer_config, explainer_config.num_attempts
        )

        # dot_reflected = (
        #     torch.sum(
        #         z_sem_before * w.view(1, 1, 1, -1),
        #         dim=-1,
        #     )
        #     .detach()
        #     .cpu()
        # )
        # with torch.no_grad():
        #     z_sem = self.encoder(x_in.to(self.device))

        #     real_pred = predictor(x_in.to(device).float()).argmax(-1).cpu()

        #     probe = list(self.gradient_predictor.children())[-1]
        #     probe_score = (z_sem @ w.to(z_sem.device)).detach().cpu()

        #     probe_pred = (probe_score > 0).long()

        # print("real pred:", real_pred)
        # print("probe pred:", probe_pred)
        # print("probe score:", probe_score)
        # print("agreement:", (real_pred == probe_pred).float().mean())
        # target_sign = (2 * target_classes.long() - 1).view(-1, 1, 1)
        # target_margin = target_sign * dot_reflected
        # print("source:", source_classes)
        # print("target:", target_classes)
        # print("dot original:", dot_zw)
        # print("dot reflected:", dot_reflected)
        # print("target margin reflected:", target_margin)
        # explainer_config.num_attempts = z_sem_before.shape[1]

        # Flatten for batch decoding
        z_sem2 = z_sem_before.reshape([-1, z_sem_before.shape[-1]])
        # Broadcast the pre-edit z_sem to match every candidate derived from it,
        # so _build_balanced_conditioning can rescale relative to the shared
        # reference norm instead of each candidate's own norm (see
        # _build_balanced_conditioning's z_sem_ref branch).
        z_sem_ref2 = z_sem
        for _ in range(z_sem_before.dim() - z_sem.dim()):
            z_sem_ref2 = z_sem_ref2.unsqueeze(1)
        z_sem_ref2 = z_sem_ref2.expand_as(z_sem_before).reshape(
            [-1, z_sem_before.shape[-1]]
        )
        # Tile stochastic code for all candidates
        wT, zs, wts = stochastic_code
        wT_decoding = wT.unsqueeze(1).unsqueeze(1)
        wT_decoding = wT_decoding.tile(
            1, z_sem_before.shape[1], len(explainer_config.linesearch_factors), 1, 1, 1
        )
        wT_decoding = wT_decoding.reshape([-1] + list(wT.shape[1:]))

        # Decode all candidates
        prompts = None
        x_counterfactuals_generator = self.decode_with_modified_embedding(
            z_sem2,
            (wT_decoding, zs, wts),
            x_in.shape,
            prompts=prompts,
            z_sem_ref=z_sem_ref2,
        )
        x_counterfactuals = x_counterfactuals_generator.detach()

        # Evaluate all candidates with the predictor
        preds = (
            F.softmax(predictor(x_counterfactuals.to(device).float()), dim=-1)
            .detach()
            .cpu()
        )
        target_classes_cpu = target_classes.detach().cpu().long()
        candidate_shape = z_sem_before.shape[:3]
        preds_by_candidate = preds.reshape(*candidate_shape, preds.shape[-1])
        target_index = target_classes_cpu.view(-1, 1, 1, 1).expand(
            -1, candidate_shape[1], candidate_shape[2], 1
        )
        y_target_end_confidence = torch.gather(
            preds_by_candidate, dim=-1, index=target_index
        ).squeeze(-1)

        # Reshape to (batch, num_attempts, linesearch_steps, ...)
        if explainer_config.num_attempts > 1:
            x_counterfactuals = torch.reshape(
                x_counterfactuals,
                list(z_sem_before.shape[:3]) + list(x_counterfactuals.shape[1:]),
            )

        # Select best counterfactual per attempt
        x_counterfactuals_out_list = []
        y_target_end_confidence_list = []
        x_out_list = []
        indices_list = []

        for i in range(explainer_config.num_attempts):
            if x_counterfactuals.shape[2] >= 2:
                y_target_diff = y_target_end_confidence.clone()
                for b in range(y_target_end_confidence.shape[0]):
                    outlier_scores = validation_dataset.calculate_outlier_score(
                        x_counterfactuals[b, i]
                    )["relative"].cpu()
                    mask = torch.logical_and(outlier_scores < 3.5, outlier_scores > 0.1)
                    masked_difference = y_target_diff[b, i] * mask
                    y_target_diff[b, i] = torch.abs(
                        masked_difference - target_confidence_goal[b]
                    )
                j = torch.argmin(y_target_diff[:, i, :], dim=-1)

            else:
                j = torch.zeros([x_counterfactuals.shape[0]], dtype=torch.long)
                # for k in range(j.shape[0]):
                #     x_counterfactuals_out_list.append(x_counterfactuals[k])
                #     y_target_end_confidence_list.append(
                #         float(y_target_end_confidence[k])
                #     )
                #     indices_list.append(k)

            for k in range(j.shape[0]):
                x_counterfactuals_out_list.append(x_counterfactuals[k, i, j[k], :])
                y_target_end_confidence_list.append(
                    float(y_target_end_confidence[k, i, j[k]])
                )
                indices_list.append(indices[k, i].unsqueeze(0))
            x_out_list.append(x_in)

        x_counterfactuals = torch.stack(x_counterfactuals_out_list, dim=0)
        x_out = torch.cat(x_out_list, dim=0)
        y_target_end_confidence = torch.tensor(y_target_end_confidence_list)
        if len(indices_list) > 1:
            indices = torch.cat(indices_list, dim=0)
        else:
            indices = torch.tensor(indices_list)
        x_difference = x_out - x_counterfactuals.cpu()

        return (
            list(x_counterfactuals.cpu()),
            list(x_difference),
            list(y_target_end_confidence),
            list(x_in),
            [],
            list(indices.cpu()),
        )

    # def _calculate_z_counterfactuals(
    #     self,
    #     z_sem: torch.Tensor,
    #     w: torch.Tensor,
    #     explainer_config=None,
    #     num_attempts=1,
    # ):
    #     """Calculate counterfactual embeddings via reflection (DiDAE Algorithm 2).

    #     Mirrors DiffusionAutoencoder._calculate_z_counterfactuals exactly.
    #     Reflects z_sem along sparse dictionary component directions to
    #     flip the classifier decision.
    #     """
    #     if explainer_config is None or explainer_config.num_attempts == 1:
    #         # Simple reflection along classifier weight direction
    #         b = w.to(z_sem.dtype)
    #         a = z_sem
    #         dot_ab = torch.sum(a * b, dim=-1, keepdim=True)
    #         dot_bb = torch.sum(b * b)
    #         proj = dot_ab / dot_bb * b
    #         reflected = a - 2 * proj

    #         return (
    #             reflected,
    #             None,
    #             torch.norm(reflected - z_sem, p=2, dim=-1, keepdim=False),
    #         )

    #     else:
    #         # Targeted Swap Logic: Precisely swap concepts with their opposites
    #         vocab = (
    #             self.sparse_dictionary.get_vocabulary()
    #             if hasattr(self.sparse_dictionary, "get_vocabulary")
    #             else None
    #         )

    #         # Resolve conceptual strings from config
    #         orig_comp_strs = self.sparse_dictionary.config.component_strings
    #         orig_opp_strs = getattr(
    #             self.sparse_dictionary.config, "opposite_component_strings", []
    #         )

    #         # Identify all concepts to track for comprehensive logging (union of targets and opposites)
    #         track_strs = list(orig_comp_strs)
    #         if orig_opp_strs:
    #             for o in orig_opp_strs:
    #                 if o and o not in track_strs:
    #                     track_strs.append(o)

    #         # Resolve all for tracking (this also triggers the loud warnings for not-found concepts)
    #         track_indices = self._resolve_component_indices(
    #             self.sparse_dictionary.config, track_strs
    #         )

    #         # Resolve primary and opposite indices for swapping logic
    #         component_indices = self._resolve_component_indices(
    #             self.sparse_dictionary.config
    #         )
    #         opp_indices = None
    #         if orig_opp_strs:
    #             opp_indices = self._resolve_component_indices(
    #                 self.sparse_dictionary.config, orig_opp_strs
    #             )

    #         if component_indices is None or len(component_indices) == 0:
    #             component_indices = list(range(num_attempts))
    #             opp_indices = None

    #         W_all = (
    #             self.sparse_dictionary.get_components().to(z_sem.device).to(z_sem.dtype)
    #         )

    #         # Initial decomposition for activation detection
    #         with torch.no_grad():
    #             activations = self.sparse_dictionary.decompose(z_sem)  # (B, K)

    #         # Step size modulation step
    #         line_search_factors = (
    #             torch.tensor(explainer_config.linesearch_factors)
    #             .to(z_sem.device)
    #             .to(z_sem.dtype)
    #         )

    #         z_reflected_list = []

    #         # Generate candidates per targeted attempt
    #         for i, comp_idx in enumerate(component_indices):
    #             opp_idx = (
    #                 opp_indices[i]
    #                 if (opp_indices is not None and i < len(opp_indices))
    #                 else None
    #             )

    #             # Concept Names for reporting
    #             comp_name = orig_comp_strs[i]
    #             opp_name = (
    #                 orig_opp_strs[i]
    #                 if (orig_opp_strs and i < len(orig_opp_strs))
    #                 else "None"
    #             )

    #             print(
    #                 f"\n--- Attempt {i}: Target '{comp_name}' (ID {comp_idx}) / Opposite '{opp_name}' (ID {opp_idx}) ---",
    #                 flush=True,
    #             )

    #             if comp_idx is None:
    #                 print(
    #                     f"  !!! Skipping attempt {i} as primary concept '{comp_name}' was not found in vocabulary !!!",
    #                     flush=True,
    #                 )
    #                 continue

    #             W_primary = W_all[:, comp_idx]
    #             W_opp = W_all[:, opp_idx] if opp_idx is not None else None

    #             is_activated = activations[:, comp_idx] > 0.01  # (B,)

    #             print("  Status BEFORE edit:", flush=True)
    #             for b in range(z_sem.shape[0]):
    #                 scores_list = []
    #                 for name, idx in zip(track_strs, track_indices):
    #                     val = f"{activations[b, idx]:.2f}" if idx is not None else "N/A"
    #                     scores_list.append(f"'{name}': {val}")

    #                 status = (
    #                     f"ACTIVE ({activations[b, comp_idx]:.2f})"
    #                     if is_activated[b]
    #                     else "ABSENT"
    #                 )
    #                 print(
    #                     f"    Sample {b}: '{comp_name}' is {status}. Scores -> {', '.join(scores_list)}",
    #                     flush=True,
    #                 )

    #             attempt_results = []
    #             for factor in line_search_factors:
    #                 z_edit = z_sem.clone()
    #                 # Logic: Swap present concepts for opposites, or introduce absent concepts.
    #                 if factor == 0.0:
    #                     z_step = z_sem  # Baseline
    #                 else:
    #                     # Logic 1: Present -> Remove primary, Add opposite
    #                     proj_primary = (
    #                         torch.sum(z_sem * W_primary, dim=-1, keepdim=True)
    #                     ) * W_primary
    #                     res1 = z_sem - proj_primary
    #                     if W_opp is not None:
    #                         res1 += factor * W_opp

    #                     # Logic 2: Absent -> Remove opposite (if accidentally active), Add primary
    #                     if W_opp is not None:
    #                         proj_opp = (
    #                             torch.sum(z_sem * W_opp, dim=-1, keepdim=True)
    #                         ) * W_opp
    #                         res2 = z_sem - proj_opp
    #                     else:
    #                         res2 = z_sem
    #                     res2 += factor * W_primary

    #                     z_step = torch.where(is_activated.unsqueeze(1), res1, res2)

    #                 attempt_results.append(z_step.unsqueeze(1))

    #             # Sanity Check (Verification decomposition of candidates)
    #             z_check = attempt_results[-1].squeeze(1)
    #             with torch.no_grad():
    #                 new_activations = self.sparse_dictionary.decompose(z_check)

    #             print(
    #                 f"  Status AFTER edit (Sanity Check, Factor {line_search_factors[-1]:.2f}):",
    #                 flush=True,
    #             )
    #             for b in range(z_sem.shape[0]):
    #                 after_list = []
    #                 for name, idx in zip(track_strs, track_indices):
    #                     val = (
    #                         f"{new_activations[b, idx]:.2f}"
    #                         if idx is not None
    #                         else "N/A"
    #                     )
    #                     after_list.append(f"'{name}': {val}")
    #                 print(
    #                     f"    Sample {b}: Scores -> {', '.join(after_list)}", flush=True
    #                 )

    #             z_reflected_list.append(torch.cat(attempt_results, dim=1).unsqueeze(1))

    #         z_reflected = torch.cat(z_reflected_list, dim=1)
    #         z_base = z_sem.unsqueeze(1).unsqueeze(1)
    #         distances = torch.norm(z_base - z_reflected, p=2, dim=-1)

    #         valid_indices = [idx for idx in component_indices if idx is not None]
    #         out_component_indices = (
    #             torch.tensor(valid_indices)
    #             .to(z_sem.device)
    #             .unsqueeze(0)
    #             .tile([z_sem.shape[0], 1])
    #         )

    #         return z_reflected, out_component_indices, distances

    def _calculate_z_counterfactuals(
        self, z_sem: torch.Tensor, w, explainer_config=None, num_attempts=1
    ):
        """Compute edited semantic codes (DiDAE Algorithm 2).

        Parameters
        ----------
        z_sem : torch.Tensor
            ``[B, D]`` PLIP embeddings.
        w : torch.Tensor
            ``[D]`` probe weight (edit direction).
        explainer_config : optional
            Provides ``num_attempts`` and ``linesearch_factors``.
        num_attempts : int
            Number of dictionary components ``K`` to try (the first ``K``
            columns of ``get_components()``).

        Returns
        -------
        tuple
            With ``num_attempts == 1`` (or no config): the Householder
            reflection ``z - 2 proj_w(z)`` as ``[B, D]``, ``None`` and the
            ``[B]`` edit distances. Otherwise ``(z_reflected,
            component_indices, distances)`` with ``z_reflected``
            ``[B, K, S, D]`` for ``S`` line-search factors,
            ``component_indices`` ``[B, K]`` (``0..K-1``) and ``distances``
            ``[B, K, S]``.

        Notes
        -----
        For component ``u_k`` the step ``s_k = (z . w) / (u_k . w)`` moves
        the probe score to zero along ``u_k`` (``u_k . w`` is clamped away
        from zero); numeric factors ``f`` give ``z - f * s_k u_k``. The
        factor ``"dynamic"`` instead targets the dataset-wide component
        activation extreme (``c_min``/``c_max`` read from
        ``c_min_and_maxes.txt`` written by :meth:`explain_all_components`)
        unless ``PEAL_PATHLDM_DYNAMIC_CMINMAX=0``, in which case it equals
        factor 1.
        """
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
            return (
                reflected,
                None,
                torch.norm(reflected - z_sem, p=2, dim=-1, keepdim=False),
            )

        else:
            # Attempt k steps along column k, or along component_indices[k] when the
            # explainer config sets it (optional; e.g. chosen atoms of an SAE).
            W, selected_components = select_dictionary_components(
                self.sparse_dictionary, explainer_config, num_attempts
            )
            W = W.to(z_sem.device)

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
            # Toggle to A/B compare the c_min/c_max dynamic-bound targeting
            # against just using the plain cross-projection scaling/
            # normalization step (same as a numeric linesearch factor) for
            # "dynamic" entries. Flip back to True to restore c_min/c_max.
            USE_DYNAMIC_C_MIN_MAX = (
                os.environ.get("PEAL_PATHLDM_DYNAMIC_CMINMAX", "1") == "1"
            )

            # "-dynamic" (optional) targets the opposite bound: the step goes to the
            # other side of the probe, so the linesearch can let the student pick the side.
            if USE_DYNAMIC_C_MIN_MAX and any(
                f in ("dynamic", "-dynamic") for f in explainer_config.linesearch_factors
            ):
                c_min_max_path = os.path.join(
                    self.sparse_dictionary.config.base_path, "c_min_and_maxes.txt"
                )
                c_mins, c_maxs = read_component_bounds(
                    c_min_max_path,
                    selected_components,
                    z_sem.device,
                    bounds_scale=getattr(explainer_config, "component_bounds_scale", 1.0),
                )

            z_base = z_sem.unsqueeze(1)  # [Batch, 1, Dim]
            z_reflected_list = []
            for f in explainer_config.linesearch_factors:
                if f in ("dynamic", "-dynamic") and USE_DYNAMIC_C_MIN_MAX:
                    c_factual = torch.matmul(z_sem, W)  # [Batch, K]
                    u_norm_sq = torch.sum(W * W, dim=0).unsqueeze(0)  # [1, K]
                    # c_int: where concept activation lands with factor=1 step
                    c_int = c_factual - proj_factors * u_norm_sq  # [Batch, K]
                    c_target = torch.where(
                        c_int > c_factual,
                        c_maxs.unsqueeze(0),
                        c_mins.unsqueeze(0),
                    )
                    if f == "-dynamic":
                        c_target = torch.where(
                            c_int > c_factual,
                            c_mins.unsqueeze(0),
                            c_maxs.unsqueeze(0),
                        )
                    step = ((c_factual - c_target) / u_norm_sq).unsqueeze(
                        -1
                    ) * W.permute(1, 0).unsqueeze(
                        0
                    )  # [Batch, K, Dim]
                    z_reflected_list.append(z_base - step)
                elif f == "dynamic":
                    # Fallback when c_min/c_max targeting is disabled: just
                    # take the same one-step cross-projection scaling used
                    # for numeric factors (factor=1.0), relying purely on
                    # _build_balanced_conditioning's norm scaling.
                    z_reflected_list.append(z_base - projections)
                elif f == "-dynamic":
                    z_reflected_list.append(z_base + projections)
                else:
                    z_reflected_list.append(z_base - float(f) * projections)

            z_reflected = torch.stack(z_reflected_list, dim=2)  # [Batch, K, S, Dim]

            # 6. Calculate Distances & Sort
            z_base_expanded = z_base.unsqueeze(2)  # [Batch, 1, 1, Dim]
            distances = torch.norm(z_base_expanded - z_reflected, p=2, dim=-1)
            sorted_indices = torch.argsort(distances, dim=1)
            component_indices = (
                torch.arange(sorted_indices.shape[1])
                .unsqueeze(0)
                .tile([sorted_indices.shape[0], 1])
            )

            return z_reflected, component_indices, distances

    # -------------------------------------------------------------------
    # Component Explanation (mirroring DiffusionAutoencoder)
    # -------------------------------------------------------------------

    def explain_all_components(self, sparse_dictionary=None):
        """Render and analyse every sparse dictionary component.

        Parameters
        ----------
        sparse_dictionary : SparseDictionary or SparseDictionaryConfig, optional
            Replaces the current dictionary (a config is loaded/fitted with
            ``act_size = encoder_dimensions``).

        Notes
        -----
        Encodes the validation split with ``self.encoder`` (batch 128) and
        computes the component activations ``c = z @ components``. Writes,
        under ``<sparse_dictionary.base_path>``, ``c_min_and_maxes.txt``
        (per-component min/max activation, consumed by the ``"dynamic"``
        line-search factor) and, under ``<base_path>/<dictionary type>/``,
        ``correlations.png`` (component/attribute correlations) plus one
        folder of contrastive collages per component from
        :meth:`explain_sparse_component`. The dataset's ``task_config`` is
        temporarily cleared so raw attributes are returned as ``y``.
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
        explanation_path = os.path.join(
            self.config.base_path,
            self.config.sparse_dictionary.sparse_dictionaries_type,
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
        batch_size = 128
        for idx, batch in enumerate(
            torch.utils.data.DataLoader(
                self.generator_datasets[1], batch_size=batch_size
            )
        ):
            _log.info(
                "%s", str(batch_size * idx) + "/" + str(len(self.generator_datasets[1]))
            )
            x, y = batch
            # z_sem returned by self.encode() is set from self.encoder(x) before
            # the VAE-encode + multi-step diffusion trajectory run (see encode()),
            # and is never modified afterward. Calling self.encoder directly gives
            # the same z but skips that unused, expensive trajectory (it was
            # causing CUDA OOM at this batch size since only z is used below).
            with torch.no_grad():
                z = self.encoder(x.to(self.device))
            c = z @ self.sparse_dictionary.get_components().to(self.device)
            y_list.append(y)
            c_list.append(c.detach().cpu())
            z_list.append(z.detach().cpu())

        y_stack = torch.cat(y_list)
        c_stack = torch.cat(c_list)
        z_stack = torch.cat(z_list)

        c_min = c_stack.min(dim=0).values
        c_max = c_stack.max(dim=0).values

        # c_min/c_max are kept in raw CLIP/PLIP embedding space — the same
        # space z_sem, the dictionary components, and c_factual are computed
        # in inside _calculate_z_counterfactuals's "dynamic" branch. They used
        # to be rescaled here into an approximation of the decoder's text-
        # conditioning space, but that rescale used one dataset-averaged
        # scalar to stand in for a per-sample, non-linear operation
        # (_build_balanced_conditioning's own matrix transform followed by a
        # per-sample norm-based rescale), which isn't reproducible from a
        # scalar projection coefficient alone. The decoder already applies
        # its own exact per-sample transform to whatever edited z_sem it's
        # given, so there's no need to anticipate that here.
        with open(
            os.path.join(
                self.sparse_dictionary.config.base_path, "c_min_and_maxes.txt"
            ),
            "w",
        ) as f:
            for i in range(c_min.shape[0]):
                f.write(
                    f"Component {i}: min={c_min[i].item():.4f}, max={c_max[i].item():.4f}\n"
                )

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

        # if not self.latent_model is None:
        #     sampled_x = self.sample_x(self.config.batch_size).detach().cpu()
        #     torchvision.utils.save_image(
        #         sampled_x,
        #         os.path.join(explanation_path, "samples.png"),
        #         nrow=int(math.sqrt(self.config.batch_size)),
        #     )

        result_list = []
        for component_idx in range(self.config.sparse_dictionary.n_components):
            result_list.append(
                self.explain_sparse_component(
                    torch.utils.data.DataLoader(
                        self.generator_datasets[1],
                        # explain_sparse_component_batch does a full encode+decode
                        # diffusion round trip per sample, not just an embedding
                        # lookup, so it needs the same (typically much smaller)
                        # batch size as the rest of the generator, not a fixed 32.
                        batch_size=self.config.batch_size,
                    ),
                    component_idx,
                )
            )

    def explain_sparse_component(self, dataloader, component_idx):
        """Render collages showing the effect of one dictionary component.

        Parameters
        ----------
        dataloader : DataLoader
            Batches of ``(x, y)`` from the validation split.
        component_idx : int
            Column of ``get_components()`` to flip.

        Returns
        -------
        tuple
            ``(x_factual_list, x_counterfactual_list)`` of CPU tensors for
            the first ``config.visualizations_per_component`` images.

        Notes
        -----
        Collages are written by the dataset's ``generate_contrastive_collage``
        into ``<base_path>/<dictionary type>/<component_idx>/``, with the
        component projections before/after the flip in place of class
        confidences.
        """
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
        """Reflect one batch across a single dictionary component and decode.

        Parameters
        ----------
        x_generator : torch.Tensor
            ``[B, 3, H, W]`` images.
        component_idx : int
            Component column to reflect across (normalised to unit length,
            projections taken relative to the dictionary mean ``mu``).

        Returns
        -------
        tuple
            ``(x_counterfactuals, (proj_before, proj_after))``: decoded
            images projected into the validation dataset's space (CPU) and
            the ``[B]`` component projections before and after the
            reflection ``z - 2 <z - mu, w> w``.
        """
        _log.info("%s", "[x_generator.min(), x_generator.max()]")
        _log.info("%s", [x_generator.min(), x_generator.max()])
        _log.info("%s", [x_generator.min(), x_generator.max()])
        _log.info("%s", [x_generator.min(), x_generator.max()])
        z_sem, xT = self.encode(x_generator.to(self.device))
        # w = self.sparse_dictionary.get_components()[:, component_idx].to(self.device)
        w_raw = self.sparse_dictionary.get_components()[:, component_idx].to(
            self.device
        )
        # Normalize w to ensure it is a unit vector
        w = w_raw / torch.norm(w_raw, p=2)
        # z_sem2 = self._calculate_z_counterfactuals(z_sem, w)
        proj_factors = (z_sem - self.sparse_dictionary.mu.to(self.device)) @ w

        _log.info(
            "%s", "proj_factors:" + str(list(proj_factors.detach().cpu().numpy()))
        )
        z_sem_after = z_sem - 2 * proj_factors.unsqueeze(1) * w
        # z_sem_after = z_sem - proj_factors.unsqueeze(1) * w
        proj_factors_after = (
            z_sem_after - self.sparse_dictionary.mu.to(self.device)
        ) @ w
        _log.info(
            "%s",
            "proj_factors_after:"
            + str(list(proj_factors_after.detach().cpu().numpy())),
        )
        x_counterfactuals_generator = self.decode((z_sem_after, xT))
        _log.info(
            "%s",
            "[x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]",
        )
        _log.info(
            "%s", [x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]
        )
        _log.info(
            "%s", [x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]
        )
        _log.info(
            "%s", [x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]
        )
        x_counterfactuals_generator = self.generator_datasets[
            1
        ].project_from_pytorch_default(x_counterfactuals_generator)
        _log.info(
            "%s",
            "[x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]",
        )
        _log.info(
            "%s", [x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]
        )
        _log.info(
            "%s", [x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]
        )
        _log.info(
            "%s", [x_counterfactuals_generator.min(), x_counterfactuals_generator.max()]
        )
        return x_counterfactuals_generator.cpu(), (
            proj_factors.cpu(),
            proj_factors_after.cpu(),
        )

    # def explain_all_components(self, sparse_dictionary=None):
    #     """Visualize all sparse dictionary components."""
    #     if self.sparse_dictionary is None or sparse_dictionary is not None:
    #         if isinstance(sparse_dictionary, SparseDictionary):
    #             self.sparse_dictionary = sparse_dictionary
    #             self.config.sparse_dictionary = copy.deepcopy(sparse_dictionary.config)
    #         else:
    #             if isinstance(sparse_dictionary, SparseDictionaryConfig):
    #                 self.config.sparse_dictionary = sparse_dictionary

    #             # Check for data override in the new sparse dictionary config
    #             sd_data = getattr(self.config.sparse_dictionary, "data", None)
    #             if sd_data:
    #                 print(
    #                     f"PathLDMAutoencoder: Reloading datasets from {sd_data} for component explanation."
    #                 )
    #                 import types

    #                 self.config.data = load_yaml_config(sd_data, DataConfig)
    #                 if isinstance(self.config.data, types.SimpleNamespace):
    #                     self.config.data = DataConfig(**vars(self.config.data))
    #                 self.generator_datasets = get_datasets(self.config.data)
    #                 self.generator_dataset = (
    #                     self.generator_datasets[0] if self.generator_datasets else None
    #                 )

    #             self.config.sparse_dictionary.act_size = self.config.encoder_dimensions
    #             self.load_sparse_dictionary()
    #             if self.sparse_dictionary is None:
    #                 self.fit_sparse_dictionary()

    #     explanation_path = os.path.join(
    #         self.config.base_path,
    #         self.config.sparse_dictionary.sparse_dictionaries_type,
    #     )
    #     Path(explanation_path).mkdir(parents=True, exist_ok=True)

    #     # Resolve which components to explain
    #     component_indices = self._resolve_component_indices(
    #         self.config.sparse_dictionary
    #     )

    #     if component_indices is None:
    #         n_comps = self.config.sparse_dictionary.n_components
    #         if n_comps is None or n_comps <= 0:
    #             n_comps = self.sparse_dictionary.get_components().shape[1]
    #         component_indices = list(range(n_comps))

    #     # Filter out duplicates and invalid indices
    #     component_indices = sorted(list(set(component_indices)))
    #     component_indices = [
    #         i
    #         for i in component_indices
    #         if i < self.sparse_dictionary.get_components().shape[1]
    #     ]

    #     # Prepare one batch of images for visualization and cache their encodings
    #     viz_batch_size = self.config.visualizations_per_component or 10
    #     viz_dataloader = torch.utils.data.DataLoader(
    #         self.generator_datasets[1], batch_size=viz_batch_size, shuffle=False
    #     )
    #     viz_batch = next(iter(viz_dataloader))
    #     x_factual_viz = viz_batch[0].to(self.device)
    #     y_factual_viz = viz_batch[1].cpu() if len(viz_batch) > 1 else []
    #     print(
    #         f"Pre-encoding {len(x_factual_viz)} images for {len(component_indices)} component visualizations..."
    #     )
    #     cached_encodings = self.encode(x_factual_viz)

    #     # Cache paths
    #     y_list_path = os.path.join(explanation_path, "y_list.pt")
    #     c_list_path = os.path.join(explanation_path, "c_list.pt")
    #     z_list_path = os.path.join(explanation_path, "z_list.pt")

    #     # Compute or load component correlations
    #     if (
    #         os.path.exists(y_list_path)
    #         and os.path.exists(c_list_path)
    #         and os.path.exists(z_list_path)
    #     ):
    #         print(f"Loading cached component correlations from {explanation_path}...")
    #         y_list = torch.load(y_list_path)
    #         c_list = torch.load(c_list_path)
    #         z_list = torch.load(z_list_path)
    #     else:
    #         print("Calculating component correlations (this may take a while)...")
    #         y_list, c_list, z_list = [], [], []
    #         task_config_buffer = (
    #             self.generator_datasets[1].task_config
    #             if hasattr(self.generator_datasets[1], "task_config")
    #             else None
    #         )
    #         self.generator_datasets[1].task_config = None

    #         batch_size = getattr(self.config.sparse_dictionary, "batch_size", 10)
    #         for idx, batch in enumerate(
    #             torch.utils.data.DataLoader(
    #                 self.generator_datasets[1], batch_size=batch_size
    #             )
    #         ):
    #             print(f"{batch_size * idx}/{len(self.generator_datasets[1])}")
    #             x, y = batch
    #             z, _ = self.encode(x.to(self.device))
    #             c = z @ self.sparse_dictionary.get_components().to(self.device).to(
    #                 z.dtype
    #             )
    #             y_list.append(y)
    #             c_list.append(c.detach().cpu())
    #             z_list.append(z.detach().cpu())

    #         self.generator_datasets[1].task_config = task_config_buffer

    #         print(f"Saving computed correlations to {explanation_path}...")
    #         torch.save(y_list, y_list_path)
    #         torch.save(c_list, c_list_path)
    #         torch.save(z_list, z_list_path)

    #     for component_idx in component_indices:
    #         print(f"Explaining component {component_idx}...")
    #         self.explain_sparse_component(
    #             None,  # Dataloader not needed if cached
    #             component_idx,
    #             cached_encodings=cached_encodings,
    #             x_factual_viz=x_factual_viz,
    #             y_factual_viz=y_factual_viz,
    #         )

    # def explain_sparse_component(
    #     self,
    #     dataloader,
    #     component_idx,
    #     cached_encodings=None,
    #     x_factual_viz=None,
    #     y_factual_viz=None,
    # ):
    #     """Visualize a single sparse dictionary component."""
    #     x_factual_list = []
    #     x_counterfactual_list = []
    #     y_factual_list = []

    #     current_base_path = os.path.join(
    #         self.config.base_path,
    #         self.config.sparse_dictionary.sparse_dictionaries_type,
    #         str(component_idx),
    #     )
    #     Path(current_base_path).mkdir(parents=True, exist_ok=True)

    #     if cached_encodings is not None:
    #         x_factual_list.extend(list(x_factual_viz.cpu()))
    #         if y_factual_viz is not None:
    #             y_factual_list.extend(list(y_factual_viz))
    #         x_counterfactual, (dot_before, dot_after) = (
    #             self.explain_sparse_component_batch(
    #                 x_factual_viz, component_idx, cached_encodings=cached_encodings
    #             )
    #         )
    #         x_counterfactual_list.extend(list(x_counterfactual.cpu()))
    #     else:
    #         start_idx = 0
    #         for i, batch in enumerate(dataloader):
    #             if (
    #                 self.config.visualizations_per_component is not None
    #                 and start_idx >= self.config.visualizations_per_component
    #             ):
    #                 break

    #             x_factual_list.extend(list(batch[0]))
    #             if len(batch) > 1:
    #                 y_factual_list.extend(list(batch[1]))
    #             x_factual = batch[0].to(self.device)
    #             x_counterfactual, (dot_before, dot_after) = (
    #                 self.explain_sparse_component_batch(x_factual, component_idx)
    #             )
    #             x_counterfactual_list.extend(list(x_counterfactual.cpu()))
    #             start_idx += len(x_factual)

    #     # Generate collage
    #     self.generator_dataset.generate_contrastive_collage(
    #         x_factual_list,
    #         x_counterfactual_list,
    #         [],
    #         [],
    #         y_factual_list,
    #         [],
    #         [],
    #         current_base_path,
    #         0,
    #     )

    #     return x_factual_list, x_counterfactual_list

    # def explain_sparse_component_batch(
    #     self, x_generator, component_idx, cached_encodings=None
    # ):
    #     """Generate counterfactuals for a single sparse component."""
    #     if cached_encodings is not None:
    #         z_sem, stochastic_code = cached_encodings
    #     else:
    #         z_sem, stochastic_code = self.encode(x_generator.to(self.device))

    #     w_raw = (
    #         self.sparse_dictionary.get_components()[:, component_idx]
    #         .to(self.device)
    #         .to(z_sem.dtype)
    #     )
    #     w = w_raw / torch.norm(w_raw, p=2)

    #     mu = (
    #         self.sparse_dictionary.mu.to(self.device).to(z_sem.dtype)
    #         if hasattr(self.sparse_dictionary, "mu")
    #         and self.sparse_dictionary.mu is not None
    #         else torch.zeros_like(z_sem[0])
    #     )

    #     proj_factors = (z_sem - mu) @ w
    #     z_sem_after = z_sem - 2 * proj_factors.unsqueeze(1) * w
    #     proj_factors_after = (z_sem_after - mu) @ w

    #     x_counterfactuals = self.decode_with_modified_embedding(
    #         z_sem_after, stochastic_code, x_generator.shape
    #     )

    #     return x_counterfactuals.cpu(), (
    #         proj_factors.cpu(),
    #         proj_factors_after.cpu(),
    #     )

    # def _resolve_component_indices(self, sd_config, strings_to_resolve=None):
    #     """Resolve component strings or config indices to vocabulary indices."""
    #     component_indices = None

    #     # 1. Try component_indices from config
    #     if (
    #         hasattr(sd_config, "component_indices")
    #         and sd_config.component_indices is not None
    #         and strings_to_resolve is None
    #     ):
    #         component_indices = sd_config.component_indices

    #     # 2. Try strings
    #     else:
    #         if strings_to_resolve is None:
    #             strings_to_resolve = getattr(sd_config, "component_strings", None)

    #         if strings_to_resolve is not None:
    #             if hasattr(self.sparse_dictionary, "get_vocabulary"):
    #                 vocab_list = self.sparse_dictionary.get_vocabulary()
    #                 vocab_list = [v.lower() for v in vocab_list]

    #                 component_indices = []
    #                 for search_term in strings_to_resolve:
    #                     search_term = search_term.lower()
    #                     # Find exact match first
    #                     try:
    #                         found_idx = vocab_list.index(search_term)
    #                         component_indices.append(found_idx)
    #                     except ValueError:
    #                         # Fallback to substring matching
    #                         found = False
    #                         for i, v in enumerate(vocab_list):
    #                             if search_term in v:
    #                                 component_indices.append(i)
    #                                 found = True
    #                                 break
    #                         if not found:
    #                             print(
    #                                 f"!!! [SpLICE] RESOLUTION FAILED: Concept '{search_term}' not found in vocabulary !!!",
    #                                 flush=True,
    #                             )
    #                             component_indices.append(
    #                                 None
    #                             )  # Keep list length consistent
    #             else:
    #                 print(
    #                     "Warning: Sparse dictionary does not support get_vocabulary()",
    #                     flush=True,
    #                 )

    #     return component_indices

    def train_model(
        self,
    ):
        """Finetune PathLDM on the generator's train split.

        Writes ``<base_path>/config.yaml``, wraps the config fields plus
        ``train_dataset`` and ``pathldm_autoencoder=self`` in a namespace
        and replaces ``self.model`` with the result of
        :func:`lora_finetune_pathldm`.

        Raises
        ------
        ValueError
            If no training dataset is available.
        """
        if not os.path.exists(self.config.base_path):
            Path(self.config.base_path).mkdir(parents=True, exist_ok=True)

        save_yaml_config(
            self.config, os.path.join(self.config.base_path, "config.yaml")
        )
        finetune_args = types.SimpleNamespace(**self.config.__dict__)
        finetune_args.train_dataset = (
            self.generator_datasets[0]
            if self.generator_datasets and self.generator_datasets[0] is not None
            else self.generator_dataset
        )
        finetune_args.resume_from_checkpoint = "latest"
        finetune_args.pathldm_autoencoder = self
        if finetune_args.train_dataset is None:
            raise ValueError("PathLDM LoRA finetuning requires a training dataset.")

        _log.info("%s", "Start PathLDM LoRA finetuning")
        self.model = lora_finetune_pathldm(finetune_args)
        _log.info("%s", "Finished PathLDM LoRA finetuning")
