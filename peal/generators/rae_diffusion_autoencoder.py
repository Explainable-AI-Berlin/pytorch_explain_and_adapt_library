"""ImageNet Representation-Autoencoder generator (RAEv2, two stages) with a frozen
OpenAI CLIP ViT-L/14 encoder, exposed through PEAL's InvertibleGenerator /
EditCapableGenerator interface.

Stage 1 (RAEv2 ``RAE``): frozen CLIP patch tokens (16x16x1024 at 256 px) are
the latent; a ViT-XL decoder trained from scratch maps them back to pixels.
Stage 2 (RAEv2 ``DiTwDDTHeadIG``): a flow-matching transformer over that
latent, conditioned on the 768-d CLIP image embedding (``clip.encode_image``),
which is z_sem here -- the same space PEAL's OpenAICLIPEncoder produces and
the MSAE dictionaries decompose. Editing z_sem and re-decoding is therefore a
dictionary-decodable edit, exactly as with the DiffusionAutoencoder.

The class subclasses DiffusionAutoencoder so the SAE / sweep / edit machinery
plugs in unchanged; only construction, training, encode and decode differ.
Latent conventions mirror the DiffAE ones so callers need no changes::

    encode(x)                      -> (z_sem, x_T)            "ddim_inv": ODE inversion
                                   -> (z_sem, x_T, zs)        "ddpm_inv": stochastic inversion
    decode((z_sem, x_T[, zs]))     -> image in generator normalization

x_T is the stage-2 latent state at t=1 (RAEv2 convention: t=1 is noise, t=0
data), zs are the per-step noise maps of the stochastic sampler, so decoding
what encode returned reproduces the input up to model error, and swapping
z_sem edits it while keeping the input's "noise identity" (edit-friendly
inversion, arXiv:2304.06140, transferred to the linear flow schedule).

With ``render_noise: fresh`` encode skips the inversion: x_T is freshly drawn
Gaussian noise (and zs fresh noise maps), so decode((z_sem_edited, x_T)) is a
new sample conditioned on the edited z_sem. The edit is realised much more
fully because nothing but z_sem carries the image, at the price of the input's
details (pose, background, exact crop) being regenerated rather than kept.

Training runs the RAEv2 scripts through peal.generators.rae_pipeline.
"""

import copy
import glob
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from peal.data.dataset_factory import get_datasets
from peal.generators.diffusion_autoencoder import (
    DenormalizationModule,
    DiffusionAutoencoder,
    DiffusionAutoencoderConfig,
    NormalizationModule,
)
from peal.generators.rae_pipeline import (
    RAEPipeline,
    default_raev2_dir,
    expand_path,
    require_raev2,
)
from peal.global_utils import load_yaml_config, save_yaml_config
from peal.log import get_logger

_log = get_logger(__name__)


# what tools/export_rae_weights.py writes; the first one found is loaded
STAGE2_WEIGHT_FILES = ("stage2_ema.pt", "stage2_model.pt")
WEIGHT_FILES = ("decoder.pt", "stats.pt") + STAGE2_WEIGHT_FILES


def resolve_weights_dir(spec):
    """A local folder as is; "hf://<repo_id>[@<revision>]" downloaded into
    $PEAL_RUNS/hf_weights/<repo_id with / -> __> (only the weight files, no
    README or configs) and returned. $PEAL_HF_WEIGHTS_DIR overrides the target."""
    spec = expand_path(spec)
    if not spec.startswith("hf://"):
        if not os.path.isdir(spec):
            raise FileNotFoundError(f"RAE weights folder not found: {spec}")
        return spec
    ref = spec[len("hf://") :]
    repo_id, _, revision = ref.partition("@")
    revision = revision or None
    target_root = os.environ.get("PEAL_HF_WEIGHTS_DIR") or os.path.join(
        os.environ.get("PEAL_RUNS", "peal_runs"), "hf_weights"
    )
    target = os.path.join(target_root, repo_id.replace("/", "__"))
    have = [f for f in WEIGHT_FILES if os.path.isfile(os.path.join(target, f))]
    if (
        "decoder.pt" in have
        and "stats.pt" in have
        and any(f in have for f in STAGE2_WEIGHT_FILES)
    ):
        return target
    from huggingface_hub import snapshot_download

    _log.info(
        "%s",
        f"[RAEDiffusionAutoencoder] downloading {repo_id} ({revision or 'main'}) to {target}",
    )
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        allow_patterns=list(WEIGHT_FILES),
    )
    return target


class RAEDiffusionAutoencoderConfig(DiffusionAutoencoderConfig):
    """Config of the RAEv2-based ImageNet generator (see module docstring)."""

    generator_type: str = "RAEDiffusionAutoencoder"
    base_path: str = "$PEAL_RUNS/imagenet/rae_clip"
    image_size: int = 256
    encoder_dimensions: int = 768
    encoder: Union[str, None] = "clip:ViT-L/14"
    encoder_input_space: str = "pytorch_default"
    # RAEv2 stage configs (OmegaConf yaml) and experiment names under base_path
    stage1_config: str
    stage2_config: str
    stage1_experiment: str = "stage1"
    stage2_experiment: str = "stage2"
    stage1_precision: str = "bf16"
    stage2_precision: str = "bf16"
    stats_num_samples: int = 100000
    stats_batch_size: int = 128
    keep_checkpoints: int = 2
    cache_dir: str = "/tmp/rae_clip_stage2_cache"
    cache_views: str = "original,hflip"
    cache_batch_size: int = 128
    cache_permutation_seed: int = 20260911
    raev2_dir: Union[str, None] = None
    # which stage-2 checkpoint to load: explicit path, else the highest ep-*.pt
    stage2_checkpoint: Union[str, None] = None
    # Pretrained weights folder instead of a training run directory: a local
    # folder or "hf://<repo_id>[@<revision>]" (downloaded once into
    # $PEAL_RUNS/hf_weights). It holds exactly the files tools/export_rae_weights.py
    # writes: decoder.pt, stats.pt (stage 1) and stage2_ema.pt (the EMA state
    # dict alone, no optimizer). When set, it wins over base_path's stage1_assets
    # and stage2/<experiment>/checkpoints, and stage2_checkpoint is ignored.
    weights: Union[str, None] = None
    stage2_weights: str = "ema"
    # Partial inversion (SDEdit-style): invert only up to flow time t_start < 1 and
    # sample back from there. At 1.0 the inverted state carries essentially the whole
    # image and a z_sem edit is ignored at decode time (measured 2026-09-14: swapping
    # z_sem changed the decode by half the reconstruction error); lower values hand
    # more of the content back to the z_sem conditioning at the cost of fidelity.
    inversion_t: float = 1.0
    # "exact": edit-friendly stochastic inversion (noise maps chosen so the sampler
    # lands back on the input; an edited z_sem changes little). "sdedit": start from
    # the noisy state at inversion_t with FRESH noise maps, so the sampler has to
    # regenerate the details from z_sem (SDEdit); only meaningful with ddpm_inv.
    inversion_mode: str = "exact"
    # eta of the stochastic ("ddpm_inv") sampler, 0 = deterministic
    inversion_eta: float = 1.0
    # Where the noise a counterfactual is decoded from comes from:
    #   "inverted": invert the input under its own z_sem (ddim_inv: ODE, ddpm_inv:
    #               edit-friendly), so decode(encode(x)) reproduces x and an edit
    #               keeps the input's details -- but the epoch-12 decoder realised
    #               only ~4 % of a requested z_sem edit this way (freight car run,
    #               2026-09-15) because the inverted state carries the whole image.
    #   "fresh":    no inversion; x_T is fresh Gaussian noise (with inversion_t < 1:
    #               the input latent noised to t_start, i.e. SDEdit) and the
    #               stochastic sampler gets fresh noise maps. The decode is a new
    #               sample from the edited z_sem: the edit shows, the details of
    #               the input are regenerated rather than preserved.
    render_noise: str = "inverted"
    # Classifier-free guidance on the z_sem conditioning (stage 2 was trained with
    # cfg_dropout_prob 0.1 against a learned null CLS): prediction =
    # uncond + scale * (cond - uncond), applied in inversion AND sampling so that
    # decode(encode(x)) still reconstructs. 1.0 = off (RAEv2's shipped sampler
    # config). > 1 amplifies exactly the conditioning difference a z_sem edit
    # introduces while an inverted x_T keeps the input's structure.
    guidance_scale: float = 1.0
    # Guidance scale used during the INVERSION only (None = same as guidance_scale).
    # Text-to-image editing papers invert at a low scale and sample at a high one
    # (edit-friendly DDPM inversion: 3.5 in, 15 out on Stable Diffusion); with the
    # edit-friendly noise maps the reconstruction survives such a mismatch.
    guidance_scale_inversion: Union[float, None] = None
    # bf16 autocast for the deterministic decode; inversion and the stochastic
    # replay always run in fp32 because bf16 rounding compounds ~x2 per step and
    # would break "decode(encode(x)) reproduces x" (measured 3e-3 after 7 steps).
    sampling_autocast: bool = True
    # Opt-in speed knobs for the stochastic (DDPM) path, off by default for the
    # reason above. sde_autocast runs inversion and replay under bf16 autocast;
    # matmul_precision is torch.set_float32_matmul_precision ("highest" = fp32,
    # "high" = TF32 on the tensor cores) and applies to every fp32 matmul.
    sde_autocast: bool = False
    matmul_precision: str = "highest"
    differentiable_decode: bool = False


class _RAEClsEncoder(nn.Module):
    """z_sem = CLIP image embedding through the RAE encoder; expects a [0, 1] image."""

    def __init__(self, rae):
        super().__init__()
        self.rae = rae

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        _, cls = self.rae.encode_with_cls(x.clamp(0, 1))
        return cls.float()


class _Stage2Bundle:
    """What DiffusionAutoencoder code expects to find under self.model."""

    def __init__(self, ddt, transport, time_dist_shift, config, checkpoint, encoder):
        self.ddt = ddt
        self.transport = transport
        self.time_dist_shift = time_dist_shift
        self.config = config
        self.checkpoint = checkpoint
        self.ema_model = types.SimpleNamespace(encoder=encoder)


class RAEDiffusionAutoencoder(DiffusionAutoencoder):
    """
    RAEv2 two-stage ImageNet generator behind the DiffusionAutoencoder API.

    Construction loads the generator datasets, resolves the RAEv2 install and
    weights (``config.weights`` or the run directory ``base_path``), builds
    the stage-1 RAE (frozen CLIP encoder + ViT decoder), the CLS encoder used
    as ``self.encoder`` (z_sem), and the stage-2 flow-matching transformer if
    a checkpoint exists. See the module docstring for the latent conventions
    and the config docstring/comments for the inversion and guidance knobs.

    Parameters
    ----------
    config : str or RAEDiffusionAutoencoderConfig
        Generator yaml path or loaded config.
    predictor_dataset : peal dataset, optional
        Dataset of the classifier being explained; supplies ``dataset_path``
        and ``task_config`` when the generator config lacks them.
    model_dir : str, optional
        Overrides ``config.base_path`` (the training run directory).
    device : str
        Ignored; CUDA is used whenever available.

    Attributes
    ----------
    rae : stage1.RAE
        Stage-1 model (``encode_with_cls``, ``decode``).
    ddt : torch.nn.Module or None
        Stage-2 network; ``None`` until a checkpoint is found.
    model : _Stage2Bundle or None
        Holds ``ddt``, the transport and the time-shift for the samplers.
    encoder : torch.nn.Module
        Maps a dataset-normalised image to the 768-d CLIP embedding (z_sem).
    sample_type : {"ddim_inv", "ddpm_inv"}
        Sampler chosen by ``set_sampler``.
    num_steps : int
        Number of sampler steps.
    latent_size : tuple
        Shape of the stage-2 latent, e.g. ``(1024, 16, 16)``.
    """

    def __init__(self, config, predictor_dataset=None, model_dir=None, device="cpu"):
        nn.Module.__init__(self)
        self.config = load_yaml_config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor_dataset = copy.deepcopy(predictor_dataset)
        if (
            getattr(self.config.data, "dataset_path", None) is None
            and self.predictor_dataset is not None
        ):
            self.config.data.dataset_path = self.predictor_dataset.config.dataset_path
        self.generator_datasets = get_datasets(self.config.data)
        if self.config.task_config is not None:
            self.generator_datasets[0].task_config = self.config.task_config
            self.generator_datasets[1].task_config = self.config.task_config
        elif self.predictor_dataset is not None:
            self.generator_datasets[0].task_config = self.predictor_dataset.task_config
            self.generator_datasets[1].task_config = self.predictor_dataset.task_config
        self.generator_dataset = self.generator_datasets[0]

        self.config.base_path = expand_path(model_dir or self.config.base_path)
        self.model_dir = self.config.base_path
        if self.config.model_type is None and self.config.encoder is not None:
            self.config.model_type = self.config.encoder

        self.pipeline = RAEPipeline(self._pipeline_config(), logger=print)
        os.environ.setdefault("RAE_STAGE2_CACHE", self.pipeline.cfg["cache_dir"])
        self._raev2_on_path()

        self.rae = None
        self.ddt = None
        self.model = None
        self.latent_model = None
        self.sparse_dictionary = None
        self.encoder = None
        self.load_models()
        self.load_sparse_dictionary()
        self.set_sampler(self.config.sampler)

    # ------------------------------------------------------------ plumbing
    def _pipeline_config(self):
        """Flatten the generator config into the plain dict ``RAEPipeline``
        expects (scalars only, paths expanded, ``raev2_dir`` defaulted)."""
        cfg = (
            self.config.model_dump()
            if hasattr(self.config, "model_dump")
            else dict(self.config.__dict__)
        )
        cfg = {
            k: v
            for k, v in cfg.items()
            if not isinstance(v, (dict, list)) or k in ("sampler",)
        }
        cfg["base_path"] = self.config.base_path
        for key in ("stage1_config", "stage2_config", "raev2_dir"):
            if cfg.get(key):
                cfg[key] = expand_path(cfg[key])
        if not cfg.get("raev2_dir"):
            cfg.pop("raev2_dir", None)
        # load_generator_config's defaults without re-reading the yaml
        defaults = {
            "raev2_dir": default_raev2_dir(),
        }
        for k, v in defaults.items():
            cfg.setdefault(k, v)
        return cfg

    def _raev2_on_path(self):
        """Put the RAEv2 checkout's ``src`` and ``.deps`` on ``sys.path`` and
        disable xformers; raises with install instructions if it is missing."""
        # Fails here, with the licence and the install command, rather than as an
        # opaque `ModuleNotFoundError: stage1` inside load_models().
        raev2 = require_raev2(self.pipeline.raev2)
        for p in (os.path.join(raev2, "src"), os.path.join(raev2, ".deps")):
            if p not in sys.path:
                sys.path.insert(0, p)
        os.environ.setdefault("XFORMERS_DISABLED", "1")

    def _weights_dir(self):
        """Folder of pretrained weights (config.weights), resolved and cached."""
        if getattr(self, "_weights_dir_cache", None) is not None:
            return self._weights_dir_cache
        spec = getattr(self.config, "weights", None)
        self._weights_dir_cache = resolve_weights_dir(spec) if spec else None
        return self._weights_dir_cache

    def _asset(self, name):
        """Path of a stage-1 asset (``decoder.pt``/``stats.pt``) in the weights
        folder or the run's assets dir, or ``None`` if absent."""
        wdir = self._weights_dir()
        if wdir is not None:
            path = os.path.join(wdir, name)
            return path if os.path.isfile(path) else None
        path = os.path.join(self.pipeline.assets, name)
        return path if os.path.isfile(path) else None

    def _find_stage2_checkpoint(self):
        """Stage-2 checkpoint to load: exported weights file, else
        ``config.stage2_checkpoint``, else the highest ``ep-*.pt`` of the run."""
        wdir = self._weights_dir()
        if wdir is not None:
            for name in STAGE2_WEIGHT_FILES:
                path = os.path.join(wdir, name)
                if os.path.isfile(path):
                    return path
            return None
        explicit = getattr(self.config, "stage2_checkpoint", None)
        if explicit:
            explicit = expand_path(explicit)
            return explicit if os.path.isfile(explicit) else None
        best, best_epoch = None, -1
        for f in glob.glob(
            os.path.join(self.pipeline.stage2_exp, "checkpoints", "ep-*.pt")
        ):
            m = re.search(r"ep-(\d+)\.pt$", f)
            if m and int(m.group(1)) > best_epoch:
                best, best_epoch = f, int(m.group(1))
        return best

    def _wrap_encoder(self, encoder):
        """Prepend the (de)normalisation that maps dataset tensors into the
        ``config.encoder_input_space`` the CLS encoder expects."""
        normalization = self.generator_dataset.config.normalization
        space = getattr(self.config, "encoder_input_space", "pytorch_default")
        if normalization is None or space == "dataset":
            return encoder
        if space == "pytorch_default":
            return nn.Sequential(
                DenormalizationModule(normalization[0], normalization[1]), encoder
            )
        if space == "legacy":
            return nn.Sequential(
                NormalizationModule(normalization[0], normalization[1]), encoder
            )
        raise ValueError(f"Unknown encoder_input_space: {space!r}")

    def set_encoder(self):
        """No-op: the encoder is the RAE's own CLIP, built in ``load_models``."""
        # the encoder is the RAE's own CLIP; built in load_models()
        return

    def load_models(self):
        """
        Build stage 1 and, if a checkpoint exists, stage 2.

        Stage 1 (``self.rae``, ``self.encoder``) is always constructed from
        the stage-1 yaml; ``decoder_available`` records whether the trained
        decoder and normalisation stats were found. Stage 2 is loaded from
        ``_find_stage2_checkpoint`` (memory-mapped, ``config.stage2_weights``
        selects the ``ema`` or ``model`` state dict) into ``self.ddt`` and
        wrapped in ``self.model``; otherwise both stay ``None`` and only
        ``encode(..., only_semantic=True)`` works.
        """
        from stage1 import RAE  # RAEv2

        precision = getattr(self.config, "matmul_precision", "highest") or "highest"
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = precision != "highest"

        s1 = self.pipeline.stage1_yaml["stage_1"]["params"]
        decoder_path, stats_path = self._asset("decoder.pt"), self._asset("stats.pt")
        # The repo yamls carry <PEAL_RAEV2>; RAEv2 opens the path as given
        # (measured 2026-09-25: "Incorrect path_or_model_id: '<PEAL_BASE>/...'"
        # for the placeholder this replaced). Resolve against this run's RAEv2,
        # not just the default, so an explicit raev2_dir in the config wins.
        decoder_config_path = expand_path(
            s1["decoder_config_path"], raev2_dir=self.pipeline.raev2
        )
        self.rae = RAE(
            encoder_name=s1["encoder_name"],
            resolution=int(s1.get("resolution", 256)),
            decoder_config_path=decoder_config_path,
            pretrained_decoder_path=decoder_path,
            normalization_stat_path=stats_path,
            noise_tau=0.0,
        ).to(self.device)
        self.rae.eval()
        self.rae.requires_grad_(False)
        self.decoder_available = decoder_path is not None and stats_path is not None
        if not self.decoder_available:
            _log.info(
                "%s",
                f"[RAEDiffusionAutoencoder] no trained stage-1 assets under {self.pipeline.assets}; "
                "only encode(..., only_semantic=True) is available",
            )
        self.encoder = self._wrap_encoder(_RAEClsEncoder(self.rae)).to(self.device)

        ckpt = self._find_stage2_checkpoint()
        if ckpt is None or not self.decoder_available:
            self.model = None
            self.ddt = None
            if ckpt is None:
                where = self._weights_dir() or self.pipeline.stage2_exp
                _log.info(
                    "%s",
                    f"[RAEDiffusionAutoencoder] no stage-2 checkpoint under {where}",
                )
            return

        from omegaconf import OmegaConf
        from configs.stage2 import Stage2Config
        from stage2.transport import create_transport
        from utils.model_utils import instantiate_from_config

        os.environ.setdefault("RAE_STAGE2_CACHE", self.pipeline.cfg["cache_dir"])
        cfg = OmegaConf.load(self.pipeline.cfg["stage2_config"])
        # the run's own stage-1 assets, whatever the yaml says (proxies, moved runs)
        cfg.stage_1.params.pretrained_decoder_path = decoder_path
        cfg.stage_1.params.normalization_stat_path = stats_path
        cfg.stage_1.params.decoder_config_path = decoder_config_path
        s2 = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(Stage2Config), cfg)
        )
        s2.post_process()
        s2.prepare_model_params()
        ddt = instantiate_from_config(s2.stage_2)
        # mmap: stage-2 checkpoints carry model+ema+optimizer (~20 GB); a plain load
        # exceeds the 16 GiB Slurm cgroup of the shared nodes, mapped pages are reclaimable
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, ValueError, TypeError):  # legacy (non-zip) format
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
        weights = getattr(self.config, "stage2_weights", "ema")
        sd = state.get(weights, state.get("model", state))
        sd = {
            k[len("module.") :] if k.startswith("module.") else k: v
            for k, v in sd.items()
        }
        missing, unexpected = ddt.load_state_dict(sd, strict=False)
        if missing or unexpected:
            _log.info(
                "%s",
                f"[RAEDiffusionAutoencoder] stage-2 load: missing={len(missing)} unexpected={len(unexpected)}",
            )
        ddt = ddt.to(self.device).eval()
        ddt.requires_grad_(False)
        self.ddt = ddt
        latent = tuple(s2.misc.latent_size)
        shift = math.sqrt(
            (s2.misc.time_dist_shift_dim or math.prod(latent))
            / s2.misc.time_dist_shift_base
        )
        transport = create_transport(config=s2.transport, time_dist_shift=shift)
        self.latent_size = latent
        self.model = _Stage2Bundle(ddt, transport, shift, s2, ckpt, self.encoder)
        _log.info(
            "%s",
            f"[RAEDiffusionAutoencoder] stage 2 loaded from {ckpt} ({weights}); epoch={state.get('epoch')} step={state.get('step')}",
        )

    def set_sampler(self, sampler):
        """
        Choose the sampler from a ``sampler`` config block.

        Parameters
        ----------
        sampler : dict or object or None
            Keys ``type``/``sampler_type`` (``"ddim"`` or ``"ddpm"``, mapped
            to ``sample_type`` ``"<type>_inv"``) and ``num_steps`` (default
            50). Without a type the previous ``sample_type`` is kept
            (initially ``"ddim_inv"``).
        """
        sampler = sampler or {}
        if not isinstance(sampler, dict):
            sampler = dict(getattr(sampler, "__dict__", {}))
        sampler_type = sampler.get("type") or sampler.get("sampler_type")
        if sampler_type is not None:
            self.sample_type = f"{sampler_type}_inv"
        elif not hasattr(self, "sample_type"):
            self.sample_type = "ddim_inv"
        self.num_steps = int(sampler.get("num_steps") or 50)

    # ------------------------------------------------------------ training
    def train_model(self):
        """
        Run both RAEv2 training stages through ``RAEPipeline`` and reload.

        Writes ``config.yaml`` into ``base_path``, drops the loaded models to
        free the GPU for the torchrun ranks, runs ``pipeline.run("all")``,
        then calls ``load_models`` and, if configured, ``fit_sparse_dictionary``.
        """
        Path(self.config.base_path).mkdir(parents=True, exist_ok=True)
        self.config.is_loaded = True
        save_yaml_config(
            self.config, os.path.join(self.config.base_path, "config.yaml")
        )
        # free the GPU for torchrun's ranks
        self.rae, self.ddt, self.model, self.encoder = None, None, None, None
        torch.cuda.empty_cache()
        self.pipeline.run("all")
        self.load_models()
        if self.config.sparse_dictionary is not None:
            self.fit_sparse_dictionary()

    # ------------------------------------------------------------ latents
    def _require_stage2(self):
        if self.model is None or self.ddt is None:
            raise RuntimeError(
                f"No trained stage-2 model under {self.pipeline.stage2_exp} (or no stage-1 decoder); "
                "inversion and decoding need it. Train the generator first, or use encode(..., only_semantic=True)."
            )

    def _time_grid(self, num_steps=None):
        """Descending flow times from 1 to 0 (``num_steps + 1`` values) with
        RAEv2's time-distribution shift, truncated at ``config.inversion_t``."""
        n = int(num_steps or self.num_steps)
        t = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
        shift = self.model.time_dist_shift
        t = shift * t / (1 + (shift - 1) * t)
        ts = t.tolist()
        t_start = float(getattr(self.config, "inversion_t", 1.0) or 1.0)
        if t_start < 1.0:  # partial inversion: same grid, truncated at t_start
            ts = [t_start] + [u for u in ts if u < t_start]
        return ts

    def _predict_x1(self, x, t, z_sem, autocast=False, inverting=False):
        """Stage-2 network output converted to the clean-latent prediction at time t."""
        t_batch = torch.full(
            (x.shape[0],), float(t), device=x.device, dtype=torch.float32
        )
        use_ac = (
            bool(autocast)
            and bool(getattr(self.config, "sampling_autocast", True))
            and x.is_cuda
        )
        scale = float(getattr(self.config, "guidance_scale", 1.0) or 1.0)
        if (
            inverting
            and getattr(self.config, "guidance_scale_inversion", None) is not None
        ):
            scale = float(self.config.guidance_scale_inversion)
        z_sem = z_sem.to(x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
            if scale == 1.0:
                out = self.ddt(x, t_batch, y=z_sem)
                if isinstance(out, tuple):
                    out = out[0]
            else:
                null = getattr(self.ddt, "null_cls", None)
                y_null = (
                    null.detach().to(z_sem.dtype).expand(x.shape[0], -1)
                    if null is not None
                    else torch.zeros_like(z_sem)
                )
                out = self.ddt(
                    torch.cat([x, x], dim=0),
                    torch.cat([t_batch, t_batch], dim=0),
                    y=torch.cat([z_sem, y_null], dim=0),
                )
                if isinstance(out, tuple):
                    out = out[0]
                out_c, out_u = out.float().chunk(2, dim=0)
                out = out_u + scale * (out_c - out_u)
        out = out.float()
        tr = self.model.transport
        if tr.prediction == "x":
            return out
        # velocity parametrisation: v = (x_t - x_1) / t
        return x - t * out

    def _sigma(self, t, s):
        """Stochastic-sampler noise scale for the step t -> s, eta x the DDPM
        posterior std of the linear schedule x_t = (1-t) x_1 + t eps read as the
        Markov chain x_t = r x_s + sqrt(t^2 - r^2 s^2) xi, r = (1-t)/(1-s):
        sigma^2 = s^2 (1 - (r s / t)^2). Until 2026-09-15 this used the
        variance-preserving DDIM formula (s/t)^2 (1 - r^2), which is larger at
        intermediate steps; both satisfy sigma <= s, so both keep the marginals
        and reconstruct exactly with the edit-friendly noise maps."""
        eta = float(getattr(self.config, "inversion_eta", 1.0))
        if eta <= 0 or s <= 0 or t <= s:
            return 0.0
        r = (1.0 - t) / (1.0 - s)
        sigma = eta * s * math.sqrt(max(0.0, 1.0 - (r * s / t) ** 2))
        return min(sigma, s)

    def _ode_sample(self, xT, z_sem, num_steps=None):
        """Deterministic Euler integration of the flow from ``xT`` (t=1) to
        the clean latent (t=0) under the ``z_sem`` conditioning."""
        ts = self._time_grid(num_steps)
        x = xT
        for i in range(len(ts) - 1):
            t, s = ts[i], ts[i + 1]
            x1_hat = self._predict_x1(x, t, z_sem, autocast=True)
            v = (x - x1_hat) / max(t, self.model.transport.t_eps)
            x = x - (t - s) * v
        return x

    def _ode_invert(self, x1, z_sem, num_steps=None):
        """Reverse of ``_ode_sample``: integrate the clean latent ``x1`` back
        to the state at t=1 (or ``inversion_t``) so decoding reproduces it."""
        ts = self._time_grid(num_steps)
        x = x1
        for i in reversed(range(len(ts) - 1)):
            t, s = (
                ts[i],
                ts[i + 1],
            )  # s < t; evaluate the drift at the current (lower) time
            x1_hat = self._predict_x1(x, s, z_sem, inverting=True) if s > 0 else x
            v = (x - x1_hat) / max(s, self.model.transport.t_eps)
            x = x + (t - s) * v
        return x

    def _sde_step(self, x, t, s, z_sem, noise, inverting=False):
        """One stochastic step t -> s; returns ``(x_s, mean, sigma)`` so the
        inversion can solve for the noise map that lands on a given state."""
        x1_hat = self._predict_x1(
            x,
            t,
            z_sem,
            autocast=bool(getattr(self.config, "sde_autocast", False)),
            inverting=inverting,
        ).float()
        eps_hat = (x - (1.0 - t) * x1_hat) / max(t, 1e-6)
        sigma = self._sigma(t, s)
        mean = (1.0 - s) * x1_hat + math.sqrt(max(s * s - sigma * sigma, 0.0)) * eps_hat
        return mean + sigma * noise, mean, sigma

    def _sde_sample(self, xT, z_sem, zs, num_steps=None):
        """Stochastic sampler from ``xT`` using the per-step noise maps
        ``zs`` (shape ``(num_steps, *latent)``)."""
        ts = self._time_grid(num_steps)
        x = xT
        for i in range(len(ts) - 1):
            x, _, _ = self._sde_step(x, ts[i], ts[i + 1], z_sem, zs[i])
        return x

    def _sde_invert(self, x1, z_sem, num_steps=None):
        """Edit-friendly inversion: independent noisy states per grid time, and the
        noise map each sampler step needs to land exactly on the next state."""
        ts = self._time_grid(num_steps)
        states = []
        for t in ts:
            eps = torch.randn_like(x1)
            states.append((1.0 - t) * x1 + t * eps if t > 0 else x1)
        xT = states[0]
        zs = torch.zeros(
            (len(ts) - 1,) + tuple(x1.shape), device=x1.device, dtype=x1.dtype
        )
        if getattr(self.config, "inversion_mode", "exact") == "sdedit":
            return xT, torch.randn_like(zs)
        x = xT
        for i in range(len(ts) - 1):
            t, s = ts[i], ts[i + 1]
            target = states[i + 1]
            _, mean, sigma = self._sde_step(
                x, t, s, z_sem, torch.zeros_like(x), inverting=True
            )
            if sigma > 0:
                zs[i] = (target - mean) / sigma
                x = target
            else:
                x = mean
        return xT, zs

    # ------------------------------------------------------------ interface
    def encode(self, x, t=1.0, only_semantic=False, sample_type=None):
        """
        Encode images into ``(z_sem, x_T[, zs])``.

        Parameters
        ----------
        x : torch.Tensor
            Images of shape ``(B, 3, H, W)`` in the generator dataset's
            normalisation.
        t : float
            Unused; kept for interface compatibility with DiffusionAutoencoder.
        only_semantic : bool
            Return just ``z_sem``; works without a stage-2 model.
        sample_type : {"ddim_inv", "ddpm_inv"}, optional
            Overrides ``self.sample_type``.

        Returns
        -------
        z_sem : torch.Tensor
            Shape ``(B, 768)``: CLIP image embedding.
        x_T : torch.Tensor
            Shape ``(B, *latent_size)``: stage-2 state at t=1 (inverted, or
            fresh noise with ``config.render_noise == "fresh"``).
        zs : torch.Tensor
            Only for ``"ddpm_inv"``: per-step noise maps of shape
            ``(num_steps, B, *latent_size)``.

        Raises
        ------
        RuntimeError
            If stage 2 is missing and ``only_semantic`` is False.
        ValueError
            On an unknown ``sample_type`` or ``config.render_noise``.
        """
        x = x.to(self.device)
        x01 = self.generator_dataset.project_to_pytorch_default(x).clamp(0, 1)
        with torch.no_grad():
            z_lat, cls = self.rae.encode_with_cls(x01)
        z_sem = cls.float()
        if only_semantic:
            return z_sem
        self._require_stage2()
        sample_type = sample_type or getattr(self, "sample_type", "ddim_inv")
        if sample_type not in ("ddim_inv", "ddpm_inv"):
            raise ValueError(f"Unknown sample_type: {sample_type}")
        z_lat = z_lat.float()
        render_noise = getattr(self.config, "render_noise", "inverted") or "inverted"
        if render_noise == "fresh":
            return self._fresh_noise_state(z_lat, z_sem, sample_type)
        if render_noise != "inverted":
            raise ValueError(
                f"Unknown render_noise: {render_noise!r} (inverted | fresh)"
            )
        with torch.no_grad():
            if sample_type == "ddim_inv":
                return z_sem, self._ode_invert(z_lat, z_sem)
            xT, zs = self._sde_invert(z_lat, z_sem)
            return z_sem, xT, zs

    def _fresh_noise_state(self, z_lat, z_sem, sample_type):
        """render_noise "fresh": the state decode() starts from without any inversion.
        At inversion_t = 1 that is pure Gaussian noise; for inversion_t < 1 it is the
        input latent noised to t_start (SDEdit), so a share of the layout survives.
        The stochastic sampler additionally gets fresh per-step noise maps. Drawn
        once per encode() call, so every direction edited from the same batch is
        rendered from the same noise and the renders differ only through z_sem."""
        t_start = float(getattr(self.config, "inversion_t", 1.0) or 1.0)
        eps = torch.randn_like(z_lat)
        xT = eps if t_start >= 1.0 else (1.0 - t_start) * z_lat + t_start * eps
        if sample_type == "ddim_inv":
            return z_sem, xT
        n = len(self._time_grid()) - 1
        zs = torch.randn(
            (n,) + tuple(z_lat.shape), device=z_lat.device, dtype=z_lat.dtype
        )
        return z_sem, xT, zs

    def decode(self, z, t=1.0, sample_type=None):
        """
        Decode ``(z_sem, x_T[, zs])`` back to an image.

        Runs the ODE (``"ddim_inv"``) or stochastic (``"ddpm_inv"``) sampler
        to the clean stage-2 latent and the stage-1 decoder to pixels.
        Gradients flow only with ``config.differentiable_decode``.

        Parameters
        ----------
        z : tuple of torch.Tensor
            As returned by ``encode``; ``zs`` is required for ``"ddpm_inv"``.
        t : float
            Unused; kept for interface compatibility.
        sample_type : {"ddim_inv", "ddpm_inv"}, optional
            Overrides ``self.sample_type``.

        Returns
        -------
        torch.Tensor
            Images of shape ``(B, 3, H, W)`` in the generator dataset's
            normalisation.
        """
        self._require_stage2()
        sample_type = sample_type or getattr(self, "sample_type", "ddim_inv")
        grad = (
            bool(getattr(self.config, "differentiable_decode", False))
            and torch.is_grad_enabled()
        )
        with torch.set_grad_enabled(grad):
            if sample_type == "ddim_inv":
                z_sem, xT = z
                x1 = self._ode_sample(
                    xT.to(self.device).float(), z_sem.to(self.device).float()
                )
            elif sample_type == "ddpm_inv":
                z_sem, xT, zs = z
                x1 = self._sde_sample(
                    xT.to(self.device).float(),
                    z_sem.to(self.device).float(),
                    zs.to(self.device).float(),
                )
            else:
                raise ValueError(f"Unknown sample_type: {sample_type}")
            img01 = self.rae.decode(x1).float().clamp(0, 1)
        return self.generator_dataset.project_from_pytorch_default(img01)

    def sample_z(self, batch_size=1):
        """
        Draw a random latent ``(z_sem, x_T)``.

        Parameters
        ----------
        batch_size : int
            Number of latents.

        Returns
        -------
        z_sem : torch.Tensor
            Shape ``(batch_size, encoder_dimensions)``: Gaussian scaled to
            the typical CLIP embedding norm (~16).
        x_T : torch.Tensor
            Shape ``(batch_size, *latent_size)``: standard Gaussian noise.
        """
        # CLIP embeddings have no simple prior; a Gaussian of matching scale keeps
        # the interface alive (ImageNet CLIP ViT-L/14 embeddings have norm ~16).
        z_sem = torch.randn(
            batch_size, self.config.encoder_dimensions, device=self.device
        )
        z_sem = z_sem * (16.0 / math.sqrt(self.config.encoder_dimensions))
        latent = getattr(self, "latent_size", (1024, 16, 16))
        xT = torch.randn((batch_size,) + tuple(latent), device=self.device)
        return z_sem, xT

    def sample_x(self, batch_size=1):
        """
        Sample images by decoding ``sample_z`` (with fresh noise maps for the
        stochastic sampler).

        Parameters
        ----------
        batch_size : int
            Number of images.

        Returns
        -------
        torch.Tensor
            Images of shape ``(batch_size, 3, H, W)``.
        """
        z_sem, xT = self.sample_z(batch_size)
        if getattr(self, "sample_type", "ddim_inv") == "ddpm_inv":
            n = len(self._time_grid()) - 1
            zs = torch.randn((n,) + tuple(xT.shape), device=self.device)
            return self.decode((z_sem, xT, zs))
        return self.decode((z_sem, xT))
