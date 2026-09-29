"""PathLDM (histopathology latent diffusion) generator for PEAL.

Wraps the vendored PathLDM model (``peal/dependencies/PathLDM``, PLIP
text-conditioned latent diffusion trained on pathology tiles) as an
``EditCapableGenerator``: images are mapped through the first-stage VAE,
noised to a time fraction ``t`` (``encode``), denoised with a DDIM-style
loop (``decode``) and edited with a latent-space FastDiME procedure driven
by classifier gradients (``FastDiME``/``edit``), which is what the
counterfactual explainers call for Camelyon-style datasets. Weights are
resolved from ``$PEAL_PLIP_DIR`` or ``peal/dependencies/plip_imagenet_finetune``.
"""

import os
import types
import shutil
import torch
import io
import blobfile as bf
from datetime import datetime
from torchvision.transforms import transforms, ToPILImage
from pathlib import Path
from torch import nn
from types import SimpleNamespace
from typing import Union
from peal._optional import require
from peal.dependencies.PathLDM.ldm.models.diffusion.ddim import DDIMSampler
from peal.generators.interfaces import EditCapableGenerator
from peal.global_utils import load_yaml_config, generate_smooth_mask
from peal.dependencies.PathLDM.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.nn import functional as F

# from peal.dependencies.DiME.main import main as dime_main
from peal.dependencies.ace.guided_diffusion import logger
from peal.dependencies.ace.guided_diffusion.resample import (
    create_named_schedule_sampler,
)
from peal.dependencies.ace.guided_diffusion.train_util import TrainLoop
from peal.data.dataloaders import get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.explainers.counterfactual_explainer import ACEConfig
from peal.training.loggers import log_images_to_writer
from peal.generators.interfaces import GeneratorConfig
from peal.data.interfaces import DataConfig
from peal.training.trainers import distill_predictor
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


# Debug image dumps. Previously hardcoded to a co-author's home directory; now under
# $PEAL_RUNS so the module is portable.
_DEBUG_DIR = os.path.join(os.environ.get("PEAL_RUNS", "peal_runs"), "debug")


class DDPMPathLDMConfig(GeneratorConfig):
    """
    Config of ``DDPMPathLDM``.

    Only a subset of the fields is read by the generator: ``ckpt_path`` and
    ``config_path`` (PathLDM checkpoint and OmegaConf yaml), ``data``,
    ``base_path``, ``prompt``, ``vae_latent_shape``, ``guidence_scale``,
    ``eta``, ``stochastic``, ``batch_size``, ``timestep_respacing``, the
    FastDiME knobs (``grad_threshold``, ``guided_iterations``,
    ``use_logits``, ``l1_loss``, ``l2_loss``, ``steps_number``,
    ``self_optimized``, ``warmup_steps``, ``use_gussian_blur_masking``) and
    the training fields consumed by ``train_model``. The remaining UNet and
    diffusion fields are inherited from the guided-diffusion DDPM config
    and currently unused because the architecture comes from
    ``config_path``.
    """

    generator_type: str = "DDPM"
    """
    The type of generator that shall be used.
    """
    base_path: str = "peal_runs/ddpm"
    """
    The path where the generator is stored.
    """
    data: DataConfig = DataConfig()
    """
    The config of the data.
    """
    num_channels: int = 256
    """
    The number of channels
    """
    ckpt_path: str = os.path.join(_PLIP_DIR, "checkpoints", "epoch_3.ckpt")
    config_path: str = os.path.join(_PLIP_DIR, "configs", "08-03T09-35-project.yaml")
    image_size: Union[int, type(None)] = None
    num_res_blocks: int = 2
    num_heads: int = 4
    num_heads_upsample: int = -1
    num_head_channels: int = -1
    attention_resolutions: str = "32,16,8"
    channel_mult: str = ""
    dropout: float = 0.0
    class_cond: bool = False
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    use_fp16: bool = False
    use_new_attention_order: bool = False
    schedule_sampler: str = "uniform"
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_anneal_steps: int = 0
    batch_size: int = 1
    microbatch: int = -1  # -1 disables microbatches
    ema_rate: str = "0.9999"  # comma-separated list of EMA values
    log_interval: int = 10
    save_interval: int = 1000
    max_steps: int = 10000
    resume_checkpoint: str = ""
    fp16_scale_growth: float = 1e-3
    output_path: str = "peal_runs/ddpm/outputs"
    gpus: str = ""
    use_hdf5: bool = False
    learn_sigma: bool = True
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str = "50"
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False
    stochastic: bool = True
    x_selection: Union[list, type(None)] = None
    is_trained: bool = False
    best_fid: float = 1e9
    prompt: str = " "
    vae_latent_shape: Union[list, type(None)] = [3, 32, 32]
    guidence_scale: float = 1.0
    eta: float = 0.0
    grad_threshold: float = 0.0
    guided_iterations: int = 9999999
    use_logits: bool = True
    l1_loss: float = 0.0
    l2_loss: float = 0.0
    steps_number: int = 25
    self_optimized: bool = False
    warmup_steps: int = 10
    use_gussian_blur_masking: bool = True


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.

    Raises
    ------
    ImportError
        If the optional ``mpi4py`` dependency is not installed.
    """
    MPI = require("mpi4py.MPI", "mpi", "broadcasting checkpoints across MPI ranks")

    chunk_size = 2**30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return torch.load(io.BytesIO(data), **kwargs)


def load_model_from_config(config, ckpt, device):
    """
    Instantiate the PathLDM model from an OmegaConf config and load weights.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Config with a ``model`` node accepted by ``instantiate_from_config``.
    ckpt : str
        Path of a Lightning checkpoint with a ``state_dict`` entry; loaded
        with ``strict=False``.
    device : torch.device or str
        Device the model is moved to.

    Returns
    -------
    torch.nn.Module
        The model in eval mode.
    """
    _log.info("%s", f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def get_model(config_path, device, checkpoint):
    """
    Load a PathLDM model from a yaml config and checkpoint.

    The ``ckpt_path`` entries of the first-stage and UNet sub-configs are
    deleted so that no separately stored sub-checkpoints are required.

    Parameters
    ----------
    config_path : str
        Path of the OmegaConf yaml.
    device : torch.device or str
        Target device.
    checkpoint : str
        Path of the full-model checkpoint.

    Returns
    -------
    torch.nn.Module
        The loaded model in eval mode.
    """
    config = OmegaConf.load(config_path)
    del config["model"]["params"]["first_stage_config"]["params"]["ckpt_path"]
    del config["model"]["params"]["unet_config"]["params"]["ckpt_path"]
    model = load_model_from_config(config, checkpoint, device)
    return model


class DDPMPathLDM(EditCapableGenerator):
    """
    PathLDM latent diffusion model as an edit-capable PEAL generator.

    Images are exchanged with the rest of PEAL in the value range of
    ``self.dataset`` (built from ``config.data``); internally the first-stage
    VAE works on ``[-1, 1]`` images and ``[B, 4, h, w]``-style latents whose
    shape is ``config.vae_latent_shape``. Text conditioning is the fixed
    ``config.prompt`` with classifier-free guidance when
    ``guidence_scale > 1``.

    Parameters
    ----------
    config : str or dict or DDPMPathLDMConfig
        Config (path or object) loaded with ``load_yaml_config``.
    model_dir : str, optional
        Run directory; defaults to ``config.base_path``.
    device : torch.device or str, optional
        Device for the model and all sampling.
    predictor_dataset : PealDataset, optional
        Unused.

    Attributes
    ----------
    model : torch.nn.Module
        The PathLDM ``LatentDiffusion`` model.
    dataset : PealDataset
        Training split of ``config.data``, used for value-range projection.
    noise_fn : callable
        ``torch.randn_like`` or ``torch.zeros_like`` per ``config.stochastic``.
    """

    def __init__(self, config, model_dir=None, device="cpu", predictor_dataset=None):
        """Load the config, the PathLDM model and the dataset; copy the edit knobs."""
        super().__init__()
        self.predictor_distilled = None
        self.config = load_yaml_config(config)
        self.model_config_path = self.config.config_path
        self.model_path = self.config.ckpt_path
        self.model = get_model(self.model_config_path, device, self.model_path)
        self.dataset = get_datasets(self.config.data)[0]
        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data_test")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals_test")
        self.prompt = self.config.prompt
        self.shape = self.config.vae_latent_shape
        # self.model, self.diffusion = create_model_and_diffusion(**self.config.__dict__)
        self.device = device
        self.guidence_scale = self.config.guidence_scale
        self.grad_threshold = self.config.grad_threshold
        self.guided_iterations = self.config.guided_iterations
        self.use_logits = self.config.use_logits
        self.l1_loss = self.config.l1_loss
        self.l2_loss = self.config.l2_loss
        self.steps_number = self.config.steps_number
        self.do_classifier_free_guidance = True if self.guidence_scale > 1.0 else False
        self.self_optamized_masking = self.config.self_optimized
        self.warmup_steps = self.config.warmup_steps
        self.use_gussian_blur_masking = self.config.use_gussian_blur_masking
        # self.model.to(device)
        # self.model_path = os.path.join(self.model_dir, "final.cpl")
        # if os.path.exists(self.model_path) and self.config.is_trained:
        #     print("load ddpm model weights!!!")
        #     self.model.load_state_dict(torch.load(self.model_path, map_location=device))

        # else:
        #     self.model_path = os.path.join(self.model_dir, "final.pt")
        #     if os.path.exists(self.model_path) and self.config.is_trained:
        #         print("load ddpm model weights!!!")
        #         self.model.load_state_dict(
        #             load_state_dict(self.model_path, map_location=device)
        #         )

        #     else:
        #         print("No ddpm model weights yet!!!")
        #         if os.path.exists(self.model_dir):
        #             shutil.move(
        #                 self.model_dir,
        #                 self.model_dir
        #                 + "_old"
        #                 + datetime.now().strftime("%Y%m%d_%H%M%S"),
        #             )

        #         Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # self.config.is_trained = True
        self.eta = self.config.eta
        self.noise_fn = torch.randn_like if self.config.stochastic else torch.zeros_like
        self.vggloss = None

    def get_unconditional_token(self, batch_size):
        """Return ``batch_size`` empty prompts for classifier-free guidance."""
        return [""] * batch_size

    def get_conditional_token(self, batch_size, summary):
        """
        Return ``batch_size`` copies of ``self.prompt``.

        Parameters
        ----------
        batch_size : int
            Number of prompts.
        summary : str
            Ignored; the fixed ``config.prompt`` is used instead.

        Returns
        -------
        list of str
        """

        # append tumor and TIL probability to the summary
        tumor = [self.prompt] * (batch_size)

        return [t for t in tumor]

    def sample_x(self, batch_size=None, renormalize=True):
        """
        Sample images from the prompt with a 50-step DDIM sampler.

        Parameters
        ----------
        batch_size : int, optional
            Defaults to ``config.batch_size``.
        renormalize : bool, optional
            Project the ``[0, 1]`` decodes into the dataset's value range.
            Must be ``True``; the code path for ``False`` leaves the result
            undefined.

        Returns
        -------
        torch.Tensor
            Images ``[batch_size, 3, H, W]``.
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
                self.shape,
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
            sample = self.dataset.project_to_pytorch_default(x_samples_ddim)

        return sample

    @torch.enable_grad()
    def vae_latent_encoder(self, image_tensor):
        """
        Encode ``[0, 1]`` images to scaled first-stage latents.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Images ``[B, 3, H, W]`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Latents with ``requires_grad`` enabled.
        """
        image_tensor = image_tensor * 2.0 - 1.0
        vae_encode = self.model.encode_first_stage(image_tensor)
        vae_encode = self.model.get_first_stage_encoding(vae_encode)

        return vae_encode.requires_grad_(True)

    @torch.enable_grad()
    def vae_latent_decoder(self, latnet_tensor):
        """
        Decode first-stage latents to ``[0, 1]`` images.

        Parameters
        ----------
        latnet_tensor : torch.Tensor
            Latents as returned by ``vae_latent_encoder``.

        Returns
        -------
        torch.Tensor
            Images ``[B, 3, H, W]`` clamped to ``[0, 1]``.
        """

        vae_decoded = self.model.decode_first_stage(latnet_tensor)
        vae_decoded = ((vae_decoded / 2.0) + 0.5).clamp(0, 1)

        return vae_decoded

    def time_steps_respacing(self, num_inference_steps, num_timesteps):
        """
        Evenly spaced timesteps ``1, 1+c, 1+2c, ...`` below ``num_timesteps``.

        Parameters
        ----------
        num_inference_steps : int
            Desired number of steps; ``c = num_timesteps // num_inference_steps``.
        num_timesteps : int
            Exclusive upper bound.

        Returns
        -------
        torch.Tensor
            1D ascending integer timesteps.
        """
        c = num_timesteps // num_inference_steps
        respaced_timesteps = torch.arange(1, num_timesteps, c)
        return respaced_timesteps

    @torch.enable_grad()
    def encode(self, x, t, stochastic="fully", num_steps=None):
        """
        Encode images and noise the latents to time fraction ``t``.

        Parameters
        ----------
        x : torch.Tensor
            Images ``[B, 3, H, W]`` in ``[0, 1]``.
        t : float
            Fraction of ``model.num_timesteps`` to noise to.
        stochastic : str, optional
            Only ``"fully"`` (fresh Gaussian noise via ``q_sample``) is
            implemented; other values leave the result undefined.
        num_steps : int, optional
            Unused.

        Returns
        -------
        torch.Tensor
            Noisy latents ``x_t`` on ``self.device``.
        """
        # encode the image
        x = self.vae_latent_encoder(x)
        # respaced_timesteps = time_steps_respacing(num_steps, model.num_timesteps)
        t = int(t * self.model.num_timesteps)

        x = x.to(self.device)
        timestep = (
            torch.tensor(
                t,
            )
            .unsqueeze(0)
            .to(x)
            .long()
        )
        _log.info("%s", f"Encoding timestep: {timestep}")
        if stochastic == "fully":
            noise = torch.randn_like(x).to(self.device)
            xt = self.model.q_sample(x, timestep, noise=noise)

        return xt

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

    def _calculate_diffusion_parameters(self, t: float, num_steps: int, eta: float):
        """
        Calculates respaced timesteps, alpha values, and sigmas for the diffusion process.

        Args:
            model: The diffusion model instance.
            t (float): The fraction of total timesteps to decode from (0.0 to 1.0).
            num_steps (int): The number of decoding steps.
            eta (float): The eta parameter for DPM-Solver (controls stochasticity).

        Returns:
            tuple: A tuple containing:
                - respaced_timesteps (torch.Tensor): The timesteps used for decoding.
                - alphas (torch.Tensor): Cumulative product of alphas for each respaced timestep (flipped).
                - alphas_prev (torch.Tensor): Cumulative product of alphas for previous respaced timesteps (flipped).
                - sqrt_one_minus_alphas (torch.Tensor): Square root of (1 - alphas) for each respaced timestep (flipped).
                - sigmas (torch.Tensor): Sigma values for each respaced timestep.
        """
        final_t = int(t * self.model.num_timesteps)
        _log.info("%s", f"Final timestep for decoding: {final_t}")

        respaced_timesteps = self.time_steps_respacing(num_steps, final_t)

        # Ensure respaced_timesteps are within bounds of model.alphas_cumprod
        # Clamp to avoid out-of-bounds indexing if respaced_timesteps go beyond model.num_timesteps - 1
        respaced_timesteps = torch.clamp(
            respaced_timesteps, 0, self.model.num_timesteps - 1
        )

        alphas = self.model.alphas_cumprod[respaced_timesteps]
        # alphas_prev calculation for DDIM/DPM-Solver like steps.
        # It takes the first alpha (which corresponds to the largest timestep after flipping)
        # and then all but the last alpha (which corresponds to the smallest timestep after flipping).
        alphas_prev = torch.cat([alphas[0:1], alphas[:-1]], dim=0).flip(0)
        alphas = alphas.flip(
            0
        )  # Flip alphas for reverse process (from t_N down to t_0)
        sqrt_one_minus_alphas = torch.sqrt(1.0 - alphas)
        sigmas = eta * torch.sqrt(
            (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
        )

        return respaced_timesteps, alphas, alphas_prev, sqrt_one_minus_alphas, sigmas

    def _apply_model_and_guidance(
        self, x_in_for_model, t_in, c_in_for_model, guidance_scale
    ):
        """
        Applies the diffusion model and performs classifier-free guidance.

        Args:
            model: The diffusion model instance.
            x_in_for_model (torch.Tensor): The input latent to the model (potentially concatenated).
            t_in (torch.Tensor): The timesteps tensor for the model.
            c_in_for_model (torch.Tensor): The conditioning tensor for the model (potentially concatenated).
            guidance_scale (float): Classifier-free guidance scale.

        Returns:
            torch.Tensor: The guided noise prediction (e_t), with original batch size.
        """
        # Model applies to potentially concatenated x_in and c_in
        model_output = self.model.apply_model(x_in_for_model, t_in, c_in_for_model)

        if guidance_scale > 1.0:
            # If guidance was applied, model_output is chunked into unconditional and conditional noise predictions
            e_t_uncond, e_t_cond = model_output.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t_cond - e_t_uncond)
        else:
            # If no guidance, model_output is directly the noise prediction
            e_t = model_output
        return e_t

    def _calculate_denoised_latent(
        self,
        x_current_latent_input_to_model: torch.Tensor,  # The latent that was fed to the model (potentially concatenated)
        e_t: torch.Tensor,  # The guided noise prediction (already has original batch_size)
        alpha_t_scalar: torch.Tensor,  # Scalar alpha_t for current step
        alpha_prev_scalar: torch.Tensor,  # Scalar alpha_prev for current step
        sigma_t_scalar: torch.Tensor,  # Scalar sigma_t for current step
        sqrt_one_minus_alpha_t_scalar: torch.Tensor,  # Scalar sqrt(1-alpha_t) for current step
        batch_size: int,
        guidance_active: bool,  # Flag to indicate if guidance was active in this step
        device: torch.device,
    ):
        """
        Performs the DDIM/DPM-Solver denoising step to predict the previous latent state.

        Args:
            x_current_latent_input_to_model (torch.Tensor): The latent input to the model for the current step.
                                                            This might be concatenated (conditional + unconditional).
            e_t (torch.Tensor): The guided noise prediction from the model.
            alpha_t_scalar (torch.Tensor): Scalar alpha_t for the current timestep.
            alpha_prev_scalar (torch.Tensor): Scalar alpha_prev for the previous timestep.
            sigma_t_scalar (torch.Tensor): Scalar sigma_t for the current timestep.
            sqrt_one_minus_alpha_t_scalar (torch.Tensor): Scalar sqrt(1 - alpha_t) for the current timestep.
            batch_size (int): The original batch size.
            guidance_active (bool): True if guidance was applied in this step (x_current_latent_input_to_model was concatenated).
            device (torch.device): The device.

        Returns:
            tuple: ``(x_prev, pred_x0, dir_xt)`` - the denoised latent, the
                predicted clean latent and the direction term.
        """
        # Reshape scalar alpha/sigma values to match latent tensor dimensions for element-wise multiplication
        a_t_reshaped = alpha_t_scalar.view(1, 1, 1, 1).expand(batch_size, 1, 1, 1)
        a_prev_reshaped = alpha_prev_scalar.view(1, 1, 1, 1).expand(batch_size, 1, 1, 1)
        sigma_t_reshaped = sigma_t_scalar.view(1, 1, 1, 1).expand(batch_size, 1, 1, 1)
        sqrt_one_minus_at_reshaped = sqrt_one_minus_alpha_t_scalar.view(
            1, 1, 1, 1
        ).expand(batch_size, 1, 1, 1)

        # In the original code, when guidance_scale > 1.0, x_in.chunk(2)[0] is used for pred_x0.
        # This corresponds to the conditional part of the concatenated latent.
        x_for_pred_x0 = (
            x_current_latent_input_to_model.chunk(2)[0]
            if guidance_active
            else x_current_latent_input_to_model
        )

        # DDIM/DPM-Solver formula for predicting x_0
        pred_x0 = (
            x_for_pred_x0 - sqrt_one_minus_at_reshaped * e_t
        ) / a_t_reshaped.sqrt()

        # Calculate direction to x_t
        dir_xt = (1.0 - a_prev_reshaped - sigma_t_reshaped**2).sqrt() * e_t

        # Calculate x_{t-1}
        x_prev = a_prev_reshaped.sqrt() * pred_x0 + dir_xt

        return x_prev, pred_x0, dir_xt

    def _save_and_display_intermediate_image(
        self, x_in_latent: torch.Tensor, step_idx: int
    ):
        """
        Decodes a latent representation, saves it as an intermediate image, and displays it.

        Args:
            latent_decoder_func (callable): Function to decode latent to image.
            x_in_latent (torch.Tensor): The latent representation to decode.
            step_idx (int): The current decoding step index (for filename).
        """
        # Detach from graph, move to CPU, and convert to PIL image
        decoded_image = self.vae_latent_decoder(x_in_latent)[0].cpu()
        from torchvision.transforms import ToPILImage

        to_pil = ToPILImage()
        pil_image = to_pil(decoded_image)
        pil_image.save(f"decoded_step_custom_{step_idx+1}.png")
        pil_image.show()  # Display the image
        _log.info(
            "%s",
            f"Saved and displayed intermediate image: decoded_step_custom_{step_idx+1}.png",
        )

    def decode(self, z, t: float, stochastic: str = "fully", num_steps: int = 50):
        """
        Denoise latents from time fraction ``t`` to 0 and decode them.

        Runs a DDIM-style loop over ``num_steps`` respaced timesteps with
        the prompt conditioning, classifier-free guidance
        (``guidence_scale``) and stochasticity ``config.eta``, then decodes
        the result with the first-stage VAE.

        Parameters
        ----------
        z : torch.Tensor
            Noisy latents, e.g. from ``encode``.
        t : float
            Fraction of ``model.num_timesteps`` the latents were noised to.
        stochastic : str, optional
            Unused; stochasticity is controlled by ``config.eta``.
        num_steps : int, optional
            Number of denoising steps. Defaults to 50.

        Returns
        -------
        torch.Tensor
            Images ``[B, 3, H, W]`` in ``[0, 1]``.
        """
        batch_size = z.size(0)
        # Fixed guidance scale as per original function

        # 1. Prepare conditioning embeddings
        uc, cc = self._prepare_conditioning(batch_size, self.prompt)

        # 2. Calculate diffusion parameters (timesteps, alphas, sigmas)
        respaced_timesteps, alphas, alphas_prev, sqrt_one_minus_alphas, sigmas = (
            self._calculate_diffusion_parameters(t, num_steps, self.eta)
        )

        x_current_latent = z.to(self.device)  # This latent will be updated in each step

        # 3. Iterative denoising loop
        for idx, timestep in enumerate(respaced_timesteps.flip(0)):
            # (A) Save and display intermediate image of the *current* latent BEFORE processing this step
            # self._save_and_display_intermediate_image(latent_decoder, x_current_latent, idx)
            # print(f"x_current_latent shape before step {idx+1} calculation: {x_current_latent.shape}")

            # (B) Prepare inputs for the model application based on guidance_scale
            # x_in_for_model and c_in_for_model will be passed to model.apply_model
            x_in_for_model = x_current_latent
            c_in_for_model = uc
            guidance_active_in_step = (
                False  # Flag to track if guidance was applied in this specific step
            )

            if self.guidence_scale > 1.0:
                c_in_for_model = torch.cat([uc, cc], dim=0)
                x_in_for_model = torch.cat([x_current_latent, x_current_latent], dim=0)
                guidance_active_in_step = True

            t_in = torch.full(
                (x_in_for_model.size(0),), timestep, device=x_in_for_model.device
            ).long()

            # print(f"Decoding step {idx+1}/{len(respaced_timesteps)}, timestep: {timestep}")
            # print(f"t_in shape: {t_in.shape}, x_in_for_model shape: {x_in_for_model.shape}, c_in_for_model shape: {c_in_for_model.shape}")

            # (C) Apply model and apply guidance to get the noise prediction (e_t)
            e_t = self._apply_model_and_guidance(
                x_in_for_model, t_in, c_in_for_model, self.guidence_scale
            )

            # (D) Perform denoising step calculation to get x_prev
            x_prev, _, _ = self._calculate_denoised_latent(
                x_in_for_model,
                e_t,
                alphas[idx],
                alphas_prev[idx],
                sigmas[idx],
                sqrt_one_minus_alphas[idx],
                batch_size,
                guidance_active_in_step,
                self.device,
            )

            # (E) Update x_current_latent for the next iteration
            x_current_latent = x_prev

            # print(f"x_current_latent shape after step {idx+1} update: {x_current_latent.shape}")

        # 4. Final decoding of the latent representation
        decoded_image = self.vae_latent_decoder(x_current_latent)
        return decoded_image

    # def decode(self, z, t, stochastic=None, num_steps=None):
    #     batch_size = z.size(0)
    #     ut = self.get_unconditional_token(batch_size)
    #     uc = self.model.get_learned_conditioning(ut)

    #     ct = self.get_conditional_token(batch_size, self.prompt)
    #     cc = self.model.get_learned_conditioning(ct)
    #     c_in = torch.cat([uc, cc], dim=0)
    #     if stochastic is None:
    #         stochastic = self.config.stochastic
    #     # decode the image
    #     noise_fn = torch.randn_like if stochastic else torch.zeros_like
    #     if isinstance(z, list) and len(z) == 1:
    #         z = z[0]
    #     z = z.to(self.device)
    #     respaced_steps = int(t * int(self.config.timestep_respacing))
    #     timesteps = list(range(respaced_steps))[::-1]
    #     for idx, timestep in enumerate(timesteps):
    #         t = torch.tensor([t] * z.size(0), device=z.device)
    #         # get the mean and variance
    #         model_mean, posterior_variance, posterior_log_variance = (
    #             self.model.p_mean_variance(x=z, t=t,c = c_in ,clip_denoised=False)
    #         )
    #         z = model_mean
    #         if idx != (respaced_steps - 1):

    #             if stochastic:
    #                 z += torch.exp(0.5 * posterior_log_variance * noise_fn(z))
    #     x = self.model.decode_first_stage(z)
    #     return x

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
        """
        Classifier-guidance gradient of the target-class score w.r.t. ``x_t``.

        The latent (or image, with ``resize``) is decoded, mapped with
        ``kwargs["generator_to_classifier"]``, resized to
        ``predictor_img_size`` and scored; the negative target logit (or
        log-softmax) scaled by ``s`` is differentiated w.r.t. the input.

        Parameters
        ----------
        x_t : torch.Tensor
            Current latents (or images when ``resize`` is true).
        y : torch.Tensor
            Target class per sample.
        resize : bool
            If true ``x_t`` is treated as an image and not decoded.
        classifier : torch.nn.Module
            Predictor providing the guidance.
        s : float, optional
            Scale of the loss (``temperature`` in the explainer config).
        use_logits : bool, optional
            Use raw logits instead of log-softmax.
        predictor_img_size : tuple of int
            Spatial size the classifier expects.
        lr, momentum, optimizer
            Unused.
        threshold : float, optional
            If set, gradient entries below ``threshold`` times the per-sample
            max magnitude are zeroed.
        **kwargs
            ``generator_to_classifier``: value-range mapping callable.

        Returns
        -------
        torch.Tensor
            Gradient with the shape of ``x_t``.
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
        Gradient of the L1/L2/perceptual distance to the original latents.

        Parameters
        ----------
        x_tau : torch.Tensor
            Latents of the original image.
        z_t : torch.Tensor
            Current noisy latents (L1/L2 terms are taken w.r.t. these).
        x_t : torch.Tensor
            Current clean latent estimate (perceptual term w.r.t. these).
        alpha_t : torch.Tensor
            Time dependent constant used to scale the perceptual gradient.
        l1_loss, l2_loss : float
            Weights of the L1 and L2 terms; 0 disables a term.
        l_perc : callable or None
            Perceptual loss ``l_perc(x_t, x_tau)``; ``None`` disables it.
        scale_grads : bool
            Divide the perceptual gradient by ``alpha_t``.
        lr, momentum
            Unused.

        Returns
        -------
        torch.Tensor or int
            Gradient w.r.t. the latents, or ``0`` if every term is disabled.
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
        Difference mask between two images, normalized and dilated (ACE-style).

        Parameters
        ----------
        x1 : torch.Tensor
            Denoised image at the current step, in ``[-1, 1]``.
        x2 : torch.Tensor
            Original input image, in ``[-1, 1]``.
        dilation : int
            Odd kernel size of the max-pool dilation.

        Returns
        -------
        tuple of torch.Tensor
            ``(mask, dil_mask)``, both ``[B, 1, H, W]`` with per-sample max 1.
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
        Latent-space FastDiME counterfactual generation.

        The image is VAE-encoded (``latents``) and noised to ``t``; each of
        ``steps_number`` DDIM steps predicts ``x_prev``/``pred_x0``, adds the
        classifier gradient (``clean_multiclass_cond_fn``) and the distance
        gradient (``dist_cond_fn``), clips the total to
        ``explainer_config.gradient_clipping`` and takes a step of size
        ``class_grad_kwargs["lr"]``. A difference mask between the input and
        the current decode is thresholded at ``explainer_config.inpaint``
        (after ``warmup_steps``, if ``config.self_optimized`` or a fixed
        ``boolmask_in`` is given) and used to keep unmasked regions equal to
        the original latents.

        Parameters
        ----------
        img : torch.Tensor
            Input images ``[B, 3, H, W]`` in ``[0, 1]`` at the generator size.
        inpaint, dilation : float
            Unused here; the explainer config values are used instead.
        t : float
            Fraction of the diffusion process to start from.
        guided_iterations : int
            Unused.
        class_grad_kwargs : dict
            Keyword arguments of ``clean_multiclass_cond_fn`` (needs ``lr``).
        dist_grad_kargs : dict
            Keyword arguments of ``dist_cond_fn``.
        explainer_config : ACEConfig
            Provides ``gradient_clipping``, ``dilation`` and ``inpaint``.
        scale_grads : bool, optional
            Divide the classifier gradient by ``t + 5e-3``.
        boolmask_in : torch.Tensor, optional
            Fixed mask ``[B, C, H, W]`` (1 = keep original) merged into the
            computed mask.

        Returns
        -------
        tuple
            ``(final_image, x_t_steps, z_t_steps)``: decoded result
            ``[B, 3, H, W]`` in ``[0, 1]`` and the per-step CPU copies of the
            clean and noisy latents.
        """
        boolmask = boolmask_in
        to_pil = ToPILImage()
        class_grad_fn = self.clean_multiclass_cond_fn
        dist_fn = self.dist_cond_fn
        guidance_scale = self.guidence_scale
        num_inference_steps = self.steps_number
        do_classifier_free_guidance = self.do_classifier_free_guidance
        self_optimized_masking = self.self_optamized_masking

        # Encode image into latent space with the VAE
        height, width = self.config.data.input_size[1:]
        _log.info("%s", f"Encoding image of size: {height}x{width}")
        with torch.no_grad():
            latents = self.vae_latent_encoder(img.to(self.device))
        # Initialize x_t as the current latent (will be updated iteratively)
        x_t = latents.clone()
        batch_size = x_t.shape[0]
        vae_scale_factor = height // x_t.shape[-1]
        z_t = self.encode(
            img.to(self.device), t, stochastic="fully", num_steps=num_inference_steps
        )
        uc, cc = self._prepare_conditioning(
            batch_size=x_t.shape[0], summary=self.prompt
        )
        c_in = uc
        respaced_timesteps, alphas, alphas_prev, sqrt_one_minus_alphas, sigmas = (
            self._calculate_diffusion_parameters(t, num_inference_steps, self.eta)
        )
        x_t_steps = []
        z_t_steps = []
        # to_pil(img.squeeze(0).cpu()).save("pathldm/input_image.png")
        # to_pil(self.vae_latent_decoder(z_t.detach()).squeeze(0).cpu()).save(
        #     "pathldm/encoded_image.png"
        # )
        # to_pil(self.vae_latent_decoder(x_t.detach()).squeeze(0).cpu()).save(
        #     "pathldm/decoded_image.png"
        # )

        # Main iterative denoising loop in latent space
        for idx, timestep in enumerate(respaced_timesteps.flip(0)):
            x_t_steps.append(x_t.detach().cpu().clone())
            z_t_steps.append(z_t.detach().cpu().clone())
            if do_classifier_free_guidance:
                x_t = torch.cat([x_t] * 2)
                z_t = torch.cat([z_t] * 2)
                c_in = torch.cat([uc, cc])

            t_in = torch.full((x_t.size(0),), timestep, device=x_t.device).long()

            # print(f"Decoding step {idx+1}/{len(respaced_timesteps)}, timestep: {timestep}")
            # print(f"t_in shape: {t_in.shape}, x_in_for_model shape: {x_in_for_model.shape}, c_in_for_model shape: {c_in_for_model.shape}")

            # (C) Apply model and apply guidance to get the noise prediction (e_t)
            with torch.no_grad():
                e_t = self._apply_model_and_guidance(
                    z_t, t_in, c_in, self.guidence_scale
                )

                # (D) Perform denoising step calculation to get x_prev
                x_prev, pred_x0, dir_xt = self._calculate_denoised_latent(
                    z_t,
                    e_t,
                    alphas[idx],
                    alphas_prev[idx],
                    sigmas[idx],
                    sqrt_one_minus_alphas[idx],
                    batch_size,
                    do_classifier_free_guidance,
                    self.device,
                )
            # print(
            #     f"x_prev shape after step {idx+1} update: {x_prev.shape}, pred_x0 shape: {pred_x0.shape}, dir_xt shape: {dir_xt.shape}"
            # )

            # (E) Update x_current_latent for the next iteration
            # Compute guidance gradients:
            grads = 0
            if class_grad_fn is not None:
                xt_in = torch.clone(x_t.detach())
                class_grad = self.clean_multiclass_cond_fn(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                ) / (t + 5.0e-3 if scale_grads else 1)
                grads += class_grad
                # visualize the gradient
                _log.info("%s %s", "grads norm", grads.norm(p=float("inf")))

                if dist_fn is not None:
                    x_t_in = torch.clone(x_t.detach())
                    z_t_in = torch.clone(z_t.detach())
                    dist_grad = self.dist_cond_fn(
                        x_tau=latents,
                        z_t=z_t_in,
                        x_t=x_t_in,
                        alpha_t=alphas[idx],
                        scale_grads=False,
                        **dist_grad_kargs,
                    )
                    # ref = torch.zeros_like(grads).to("cpu")
                    # grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))
                    # self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                    #     f"{_DEBUG_DIR}/grads_class_dist{i}.png"
                    # )
                    grads = grads + dist_grad
                    # visualize the gradient
                    # print("grads1 norm", grads.norm(p=float("inf")))

                    # grads_pixels = self.latent_decoder(grads).squeeze(0)
                    # ref = torch.zeros_like(grads).to("cpu")
                    # grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))

            # scaling the gradeint
            z_t = x_prev
            x_t = pred_x0
            norm = grads.norm(p=float("inf"))
            # print("norm", norm)
            if norm > explainer_config.gradient_clipping:
                _log.info("%s", "gradient clipping")
                rescale_factor = explainer_config.gradient_clipping / norm
                grads = grads * rescale_factor
            z_t = z_t - class_grad_kwargs["lr"] * grads
            # print(
            #     f"z_t shape after step {idx+1} update: {z_t.shape}, x_t shape: {x_t.shape}"
            # )
            # Apply FASTDiME self-optimized masking if enabled:
            with torch.no_grad():
                x_0_denoised = self.vae_latent_decoder(x_t.detach())
                # mask_t, dil_mask = generate_smooth_mask(
                #     img.to(self.device), x_0_denoised, explainer_config.dilation
                # )
                if self.use_gussian_blur_masking:

                    mask_t, dil_mask = generate_smooth_mask(
                        img.to(self.device), x_0_denoised, explainer_config.dilation
                    )
                else:
                    mask_t, dil_mask = self.generate_mask(
                        img.to(self.device), x_0_denoised, explainer_config.dilation
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

            if (
                self_optimized_masking
                and idx > self.warmup_steps
                and boolmask_in is None
            ):
                _log.info("%s", "self optimized mask")
                # extract time-depedent mask (Eq. 6)

                # masking denoised and sampled images (Eq. 7 & 8)
                with torch.no_grad():
                    boolmask_latent = torch.nn.functional.interpolate(
                        boolmask,
                        size=(height // vae_scale_factor, width // vae_scale_factor),
                    )
                    x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents
                    noise = torch.randn_like(z_t).to(self.device)
                    noise = self.model.q_sample(z_t, t_in, noise=noise)
                    z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise
            # apply masking with fixed mask when available with our 2-step approaches
            if boolmask_in is not None and idx > self.warmup_steps:
                # fixed mask

                height, width = img.shape[2:]

                boolmask_latent = torch.nn.functional.interpolate(
                    boolmask,
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                )
                # masking denoised and sampled images (Eq. 7 & 8)
                with torch.no_grad():
                    x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents
                    noise = torch.randn_like(z_t).to(self.device)
                    noise = self.model.q_sample(z_t, t_in, noise=noise)

                    z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise
            with torch.no_grad():
                if boolmask_latent is not None:
                    z_t = z_t * (1 - boolmask_latent) + boolmask_latent * latents

            final_image = self.vae_latent_decoder(z_t)

        return final_image, x_t_steps, z_t_steps

    def repaint(
        self,
        x,
        pe,
        inpaint,
        dilation,
        t,
        stochastic,
        old_mask=None,
        mask_momentum=0.5,
        boolmask_in=None,
        max_avg_combination=0.5,
        exceptions=None,
    ):
        """
        Blend a counterfactual back into the original outside a difference mask.

        Legacy pixel-space RePaint loop carried over from the guided-diffusion
        DDPM generator: a smooth mask between ``x`` and ``pe`` is
        thresholded at ``inpaint``, the masked region is re-noised from ``x``
        at every step and the model's ``p_mean_variance`` is applied.

        Parameters
        ----------
        x : torch.Tensor
            Original images.
        pe : torch.Tensor
            Counterfactual (edited) images.
        inpaint : float
            Mask threshold; 0 disables re-noising from ``x``.
        dilation : int
            Mask dilation.
        t : float
            Fraction of ``config.timestep_respacing`` steps to run.
        stochastic : bool
            Add posterior noise between steps.
        old_mask, mask_momentum
            Unused.
        boolmask_in : torch.Tensor, optional
            Additional fixed mask merged into the computed one.
        max_avg_combination : float, optional
            Forwarded to ``generate_smooth_mask``.
        exceptions : torch.Tensor, optional
            Per-sample flags; samples with value 1 get an all-zero mask.

        Returns
        -------
        tuple
            ``(ce, boolmask)``: repainted images and the CPU mask.

        Notes
        -----
        Calls ``self.model.p_mean_variance(self.model, ...)`` with the
        guided-diffusion signature, which the PathLDM model does not
        provide; this method is not exercised by the current edit path.
        """
        batch_size = x.size(0)
        uc, cc = self._prepare_conditioning(batch_size, self.prompt)
        c_in = torch.cat([uc, cc], dim=0)
        respaced_steps = int(t * int(self.config.timestep_respacing))
        indices = list(range(respaced_steps))[::-1]
        x_normalized = self.dataset.project_to_pytorch_default(x)
        pe_normalized = self.dataset.project_to_pytorch_default(pe)
        mask, dil_mask = generate_smooth_mask(
            x_normalized, pe_normalized, dilation, max_avg_combination
        )

        boolmask = (dil_mask < inpaint).float()
        if boolmask_in is not None:
            added_term = torch.ones_like(boolmask) - boolmask_in[:, 0:1].to(boolmask)
            new_candidate = boolmask + added_term
            boolmask = torch.minimum(torch.ones_like(boolmask), new_candidate)

        if not exceptions is None:
            for i in range(exceptions.shape[0]):
                # No repainting!
                if exceptions[i] == 1:
                    boolmask[i] = 0

                else:
                    _log.info(
                        "%s",
                        "boolmask1: "
                        + str(
                            torch.sum(1 == boolmask[i])
                            / torch.sum(1 == torch.ones_like(boolmask[i]))
                        ),
                    )

        noise_fn = torch.randn_like if stochastic else torch.zeros_like

        ce = torch.clone(pe)
        for idx, t in enumerate(indices):
            # filter the with the diffusion model
            t = torch.tensor([t] * ce.size(0), device=ce.device)

            if idx == 0:
                ce = self.model.q_sample(ce, t, noise=noise_fn(ce))

            if inpaint != 0:
                ce = ce * (1 - boolmask) + boolmask * self.model.q_sample(
                    x, t, noise=noise_fn(ce)
                )

            model_mean, posterior_variance, posterior_log_variance = (
                self.model.p_mean_variance(
                    self.model, ce, t, c=c_in, clip_denoised=True
                )
            )

            ce = model_mean

            if stochastic and (idx != (respaced_steps - 1)):
                noise = torch.randn_like(ce)
                ce += torch.exp(0.5 * posterior_log_variance) * noise

        ce = ce * (1 - boolmask) + boolmask * x
        return ce, boolmask.cpu()

    def train_model(
        self,
    ):
        """
        Train with the guided-diffusion ``TrainLoop`` (legacy, see Notes).

        Moves an existing untrained ``model_dir`` aside, builds a training
        dataloader with ``config.max_steps`` steps per epoch, logs sample
        images to TensorBoard under ``<model_dir>/logs`` and runs the loop
        with the optimizer/EMA/checkpoint settings from the config.

        Notes
        -----
        Uses ``self.diffusion``, which the current constructor never sets,
        so this method cannot run against the PathLDM model as is.
        """
        if not self.config.is_trained and os.path.exists(self.model_dir):
            shutil.move(
                self.model_dir,
                self.model_dir + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            )

        self.config.is_trained = True
        # dist_util.setup_dist(self.config.gpus)
        logger.configure(dir=self.model_dir)

        schedule_sampler = create_named_schedule_sampler(
            self.config.schedule_sampler, self.diffusion
        )

        if not self.config.x_selection is None:
            self.dataset.task_config = SimpleNamespace(
                **{"x_selection": self.config.x_selection}
            )
            _log.info("%s", "self.dataset.task_config1")
            _log.info("%s", "self.dataset.task_config1")
            _log.info("%s", "self.dataset.task_config1")
            _log.info("%s", "self.dataset.task_config1")
            _log.info("%s", "self.dataset.task_config1")
            _log.info("%s", self.dataset.task_config)

        logger.log("creating data loader...")
        dataloader = get_dataloader(
            self.dataset,
            mode="train",
            batch_size=self.config.batch_size,
            training_config=types.SimpleNamespace(
                **{"steps_per_epoch": self.config.max_steps}
            ),
        )

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(self.model_dir, "logs"))
        if not self.config.x_selection is None:
            self.dataset.task_config = SimpleNamespace(
                **{"x_selection": self.config.x_selection}
            )
            _log.info("%s", "self.dataset.task_config2")
            _log.info("%s", "self.dataset.task_config2")
            _log.info("%s", "self.dataset.task_config2")
            _log.info("%s", "self.dataset.task_config2")
            _log.info("%s", "self.dataset.task_config2")
            _log.info("%s", self.dataset.task_config)

        log_images_to_writer(dataloader, writer, "train")
        data = iter(dataloader)

        logger.log("training...")
        train_loop = TrainLoop(
            model=self.model,
            diffusion=self.diffusion,
            data=data,
            batch_size=self.config.batch_size,
            microbatch=self.config.microbatch,
            lr=self.config.lr,
            ema_rate=self.config.ema_rate,
            log_interval=self.config.log_interval,
            save_interval=self.config.save_interval,
            resume_checkpoint=self.config.resume_checkpoint,
            use_fp16=self.config.use_fp16,
            fp16_scale_growth=self.config.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=self.config.weight_decay,
            lr_anneal_steps=self.config.lr_anneal_steps,
            model_dir=self.model_dir,
        )
        train_loop.run_loop(self.config, writer)

    def edit(
        self,
        x_in: torch.Tensor,
        target_confidence_goal: float,
        source_classes: torch.Tensor,
        target_classes: torch.Tensor,
        predictor: nn.Module,
        explainer_config: ACEConfig,
        predictor_datasets: list,
        boolmask_in: torch.Tensor,
        attempt_number: int,
        pbar=None,
        base_path: str = "",
        mode: str = "",
    ):
        """
        Create counterfactuals for a batch (the explainer entry point).

        Optionally loads or distills a gradient predictor
        (``explainer_config.distilled_predictor``; ``<PEAL_BASE>`` in its
        ``model_path`` is resolved against ``base_path``), maps the batch
        from the predictor's value range and size to the generator's, runs
        ``FastDiME`` and maps the result back. Difference masks and 2x2
        collages (original, counterfactual, mask, masked blend) are written
        to ``<base_path>/masks/``.

        Parameters
        ----------
        x_in : torch.Tensor
            Factual images ``[B, C, H, W]`` in the predictor's value range.
        target_confidence_goal : float
            Unused.
        source_classes, target_classes : torch.Tensor
            Source and target class per sample.
        predictor : torch.nn.Module
            Classifier being explained (used for the final confidences).
        explainer_config : ACEConfig
            Provides ``inpaint``, ``dilation``, ``sampling_time_fraction``,
            ``temperature``, ``learning_rate``, ``momentum``, ``optimizer``,
            ``gradient_clipping`` and ``distilled_predictor``.
        predictor_datasets : list
            Dataloaders of the predictor; the first dataset gives the value
            range and ``input_size``.
        boolmask_in : torch.Tensor or None
            Fixed edit mask forwarded to ``FastDiME``.
        attempt_number : int
            Used in the mask/collage filenames.
        pbar, mode
            Unused.
        base_path : str, optional
            Output directory for masks and the distilled predictor.

        Returns
        -------
        tuple
            ``(x_counterfactuals, x_differences, y_target_end_confidence,
            x_in, None, boolmask)`` with the first four as lists over the
            batch (counterfactuals and factuals in the predictor's range and
            size) and ``boolmask`` the ``[B, 1, H, W]`` difference mask.
        """
        if not explainer_config.distilled_predictor is None:
            model_path = "peal_run/predictor1"
            if explainer_config.distilled_predictor.get("model_path") is not None:

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
            _log.info("%s %s", "distilled_path", distilled_path)
            if not os.path.exists(distilled_path):
                gradient_predictor = distill_predictor(
                    explainer_config.distilled_predictor,
                    base_path,
                    predictor,
                    predictor_datasets,
                )

            else:
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

        classifier_to_generator = lambda x: self.dataset.project_from_pytorch_default(
            predictor_datasets[0].dataset.project_to_pytorch_default(x)
        )
        generator_to_classifier = lambda x: predictor_datasets[
            0
        ].dataset.project_from_pytorch_default(
            self.dataset.project_to_pytorch_default(x)
        )
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
            "l_perc": None,
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

        x_counterfactuals = transforms.Resize(predictor_data_size)(x_counterfactuals)
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
