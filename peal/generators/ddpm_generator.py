"""Pixel-space DDPM generator and the ACE/FastDiME counterfactual edit path.

This module wraps the guided-diffusion UNet vendored under
``peal.dependencies.ace`` as a PEAL ``EditCapableGenerator`` and
``InvertibleGenerator``: it builds model and diffusion from a
:class:`DDPMConfig`, trains it with the guided-diffusion ``TrainLoop``, and
offers (DDIM or noising) encode, decode, repaint and sampling in pixel space.
``DDPM.edit`` produces counterfactuals by handing the model, the diffusion and
the (optionally distilled) classifier to the vendored ACE or FastDiME main
functions, sweeping their attack hyper-parameters over several attempts.
"""

import os
import types
import shutil
import copy
import torch
import io
import blobfile as bf

from datetime import datetime
from pathlib import Path

import wget
from torch import nn
from types import SimpleNamespace
from typing import Union

from peal._optional import require
from peal.dependencies.FastDiME_CelebA.core.sample_utils import PerceptualLoss
from peal.generators.interfaces import EditCapableGenerator, InvertibleGenerator
from peal.global_utils import load_yaml_config, generate_smooth_mask

# from peal.dependencies.DiME.main import main as dime_main
from peal.dependencies.ace.guided_diffusion import logger
from peal.dependencies.ace.guided_diffusion.resample import (
    create_named_schedule_sampler,
)
from peal.dependencies.ace.guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from peal.data.dataloaders import get_dataloader
from peal.data.dataset_factory import get_datasets
from peal.explainers.counterfactual_explainer import ACEConfig
from peal.training.loggers import log_images_to_writer
from peal.generators.interfaces import GeneratorConfig
from peal.data.interfaces import DataConfig
from peal.training.trainers import distill_predictor
from peal.log import get_logger

_log = get_logger(__name__)


class DDPMConfig(GeneratorConfig):
    """
    TODO actually implement this class properly
    This class defines the config of a DDPM.
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
    num_channels: int = 128
    """
    The number of channels
    """
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
    download_weights: Union[str, type(None)] = None


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


class DDPM(EditCapableGenerator, InvertibleGenerator):
    """Pixel-space denoising diffusion model used as a PEAL generator.

    Wraps the guided-diffusion UNet and gaussian diffusion created by
    ``create_model_and_diffusion`` from the vendored ACE code. Besides plain
    sampling it provides the invertible interface (``encode``/``decode``), a
    ``repaint`` step that keeps everything outside a smooth change mask, and
    ``edit``, which runs the ACE or FastDiME counterfactual attack.

    Parameters
    ----------
    config : str or DDPMConfig
        Config path or object; loaded with ``load_yaml_config``. The image
        size is overwritten with the last entry of ``config.data.input_size``.
    model_dir : str, optional
        Run directory holding the weights, logs and outputs. Defaults to
        ``config.base_path``.
    device : str, optional
        Device the UNet is moved to.
    predictor_dataset : optional
        Accepted for interface compatibility; unused.

    Attributes
    ----------
    model, diffusion
        The UNet and the gaussian diffusion process.
    dataset
        First dataset built from ``config.data``; used for the normalisation
        between the model range and the pytorch default range.
    model_path : str
        ``final.cpl`` in the run directory if it exists, else ``final.pt``
        (downloaded from ``config.download_weights`` when missing). If no
        weights are found, an existing run directory is moved aside with a
        timestamp suffix and a fresh one is created.
    noise_fn : callable
        ``torch.randn_like`` for stochastic sampling, ``torch.zeros_like``
        otherwise.
    """

    def __init__(self, config, model_dir=None, device="cpu", predictor_dataset=None):
        """Load the config, build the UNet and diffusion and restore weights."""
        super().__init__()
        self.predictor_distilled = None
        self.config = load_yaml_config(config)
        self.config.image_size = self.config.data.input_size[-1]

        self.dataset = get_datasets(self.config.data)[0]

        if not model_dir is None:
            self.model_dir = model_dir

        else:
            self.model_dir = self.config.base_path

        self.data_dir = os.path.join(self.model_dir, "data_test")
        self.counterfactual_path = os.path.join(self.model_dir, "counterfactuals_test")

        self.model, self.diffusion = create_model_and_diffusion(**self.config.__dict__)
        self.device = device
        self.model.to(device)
        self.model_path = os.path.join(self.model_dir, "final.cpl")
        if os.path.exists(self.model_path) and self.config.is_trained:
            _log.info("%s", "load ddpm model weights!!!")
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
            # save_yaml_config(self.config, os.path.join(self.model_dir, "config.yaml"))

        else:
            self.model_path = os.path.join(self.model_dir, "final.pt")
            if not os.path.exists(self.model_path) and self.config.download_weights:
                Path(self.model_dir).mkdir(exist_ok=True)
                wget.download(self.config.download_weights, self.model_path)

            if os.path.exists(self.model_path) and self.config.is_trained:
                state_dict = load_state_dict(self.model_path, map_location=device)
                _log.info("%s", "load ddpm model weights!!!")
                self.model.load_state_dict(state_dict)

            else:
                _log.info("%s", "No ddpm model weights yet!!!")
                if os.path.exists(self.model_dir):
                    shutil.move(
                        self.model_dir,
                        self.model_dir
                        + "_old"
                        + datetime.now().strftime("%Y%m%d_%H%M%S"),
                    )

                Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.config.is_trained = True
        self.noise_fn = torch.randn_like if self.config.stochastic else torch.zeros_like
        self.vggloss = None

    def sample_x(self, batch_size=None, renormalize=True):
        """Draw images from the prior with the full ancestral sampling loop.

        Parameters
        ----------
        batch_size : int, optional
            Number of images; defaults to ``config.batch_size``.
        renormalize : bool, optional
            Map the samples from the model range into the pytorch default
            range via ``dataset.project_to_pytorch_default``.

        Returns
        -------
        torch.Tensor
            Samples of shape ``(batch_size, *config.data.input_size)``.
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        sample = self.diffusion.p_sample_loop(
            self.model, [batch_size] + self.config.data.input_size
        )
        if renormalize:
            sample = self.dataset.project_to_pytorch_default(sample)

        return sample

    def encode(self, x, t=1.0, stochastic=None, num_steps=None):
        """Map images to the noisy latent at a fraction ``t`` of the schedule.

        With ``stochastic == "fully"`` the latent is a single ``q_sample``
        draw at the respaced timestep, clamped to ``[-1, 1]``. Otherwise the
        image is inverted deterministically by iterating
        ``ddim_reverse_sample`` over the first ``t * timestep_respacing``
        steps.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images in the model range, shape ``(B, C, H, W)``.
        t : float, optional
            Fraction of ``config.timestep_respacing`` to noise up to.
        stochastic : bool or str, optional
            ``"fully"`` for one-shot noising, anything else for DDIM
            inversion. Defaults to ``config.stochastic``.
        num_steps : optional
            Accepted for interface compatibility; unused.

        Returns
        -------
        torch.Tensor
            Latent with the same shape as ``x``.
        """
        if stochastic is None:
            stochastic = self.config.stochastic

        respaced_steps = int(t * int(self.config.timestep_respacing))
        if stochastic == "fully":
            noise = torch.randn_like(x)
            timestep = torch.tensor(respaced_steps).to(x).long()
            x = torch.clamp(self.diffusion.q_sample(x, timestep, noise=noise), -1, 1)

        else:
            timesteps = list(range(respaced_steps))
            for idx, t in enumerate(timesteps):
                t = torch.tensor([t] * x.size(0), device=x.device)
                x = self.diffusion.ddim_reverse_sample(self.model, x, t)["sample"]

        # TODO why are gradients in ACE scaled???
        # t = torch.tensor([self.steps - 1] * x.size(0), device=x.device)
        return x

    def decode(self, z, t=1.0, stochastic=None, num_steps=None):
        """Denoise a latent back to an image with the reverse diffusion loop.

        Iterates ``p_mean_variance`` from step ``t * timestep_respacing - 1``
        down to 0, taking the posterior mean and adding the posterior noise in
        every step but the last when ``config.stochastic`` is set.

        Parameters
        ----------
        z : torch.Tensor or list
            Latent of shape ``(B, C, H, W)``; a one-element list is unwrapped.
        t : float, optional
            Fraction of the respaced schedule the latent was noised to; must
            match the value used in :meth:`encode`.
        stochastic : bool, optional
            Defaults to ``config.stochastic``. Note that the per-step noise is
            gated by ``config.stochastic`` rather than by this argument.
        num_steps : optional
            Accepted for interface compatibility; unused.

        Returns
        -------
        torch.Tensor
            Reconstructed images in the model range.
        """
        if isinstance(z, list) and len(z) == 1:
            z = z[0]

        if stochastic is None:
            stochastic = self.config.stochastic

        # TODO test decode function via sampling function
        respaced_steps = int(t * int(self.config.timestep_respacing))
        timesteps = list(range(respaced_steps))[::-1]
        for idx, t in enumerate(timesteps):
            t = torch.tensor([t] * z.size(0), device=z.device)

            out = self.diffusion.p_mean_variance(self.model, z, t, clip_denoised=True)

            z = out["mean"]

            if idx != (respaced_steps - 1):
                if self.config.stochastic:
                    z += torch.exp(0.5 * out["log_variance"]) * self.noise_fn(z)

        return z

    def repaint(
        self,
        x,
        pe,
        inpaint,
        dilation,
        t,
        stochastic,
        boolmask_in=None,
        max_avg_combination=0.5,
        exceptions=None,
    ):
        """Re-diffuse an edited image while pinning the unchanged region.

        A smooth change mask between ``x`` and the edited image ``pe`` is
        built with ``generate_smooth_mask``; everything whose dilated mask
        value stays below ``inpaint`` is considered background and is, in
        every reverse step, replaced by the correspondingly noised original.
        The final image also takes the background pixels straight from ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Original images, shape ``(B, C, H, W)``.
        pe : torch.Tensor
            Edited ("pre-explanation") images of the same shape.
        inpaint : float
            Mask threshold; ``0`` disables the per-step re-injection of the
            original but not the final composition.
        dilation : float
            Dilation of the change mask, passed to ``generate_smooth_mask``.
        t : float
            Fraction of ``config.timestep_respacing`` to restart from.
        stochastic : bool
            Add posterior noise during the reverse loop and noise when
            re-injecting the original.
        boolmask_in : torch.Tensor, optional
            Background mask of a previous attempt; its complement is unioned
            into the current background mask, so already-edited regions stay
            editable.
        max_avg_combination : float, optional
            Mixing weight between the max and the mean channel difference in
            ``generate_smooth_mask``.
        exceptions : torch.Tensor, optional
            Per-sample flags; where the entry is 1 the background mask is
            zeroed, i.e. that sample is not repainted at all.

        Returns
        -------
        ce : torch.Tensor
            Repainted images.
        boolmask : torch.Tensor
            The background mask used, on the CPU.
        """
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

        noise_fn = torch.randn_like if stochastic else torch.zeros_like

        ce = torch.clone(pe)
        for idx, t in enumerate(indices):
            # filter the with the diffusion model
            t = torch.tensor([t] * ce.size(0), device=ce.device)

            if idx == 0:
                ce = self.diffusion.q_sample(ce, t, noise=noise_fn(ce))

            if inpaint != 0:
                ce = ce * (1 - boolmask) + boolmask * self.diffusion.q_sample(
                    x, t, noise=noise_fn(ce)
                )

            out = self.diffusion.p_mean_variance(self.model, ce, t, clip_denoised=True)

            ce = out["mean"]

            if stochastic and (idx != (respaced_steps - 1)):
                noise = torch.randn_like(ce)
                ce += torch.exp(0.5 * out["log_variance"]) * noise

        ce = ce * (1 - boolmask) + boolmask * x
        return ce, boolmask.cpu()

    def train_model(
        self,
    ):
        """Train the UNet with the guided-diffusion ``TrainLoop``.

        Configures the guided-diffusion logger and the schedule sampler named
        by ``config.schedule_sampler``, builds a training dataloader whose
        epoch length is ``config.max_steps``, logs a batch of training images
        to ``<model_dir>/logs`` and runs the loop, which writes checkpoints
        into ``model_dir`` every ``config.save_interval`` steps. If the model
        was not trained before, an existing run directory is moved aside with
        a timestamp suffix.

        Notes
        -----
        ``config.x_selection`` overrides the dataset's ``task_config`` so that
        the generator can be trained on a different input column than the one
        the dataset was built for.
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
        # guided-diffusion's TrainLoop imports dist_util, and with it
        # mpi4py, so it is imported at the point of use rather than
        # making a system MPI a hard requirement of every PEAL install.
        from peal.dependencies.ace.guided_diffusion.train_util import TrainLoop

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
        pbar=None,
        base_path: str = "",
        mode: str = "",
        boolmask_in=None,
        attempt_number=None,
    ):
        """Create counterfactuals for ``x_in`` with ACE or FastDiME.

        Trains the diffusion model first if needed, optionally distils the
        predictor into a surrogate that supplies the guidance gradients
        (cached at ``<base_path>/explainer/distilled_predictor/model.cpl``),
        and then calls the vendored ``ace_main`` or ``fastdime_main``
        depending on ``explainer_config.subtype``.

        The attack is repeated ``explainer_config.num_attempts`` times. Any of
        ``attack_iterations``, ``sampling_time_fraction`` and
        ``sampling_inpaint`` given as a two-element list is interpolated
        linearly between the attempts, ``dist_l1``/``dist_l2`` geometrically,
        and the seed is incremented per attempt. All attempts are concatenated
        into the returned batch rather than being filtered by success.

        Parameters
        ----------
        x_in : torch.Tensor
            Images to explain, shape ``(B, C, H, W)``.
        target_confidence_goal : float
            Accepted for interface compatibility; the stopping criterion is
            taken from ``explainer_config``.
        source_classes, target_classes : torch.Tensor
            Current and desired class index per sample, shape ``(B,)``.
        predictor : nn.Module
            Classifier whose decision the counterfactual has to flip; also
            used to score the results.
        explainer_config : ACEConfig
            Attack hyper-parameters; ``subtype`` selects ``"ACE"`` or
            ``"FastDiME"``, ``l_perc``/``l_perc_layer`` enable the VGG
            perceptual loss, ``distilled_predictor`` the surrogate.
        predictor_datasets : list
            ``[train, val, test]``-style datasets; entry 1 is passed to the
            attack as the predictor's dataset and entry set is used for
            distillation.
        pbar : optional
            Progress bar, unused here.
        base_path : str, optional
            Run directory used to cache the distilled predictor.
        mode : str, optional
            Unused; kept for interface compatibility.
        boolmask_in : optional
            Unused; kept for interface compatibility with other generators.
        attempt_number : optional
            Unused; attempts are looped over internally.

        Returns
        -------
        tuple of list
            ``(counterfactuals, differences, target confidences, originals,
            histories, zeros)``, each with ``B * num_attempts`` entries. The
            differences are ``x_in - counterfactual`` and the trailing zeros
            stand in for the per-sample masks other generators return.

        Raises
        ------
        Exception
            If ``explainer_config.subtype`` is neither ACE nor FastDiME.
        """
        if not self.config.is_trained:
            _log.info("%s", "Model not trained yet. Model will be trained now!")
            self.train_model()
        if not explainer_config.distilled_predictor is None:
            distilled_path = os.path.join(
                base_path, "explainer", "distilled_predictor", "model.cpl"
            )
            if not os.path.exists(distilled_path):
                gradient_predictor = distill_predictor(
                    explainer_config.distilled_predictor,
                    base_path,
                    predictor,
                    predictor_datasets,
                )

            else:
                try:
                    gradient_predictor = torch.load(
                        distilled_path, map_location=self.device
                    )
                except Exception:
                    gradient_predictor = torch.load(
                        distilled_path, map_location=self.device, weights_only=False
                    )

        else:
            gradient_predictor = predictor

        dataset = [
            (
                torch.zeros([len(x_in)], dtype=torch.long),
                x_in,
                [source_classes, target_classes],
            )
        ]
        args = copy.deepcopy(self.config).dict()
        # args = copy.deepcopy(self.config.full_args)
        args = SimpleNamespace(**args)
        args.output_path = self.counterfactual_path
        args.batch_size = x_in.shape[0]
        if explainer_config.l_perc != 0:
            if self.vggloss is None:
                _log.info("%s", "Loading VGG loss!")
                self.vggloss = PerceptualLoss(
                    layer=explainer_config.l_perc_layer, c=explainer_config.l_perc
                ).to(self.device)
                self.vggloss.eval()

            args.vggloss = self.vggloss
        #
        x_counterfactuals = None
        x_in_out = torch.clone(x_in)
        for idx in range(explainer_config.num_attempts):
            # a0_0 = a_min + (a_max - a_min) / 2
            # a1_0 = a_min + 0 * (a_max - a_min) / (explainer_config.num_attempts - 1)
            # a1_1 = a_min + 1 * (a_max - a_min) / (explainer_config.num_attempts - 1)
            # a2_0 = a_min + 0 * (a_max - a_min) / (explainer_config.num_attempts - 1)
            # a2_0 = a_min + 1 * (a_max - a_min) / (explainer_config.num_attempts - 1)
            # a2_0 = a_min + 2 * (a_max - a_min) / (explainer_config.num_attempts - 1)
            if explainer_config.num_attempts > 1:
                multiplier = idx / (explainer_config.num_attempts - 1)

            else:
                multiplier = 0.5

            args.attack_iterations = int(
                explainer_config.attack_iterations
                if not isinstance(explainer_config.attack_iterations, list)
                else int(
                    explainer_config.attack_iterations[0]
                    + (
                        explainer_config.attack_iterations[1]
                        - explainer_config.attack_iterations[0]
                    )
                    * multiplier
                )
            )
            _log.info("%s", "args.attack_iterations")
            _log.info("%s", args.attack_iterations)
            args.sampling_time_fraction = float(
                explainer_config.sampling_time_fraction
                if not isinstance(explainer_config.sampling_time_fraction, list)
                else float(
                    explainer_config.sampling_time_fraction[0]
                    + (
                        explainer_config.sampling_time_fraction[1]
                        - explainer_config.sampling_time_fraction[0]
                    )
                    * multiplier
                )
            )
            _log.info("%s", "args.sampling_time_fraction")
            _log.info("%s", args.sampling_time_fraction)
            args.dist_l1 = float(
                explainer_config.dist_l1
                if not isinstance(explainer_config.dist_l1, list)
                else explainer_config.dist_l1[0]
                * (
                    (explainer_config.dist_l1[1] / explainer_config.dist_l1[0])
                    ** (1 / (explainer_config.num_attempts - 1))
                )
                ** idx
            )
            _log.info("%s", "args.dist_l1")
            _log.info("%s", args.dist_l1)
            args.dist_l2 = float(
                explainer_config.dist_l2
                if not isinstance(explainer_config.dist_l2, list)
                else explainer_config.dist_l1[0]
                * (
                    (explainer_config.dist_l2[1] / explainer_config.dist_l2[0])
                    ** (1 / (explainer_config.num_attempts - 1))
                )
                ** idx
            )
            args.sampling_inpaint = float(
                explainer_config.sampling_inpaint
                if not isinstance(explainer_config.sampling_inpaint, list)
                else float(
                    explainer_config.sampling_inpaint[0]
                    - (
                        explainer_config.sampling_inpaint[0]
                        - explainer_config.sampling_inpaint[1]
                    )
                    * multiplier
                )
            )
            _log.info("%s", "args.sampling_inpaint")
            _log.info("%s", args.sampling_inpaint)
            args.__dict__.update(
                {
                    k: v
                    for k, v in explainer_config.__dict__.items()
                    if k not in args.__dict__
                }
            )
            args.timestep_respacing = explainer_config.timestep_respacing
            args.dataset = dataset
            args.predictor_dataset = predictor_datasets[1]
            args.generator_dataset = self.dataset
            args.model_path = os.path.join(self.model_dir, "final.pt")
            args.classifier = gradient_predictor
            args.original_classifier = predictor
            args.diffusion = self.diffusion
            args.model = self.model
            args.seed += idx
            #
            if args.subtype == "ACE":
                # Imported here for the same reason as FastDiME below:
                # the vendored ACE entry point reaches mpi4py through
                # guided-diffusion's dist_util.
                from peal.dependencies.ace.run_ace import main as ace_main

                x_counterfactuals_current, histories = ace_main(args=args)

            elif args.subtype == "FastDiME":

                # Imported here: the vendored FastDiME package reaches
                # mpi4py through its dist_util, and a module-level
                # import would make a system MPI a hard requirement of
                # every PEAL install instead of the `mpi` extra.
                from peal.dependencies.FastDiME_CelebA.main import (
                    main as fastdime_main,
                )

                x_counterfactuals_current, histories = fastdime_main(args=args)

            else:
                raise Exception(args.subtype + " does not exist!")

            x_counterfactuals_current = torch.cat(x_counterfactuals_current, dim=0)

            device = [p for p in predictor.parameters()][0].device
            preds = torch.nn.Softmax(dim=-1)(
                predictor(x_counterfactuals_current.to(device)).detach().cpu()
            )
            _log.info(
                "%s",
                "preds_final: "
                + str(
                    [
                        float(preds[i][target_classes[i]])
                        for i in range(len(target_classes))
                    ]
                ),
            )
            y_target_end_confidence_current = torch.zeros([x_in.shape[0]])
            for i in range(x_in.shape[0]):
                y_target_end_confidence_current[i] = preds[i, target_classes[i]]

            if x_counterfactuals is None:
                x_counterfactuals = x_counterfactuals_current
                y_target_end_confidence = y_target_end_confidence_current

            else:
                x_counterfactuals = torch.cat(
                    [x_counterfactuals, x_counterfactuals_current], 0
                )
                y_target_end_confidence = torch.cat(
                    [y_target_end_confidence, y_target_end_confidence_current], 0
                )
                x_in_out = torch.cat([x_in_out, torch.clone(x_in)], 0)

        return (
            list(x_counterfactuals),
            list(x_in_out - x_counterfactuals),
            list(y_target_end_confidence),
            list(x_in),
            list(histories),
            [0] * len(x_counterfactuals),
        )
