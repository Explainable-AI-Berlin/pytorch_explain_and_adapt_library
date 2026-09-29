import copy
import json
import math
import re
import numpy as np

import lightning as L
from peal.dependencies.edit_friendly_ddpm_inversion.edit_friendly_ddpm_inversion import (
    DiffusionSampler,
)
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.plugins.precision import amp
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import TensorDataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import lpips

from .lmdb_writer import *
from .metrics import evaluate_fid, evaluate_lpips
from .renderer import *


class LitModel(L.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            L.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf

        self.model = conf.make_model_conf().make_model()
        if not conf.encoder is None:
            self.model.encoder = conf.encoder

        self.first_encoder_parameter = torch.tensor(
            list(self.model.encoder.parameters())[0]
        )
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print("Model params: %.2f M" % (model_size / 1024 / 1024))

        self.sampler: SpacedDiffusionBeatGans = (
            conf.make_diffusion_conf().make_sampler()
        )
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()
        self.ddpm_sampler = self.create_ddpm_sampler()

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler()
            self.eval_latent_sampler = (
                conf.make_latent_eval_diffusion_conf().make_sampler()
            )
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # initial variables for consistent sampling
        self.register_buffer(
            "x_T", torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size)
        )

        if conf.pretrain is not None:
            print(f"loading pretrain ... {conf.pretrain.name}")
            state = torch.load(conf.pretrain.path, map_location="cpu")
            print("step:", state["global_step"])
            self.load_state_dict(state["state_dict"], strict=False)

        if conf.latent_infer_path is not None:
            print("loading latent stats ...")
            state = torch.load(conf.latent_infer_path)
            self.conds = state["conds"]
            self.register_buffer("conds_mean", state["conds_mean"][None, :])
            self.register_buffer("conds_std", state["conds_std"][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None

        self.best_lpips = float("inf")
        self.best_fid = float("inf")

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(self.device)
        return cond

    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(
                T_latent
            ).make_sampler()

        noise = torch.randn(N, 3, self.conf.img_size, self.conf.img_size, device=device)
        pred_img = render_uncondition(
            self.conf,
            self.ema_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.conds_mean,
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def render(self, noise, cond=None, T=None, grads=False):
        """
        Renders an image based on the provided noise and optional conditioning.

        DAE (conditioned DDIM) backwards pass.

        Parameters:
        noise (torch.Tensor): The input noise tensor for the rendering process.
        cond (torch.Tensor, optional): The conditioning tensor. If provided, the rendering will be conditioned on this tensor. Defaults to None.
        T (int, optional): The number of timesteps for the diffusion process. If None, the default evaluation sampler is used. Defaults to None.

        Returns:
        torch.Tensor: The rendered image tensor, with values scaled to the range [0, 1].
        """
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler(grads=grads)

        if cond is not None:
            pred_img = render_condition(
                self.conf, self.ema_model, noise, sampler=sampler, cond=cond
            )
        else:
            pred_img = render_uncondition(
                self.conf, self.ema_model, noise, sampler=sampler, latent_sampler=None
            )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x):
        # TODO:
        assert self.conf.model_type.has_autoenc()
        cond = self.ema_model.encoder.forward(x)
        return cond

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(
            self.ema_model, x, model_kwargs={"cond": cond}
        )
        return out["sample"]

    def encode_ddpm_inversion(self, x, z_sem):
        xT, zs = self.ddpm_sampler.ddpm_edit_friendly_invert(x, z_sem, self.ema_model)
        return xT, zs

    def decode_ddpm_inversion(self, xT, zs, z_sem):
        x = self.ddpm_sampler.ddpm_edit_friendly_sample(xT, zs, z_sem, self.ema_model)
        return x

    def create_ddpm_sampler(self, T=20, spacing="uniform"):
        device = next(self.model.parameters()).device
        betas = torch.from_numpy(self.sampler.betas).to(device)
        sampler = DiffusionSampler(betas)
        return DiffusionSampler.respaced(sampler, num_steps=T, spacing=spacing)

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        if ema_model:
            model = self.ema_model
        else:
            model = self.model
        gen = self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)
        return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("local seed:", seed)
        ##############################################

        self.train_data = self.conf.make_dataset()
        print("train data:", len(self.train_data))
        self.val_data = self.train_data
        print("val data:", len(self.val_data))

    def _train_dataloader(self, drop_last=True):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataloader = conf.make_loader(
            self.train_data, shuffle=True, drop_last=drop_last
        )
        return dataloader

    def train_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        if latent mode => return the inferred latent dataset
        """
        print("on train dataloader start ...")
        if self.conf.train_mode.require_dataset_infer():
            if self.conds is None:
                # usually we load self.conds from a file
                # so we do not need to do this again!
                self.conds = self.infer_whole_dataset()
                # need to use float32! unless the mean & std will be off!
                # (1, c)
                self.conds_mean.data = self.conds.float().mean(dim=0, keepdim=True)
                self.conds_std.data = self.conds.float().std(dim=0, keepdim=True)
            print("mean:", self.conds_mean.mean(), "std:", self.conds_std.mean())

            # return the dataset with pre-calculated conds
            conf = self.conf.clone()
            conf.batch_size = self.batch_size
            data = TensorDataset(self.conds)
            return conf.make_loader(data, shuffle=True)
        else:
            return self._train_dataloader()

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop?
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def infer_whole_dataset(
        self, with_render=False, T_render=None, render_save_path=None
    ):
        """
        predicting the latents given images using the encoder

        Args:
            both_flips: include both original and flipped images; no need, it's not an improvement
            with_render: whether to also render the images corresponding to that latent
            render_save_path: lmdb output for the rendered images
        """
        data = self.conf.make_dataset()
        if isinstance(data, CelebAlmdb) and data.crop_d2c:
            # special case where we need the d2c crop
            data.transform = make_transform(
                self.conf.img_size, flip_prob=0, crop_d2c=True
            )
        else:
            data.transform = make_transform(self.conf.img_size, flip_prob=0)

        # data = SubsetDataset(data, 21)

        loader = self.conf.make_loader(
            data,
            shuffle=False,
            drop_last=False,
            batch_size=self.conf.batch_size_eval,
            parallel=True,
        )
        model = self.ema_model
        model.eval()
        conds = []

        if with_render:
            sampler = self.conf._make_diffusion_conf(
                T=T_render or self.conf.T_eval
            ).make_sampler()

            if self.global_rank == 0:
                writer = LMDBImageWriter(render_save_path, format="webp", quality=100)
            else:
                writer = nullcontext()
        else:
            writer = nullcontext()

        with writer:
            for batch in tqdm(loader, total=len(loader), desc="infer"):
                with torch.no_grad():
                    # (n, c)
                    # print('idx:', batch['index'])
                    cond = model.encoder(batch["x"].to(self.device))

                    # used for reordering to match the original dataset
                    idx = batch["index"]
                    idx = self.all_gather(idx)
                    if idx.dim() == 2:
                        idx = idx.flatten(0, 1)
                    argsort = idx.argsort()

                    if with_render:
                        noise = torch.randn(
                            len(cond),
                            3,
                            self.conf.img_size,
                            self.conf.img_size,
                            device=self.device,
                        )
                        render = sampler.sample(model, noise=noise, cond=cond)
                        render = (render + 1) / 2
                        # print('render:', render.shape)
                        # (k, n, c, h, w)
                        render = self.all_gather(render)
                        if render.dim() == 5:
                            # (k*n, c)
                            render = render.flatten(0, 1)

                        # print('global_rank:', self.global_rank)

                        if self.global_rank == 0:
                            writer.put_images(render[argsort])

                    # (k, n, c)
                    cond = self.all_gather(cond)

                    if cond.dim() == 3:
                        # (k*n, c)
                        cond = cond.flatten(0, 1)

                    conds.append(cond[argsort].cpu())
                # break
        model.train()
        # (N, c) cpu

        conds = torch.cat(conds).float()
        return conds

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        # batch size here is local!
        # forward
        # self.first_encoder_parameter - torch.tensor(list(self.model.encoder.parameters())[0]).cpu()
        if self.conf.train_mode.require_dataset_infer():
            # this mode as pre-calculated cond
            cond = batch[0]
            if self.conf.latent_znormalize:
                cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
                    self.device
                )
        else:
            imgs, idxs = batch["x"], batch["index"]
            # print(f'(rank {self.global_rank}) batch size:', len(imgs))
            x_start = imgs

        if self.conf.train_mode == TrainMode.diffusion:
            """
            main training mode!!!
            """
            # with numpy seed we have the problem that the sample t's are related!
            t, weight = self.T_sampler.sample(len(x_start), x_start.device)
            losses = self.sampler.training_losses(
                model=self.model, x_start=x_start, t=t
            )
        elif self.conf.train_mode.is_latent_diffusion():
            """
            training the latent variables!
            """
            # diffusion on the latent
            t, weight = self.T_sampler.sample(len(cond), cond.device)
            latent_losses = self.latent_sampler.training_losses(
                model=self.model.latent_net, x_start=cond, t=t
            )
            # train only do the latent diffusion
            losses = {
                "latent": latent_losses["loss"],
                "loss": latent_losses["loss"],
            }
        else:
            raise NotImplementedError()

        loss = losses["loss"].mean()
        # divide by accum batches to make the accumulated gradient exact!
        for key in ["loss", "vae", "latent", "mmd", "chamfer", "arg_cnt"]:
            if key in losses:
                losses[key] = self.all_gather(losses[key]).mean()

        # Logged on EVERY rank: the values were all_gather-averaged above, so
        # they are identical everywhere, and ModelCheckpoint(monitor="loss")
        # needs the key on every rank. Logging on rank 0 only made rank 0 alone
        # decide to save and block in Trainer.save_checkpoint's barrier while
        # the other ranks never entered it (DDP deadlock at the first save).
        self.log("loss", losses["loss"], True)
        for key in ["vae", "latent", "mmd", "chamfer", "arg_cnt"]:
            if key in losses:
                self.log(f"loss/{key}", losses[key])

        return {"loss": loss}

    def on_train_start(self) -> None:
        """
        Evaluate and log step 0 baseline before training begins.
        """
        if self.global_step == 0:
            print("[Step 0 Baseline] Evaluating initial model before training...")
            x_start = None
            if not self.conf.train_mode.require_dataset_infer():
                try:
                    dataloader = self.trainer.train_dataloader
                    if dataloader is not None:
                        first_batch = next(iter(dataloader))
                        if isinstance(first_batch, dict) and "x" in first_batch:
                            x_start = first_batch["x"].to(self.device)
                        elif isinstance(first_batch, (tuple, list)):
                            x_start = first_batch[0].to(self.device)
                except Exception as e:
                    print(f"[Step 0 Baseline] Could not fetch first batch: {e}")

            self.log_sample(x_start=x_start)
            if getattr(self.conf, "eval_fid", True):
                self.evaluate_scores()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(
                    self.model.latent_net,
                    self.ema_model.latent_net,
                    self.conf.ema_decay,
                )
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

            # logging
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            else:
                imgs = batch["x"]

            if self.conf.sample_every_samples > 0 and is_time(
                self.num_samples,
                self.conf.sample_every_samples,
                self.conf.batch_size_effective,
            ):
                self.log_sample(x_start=imgs)
                if getattr(self.conf, "eval_fid", True):
                    self.evaluate_scores()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.conf.grad_clip)

    def log_sample(self, x_start):
        """
        put images to the tensorboard
        """

        def do(
            model,
            postfix,
            use_xstart,
            save_real=False,
            no_latent_diff=False,
            interpolate=False,
        ):
            model.eval()
            with torch.no_grad():
                all_x_T = self.split_tensor(self.x_T)
                batch_size = min(len(all_x_T), self.conf.batch_size_eval)
                # allow for superlarge models
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen = []
                curr_idx = 0
                for x_T in loader:
                    if use_xstart and x_start is not None:
                        _xstart = x_start[curr_idx : curr_idx + len(x_T)]
                        if len(_xstart) == 0:
                            break
                        if len(x_T) > len(_xstart):
                            x_T = x_T[: len(_xstart)]
                        curr_idx += len(_xstart)
                    else:
                        _xstart = None

                    if self.conf.train_mode.is_latent_diffusion() and not use_xstart:
                        # diffusion of the latent first
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.eval_sampler,
                            latent_sampler=self.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std,
                        )
                    else:
                        if not use_xstart and self.conf.model_type.has_noise_to_cond():
                            model: BeatGANsAutoencModel
                            # special case, it may not be stochastic, yet can sample
                            cond = torch.randn(
                                len(x_T), self.conf.style_ch, device=self.device
                            )
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate and _xstart is not None:
                                cond = model.encoder(_xstart)
                                i = torch.randperm(len(cond))
                                cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.eval_sampler.sample(
                            model=model, noise=x_T, cond=cond, x_start=_xstart
                        )
                    Gen.append(gen)

                gen_local = torch.cat(Gen)
                gen = self.all_gather(gen_local)
                if gen.dim() == 5:
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart and x_start is not None:
                    # save the original images to the tensorboard
                    x_start_local = x_start[: len(gen_local)]
                    real = self.all_gather(x_start_local)
                    if real.dim() == 5:
                        real = real.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_real = (make_grid(real) + 1) / 2
                        self.logger.experiment.add_image(
                            f"sample{postfix}/real", grid_real, self.num_samples
                        )

                if use_xstart and x_start is not None:
                    # Calculate pairwise reconstruction metrics (MSE, PSNR, L1, LPIPS) for Stage 1 autoencoding
                    x_start_local = x_start[: len(gen_local)]
                    mse_val = torch.mean((x_start_local - gen_local) ** 2).item()
                    psnr_val = 20 * math.log10(2.0 / (math.sqrt(mse_val) + 1e-8))
                    l1_val = torch.mean(torch.abs(x_start_local - gen_local)).item()
                    # LPIPS is computed on every rank and averaged across ranks,
                    # so that self.log(sync_dist=True) and the final.ckpt
                    # decision below are taken by all ranks together. Both are
                    # collectives (Trainer.save_checkpoint ends in a barrier);
                    # calling them from rank 0 only deadlocked DDP runs.
                    if not hasattr(self, "_dino_eval"):
                        from peal.global_utils import DINOEvaluator
                        self._dino_eval = DINOEvaluator(device=self.device)
                    lpips_local = torch.as_tensor(
                        float(self._dino_eval.compute_lpips(x_start_local, gen_local)),
                        device=self.device,
                    )
                    lpips_val = self.all_gather(lpips_local).float().mean().item()
                    if self.global_rank == 0:
                        self.logger.experiment.add_scalar(f"rec_loss{postfix}/mse", mse_val, self.num_samples)
                        self.logger.experiment.add_scalar(f"rec_loss{postfix}/psnr", psnr_val, self.num_samples)
                        self.logger.experiment.add_scalar(f"rec_loss{postfix}/l1", l1_val, self.num_samples)
                        self.logger.experiment.add_scalar(f"rec_loss{postfix}/lpips", lpips_val, self.num_samples)
                    self.log("lpips", lpips_val, sync_dist=True)

                    # Outer diffusion model: Update final.ckpt if LPIPS is minimal
                    if lpips_val < self.best_lpips:
                        self.best_lpips = lpips_val
                        final_ckpt_path = os.path.join(self.conf.logdir, "final.ckpt")
                        if hasattr(self, "trainer") and self.trainer is not None:
                            # every rank calls it; Lightning writes on rank 0
                            self.trainer.save_checkpoint(final_ckpt_path)
                            if self.global_rank == 0:
                                print(f"[Best LPIPS] Updated final.ckpt with LPIPS={lpips_val:.4f} at sample {self.num_samples}")

                if self.global_rank == 0:
                    # save samples to single samples directory
                    grid = (make_grid(gen) + 1) / 2
                    sample_dir = os.path.join(
                        self.conf.logdir,
                        "samples" if not postfix else f"samples{postfix}",
                    )
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    path = os.path.join(sample_dir, "%d.png" % self.num_samples)
                    save_image(grid, path)
                    self.logger.experiment.add_image(
                        f"sample{postfix}", grid, self.num_samples
                    )
            model.train()

        if self.conf.train_mode.require_dataset_infer():
            do(self.model, "", use_xstart=False)
            do(self.ema_model, "_ema", use_xstart=False)
        else:
            if self.conf.model_type.has_autoenc() and self.conf.model_type.can_sample():
                do(self.model, "", use_xstart=False)
                do(self.ema_model, "_ema", use_xstart=False)
                # autoencoding mode
                do(self.model, "_enc", use_xstart=True, save_real=True)
                do(self.ema_model, "_enc_ema", use_xstart=True, save_real=True)
            elif self.conf.train_mode.use_latent_net():
                do(self.model, "", use_xstart=False)
                do(self.ema_model, "_ema", use_xstart=False)
                # autoencoding mode
                do(self.model, "_enc", use_xstart=True, save_real=True)
                do(
                    self.model,
                    "_enc_nodiff",
                    use_xstart=True,
                    save_real=True,
                    no_latent_diff=True,
                )
                do(self.ema_model, "_enc_ema", use_xstart=True, save_real=True)
            else:
                do(self.model, "", use_xstart=True, save_real=True)
                do(self.ema_model, "_ema", use_xstart=True, save_real=True)

        # Always save checkpoint when images are created
        # All ranks must call save_checkpoint (it ends in a barrier); Lightning
        # writes the file on rank 0 only.
        if hasattr(self, "trainer") and self.trainer is not None:
            last_ckpt_path = os.path.join(self.conf.logdir, "last.ckpt")
            self.trainer.save_checkpoint(last_ckpt_path)
            if self.global_rank == 0:
                print(f"[Checkpoint] Always saved last.ckpt at sample {self.num_samples}")

    def evaluate_scores(self):
        """
        evaluate FID and other scores during training (logged when sampling images)
        """
        def fid(model, postfix):
            score = evaluate_fid(
                self.eval_sampler,
                model,
                self.conf,
                device=self.device,
                train_data=self.train_data,
                val_data=self.val_data,
                latent_sampler=self.eval_latent_sampler,
                conds_mean=self.conds_mean,
                conds_std=self.conds_std,
            )
            # evaluate_fid computes the score on rank 0; make every rank agree
            # on it before the collective calls below (self.log with a
            # checkpoint monitor, Trainer.save_checkpoint's barrier).
            if hasattr(self, "trainer") and self.trainer is not None:
                score = self.trainer.strategy.broadcast(
                    float(score) if self.global_rank == 0 else 0.0, src=0
                )
            self.log(f"FID{postfix}", score, prog_bar=True)
            if self.global_rank == 0:
                self.logger.experiment.add_scalar(f"FID{postfix}", score, self.num_samples)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, "eval.txt"), "a") as f:
                    metrics_dict = {
                        f"FID{postfix}": score,
                        "num_samples": self.num_samples,
                    }
                    f.write(json.dumps(metrics_dict) + "\n")

            # Latent / Outer model: Update final.ckpt if FID is minimal
            # (all ranks: save_checkpoint is a collective; rank 0 writes)
            if score < self.best_fid:
                self.best_fid = score
                final_ckpt_path = os.path.join(self.conf.logdir, "final.ckpt")
                if hasattr(self, "trainer") and self.trainer is not None:
                    self.trainer.save_checkpoint(final_ckpt_path)
                    if self.global_rank == 0:
                        print(f"[Best FID] Updated final.ckpt with FID={score:.4f} at sample {self.num_samples}")

        print(f"eval fid @ {self.num_samples}")
        fid(self.model, "")
        if hasattr(self, "ema_model"):
            fid(self.ema_model, "_ema")
            fid(self.ema_model, "_ema")
            # it's too slow
            # lpips(self.ema_model, '_ema')

    def configure_optimizers(self):
        out = {}
        print(f"Optimizer {self.conf.optimizer} with lr {self.conf.lr}")
        self.first_encoder_parameter2 = torch.tensor(
            list(self.model.encoder.parameters())[0]
        )
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        if self.conf.encoder is None:
            parameters = self.model.parameters()

        else:
            parameters = [
                param
                for param in self.model.parameters()
                if param not in set(self.model.encoder.parameters())
            ]

        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(
                parameters,
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay,
            )
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(
                parameters,
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay,
            )
        else:
            raise NotImplementedError()
        out["optimizer"] = optim
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(
                optim, lr_lambda=WarmupLR(self.conf.warmup)
            )
            out["lr_scheduler"] = {
                "scheduler": sched,
                "interval": "step",
            }
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank : (rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        for the "eval" mode.
        We first select what to do according to the "conf.eval_programs".
        test_step will only run for "one iteration" (it's a hack!).

        We just want the multi-gpu support.
        """
        # make sure you seed each worker differently!
        self.setup()

        # it will run only one step!
        print("global step:", self.global_step)
        """
        "infer" = predict the latent variables using the encoder on the whole dataset
        """
        if "infer" in self.conf.eval_programs:
            if "infer" in self.conf.eval_programs:
                print("infer ...")
                conds = self.infer_whole_dataset().float()
                save_path = os.path.join(self.conf.logdir, "latent.pkl")
            else:
                raise NotImplementedError()

            if self.global_rank == 0:
                conds_mean = conds.mean(dim=0)
                conds_std = conds.std(dim=0)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(
                    {
                        "conds": conds,
                        "conds_mean": conds_mean,
                        "conds_std": conds_std,
                    },
                    save_path,
                )
        """
        "infer+render" = predict the latent variables using the encoder on the whole dataset
        THIS ALSO GENERATE CORRESPONDING IMAGES
        """
        # infer + reconstruction quality of the input
        for each in self.conf.eval_programs:
            if each.startswith("infer+render"):
                m = re.match(r"infer\+render([0-9]+)", each)
                if m is not None:
                    T = int(m[1])
                    self.setup()
                    print(f"infer + reconstruction T{T} ...")
                    conds = self.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=f"latent_infer_render{T}/{self.conf.name}.lmdb",
                    )
                    save_path = f"latent_infer_render{T}/{self.conf.name}.pkl"
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    torch.save(
                        {
                            "conds": conds,
                            "conds_mean": conds_mean,
                            "conds_std": conds_std,
                        },
                        save_path,
                    )

        # evals those "fidXX"
        """
        "fid<T>" = unconditional generation (conf.train_mode = diffusion).
            Note:   Diff. autoenc will still receive real images in this mode.
        "fid<T>,<T_latent>" = unconditional generation for latent models (conf.train_mode = latent_diffusion).
            Note:   Diff. autoenc will still NOT receive real images in this made.
                    but you need to make sure that the train_mode is latent_diffusion.
        """
        for each in self.conf.eval_programs:
            if each.startswith("fid"):
                m = re.match(r"fid\(([0-9]+),([0-9]+)\)", each)
                clip_latent_noise = False
                if m is not None:
                    # eval(T1,T2)
                    T = int(m[1])
                    T_latent = int(m[2])
                    print(f"evaluating FID T = {T}... latent T = {T_latent}")
                else:
                    m = re.match(r"fidclip\(([0-9]+),([0-9]+)\)", each)
                    if m is not None:
                        # fidclip(T1,T2)
                        T = int(m[1])
                        T_latent = int(m[2])
                        clip_latent_noise = True
                        print(
                            f"evaluating FID (clip latent noise) T = {T}... latent T = {T_latent}"
                        )
                    else:
                        # evalT
                        _, T = each.split("fid")
                        T = int(T)
                        T_latent = None
                        print(f"evaluating FID T = {T}...")

                self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                if T_latent is not None:
                    latent_sampler = self.conf._make_latent_diffusion_conf(
                        T=T_latent
                    ).make_sampler()
                else:
                    latent_sampler = None

                conf = self.conf.clone()
                conf.eval_num_images = 50_000
                score = 0.0
                """score = evaluate_fid(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    latent_sampler=latent_sampler,
                    conds_mean=self.conds_mean,
                    conds_std=self.conds_std,
                    remove_cache=False,
                    clip_latent_noise=clip_latent_noise,
                )"""
                if T_latent is None:
                    self.log(f"fid_ema_T{T}", score)
                else:
                    name = "fid"
                    if clip_latent_noise:
                        name += "_clip"
                    name += f"_ema_T{T}_Tlatent{T_latent}"
                    self.log(name, score)
        """
        "recon<T>" = reconstruction & autoencoding (without noise inversion)
        """
        for each in self.conf.eval_programs:
            if each.startswith("recon"):
                self.model: BeatGANsAutoencModel
                _, T = each.split("recon")
                T = int(T)
                print(f"evaluating reconstruction T = {T}...")

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = 0.0
                """score = evaluate_lpips(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    val_data=self.val_data,
                    latent_sampler=None,
                )"""
                for k, v in score.items():
                    self.log(f"{k}_ema_T{T}", v)
        """
        "inv<T>" = reconstruction with noise inversion
        """
        for each in self.conf.eval_programs:
            if each.startswith("inv"):
                self.model: BeatGANsAutoencModel
                _, T = each.split("inv")
                T = int(T)
                print(f"evaluating reconstruction with noise inversion T = {T}...")

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = 0.0
                """score = evaluate_lpips(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    val_data=self.val_data,
                    latent_sampler=None,
                    use_inverted_noise=True,
                )"""
                for k, v in score.items():
                    self.log(f"{k}_inv_ema_T{T}", v)


def ema(source, target, decay):
    """
    Updates the target model parameters using Exponential Moving Average (EMA) of
    the source model parameters.

    Parameters
    ----------
    source : torch.nn.Module
        The source model whose parameters are used to update the target model.
    target : torch.nn.Module
        The target model whose parameters are updated.
    decay : float
        The decay rate for the EMA. Should be a value between 0 and 1, where a higher
        value means more weight is given to the previous target parameters.

    Returns
    -------
    None
    """

    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


class EMAModelCheckpoint(ModelCheckpoint):
    @property
    def state_key(self) -> str:
        return f"{super().state_key}_ema"


def train(
    conf: TrainConfig,
    nodes=1,
    mode: str = "train",
    max_time=None,
    checkpoint_name="last.ckpt",
):
    print("conf:", conf)
    # Fixed input shapes (batch, 3, img_size, img_size) all run long: let cuDNN
    # autotune the conv algorithms once instead of using the heuristic pick.
    torch.backends.cudnn.benchmark = True
    # assert not (conf.fp16 and conf.grad_clip > 0
    #             ), 'pytorch lightning has bug with amp + gradient clipping'

    # If we are not using latent diffusion, we can use strict loading
    # Otherwise we init an untrained latent model and train it now
    strict_loading = not conf.train_mode is TrainMode.latent_diffusion
    # Always resume from this stage's own logdir. A conf.pretrain (the latent
    # stage initialised from the trained autoencoder) is loaded with
    # strict=False inside LitModel.__init__; handing that path to
    # trainer.fit(ckpt_path=...) instead made Lightning restore it strictly,
    # which fails on the missing latent_net / conds_* keys and also drags the
    # autoencoder's optimizer and loop state into the fresh latent training.
    checkpoint_path = f"{conf.logdir}/{checkpoint_name}"
    print("ckpt path:", checkpoint_path)
    resume_ckpt = None
    if os.path.exists(checkpoint_path):
        resume_ckpt = checkpoint_path
        print("resume from checkpoint:", resume_ckpt)
        model = LitModel.load_from_checkpoint(
            resume_ckpt, conf=conf, strict=strict_loading
        )
    else:
        if conf.continue_from is not None and os.path.exists(conf.continue_from.path):
            resume_ckpt = conf.continue_from.path
            print("resume from continue_from path:", resume_ckpt)
            model = LitModel.load_from_checkpoint(
                resume_ckpt, conf=conf, strict=strict_loading
            )
        else:
            model = LitModel(conf=conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)

    print("logdir:", conf.logdir)
    save_train_steps = conf.save_every_samples // conf.batch_size_effective
    print("save train steps:", save_train_steps)

    checkpoint = ModelCheckpoint(
        dirpath=f"{conf.logdir}",
        save_last=True,
        save_top_k=2,
        monitor="loss",
        every_n_train_steps=save_train_steps,
        verbose=True,
        filename="{step:07d}",
        # Overwrite last.ckpt on every save instead of writing last-v1.ckpt,
        # last-v2.ckpt, ... after a resume (Lightning refuses to clobber a
        # last.ckpt it did not write itself). train() resumes from
        # <logdir>/last.ckpt unconditionally, so with the version counter
        # on, every restart silently rewound to the first crash.
        enable_version_counter=False,
    )
    checkpoint_ema = EMAModelCheckpoint(
        dirpath=os.path.join(conf.logdir, "ema"),
        save_last=True,
        save_top_k=2,
        monitor="loss",
        every_n_train_steps=save_train_steps,
        verbose=True,
        filename="{step:07d}",
        # Overwrite last.ckpt on every save instead of writing last-v1.ckpt,
        # last-v2.ckpt, ... after a resume (Lightning refuses to clobber a
        # last.ckpt it did not write itself). train() resumes from
        # <logdir>/last.ckpt unconditionally, so with the version counter
        # on, every restart silently rewound to the first crash.
        enable_version_counter=False,
    )

    tb_logger = TensorBoardLogger(save_dir=conf.logdir, name="", version="")

    max_steps = conf.total_samples // conf.batch_size_effective
    print("max steps:", max_steps)
    trainer = L.Trainer(
        max_steps=max_steps,
        num_nodes=nodes,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else "auto"
        ),
        # conf.precision (a Lightning string such as "bf16-mixed") overrides the
        # legacy fp16 flag; bf16 needs no loss scaler and cannot overflow.
        precision=(conf.precision or (16 if conf.fp16 else 32)),
        callbacks=[
            checkpoint,
            checkpoint_ema,
            LearningRateMonitor(),
        ],
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        log_every_n_steps=5,
        max_time=max_time or conf.max_time,
    )

    if mode == "train":
        trainer.fit(
            model,
            ckpt_path=(
                resume_ckpt if (resume_ckpt and os.path.exists(resume_ckpt)) else None
            ),
        )
    elif mode == "eval":
        # load the latest checkpoint
        # perform lpips
        # dummy loader to allow calling "test_step"
        dummy = DataLoader(
            TensorDataset(torch.tensor([0.0] * conf.batch_size)),
            batch_size=conf.batch_size,
        )
        eval_path = conf.eval_path or checkpoint_path
        # conf.eval_num_images = 50
        print("loading from:", eval_path)
        state = torch.load(eval_path, map_location="cpu")
        print("step:", state["global_step"])
        model.load_state_dict(state["state_dict"])
        # trainer.fit(model)
        out = trainer.test(model, dataloaders=dummy)
        # first (and only) loader
        out = out[0]
        print(out)

        if get_rank() == 0:
            # save to tensorboard
            for k, v in out.items():
                tb_logger.experiment.add_scalar(
                    k, v, state["global_step"] * conf.batch_size_effective
                )

            # # save to file
            # # make it a dict of list
            # for k, v in out.items():
            #     out[k] = [v]
            tgt = f"evals/{conf.name}.txt"
            dirname = os.path.dirname(tgt)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(tgt, "a") as f:
                f.write(json.dumps(out) + "\n")
            # pd.DataFrame(out).to_csv(tgt)
    else:
        raise NotImplementedError()
