import copy
import functools
import os
import glob
import copy
import shutil
import yaml

import blobfile as bf
import torch as th
import torch.distributed as dist
import torchvision.utils
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .sample_utils import load_from_DDP_model

from peal.global_utils import save_yaml_config

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        model_dir,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.model_dir = model_dir
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                print(f"loading model from checkpoint: {resume_checkpoint}...")
                # self.model.load_state_dict(
                #     dist_util.load_state_dict(
                #         resume_checkpoint, map_location=dist_util.dev()
                #     )
                # )
                self.model.load_state_dict(
                    load_from_DDP_model(
                        th.load(resume_checkpoint, map_location=dist_util.dev())
                    )
                )
                print("done")

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                print(f"loading EMA from checkpoint: {ema_checkpoint}...")
                # state_dict = dist_util.load_state_dict(
                #     ema_checkpoint, map_location=dist_util.dev()
                # )
                state_dict = load_from_DDP_model(
                    th.load(ema_checkpoint, map_location=dist_util.dev())
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
                print("done")

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self, config):
        print("Running training loop")
        pbar = tqdm(
            total=self.data.dataloader.train_config.steps_per_epoch
            * self.batch_size
            / self.microbatch
        )
        pbar.num_steps = self.data.dataloader.train_config.steps_per_epoch
        pbar.config = config
        shutil.rmtree(self.model_dir, ignore_errors=True)
        os.makedirs(os.path.join(self.model_dir, "ema"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "model"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "optimizer"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "outputs"), exist_ok=True)
        pbar.writer = SummaryWriter(os.path.join(self.model_dir, "logs"))
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            pbar.step = self.step
            batch, cond = next(self.data)
            self.run_step(batch, cond, pbar)
            if self.step % self.save_interval == 0:
                # save intermediate images as outputs
                imgs = self.diffusion.p_sample_loop(self.model, batch.shape)
                torchvision.utils.save_image(
                    imgs,
                    os.path.join(
                        copy.deepcopy(self.model_dir),
                        os.path.join(
                            "outputs", f"{(self.step+self.resume_step):06d}.png"
                        ),
                    ),
                )
                train_generator_performance = (
                    self.data.dataloader.dataset.track_generator_performance(
                        imgs, self.data.dataloader.batch_size
                    )
                )
                pbar.config.current_fid = train_generator_performance["fid"]
                pbar.writer.add_scalar(
                    "fid", train_generator_performance["fid"], self.step + self.resume_step
                )
                self.save()
                save_yaml_config(pbar.config, os.path.join(self.model_dir, "config.yaml"))

                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, pbar=None):
        self.forward_backward(batch, cond, pbar)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        # self.log_step()

    def forward_backward(self, batch, cond, pbar=None):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            pbar.set_description(
                f"Step: {pbar.step} / {pbar.num_steps}, "
                + f"Batch: {i} / {int(batch.shape[0] / self.microbatch + 0.99)}"
            )
            pbar.update(1)
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            # log_loss_dict(
            #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
            # )
            self.mp_trainer.backward(loss)
            pbar.writer.add_scalar("loss", loss.detach().item(), self.step + self.resume_step)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, pbar = None):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                print(f"saving model {rate}...")
                if not rate:
                    filename = os.path.join(f"model",  f"{(self.step+self.resume_step):06d}.pt")
                    if not pbar is None and pbar.config.current_fid < pbar.config.best_fid:
                        pbar.config.best_fid = pbar.config.current_fid
                        with bf.BlobFile(
                            bf.join(copy.deepcopy(self.model_dir), f"final.pt"), "wb"
                        ) as f:
                            th.save(state_dict, f)
                else:
                    filename = os.path.join(f"ema", f"{rate}_{(self.step+self.resume_step):06d}.pt")

                with bf.BlobFile(
                    bf.join(copy.deepcopy(self.model_dir), filename), "wb"
                ) as f:
                    th.save(state_dict, f)

        # delete old checkpoints
        """if dist.get_rank() == 0:
            for f in glob.glob(os.path.join(copy.deepcopy(self.model_dir), "*.pt")):
                os.remove(os.path.join(copy.deepcopy(self.model_dir), f))"""

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(
                    copy.deepcopy(self.model_dir),
                    os.path.join(f"opt", f"{(self.step+self.resume_step):06d}.pt"),
                ),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
