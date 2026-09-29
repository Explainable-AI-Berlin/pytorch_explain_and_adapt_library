"""Stage 2 training script for flow matching on RAE latents."""

import argparse
import dataclasses
import json
import math
import os
from functools import partial
from typing import Optional

os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from copy import deepcopy

import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from tqdm.auto import tqdm

from configs.stage2 import Stage2Config
from data import prepare_unified_dataloader
from encoders.vision_encoder import load_encoders
from eval.datasets import normalize_eval_datasets, prepare_eval_datasets
from stage1 import RAE
from stage2.engine import train_one_epoch
from stage2.models import Stage2ModelProtocol
from stage2.state_utils import (
    AuxStateSpec,
    infer_aux_state_spec,
    make_initial_sample_state,
    validate_decode_target_compatibility,
)
from stage2.transport import create_sampler, create_transport
from stage2.utils import setup_text_encoder, validate_stage2_config
from utils.checkpoint import load_stage2_checkpoint, save_stage2_checkpoint
from utils.dist_utils import cleanup_distributed, main_process_first, setup_distributed
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import configure_experiment_dirs, find_resume_checkpoint, save_worktree
from utils.sync_utils import sync_checkpoint_blocking, sync_evals_blocking
from utils.train_utils import center_crop_arr, get_autocast_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--results-dir", type=str, default="ckpts")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sync-checkpoints", action="store_true")
    parser.add_argument("--compile", action="store_true", help="torch.compile the training loss function")
    return parser.parse_args()


def main():
    """Train Stage 2 model using config-driven hyperparameters."""
    args = parse_args()

    #########################################################
    # Distributed + config setup
    #########################################################
    rank, world_size, device = setup_distributed()
    config: Stage2Config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
    config.post_process()
    validate_stage2_config(config)
    uses_stage2_latent_cache = (
        config.dataset.target == "stage2_latent_cache"
        or config.dataset.type == "stage2_latent_cache"
    )

    seed = config.training.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    autocast_kwargs = get_autocast_kwargs(args)

    #########################################################
    # Data setup; train and eval
    #########################################################
    global_batch_size = config.training.global_batch_size or (config.training.batch_size * world_size * config.training.grad_accum_steps)
    assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    micro_batch_size = global_batch_size // (world_size * config.training.grad_accum_steps)
    stage2_transform = transforms.Compose([
        transforms.Lambda(partial(center_crop_arr, image_size=config.training.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    needs_transform = (
        config.dataset.type not in ("hf", "wds", "histo_wds", "stage2_latent_cache")
        and not uses_stage2_latent_cache
    )
    # train dataloader
    dataloader = prepare_unified_dataloader(
        config=dataclasses.asdict(config.dataset),
        image_size=config.training.image_size,
        batch_size=micro_batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        transform=stage2_transform if needs_transform else None,
        condition_type=config.conditioning.type,
        virtual_epoch_steps=config.training.virtual_epoch_steps,
    )
    train_dataset = getattr(dataloader.loader, "dataset", None)
    if (train_dataset is None or not hasattr(train_dataset, "y_vocab")) and getattr(dataloader, "_wds_pipeline", None) is not None:
        train_dataset = dataloader._wds_pipeline

    # eval setup
    eval_datasets, eval_dir = None, None
    if config.eval is not None:
        eval_datasets_config = normalize_eval_datasets(config.eval.datasets)
        if eval_datasets_config is not None:
            eval_datasets = prepare_eval_datasets(
                eval_datasets_config,
                image_size=config.training.image_size,
                batch_size=micro_batch_size,
                num_workers=config.training.num_workers,
                rank=rank,
                world_size=world_size,
            )
        eval_dir = config.eval.eval_dir

    #########################################################
    # Models setup
    #########################################################
    latent_size = tuple(config.misc.latent_size)

    # stage1: rae - frozen
    rae: RAE = instantiate_from_config(config.stage_1).to(device)
    rae.eval()
    if uses_stage2_latent_cache and hasattr(rae, "encoder"):
        rae.encoder.to("cpu")
        if rank == 0:
            logger.info("Stage-2 latent cache active: moved frozen RAE encoder to CPU; decoder remains on GPU.")
    decode_rae: Optional[RAE] = None
    if config.decode_stage_1 is not None and config.decode_stage_1.target:
        decode_rae = instantiate_from_config(config.decode_stage_1).to(device)
        decode_rae.eval()
        validate_decode_target_compatibility(rae, decode_rae, latent_size=latent_size)

    if train_dataset is not None and hasattr(train_dataset, "y_vocab"):
        label_count = None
        labels_attr = getattr(train_dataset, "labels", None)
        if labels_attr is not None:
            label_count = int(labels_attr.max()) + 1
        y_vocab = getattr(train_dataset, "y_vocab", {}) or {}
        vocab_count = len(set(int(v) for v in y_vocab.values())) if isinstance(y_vocab, dict) else 0
        config.misc.num_classes = max(int(config.misc.num_classes), int(label_count or 0), int(vocab_count))
        if hasattr(train_dataset, "meta_cardinalities") and config.conditioning.type == "label_meta":
            config.stage_2.params["meta_num_classes"] = list(train_dataset.meta_cardinalities)

        if rank == 0:
            with open(os.path.join(experiment_dir, "y_vocab.json"), "w", encoding="utf-8") as handle:
                json.dump(getattr(train_dataset, "y_vocab", {}), handle, indent=2, sort_keys=True)
            if getattr(train_dataset, "meta_vocabs", None):
                with open(os.path.join(experiment_dir, "meta_vocabs.json"), "w", encoding="utf-8") as handle:
                    json.dump(train_dataset.meta_vocabs, handle, indent=2, sort_keys=True)

    # repa target encoder
    repa_target_encoder = None
    if config.repa.use_repa:
        with main_process_first(rank):
            repa_target_encoder = load_encoders(config.repa.target_encoder, device, config.repa.target_encoder_resolution)[0]
        repa_target_encoder.eval()
        repa_target_encoder.model.requires_grad_(False)
        config.repa.z_dim = repa_target_encoder.embed_dim
        logger.info(f"REPA target encoder: {config.repa.target_encoder}, embed_dim={repa_target_encoder.embed_dim}")

    # text encoder for text conditioning; None if not using text conditioning
    text_encoder = setup_text_encoder(config, rank, device)

    # prepare model params (must be called before model instantiation so that
    # condition_type, context_dim, repa z_dim etc. are set)
    config.prepare_model_params()

    # stage2: model - trainable
    model: Stage2ModelProtocol = instantiate_from_config(config.stage_2).to(device)
    model.requires_grad_(True)
    # stage2 ema model
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()

    # ddp wrapper for stage2 model
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    model = ddp_model.module
    ddp_model.train()
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if args.wandb and rank == 0:
        import wandb
        wandb.config.update({
            "model_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "trainable_params_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1),
        }, allow_val_change=True)

    #########################################################
    # Optimizer + Scheduler setup
    #########################################################
    optimizer, _ = build_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        config.training.optimizer,
    )

    #########################################################
    # Steps per epoch setup
    #########################################################
    steps_per_epoch = len(dataloader) // config.training.grad_accum_steps
    logger.info(f"Using {steps_per_epoch} steps per epoch (virtual={config.training.virtual_epoch_steps is not None})")

    # Build scheduler (needs steps_per_epoch)
    scheduler = None
    sched_msg = None
    if config.training.scheduler is not None:
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, config.training.scheduler)

    #########################################################
    # Transport + Sampler setup
    #########################################################
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base
    )
    transport = create_transport(
        config=config.transport,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = create_sampler(transport, guidance_config=config.guidance)
    eval_sampler = transport_sampler.sample_ode(**dataclasses.asdict(config.sampler))

    if args.compile:
        transport.training_losses = torch.compile(transport.training_losses)

    #########################################################
    # Resume setup
    #########################################################
    start_epoch = 0
    global_step = 0
    ckpt_path = find_resume_checkpoint(experiment_dir, args.ckpt)
    if ckpt_path:
        start_epoch, global_step = load_stage2_checkpoint(ckpt_path, ddp_model, ema_model, optimizer, scheduler)
        logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
    else:
        if rank == 0:
            save_worktree(experiment_dir, config)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")

    total_steps = config.training.epochs * steps_per_epoch
    progress_bar = tqdm(total=total_steps, initial=global_step, desc="Training", disable=rank != 0)

    # fixed state for consistent visualization across epochs (populated from first batch)
    num_viz_samples = min(micro_batch_size, 32)
    uses_joint_state = bool(getattr(model, "predict_aux", False) or getattr(model, "predict_aux_tokens", False))
    aux_state_spec = (
        infer_aux_state_spec(rae, latent_size=latent_size)
        if uses_joint_state
        else AuxStateSpec(mode="none", shape=None)
    )
    viz_fixed = {
        'zs': make_initial_sample_state(
            num_viz_samples,
            latent_size,
            aux_state_spec,
            device,
        ),
        'context': None,
        'attn_mask': None,
        'meta': None,
        'cls': None,
    }

    #########################################################
    # Training loop
    #########################################################
    dist.barrier()
    for epoch in range(start_epoch, config.training.epochs):
        model.train()

        global_step = train_one_epoch(
            ddp_model=ddp_model,
            ema_model=ema_model,
            rae=rae,
            transport=transport,
            eval_sampler=eval_sampler,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            autocast_kwargs=autocast_kwargs,
            device=device,
            epoch=epoch,
            global_step=global_step,
            config=config,
            args=args,
            rank=rank,
            world_size=world_size,
            micro_batch_size=micro_batch_size,
            checkpoint_dir=checkpoint_dir,
            experiment_dir=experiment_dir,
            progress_bar=progress_bar,
            text_encoder=text_encoder,
            repa_target_encoder=repa_target_encoder,
            decode_rae=decode_rae,
            eval_datasets=eval_datasets,
            viz_fixed=viz_fixed,
        )
    progress_bar.close()

    #########################################################
    # final checkpoint setup and cleanup
    #########################################################
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {config.training.epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-{config.training.epochs:07d}.pt"
        save_stage2_checkpoint(ckpt_path, global_step, config.training.epochs, ddp_model, ema_model, optimizer, scheduler)
        if args.sync_checkpoints:
            sync_checkpoint_blocking(checkpoint_dir, logger)
            if eval_dir: sync_evals_blocking(eval_dir, logger)

    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
