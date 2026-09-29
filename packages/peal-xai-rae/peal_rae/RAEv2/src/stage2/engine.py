"""Stage 2 training engine: train_one_epoch and helpers."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, Optional

from PIL import Image
import torch
import torch.distributed as dist
import wandb
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.stage2 import Stage2Config
from stage2 import nwm_cond
from stage2.state_utils import state_batch_size
from stage2.utils import (
    encode_text,
    get_fixed_viz_batch_conditions,
    get_null_cond,
    sample_and_decode,
)
from utils import wandb_utils
from utils.checkpoint import save_stage2_checkpoint
from utils.guidance_utils import get_model_forward_fn
from utils.logging import save_eval_to_csv
from utils.sync_utils import sync_checkpoint_async, sync_evals_async
from utils.train_utils import update_ema

logger = logging.getLogger("rae")


#########################################################
# Main training function
#########################################################
def train_one_epoch(
    *, # * forces all arguments to be passed as keyword arguments
    ddp_model: DDP,
    ema_model: torch.nn.Module,
    rae,
    transport,
    eval_sampler,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    autocast_kwargs: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: Stage2Config,
    args,
    rank: int,
    world_size: int,
    micro_batch_size: int,
    checkpoint_dir: str,
    experiment_dir: str,
    progress_bar,
    text_encoder=None,
    repa_target_encoder=None,
    decode_rae=None,
    eval_datasets: Optional[Dict] = None,
    viz_fixed: Optional[Dict] = None,
) -> int:
    """Run one epoch of Stage 2 training. Returns updated global_step.

    Args:
        viz_fixed: Mutable dict with keys 'zs', 'y', 'encoder_hidden_states',
            'encoder_attention_mask'. Populated from first batch, persists across epochs.
    """
    #########################################################
    # Setup
    #########################################################
    model = ddp_model.module
    uses_joint_state = bool(getattr(model, "predict_aux", False) or getattr(model, "predict_aux_tokens", False))
    use_cls_conditioning = bool(getattr(model, "cls_conditioning", False))
    logged_cls_condition = False
    logged_first_step_grad = False

    if rank == 0:
        stage2_init = "pretrained" if getattr(config.stage_2, "ckpt", None) else "random"
        logger.info(
            "Stage2 launch audit: target=%s, prediction=%s, repa.use_repa=%s, "
            "internal_guidance.base_model_coeff=%s, stage2_initialization=%s, checkpoint=%s, "
            "global_batch=%s, micro_batch=%s, grad_accum=%s, cls_conditioning=%s, "
            "condition_dropout=joint",
            config.stage_2.target,
            config.transport.prediction,
            config.repa.use_repa,
            config.internal_guidance.base_model_coeff,
            stage2_init,
            getattr(config.stage_2, "ckpt", None),
            config.training.global_batch_size,
            micro_batch_size,
            config.training.grad_accum_steps,
            use_cls_conditioning,
        )
        if use_cls_conditioning:
            cls_param_count = sum(
                p.numel()
                for name, p in model.named_parameters()
                if name.startswith("cls_embedder") or name.startswith("null_cls")
            )
            logger.info(
                "CLS/global conditioner audit: cls_dim=%s, condition_source=%s, trainable_params=%s",
                getattr(model, "cls_dim", None),
                getattr(model, "cls_condition_source", "unknown"),
                cls_param_count,
            )

    # Guidance: derive model_fn / ema_model_fn / sample_kwargs from config
    model_fn, sample_model_kwargs = get_model_forward_fn(model, config.guidance)
    ema_model_fn, _ = get_model_forward_fn(ema_model, config.guidance)
    use_guidance = config.guidance.any_guidance_active

    # Eval settings
    do_eval = config.eval is not None and eval_datasets is not None
    if do_eval: eval_dir = config.eval.eval_dir
    experiment_name = os.environ.get("EXPERIMENT_NAME")

    # Get null conditions for CFG dropout
    if config.conditioning.type == "nwm":
        model_kwargs_null = nwm_cond.null_context(config, micro_batch_size, device)
    elif config.conditioning.type == "cls":
        null_cls = getattr(model, "null_cls", None)
        if null_cls is None:
            null_context = torch.zeros(micro_batch_size, getattr(model, "cls_dim", 0), device=device)
        else:
            null_context = null_cls.expand(micro_batch_size, -1).to(device=device)
        model_kwargs_null = dict(context=null_context, attn_mask=None)
    else:
        model_kwargs_null = get_null_cond(
            text_encoder,
            config.conditioning.type,
            config.misc.num_classes,
            micro_batch_size,
            device,
            meta_num_fields=len(config.dataset.meta_fields or []),
        )

    # per-epoch state
    num_viz_samples = state_batch_size(viz_fixed['zs']) if viz_fixed is not None else 0
    epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
    num_batches = 0
    optimizer.zero_grad()

    # save checkpoint at epoch start
    if config.training.checkpoint_interval > 0 and epoch % config.training.checkpoint_interval == 0 and rank == 0:
        logger.info(f"Saving checkpoint at epoch {epoch}...")
        ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
        save_stage2_checkpoint(ckpt_path, global_step, epoch, ddp_model, ema_model, optimizer, scheduler)
        if args.sync_checkpoints:
            sync_checkpoint_async(checkpoint_dir, logger)
            if do_eval: sync_evals_async(eval_dir, logger)

    #########################################################
    # Training loop
    #########################################################
    dataloader.set_epoch(epoch)
    for step, batch in enumerate(dataloader):
        cached_batch = isinstance(batch, dict) and "z" in batch
        if cached_batch:
            images = None
            z_tensor = batch["z"].to(device, non_blocking=True)
            aux = batch.get("aux", None)
            z = (
                z_tensor,
                aux.to(device, non_blocking=True),
            ) if aux is not None else z_tensor
            y = batch["y"].to(device, non_blocking=True)
            meta = batch.get("meta", None)
            if meta is not None:
                meta = meta.to(device, non_blocking=True)
            # A shared cache may contain CLS for all ablations. Do not transfer,
            # log, drop, or forward it when this model has no CLS conditioner.
            cls_context = batch.get("cls", None) if use_cls_conditioning else None
            if cls_context is not None:
                cls_context = cls_context.to(device, non_blocking=True)
            if use_cls_conditioning and cls_context is None:
                raise ValueError(
                    "Stage-2 model has CLS/global conditioning enabled, but latent-cache "
                    "batch does not contain 'cls'. Rebuild the cache with --include-cls."
                )
            if repa_target_encoder is not None:
                raise ValueError("REPA targets require image batches; disable REPA for latent-cache training.")
            z_clean = None
        else:
            if config.conditioning.type == "label_meta":
                images, y, meta = batch
            else:
                images, y = batch[:2]
                meta = None
            images = images.to(device)

            # Encode images to latents and compute REPA targets
            with torch.no_grad():
                cls_context = None
                if use_cls_conditioning and hasattr(rae, "encode_with_cls"):
                    z, cls_context = rae.encode_with_cls(images)
                elif uses_joint_state and hasattr(rae, "encode_for_stage2"):
                    z = rae.encode_for_stage2(images)
                else:
                    z = rae.encode(images)
                if repa_target_encoder is not None:
                    raw_images = images.clone() * 255.0
                    raw_img_preprocessed = repa_target_encoder.preprocess(raw_images)
                    z_clean = repa_target_encoder.forward_features(raw_img_preprocessed)['x_norm_patchtokens']
                else:
                    z_clean = None

        if cls_context is not None and rank == 0 and not logged_cls_condition:
            cls_float = cls_context.detach().float()
            encoder = getattr(rae, "encoder", None)
            encoder_name = getattr(rae, "encoder_name", None)
            encoder_class = encoder.__class__.__name__ if encoder is not None else "unknown"
            selected_layers = getattr(encoder, "layer_indices", None)
            condition_source = getattr(model, "cls_condition_source", "encoder_x_norm_clstoken")
            class_token_requested = None
            if encoder_class == "DINOv3MultiLayerSimpleAddEncoder":
                condition_source = "final_selected_layer_patch_mean"
                class_token_requested = False
            logger.info(
                "First batch CLS/global condition: source=%s, shape=%s, encoder_name=%s, "
                "encoder_class=%s, selected_layers=%s, class_token_requested=%s, "
                "pre_norm_mean=%.6f, pre_norm_std=%.6f, pre_norm_l2_mean=%.6f",
                f"latent_cache:{condition_source}" if cached_batch else condition_source,
                tuple(cls_context.shape),
                encoder_name,
                encoder_class,
                selected_layers,
                class_token_requested,
                cls_float.mean().item(),
                cls_float.std(unbiased=False).item(),
                cls_float.norm(dim=-1).mean().item(),
            )
            logged_cls_condition = True

        # Capture fixed conditions from first batch
        if viz_fixed is not None:
            if config.conditioning.type == "nwm":
                if viz_fixed['context'] is None:
                    viz_fixed['context'] = nwm_cond.viz_context(y, state_batch_size(viz_fixed['zs']), rae, device)
            elif config.conditioning.type == "cls":
                if cls_context is None:
                    raise ValueError("conditioning.type='cls' requires a CLS/global condition for visualization.")
                if viz_fixed['context'] is None:
                    n = state_batch_size(viz_fixed['zs'])
                    viz_fixed['context'] = cls_context[:n].clone().to(device)
            else:
                viz_payload = (y, meta) if config.conditioning.type == "label_meta" else y
                viz_fixed = get_fixed_viz_batch_conditions(viz_fixed, viz_payload, config.conditioning.type, text_encoder, device)
                if (
                    cls_context is not None
                    and config.conditioning.type != "cls"
                    and viz_fixed.get("cls") is None
                ):
                    n = state_batch_size(viz_fixed['zs'])
                    viz_fixed["cls"] = cls_context[:n].clone().to(device)

        # Encode conditions
        if config.conditioning.type == "text":
            context, context_attn_mask = encode_text(text_encoder, y)
        elif config.conditioning.type == "nwm":
            context = nwm_cond.encode_train_context(y, rae, device)
            context_attn_mask = None
        elif config.conditioning.type == "cls":
            if cls_context is None:
                raise ValueError("conditioning.type='cls' requires batch['cls'] or rae.encode_with_cls output.")
            context, context_attn_mask = cls_context.to(device=device), None
        else:
            context, context_attn_mask = y.to(device), None
            if meta is not None:
                meta = meta.to(device)

        #########################################################
        # Forward + backward
        #########################################################
        model_kwargs = dict(context=context, attn_mask=context_attn_mask)
        model_kwargs_null_step = model_kwargs_null
        if config.conditioning.type == "cls":
            if getattr(model, "null_cls", None) is not None:
                null_context = model.null_cls.expand(context.shape[0], -1).to(device=device, dtype=context.dtype)
            else:
                null_context = torch.zeros_like(context)
            model_kwargs_null_step = dict(context=null_context, attn_mask=None)
        if meta is not None:
            model_kwargs["meta"] = meta
        if use_cls_conditioning and cls_context is not None and config.conditioning.type != "cls":
            cls_value = cls_context.to(device=device)
            if getattr(model, "null_cls", None) is not None:
                null_cls = model.null_cls.expand(cls_context.shape[0], -1).to(device=device, dtype=cls_context.dtype)
            else:
                null_cls = torch.zeros_like(cls_context)
            cls_dropout_prob = float(config.conditioning.cls_dropout_prob)
            if cls_dropout_prob > 0:
                cls_drop = torch.rand(cls_value.shape[0], device=device) < cls_dropout_prob
                cls_value = torch.where(cls_drop.unsqueeze(-1), null_cls, cls_value)
            model_kwargs["cls"] = cls_value
            model_kwargs_null_step = dict(model_kwargs_null)
            model_kwargs_null_step["cls"] = null_cls

        is_accum_step = (step + 1) % config.training.grad_accum_steps != 0
        # DDP decides whether to prepare gradient reduction during forward. The
        # no_sync context must therefore cover both forward and backward.
        sync_context = ddp_model.no_sync() if is_accum_step else nullcontext()
        with sync_context:
            with autocast(**autocast_kwargs):
                loss_dict = transport.training_losses(
                    ddp_model, z, model_kwargs, model_kwargs_null_step,
                    z_clean=z_clean,
                    repa_coeff=config.repa.repa_coeff if config.repa.use_repa else None,
                    base_model_coeff=config.internal_guidance.base_model_coeff,
                    cfg_dropout_prob=config.conditioning.cfg_dropout_prob,
                    aux_loss_weight=config.joint_state_loss.aux_loss_weight,
                )
                loss_diff = loss_dict["loss"].mean()
                loss_repa_value = loss_dict.get("loss_repa")
                loss_repa = loss_repa_value.mean() if loss_repa_value is not None else loss_diff.new_zeros(())
                loss = loss_diff + loss_repa if config.repa.use_repa else loss_diff

            loss = loss / config.training.grad_accum_steps
            loss.backward()

        if not is_accum_step:
            if config.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), config.training.clip_grad)
            if use_cls_conditioning and rank == 0 and not logged_first_step_grad:
                grad_sq = 0.0
                param_sq = 0.0
                for name, param in model.named_parameters():
                    if name.startswith("cls_embedder") or name.startswith("null_cls"):
                        param_sq += float(param.detach().float().pow(2).sum().item())
                        if param.grad is not None:
                            grad_sq += float(param.grad.detach().float().pow(2).sum().item())
                peak_mem_gib = (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                logger.info(
                    "First optimizer-step CLS/global audit: grad_norm=%.6e, param_norm=%.6e, gpu_peak_mem_gib=%.3f",
                    grad_sq ** 0.5,
                    param_sq ** 0.5,
                    peak_mem_gib,
                )
                logged_first_step_grad = True
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            update_ema(ema_model, ddp_model.module, decay=config.training.ema_decay)
            global_step += 1

        epoch_metrics['loss'] += loss_diff.detach()
        if "loss_patch" in loss_dict:
            epoch_metrics['loss_patch'] += loss_dict["loss_patch"].detach().mean()
        if "loss_aux" in loss_dict:
            epoch_metrics['loss_aux'] += loss_dict["loss_aux"].detach().mean()
        num_batches += 1
        progress_bar.update(1)

        # Skip logging/viz/eval on non-boundary micro-steps
        if is_accum_step:
            continue

        #########################################################
        # Logging and visualization
        #########################################################
        if config.training.log_interval > 0 and global_step % config.training.log_interval == 0 and rank == 0:
            cur_loss = loss_diff.item()
            stats = {"train/loss": cur_loss, "train/lr": optimizer.param_groups[0]["lr"]}
            if "loss_patch" in loss_dict:
                stats["train/loss_patch"] = loss_dict["loss_patch"].mean().item()
            if "loss_aux" in loss_dict:
                stats["train/loss_aux"] = loss_dict["loss_aux"].mean().item()
            if config.repa.use_repa:
                stats["train/loss_repa"] = loss_repa.item()
            if "loss_base" in loss_dict:
                stats["train/loss_base"] = loss_dict["loss_base"].mean().item()
            logger.info(
                f"[Epoch {epoch} | Step {global_step}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
            )
            if args.wandb:
                wandb_utils.log(stats, step=global_step)
            progress_bar.set_postfix(loss=cur_loss, lr=optimizer.param_groups[0]["lr"])

        # Sampling visualization
        if global_step % config.training.sample_every == 0:
            model.eval()
            logger.info("Generating EMA samples...")
            sample_args = dict(
                eval_sampler=eval_sampler, model_fn=ema_model_fn,
                sample_model_kwargs=sample_model_kwargs, rae=rae,
                decode_rae=decode_rae,
                use_guidance=use_guidance, condition_type=config.conditioning.type,
                text_encoder=text_encoder, num_classes=config.misc.num_classes,
                device=device, autocast_kwargs=autocast_kwargs,
                cls_null=getattr(ema_model, "null_cls", None),
            )
            if rank == 0:
                with torch.no_grad():
                    samples_dict = {}
                    # 1. Batch samples (from current batch conditions)
                    is_dict_ctx = isinstance(context, dict)
                    batch_n = min(num_viz_samples, nwm_cond.batch_size(context) if is_dict_ctx else context.shape[0])
                    zs_batch = torch.randn(batch_n, *config.misc.latent_size, device=device, dtype=torch.float32)
                    samples_dict["samples/batch"] = sample_and_decode(
                        zs_batch, nwm_cond.slice(context, batch_n) if is_dict_ctx else context[:batch_n],
                        context_attn_mask[:batch_n] if context_attn_mask is not None else None,
                        meta=meta[:batch_n] if meta is not None else None,
                        cls=(
                            cls_context[:batch_n]
                            if cls_context is not None and config.conditioning.type != "cls"
                            else None
                        ),
                        **sample_args,
                    )
                    # 2. Fixed samples (consistent across epochs)
                    if viz_fixed is not None and viz_fixed['context'] is not None:
                        fixed_ctx = viz_fixed['context']
                        fixed_ctx_clone = nwm_cond.clone_context(fixed_ctx) if isinstance(fixed_ctx, dict) else fixed_ctx.clone()
                        samples_dict["samples/fixed"] = sample_and_decode(
                            tuple(part.clone() for part in viz_fixed['zs']) if isinstance(viz_fixed['zs'], tuple) else viz_fixed['zs'].clone(), fixed_ctx_clone,
                            viz_fixed['attn_mask'].clone() if viz_fixed['attn_mask'] is not None else None,
                            meta=viz_fixed.get('meta').clone() if viz_fixed.get('meta') is not None else None,
                            cls=viz_fixed.get('cls').clone() if viz_fixed.get('cls') is not None else None,
                            **sample_args,
                        )
                    if args.wandb: # log samples to wandb
                        for name, samples in samples_dict.items():
                            grid = wandb_utils.array2grid(samples)
                            wandb.log({name: wandb.Image(grid)}, step=global_step)
                    sample_dir = os.path.join(experiment_dir, "ema_samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    for name, samples in samples_dict.items():
                        grid = wandb_utils.array2grid(samples)
                        safe_name = name.replace("/", "_")
                        sample_path = os.path.join(
                            sample_dir,
                            f"epoch{epoch:04d}_step{global_step:07d}_{safe_name}.png",
                        )
                        Image.fromarray(grid).save(sample_path)
                    if samples_dict:
                        logger.info("Saved EMA sample grids to %s", sample_dir)
            dist.barrier()
            logger.info("Generating EMA samples done.")
            model.train() # set model back to train mode

        #########################################################
        # Evaluation; distributed evaluation
        #########################################################
        if do_eval and config.eval.eval_interval > 0 and global_step % config.eval.eval_interval == 0:
            from eval import evaluate_generation_distributed
            logger.info("Starting evaluation...")
            model.eval()
            # eval ema or both ema and running model if eval_model is True
            eval_models = [(ema_model_fn, "ema")] if not config.eval.eval_model else [(ema_model_fn, "ema"), (model_fn, "model")]
            for fn, mod_name in eval_models:
                for ds_name, ds_info in eval_datasets.items():
                    logger.info(f"Evaluating {mod_name} on {ds_name}...")
                    eval_dataset = ds_info.require_indexable_dataset(ds_name)
                    eval_n = min(ds_info.num_samples or len(ds_info), len(ds_info))
                    eval_stats = evaluate_generation_distributed(
                        fn, eval_sampler, tuple(config.misc.latent_size), sample_model_kwargs,
                        use_guidance, rae, eval_dataset, eval_n,
                        rank=rank, world_size=world_size, device=device,
                        batch_size=micro_batch_size, experiment_dir=experiment_dir,
                        global_step=global_step, autocast_kwargs=autocast_kwargs,
                        reference_npz_path=ds_info.reference_npz,
                        shared_tmpdir=config.dataset.shared_tmpdir,
                        condition_type=ds_info.condition_type,
                        null_label=config.misc.num_classes,
                        text_encoder=text_encoder if ds_info.condition_type == "text" else None,
                        metrics_to_compute=ds_info.metrics,
                        data_dir=ds_info.data_dir,
                        decode_rae=decode_rae,
                    )
                    if eval_stats is not None and rank == 0:
                        save_eval_to_csv(experiment_name, mod_name, global_step, {'dataset': ds_name, **eval_stats}, eval_dir)
                        if args.wandb:
                            wandb_utils.log({f"eval_{mod_name}/{k}_{ds_name}": v for k, v in eval_stats.items()}, step=global_step)
            model.train() # set model back to train mode
            logger.info("Evaluation done.")


    #########################################################
    # Epoch summary
    #########################################################
    if rank == 0 and num_batches > 0:
        avg_loss = epoch_metrics['loss'].item() / num_batches
        epoch_stats = {"epoch/loss": avg_loss}
        if 'loss_patch' in epoch_metrics:
            epoch_stats["epoch/loss_patch"] = epoch_metrics['loss_patch'].item() / num_batches
        if 'loss_aux' in epoch_metrics:
            epoch_stats["epoch/loss_aux"] = epoch_metrics['loss_aux'].item() / num_batches
        logger.info(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
        if args.wandb:
            wandb_utils.log(epoch_stats, step=global_step)

    return global_step
