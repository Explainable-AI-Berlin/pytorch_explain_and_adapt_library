"""Stage 1 training engine."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.utils import make_grid

from .disc import select_gan_losses, calculate_adaptive_weight
from eval import evaluate_reconstruction_distributed
from utils import wandb_utils
from utils.checkpoint import save_stage1_checkpoint
from utils.logging import save_eval_to_csv
from utils.sync_utils import sync_checkpoint_async, sync_evals_async
from utils.train_utils import update_ema


def auxiliary_bank_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str = "mse",
    temperature: float = 0.07,
) -> torch.Tensor:
    loss_type = str(loss_type).lower()
    pred_f = pred.float()
    target_f = target.float()

    if loss_type == "mse":
        return F.mse_loss(pred_f, target_f)

    if loss_type == "cosine":
        pred_flat = F.normalize(pred_f.flatten(1), dim=1, eps=1e-12)
        target_flat = F.normalize(target_f.flatten(1), dim=1, eps=1e-12)
        return (1.0 - (pred_flat * target_flat).sum(dim=1)).mean()

    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 for contrastive aux losses, got {temperature}")

    def _gather_global_targets(x: torch.Tensor):
        if not dist.is_available() or not dist.is_initialized():
            return x, 0
        world_size = dist.get_world_size()
        if world_size <= 1:
            return x, 0
        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x.contiguous())
        return torch.cat(gathered, dim=0), dist.get_rank() * x.shape[0]

    if loss_type == "infonce_bank":
        pred_flat = F.normalize(pred_f.flatten(1), dim=1, eps=1e-12)
        target_flat = F.normalize(target_f.flatten(1), dim=1, eps=1e-12)
        target_global, rank_offset = _gather_global_targets(target_flat)
        logits = pred_flat @ target_global.T / temperature
        labels = torch.arange(pred_f.size(0), device=pred_f.device) + rank_offset
        return F.cross_entropy(logits, labels)

    if loss_type == "infonce_token":
        if pred_f.ndim != 3 or target_f.ndim != 3:
            raise ValueError(
                "infonce_token expects token banks with shape [B, K, C], "
                f"got pred={tuple(pred_f.shape)} target={tuple(target_f.shape)}"
            )
        if pred_f.shape != target_f.shape:
            raise ValueError(
                "infonce_token requires pred and target to have identical shapes, "
                f"got pred={tuple(pred_f.shape)} target={tuple(target_f.shape)}"
            )
        batch_size, num_tokens, _ = pred_f.shape
        pred_norm = F.normalize(pred_f, dim=-1, eps=1e-12)
        target_norm = F.normalize(target_f, dim=-1, eps=1e-12)
        target_global, rank_offset = _gather_global_targets(target_norm)
        logits = torch.einsum("bkc,tkc->kbt", pred_norm, target_global) / temperature
        labels = (torch.arange(batch_size, device=pred_f.device) + rank_offset).repeat(num_tokens)
        logits_fwd = logits.reshape(num_tokens * batch_size, target_global.shape[0])
        return F.cross_entropy(logits_fwd, labels)

    raise ValueError(
        f"Unsupported auxiliary prediction loss_type={loss_type!r}; "
        "expected one of ['mse', 'cosine', 'infonce_bank', 'infonce_token']."
    )


def vicreg_token_diversity_loss(
    tokens: torch.Tensor,
    *,
    variance_weight: float = 1.0,
    covariance_weight: float = 1.0,
    variance_target_std: float = 0.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(
            "vicreg_token_diversity_loss expects tokens with shape [B, K, C], "
            f"got {tuple(tokens.shape)}"
        )

    x = tokens.float()
    _, num_tokens, _ = x.shape
    total = torch.zeros((), device=x.device, dtype=x.dtype)

    if variance_weight > 0.0:
        slot_centered = x - x.mean(dim=0, keepdim=True)
        slot_std = torch.sqrt(slot_centered.var(dim=0, unbiased=False) + eps)
        variance_loss = F.relu(variance_target_std - slot_std).mean()
        total = total + variance_weight * variance_loss

    if covariance_weight > 0.0 and num_tokens > 1:
        slot_norm = F.normalize(x, dim=-1, eps=1e-12)
        slot_similarity = torch.matmul(slot_norm, slot_norm.transpose(1, 2))
        off_diag = slot_similarity - torch.diag_embed(torch.diagonal(slot_similarity, dim1=1, dim2=2))
        covariance_loss = off_diag.pow(2).sum(dim=(1, 2)) / float(num_tokens * (num_tokens - 1))
        covariance_loss = covariance_loss.mean()
        total = total + covariance_weight * covariance_loss

    return total


def train_one_epoch(
    ddp_model,
    ema_model,
    ddp_disc,
    disc_aug,
    lpips_model,
    dataloader,
    optimizer,
    disc_optimizer,
    scheduler,
    disc_scheduler,
    autocast_kwargs: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    micro_batch_size: int,
    config,
    args,
    logger,
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    experiment_dir: str,
    progress_bar,
    eval_datasets: Optional[Dict] = None,
    viz_samples: Optional[torch.Tensor] = None,
) -> int:
    """Train one epoch. Returns updated global_step.

    Args:
        eval_datasets: Dict of {name: EvalDatasetInfo} for unified eval, or None to skip eval.
    """
    #########################################################
    # Setup
    #########################################################
    ddp_model.train()

    disc = ddp_disc.module if ddp_disc is not None else None
    decoder = ddp_model.module.decoder
    last_layer = decoder.decoder_pred.weight

    grad_accum_steps = config.training.grad_accum_steps
    steps_per_epoch = (
        config.training.virtual_epoch_steps
        if config.training.virtual_epoch_steps
        else len(dataloader) // grad_accum_steps
    )
    disc_loss_fn, gen_loss_fn = select_gan_losses(config.gan.loss.disc_loss, config.gan.loss.gen_loss)
    aux_slot_pred_weight = float(getattr(config.gan.loss, "aux_slot_pred_weight", 0.0))
    aux_slot_pred_normalize_target = bool(getattr(config.gan.loss, "aux_slot_pred_normalize_target", True))
    aux_slot_pred_loss_type = str(getattr(config.gan.loss, "aux_slot_pred_loss_type", "mse"))
    aux_slot_pred_loss_temperature = float(getattr(config.gan.loss, "aux_slot_pred_loss_temperature", 0.07))
    aux_slot_vicreg_weight = float(getattr(config.gan.loss, "aux_slot_vicreg_weight", 0.0))
    aux_slot_vicreg_var_weight = float(getattr(config.gan.loss, "aux_slot_vicreg_var_weight", 1.0))
    aux_slot_vicreg_cov_weight = float(getattr(config.gan.loss, "aux_slot_vicreg_cov_weight", 1.0))
    aux_slot_vicreg_target_std = float(getattr(config.gan.loss, "aux_slot_vicreg_target_std", 0.5))
    if aux_slot_vicreg_weight > 0.0 and aux_slot_pred_weight <= 0.0:
        raise ValueError("aux_slot_vicreg_weight requires aux_slot_pred_weight > 0.0.")

    gan_start_step = config.gan.loss.disc_start * steps_per_epoch
    disc_update_step = config.gan.loss.disc_upd_start * steps_per_epoch
    lpips_start_step = config.gan.loss.lpips_start * steps_per_epoch

    do_eval = config.eval is not None and eval_datasets is not None
    epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
    num_batches = 0

    # Checkpoint at epoch start
    if config.training.checkpoint_interval > 0 and epoch % config.training.checkpoint_interval == 0 and rank == 0:
        logger.info(f"Saving checkpoint at epoch {epoch}...")
        ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
        save_stage1_checkpoint(
            ckpt_path, global_step, epoch, ddp_model, ema_model,
            optimizer, scheduler, disc, disc_optimizer, disc_scheduler,
        )
        if args.sync_checkpoints:
            sync_checkpoint_async(checkpoint_dir, logger)
            sync_evals_async("evals/stage1", logger)

    #########################################################
    # Train loop
    #########################################################
    optimizer.zero_grad(set_to_none=True)
    if disc_optimizer is not None:
        disc_optimizer.zero_grad(set_to_none=True)

    accum_images = []
    accum_recon = 0.0
    accum_lpips = 0.0
    accum_gan = 0.0
    accum_total = 0.0
    accum_adaptive_weight = 0.0
    accum_aux_slot_pred = 0.0
    accum_aux_slot_vicreg = 0.0

    max_micro_steps = steps_per_epoch * grad_accum_steps
    for step, batch in enumerate(dataloader):
        if step >= max_micro_steps:
            break
        images = batch[0]
        use_gan = disc is not None and global_step >= gan_start_step and config.gan.loss.disc_weight > 0.0
        train_disc = ddp_disc is not None and global_step >= disc_update_step and config.gan.loss.disc_weight > 0.0
        use_lpips = global_step >= lpips_start_step and config.gan.loss.perceptual_weight > 0.0
        is_accum_step = (step + 1) % grad_accum_steps != 0

        images = images.to(device, non_blocking=True)
        real_normed = images * 2.0 - 1.0
        accum_images.append(images.detach())

        #########################################################
        # Train generator
        #########################################################
        if disc is not None:
            disc.eval()

        with autocast(**autocast_kwargs):
            aux_slot_pred_loss = torch.zeros((), device=device)
            aux_slot_vicreg_loss = torch.zeros((), device=device)
            if aux_slot_pred_weight > 0.0:
                recon, aux_slot_pred, aux_slot_target, aux_slot_states = ddp_model(
                    images,
                    return_aux_slot_pred=True,
                    normalize_aux_slot_pred_target=aux_slot_pred_normalize_target,
                )
                aux_slot_pred_loss = auxiliary_bank_prediction_loss(
                    aux_slot_pred,
                    aux_slot_target,
                    loss_type=aux_slot_pred_loss_type,
                    temperature=aux_slot_pred_loss_temperature,
                )
                if aux_slot_vicreg_weight > 0.0:
                    aux_slot_vicreg_loss = vicreg_token_diversity_loss(
                        aux_slot_states,
                        variance_weight=aux_slot_vicreg_var_weight,
                        covariance_weight=aux_slot_vicreg_cov_weight,
                        variance_target_std=aux_slot_vicreg_target_std,
                    )
            else:
                recon = ddp_model(images)
            recon_normed = recon * 2.0 - 1.0
            rec_loss = (recon - images).abs().mean()
            lpips_loss = lpips_model(real_normed, recon_normed) if use_lpips else rec_loss.new_zeros(())
            recon_total = (
                rec_loss
                + config.gan.loss.perceptual_weight * lpips_loss
                + aux_slot_pred_weight * aux_slot_pred_loss
                + aux_slot_vicreg_weight * aux_slot_vicreg_loss
            )

            if use_gan:
                fake_aug = disc_aug.aug(recon_normed)
                logits_fake, _ = ddp_disc(fake_aug, None)
                gan_loss = gen_loss_fn(logits_fake)
            else:
                gan_loss = torch.zeros_like(recon_total)

        if use_gan:
            adaptive_weight = calculate_adaptive_weight(recon_total, gan_loss, last_layer, config.gan.loss.max_d_weight)
            total_loss = recon_total + config.gan.loss.disc_weight * adaptive_weight * gan_loss
        else:
            adaptive_weight = torch.zeros_like(recon_total)
            total_loss = recon_total

        scaled_total_loss = total_loss / grad_accum_steps
        if is_accum_step:
            with ddp_model.no_sync():
                scaled_total_loss.backward()
        else:
            scaled_total_loss.backward()

        accum_recon += rec_loss.detach().item()
        accum_lpips += lpips_loss.detach().item()
        accum_gan += gan_loss.detach().item()
        accum_total += total_loss.detach().item()
        accum_adaptive_weight += adaptive_weight.detach().item()
        accum_aux_slot_pred += aux_slot_pred_loss.detach().item()
        accum_aux_slot_vicreg += aux_slot_vicreg_loss.detach().item()

        if is_accum_step:
            continue

        if config.training.clip_grad:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), config.training.clip_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()
        update_ema(ema_model, ddp_model.module, config.training.ema_decay)

        #########################################################
        # Train discriminator
        #########################################################
        disc_metrics: Dict[str, torch.Tensor] = {}
        if train_disc:
            ddp_model.eval()
            ddp_disc.train()
            for _ in range(config.gan.loss.disc_updates):
                disc_optimizer.zero_grad(set_to_none=True)
                disc_loss_total = 0.0
                disc_acc_total = 0.0
                logits_real_mean = 0.0
                logits_fake_mean = 0.0
                for micro_idx, micro_images in enumerate(accum_images):
                    real_micro = micro_images * 2.0 - 1.0
                    with autocast(**autocast_kwargs):
                        with torch.no_grad():
                            recon_disc = ddp_model(micro_images)
                            recon_disc_normed = recon_disc * 2.0 - 1.0
                        fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                        fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_micro)
                        logits_fake, logits_real = ddp_disc(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)
                        accuracy = (logits_real > logits_fake).float().mean()

                    scaled_d_loss = d_loss / grad_accum_steps
                    is_disc_accum_step = micro_idx < len(accum_images) - 1
                    if is_disc_accum_step:
                        with ddp_disc.no_sync():
                            scaled_d_loss.backward()
                    else:
                        scaled_d_loss.backward()

                    disc_loss_total += d_loss.detach().item()
                    disc_acc_total += accuracy.detach().item()
                    logits_real_mean += logits_real.detach().mean().item()
                    logits_fake_mean += logits_fake.detach().mean().item()

                disc_optimizer.step()

                disc_metrics = {
                    "disc_loss": torch.tensor(disc_loss_total / len(accum_images), device=device),
                    "logits_real": torch.tensor(logits_real_mean / len(accum_images), device=device),
                    "logits_fake": torch.tensor(logits_fake_mean / len(accum_images), device=device),
                    "disc_accuracy": torch.tensor(disc_acc_total / len(accum_images), device=device),
                }
                epoch_metrics["disc_loss"] += disc_metrics["disc_loss"]
                epoch_metrics["disc_accuracy"] += disc_metrics["disc_accuracy"]
                if disc_scheduler is not None:
                    disc_scheduler.step()

            ddp_disc.eval()
            ddp_model.train()

        epoch_metrics["recon"] += torch.tensor(accum_recon / grad_accum_steps, device=device)
        epoch_metrics["lpips"] += torch.tensor(accum_lpips / grad_accum_steps, device=device)
        epoch_metrics["gan"] += torch.tensor(accum_gan / grad_accum_steps, device=device)
        epoch_metrics["total"] += torch.tensor(accum_total / grad_accum_steps, device=device)
        epoch_metrics["aux_slot_pred"] += torch.tensor(accum_aux_slot_pred / grad_accum_steps, device=device)
        epoch_metrics["aux_slot_vicreg"] += torch.tensor(accum_aux_slot_vicreg / grad_accum_steps, device=device)
        num_batches += 1
        progress_bar.update(1)
        global_step += 1

        #########################################################
        # Logging and visualization
        #########################################################
        if config.training.log_interval > 0 and global_step % config.training.log_interval == 0 and rank == 0:
            mean_recon = accum_recon / grad_accum_steps
            mean_lpips = accum_lpips / grad_accum_steps
            mean_gan = accum_gan / grad_accum_steps
            mean_total = accum_total / grad_accum_steps
            mean_adaptive_weight = accum_adaptive_weight / grad_accum_steps
            mean_aux_slot_pred = accum_aux_slot_pred / grad_accum_steps
            mean_aux_slot_vicreg = accum_aux_slot_vicreg / grad_accum_steps
            stats = {
                "loss/total": mean_total,
                "loss/recon": mean_recon,
                "loss/lpips": mean_lpips,
                "loss/gan": mean_gan,
                "lr/generator": optimizer.param_groups[0]["lr"],
            }
            if aux_slot_pred_weight > 0.0:
                stats["loss/aux_slot_pred"] = mean_aux_slot_pred
            if aux_slot_vicreg_weight > 0.0:
                stats["loss/aux_slot_vicreg"] = mean_aux_slot_vicreg
            if disc_metrics:
                stats.update({
                    "loss/disc": disc_metrics["disc_loss"].item(),
                    "disc/logits_real": disc_metrics["logits_real"].item(),
                    "disc/logits_fake": disc_metrics["logits_fake"].item(),
                    "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                    "disc/accuracy": disc_metrics["disc_accuracy"].item(),
                    "disc/weight": mean_adaptive_weight,
                })
            logger.info(f"[Epoch {epoch} | Step {global_step}] " + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
            if args.wandb:
                wandb_utils.log(stats, step=global_step)
            progress_bar.set_postfix(loss=mean_total, lr=optimizer.param_groups[0]["lr"])

        # Visualization
        if global_step % config.training.sample_every == 0 and do_eval and viz_samples is not None:
            logger.info("Generating EMA samples...")
            with torch.no_grad():
                samples = ema_model(viz_samples)
                original_grid = make_grid(viz_samples.cpu().float(), nrow=8)
                recon_grid = make_grid(samples.cpu().float(), nrow=8)
                if args.wandb:
                    wandb_utils.log_images({"viz/original": original_grid, "viz/reconstructed": recon_grid}, step=global_step)
            logger.info("Generating EMA samples done.")

        #########################################################
        # Evaluation (unified: iterate over all eval datasets)
        #########################################################
        if do_eval and config.eval.eval_interval > 0 and global_step > 0 and global_step % config.eval.eval_interval == 0:
            logger.info("Starting evaluation...")
            eval_models = [(ema_model, "ema")]
            if config.eval.eval_model:
                eval_models.append((ddp_model.module, "model"))
            experiment_name = os.environ.get("EXPERIMENT_NAME", "unknown")

            for ds_name, ds_info in eval_datasets.items():
                eval_dataset = ds_info.require_indexable_dataset(ds_name)
                eval_n = min(ds_info.num_samples or len(ds_info), len(ds_info))
                for eval_mod, mod_name in eval_models:
                    eval_stats = evaluate_reconstruction_distributed(
                        eval_mod, eval_dataset, eval_n,
                        rank=rank, world_size=world_size, device=device, batch_size=micro_batch_size,
                        metrics_to_compute=ds_info.metrics, experiment_dir=experiment_dir,
                        global_step=global_step, autocast_kwargs=autocast_kwargs,
                        reference_npz_path=ds_info.reference_npz, shared_tmpdir=config.dataset.shared_tmpdir,
                    )
                    if rank == 0 and eval_stats is not None:
                        save_eval_to_csv(experiment_name, f"{mod_name}_{ds_name}", global_step, eval_stats)
                    if eval_stats:
                        wandb_stats = {f"eval_{mod_name}/{k}_{ds_name}": v for k, v in eval_stats.items()}
                        if args.wandb:
                            wandb_utils.log(wandb_stats, step=global_step)
            logger.info("Evaluation done.")

        accum_images.clear()
        accum_recon = 0.0
        accum_lpips = 0.0
        accum_gan = 0.0
        accum_total = 0.0
        accum_adaptive_weight = 0.0
        accum_aux_slot_pred = 0.0
        accum_aux_slot_vicreg = 0.0

    #########################################################
    # Epoch summary
    #########################################################
    if rank == 0 and num_batches > 0:
        epoch_stats = {
            "epoch/loss_total": (epoch_metrics["total"] / num_batches).item(),
            "epoch/loss_recon": (epoch_metrics["recon"] / num_batches).item(),
            "epoch/loss_lpips": (epoch_metrics["lpips"] / num_batches).item(),
            "epoch/loss_gan": (epoch_metrics["gan"] / num_batches).item(),
            "epoch/loss_aux_slot_pred": (epoch_metrics["aux_slot_pred"] / num_batches).item(),
            "epoch/loss_aux_slot_vicreg": (epoch_metrics["aux_slot_vicreg"] / num_batches).item(),
        }
        if disc_metrics:
            epoch_stats.update({
                "epoch/loss_disc": (epoch_metrics["disc_loss"] / num_batches).item(),
                "epoch/disc_logits_real": disc_metrics["logits_real"].item(),
                "epoch/disc_logits_fake": disc_metrics["logits_fake"].item(),
                "epoch/disc_accuracy": (epoch_metrics["disc_accuracy"] / num_batches).item(),
            })
        logger.info(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
        if args.wandb:
            wandb_utils.log(epoch_stats, step=global_step)

    return global_step
