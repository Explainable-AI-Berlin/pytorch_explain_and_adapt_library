"""Stage 2 shared utilities.

Contains config validation + helpers shared between stage2/engine.py
"""

from __future__ import annotations

import dataclasses
from copy import deepcopy

import torch
from torch.cuda.amp import autocast

from configs.stage2 import Stage2Config
from stage2.models.embedders import TextEncoder
from stage2.state_utils import (
    decode_stage2_state,
    duplicate_state_for_guidance,
    final_state_from_trajectory,
    split_guided_state,
    state_batch_size,
)
from utils.dist_utils import main_process_first


def validate_stage2_config(config: Stage2Config) -> None:
    """Validate a Stage2Config for consistency."""
    if not config.stage_1.target:
        raise ValueError("Config must provide stage_1.target (RAE model).")
    if config.decode_stage_1 is not None and not config.decode_stage_1.target:
        raise ValueError("decode_stage_1.target must be set when decode_stage_1 is provided.")
    if not config.stage_2.target:
        raise ValueError("Config must provide stage_2.target (DiT model).")

    # REPA validation
    repa = config.repa
    if repa.use_repa:
        if not repa.target_encoder:
            raise ValueError("repa.target_encoder is required when use_repa=True.")

    # Gradient accumulation
    if config.training.grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps must be >= 1.")
    if config.joint_state_loss.aux_loss_weight < 0:
        raise ValueError("joint_state_loss.aux_loss_weight must be >= 0.")

    # Conditioning
    cond = config.conditioning
    if cond.type == "text" and cond.text_encoder is None:
        raise ValueError("conditioning.text_encoder must be set when conditioning.type='text'.")
    if cond.type not in {"label", "label_meta", "text", "nwm", "cls"}:
        raise ValueError(f"Unsupported conditioning.type={cond.type!r}")


##############################################################
# Shared helpers used by both stage2/engine
##############################################################
def apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob=0.1):
    if isinstance(model_conds['context'], dict):
        from stage2 import nwm_cond
        return nwm_cond.apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob)
    mask = torch.rand(model_conds['context'].shape[0], device=model_conds['context'].device) < cfg_dropout_prob
    return {
        k: torch.where(mask.view(-1, *([1]*(v.ndim-1))), model_conds_null[k], v) if v is not None else None
        for k, v in model_conds.items()
    }, mask


def get_null_cond(text_encoder, conditioning_type, num_classes, batch_size, device, meta_num_fields: int = 0):
    if conditioning_type == "text":
        _null_context, _null_attn_mask = encode_text(text_encoder, [""])
        rtn = dict(context=_null_context, attn_mask=_null_attn_mask)
    elif conditioning_type == "label_meta":
        _null_context, _null_attn_mask = torch.tensor([num_classes], device=device), None
        rtn = dict(
            context=_null_context,
            attn_mask=_null_attn_mask,
            meta=torch.zeros(1, max(int(meta_num_fields), 1), device=device, dtype=torch.long),
        )
    else:
        _null_context, _null_attn_mask = torch.tensor([num_classes], device=device), None
        rtn = dict(context=_null_context, attn_mask=_null_attn_mask)
    rtn = {k: v.expand(batch_size, *v.shape[1:]) if v is not None else None for k, v in rtn.items()}
    return rtn


def setup_text_encoder(config, rank, device):
    """Build text encoder if conditioning.type == 'text', else return None.

    Side effect: sets config.conditioning.context_dim from the encoder's feature_dim.
    """
    if config.conditioning.type != "text":
        return None
    with main_process_first(rank):
        text_encoder = TextEncoder(**dataclasses.asdict(config.conditioning.text_encoder)).to(device)
    config.conditioning.context_dim = text_encoder.feature_dim
    return text_encoder


def encode_text(text_encoder, y):
    """Encode text conditions. Returns (encoder_hidden_states, encoder_attention_mask)."""
    with torch.no_grad():
        enc_out = text_encoder(y)
        return enc_out["tokens"], enc_out["attention_mask"]


def get_fixed_viz_batch_conditions(viz_fixed, y, condition_type, text_encoder, device):
    """Get fixed conditions for the first batch for consistent visualization."""
    if viz_fixed['context'] is not None:
        return viz_fixed
    n = state_batch_size(viz_fixed['zs'])
    if condition_type in {"label", "cls"}:
        viz_fixed['context'] = y[:n].clone().to(device)
    elif condition_type == "label_meta":
        labels, meta = y
        viz_fixed['context'] = labels[:n].clone().to(device)
        viz_fixed['meta'] = meta[:n].clone().to(device)
    else:
        with torch.no_grad():
            enc_out = text_encoder(y[:n])
            viz_fixed['context'] = enc_out["tokens"]
            viz_fixed['attn_mask'] = enc_out["attention_mask"]
    return viz_fixed


def sample_and_decode(
    zs, context, attn_mask,
    eval_sampler, model_fn, sample_model_kwargs, rae,
    use_guidance, condition_type, text_encoder, num_classes, device, autocast_kwargs,
    decode_rae=None,
    meta=None,
    cls=None,
    cls_null=None,
):
    """Generate and decode samples, handling guidance doubling."""
    n = state_batch_size(zs)
    if use_guidance:
        zs = duplicate_state_for_guidance(zs)
        if isinstance(context, dict):
            null = {k: torch.zeros_like(v) for k, v in context.items()}
            context = {k: torch.cat([context[k], null[k]], dim=0) for k in context}
            attn_mask = None
        else:
            if condition_type == "text":
                context_null, attn_mask_null = encode_text(text_encoder, [""] * n)
                meta_null = None
            elif condition_type == "cls":
                context_null = (
                    cls_null.expand(n, -1).to(device=device, dtype=context.dtype)
                    if cls_null is not None
                    else torch.zeros_like(context)
                )
                attn_mask_null = None
                meta_null = None
            else:
                context_null = torch.full((n,), num_classes, device=device)
                attn_mask_null = None
                meta_null = torch.zeros_like(meta) if meta is not None else None
            context = torch.cat([context, context_null], dim=0)
            if attn_mask is not None and attn_mask_null is not None:
                attn_mask = torch.cat([attn_mask, attn_mask_null], dim=0)
            if meta is not None and meta_null is not None:
                meta = torch.cat([meta, meta_null], dim=0)
        if cls is not None:
            cls_uncond = (
                cls_null.expand(n, -1).to(device=device, dtype=cls.dtype)
                if cls_null is not None
                else torch.zeros_like(cls)
            )
            cls = torch.cat([cls, cls_uncond], dim=0)

    kwargs = deepcopy(sample_model_kwargs)
    kwargs.update(context=context, attn_mask=attn_mask)
    if meta is not None:
        kwargs["meta"] = meta
    if cls is not None:
        kwargs["cls"] = cls
    with autocast(**autocast_kwargs):
        samples = final_state_from_trajectory(eval_sampler(zs, model_fn, **kwargs))
        if use_guidance:
            samples = split_guided_state(samples)
    decode_target = decode_rae if decode_rae is not None else rae
    return decode_stage2_state(decode_target, samples).cpu().float()
