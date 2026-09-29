"""Checkpoint save/load utilities for Stage 1 and Stage 2 training."""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


def _checkpoint_epoch_key(path: str) -> int:
    """Best-effort epoch key for sorting checkpoint fallbacks."""
    basename = os.path.basename(path)
    if basename == "ep-last.pt":
        return 10**12
    match = re.search(r"ep-(\d+)", basename)
    if match:
        return int(match.group(1))
    return -1


def _fallback_candidates(path: str) -> list[str]:
    """Return resume candidates in descending priority, starting with ``path``."""
    checkpoint_dir = os.path.dirname(path)
    candidates = [path]
    if not os.path.isdir(checkpoint_dir):
        return candidates

    others = []
    for name in os.listdir(checkpoint_dir):
        if not name.endswith(".pt"):
            continue
        candidate = os.path.join(checkpoint_dir, name)
        if os.path.abspath(candidate) == os.path.abspath(path):
            continue
        others.append(candidate)

    others.sort(key=lambda candidate: (_checkpoint_epoch_key(candidate), candidate), reverse=True)
    return candidates + others


def _load_checkpoint_with_fallback(path: str) -> dict:
    """Load checkpoint, falling back to older siblings if the preferred file is corrupted."""
    first_error: Optional[Exception] = None
    for candidate in _fallback_candidates(path):
        try:
            if os.path.abspath(candidate) != os.path.abspath(path):
                print(f"[checkpoint] preferred resume file failed, trying fallback {candidate}")
            return torch.load(candidate, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            if first_error is None:
                first_error = exc
            print(f"[checkpoint] failed loading {candidate}: {exc}")
            continue

    assert first_error is not None
    raise first_error


def _update_last_checkpoint_alias(path: str, alias_name: str = "ep-last.pt") -> None:
    """Refresh a stable alias for the latest checkpoint in the same directory.

    Prefer a hard link to avoid duplicating large checkpoints, and fall back to
    a byte-for-byte copy when linking is not available.
    """
    checkpoint_dir = os.path.dirname(path)
    if not checkpoint_dir:
        return

    alias_path = os.path.join(checkpoint_dir, alias_name)
    if os.path.abspath(path) == os.path.abspath(alias_path):
        return

    temp_alias = f"{alias_path}.tmp"
    if os.path.lexists(temp_alias):
        os.remove(temp_alias)

    try:
        os.link(path, temp_alias)
    except OSError:
        shutil.copy2(path, temp_alias)

    os.replace(temp_alias, alias_path)


def save_stage1_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: Optional[torch.nn.Module],
    disc_optimizer: Optional[torch.optim.Optimizer],
    disc_scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 1 training checkpoint (model + discriminator)."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict() if disc is not None else None,
        "disc_optimizer": disc_optimizer.state_dict() if disc_optimizer is not None else None,
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    _update_last_checkpoint_alias(path)


def load_stage1_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: Optional[torch.nn.Module],
    disc_optimizer: Optional[torch.optim.Optimizer],
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 1 training checkpoint. Returns (epoch, step)."""
    checkpoint = _load_checkpoint_with_fallback(path)
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if disc is not None and checkpoint.get("disc") is not None:
        disc.load_state_dict(checkpoint["disc"])
    if disc_optimizer is not None and checkpoint.get("disc_optimizer") is not None:
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def save_stage2_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 2 training checkpoint."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    _update_last_checkpoint_alias(path)


def load_stage2_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 2 training checkpoint. Returns (epoch, step)."""
    checkpoint = _load_checkpoint_with_fallback(path)
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


__all__ = [
    "save_stage1_checkpoint",
    "load_stage1_checkpoint",
    "save_stage2_checkpoint",
    "load_stage2_checkpoint",
]
