#!/usr/bin/env python3
"""Validate hashes and strict model loading for an exported ADA RAEv2 bundle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--raev2-root", type=Path, required=True)
    parser.add_argument(
        "--verify-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, mmap=True, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str((args.raev2_root / "src").resolve()))
    os.environ.setdefault("ADA_STAGE2_CACHE", "/tmp/unused-stage2-cache")

    from configs.stage2 import Stage2Config
    from stage2.utils import validate_stage2_config
    from utils.model_utils import instantiate_from_config

    manifest = json.loads((args.bundle_root / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["variants"]:
        variant_dir = args.bundle_root / record["variant"]
        checkpoint_path = variant_dir / record["file"]
        config_path = variant_dir / record["config"]
        if args.verify_hashes:
            observed = sha256(checkpoint_path)
            if observed != record["sha256"]:
                raise RuntimeError(
                    f"Hash mismatch for {checkpoint_path}: {observed} != {record['sha256']}"
                )

        config: Stage2Config = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(config_path))
        )
        config.post_process()
        validate_stage2_config(config)
        config.prepare_model_params()
        model = instantiate_from_config(config.stage_2).eval()
        checkpoint = load_checkpoint(checkpoint_path)
        state = checkpoint.get("ema", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{record['variant']} mismatch: missing={missing[:10]}, "
                f"unexpected={unexpected[:10]}"
            )
        print(
            f"[ok] {record['variant']} epoch={checkpoint.get('epoch')} "
            f"step={checkpoint.get('step')} parameters={record['parameter_count']}",
            flush=True,
        )
        del state, checkpoint, model
        gc.collect()

    print("[ok] model bundle hashes and state dictionaries are valid")


if __name__ == "__main__":
    main()
