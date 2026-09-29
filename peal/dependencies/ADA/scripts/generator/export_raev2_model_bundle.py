#!/usr/bin/env python3
"""Export compact EMA-only snapshots for the three ADA RAEv2 generators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch


VARIANTS = {
    "cls_only": {
        "experiment": "in1k_raev2_dinov2l_k1_patch_cls_only_premix64_80ep_4gpu_b1024",
        "config": "imagenet-dinov2l-k1-patch-cls-only-cache-premix64-80ep-4gpu-accum32.yaml",
        "conditioning": "cls",
    },
    "cls_class": {
        "experiment": "in1k_raev2_dinov2l_k1_patch_cls_class_premix64_80ep_4gpu_b1024",
        "config": "imagenet-dinov2l-k1-patch-cls-cache-premix64-80ep-4gpu-accum32.yaml",
        "conditioning": "class+cls",
    },
    "class_only": {
        "experiment": "in1k_raev2_dinov2l_k1_patch_class_only_premix64_80ep_4gpu_b1024",
        "config": "imagenet-dinov2l-k1-patch-class-only-cache-premix64-80ep-4gpu-accum32.yaml",
        "conditioning": "class",
    },
    "cls_ca8": {
        "experiment": "in1k_raev2_dinov2l_k1_patch_cls_only_ca8_every4_80ep_4gpu_b1024",
        "config": "imagenet-dinov2l-k1-patch-cls-only-cache-premix64-80ep-4gpu-ca8-every4.yaml",
        "conditioning": "cls_cross_attention_8",
    },
}

STAGE1_ASSETS = {
    "decoder.pt": {
        "bytes": 1662766063,
        "sha256": "12e40ca9d74b7c45441256d00ada5e6a6b109c1af13f520faf838f13387c861b",
    },
    "stats.pt": {
        "bytes": 2098901,
        "sha256": "b82fe80fae9b27f07e324ecf4222c3aa1a190dba8536343d1a500a718a21b90d",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="cls_only,cls_class,class_only",
        help="Comma-separated variants to export; cls_ca8 is available explicitly.",
    )
    parser.add_argument("--source-repo-commit", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def newest_numbered_checkpoint(checkpoint_dir: Path) -> tuple[int, Path]:
    matches = []
    pattern = re.compile(r"ep-(\d+)\.pt$")
    for path in checkpoint_dir.glob("ep-*.pt"):
        match = pattern.fullmatch(path.name)
        if match:
            matches.append((int(match.group(1)), path))
    if not matches:
        raise FileNotFoundError(f"No numbered checkpoints in {checkpoint_dir}")
    return max(matches, key=lambda item: item[0])


def load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, mmap=True, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def export_variant(
    *,
    variant: str,
    spec: dict[str, str],
    checkpoint_root: Path,
    config_root: Path,
    output_root: Path,
    source_repo_commit: str,
    overwrite: bool,
) -> dict[str, Any]:
    numbered_epoch, source = newest_numbered_checkpoint(
        checkpoint_root / spec["experiment"] / "checkpoints"
    )
    config_source = config_root / spec["config"]
    if not config_source.is_file():
        raise FileNotFoundError(config_source)

    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    output = variant_dir / "ema.pt"
    config_output = variant_dir / "config.yaml"
    metadata_output = variant_dir / "metadata.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    checkpoint = load_checkpoint(source)
    if "ema" not in checkpoint or not isinstance(checkpoint["ema"], dict):
        raise KeyError(f"{source} has no EMA state dictionary")
    state = checkpoint["ema"]
    epoch = int(checkpoint.get("epoch", numbered_epoch))
    step = int(checkpoint.get("step", -1))
    tensor_count = sum(torch.is_tensor(value) for value in state.values())
    parameter_count = sum(value.numel() for value in state.values() if torch.is_tensor(value))
    tensor_bytes = sum(
        value.numel() * value.element_size()
        for value in state.values()
        if torch.is_tensor(value)
    )

    payload = {
        "format": "ada-raev2-stage2-ema-v1",
        "variant": variant,
        "conditioning": spec["conditioning"],
        "epoch": epoch,
        "step": step,
        "ema": state,
    }
    temporary = output.with_suffix(".pt.tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    os.replace(temporary, output)
    shutil.copy2(config_source, config_output)

    output_hash = sha256(output)
    config_hash = sha256(config_output)
    metadata = {
        "format": payload["format"],
        "variant": variant,
        "conditioning": spec["conditioning"],
        "experiment": spec["experiment"],
        "source_checkpoint": source.name,
        "source_numbered_epoch": numbered_epoch,
        "checkpoint_epoch": epoch,
        "checkpoint_step": step,
        "weights": "ema",
        "optimizer_state_included": False,
        "scheduler_state_included": False,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "tensor_bytes": tensor_bytes,
        "file": output.name,
        "bytes": output.stat().st_size,
        "sha256": output_hash,
        "config": config_output.name,
        "config_sha256": config_hash,
        "source_repo_commit": source_repo_commit or None,
    }
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(output, 0o640)
    os.chmod(config_output, 0o640)
    os.chmod(metadata_output, 0o640)
    del checkpoint
    return metadata


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    selected = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(selected).difference(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if not selected:
        raise ValueError("--variants must select at least one model")
    for variant in selected:
        spec = VARIANTS[variant]
        print(f"[export] {variant}", flush=True)
        records.append(
            export_variant(
                variant=variant,
                spec=spec,
                checkpoint_root=args.checkpoint_root,
                config_root=args.config_root,
                output_root=args.output_root,
                source_repo_commit=args.source_repo_commit,
                overwrite=args.overwrite,
            )
        )

    manifest = {
        "format": "ada-raev2-model-bundle-v1",
        "model_family": "RAEv2 DDT over DINOv2-L K=1 patch latents",
        "variants": records,
        "stage1_assets": {
            "repo_id": "nyu-visionx/RAEv2-models",
            "revision": "9770b7b980fa1875c8e6d65f226c615c0ce908a8",
            "artifact_path": "stage1/imagenet/dinov2l-k1",
            "files": STAGE1_ASSETS,
        },
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(manifest_path, 0o640)
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
