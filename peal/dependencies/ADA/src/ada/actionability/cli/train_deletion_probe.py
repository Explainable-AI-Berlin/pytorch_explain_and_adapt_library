from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.actionability.train_deletion_probe import (
    DeletionProbeConfig,
    train_deletion_probe_for_index,
    write_probe_index,
)
from ada.atlas.hashing import file_sha1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one DINO CLS linear probe for a deletion-control exposure manifest.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--deletion-controls-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--model-id")
    parser.add_argument("--feature-id")
    parser.add_argument("--manifest-index", type=int, required=True)
    parser.add_argument("--fixed-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    metadata = train_deletion_probe_for_index(config, int(args.manifest_index))
    index = write_probe_index(config.output_dir)
    print(json.dumps({"metadata_json": str(Path(metadata["predictions_csv"]).parent / "metadata.json"), "probe_index": str(index)}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> DeletionProbeConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = DeletionProbeConfig(
        train_cache=Path(raw.get("train_cache", "")),
        validation_cache=Path(raw.get("validation_cache", "")),
        deletion_controls_dir=Path(raw.get("deletion_controls_dir", "")),
        output_dir=Path(raw.get("probe_output_dir", raw.get("output_dir", ""))),
        experiment_id=str(raw.get("experiment_id", "e4a_in100_k10_deletion_probe")),
        model_id=str(raw.get("model_id", "dino_cls_deletion_probe")),
        feature_id=str(raw.get("feature_id", "")),
        fixed_steps=int(raw.get("fixed_steps", 1000)),
        batch_size=int(raw.get("batch_size", 2048)),
        lr=float(raw.get("lr", 1.0e-2)),
        weight_decay=float(raw.get("weight_decay", 1.0e-4)),
        device=str(raw.get("device", "auto")),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "train_cache",
        "validation_cache",
        "deletion_controls_dir",
        "output_dir",
        "experiment_id",
        "model_id",
        "feature_id",
        "fixed_steps",
        "batch_size",
        "lr",
        "weight_decay",
        "device",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    cfg = replace(cfg, **overrides)
    if not str(cfg.train_cache):
        raise ValueError("train_cache is required")
    if not str(cfg.validation_cache):
        raise ValueError("validation_cache is required")
    if not str(cfg.deletion_controls_dir):
        raise ValueError("deletion_controls_dir is required")
    if not str(cfg.output_dir):
        raise ValueError("output_dir is required")
    return cfg


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = Path(path).read_text()
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    data["config_path"] = str(path)
    data["config_file_sha1"] = file_sha1(path)
    return data


if __name__ == "__main__":
    main()
