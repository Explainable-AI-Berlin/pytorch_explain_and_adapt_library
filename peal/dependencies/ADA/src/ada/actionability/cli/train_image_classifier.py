from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.actionability.train_image_classifier import (
    ImageClassifierConfig,
    train_image_classifier_for_index,
    write_image_probe_index,
)
from ada.atlas.hashing import file_sha1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one image classifier on an actionability exposure manifest.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--deletion-controls-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-image-root", type=Path)
    parser.add_argument("--validation-image-root", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--model-id")
    parser.add_argument("--architecture")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--scratch", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-full-model", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=None)
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.set_defaults(allow_weight_download=None)
    parser.add_argument("--manifest-index", type=int, required=True)
    parser.add_argument("--fixed-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--train-augmentation")
    parser.add_argument("--precision")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    metadata = train_image_classifier_for_index(config, int(args.manifest_index))
    index = write_image_probe_index(config.output_dir)
    print(json.dumps({"metadata_json": str(Path(metadata["predictions_csv"]).parent / "metadata.json"), "probe_index": str(index)}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> ImageClassifierConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = ImageClassifierConfig(
        train_cache=Path(raw.get("train_cache", "")),
        validation_cache=Path(raw.get("validation_cache", "")),
        deletion_controls_dir=Path(raw.get("deletion_controls_dir", "")),
        output_dir=Path(raw.get("probe_output_dir", raw.get("output_dir", ""))),
        train_image_root=_optional_path(raw.get("train_image_root")),
        validation_image_root=_optional_path(raw.get("validation_image_root")),
        experiment_id=str(raw.get("experiment_id", "e8a_in100_image_classifier_causal_screen")),
        model_id=str(raw.get("model_id", "resnet18_scratch_image_classifier")),
        architecture=str(raw.get("architecture", "torchvision_resnet18")),
        pretrained=bool(raw.get("pretrained", False)),
        freeze_backbone=bool(raw.get("freeze_backbone", False)),
        allow_weight_download=bool(raw.get("allow_weight_download", False)),
        fixed_steps=int(raw.get("fixed_steps", 2000)),
        batch_size=int(raw.get("batch_size", 128)),
        eval_batch_size=int(raw.get("eval_batch_size", raw.get("batch_size", 256))),
        lr=float(raw.get("lr", 3.0e-4)),
        weight_decay=float(raw.get("weight_decay", 1.0e-4)),
        image_size=int(raw.get("image_size", 224)),
        train_augmentation=str(raw.get("train_augmentation", "standard")),
        precision=str(raw.get("precision", "fp32")),
        num_workers=int(raw.get("num_workers", 8)),
        device=str(raw.get("device", "auto")),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "train_cache",
        "validation_cache",
        "deletion_controls_dir",
        "output_dir",
        "train_image_root",
        "validation_image_root",
        "experiment_id",
        "model_id",
        "architecture",
        "pretrained",
        "freeze_backbone",
        "allow_weight_download",
        "fixed_steps",
        "batch_size",
        "eval_batch_size",
        "lr",
        "weight_decay",
        "image_size",
        "train_augmentation",
        "precision",
        "num_workers",
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


def _optional_path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


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
