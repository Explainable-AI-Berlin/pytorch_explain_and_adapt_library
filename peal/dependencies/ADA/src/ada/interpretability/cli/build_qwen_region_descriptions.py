from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.atlas.hashing import file_sha1
from ada.interpretability.qwen_region_descriptions import QwenRegionDescriptionConfig, build_qwen_region_descriptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run Qwen3-VL contrastive region descriptions.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--region-cards-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-image-root", type=Path)
    parser.add_argument("--validation-image-root", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--max-regions", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--images-per-set", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    parser.add_argument("--run-inference", action="store_true", default=None)
    parser.add_argument("--include-swapped-order", action="store_true", default=None)
    parser.add_argument("--include-null-control", action="store_true", default=None)
    parser.add_argument("--deterministic-repeats", type=int)
    parser.add_argument("--prompt-variant")
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    config = _build_config(parse_args())
    metadata = build_qwen_region_descriptions(config)
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), **metadata["outputs"]}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> QwenRegionDescriptionConfig:
    raw = _load_config(args.config) if args.config else {}
    cards_dir = Path(raw.get("region_cards_dir", ""))
    cfg = QwenRegionDescriptionConfig(
        manifest_dir=Path(raw.get("interpretability_manifest_dir", raw.get("manifest_dir", ""))),
        region_cards_csv=Path(raw.get("region_cards_csv", cards_dir / "region_language_cards.csv")),
        output_dir=Path(raw.get("qwen_output_dir", raw.get("output_dir", ""))),
        train_image_root=Path(raw.get("train_image_root", "")),
        validation_image_root=Path(raw.get("validation_image_root", "")),
        model_name=str(raw.get("model_name", "Qwen/Qwen3-VL-4B-Instruct")),
        model_path=_optional_path(raw.get("model_path")),
        model_revision=str(raw.get("model_revision", "")),
        max_regions=_optional_int(raw.get("max_regions")),
        max_tasks=_optional_int(raw.get("max_tasks")),
        images_per_set=int(raw.get("images_per_set", 8)),
        max_new_tokens=int(raw.get("max_new_tokens", 768)),
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        device=str(raw.get("device", "auto")),
        dtype=str(raw.get("dtype", "auto")),
        run_inference=bool(raw.get("run_inference", False)),
        include_swapped_order=bool(raw.get("include_swapped_order", False)),
        include_null_control=bool(raw.get("include_null_control", False)),
        deterministic_repeats=int(raw.get("deterministic_repeats", 1)),
        prompt_variant=str(raw.get("prompt_variant", "strict")),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "manifest_dir",
        "region_cards_csv",
        "output_dir",
        "train_image_root",
        "validation_image_root",
        "model_name",
        "model_path",
        "model_revision",
        "max_regions",
        "max_tasks",
        "images_per_set",
        "max_new_tokens",
        "temperature",
        "top_p",
        "device",
        "dtype",
        "run_inference",
        "include_swapped_order",
        "include_null_control",
        "deterministic_repeats",
        "prompt_variant",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    cfg = replace(cfg, **overrides)
    for key in ("manifest_dir", "region_cards_csv", "output_dir", "train_image_root", "validation_image_root"):
        if not str(getattr(cfg, key)):
            raise ValueError(f"{key} is required")
    return cfg


def _optional_path(value: object) -> Path | None:
    if value in ("", None):
        return None
    return Path(str(value))


def _optional_int(value: object) -> int | None:
    if value in ("", None):
        return None
    return int(value)


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
