from __future__ import annotations

import os

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.atlas.hashing import file_sha1
from ada.interpretability.phrase_bank import PhraseBankConfig, build_phrase_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a text phrase bank for ADA region language cards.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--imagenet-meta", type=Path)
    parser.add_argument("--extra-phrases-csv", type=Path)
    parser.add_argument("--confusion-prompts-csv", type=Path)
    parser.add_argument("--no-general-phrases", dest="include_general_phrases", action="store_false")
    parser.add_argument("--no-class-prompts", dest="include_class_prompts", action="store_false")
    parser.add_argument("--no-class-attribute-prompts", dest="include_class_attribute_prompts", action="store_false")
    parser.set_defaults(include_general_phrases=None, include_class_prompts=None, include_class_attribute_prompts=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = _build_config(parse_args())
    metadata = build_phrase_bank(config)
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), **metadata["outputs"]}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> PhraseBankConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = PhraseBankConfig(
        train_cache=Path(raw.get("train_cache", "")),
        output_dir=Path(raw.get("phrase_bank_dir", raw.get("output_dir", ""))),
        imagenet_meta=_optional_path(raw.get("imagenet_meta", os.path.join(os.environ.get("PEAL_DATA", "datasets"), "imagenet_torchvision/data/meta.bin"))),
        extra_phrases_csv=_optional_path(raw.get("extra_phrases_csv")),
        confusion_prompts_csv=_optional_path(raw.get("confusion_prompts_csv")),
        include_general_phrases=bool(raw.get("include_general_phrases", True)),
        include_class_prompts=bool(raw.get("include_class_prompts", True)),
        include_class_attribute_prompts=bool(raw.get("include_class_attribute_prompts", True)),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "train_cache",
        "output_dir",
        "imagenet_meta",
        "extra_phrases_csv",
        "confusion_prompts_csv",
        "include_general_phrases",
        "include_class_prompts",
        "include_class_attribute_prompts",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    cfg = replace(cfg, **overrides)
    if not str(cfg.train_cache):
        raise ValueError("train_cache is required")
    if not str(cfg.output_dir):
        raise ValueError("output_dir is required")
    return cfg


def _optional_path(value: object) -> Path | None:
    if value in ("", None):
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
