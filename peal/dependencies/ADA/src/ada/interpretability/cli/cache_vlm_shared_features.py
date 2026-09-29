from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.atlas.hashing import file_sha1
from ada.interpretability.vlm_shared_cache import VLMSharedCacheConfig, cache_vlm_shared_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache shared image/text embeddings from a local SigLIP/CLIP model.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--phrase-bank-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interpretability-manifest-dir", type=Path)
    parser.add_argument("--image-cache-mode")
    parser.add_argument("--shared-image-train-cache", type=Path)
    parser.add_argument("--shared-image-validation-cache", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--train-image-root", type=Path)
    parser.add_argument("--validation-image-root", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--text-batch-size", type=int)
    parser.add_argument("--feature-precision")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--canary-max-images", type=int)
    parser.add_argument("--no-cache-equivalence-canary", dest="cache_equivalence_canary", action="store_false")
    parser.set_defaults(cache_equivalence_canary=None)
    parser.add_argument("--cache-equivalence-min-mean-cosine", type=float)
    parser.add_argument("--cache-equivalence-min-min-cosine", type=float)
    parser.add_argument("--cache-equivalence-max-abs-diff", type=float)
    parser.add_argument("--official-logits-max-abs-diff", type=float)
    parser.add_argument("--official-logits-rtol", type=float)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = _build_config(parse_args())
    metadata = cache_vlm_shared_features(config)
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), **metadata["outputs"]}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> VLMSharedCacheConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = VLMSharedCacheConfig(
        train_cache=Path(raw.get("train_cache", "")),
        validation_cache=Path(raw.get("validation_cache", "")),
        phrase_bank_csv=Path(raw.get("phrase_bank_csv", "")),
        output_dir=Path(raw.get("vlm_shared_dir", raw.get("output_dir", ""))),
        interpretability_manifest_dir=_optional_path(raw.get("interpretability_manifest_dir")),
        image_cache_mode=str(raw.get("image_cache_mode", "encode")),
        shared_image_train_cache=_optional_path(raw.get("shared_image_train_cache")),
        shared_image_validation_cache=_optional_path(raw.get("shared_image_validation_cache")),
        model_name=str(raw.get("model_name", "google/siglip2-base-patch16-256")),
        model_path=_optional_path(raw.get("model_path")),
        model_revision=str(raw.get("model_revision", "")),
        asset_manifest=_optional_path(raw.get("asset_manifest")),
        train_image_root=_optional_path(raw.get("train_image_root")),
        validation_image_root=_optional_path(raw.get("validation_image_root")),
        batch_size=int(raw.get("batch_size", 64)),
        text_batch_size=int(raw.get("text_batch_size", 256)),
        feature_precision=str(raw.get("feature_precision", "fp32")),
        max_train_samples=_optional_int(raw.get("max_train_samples")),
        max_validation_samples=_optional_int(raw.get("max_validation_samples")),
        canary_max_images=int(raw.get("canary_max_images", 128)),
        cache_equivalence_canary=bool(raw.get("cache_equivalence_canary", True)),
        cache_equivalence_min_mean_cosine=float(raw.get("cache_equivalence_min_mean_cosine", 0.99999)),
        cache_equivalence_min_min_cosine=float(raw.get("cache_equivalence_min_min_cosine", 0.9999)),
        cache_equivalence_max_abs_diff=float(raw.get("cache_equivalence_max_abs_diff", 1.0e-4)),
        official_logits_max_abs_diff=float(raw.get("official_logits_max_abs_diff", 1.0e-4)),
        official_logits_rtol=float(raw.get("official_logits_rtol", 1.0e-4)),
        device=str(raw.get("device", "auto")),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "train_cache",
        "validation_cache",
        "phrase_bank_csv",
        "output_dir",
        "interpretability_manifest_dir",
        "image_cache_mode",
        "shared_image_train_cache",
        "shared_image_validation_cache",
        "model_name",
        "model_path",
        "model_revision",
        "asset_manifest",
        "train_image_root",
        "validation_image_root",
        "batch_size",
        "text_batch_size",
        "feature_precision",
        "max_train_samples",
        "max_validation_samples",
        "canary_max_images",
        "cache_equivalence_canary",
        "cache_equivalence_min_mean_cosine",
        "cache_equivalence_min_min_cosine",
        "cache_equivalence_max_abs_diff",
        "official_logits_max_abs_diff",
        "official_logits_rtol",
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
    if not str(cfg.phrase_bank_csv):
        raise ValueError("phrase_bank_csv is required")
    if not str(cfg.output_dir):
        raise ValueError("output_dir is required")
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
