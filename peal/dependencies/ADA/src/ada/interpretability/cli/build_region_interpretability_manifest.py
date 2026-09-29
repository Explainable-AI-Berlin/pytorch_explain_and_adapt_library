from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.atlas.hashing import file_sha1
from ada.interpretability.region_manifest import RegionInterpretabilityManifestConfig, build_region_interpretability_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen sample manifest for eight-region VLM interpretation.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--enriched-regions-dir", type=Path)
    parser.add_argument("--pilot-regions-dir", type=Path)
    parser.add_argument("--deletion-controls-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-predictions-csv", type=Path)
    parser.add_argument("--control-count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = _build_config(parse_args())
    metadata = build_region_interpretability_manifest(config)
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), **metadata["outputs"]}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> RegionInterpretabilityManifestConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = RegionInterpretabilityManifestConfig(
        train_cache=Path(raw.get("train_cache", "")),
        validation_cache=Path(raw.get("validation_cache", "")),
        enriched_regions_dir=Path(raw.get("enriched_regions_dir", "")),
        pilot_regions_dir=Path(raw.get("pilot_regions_dir", "")),
        deletion_controls_dir=Path(raw.get("deletion_controls_dir", "")),
        output_dir=Path(raw.get("interpretability_manifest_dir", raw.get("output_dir", ""))),
        baseline_predictions_csv=_optional_path(raw.get("baseline_predictions_csv")),
        control_count=int(raw.get("control_count", raw.get("control_max_count", 256))),
        seed=int(raw.get("seed", 0)),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "train_cache",
        "validation_cache",
        "enriched_regions_dir",
        "pilot_regions_dir",
        "deletion_controls_dir",
        "output_dir",
        "baseline_predictions_csv",
        "control_count",
        "seed",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    cfg = replace(cfg, **overrides)
    for key in ("train_cache", "validation_cache", "enriched_regions_dir", "pilot_regions_dir", "deletion_controls_dir", "output_dir"):
        if not str(getattr(cfg, key)):
            raise ValueError(f"{key} is required")
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
