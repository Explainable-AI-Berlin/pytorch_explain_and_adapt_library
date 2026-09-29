from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.actionability.deletion import DeletionBuildConfig, build_deletion_artifact
from ada.actionability.region_filters import EligibilityThresholds
from ada.atlas.hashing import file_sha1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ADA controlled regional deletion manifests.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--regions-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--retention-levels", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-regions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    if args.dry_run:
        payload = {
            "dry_run": True,
            "train_cache": str(config.train_cache),
            "regions_dir": str(config.regions_dir),
            "output_dir": str(config.output_dir),
            "retention_levels": list(config.retention_levels),
            "max_regions": config.max_regions,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    metadata = build_deletion_artifact(config)
    report_path = write_deletion_report(metadata, Path("notes/ada/actionability/deletion_manifest_report.md"))
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), "report": str(report_path)}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> DeletionBuildConfig:
    raw = _load_config(args.config) if args.config else {}
    thresholds_raw = dict(raw.get("eligibility", {}))
    thresholds = EligibilityThresholds(
        min_train_count=int(thresholds_raw.get("min_train_count", 100)),
        min_val_count=int(thresholds_raw.get("min_val_count", 10)),
        same_class_purity=float(thresholds_raw.get("same_class_purity", 0.90)),
        min_positive_class_margin=float(thresholds_raw.get("min_positive_class_margin", 0.0)),
        max_duplicate_fraction=float(thresholds_raw.get("max_duplicate_fraction", 0.10)),
    )
    cfg = DeletionBuildConfig(
        train_cache=Path(raw.get("train_cache", "")),
        regions_dir=Path(raw.get("regions_dir", "")),
        output_dir=Path(raw.get("output_dir", "")),
        retention_levels=tuple(float(x) for x in raw.get("retention_levels", [1.0, 0.5, 0.25, 0.1, 0.0])),
        seed=int(raw.get("seed", 0)),
        max_regions=int(raw["max_regions"]) if raw.get("max_regions") is not None else None,
        overwrite=bool(raw.get("overwrite", False)),
        thresholds=thresholds,
    )
    overrides: dict[str, Any] = {}
    for key in ("train_cache", "regions_dir", "output_dir", "seed", "max_regions", "overwrite"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.retention_levels:
        overrides["retention_levels"] = tuple(_parse_levels(args.retention_levels))
    cfg = replace(cfg, **overrides)
    if not str(cfg.train_cache):
        raise ValueError("train_cache is required")
    if not str(cfg.regions_dir):
        raise ValueError("regions_dir is required")
    if not str(cfg.output_dir):
        raise ValueError("output_dir is required")
    return cfg


def _parse_levels(raw: str) -> list[float]:
    values = [float(item) for item in raw.replace(":", ",").split(",") if item.strip()]
    if not values or min(values) < 0.0 or max(values) > 1.0:
        raise ValueError(f"invalid retention levels: {raw!r}")
    return values


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


def write_deletion_report(metadata: dict[str, object], path: Path) -> Path:
    lines = [
        "# ADA Actionability Deletion Manifest Report",
        "",
        f"- experiment: `{metadata.get('experiment_id', '')}`",
        f"- artifact: `{metadata.get('artifact_id', '')}`",
        f"- train cache: `{metadata.get('train_cache', '')}`",
        f"- regions dir: `{metadata.get('regions_dir', '')}`",
        f"- output dir: `{metadata.get('output_dir', '')}`",
        f"- region artifact: `{metadata.get('region_artifact_id', '')}`",
        f"- region hash: `{metadata.get('region_hash', '')}`",
        f"- selected regions: `{metadata.get('selected_region_count', 0)}`",
        f"- deletion manifests: `{metadata.get('manifest_count', 0)}`",
        f"- retention levels: `{metadata.get('retention_levels', [])}`",
        "",
        "## Leakage Declaration",
        "",
        str(metadata.get("leakage_declaration", "")),
        "",
        "## Status",
        "",
        "Only deletion manifests were written. No classifiers were trained, no generator code was modified, and no full experiments were launched.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    main()
