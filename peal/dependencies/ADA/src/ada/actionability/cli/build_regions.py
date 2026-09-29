from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.actionability.region_filters import EligibilityThresholds
from ada.actionability.regions import RegionBuildConfig, build_region_artifact, load_embedding_bank
from ada.atlas.hashing import file_sha1, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only within-class ADA actionability regions.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--k", default=None, help="Comma/colon-separated K values, e.g. 5:10.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--residual-mode", choices=["raw", "centered", "tangent"])
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--local-purity-k", type=int)
    parser.add_argument("--duplicate-distance-threshold", type=float)
    parser.add_argument("--exemplar-count", type=int)
    parser.add_argument("--metric-batch-size", type=int)
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    if args.dry_run:
        train = load_embedding_bank(config.train_cache, mmap=config.mmap)
        validation = load_embedding_bank(config.validation_cache, mmap=config.mmap) if config.validation_cache else None
        payload = {
            "dry_run": True,
            "train_cache": str(config.train_cache),
            "train_shape": list(map(int, train.embeddings.shape)),
            "train_rows": len(train.rows),
            "validation_cache": str(config.validation_cache) if config.validation_cache else "",
            "validation_shape": list(map(int, validation.embeddings.shape)) if validation else [],
            "k_values": list(config.k_values),
            "output_dir": str(config.output_dir),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    metadata = build_region_artifact(config)
    raw = _load_config(args.config) if args.config else {}
    report_path = write_region_report(metadata, args.report_path or Path(raw.get("report_path", "notes/ada/actionability/region_build_report.md")))
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), "report": str(report_path)}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> RegionBuildConfig:
    raw = _load_config(args.config) if args.config else {}
    thresholds_raw = dict(raw.get("eligibility", {}))
    thresholds = EligibilityThresholds(
        min_train_count=int(thresholds_raw.get("min_train_count", 100)),
        min_val_count=int(thresholds_raw.get("min_val_count", 10)),
        same_class_purity=float(thresholds_raw.get("same_class_purity", 0.90)),
        min_positive_class_margin=float(thresholds_raw.get("min_positive_class_margin", 0.0)),
        max_duplicate_fraction=float(thresholds_raw.get("max_duplicate_fraction", 0.10)),
    )
    cfg = RegionBuildConfig(
        train_cache=Path(raw.get("train_cache", "")),
        validation_cache=Path(raw["validation_cache"]) if raw.get("validation_cache") else None,
        output_dir=Path(raw.get("output_dir", "")),
        experiment_id=str(raw.get("experiment_id", "e4a_in100_controlled_regional_deletion_regions")),
        k_values=tuple(int(x) for x in raw.get("k_values", [5, 10])),
        seed=int(raw.get("seed", 0)),
        residual_mode=str(raw.get("residual_mode", "tangent")),
        max_iter=int(raw.get("max_iter", 50)),
        tol=float(raw.get("tol", 1.0e-5)),
        local_purity_k=int(raw.get("local_purity_k", 50)),
        duplicate_distance_threshold=float(raw.get("duplicate_distance_threshold", 1.0e-4)),
        exemplar_count=int(raw.get("exemplar_count", 8)),
        metric_batch_size=int(raw.get("metric_batch_size", 128)),
        mmap=bool(raw.get("mmap", True)),
        overwrite=bool(raw.get("overwrite", False)),
        thresholds=thresholds,
    )
    overrides: dict[str, Any] = {}
    for key in (
        "train_cache",
        "validation_cache",
        "output_dir",
        "experiment_id",
        "seed",
        "residual_mode",
        "max_iter",
        "tol",
        "local_purity_k",
        "duplicate_distance_threshold",
        "exemplar_count",
        "metric_batch_size",
        "overwrite",
    ):
        arg_key = key.replace("_", "-")
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.k:
        overrides["k_values"] = tuple(_parse_k(args.k))
    if args.no_mmap:
        overrides["mmap"] = False
    cfg = replace(cfg, **overrides)
    if not str(cfg.train_cache):
        raise ValueError("train_cache is required")
    if not str(cfg.output_dir):
        raise ValueError("output_dir is required")
    return cfg


def _parse_k(raw: str) -> list[int]:
    values = [int(item) for item in raw.replace(":", ",").split(",") if item.strip()]
    if not values or min(values) < 1:
        raise ValueError(f"invalid K list: {raw!r}")
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


def write_region_report(metadata: dict[str, object], path: Path) -> Path:
    summary = dict(metadata.get("summary", {}))
    config = dict(metadata.get("config", {}))
    lines = [
        "# ADA Actionability Region Build Report",
        "",
        f"- experiment: `{metadata.get('experiment_id', '')}`",
        f"- artifact: `{metadata.get('artifact_id', '')}`",
        f"- train cache: `{config.get('train_cache', '')}`",
        f"- validation cache: `{config.get('validation_cache', '')}`",
        f"- K values: `{config.get('k_values', [])}`",
        f"- residual mode: `{config.get('residual_mode', '')}`",
        f"- config hash: `{metadata.get('config_hash', '')}`",
        f"- region hash: `{metadata.get('region_hash', '')}`",
        f"- membership hash: `{metadata.get('membership_hash', '')}`",
        f"- validation assignment hash: `{metadata.get('validation_assignment_hash', '')}`",
        "",
        "## Summary",
        "",
        f"- regions: `{summary.get('region_count', 0)}`",
        f"- membership rows: `{summary.get('membership_rows', 0)}`",
        f"- validation assignment rows: `{summary.get('validation_assignment_rows', 0)}`",
        f"- primary eligible regions: `{summary.get('eligible_primary_count', 0)}`",
        "",
        "## Leakage Declaration",
        "",
        str(metadata.get("leakage_declaration", "")),
        "",
        "## Outputs",
        "",
    ]
    for key, value in dict(metadata.get("outputs", {})).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    main()
