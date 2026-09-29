from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.actionability.deletion_controls import DeletionControlConfig, build_deletion_controls
from ada.atlas.hashing import file_sha1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build counted exposure manifests for the K=10 causal deletion pilot.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--enriched-regions-dir", type=Path)
    parser.add_argument("--pilot-regions-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--superseded-deletion-dir", type=Path)
    parser.add_argument("--retention-levels", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": _jsonable_config(config)}, indent=2, sort_keys=True))
        return
    metadata = build_deletion_controls(config)
    raw = _load_config(args.config) if args.config else {}
    report = write_deletion_control_report(metadata, args.report_path or Path(raw.get("report_path", "notes/ada/actionability/k10_deletion_control_manifest_report.md")))
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), "report": str(report)}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> DeletionControlConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = DeletionControlConfig(
        train_cache=Path(raw.get("train_cache", "")),
        enriched_regions_dir=Path(raw.get("enriched_regions_dir", "")),
        pilot_regions_dir=Path(raw.get("pilot_regions_dir", "")),
        output_dir=Path(raw.get("output_dir", "")),
        experiment_id=str(raw.get("experiment_id", "e4a_in100_k10_causal_pilot_deletion_controls")),
        superseded_deletion_dir=Path(raw["superseded_deletion_dir"]) if raw.get("superseded_deletion_dir") else None,
        retention_levels=tuple(float(x) for x in raw.get("retention_levels", [1.0, 0.25, 0.0])),
        seeds=tuple(int(x) for x in raw.get("seeds", [0, 1, 2])),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "train_cache",
        "enriched_regions_dir",
        "pilot_regions_dir",
        "output_dir",
        "experiment_id",
        "superseded_deletion_dir",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.retention_levels:
        overrides["retention_levels"] = tuple(_parse_floats(args.retention_levels))
    if args.seeds:
        overrides["seeds"] = tuple(int(x) for x in _parse_floats(args.seeds))
    cfg = replace(cfg, **overrides)
    if not str(cfg.train_cache):
        raise ValueError("train_cache is required")
    if not str(cfg.enriched_regions_dir):
        raise ValueError("enriched_regions_dir is required")
    if not str(cfg.pilot_regions_dir):
        raise ValueError("pilot_regions_dir is required")
    if not str(cfg.output_dir):
        raise ValueError("output_dir is required")
    return cfg


def _parse_floats(raw: str) -> list[float]:
    return [float(item) for item in raw.replace(":", ",").split(",") if item.strip()]


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


def _jsonable_config(config: DeletionControlConfig) -> dict[str, object]:
    out = dict(config.__dict__)
    for key in ("train_cache", "enriched_regions_dir", "pilot_regions_dir", "output_dir", "superseded_deletion_dir"):
        out[key] = str(out[key]) if out[key] is not None else ""
    return out


def write_deletion_control_report(metadata: dict[str, object], path: Path) -> Path:
    summary = dict(metadata.get("summary", {}))
    config = dict(metadata.get("config", {}))
    lines = [
        "# ADA K=10 Deletion-Control Manifest Report",
        "",
        f"- experiment: `{metadata.get('experiment_id', '')}`",
        f"- artifact: `{metadata.get('artifact_id', '')}`",
        f"- enrichment artifact: `{metadata.get('enrichment_artifact_id', '')}`",
        f"- pilot region artifact: `{metadata.get('pilot_region_artifact_id', '')}`",
        f"- source region artifact: `{metadata.get('source_region_artifact_id', '')}`",
        f"- frozen validation assignment hash: `{metadata.get('source_validation_assignment_hash', '')}`",
        f"- output dir: `{config.get('output_dir', '')}`",
        "",
        "## Summary",
        "",
        f"- selected regions: `{summary.get('selected_regions', 0)}`",
        f"- retention levels: `{summary.get('retention_levels', [])}`",
        f"- seeds: `{summary.get('seeds', [])}`",
        f"- manifests: `{summary.get('manifest_count', 0)}`",
        f"- by control family: `{summary.get('by_control_family', {})}`",
        "",
        "## Controls",
        "",
        "- `baseline`: full training set, one per region/seed at retention 1.0.",
        "- `regional_drop`: remove target-region samples.",
        "- `same_class_random_drop`: remove the same number of same-class off-target samples.",
        "- `global_random_drop`: remove the same number of random training samples.",
        "- `same_class_count_preserving_replacement`: remove target-region samples and reallocate exposures to off-target same-class samples.",
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
