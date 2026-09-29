from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.actionability.select_pilot_regions import PilotSelectionConfig, select_pilot_regions
from ada.atlas.hashing import file_sha1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select eight K=10 pilot regions from train-only geometry.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--enriched-regions-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--excluded-wnids-path", type=Path)
    parser.add_argument("--excluded-wnids-sha256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": _jsonable_config(config)}, indent=2, sort_keys=True))
        return
    metadata = select_pilot_regions(config)
    raw = _load_config(args.config) if args.config else {}
    report = write_selection_report(metadata, args.report_path or Path(raw.get("report_path", "notes/ada/actionability/k10_pilot_region_selection_report.md")))
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), "report": str(report)}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> PilotSelectionConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = PilotSelectionConfig(
        enriched_regions_dir=Path(raw.get("enriched_regions_dir", "")),
        output_dir=Path(raw.get("output_dir", "")),
        experiment_id=str(raw.get("experiment_id", "e4a_in100_k10_causal_pilot_region_selection")),
        sparse_interior_count=int(raw.get("sparse_interior_count", 4)),
        dense_interior_count=int(raw.get("dense_interior_count", 2)),
        sparse_boundary_count=int(raw.get("sparse_boundary_count", 2)),
        min_train_count=int(raw.get("min_train_count", 100)),
        min_validation_count=int(raw.get("min_validation_count", 15)),
        sparse_support_pct_min=float(raw.get("sparse_support_pct_min", 0.75)),
        dense_support_pct_max=float(raw.get("dense_support_pct_max", 0.25)),
        purity_min=float(raw.get("purity_min", 0.90)),
        entropy_max=float(raw.get("entropy_max", 0.20)),
        margin_min=float(raw.get("margin_min", 0.0)),
        boundary_entropy_min=float(raw.get("boundary_entropy_min", 0.40)),
        boundary_margin_max=float(raw.get("boundary_margin_max", 0.0)),
        one_region_per_class=bool(raw.get("one_region_per_class", True)),
        excluded_wnids_path=Path(raw["excluded_wnids_path"]) if raw.get("excluded_wnids_path") else None,
        excluded_wnids_sha256=str(raw.get("excluded_wnids_sha256", "")),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "enriched_regions_dir",
        "output_dir",
        "experiment_id",
        "excluded_wnids_path",
        "excluded_wnids_sha256",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    cfg = replace(cfg, **overrides)
    if not str(cfg.enriched_regions_dir):
        raise ValueError("enriched_regions_dir is required")
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


def _jsonable_config(config: PilotSelectionConfig) -> dict[str, object]:
    out = dict(config.__dict__)
    for key in ("enriched_regions_dir", "output_dir"):
        out[key] = str(out[key])
    out["excluded_wnids_path"] = str(out["excluded_wnids_path"] or "")
    return out


def write_selection_report(metadata: dict[str, object], path: Path) -> Path:
    summary = dict(metadata.get("summary", {}))
    lines = [
        "# ADA K=10 Pilot Region Selection Report",
        "",
        f"- experiment: `{metadata.get('experiment_id', '')}`",
        f"- artifact: `{metadata.get('artifact_id', '')}`",
        f"- source enrichment artifact: `{metadata.get('source_enrichment_artifact_id', '')}`",
        f"- source region artifact: `{metadata.get('source_region_artifact_id', '')}`",
        f"- frozen validation assignment hash: `{metadata.get('source_validation_assignment_hash', '')}`",
        "",
        "## Summary",
        "",
        f"- selected regions: `{summary.get('selected_regions', 0)}`",
        f"- distinct classes: `{summary.get('distinct_classes', 0)}`",
        f"- by type: `{summary.get('by_type', {})}`",
        f"- threshold relaxations: `{summary.get('relaxation_count', 0)}`",
        "",
        "## Exclusion",
        "",
        f"- excluded WNID file: `{dict(metadata.get('excluded_wnids', {})).get('path', '')}`",
        f"- excluded WNID SHA256: `{dict(metadata.get('excluded_wnids', {})).get('sha256', '')}`",
        f"- excluded WNID count: `{dict(metadata.get('excluded_wnids', {})).get('count', 0)}`",
        f"- selected/excluded overlap: `{dict(metadata.get('excluded_wnids', {})).get('selected_overlap_count', 0)}`",
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
    if metadata.get("relaxations"):
        lines.extend(["## Relaxations", ""])
        for item in metadata["relaxations"]:
            lines.append(f"- `{item['region_id']}` `{item['pilot_type']}`: {item['reason']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    main()
