from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.actionability.enrich_regions import RegionEnrichmentConfig, enrich_regions
from ada.atlas.hashing import file_sha1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a derived K=10 actionability region-enrichment artifact.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--regions-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--primary-k", type=int)
    parser.add_argument("--parent-k", type=int)
    parser.add_argument("--support-k", type=int)
    parser.add_argument("--global-neighbor-k", type=int)
    parser.add_argument("--margin-core-k", type=int)
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": _jsonable_config(config)}, indent=2, sort_keys=True))
        return
    metadata = enrich_regions(config)
    raw = _load_config(args.config) if args.config else {}
    report = write_enrichment_report(metadata, args.report_path or Path(raw.get("report_path", "notes/ada/actionability/k10_region_enrichment_report.md")))
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), "report": str(report)}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> RegionEnrichmentConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = RegionEnrichmentConfig(
        train_cache=Path(raw.get("train_cache", "")),
        regions_dir=Path(raw.get("regions_dir", "")),
        output_dir=Path(raw.get("output_dir", "")),
        experiment_id=str(raw.get("experiment_id", "e4a_in100_k10_region_enrichment")),
        primary_k=int(raw.get("primary_k", 10)),
        parent_k=int(raw.get("parent_k", 5)),
        support_k=int(raw.get("support_k", 50)),
        global_neighbor_k=int(raw.get("global_neighbor_k", 50)),
        margin_core_k=int(raw.get("margin_core_k", 10)),
        mmap=bool(raw.get("mmap", True)),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "train_cache",
        "regions_dir",
        "output_dir",
        "experiment_id",
        "primary_k",
        "parent_k",
        "support_k",
        "global_neighbor_k",
        "margin_core_k",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.no_mmap:
        overrides["mmap"] = False
    cfg = replace(cfg, **overrides)
    if not str(cfg.train_cache):
        raise ValueError("train_cache is required")
    if not str(cfg.regions_dir):
        raise ValueError("regions_dir is required")
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


def _jsonable_config(config: RegionEnrichmentConfig) -> dict[str, object]:
    out = dict(config.__dict__)
    for key in ("train_cache", "regions_dir", "output_dir"):
        out[key] = str(out[key])
    return out


def write_enrichment_report(metadata: dict[str, object], path: Path) -> Path:
    summary = dict(metadata.get("summary", {}))
    config = dict(metadata.get("config", {}))
    lines = [
        "# ADA K=10 Region Enrichment Report",
        "",
        f"- experiment: `{metadata.get('experiment_id', '')}`",
        f"- artifact: `{metadata.get('artifact_id', '')}`",
        f"- source region artifact: `{metadata.get('source_region_artifact_id', '')}`",
        f"- source validation assignment hash: `{metadata.get('source_validation_assignment_hash', '')}`",
        f"- train cache: `{config.get('train_cache', '')}`",
        f"- regions dir: `{config.get('regions_dir', '')}`",
        f"- primary K: `{summary.get('primary_k', '')}`",
        f"- support k: `{summary.get('support_k', '')}`",
        f"- global neighbour k: `{summary.get('global_neighbor_k', '')}`",
        f"- margin core k: `{summary.get('margin_core_k', '')}`",
        "",
        "## Summary",
        "",
        f"- enriched K=10 regions: `{summary.get('regions', 0)}`",
        f"- membership rows: `{summary.get('membership_rows', 0)}`",
        f"- validation assignment rows: `{summary.get('validation_assignment_rows', 0)}`",
        f"- distinct classes: `{summary.get('distinct_classes', 0)}`",
        f"- primary-eligible rows retained from source: `{summary.get('eligible_primary_regions', 0)}`",
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
