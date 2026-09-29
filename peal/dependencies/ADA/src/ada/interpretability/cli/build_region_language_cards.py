from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.atlas.hashing import file_sha1
from ada.interpretability.region_cards import RegionLanguageCardConfig, build_region_language_cards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build language cards for selected ADA regions.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vlm-shared-dir", type=Path)
    parser.add_argument("--enriched-regions-dir", type=Path)
    parser.add_argument("--pilot-regions-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interpretability-manifest-dir", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--deletion-evaluation-csv", type=Path)
    parser.add_argument("--restoration-evaluation-csv", type=Path)
    parser.add_argument("--baseline-predictions-csv", type=Path)
    parser.add_argument("--control-max-count", type=int)
    parser.add_argument("--top-k-phrases", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--bootstrap-top-k", type=int)
    parser.add_argument("--permutation-samples", type=int)
    parser.add_argument("--min-bootstrap-frequency", type=float)
    parser.add_argument("--min-supporting-images", type=int)
    parser.add_argument("--no-sparse", dest="sparse_enabled", action="store_false")
    parser.set_defaults(sparse_enabled=None)
    parser.add_argument("--sparse-max-terms", type=int)
    parser.add_argument("--text-temperature", type=float)
    parser.add_argument("--text-logit-scale", type=float)
    parser.add_argument("--temperature-calibration-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = _build_config(parse_args())
    metadata = build_region_language_cards(config)
    report = write_region_cards_report(metadata, Path(config.output_dir) / "report.md")
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), "report": str(report), **metadata["outputs"]}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> RegionLanguageCardConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = RegionLanguageCardConfig(
        vlm_shared_dir=Path(raw.get("vlm_shared_dir", "")),
        enriched_regions_dir=Path(raw.get("enriched_regions_dir", "")),
        pilot_regions_dir=Path(raw.get("pilot_regions_dir", "")),
        output_dir=Path(raw.get("region_cards_dir", raw.get("output_dir", ""))),
        interpretability_manifest_dir=_optional_path(raw.get("interpretability_manifest_dir")),
        train_cache=_optional_path(raw.get("train_cache")),
        validation_cache=_optional_path(raw.get("validation_cache")),
        deletion_evaluation_csv=_optional_path(raw.get("deletion_evaluation_csv")),
        restoration_evaluation_csv=_optional_path(raw.get("restoration_evaluation_csv")),
        baseline_predictions_csv=_optional_path(raw.get("baseline_predictions_csv")),
        control_max_count=int(raw.get("control_max_count", 256)),
        top_k_phrases=int(raw.get("top_k_phrases", 12)),
        bootstrap_samples=int(raw.get("bootstrap_samples", 100)),
        bootstrap_top_k=int(raw.get("bootstrap_top_k", 20)),
        permutation_samples=int(raw.get("permutation_samples", 100)),
        min_bootstrap_frequency=float(raw.get("min_bootstrap_frequency", 0.0)),
        min_supporting_images=int(raw.get("min_supporting_images", 5)),
        sparse_enabled=bool(raw.get("sparse_enabled", True)),
        sparse_max_terms=int(raw.get("sparse_max_terms", 8)),
        text_temperature=_optional_float(raw.get("text_temperature")),
        text_logit_scale=_optional_float(raw.get("text_logit_scale")),
        temperature_calibration_samples=int(raw.get("temperature_calibration_samples", 2048)),
        seed=int(raw.get("seed", 0)),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "vlm_shared_dir",
        "enriched_regions_dir",
        "pilot_regions_dir",
        "output_dir",
        "interpretability_manifest_dir",
        "train_cache",
        "validation_cache",
        "deletion_evaluation_csv",
        "restoration_evaluation_csv",
        "baseline_predictions_csv",
        "control_max_count",
        "top_k_phrases",
        "bootstrap_samples",
        "bootstrap_top_k",
        "permutation_samples",
        "min_bootstrap_frequency",
        "min_supporting_images",
        "sparse_enabled",
        "sparse_max_terms",
        "text_temperature",
        "text_logit_scale",
        "temperature_calibration_samples",
        "seed",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    cfg = replace(cfg, **overrides)
    if not str(cfg.vlm_shared_dir):
        raise ValueError("vlm_shared_dir is required")
    if not str(cfg.enriched_regions_dir):
        raise ValueError("enriched_regions_dir is required")
    if not str(cfg.pilot_regions_dir):
        raise ValueError("pilot_regions_dir is required")
    if not str(cfg.output_dir):
        raise ValueError("output_dir is required")
    return cfg


def write_region_cards_report(metadata: dict[str, object], path: Path) -> Path:
    summary = dict(metadata.get("summary", {}))
    lines = [
        "# ADA Region Language Cards",
        "",
        f"- artifact: `{metadata.get('artifact_id', '')}`",
        f"- regions: `{summary.get('regions', 0)}`",
        f"- concept rows: `{summary.get('concept_rows', 0)}`",
        f"- ambiguity rows: `{summary.get('ambiguity_rows', 0)}`",
        f"- deleted-vs-retained rows: `{summary.get('deleted_vs_retained_rows', 0)}`",
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


def _optional_path(value: object) -> Path | None:
    if value in ("", None):
        return None
    return Path(str(value))


def _optional_float(value: object) -> float | None:
    if value in ("", None):
        return None
    return float(value)


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
