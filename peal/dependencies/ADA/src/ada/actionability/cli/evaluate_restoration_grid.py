from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ada.actionability.evaluate_restoration import RestorationEvaluationConfig, evaluate_restoration_grid
from ada.atlas.hashing import file_sha1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the ImageNet-100 K=10 real-restoration pilot.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--enriched-regions-dir", type=Path)
    parser.add_argument("--restoration-dir", type=Path)
    parser.add_argument("--restoration-probe-output-dir", type=Path)
    parser.add_argument("--deletion-probe-output-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--support-k", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    metadata = evaluate_restoration_grid(config)
    raw = _load_config(args.config) if args.config else {}
    report = write_restoration_evaluation_report(metadata, args.report_path or Path(raw.get("report_path", "notes/ada/actionability/real_restoration_evaluation_report.md")))
    print(json.dumps({"metadata_json": str(Path(config.output_dir) / "metadata.json"), "report": str(report)}, indent=2, sort_keys=True))


def _build_config(args: argparse.Namespace) -> RestorationEvaluationConfig:
    raw = _load_config(args.config) if args.config else {}
    cfg = RestorationEvaluationConfig(
        train_cache=Path(raw.get("train_cache", "")),
        validation_cache=Path(raw.get("validation_cache", "")),
        enriched_regions_dir=Path(raw.get("enriched_regions_dir", "")),
        restoration_dir=Path(raw.get("restoration_dir", "")),
        restoration_probe_output_dir=Path(raw.get("restoration_probe_output_dir", raw.get("probe_output_dir", ""))),
        deletion_probe_output_dir=Path(raw.get("deletion_probe_output_dir", "")),
        output_dir=Path(raw.get("restoration_eval_output_dir", raw.get("eval_output_dir", raw.get("output_dir", "")))),
        experiment_id=str(raw.get("restoration_eval_experiment_id", raw.get("eval_experiment_id", raw.get("experiment_id", "e5a_in100_real_restoration_pilot_evaluation")))),
        support_k=int(raw.get("support_k", 50)),
        overwrite=bool(raw.get("overwrite", False)),
    )
    overrides: dict[str, Any] = {}
    for key in (
        "train_cache",
        "validation_cache",
        "enriched_regions_dir",
        "restoration_dir",
        "restoration_probe_output_dir",
        "deletion_probe_output_dir",
        "output_dir",
        "experiment_id",
        "support_k",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    cfg = replace(cfg, **overrides)
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


def write_restoration_evaluation_report(metadata: dict[str, object], path: Path) -> Path:
    summary = dict(metadata.get("summary", {}))
    lines = [
        "# ADA Real Restoration Evaluation Report",
        "",
        f"- experiment: `{metadata.get('experiment_id', '')}`",
        f"- artifact: `{metadata.get('artifact_id', '')}`",
        f"- evaluated rows: `{summary.get('rows', 0)}`",
        f"- unique regions: `{summary.get('unique_regions', 0)}`",
        f"- support k: `{summary.get('support_k', '')}`",
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
