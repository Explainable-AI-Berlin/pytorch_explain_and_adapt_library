from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from ada.atlas.hashing import file_sha1, hash_rows, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize class distribution of DINO-CLS outlier validation examples.")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--description-csv", type=Path)
    parser.add_argument("--deletion-evaluation-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-seed", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.9, 0.8, 0.7, 0.5])
    parser.add_argument("--top-n", type=int, nargs="+", default=[12, 25, 50])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not args.overwrite:
        raise FileExistsError(f"outlier class distribution artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_dir) / "samples.csv"
    rows = [
        row
        for row in _read_csv(manifest_path)
        if str(row.get("analysis_group", "")) == "region_val_member"
    ]
    descriptions = _read_optional_by_sample(args.description_csv)
    correctness = (
        _baseline_correctness(
            args.deletion_evaluation_csv,
            baseline_seed=int(args.baseline_seed),
            wanted_sample_ids=[str(row["sample_id"]) for row in rows],
        )
        if args.deletion_evaluation_csv
        else {}
    )

    joined: list[dict[str, object]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        desc = descriptions.get(sample_id, {})
        pred = correctness.get(sample_id, {})
        joined.append(
            {
                "sample_id": sample_id,
                "class_id": int(float(row["class_id"])),
                "class_name": str(row["class_name"]),
                "region_id": str(row["region_id"]),
                "pilot_type": str(row["pilot_type"]),
                "relative_path": str(row["relative_path"]),
                "dino_cls_outlier_percentile": _float(row.get("support_percentile")),
                "text_margin": _float(desc.get("text_margin")),
                "top_competing_class_name": str(desc.get("top_competing_class_name", "")),
                "downstream_correct": _optional_int(pred.get("correct")),
                "downstream_predicted_label": pred.get("predicted_label", ""),
            }
        )
    joined.sort(key=lambda row: float(row["dino_cls_outlier_percentile"]), reverse=True)

    class_rows = _class_rows(joined)
    threshold_rows = _threshold_rows(joined, args.thresholds)
    topn_rows = _topn_rows(joined, args.top_n)

    joined_csv = output / "outlier_validation_rows.csv"
    class_csv = output / "class_distribution.csv"
    threshold_csv = output / "threshold_summary.csv"
    topn_csv = output / "topn_summary.csv"
    report_md = output / "report.md"
    _write_csv(joined_csv, joined)
    _write_csv(class_csv, class_rows)
    _write_csv(threshold_csv, threshold_rows)
    _write_csv(topn_csv, topn_rows)
    report_md.write_text(_report(class_rows, threshold_rows, topn_rows), encoding="utf-8")

    result_hash = hash_rows(class_rows + threshold_rows + topn_rows, prefix="outlier-class-distribution")
    metadata = {
        "artifact_id": stable_hash(
            {
                "result_hash": result_hash,
                "manifest_dir": str(args.manifest_dir),
                "description_csv": str(args.description_csv or ""),
                "deletion_evaluation_csv": str(args.deletion_evaluation_csv or ""),
            },
            prefix="outlier-class-dist-artifact",
        ),
        "result_hash": result_hash,
        "summary": {
            "validation_rows": len(joined),
            "classes": len({row["class_name"] for row in joined}),
            "thresholds": [float(x) for x in args.thresholds],
            "top_n": [int(x) for x in args.top_n],
        },
        "outputs": {
            "outlier_validation_rows_csv": str(joined_csv),
            "class_distribution_csv": str(class_csv),
            "threshold_summary_csv": str(threshold_csv),
            "topn_summary_csv": str(topn_csv),
            "report_md": str(report_md),
        },
        "config": {
            "manifest_dir": str(args.manifest_dir),
            "manifest_sha1": file_sha1(manifest_path),
            "description_csv": str(args.description_csv or ""),
            "description_csv_sha1": file_sha1(args.description_csv) if args.description_csv else "",
            "deletion_evaluation_csv": str(args.deletion_evaluation_csv or ""),
            "baseline_seed": int(args.baseline_seed),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    completed.write_text("ok\n", encoding="utf-8")
    print(json.dumps({"metadata_json": str(output / "metadata.json"), **metadata["outputs"]}, indent=2, sort_keys=True))


def _class_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_class: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[(int(row["class_id"]), str(row["class_name"]))].append(row)
    out = []
    total = len(rows)
    for (class_id, class_name), group in sorted(
        by_class.items(),
        key=lambda item: (max(float(row["dino_cls_outlier_percentile"]) for row in item[1]), len(item[1])),
        reverse=True,
    ):
        correct_values = [int(row["downstream_correct"]) for row in group if row.get("downstream_correct") not in ("", None)]
        out.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "validation_members": len(group),
                "fraction_of_validation_members": len(group) / max(total, 1),
                "max_dino_cls_outlier_percentile": max(float(row["dino_cls_outlier_percentile"]) for row in group),
                "mean_dino_cls_outlier_percentile": _mean(float(row["dino_cls_outlier_percentile"]) for row in group),
                "mean_text_margin": _mean(_finite(float(row["text_margin"])) for row in group),
                "downstream_accuracy": _mean(correct_values),
                "pilot_types": ";".join(sorted({str(row["pilot_type"]) for row in group})),
                "regions": ";".join(sorted({str(row["region_id"]) for row in group})),
            }
        )
    return out


def _threshold_rows(rows: Sequence[Mapping[str, object]], thresholds: Sequence[float]) -> list[dict[str, object]]:
    out = []
    for threshold in thresholds:
        group = [row for row in rows if float(row["dino_cls_outlier_percentile"]) >= float(threshold)]
        by_class = defaultdict(int)
        for row in group:
            by_class[str(row["class_name"])] += 1
        correct = [int(row["downstream_correct"]) for row in group if row.get("downstream_correct") not in ("", None)]
        leader = ""
        leader_count = 0
        if by_class:
            leader, leader_count = sorted(by_class.items(), key=lambda kv: kv[1], reverse=True)[0]
        out.append(
            {
                "threshold": float(threshold),
                "n_outliers": len(group),
                "n_classes": len(by_class),
                "top_class": leader,
                "top_class_count": leader_count,
                "top_class_fraction": leader_count / max(len(group), 1),
                "downstream_accuracy": _mean(correct),
                "class_counts": ";".join(f"{k}:{v}" for k, v in sorted(by_class.items(), key=lambda kv: kv[1], reverse=True)),
            }
        )
    return out


def _topn_rows(rows: Sequence[Mapping[str, object]], top_ns: Sequence[int]) -> list[dict[str, object]]:
    out = []
    for n in top_ns:
        group = list(rows[: int(n)])
        by_class = defaultdict(int)
        for row in group:
            by_class[str(row["class_name"])] += 1
        correct = [int(row["downstream_correct"]) for row in group if row.get("downstream_correct") not in ("", None)]
        leader = ""
        leader_count = 0
        if by_class:
            leader, leader_count = sorted(by_class.items(), key=lambda kv: kv[1], reverse=True)[0]
        out.append(
            {
                "top_n": int(n),
                "n_rows": len(group),
                "n_classes": len(by_class),
                "top_class": leader,
                "top_class_count": leader_count,
                "top_class_fraction": leader_count / max(len(group), 1),
                "downstream_accuracy": _mean(correct),
                "class_counts": ";".join(f"{k}:{v}" for k, v in sorted(by_class.items(), key=lambda kv: kv[1], reverse=True)),
            }
        )
    return out


def _report(
    class_rows: Sequence[Mapping[str, object]],
    threshold_rows: Sequence[Mapping[str, object]],
    topn_rows: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# DINOv2-CLS Outlier Class Distribution",
        "",
        "Scope: validation members of the eight causally tested ImageNet-100 DINOv2-CLS regions.",
        "The outlier score is the region support percentile from the frozen manifest; larger means weaker same-class DINOv2-CLS support.",
        "",
        "## Thresholds",
        "",
        "| threshold | n | classes | top class | top class fraction | downstream acc |",
        "|---:|---:|---:|---|---:|---:|",
    ]
    for row in threshold_rows:
        lines.append(
            f"| {float(row['threshold']):.2f} | {row['n_outliers']} | {row['n_classes']} | "
            f"`{row['top_class']}` | {float(row['top_class_fraction']):.3f} | {_fmt(row['downstream_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Top-N",
            "",
            "| top n | classes | top class | top class fraction | downstream acc | class counts |",
            "|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in topn_rows:
        lines.append(
            f"| {row['top_n']} | {row['n_classes']} | `{row['top_class']}` | "
            f"{float(row['top_class_fraction']):.3f} | {_fmt(row['downstream_accuracy'])} | `{row['class_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Classes",
            "",
            "| class | validation members | max outlier percentile | mean outlier percentile | downstream acc | pilot types |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| `{row['class_name']}` | {row['validation_members']} | "
            f"{float(row['max_dino_cls_outlier_percentile']):.4f} | "
            f"{float(row['mean_dino_cls_outlier_percentile']):.4f} | "
            f"{_fmt(row['downstream_accuracy'])} | `{row['pilot_types']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _baseline_correctness(path: Path, *, baseline_seed: int, wanted_sample_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    wanted = set(wanted_sample_ids)
    eval_rows = _read_csv(path)
    pred_paths: list[Path] = []
    for row in eval_rows:
        if str(row.get("control_family", "")) != "baseline":
            continue
        if not math.isclose(_float(row.get("retention_level")), 1.0):
            continue
        if int(float(row.get("seed", "-1") or "-1")) != int(baseline_seed):
            continue
        pred = row.get("probe_predictions_csv", "")
        if pred:
            pred_paths.append(Path(str(pred)))

    out: dict[str, dict[str, str]] = {}
    for pred_path in pred_paths:
        for row in _read_csv(pred_path):
            sample_id = str(row.get("sample_id", ""))
            if sample_id in wanted and sample_id not in out:
                out[sample_id] = row
    return out


def _read_optional_by_sample(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    return {str(row["sample_id"]): row for row in _read_csv(path)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _float(value: object) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def _optional_int(value: object) -> int | str:
    if value in ("", None):
        return ""
    return int(float(value))


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _mean(values) -> float:
    materialized = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not materialized:
        return float("nan")
    return sum(materialized) / len(materialized)


def _fmt(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.3f}"


if __name__ == "__main__":
    main()
