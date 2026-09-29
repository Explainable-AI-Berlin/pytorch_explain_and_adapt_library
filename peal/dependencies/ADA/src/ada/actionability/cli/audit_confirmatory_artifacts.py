from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping


DEFAULT_SELECTION_DIR = Path(
    "artifacts/ada/actionability/pilot_regions/"
    "e7a_in1k_dinov2_cls_k10_confirmatory_exclude_in100_seed0"
)
DEFAULT_DELETION_DIR = Path(
    "artifacts/ada/actionability/deletion_confirmatory/"
    "e7b_in1k_dinov2_cls_k10_confirmatory_exclude_in100_seed0"
)
DEFAULT_RESTORATION_DIR = Path(
    "artifacts/ada/actionability/restoration_confirmatory/"
    "e7c_in1k_dinov2_cls_k10_exclude_in100_real_restoration_seed0"
)
DEFAULT_EXCLUDED_WNIDS = Path(
    "artifacts/ada/actionability/exclusions/"
    "e7a_in1k_excluded_development_wnids_in100/excluded_development_wnids.txt"
)
DEFAULT_REPORT = Path("notes/ada/actionability/in1k_k10_confirmatory_exclude_in100_artifact_audit.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit clean ImageNet-1K ADA confirmatory artifacts.")
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--deletion-dir", type=Path, default=DEFAULT_DELETION_DIR)
    parser.add_argument("--restoration-dir", type=Path, default=DEFAULT_RESTORATION_DIR)
    parser.add_argument("--excluded-wnids-path", type=Path, default=DEFAULT_EXCLUDED_WNIDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--expected-regions", type=int, default=30)
    parser.add_argument("--expected-sparse-interiors", type=int, default=12)
    parser.add_argument("--expected-sparse-boundaries", type=int, default=10)
    parser.add_argument("--expected-dense-interiors", type=int, default=8)
    parser.add_argument("--expected-deletion-rows", type=int, default=810)
    parser.add_argument("--expected-restoration-rows", type=int, default=2070)
    parser.add_argument("--expected-full-restoration-rows", type=int, default=180)
    parser.add_argument("--min-validation-count", type=int, default=15)
    parser.add_argument("--min-retained-q025-count", type=int, default=25)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args)
    write_report(args.output, result)
    json_output = args.json_output or args.output.with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(json_output)
    if args.strict and result["failed_checks"]:
        raise SystemExit(1)


def audit(args: argparse.Namespace) -> dict[str, object]:
    selection_csv = args.selection_dir / "selected_pilot_regions.csv"
    deletion_csv = args.deletion_dir / "deletion_control_manifests.csv"
    restoration_csv = args.restoration_dir / "restoration_manifests.csv"

    selected = _read_csv(selection_csv)
    deletion = _read_csv(deletion_csv)
    restoration = _read_csv(restoration_csv)
    excluded = _read_lines(args.excluded_wnids_path)

    by_type = Counter(row["pilot_type"] for row in selected)
    selected_wnids = [row["class_name"] for row in selected]
    selected_classes = [row["class_id"] for row in selected]
    selected_excluded_overlap = sorted(set(selected_wnids) & set(excluded))
    duplicate_classes = sorted(cls for cls, count in Counter(selected_classes).items() if count > 1)
    train_counts = [_int(row, "train_count") for row in selected]
    val_counts = [_int(row, "validation_count") for row in selected]

    deletion_by_family = Counter(row["control_family"] for row in deletion)
    deletion_retention_levels = sorted({_float(row, "retention_level") for row in deletion})
    deletion_seeds = sorted({_int(row, "seed") for row in deletion})

    train_count_by_region = {row["region_id"]: _int(row, "train_count") for row in selected}
    retained_q025_counts = []
    for row in deletion:
        if row.get("control_family") == "regional_drop" and row.get("retention_label") == "retain_025":
            retained_q025_counts.append(
                int(train_count_by_region[row["region_id"]]) - _int(row, "deleted_sample_count")
            )

    restoration_by_condition = Counter(row["restoration_condition"] for row in restoration)
    restoration_by_retention = Counter(row["retention_level"] for row in restoration)
    restoration_seeds = sorted({_int(row, "seed") for row in restoration})
    full_rows = [row for row in restoration if _truthy(row.get("full_restoration_expected"))]
    full_matches = [row for row in full_rows if _truthy(row.get("full_restoration_matches_baseline_manifest"))]

    checks = {
        "selected_regions": len(selected) == args.expected_regions,
        "distinct_selected_classes": len(set(selected_classes)) == args.expected_regions,
        "sparse_interior_count": by_type.get("sparse_interior", 0) == args.expected_sparse_interiors,
        "sparse_boundary_count": by_type.get("sparse_boundary", 0) == args.expected_sparse_boundaries,
        "dense_interior_count": by_type.get("dense_interior", 0) == args.expected_dense_interiors,
        "selected_excluded_overlap_zero": len(selected_excluded_overlap) == 0,
        "minimum_validation_count": bool(val_counts) and min(val_counts) >= args.min_validation_count,
        "minimum_retained_q025_count": bool(retained_q025_counts)
        and min(retained_q025_counts) >= args.min_retained_q025_count,
        "deletion_rows": len(deletion) == args.expected_deletion_rows,
        "restoration_rows": len(restoration) == args.expected_restoration_rows,
        "full_restoration_rows": len(full_rows) == args.expected_full_restoration_rows,
        "full_restoration_matches": len(full_matches) == args.expected_full_restoration_rows,
        "duplicate_selected_classes_zero": len(duplicate_classes) == 0,
        "completion_markers": all(
            [
                (args.selection_dir / "COMPLETED").exists(),
                (args.deletion_dir / "COMPLETED").exists(),
                (args.restoration_dir / "COMPLETED").exists(),
            ]
        ),
    }

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "checks": checks,
        "paths": {
            "selection_dir": str(args.selection_dir),
            "deletion_dir": str(args.deletion_dir),
            "restoration_dir": str(args.restoration_dir),
            "excluded_wnids_path": str(args.excluded_wnids_path),
            "selection_csv": str(selection_csv),
            "deletion_csv": str(deletion_csv),
            "restoration_csv": str(restoration_csv),
        },
        "selection": {
            "selected_regions": len(selected),
            "distinct_classes": len(set(selected_classes)),
            "by_type": dict(sorted(by_type.items())),
            "excluded_wnid_count": len(excluded),
            "selected_excluded_overlap_count": len(selected_excluded_overlap),
            "selected_excluded_overlap": selected_excluded_overlap,
            "duplicate_selected_classes": duplicate_classes,
            "train_count": _summary(train_counts),
            "validation_count": _summary(val_counts),
        },
        "deletion": {
            "rows": len(deletion),
            "by_control_family": dict(sorted(deletion_by_family.items())),
            "retention_levels": deletion_retention_levels,
            "seeds": deletion_seeds,
            "retained_q025_count": _summary(retained_q025_counts),
        },
        "restoration": {
            "rows": len(restoration),
            "by_condition": dict(sorted(restoration_by_condition.items())),
            "by_retention": dict(sorted(restoration_by_retention.items())),
            "seeds": restoration_seeds,
            "full_restoration_rows": len(full_rows),
            "full_restoration_matches": len(full_matches),
        },
    }


def write_report(path: Path, result: Mapping[str, object]) -> None:
    checks = result["checks"]
    lines = [
        "# IN1K K=10 Confirmatory Exclude-IN100 Artifact Audit",
        "",
        f"- status: `{result['status']}`",
        f"- failed checks: `{result['failed_checks']}`",
        "",
        "## Paths",
        "",
    ]
    for key, value in result["paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for key, passed in checks.items():
        lines.append(f"- {key}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Selection", ""])
    _append_mapping(lines, result["selection"])
    lines.extend(["", "## Deletion Controls", ""])
    _append_mapping(lines, result["deletion"])
    lines.extend(["", "## Real Restoration", ""])
    _append_mapping(lines, result["restoration"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _append_mapping(lines: list[str], values: Mapping[str, object]) -> None:
    for key, value in values.items():
        lines.append(f"- {key}: `{value}`")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _int(row: Mapping[str, str], key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def _float(row: Mapping[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _summary(values: Iterable[int]) -> dict[str, int | float | None]:
    vals = sorted(int(value) for value in values)
    if not vals:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(vals), "min": min(vals), "median": float(median(vals)), "max": max(vals)}


if __name__ == "__main__":
    main()
