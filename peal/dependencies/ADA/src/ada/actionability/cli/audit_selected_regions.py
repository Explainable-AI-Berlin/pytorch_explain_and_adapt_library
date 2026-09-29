from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping

from ada.actionability.region_filters import parse_float


DEFAULT_REGIONS_DIR = Path(
    "artifacts/ada/actionability/regions/e4a_in100_dinov2_cls_train_regions_k5_k10_seed0"
)
DEFAULT_DELETION_DIR = Path(
    "artifacts/ada/actionability/deletion/e4a_in100_dinov2_cls_train_regions_k5_k10_seed0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit selected ADA actionability regions before training.")
    parser.add_argument("--regions-dir", type=Path, default=DEFAULT_REGIONS_DIR)
    parser.add_argument("--deletion-dir", type=Path, default=DEFAULT_DELETION_DIR)
    parser.add_argument("--output", type=Path, default=Path("notes/ada/actionability/selected_region_audit.md"))
    parser.add_argument("--parent-child-csv", type=Path, default=Path("notes/ada/actionability/selected_region_parent_child_overlap.csv"))
    parser.add_argument("--primary-k10-csv", type=Path, default=Path("notes/ada/actionability/primary_k10_one_per_class_candidates.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regions = _read_csv(args.regions_dir / "regions.csv")
    memberships = _read_csv(args.regions_dir / "region_membership.csv")
    validation = _read_csv(args.regions_dir / "validation_assignments.csv")
    selected_ids = _selected_region_ids(args.deletion_dir, regions)
    selected = [row for row in regions if row["region_id"] in selected_ids]

    parent_rows = _parent_child_overlap(regions, memberships, selected)
    primary_k10 = _primary_k10_one_per_class(regions)

    _write_csv(args.parent_child_csv, parent_rows)
    _write_csv(args.primary_k10_csv, primary_k10)
    _write_report(
        args.output,
        regions_dir=args.regions_dir,
        deletion_dir=args.deletion_dir,
        regions=regions,
        selected=selected,
        validation=validation,
        parent_rows=parent_rows,
        primary_k10=primary_k10,
        parent_child_csv=args.parent_child_csv,
        primary_k10_csv=args.primary_k10_csv,
    )
    print(args.output)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
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
        writer.writerows(rows)


def _selected_region_ids(deletion_dir: Path, regions: list[dict[str, str]]) -> set[str]:
    manifest_csv = deletion_dir / "deletion_manifests.csv"
    if manifest_csv.exists():
        selected = {
            row["region_id"]
            for row in _read_csv(manifest_csv)
            if row.get("retention_label") == "retain_100"
        }
        if selected:
            return selected
    return {row["region_id"] for row in regions if row.get("eligible_primary") in {"1", "true", "True"}}


def _float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    value = parse_float(row.get(key))
    return default if value is None else float(value)


def _int(row: Mapping[str, str], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _primary_sort_key(row: Mapping[str, str]) -> tuple[int, int, float, str]:
    return (
        -_int(row, "validation_count"),
        -_int(row, "train_count"),
        -_float(row, "class_margin"),
        str(row.get("region_id", "")),
    )


def _primary_k10_one_per_class(regions: list[dict[str, str]]) -> list[dict[str, object]]:
    eligible = [
        row
        for row in regions
        if row.get("eligible_primary") in {"1", "true", "True"} and _int(row, "k_regions") == 10
    ]
    eligible.sort(key=_primary_sort_key)
    by_class: dict[str, dict[str, str]] = {}
    for row in eligible:
        by_class.setdefault(row["class_id"], row)
    out = []
    for rank, row in enumerate(sorted(by_class.values(), key=_primary_sort_key), start=1):
        item = dict(row)
        item["primary_rank"] = rank
        out.append(item)
    return out


def _parent_child_overlap(
    regions: list[dict[str, str]],
    memberships: list[dict[str, str]],
    selected: list[dict[str, str]],
) -> list[dict[str, object]]:
    region_by_id = {row["region_id"]: row for row in regions}
    wanted: set[str] = set()
    wanted.update(row["region_id"] for row in selected if _int(row, "k_regions") == 10)
    for row in regions:
        if _int(row, "k_regions") == 5:
            wanted.add(row["region_id"])

    members: dict[str, set[str]] = defaultdict(set)
    for row in memberships:
        region_id = row["region_id"]
        if region_id in wanted:
            members[region_id].add(row["sample_id"])

    parents_by_class: dict[str, list[str]] = defaultdict(list)
    for row in regions:
        if _int(row, "k_regions") == 5:
            parents_by_class[row["class_id"]].append(row["region_id"])

    out: list[dict[str, object]] = []
    for child in sorted((row for row in selected if _int(row, "k_regions") == 10), key=lambda row: row["region_id"]):
        child_id = child["region_id"]
        child_members = members.get(child_id, set())
        best_parent = ""
        best_overlap = 0
        for parent_id in parents_by_class.get(child["class_id"], []):
            overlap = len(child_members & members.get(parent_id, set()))
            if overlap > best_overlap or (overlap == best_overlap and parent_id < best_parent):
                best_parent = parent_id
                best_overlap = overlap
        parent_count = len(members.get(best_parent, set())) if best_parent else 0
        child_count = len(child_members)
        out.append(
            {
                "k10_region_id": child_id,
                "parent_k5_region_id": best_parent,
                "class_id": child["class_id"],
                "class_name": child["class_name"],
                "child_train_count": child_count,
                "parent_train_count": parent_count,
                "membership_overlap": best_overlap,
                "overlap_frac_of_child": best_overlap / child_count if child_count else 0.0,
                "overlap_frac_of_parent": best_overlap / parent_count if parent_count else 0.0,
            }
        )
    return out


def _summarize_numbers(rows: list[Mapping[str, str]], key: str) -> str:
    values = [_float(row, key) for row in rows if parse_float(row.get(key)) is not None]
    if not values:
        return "n/a"
    return f"min={min(values):.4g}, median={median(values):.4g}, max={max(values):.4g}"


def _write_report(
    path: Path,
    *,
    regions_dir: Path,
    deletion_dir: Path,
    regions: list[dict[str, str]],
    selected: list[dict[str, str]],
    validation: list[dict[str, str]],
    parent_rows: list[dict[str, object]],
    primary_k10: list[dict[str, object]],
    parent_child_csv: Path,
    primary_k10_csv: Path,
) -> None:
    selected_class_counts = Counter(row["class_id"] for row in selected)
    duplicated_classes = {key: value for key, value in selected_class_counts.items() if value > 1}
    validation_by_k = Counter(row["k_regions"] for row in validation)
    region_val_counts = Counter(row["region_id"] for row in validation)
    val_count_mismatches = [
        row["region_id"]
        for row in regions
        if _int(row, "validation_count") != int(region_val_counts.get(row["region_id"], 0))
    ]
    purity_unique = len({row.get("same_class_purity", "") for row in regions})
    purity_note = (
        "nontrivial global-neighbour purity"
        if purity_unique > 1
        else "constant purity; inspect implementation before training"
    )

    lines = [
        "# ADA Selected Region Audit",
        "",
        "Date: 2026-08-06",
        "",
        f"- regions dir: `{regions_dir}`",
        f"- deletion dir: `{deletion_dir}`",
        f"- parent-child overlap CSV: `{parent_child_csv}`",
        f"- one-per-class K=10 candidate CSV: `{primary_k10_csv}`",
        "",
        "## Validation Assignment Audit",
        "",
        f"- validation assignment rows: `{len(validation)}`",
        f"- validation rows by K: `{dict(sorted(validation_by_k.items()))}`",
        f"- validation count mismatched regions: `{len(val_count_mismatches)}`",
        "",
        "Expected fixed artifact behavior: long format with one row per `(validation sample, K)`.",
        "",
        "## Purity Audit",
        "",
        f"- same_class_purity unique values: `{purity_unique}`",
        f"- interpretation: `{purity_note}`",
        f"- same_class_purity range: `{_summarize_numbers(regions, 'same_class_purity')}`",
        f"- class_margin range: `{_summarize_numbers(regions, 'class_margin')}`",
        "",
        "The implementation computes purity by querying global train-neighbour labels around each region prototype, not by reading within-class cluster membership.",
        "",
        "## Current Selected Regions",
        "",
        f"- selected regions: `{len(selected)}`",
        f"- distinct selected classes: `{len(selected_class_counts)}`",
        f"- selected K distribution: `{dict(sorted(Counter(row['k_regions'] for row in selected).items()))}`",
        f"- classes with multiple selected regions: `{duplicated_classes}`",
        f"- train count range: `{_summarize_numbers(selected, 'train_count')}`",
        f"- validation count range: `{_summarize_numbers(selected, 'validation_count')}`",
        f"- purity range: `{_summarize_numbers(selected, 'same_class_purity')}`",
        f"- class margin range: `{_summarize_numbers(selected, 'class_margin')}`",
        "",
        "## Primary K=10 Recommendation",
        "",
        f"- eligible K=10 one-per-class candidates: `{len(primary_k10)}`",
        "",
        "Use K=10 as the primary causal experiment and K=5 as a resolution sensitivity analysis. Do not treat overlapping K=5 and K=10 regions as independent experimental units.",
        "",
        "## Parent-Child Overlap",
        "",
        f"- selected K=10 rows with parent K=5 mapping: `{len(parent_rows)}`",
        f"- K10 children wholly contained in best K5 parent: `{sum(float(row['overlap_frac_of_child']) >= 0.999 for row in parent_rows)}`",
        "",
        "## Missing Before Training",
        "",
        "- support percentile columns are not present in `regions.csv` yet.",
        "- local label entropy columns are not present in `regions.csv` yet.",
        "- count-preserving deletion controls are not built yet.",
        "- no deletion-probe or restoration classifier training has been launched.",
        "",
        "## First Selected Rows",
        "",
    ]
    for row in sorted(selected, key=_primary_sort_key)[:12]:
        lines.append(
            "- "
            f"`{row['region_id']}` K={row['k_regions']} class={row['class_id']} "
            f"train={row['train_count']} val={row['validation_count']} "
            f"purity={row['same_class_purity']} margin={row['class_margin']}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
