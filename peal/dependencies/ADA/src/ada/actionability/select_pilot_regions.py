from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ada.atlas.hashing import hash_rows, stable_hash


@dataclass(frozen=True)
class PilotSelectionConfig:
    enriched_regions_dir: Path
    output_dir: Path
    experiment_id: str = "e4a_in100_k10_causal_pilot_region_selection"
    sparse_interior_count: int = 4
    dense_interior_count: int = 2
    sparse_boundary_count: int = 2
    min_train_count: int = 100
    min_validation_count: int = 15
    sparse_support_pct_min: float = 0.75
    dense_support_pct_max: float = 0.25
    purity_min: float = 0.90
    entropy_max: float = 0.20
    margin_min: float = 0.0
    boundary_entropy_min: float = 0.40
    boundary_margin_max: float = 0.0
    one_region_per_class: bool = True
    excluded_wnids_path: Path | None = None
    excluded_wnids_sha256: str = ""
    overwrite: bool = False


def select_pilot_regions(config: PilotSelectionConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"pilot selection artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_metadata_path = Path(config.enriched_regions_dir) / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else {}
    rows = _read_csv(Path(config.enriched_regions_dir) / "regions_enriched.csv")
    rows = [row for row in rows if int(row["k_regions"]) == 10]
    if not rows:
        raise ValueError("pilot selection requires K=10 enriched regions")
    excluded_info = _load_excluded_wnids(config)
    excluded_wnids = set(excluded_info["wnids"])
    excluded_overlap_in_source = sorted({str(row.get("class_name", "")) for row in rows} & excluded_wnids)
    if excluded_wnids:
        rows = [row for row in rows if str(row.get("class_name", "")) not in excluded_wnids]
    if not rows:
        raise ValueError("pilot selection has no K=10 enriched regions after excluded-WNID filtering")

    selected: list[dict[str, object]] = []
    used_classes: set[int] = set()
    relaxations: list[dict[str, object]] = []

    selected.extend(
        _select_category(
            rows,
            used_classes,
            category="sparse_interior",
            count=int(config.sparse_interior_count),
            config=config,
        )
    )
    selected.extend(
        _select_category(
            rows,
            used_classes,
            category="dense_interior",
            count=int(config.dense_interior_count),
            config=config,
        )
    )
    selected.extend(
        _select_category(
            rows,
            used_classes,
            category="sparse_boundary",
            count=int(config.sparse_boundary_count),
            config=config,
        )
    )

    total_requested = int(config.sparse_interior_count + config.dense_interior_count + config.sparse_boundary_count)
    if len(selected) != total_requested:
        raise ValueError(f"selected {len(selected)} pilot regions, expected {total_requested}")

    for rank, row in enumerate(selected):
        row["pilot_rank"] = int(rank)
    selected_wnids = {str(row.get("class_name", "")) for row in selected}
    selected_excluded_overlap = sorted(selected_wnids & excluded_wnids)
    if selected_excluded_overlap:
        raise AssertionError(
            "selected regions overlap excluded development WNIDs: "
            + ",".join(selected_excluded_overlap)
        )

    for row in selected:
        if int(row.pop("_relaxed", 0)):
            relaxations.append(
                {
                    "region_id": row["region_id"],
                    "pilot_type": row["pilot_type"],
                    "reason": "strict category quota was underfilled; selected by prespecified fallback score",
                }
            )

    selected_hash = hash_rows(selected, prefix="pilot-regions")
    config_payload = asdict(config)
    config_payload["enriched_regions_dir"] = str(config.enriched_regions_dir)
    config_payload["output_dir"] = str(config.output_dir)
    config_payload["excluded_wnids_path"] = str(config.excluded_wnids_path or "")
    config_hash = stable_hash(config_payload, prefix="pilot-select-config")
    artifact_id = stable_hash(
        {
            "config_hash": config_hash,
            "selected_hash": selected_hash,
            "source_enrichment_artifact_id": source_metadata.get("artifact_id", ""),
        },
        prefix="pilot-regions-artifact",
    )

    _write_csv(output / "selected_pilot_regions.csv", selected)
    metadata = {
        "artifact_id": artifact_id,
        "experiment_id": config.experiment_id,
        "config": config_payload,
        "config_hash": config_hash,
        "selected_hash": selected_hash,
        "source_enrichment_artifact_id": source_metadata.get("artifact_id", ""),
        "source_region_artifact_id": source_metadata.get("source_region_artifact_id", ""),
        "source_validation_assignment_hash": source_metadata.get("source_validation_assignment_hash", ""),
        "excluded_wnids": {
            "path": str(config.excluded_wnids_path or ""),
            "sha256": excluded_info["sha256"],
            "expected_sha256": str(config.excluded_wnids_sha256 or ""),
            "count": len(excluded_wnids),
            "overlap_in_source_count": len(excluded_overlap_in_source),
            "selected_overlap_count": len(selected_excluded_overlap),
            "selected_overlap": selected_excluded_overlap,
        },
        "outputs": {"selected_pilot_regions_csv": str(output / "selected_pilot_regions.csv")},
        "summary": {
            "selected_regions": len(selected),
            "distinct_classes": len({int(row["class_id"]) for row in selected}),
            "by_type": _count_by(selected, "pilot_type"),
            "relaxation_count": len(relaxations),
        },
        "relaxations": relaxations,
        "leakage_declaration": (
            "Pilot regions are selected from train-only geometry and frozen validation-count metadata only. "
            "No validation errors or classifier predictions are used."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _select_category(
    rows: Sequence[Mapping[str, str]],
    used_classes: set[int],
    *,
    category: str,
    count: int,
    config: PilotSelectionConfig,
) -> list[dict[str, object]]:
    strict = [row for row in rows if _common_ok(row, config) and _category_ok(row, category, config)]
    strict_sorted = _sort_candidates(strict, category)
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    for row in strict_sorted:
        if len(selected) >= int(count):
            break
        if _blocked(row, used_classes, config):
            continue
        selected.append(_selected_row(row, category, relaxed=False, score=_score(row, category)))
        selected_ids.add(str(row["region_id"]))
        used_classes.add(int(row["class_id"]))

    if len(selected) >= int(count):
        return selected

    fallback = [row for row in rows if _common_minimal_ok(row, config) and str(row["region_id"]) not in selected_ids]
    fallback_sorted = _sort_candidates(fallback, category)
    for row in fallback_sorted:
        if len(selected) >= int(count):
            break
        if _blocked(row, used_classes, config):
            continue
        selected.append(_selected_row(row, category, relaxed=True, score=_score(row, category)))
        used_classes.add(int(row["class_id"]))
    return selected


def _common_ok(row: Mapping[str, str], config: PilotSelectionConfig) -> bool:
    return (
        int(row["train_count"]) >= int(config.min_train_count)
        and int(row["validation_count"]) >= int(config.min_validation_count)
    )


def _common_minimal_ok(row: Mapping[str, str], config: PilotSelectionConfig) -> bool:
    return int(row["validation_count"]) >= int(config.min_validation_count)


def _category_ok(row: Mapping[str, str], category: str, config: PilotSelectionConfig) -> bool:
    support = _float(row["member_support_pct_median"])
    purity = _float(row["global_neighbor_purity"])
    entropy = _float(row["local_label_entropy"])
    margin = _float(row["robust_class_margin"])
    interior = purity >= float(config.purity_min) and entropy <= float(config.entropy_max) and margin > float(config.margin_min)
    if category == "sparse_interior":
        return support >= float(config.sparse_support_pct_min) and interior
    if category == "dense_interior":
        return support <= float(config.dense_support_pct_max) and interior
    if category == "sparse_boundary":
        return support >= float(config.sparse_support_pct_min) and (
            margin <= float(config.boundary_margin_max) or entropy >= float(config.boundary_entropy_min)
        )
    raise ValueError(f"unknown pilot category: {category}")


def _sort_candidates(rows: Sequence[Mapping[str, str]], category: str) -> list[Mapping[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            -_score(row, category),
            -int(row["validation_count"]),
            -int(row["train_count"]),
            str(row["region_id"]),
        ),
    )


def _score(row: Mapping[str, str], category: str) -> float:
    support = _float(row["member_support_pct_median"])
    purity = _float(row["global_neighbor_purity"])
    entropy = _float(row["local_label_entropy"])
    margin = _float(row["robust_class_margin"])
    if category == "dense_interior":
        return (1.0 - support) + purity - entropy + max(margin, -1.0)
    if category == "sparse_boundary":
        return support + entropy - margin
    return support + purity - entropy + max(margin, -1.0)


def _blocked(row: Mapping[str, str], used_classes: set[int], config: PilotSelectionConfig) -> bool:
    return bool(config.one_region_per_class) and int(row["class_id"]) in used_classes


def _load_excluded_wnids(config: PilotSelectionConfig) -> dict[str, object]:
    path = config.excluded_wnids_path
    if path is None or not str(path):
        return {"wnids": [], "sha256": ""}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"excluded WNID file does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = str(config.excluded_wnids_sha256 or "")
    if expected and digest != expected:
        raise ValueError(f"excluded WNID SHA256 mismatch for {path}: expected {expected}, got {digest}")
    wnids = []
    seen = set()
    for raw in path.read_text().splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        if item in seen:
            raise ValueError(f"duplicate excluded WNID {item!r} in {path}")
        seen.add(item)
        wnids.append(item)
    return {"wnids": sorted(wnids), "sha256": digest}


def _selected_row(row: Mapping[str, str], category: str, *, relaxed: bool, score: float) -> dict[str, object]:
    item = dict(row)
    item["pilot_type"] = category
    item["selection_score"] = float(score)
    item["selection_relaxed"] = int(relaxed)
    item["_relaxed"] = int(relaxed)
    return item


def _count_by(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _float(value: object) -> float:
    return float(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in dict(row).items() if key in fieldnames})
