from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ada.actionability.region_filters import EligibilityThresholds, select_eligible_regions
from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.hashing import hash_rows, stable_hash


@dataclass(frozen=True)
class DeletionBuildConfig:
    train_cache: Path
    regions_dir: Path
    output_dir: Path
    retention_levels: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.0)
    seed: int = 0
    max_regions: int | None = 30
    overwrite: bool = False
    thresholds: EligibilityThresholds = EligibilityThresholds()


def build_deletion_artifact(config: DeletionBuildConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"deletion artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest_csv(Path(config.train_cache) / "manifest.csv")
    sample_ids = [row.sample_id for row in train_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("train cache manifest contains duplicate sample IDs")
    train_by_id = {row.sample_id: row for row in train_rows}
    train_class_counts = Counter(int(row.class_id) for row in train_rows)
    original_manifest_hash = hash_rows((asdict(row) for row in train_rows), prefix="manifest")

    regions = _read_csv(Path(config.regions_dir) / "regions.csv")
    membership = _read_csv(Path(config.regions_dir) / "region_membership.csv")
    region_metadata_path = Path(config.regions_dir) / "metadata.json"
    region_metadata = json.loads(region_metadata_path.read_text()) if region_metadata_path.exists() else {}
    selected_regions = select_eligible_regions(regions, config.thresholds, max_regions=config.max_regions)
    if not selected_regions:
        raise ValueError("no eligible regions found for deletion manifests")

    members_by_region: dict[str, list[str]] = defaultdict(list)
    for row in membership:
        members_by_region[str(row["region_id"])].append(str(row["sample_id"]))
    for region_id, ids in members_by_region.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate sample IDs in region membership: {region_id}")
        missing = sorted(set(ids).difference(train_by_id))
        if missing:
            raise ValueError(f"region {region_id} contains sample IDs missing from train manifest")

    retention_levels = _normalize_retention_levels(config.retention_levels)
    manifest_rows: list[dict[str, object]] = []
    for region in selected_regions:
        region_id = str(region["region_id"])
        region_ids = sorted(members_by_region.get(region_id, []))
        if not region_ids:
            raise ValueError(f"selected region has no member rows: {region_id}")
        for level in retention_levels:
            retain_count = _retained_count(len(region_ids), level)
            retained_region_ids, deleted_region_ids = _split_region_ids(
                region_ids,
                retain_count=retain_count,
                seed=_derived_seed(config.seed, region_id, level),
            )
            retained_set = set(retained_region_ids)
            deleted_set = set(deleted_region_ids)
            if retained_set.intersection(deleted_set):
                raise AssertionError("retained/deleted region sets overlap")
            full_retained_ids = [sid for sid in sample_ids if sid not in deleted_set]
            class_counts = Counter(int(train_by_id[sid].class_id) for sid in full_retained_ids)
            retain_label = _retention_label(level)
            out_dir = output / region_id / retain_label
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_lines(out_dir / "retained_region_sample_ids.txt", retained_region_ids)
            _write_lines(out_dir / "deleted_region_sample_ids.txt", deleted_region_ids)
            payload = {
                "manifest_type": "ada_controlled_regional_deletion",
                "region_id": region_id,
                "retention_level": float(level),
                "retention_label": retain_label,
                "seed": int(config.seed),
                "split_seed": _derived_seed(config.seed, region_id, level),
                "train_cache": str(Path(config.train_cache)),
                "original_train_count": len(train_rows),
                "original_manifest_hash": original_manifest_hash,
                "region_artifact_id": region_metadata.get("artifact_id", ""),
                "region_hash": region_metadata.get("region_hash", ""),
                "region_definition_hash": stable_hash(dict(region), prefix="region-row"),
                "region_train_count": len(region_ids),
                "retained_region_count": len(retained_region_ids),
                "deleted_region_count": len(deleted_region_ids),
                "retained_total_count": len(full_retained_ids),
                "deleted_total_count": len(deleted_region_ids),
                "retained_sample_ids_hash": hash_rows(({"sample_id": sid} for sid in full_retained_ids), prefix="retained"),
                "deleted_sample_ids_hash": hash_rows(({"sample_id": sid} for sid in deleted_region_ids), prefix="deleted"),
                "retained_region_sample_ids_path": str(out_dir / "retained_region_sample_ids.txt"),
                "deleted_region_sample_ids_path": str(out_dir / "deleted_region_sample_ids.txt"),
                "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
                "duplicate_checks": {
                    "train_sample_ids_unique": len(sample_ids) == len(set(sample_ids)),
                    "region_sample_ids_unique": len(region_ids) == len(set(region_ids)),
                    "retained_region_sample_ids_unique": len(retained_region_ids) == len(set(retained_region_ids)),
                    "deleted_region_sample_ids_unique": len(deleted_region_ids) == len(set(deleted_region_ids)),
                },
                "leakage_declaration": (
                    "Deletion manifests use train region memberships only. "
                    "Validation assignments and labels are not used to choose retained samples."
                ),
            }
            payload["manifest_id"] = stable_hash(payload, prefix="deletion")
            (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
            (out_dir / "COMPLETED").write_text("ok\n")
            manifest_rows.append(
                {
                    "manifest_id": payload["manifest_id"],
                    "region_id": region_id,
                    "retention_level": float(level),
                    "retention_label": retain_label,
                    "class_id": int(region["class_id"]),
                    "class_name": str(region["class_name"]),
                    "region_train_count": len(region_ids),
                    "retained_region_count": len(retained_region_ids),
                    "deleted_region_count": len(deleted_region_ids),
                    "retained_total_count": len(full_retained_ids),
                    "deleted_total_count": len(deleted_region_ids),
                    "manifest_json": str(out_dir / "manifest.json"),
                }
            )

    summary = {
        "artifact_id": stable_hash(
            {
                "regions_dir": str(config.regions_dir),
                "output_dir": str(config.output_dir),
                "retention_levels": list(retention_levels),
                "selected_region_ids": [row["region_id"] for row in selected_regions],
                "manifest_rows_hash": hash_rows(manifest_rows, prefix="deletion-index"),
            },
            prefix="deletion-artifact",
        ),
        "experiment_id": "e4a_in100_controlled_regional_deletion_manifests",
        "train_cache": str(Path(config.train_cache)),
        "regions_dir": str(Path(config.regions_dir)),
        "output_dir": str(output),
        "original_manifest_hash": original_manifest_hash,
        "region_artifact_id": region_metadata.get("artifact_id", ""),
        "region_hash": region_metadata.get("region_hash", ""),
        "retention_levels": list(retention_levels),
        "selected_region_count": len(selected_regions),
        "manifest_count": len(manifest_rows),
        "max_regions": config.max_regions,
        "thresholds": asdict(config.thresholds),
        "leakage_declaration": (
            "Deletion split construction uses only the train manifest and train-built region memberships. "
            "Validation assignments are ignored except for region eligibility metadata already recorded in regions.csv."
        ),
    }
    _write_csv(output / "deletion_manifests.csv", manifest_rows)
    (output / "metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return summary


def _normalize_retention_levels(levels: Sequence[float]) -> tuple[float, ...]:
    parsed = tuple(sorted({float(level) for level in levels}, reverse=True))
    if any(level < 0.0 or level > 1.0 for level in parsed):
        raise ValueError("retention levels must be between 0 and 1")
    return parsed


def _retained_count(total: int, level: float) -> int:
    if float(level) >= 1.0:
        return int(total)
    if float(level) <= 0.0:
        return 0
    return int(math.floor(int(total) * float(level)))


def _split_region_ids(region_ids: Sequence[str], *, retain_count: int, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(str(sid) for sid in region_ids)
    rng = np.random.default_rng(int(seed))
    order = np.asarray(ids, dtype=object)
    rng.shuffle(order)
    retained = sorted(str(sid) for sid in order[: int(retain_count)].tolist())
    deleted = sorted(str(sid) for sid in order[int(retain_count) :].tolist())
    return retained, deleted


def _derived_seed(seed: int, region_id: str, level: float) -> int:
    digest = stable_hash({"seed": int(seed), "region_id": str(region_id), "retention_level": float(level)})
    return int(digest[:8], 16)


def _retention_label(level: float) -> str:
    return f"retain_{int(round(float(level) * 100)):03d}"


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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
        for row in rows:
            writer.writerow(dict(row))


def _write_lines(path: Path, values: Sequence[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values))
