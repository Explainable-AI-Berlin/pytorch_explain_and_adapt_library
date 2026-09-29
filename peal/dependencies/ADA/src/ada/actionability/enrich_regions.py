from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ada.actionability.region_assignment import l2_normalize
from ada.actionability.regions import load_embedding_bank
from ada.atlas.hashing import hash_rows, stable_hash


@dataclass(frozen=True)
class RegionEnrichmentConfig:
    train_cache: Path
    regions_dir: Path
    output_dir: Path
    experiment_id: str = "e4a_in100_k10_region_enrichment"
    primary_k: int = 10
    parent_k: int = 5
    support_k: int = 50
    global_neighbor_k: int = 50
    margin_core_k: int = 10
    mmap: bool = True
    overwrite: bool = False


def enrich_regions(config: RegionEnrichmentConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"enrichment artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train = load_embedding_bank(config.train_cache, mmap=config.mmap)
    train_embeddings = l2_normalize(np.asarray(train.embeddings, dtype=np.float32))
    train_labels = train.labels
    sample_to_index = {row.sample_id: idx for idx, row in enumerate(train.rows)}
    sample_to_class = {row.sample_id: int(row.class_id) for row in train.rows}

    regions_dir = Path(config.regions_dir)
    source_metadata_path = regions_dir / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else {}
    source_regions = _read_csv(regions_dir / "regions.csv")
    source_membership = _read_csv(regions_dir / "region_membership.csv")
    source_val = _read_csv(regions_dir / "validation_assignments.csv")
    prototypes = np.load(regions_dir / "prototypes.npy", mmap_mode="r" if config.mmap else None)

    primary_k = int(config.primary_k)
    parent_k = int(config.parent_k)
    primary_regions = [dict(row) for row in source_regions if int(row["k_regions"]) == primary_k]
    primary_ids = {str(row["region_id"]) for row in primary_regions}
    primary_membership = [dict(row) for row in source_membership if int(row["k_regions"]) == primary_k]
    primary_val = [dict(row) for row in source_val if int(row["k_regions"]) == primary_k]

    if not primary_regions:
        raise ValueError(f"no regions found for primary_k={primary_k}")
    _assert_membership_accounting(primary_membership, train.sample_ids)
    _assert_validation_assignments(primary_val, primary_ids)

    members_by_region: dict[str, list[str]] = defaultdict(list)
    region_by_sample_primary: dict[str, str] = {}
    for row in primary_membership:
        region_id = str(row["region_id"])
        sample_id = str(row["sample_id"])
        members_by_region[region_id].append(sample_id)
        region_by_sample_primary[sample_id] = region_id

    members_by_parent: dict[str, list[str]] = defaultdict(list)
    parent_by_sample: dict[str, str] = {}
    for row in source_membership:
        if int(row["k_regions"]) != parent_k:
            continue
        region_id = str(row["region_id"])
        sample_id = str(row["sample_id"])
        members_by_parent[region_id].append(sample_id)
        parent_by_sample[sample_id] = region_id

    class_counts = Counter(sample_to_class[sid] for sid in sample_to_index)
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(train.rows):
        class_to_indices[int(row.class_id)].append(idx)

    train_support = _leave_one_out_support_by_class(
        train_embeddings=train_embeddings,
        class_to_indices=class_to_indices,
        support_k=int(config.support_k),
    )
    support_by_sample = {train.rows[idx].sample_id: train_support[idx] for idx in range(len(train.rows))}

    enriched_rows: list[dict[str, object]] = []
    for row in primary_regions:
        region_id = str(row["region_id"])
        class_id = int(row["class_id"])
        member_ids = sorted(members_by_region.get(region_id, []))
        if not member_ids:
            raise ValueError(f"primary region has no members: {region_id}")
        member_indices = [sample_to_index[sid] for sid in member_ids]
        if any(sample_to_class[sid] != class_id for sid in member_ids):
            raise AssertionError(f"region {region_id} contains mixed class membership")

        member_pct = np.asarray([support_by_sample[sid]["percentile"] for sid in member_ids], dtype=np.float64)
        proto_index = int(row["prototype_index"])
        proto = l2_normalize(np.asarray(prototypes[proto_index], dtype=np.float32).reshape(1, -1))[0]
        global_metrics = _prototype_global_metrics(
            proto,
            class_id=class_id,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            global_neighbor_k=int(config.global_neighbor_k),
            margin_core_k=int(config.margin_core_k),
        )
        prototype_support_distance = _kth_distance(
            1.0 - (train_embeddings[class_to_indices[class_id]] @ proto),
            k=min(int(config.support_k), len(class_to_indices[class_id])),
        )
        prototype_support_pct = _percentile_from_sorted(
            train_support["class_sorted_distances"][class_id],
            prototype_support_distance,
        )
        parent_region_id, parent_overlap = _best_parent(member_ids, parent_by_sample)
        enriched = dict(row)
        enriched.update(
            {
                "region_mass_within_class": float(len(member_ids) / max(class_counts[class_id], 1)),
                "member_support_pct_median": float(np.median(member_pct)),
                "member_support_pct_q75": float(np.quantile(member_pct, 0.75)),
                "member_support_pct_q90": float(np.quantile(member_pct, 0.90)),
                "prototype_support_pct": float(prototype_support_pct),
                "prototype_support_distance": float(prototype_support_distance),
                "global_neighbor_purity": float(global_metrics["global_neighbor_purity"]),
                "local_label_entropy": float(global_metrics["local_label_entropy"]),
                "robust_class_margin": float(global_metrics["robust_class_margin"]),
                "robust_same_class_core_distance": float(global_metrics["robust_same_class_core_distance"]),
                "robust_competing_core_distance": float(global_metrics["robust_competing_core_distance"]),
                "parent_k5_region_id": parent_region_id,
                "parent_overlap_fraction": float(parent_overlap),
            }
        )
        if int(enriched["train_count"]) != len(member_indices):
            raise AssertionError(f"train_count mismatch for {region_id}")
        enriched_rows.append(enriched)

    train_support_rows = [
        {
            "sample_id": row.sample_id,
            "class_id": int(row.class_id),
            "class_name": row.class_name,
            "loo_support_k": int(config.support_k),
            "loo_same_class_support_distance": float(train_support[idx]["distance"]),
            "loo_same_class_support_percentile": float(train_support[idx]["percentile"]),
            "region_id_k10": region_by_sample_primary[row.sample_id],
        }
        for idx, row in enumerate(train.rows)
    ]

    enriched_hash = hash_rows(enriched_rows, prefix="region-enrichment")
    support_hash = hash_rows(train_support_rows, prefix="train-support")
    validation_hash = hash_rows(primary_val, prefix="val-assign-k10")
    config_payload = {
        "train_cache": str(Path(config.train_cache)),
        "regions_dir": str(Path(config.regions_dir)),
        "source_region_artifact_id": source_metadata.get("artifact_id", ""),
        "source_region_hash": source_metadata.get("region_hash", ""),
        "source_membership_hash": source_metadata.get("membership_hash", ""),
        "source_validation_assignment_hash": source_metadata.get("validation_assignment_hash", ""),
        "experiment_id": config.experiment_id,
        "primary_k": int(config.primary_k),
        "parent_k": int(config.parent_k),
        "support_k": int(config.support_k),
        "global_neighbor_k": int(config.global_neighbor_k),
        "margin_core_k": int(config.margin_core_k),
    }
    config_hash = stable_hash(config_payload, prefix="region-enrich-config")
    artifact_id = stable_hash(
        {
            "config_hash": config_hash,
            "enriched_hash": enriched_hash,
            "support_hash": support_hash,
            "validation_hash": validation_hash,
        },
        prefix="region-enrich-artifact",
    )

    _write_csv(output / "regions_enriched.csv", enriched_rows)
    _write_csv(output / "region_membership_k10.csv", primary_membership)
    _write_csv(output / "validation_assignments_k10.csv", primary_val)
    _write_csv(output / "train_member_support.csv", train_support_rows)
    metadata = {
        "artifact_id": artifact_id,
        "experiment_id": config.experiment_id,
        "config": config_payload,
        "config_hash": config_hash,
        "enriched_region_hash": enriched_hash,
        "train_support_hash": support_hash,
        "validation_assignment_hash": validation_hash,
        "source_region_artifact_id": source_metadata.get("artifact_id", ""),
        "source_region_hash": source_metadata.get("region_hash", ""),
        "source_membership_hash": source_metadata.get("membership_hash", ""),
        "source_validation_assignment_hash": source_metadata.get("validation_assignment_hash", ""),
        "outputs": {
            "regions_enriched_csv": str(output / "regions_enriched.csv"),
            "region_membership_k10_csv": str(output / "region_membership_k10.csv"),
            "validation_assignments_k10_csv": str(output / "validation_assignments_k10.csv"),
            "train_member_support_csv": str(output / "train_member_support.csv"),
        },
        "summary": {
            "primary_k": int(config.primary_k),
            "regions": len(enriched_rows),
            "membership_rows": len(primary_membership),
            "validation_assignment_rows": len(primary_val),
            "distinct_classes": len({int(row["class_id"]) for row in enriched_rows}),
            "support_k": int(config.support_k),
            "global_neighbor_k": int(config.global_neighbor_k),
            "margin_core_k": int(config.margin_core_k),
            "eligible_primary_regions": int(sum(int(row.get("eligible_primary", 0)) for row in enriched_rows)),
        },
        "leakage_declaration": (
            "Enrichment uses the original train-only region memberships, train embeddings, "
            "train labels, and frozen validation assignments. It does not use validation errors "
            "or classifier predictions."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _leave_one_out_support_by_class(
    *,
    train_embeddings: np.ndarray,
    class_to_indices: Mapping[int, Sequence[int]],
    support_k: int,
) -> dict[object, object]:
    per_index: dict[int, dict[str, float]] = {}
    class_sorted_distances: dict[int, np.ndarray] = {}
    for class_id, indices_raw in sorted(class_to_indices.items()):
        indices = np.asarray(list(indices_raw), dtype=np.int64)
        x = np.asarray(train_embeddings[indices], dtype=np.float32)
        if len(indices) < 2:
            raise ValueError(f"class_id={class_id} has fewer than two examples")
        sims = x @ x.T
        np.fill_diagonal(sims, -np.inf)
        distances = 1.0 - sims
        k_eff = min(int(support_k), len(indices) - 1)
        kth = np.partition(distances, kth=k_eff - 1, axis=1)[:, k_eff - 1]
        sorted_kth = np.sort(kth.astype(np.float64, copy=False))
        class_sorted_distances[int(class_id)] = sorted_kth
        for local_pos, idx in enumerate(indices.tolist()):
            dist = float(kth[local_pos])
            per_index[int(idx)] = {
                "distance": dist,
                "percentile": _percentile_from_sorted(sorted_kth, dist),
            }
    per_index["class_sorted_distances"] = class_sorted_distances
    return per_index


def _prototype_global_metrics(
    prototype: np.ndarray,
    *,
    class_id: int,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    global_neighbor_k: int,
    margin_core_k: int,
) -> dict[str, float]:
    sims = np.asarray(train_embeddings, dtype=np.float32) @ np.asarray(prototype, dtype=np.float32)
    distances = 1.0 - sims
    k_global = min(int(global_neighbor_k), len(distances))
    top_idx = np.argpartition(distances, kth=k_global - 1)[:k_global]
    labels = np.asarray(train_labels, dtype=np.int64)
    top_labels = labels[top_idx]
    counts = Counter(int(label) for label in top_labels.tolist())
    purity = counts.get(int(class_id), 0) / max(k_global, 1)
    entropy = _entropy([count / max(k_global, 1) for count in counts.values()])

    same_dist = distances[labels == int(class_id)]
    other_dist = distances[labels != int(class_id)]
    same_core = _kth_distance(same_dist, k=min(int(margin_core_k), same_dist.size))
    other_core = _kth_distance(other_dist, k=min(int(margin_core_k), other_dist.size))
    return {
        "global_neighbor_purity": float(purity),
        "local_label_entropy": float(entropy),
        "robust_same_class_core_distance": float(same_core),
        "robust_competing_core_distance": float(other_core),
        "robust_class_margin": float(other_core - same_core),
    }


def _kth_distance(distances: np.ndarray, *, k: int) -> float:
    arr = np.asarray(distances, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return math.nan
    k_eff = min(max(int(k), 1), arr.size)
    return float(np.partition(arr, kth=k_eff - 1)[k_eff - 1])


def _percentile_from_sorted(sorted_values: np.ndarray, value: float) -> float:
    arr = np.asarray(sorted_values, dtype=np.float64)
    if arr.size == 0:
        return math.nan
    return float(np.searchsorted(arr, float(value), side="right") / arr.size)


def _entropy(probs: Sequence[float]) -> float:
    total = 0.0
    for p in probs:
        p = float(p)
        if p > 0.0:
            total -= p * math.log(p)
    return total


def _best_parent(member_ids: Sequence[str], parent_by_sample: Mapping[str, str]) -> tuple[str, float]:
    counts = Counter(parent_by_sample.get(sample_id, "") for sample_id in member_ids)
    counts.pop("", None)
    if not counts:
        return "", 0.0
    parent_id, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return str(parent_id), float(count / max(len(member_ids), 1))


def _assert_membership_accounting(membership_rows: Sequence[Mapping[str, object]], sample_ids: Sequence[str]) -> None:
    seen = [str(row["sample_id"]) for row in membership_rows]
    if len(seen) != len(set(seen)):
        raise AssertionError("primary-K membership must contain each training sample exactly once")
    missing = set(sample_ids).difference(seen)
    extra = set(seen).difference(sample_ids)
    if missing or extra:
        raise AssertionError(f"primary-K membership mismatch: missing={len(missing)} extra={len(extra)}")


def _assert_validation_assignments(validation_rows: Sequence[Mapping[str, object]], region_ids: set[str]) -> None:
    seen_samples = [str(row["sample_id"]) for row in validation_rows]
    if len(seen_samples) != len(set(seen_samples)):
        raise AssertionError("primary-K validation assignments must contain each validation sample exactly once")
    unknown = {str(row["region_id"]) for row in validation_rows}.difference(region_ids)
    if unknown:
        raise AssertionError(f"validation assignments reference unknown primary regions: {sorted(unknown)[:5]}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
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
