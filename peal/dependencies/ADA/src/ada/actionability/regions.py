from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ada.actionability.region_assignment import assign_rows_to_regions, class_residuals, l2_normalize, normalize_vector
from ada.actionability.region_filters import EligibilityThresholds, annotate_eligibility
from ada.atlas.data.manifests import ManifestRow, load_manifest_csv
from ada.atlas.hashing import file_sha1, hash_rows, stable_hash


@dataclass(frozen=True)
class CachedEmbeddingBank:
    cache_dir: Path
    embeddings: np.ndarray
    rows: list[ManifestRow]
    metadata: dict[str, object]

    @property
    def sample_ids(self) -> list[str]:
        return [row.sample_id for row in self.rows]

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([int(row.class_id) for row in self.rows], dtype=np.int64)


@dataclass(frozen=True)
class RegionBuildConfig:
    train_cache: Path
    output_dir: Path
    experiment_id: str = "e4a_in100_controlled_regional_deletion_regions"
    validation_cache: Path | None = None
    k_values: tuple[int, ...] = (5, 10)
    seed: int = 0
    residual_mode: str = "tangent"
    max_iter: int = 50
    tol: float = 1.0e-5
    local_purity_k: int = 50
    duplicate_distance_threshold: float = 1.0e-4
    exemplar_count: int = 8
    metric_batch_size: int = 128
    mmap: bool = True
    overwrite: bool = False
    thresholds: EligibilityThresholds = EligibilityThresholds()


def load_embedding_bank(cache_dir: str | Path, *, mmap: bool = True) -> CachedEmbeddingBank:
    path = Path(cache_dir)
    for name in ("embeddings.npy", "manifest.csv", "metadata.json"):
        if not (path / name).exists():
            raise FileNotFoundError(f"missing {name} in cache: {path}")
    mmap_mode = "r" if mmap else None
    embeddings = np.load(path / "embeddings.npy", mmap_mode=mmap_mode).astype("float32", copy=False)
    rows = load_manifest_csv(path / "manifest.csv")
    metadata = json.loads((path / "metadata.json").read_text())
    if int(embeddings.shape[0]) != len(rows):
        raise ValueError(f"cache row mismatch for {path}: embeddings={embeddings.shape[0]} manifest={len(rows)}")
    sample_ids = [row.sample_id for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"cache contains duplicate sample IDs: {path}")
    return CachedEmbeddingBank(cache_dir=path, embeddings=embeddings, rows=rows, metadata=metadata)


def build_region_artifact(config: RegionBuildConfig) -> dict[str, object]:
    if not config.k_values:
        raise ValueError("k_values must not be empty")
    k_values = tuple(sorted({int(k) for k in config.k_values}))
    if min(k_values) < 1:
        raise ValueError("k_values must be positive")
    if config.residual_mode not in {"raw", "centered", "tangent"}:
        raise ValueError("residual_mode must be one of raw, centered, tangent")

    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"region artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train = load_embedding_bank(config.train_cache, mmap=config.mmap)
    validation = load_embedding_bank(config.validation_cache, mmap=config.mmap) if config.validation_cache is not None else None
    if validation is not None:
        overlap = set(train.sample_ids).intersection(validation.sample_ids)
        if overlap:
            raise ValueError(f"train/validation sample_id overlap is forbidden; found {len(overlap)} overlaps")
        if int(train.embeddings.shape[1]) != int(validation.embeddings.shape[1]):
            raise ValueError("train and validation embedding dimensions differ")

    train_embeddings = l2_normalize(np.asarray(train.embeddings, dtype=np.float32))
    train_labels = train.labels
    class_indices = _group_indices_by_class(train.rows)
    class_prototypes: dict[int, np.ndarray] = {}
    for class_id, indices in class_indices.items():
        class_prototypes[int(class_id)] = normalize_vector(train_embeddings[indices].mean(axis=0))

    raw_region_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    raw_prototypes: list[np.ndarray] = []
    residual_prototypes: list[np.ndarray] = []

    for k_regions in k_values:
        for class_id, indices in sorted(class_indices.items()):
            if len(indices) < int(k_regions):
                raise ValueError(f"class_id={class_id} has only {len(indices)} rows, fewer than K={k_regions}")
            class_rows = [train.rows[idx] for idx in indices]
            local = class_residuals(train_embeddings[indices], class_prototypes[int(class_id)], mode=config.residual_mode)
            labels = spherical_kmeans(
                local,
                int(k_regions),
                seed=_derived_seed(config.seed, int(k_regions), int(class_id)),
                max_iter=int(config.max_iter),
                tol=float(config.tol),
            )
            clusters = _sorted_clusters(labels, class_rows)
            for cluster_index, local_member_positions in enumerate(clusters):
                member_indices = [indices[pos] for pos in local_member_positions]
                member_ids = [train.rows[idx].sample_id for idx in member_indices]
                member_ids_sorted = sorted(member_ids)
                region_id = stable_hash(
                    {
                        "dataset": train.metadata.get("dataset"),
                        "split": train.metadata.get("split"),
                        "train_cache_id": train.metadata.get("cache_id"),
                        "train_manifest_hash": train.metadata.get("cached_manifest_hash"),
                        "k_regions": int(k_regions),
                        "class_id": int(class_id),
                        "member_ids_hash": hash_rows(({"sample_id": sid} for sid in member_ids_sorted), prefix="members"),
                    },
                    prefix="region",
                )
                raw_proto = normalize_vector(train_embeddings[member_indices].mean(axis=0))
                residual_proto = normalize_vector(local[local_member_positions].mean(axis=0))
                prototype_index = len(raw_prototypes)
                raw_prototypes.append(raw_proto.astype("float32", copy=False))
                residual_prototypes.append(residual_proto.astype("float32", copy=False))
                class_name = train.rows[member_indices[0]].class_name
                duplicate_fraction = _duplicate_fraction(
                    train_embeddings[member_indices],
                    threshold=float(config.duplicate_distance_threshold),
                )
                exemplar_ids = _nearest_member_ids(
                    raw_proto,
                    train_embeddings[member_indices],
                    [train.rows[idx] for idx in member_indices],
                    count=int(config.exemplar_count),
                )
                raw_region_rows.append(
                    {
                        "region_id": region_id,
                        "k_regions": int(k_regions),
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "cluster_index": int(cluster_index),
                        "prototype_index": int(prototype_index),
                        "train_count": int(len(member_indices)),
                        "validation_count": 0,
                        "same_class_purity": "",
                        "nearest_same_class_distance": "",
                        "nearest_competing_class_distance": "",
                        "class_margin": "",
                        "duplicate_fraction": float(duplicate_fraction),
                        "visual_exemplar_ids": ";".join(exemplar_ids),
                    }
                )
                for idx in member_indices:
                    row = train.rows[idx]
                    membership_rows.append(
                        {
                            "region_id": region_id,
                            "sample_id": row.sample_id,
                            "relative_path": row.relative_path,
                            "class_id": int(row.class_id),
                            "class_name": row.class_name,
                            "k_regions": int(k_regions),
                            "cluster_index": int(cluster_index),
                        }
                    )

    prototypes_arr = np.stack(raw_prototypes).astype("float32", copy=False)
    residual_prototypes_arr = np.stack(residual_prototypes).astype("float32", copy=False)
    region_rows = _add_region_geometry_metrics(
        raw_region_rows,
        prototypes=prototypes_arr,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        train_rows=train.rows,
        local_purity_k=int(config.local_purity_k),
        batch_size=int(config.metric_batch_size),
    )

    validation_rows: list[dict[str, object]] = []
    if validation is not None:
        validation_assignments = assign_rows_to_regions(
            embeddings=np.asarray(validation.embeddings, dtype=np.float32),
            rows=validation.rows,
            region_rows=region_rows,
            class_prototypes=class_prototypes,
            residual_prototypes=residual_prototypes_arr,
            residual_mode=config.residual_mode,
        )
        val_counts = Counter(item.region_id for item in validation_assignments)
        for row in region_rows:
            row["validation_count"] = int(val_counts.get(str(row["region_id"]), 0))
        validation_rows = [asdict(item) for item in validation_assignments]
        _assert_validation_assignments(
            validation_rows,
            region_rows,
            n_validation=len(validation.rows),
            k_values=config.k_values,
        )

    region_rows = annotate_eligibility(region_rows, config.thresholds)
    region_hash = hash_rows(region_rows, prefix="regions")
    membership_hash = hash_rows(membership_rows, prefix="membership")
    validation_hash = hash_rows(validation_rows, prefix="val-assign") if validation_rows else ""
    config_payload = _config_payload(config, train, validation)
    config_hash = stable_hash(config_payload, prefix="region-config")
    artifact_id = stable_hash(
        {
            "config_hash": config_hash,
            "regions_hash": region_hash,
            "membership_hash": membership_hash,
            "validation_hash": validation_hash,
        },
        prefix="regions-artifact",
    )

    _write_csv(output / "regions.csv", region_rows)
    _write_csv(output / "region_membership.csv", membership_rows)
    if validation_rows:
        _write_csv(output / "validation_assignments.csv", validation_rows)
    np.save(output / "prototypes.npy", prototypes_arr)
    np.save(output / "residual_prototypes.npy", residual_prototypes_arr)
    class_proto_arr, class_proto_ids = _pack_class_prototypes(class_prototypes)
    np.save(output / "class_prototypes.npy", class_proto_arr)
    _write_json(output / "class_prototypes.json", {"class_ids": class_proto_ids})
    metadata = {
        "artifact_id": artifact_id,
        "experiment_id": config.experiment_id,
        "leakage_declaration": (
            "Region prototypes and memberships are fit using train embeddings only. "
            "Validation embeddings are assigned after fitting and do not affect boundaries."
        ),
        "config": config_payload,
        "config_hash": config_hash,
        "region_hash": region_hash,
        "membership_hash": membership_hash,
        "validation_assignment_hash": validation_hash,
        "outputs": {
            "regions_csv": str(output / "regions.csv"),
            "region_membership_csv": str(output / "region_membership.csv"),
            "validation_assignments_csv": str(output / "validation_assignments.csv") if validation_rows else "",
            "prototypes_npy": str(output / "prototypes.npy"),
            "residual_prototypes_npy": str(output / "residual_prototypes.npy"),
            "class_prototypes_npy": str(output / "class_prototypes.npy"),
        },
        "summary": _region_summary(region_rows, membership_rows, validation_rows),
    }
    _write_json(output / "metadata.json", metadata)
    completed.write_text("ok\n")
    return metadata


def spherical_kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    seed: int,
    max_iter: int = 50,
    tol: float = 1.0e-5,
) -> np.ndarray:
    x = l2_normalize(np.asarray(vectors, dtype=np.float32))
    n = int(x.shape[0])
    if k < 1:
        raise ValueError("k must be positive")
    if n < k:
        raise ValueError(f"cannot cluster {n} rows into {k} regions")
    rng = np.random.default_rng(int(seed))
    initial = rng.choice(n, size=int(k), replace=False)
    centroids = l2_normalize(x[initial].copy())
    labels = np.full(n, -1, dtype=np.int64)
    for _iteration in range(int(max_iter)):
        sims = x @ centroids.T
        new_labels = np.argmax(sims, axis=1).astype(np.int64)
        new_labels = _repair_empty_clusters(x, centroids, new_labels)
        new_centroids = np.zeros_like(centroids)
        for cluster in range(int(k)):
            members = x[new_labels == cluster]
            new_centroids[cluster] = normalize_vector(members.mean(axis=0))
        shift = float(np.max(1.0 - np.sum(centroids * new_centroids, axis=1)))
        centroids = new_centroids
        labels = new_labels
        if shift <= float(tol):
            break
    return labels


def _repair_empty_clusters(x: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = labels.copy()
    k = int(centroids.shape[0])
    counts = np.bincount(out, minlength=k)
    if np.all(counts > 0):
        return out
    nearest_sim = np.max(x @ centroids.T, axis=1)
    candidate_order = np.argsort(nearest_sim)
    used: set[int] = set()
    for empty in np.where(counts == 0)[0]:
        for candidate in candidate_order:
            candidate = int(candidate)
            if candidate in used:
                continue
            old = int(out[candidate])
            if counts[old] <= 1:
                continue
            out[candidate] = int(empty)
            counts[old] -= 1
            counts[int(empty)] += 1
            used.add(candidate)
            break
    if np.any(np.bincount(out, minlength=k) == 0):
        raise RuntimeError("failed to repair empty spherical k-means cluster")
    return out


def _group_indices_by_class(rows: Sequence[ManifestRow]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[int(row.class_id)].append(int(idx))
    return dict(grouped)


def _sorted_clusters(labels: np.ndarray, rows: Sequence[ManifestRow]) -> list[list[int]]:
    clusters: list[list[int]] = []
    for label in sorted(set(int(x) for x in labels.tolist())):
        members = [idx for idx, value in enumerate(labels.tolist()) if int(value) == label]
        clusters.append(members)
    clusters.sort(key=lambda members: (min(rows[idx].sample_id for idx in members), len(members)))
    return clusters


def _derived_seed(seed: int, k_regions: int, class_id: int) -> int:
    digest = stable_hash({"seed": int(seed), "k_regions": int(k_regions), "class_id": int(class_id)})
    return int(digest[:8], 16)


def _duplicate_fraction(embeddings: np.ndarray, *, threshold: float) -> float:
    x = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    n = int(x.shape[0])
    if n < 2:
        return 0.0
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    nearest_distance = 1.0 - np.max(sims, axis=1)
    return float(np.mean(nearest_distance <= float(threshold)))


def _nearest_member_ids(
    prototype: np.ndarray,
    embeddings: np.ndarray,
    rows: Sequence[ManifestRow],
    *,
    count: int,
) -> list[str]:
    sims = l2_normalize(np.asarray(embeddings, dtype=np.float32)) @ normalize_vector(prototype)
    order = sorted(range(len(rows)), key=lambda idx: (-(float(sims[idx])), rows[idx].sample_id))
    return [rows[idx].sample_id for idx in order[: int(count)]]


def _add_region_geometry_metrics(
    region_rows: Sequence[dict[str, object]],
    *,
    prototypes: np.ndarray,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_rows: Sequence[ManifestRow],
    local_purity_k: int,
    batch_size: int,
) -> list[dict[str, object]]:
    out = [dict(row) for row in region_rows]
    labels = np.asarray(train_labels, dtype=np.int64)
    max_k = min(int(local_purity_k), int(train_embeddings.shape[0]))
    for start in range(0, int(prototypes.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(prototypes.shape[0]))
        sims = np.asarray(prototypes[start:end], dtype=np.float32) @ np.asarray(train_embeddings, dtype=np.float32).T
        for offset, sim_row in enumerate(sims):
            row_index = start + offset
            class_id = int(out[row_index]["class_id"])
            same_mask = labels == class_id
            other_mask = ~same_mask
            same_sims = sim_row[same_mask]
            other_sims = sim_row[other_mask]
            same_best = float(np.max(same_sims)) if same_sims.size else math.nan
            other_best = float(np.max(other_sims)) if other_sims.size else math.nan
            order = np.argpartition(-sim_row, kth=max_k - 1)[:max_k]
            order = order[np.argsort(-sim_row[order])]
            same_count = int(np.sum(labels[order] == class_id))
            out[row_index]["same_class_purity"] = float(same_count / max_k)
            out[row_index]["nearest_same_class_distance"] = float(1.0 - same_best) if math.isfinite(same_best) else ""
            out[row_index]["nearest_competing_class_distance"] = float(1.0 - other_best) if math.isfinite(other_best) else ""
            if math.isfinite(same_best) and math.isfinite(other_best):
                out[row_index]["class_margin"] = float((1.0 - other_best) - (1.0 - same_best))
            else:
                out[row_index]["class_margin"] = ""
            out[row_index]["nearest_same_class_sample_id"] = _nearest_id_with_mask(sim_row, same_mask, train_rows)
            out[row_index]["nearest_competing_sample_id"] = _nearest_id_with_mask(sim_row, other_mask, train_rows)
    return out


def _nearest_id_with_mask(sim_row: np.ndarray, mask: np.ndarray, rows: Sequence[ManifestRow]) -> str:
    if not bool(np.any(mask)):
        return ""
    masked = np.where(mask, sim_row, -np.inf)
    idx = int(np.argmax(masked))
    return str(rows[idx].sample_id)


def _assert_validation_assignments(
    validation_rows: Sequence[dict[str, object]],
    region_rows: Sequence[dict[str, object]],
    *,
    n_validation: int,
    k_values: Sequence[int],
) -> None:
    expected_total = int(n_validation) * len(tuple(k_values))
    if len(validation_rows) != expected_total:
        raise AssertionError(
            "validation_assignments.csv is long-format and must contain one row per "
            f"(validation sample, K); expected {expected_total}, got {len(validation_rows)}"
        )

    expected_k = {int(k) for k in k_values}
    valid_region_ids = {str(row["region_id"]) for row in region_rows}
    sample_to_k: dict[str, set[int]] = defaultdict(set)
    assignment_counts: Counter[str] = Counter()
    for row in validation_rows:
        region_id = str(row["region_id"])
        if region_id not in valid_region_ids:
            raise AssertionError(f"validation assignment references unknown region_id={region_id}")
        sample_to_k[str(row["sample_id"])].add(int(row["k_regions"]))
        assignment_counts[region_id] += 1

    if len(sample_to_k) != int(n_validation):
        raise AssertionError(
            f"validation assignments cover {len(sample_to_k)} samples, expected {n_validation}"
        )
    for sample_id, seen_k in sample_to_k.items():
        if seen_k != expected_k:
            raise AssertionError(
                "validation assignments must include every requested K for each sample; "
                f"sample_id={sample_id} seen={sorted(seen_k)} expected={sorted(expected_k)}"
            )

    for region in region_rows:
        region_id = str(region["region_id"])
        reported = int(region.get("validation_count", 0))
        reproduced = int(assignment_counts.get(region_id, 0))
        if reported != reproduced:
            raise AssertionError(
                f"validation_count mismatch for {region_id}: reported={reported}, reproduced={reproduced}"
            )


def _pack_class_prototypes(class_prototypes: Mapping[int, np.ndarray]) -> tuple[np.ndarray, list[int]]:
    ids = sorted(int(key) for key in class_prototypes)
    arr = np.stack([np.asarray(class_prototypes[key], dtype=np.float32) for key in ids]).astype("float32", copy=False)
    return arr, ids


def _config_payload(
    config: RegionBuildConfig,
    train: CachedEmbeddingBank,
    validation: CachedEmbeddingBank | None,
) -> dict[str, object]:
    payload = {
        "train_cache": str(Path(config.train_cache)),
        "train_cache_id": train.metadata.get("cache_id"),
        "train_manifest_hash": train.metadata.get("cached_manifest_hash"),
        "train_embedding_shape": list(map(int, train.embeddings.shape)),
        "validation_cache": str(Path(config.validation_cache)) if config.validation_cache is not None else "",
        "validation_cache_id": validation.metadata.get("cache_id") if validation is not None else "",
        "validation_manifest_hash": validation.metadata.get("cached_manifest_hash") if validation is not None else "",
        "validation_embedding_shape": list(map(int, validation.embeddings.shape)) if validation is not None else [],
        "experiment_id": config.experiment_id,
        "k_values": list(map(int, sorted(config.k_values))),
        "seed": int(config.seed),
        "residual_mode": str(config.residual_mode),
        "max_iter": int(config.max_iter),
        "tol": float(config.tol),
        "local_purity_k": int(config.local_purity_k),
        "duplicate_distance_threshold": float(config.duplicate_distance_threshold),
        "exemplar_count": int(config.exemplar_count),
        "metric_batch_size": int(config.metric_batch_size),
        "thresholds": asdict(config.thresholds),
    }
    for key, path in (("train_cache_embeddings_sha1", Path(config.train_cache) / "embeddings.npy"),):
        if path.exists():
            payload[key] = file_sha1(path)
    return payload


def _region_summary(
    region_rows: Sequence[Mapping[str, object]],
    membership_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_k: dict[int, dict[str, int]] = {}
    for row in region_rows:
        k = int(row["k_regions"])
        by_k.setdefault(k, {"regions": 0, "eligible_primary": 0})
        by_k[k]["regions"] += 1
        by_k[k]["eligible_primary"] += int(row.get("eligible_primary", 0))
    return {
        "region_count": len(region_rows),
        "membership_rows": len(membership_rows),
        "validation_assignment_rows": len(validation_rows),
        "eligible_primary_count": int(sum(int(row.get("eligible_primary", 0)) for row in region_rows)),
        "by_k": {str(key): value for key, value in sorted(by_k.items())},
    }


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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
