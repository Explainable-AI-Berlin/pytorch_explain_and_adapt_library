#!/usr/bin/env python3
"""Build paired direct-SLERP and finite-atlas graph condition paths."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ada.generation.conditional_memorization import slerp
from ada.generation.graph_geodesic import (
    normalize_rows,
    piecewise_slerp_path,
    shortest_path,
    symmetric_knn_adjacency,
)


STRATUM_ROLES = {
    "coherent": "p95_periphery",
    "entangled": "typical_crossed_example",
    "diffuse": "p95_periphery",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--subsample-size", type=int, default=512)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--path-points", type=int, default=7)
    parser.add_argument("--support-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--use-all-sources",
        action="store_true",
        help="Use every row in source-manifest in its frozen order.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_sources(rows: list[dict], per_stratum: int) -> list[dict]:
    output: list[dict] = []
    for stratum, role in STRATUM_ROLES.items():
        candidates = [
            row for row in rows
            if row["atlas_stratum"] == stratum and row["atlas_role"] == role
        ]
        candidates.sort(key=lambda row: (int(row["class_index"]), int(row["cache_index"])))
        if len(candidates) < per_stratum:
            raise ValueError(f"Only {len(candidates)} candidates for {stratum}/{role}")
        output.extend(candidates[:per_stratum])
    return output


def class_medoid(features: np.ndarray, rows: np.ndarray) -> int:
    values = np.asarray(features[rows], dtype=np.float32)
    centroid = values.mean(axis=0)
    return int(rows[np.square(values - centroid).sum(axis=1).argmin()])


def support_distance(query: np.ndarray, references: np.ndarray, k: int) -> float:
    q = query / max(float(np.linalg.norm(query)), 1.0e-12)
    refs = normalize_rows(references)
    distance = 1.0 - refs @ q
    index = min(int(k), len(distance)) - 1
    return float(np.partition(distance, index)[index])


def build_local_rows(
    class_rows: np.ndarray,
    anchor: int,
    endpoint: int,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    required = np.asarray(sorted({int(anchor), int(endpoint)}), dtype=np.int64)
    remaining = class_rows[~np.isin(class_rows, required)]
    count = min(max(size - len(required), 0), len(remaining))
    sampled = rng.choice(remaining, size=count, replace=False) if count else np.empty(0, dtype=np.int64)
    return np.concatenate([required, np.sort(sampled)])


def main() -> None:
    args = parse_args()
    completed = args.output_dir / "COMPLETED"
    if completed.exists() and not args.overwrite:
        print(f"[reuse] {args.output_dir}")
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    sources = (
        list(payload["rows"])
        if args.use_all_sources
        else select_sources(payload["rows"], args.per_stratum)
    )
    if not sources:
        raise ValueError("The source manifest is empty")
    features = np.load(args.geometry_bundle / "cls.float32.npy", mmap_mode="r")
    labels = np.load(args.geometry_bundle / "y.int16.npy", mmap_mode="r")
    source_ids = np.load(args.geometry_bundle / "source_index.int32.npy", mmap_mode="r")
    progress = np.linspace(0.0, 1.0, args.path_points)
    rng = np.random.default_rng(args.seed)
    conditions: list[np.ndarray] = []
    records: list[dict] = []
    path_summaries: list[dict] = []

    for source_slot, source in enumerate(sources):
        anchor_index = int(source["cache_index"])
        class_index = int(source["class_index"])
        rows = np.flatnonzero(labels == class_index)
        endpoint_index = class_medoid(features, rows)
        local_rows = build_local_rows(
            rows, anchor_index, endpoint_index, args.subsample_size, rng
        )
        local_values = np.asarray(features[local_rows], dtype=np.float32)
        local_anchor = int(np.flatnonzero(local_rows == anchor_index)[0])
        local_endpoint = int(np.flatnonzero(local_rows == endpoint_index)[0])

        used_k = args.knn_k
        while True:
            graph = symmetric_knn_adjacency(local_values, used_k)
            try:
                vertex_path, graph_length = shortest_path(graph, local_anchor, local_endpoint)
                break
            except RuntimeError:
                used_k *= 2
                if used_k >= len(local_rows):
                    raise
        graph_geometry_rows = local_rows[np.asarray(vertex_path, dtype=np.int64)]
        graph_vertices = np.asarray(features[graph_geometry_rows], dtype=np.float32)
        graph_conditions = piecewise_slerp_path(graph_vertices, progress).astype(np.float32)
        direct_conditions = slerp(
            np.asarray(features[anchor_index], dtype=np.float32),
            np.asarray(features[endpoint_index], dtype=np.float32),
            progress,
        ).astype(np.float32)

        for path_mode, path_values in (
            ("direct_slerp", direct_conditions),
            ("graph_geodesic", graph_conditions),
        ):
            for step_index, (alpha, value) in enumerate(zip(progress, path_values)):
                condition_index = len(conditions)
                conditions.append(value)
                records.append(
                    {
                        "condition_index": condition_index,
                        "source_slot": source_slot,
                        "path_mode": path_mode,
                        "path_step": step_index,
                        "progress": float(alpha),
                        "source_geometry_index": anchor_index,
                        "source_index": int(source["source_index"]),
                        "class_index": class_index,
                        "wnid": source["wnid"],
                        "class_name": source["class_name"],
                        "atlas_stratum": source["atlas_stratum"],
                        "atlas_role": source["atlas_role"],
                        "endpoint_geometry_index": endpoint_index,
                        "endpoint_source_index": int(source_ids[endpoint_index]),
                        "graph_vertices": [int(value) for value in graph_geometry_rows],
                        "graph_vertex_count": len(graph_geometry_rows),
                        "graph_knn_k": used_k,
                        "condition_same_class_kth_cosine_distance": support_distance(
                            value,
                            np.asarray(features[rows], dtype=np.float32),
                            args.support_k,
                        ),
                    }
                )
        direct_support = [
            row["condition_same_class_kth_cosine_distance"]
            for row in records
            if row["source_slot"] == source_slot and row["path_mode"] == "direct_slerp"
        ]
        graph_support = [
            row["condition_same_class_kth_cosine_distance"]
            for row in records
            if row["source_slot"] == source_slot and row["path_mode"] == "graph_geodesic"
        ]
        path_summaries.append(
            {
                "source_slot": source_slot,
                "class_index": class_index,
                "wnid": source["wnid"],
                "atlas_stratum": source["atlas_stratum"],
                "graph_vertex_count": len(graph_geometry_rows),
                "graph_knn_k": used_k,
                "graph_angular_length": graph_length,
                "direct_max_support_distance": max(direct_support),
                "graph_max_support_distance": max(graph_support),
                "graph_minus_direct_max_support": max(graph_support) - max(direct_support),
            }
        )
        print(
            f"[graph] {source_slot + 1}/{len(sources)} {source['wnid']} "
            f"vertices={len(graph_geometry_rows)} k={used_k}",
            flush=True,
        )

    np.save(
        args.output_dir / "conditions.float32.npy",
        np.asarray(conditions, dtype=np.float32),
    )
    np.save(
        args.output_dir / "y.int16.npy",
        np.asarray([row["class_index"] for row in records], dtype=np.int16),
    )
    np.save(
        args.output_dir / "source_index.int32.npy",
        np.asarray([row["source_index"] for row in records], dtype=np.int32),
    )
    (args.output_dir / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "path_summaries.json").write_text(
        json.dumps(path_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "purpose": "paired_direct_slerp_vs_same_class_graph_geodesic_condition_bank",
        "source_manifest": str(args.source_manifest.resolve()),
        "geometry_bundle": str(args.geometry_bundle.resolve()),
        "num_sources": len(sources),
        "num_conditions": len(conditions),
        "path_modes": ["direct_slerp", "graph_geodesic"],
        "fixed_endpoints_across_modes": True,
        "progress": progress.tolist(),
        "subsample_size": args.subsample_size,
        "graph": "symmetric union-kNN with angular edge lengths",
        "requested_knn_k": args.knn_k,
        "support_k": args.support_k,
        "seed": args.seed,
        "arrays": {
            "conditions": "conditions.float32.npy",
            "labels": "y.int16.npy",
            "source_indices": "source_index.int32.npy",
        },
        "array_rows_aligned": True,
        "source_selection": "all_manifest_rows" if args.use_all_sources else "stratified",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completed.write_text("complete\n", encoding="utf-8")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
