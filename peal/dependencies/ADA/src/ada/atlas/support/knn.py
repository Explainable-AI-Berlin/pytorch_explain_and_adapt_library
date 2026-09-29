from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


Vector = Sequence[float]


@dataclass(frozen=True)
class KNNResult:
    indices: list[list[int]]
    reference_ids: list[list[str]]
    distances: list[list[float]]
    similarities: list[list[float]]
    kth_distance: list[float]
    mean_distance: list[float]


def l2_normalize_rows(rows: Sequence[Vector], *, eps: float = 1.0e-12) -> list[list[float]]:
    out: list[list[float]] = []
    for row in rows:
        norm = math.sqrt(sum(float(x) * float(x) for x in row))
        denom = max(norm, eps)
        out.append([float(x) / denom for x in row])
    return out


def cosine_similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} vs {len(b)}")
    return sum(float(x) * float(y) for x, y in zip(a, b))


def exact_cosine_knn(
    query: Sequence[Vector],
    reference: Sequence[Vector],
    *,
    k: int,
    query_ids: Sequence[str] | None = None,
    reference_ids: Sequence[str] | None = None,
    leave_one_out: bool = False,
    assume_normalized: bool = False,
) -> KNNResult:
    if k < 1:
        raise ValueError("k must be positive")
    if not query:
        return KNNResult([], [], [], [], [], [])
    if not reference:
        raise ValueError("reference must contain at least one vector")

    dim = len(query[0])
    if any(len(row) != dim for row in query):
        raise ValueError("query vectors must all have the same dimension")
    if any(len(row) != dim for row in reference):
        raise ValueError("reference vectors must match query dimension")

    if leave_one_out and (query_ids is None or reference_ids is None):
        raise ValueError("leave_one_out requires query_ids and reference_ids")
    if query_ids is not None and len(query_ids) != len(query):
        raise ValueError("query_ids length does not match query rows")
    if reference_ids is not None and len(reference_ids) != len(reference):
        raise ValueError("reference_ids length does not match reference rows")

    q_rows = [list(map(float, row)) for row in query] if assume_normalized else l2_normalize_rows(query)
    r_rows = [list(map(float, row)) for row in reference] if assume_normalized else l2_normalize_rows(reference)
    ref_ids = list(reference_ids) if reference_ids is not None else [str(i) for i in range(len(reference))]

    all_indices: list[list[int]] = []
    all_ids: list[list[str]] = []
    all_distances: list[list[float]] = []
    all_similarities: list[list[float]] = []
    kth_distance: list[float] = []
    mean_distance: list[float] = []

    for qi, q in enumerate(q_rows):
        qid = None if query_ids is None else str(query_ids[qi])
        candidates: list[tuple[float, str, int, float]] = []
        for ri, r in enumerate(r_rows):
            rid = ref_ids[ri]
            if leave_one_out and qid == rid:
                continue
            sim = cosine_similarity(q, r)
            distance = 1.0 - sim
            candidates.append((distance, rid, ri, sim))

        if len(candidates) < k:
            raise ValueError(f"Not enough reference neighbours for k={k}; found {len(candidates)}")
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        chosen = candidates[:k]
        distances = [float(item[0]) for item in chosen]
        similarities = [float(item[3]) for item in chosen]
        all_indices.append([int(item[2]) for item in chosen])
        all_ids.append([str(item[1]) for item in chosen])
        all_distances.append(distances)
        all_similarities.append(similarities)
        kth_distance.append(distances[-1])
        mean_distance.append(sum(distances) / len(distances))

    return KNNResult(
        indices=all_indices,
        reference_ids=all_ids,
        distances=all_distances,
        similarities=all_similarities,
        kth_distance=kth_distance,
        mean_distance=mean_distance,
    )
