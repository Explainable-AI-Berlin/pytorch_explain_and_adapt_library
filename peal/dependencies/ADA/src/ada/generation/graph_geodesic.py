"""Finite-atlas graph paths for representation conditions."""

from __future__ import annotations

import heapq

import numpy as np

from ada.generation.conditional_memorization import slerp


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("values must be a finite rank-two array")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1.0e-12)


def symmetric_knn_adjacency(values: np.ndarray, k: int) -> list[dict[int, float]]:
    """Build a symmetric union-kNN graph with angular edge lengths."""

    unit = normalize_rows(values)
    count = len(unit)
    neighbors = int(k)
    if count < 2 or not 1 <= neighbors < count:
        raise ValueError("k must lie in [1, num_rows - 1]")
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    np.fill_diagonal(cosine, -np.inf)
    selected = np.argpartition(-cosine, kth=neighbors - 1, axis=1)[:, :neighbors]
    adjacency: list[dict[int, float]] = [dict() for _ in range(count)]
    for left in range(count):
        for right in selected[left].tolist():
            distance = float(np.arccos(np.clip(cosine[left, right], -1.0, 1.0)))
            old = adjacency[left].get(right)
            if old is None or distance < old:
                adjacency[left][right] = distance
                adjacency[right][left] = distance
    return adjacency


def shortest_path(
    adjacency: list[dict[int, float]],
    source: int,
    target: int,
) -> tuple[list[int], float]:
    """Dijkstra shortest path for a non-negative adjacency list."""

    count = len(adjacency)
    start = int(source)
    end = int(target)
    if not 0 <= start < count or not 0 <= end < count:
        raise ValueError("source and target must be graph vertices")
    distance = np.full(count, np.inf, dtype=np.float64)
    previous = np.full(count, -1, dtype=np.int64)
    distance[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance != distance[current]:
            continue
        if current == end:
            break
        for neighbor, edge in adjacency[current].items():
            candidate = current_distance + float(edge)
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    if not np.isfinite(distance[end]):
        raise RuntimeError("source and target are disconnected")
    path = [end]
    while path[-1] != start:
        path.append(int(previous[path[-1]]))
    path.reverse()
    return path, float(distance[end])


def piecewise_slerp_path(vertices: np.ndarray, progress: np.ndarray) -> np.ndarray:
    """Resample a polyline of spherical local edges at arc-length progress."""

    values = np.asarray(vertices, dtype=np.float64)
    alpha = np.asarray(progress, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("vertices must contain at least two finite vectors")
    if alpha.ndim != 1 or len(alpha) < 2 or alpha[0] != 0.0 or alpha[-1] != 1.0:
        raise ValueError("progress must be a vector beginning at zero and ending at one")
    if np.any(np.diff(alpha) <= 0.0):
        raise ValueError("progress must be strictly increasing")
    unit = normalize_rows(values)
    lengths = np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] <= 1.0e-12:
        return np.repeat(values[:1], len(alpha), axis=0)
    targets = alpha * cumulative[-1]
    result = []
    for target in targets:
        segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
        local = (target - cumulative[segment]) / max(lengths[segment], 1.0e-12)
        result.append(slerp(values[segment], values[segment + 1], float(local)))
    return np.asarray(result, dtype=np.float64)


__all__ = [
    "normalize_rows",
    "piecewise_slerp_path",
    "shortest_path",
    "symmetric_knn_adjacency",
]
