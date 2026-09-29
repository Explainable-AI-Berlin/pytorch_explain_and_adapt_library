"""Metrics that separate condition fidelity from pointwise source locking.

The functions in this module operate on generic feature arrays. They are used by
the synthetic benchmark and are also intended for re-encoded CLS, DINO, SigLIP,
or perceptual features from the image generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _rows(x: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a rank-two array, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1.0e-12)


def slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Spherical interpolation with linearly interpolated endpoint norms."""

    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError(f"a and b must be matching vectors, got {left.shape} and {right.shape}")
    alpha = np.asarray(t, dtype=np.float64)
    if not np.isfinite(alpha).all():
        raise ValueError("t contains non-finite values")

    norm_left = float(np.linalg.norm(left))
    norm_right = float(np.linalg.norm(right))
    unit_left = left / max(norm_left, 1.0e-12)
    unit_right = right / max(norm_right, 1.0e-12)
    dot = float(np.clip(unit_left @ unit_right, -1.0, 1.0))
    omega = float(np.arccos(dot))

    flat_alpha = alpha.reshape(-1)
    if omega < 1.0e-7:
        directions = (1.0 - flat_alpha[:, None]) * unit_left + flat_alpha[:, None] * unit_right
        directions = _normalize(directions)
    else:
        denominator = np.sin(omega)
        directions = (
            np.sin((1.0 - flat_alpha) * omega)[:, None] / denominator * unit_left
            + np.sin(flat_alpha * omega)[:, None] / denominator * unit_right
        )
    norms = (1.0 - flat_alpha) * norm_left + flat_alpha * norm_right
    result = directions * norms[:, None]
    return result.reshape(alpha.shape + left.shape)


def effective_rank(samples: np.ndarray) -> float:
    """Participation-ratio rank of a sample cloud covariance."""

    values = _rows(samples, name="samples")
    if values.shape[0] < 2:
        return 0.0
    singular_values = np.linalg.svd(values - values.mean(axis=0, keepdims=True), compute_uv=False)
    eigenvalues = singular_values**2 / max(1, values.shape[0] - 1)
    denominator = float(np.square(eigenvalues).sum())
    if denominator <= 1.0e-20:
        return 0.0
    return float(np.square(eigenvalues.sum()) / denominator)


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    queries = _rows(query, name="query")
    references = _rows(reference, name="reference")
    if queries.shape[1] != references.shape[1]:
        raise ValueError("query and reference feature dimensions differ")
    q2 = np.square(queries).sum(axis=1, keepdims=True)
    r2 = np.square(references).sum(axis=1, keepdims=True).T
    distances2 = np.maximum(q2 + r2 - 2.0 * queries @ references.T, 0.0)
    indices = np.argmin(distances2, axis=1)
    return np.sqrt(distances2[np.arange(len(queries)), indices]), indices


def calibrated_nearest_train_ratio(
    generated: np.ndarray,
    fresh_reference: np.ndarray,
    train: np.ndarray,
) -> float:
    """Nearest-train distance relative to independent population draws.

    A value near one means generated and fresh held-out samples have comparable
    proximity to the training set. Values far below one indicate excess
    concentration around training examples. The arrays must represent matched
    query conditions, but may contain multiple samples per condition.
    """

    generated_distance, _ = nearest_distances(generated, train)
    fresh_distance, _ = nearest_distances(fresh_reference, train)
    denominator = float(np.mean(fresh_distance))
    if denominator <= 1.0e-12:
        return float("nan")
    return float(np.mean(generated_distance) / denominator)


@dataclass(frozen=True)
class PathDiagnostics:
    num_points: int
    condition_output_distance_correlation: float
    progress_spearman: float
    stationary_step_fraction: float
    max_to_mean_step_ratio: float
    nearest_source_switches: int
    longest_source_lock_fraction: float
    output_effective_rank: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) <= 1.0e-12 or float(np.std(b)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def conditional_path_diagnostics(
    conditions: np.ndarray,
    outputs: np.ndarray,
    train_outputs: np.ndarray,
    *,
    progress: np.ndarray | None = None,
    stationary_relative_threshold: float = 0.1,
) -> PathDiagnostics:
    """Measure whether a fixed-noise conditional path moves smoothly or snaps.

    ``conditions`` and ``outputs`` are ordered along one interpolation path.
    ``outputs`` should be features re-encoded from images generated with the
    same initial diffusion noise. A nearest-neighbour lookup has long stationary
    runs and abrupt source switches; a smooth conditional map tracks progress.
    """

    cond = _rows(conditions, name="conditions")
    out = _rows(outputs, name="outputs")
    train = _rows(train_outputs, name="train_outputs")
    if len(cond) != len(out):
        raise ValueError("conditions and outputs must have the same number of rows")
    if len(out) < 3:
        raise ValueError("a conditional path needs at least three points")

    cond_steps = np.linalg.norm(np.diff(cond, axis=0), axis=1)
    output_steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    positive_cond = cond_steps[cond_steps > 1.0e-12]
    expected_step = float(np.median(positive_cond)) if len(positive_cond) else 0.0
    output_scale = float(np.linalg.norm(np.ptp(out, axis=0)))
    threshold = max(1.0e-10, stationary_relative_threshold * expected_step * max(output_scale, 1.0))
    stationary_fraction = float(np.mean(output_steps <= threshold))

    mean_step = float(np.mean(output_steps))
    jump_ratio = float(np.max(output_steps) / max(mean_step, 1.0e-12))

    _, nearest = nearest_distances(out, train)
    switches = int(np.count_nonzero(np.diff(nearest)))
    longest = 1
    run = 1
    for changed in np.diff(nearest) != 0:
        if changed:
            longest = max(longest, run)
            run = 1
        else:
            run += 1
    longest = max(longest, run)

    cond_pair = np.linalg.norm(cond[:, None, :] - cond[None, :, :], axis=2)
    out_pair = np.linalg.norm(out[:, None, :] - out[None, :, :], axis=2)
    upper = np.triu_indices(len(out), k=1)
    geometry_corr = _correlation(cond_pair[upper], out_pair[upper])

    if progress is None:
        progress_values = np.linspace(0.0, 1.0, len(out), dtype=np.float64)
    else:
        progress_values = np.asarray(progress, dtype=np.float64)
        if progress_values.shape != (len(out),):
            raise ValueError(f"progress must have shape {(len(out),)}, got {progress_values.shape}")
    direction = out[-1] - out[0]
    direction_norm2 = float(direction @ direction)
    if direction_norm2 <= 1.0e-20:
        output_progress = np.zeros(len(out), dtype=np.float64)
    else:
        output_progress = (out - out[0]) @ direction / direction_norm2
    progress_corr = _correlation(_rankdata(progress_values), _rankdata(output_progress))

    return PathDiagnostics(
        num_points=len(out),
        condition_output_distance_correlation=geometry_corr,
        progress_spearman=progress_corr,
        stationary_step_fraction=stationary_fraction,
        max_to_mean_step_ratio=jump_ratio,
        nearest_source_switches=switches,
        longest_source_lock_fraction=float(longest / len(out)),
        output_effective_rank=effective_rank(out),
    )
