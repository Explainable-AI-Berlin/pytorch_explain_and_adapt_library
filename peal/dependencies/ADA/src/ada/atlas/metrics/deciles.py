from __future__ import annotations

from typing import Mapping, Sequence


def quantile_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    score_column: str,
    error_column: str = "is_error",
    confidence_column: str | None = "confidence_raw",
    bins: int = 10,
    high_score_is_risky: bool = True,
) -> list[dict[str, float | int]]:
    """Summarize error and confidence by score quantile.

    Use `high_score_is_risky=True` for kNN distances where larger distance
    means lower support. Use `False` for support scores where larger is safer.
    """
    if bins < 1:
        raise ValueError("bins must be positive")
    if not rows:
        raise ValueError("rows must not be empty")

    sorted_rows = sorted(rows, key=lambda row: float(row[score_column]), reverse=high_score_is_risky)
    n = len(sorted_rows)
    summaries: list[dict[str, float | int]] = []
    for bin_idx in range(bins):
        start = (bin_idx * n) // bins
        end = ((bin_idx + 1) * n) // bins
        chunk = sorted_rows[start:end]
        if not chunk:
            continue
        errors = [int(row[error_column]) for row in chunk]
        scores = [float(row[score_column]) for row in chunk]
        out: dict[str, float | int] = {
            "bin": bin_idx + 1,
            "count": len(chunk),
            "score_min": min(scores),
            "score_max": max(scores),
            "score_mean": sum(scores) / len(scores),
            "error_rate": sum(errors) / len(errors),
        }
        if confidence_column is not None:
            confidences = [float(row[confidence_column]) for row in chunk]
            accuracy = 1.0 - out["error_rate"]
            out["confidence_mean"] = sum(confidences) / len(confidences)
            out["calibration_gap"] = out["confidence_mean"] - accuracy
        summaries.append(out)
    return summaries
