from __future__ import annotations

from typing import Sequence


def _validate_binary(labels: Sequence[int | bool]) -> list[int]:
    out = [int(x) for x in labels]
    if any(x not in {0, 1} for x in out):
        raise ValueError("labels must be binary with errors/positives encoded as 1")
    return out


def roc_auc_score(scores: Sequence[float], labels: Sequence[int | bool]) -> float:
    """Compute AUROC with average ranks for ties.

    `scores` should be larger for more likely errors.
    """
    y = _validate_binary(labels)
    if len(scores) != len(y):
        raise ValueError("scores and labels must have the same length")
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires at least one positive and one negative")

    order = sorted(range(len(scores)), key=lambda i: float(scores[i]))
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        score = float(scores[order[cursor]])
        while end < len(order) and float(scores[order[end]]) == score:
            end += 1
        avg_rank = (cursor + 1 + end) / 2.0
        for pos in range(cursor, end):
            ranks[order[pos]] = avg_rank
        cursor = end

    pos_rank_sum = sum(rank for rank, label in zip(ranks, y) if label == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_score(scores: Sequence[float], labels: Sequence[int | bool]) -> float:
    """Compute average precision with errors/positives encoded as 1."""
    y = _validate_binary(labels)
    if len(scores) != len(y):
        raise ValueError("scores and labels must have the same length")
    total_pos = sum(y)
    if total_pos == 0:
        raise ValueError("average precision requires at least one positive")

    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    tp = 0
    fp = 0
    last_recall = 0.0
    ap = 0.0
    for idx in order:
        if y[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / max(tp + fp, 1)
        if y[idx] == 1:
            ap += (recall - last_recall) * precision
            last_recall = recall
    return ap


def brier_score(probabilities: Sequence[float], labels: Sequence[int | bool]) -> float:
    y = _validate_binary(labels)
    if len(probabilities) != len(y):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    return sum((float(p) - label) ** 2 for p, label in zip(probabilities, y)) / len(y)


def confidence_to_error_risk(confidences: Sequence[float]) -> list[float]:
    return [1.0 - float(conf) for conf in confidences]
