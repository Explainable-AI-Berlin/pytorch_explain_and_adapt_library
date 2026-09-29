from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

from ada.atlas.metrics.deciles import quantile_summary
from ada.atlas.metrics.error_detection import (
    average_precision_score,
    brier_score,
    confidence_to_error_risk,
    roc_auc_score,
)


def summarize_predictions_with_support(
    *,
    support_csv: str | Path,
    prediction_csvs: Sequence[str | Path],
    output_dir: str | Path,
    support_columns: Sequence[str] = (
        "support_k5_kth_distance",
        "support_k10_kth_distance",
        "support_k50_kth_distance",
        "class_support_k5_kth_distance",
        "class_support_k10_kth_distance",
        "class_support_k50_kth_distance",
    ),
    primary_support_column: str = "support_k50_kth_distance",
    primary_class_support_column: str = "class_support_k50_kth_distance",
    bins: int = 10,
    include_crossfit: bool = True,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    support_by_id = _read_support_by_sample_id(Path(support_csv))
    rows: list[dict[str, str]] = []
    for prediction_csv in prediction_csvs:
        with Path(prediction_csv).open("r", newline="") as f:
            for row in csv.DictReader(f):
                support = support_by_id.get(row["sample_id"])
                if support is None:
                    raise KeyError(f"prediction sample_id missing from support table: {row['sample_id']}")
                joined = dict(row)
                joined["is_error"] = str(1 - int(row["correct"]))
                for column in support_columns:
                    joined[column] = support.get(column, "")
                rows.append(joined)

    joined_csv = output / "joined_predictions_support.csv"
    _write_rows(joined_csv, rows)

    metrics = _summarize_rows(
        rows,
        support_columns=support_columns,
        primary_support_column=primary_support_column,
        primary_class_support_column=primary_class_support_column,
        bins=bins,
        include_crossfit=include_crossfit,
    )
    metrics["support_csv"] = str(support_csv)
    metrics["prediction_csvs"] = [str(path) for path in prediction_csvs]
    metrics["joined_csv"] = str(joined_csv)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    _write_deciles(output / "deciles.csv", metrics["models"])
    return output / "metrics.json"


def _read_support_by_sample_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="") as f:
        rows = {row["sample_id"]: row for row in csv.DictReader(f)}
    if not rows:
        raise ValueError(f"support table is empty: {path}")
    return rows


def _write_rows(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("no joined rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_rows(
    rows: Sequence[dict[str, str]],
    *,
    support_columns: Sequence[str],
    primary_support_column: str,
    primary_class_support_column: str,
    bins: int,
    include_crossfit: bool,
) -> dict:
    out = {"rows": len(rows), "models": {}}
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row)

    for model_id, model_rows in sorted(by_model.items()):
        errors = [int(row["is_error"]) for row in model_rows]
        confidence = [_confidence_value(row) for row in model_rows]
        confidence_risk = confidence_to_error_risk(confidence)
        model_report = {
            "rows": len(model_rows),
            "errors": int(sum(errors)),
            "accuracy": 1.0 - (sum(errors) / len(errors)),
            "mean_confidence": sum(confidence) / len(confidence),
            "scores": {},
            "deciles": {},
            "crossfit_logistic": {},
        }
        model_report["scores"]["confidence_risk"] = _error_metrics(confidence_risk, errors, probability_like=True)
        for column in support_columns:
            values = [_float(row[column]) for row in model_rows]
            model_report["scores"][column] = _error_metrics(values, errors)
            decile_rows = [dict(row, confidence_raw=str(conf)) for row, conf in zip(model_rows, confidence)]
            model_report["deciles"][column] = quantile_summary(
                decile_rows,
                score_column=column,
                error_column="is_error",
                confidence_column="confidence_raw",
                bins=bins,
                high_score_is_risky=True,
            )
        model_report["scores"][f"rankavg_confidence_{primary_support_column}"] = _rank_average_metrics(
            confidence_risk,
            [_float(row[primary_support_column]) for row in model_rows],
            errors,
        )
        if include_crossfit:
            model_report["crossfit_logistic"] = _crossfit_logistic_report(
                model_rows,
                errors,
                confidence,
                primary_support_column=primary_support_column,
                primary_class_support_column=primary_class_support_column,
            )
        else:
            model_report["crossfit_logistic"] = {"available": False, "reason": "disabled"}
        out["models"][model_id] = model_report
    return out


def _confidence_value(row: dict[str, str]) -> float:
    calibrated = row.get("max_probability_calibrated", "")
    if calibrated not in {"", "nan", "NaN", "None"}:
        return _float(calibrated)
    return _float(row["max_probability_raw"])


def _float(value: str) -> float:
    return float(value)


def _error_metrics(scores: Sequence[float], errors: Sequence[int], *, probability_like: bool = False) -> dict[str, float | None]:
    return {
        "auroc": roc_auc_score(scores, errors),
        "auprc": average_precision_score(scores, errors),
        "brier": brier_score(scores, errors) if probability_like else None,
    }


def _rank_average_metrics(scores_a: Sequence[float], scores_b: Sequence[float], errors: Sequence[int]) -> dict[str, float]:
    ranks_a = _percentile_ranks(scores_a)
    ranks_b = _percentile_ranks(scores_b)
    combined = [(a + b) / 2.0 for a, b in zip(ranks_a, ranks_b)]
    return _error_metrics(combined, errors)


def _percentile_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: float(values[idx]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[cursor]]):
            end += 1
        rank = (cursor + end - 1) / 2.0
        denom = max(len(values) - 1, 1)
        for pos in range(cursor, end):
            ranks[order[pos]] = rank / denom
        cursor = end
    return ranks


def _crossfit_logistic_report(
    rows: Sequence[dict[str, str]],
    errors: Sequence[int],
    confidence: Sequence[float],
    *,
    primary_support_column: str,
    primary_class_support_column: str,
) -> dict:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except Exception as exc:  # pragma: no cover - environment-dependent
        return _pure_python_crossfit_logistic_report(
            rows,
            errors,
            confidence,
            primary_support_column=primary_support_column,
            primary_class_support_column=primary_class_support_column,
            sklearn_reason=repr(exc),
        )

    y = np.asarray(errors, dtype=np.int64)
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        return {"available": False, "reason": "not enough positive or negative examples"}
    feature_sets = {
        "confidence": [[1.0 - confidence[i]] for i in range(len(rows))],
        "confidence_plus_support": [
            [1.0 - confidence[i], _float(row[primary_support_column])]
            for i, row in enumerate(rows)
        ],
        "confidence_plus_global_and_class_support": [
            [
                1.0 - confidence[i],
                _float(row[primary_support_column]),
                _float(row[primary_class_support_column]),
            ]
            for i, row in enumerate(rows)
        ],
    }
    report = {"available": True, "folds": 5, "models": {}}
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, matrix in feature_sets.items():
        x = np.asarray(matrix, dtype=np.float64)
        pred = np.zeros(len(rows), dtype=np.float64)
        for train_idx, test_idx in splits.split(x, y):
            mean = x[train_idx].mean(axis=0, keepdims=True)
            std = x[train_idx].std(axis=0, keepdims=True)
            std[std < 1.0e-8] = 1.0
            x_train = (x[train_idx] - mean) / std
            x_test = (x[test_idx] - mean) / std
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
            clf.fit(x_train, y[train_idx])
            pred[test_idx] = clf.predict_proba(x_test)[:, 1]
        scores = [float(value) for value in pred.tolist()]
        model_metrics = _error_metrics(scores, errors, probability_like=True)
        model_metrics["log_loss"] = _log_loss(scores, errors)
        report["models"][name] = model_metrics
    return report


def _pure_python_crossfit_logistic_report(
    rows: Sequence[dict[str, str]],
    errors: Sequence[int],
    confidence: Sequence[float],
    *,
    primary_support_column: str,
    primary_class_support_column: str,
    sklearn_reason: str,
) -> dict:
    feature_sets = {
        "confidence": [[1.0 - confidence[i]] for i in range(len(rows))],
        "confidence_plus_support": [
            [1.0 - confidence[i], _float(row[primary_support_column])]
            for i, row in enumerate(rows)
        ],
        "confidence_plus_global_and_class_support": [
            [
                1.0 - confidence[i],
                _float(row[primary_support_column]),
                _float(row[primary_class_support_column]),
            ]
            for i, row in enumerate(rows)
        ],
    }
    if sum(errors) < 2 or (len(errors) - sum(errors)) < 2:
        return {"available": False, "reason": "not enough positive or negative examples"}
    folds = _stratified_folds(errors, n_splits=5, seed=0)
    report = {
        "available": True,
        "backend": "pure_python",
        "sklearn_unavailable_reason": sklearn_reason,
        "folds": len(folds),
        "models": {},
    }
    for name, matrix in feature_sets.items():
        pred = [0.0] * len(rows)
        for test_idx in folds:
            test_set = set(test_idx)
            train_idx = [idx for idx in range(len(rows)) if idx not in test_set]
            x_train_raw = [matrix[idx] for idx in train_idx]
            x_test_raw = [matrix[idx] for idx in test_idx]
            x_train, x_test = _standardize_train_test(x_train_raw, x_test_raw)
            y_train = [errors[idx] for idx in train_idx]
            weights = _fit_logistic_gd(x_train, y_train, epochs=250, lr=0.1, l2=1.0e-3)
            for idx, x_vec in zip(test_idx, x_test):
                pred[idx] = _sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], x_vec)))
        metrics = _error_metrics(pred, errors, probability_like=True)
        metrics["log_loss"] = _log_loss(pred, errors)
        report["models"][name] = metrics
    return report


def _stratified_folds(labels: Sequence[int], *, n_splits: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)
    for indices in by_label.values():
        rng.shuffle(indices)
    folds = [[] for _ in range(n_splits)]
    for indices in by_label.values():
        for offset, idx in enumerate(indices):
            folds[offset % n_splits].append(idx)
    for fold in folds:
        fold.sort()
    return folds


def _standardize_train_test(
    x_train: Sequence[Sequence[float]],
    x_test: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    dim = len(x_train[0])
    mean = [sum(row[j] for row in x_train) / len(x_train) for j in range(dim)]
    std = []
    for j in range(dim):
        var = sum((row[j] - mean[j]) ** 2 for row in x_train) / len(x_train)
        std.append(max(math.sqrt(var), 1.0e-8))

    def transform(rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(row[j] - mean[j]) / std[j] for j in range(dim)] for row in rows]

    return transform(x_train), transform(x_test)


def _fit_logistic_gd(
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> list[float]:
    dim = len(x_train[0])
    weights = [0.0] * (dim + 1)
    n_pos = max(sum(y_train), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    pos_weight = len(y_train) / (2.0 * n_pos)
    neg_weight = len(y_train) / (2.0 * n_neg)
    for _ in range(epochs):
        grad = [0.0] * (dim + 1)
        for x_vec, label in zip(x_train, y_train):
            z = weights[0] + sum(w * x for w, x in zip(weights[1:], x_vec))
            p = _sigmoid(z)
            sample_weight = pos_weight if int(label) == 1 else neg_weight
            diff = (p - int(label)) * sample_weight
            grad[0] += diff
            for j, value in enumerate(x_vec):
                grad[j + 1] += diff * value
        inv_n = 1.0 / len(x_train)
        weights[0] -= lr * grad[0] * inv_n
        for j in range(1, len(weights)):
            weights[j] -= lr * ((grad[j] * inv_n) + l2 * weights[j])
    return weights


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _log_loss(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    losses = []
    for prob, label in zip(probabilities, labels):
        p = min(max(float(prob), 1.0e-12), 1.0 - 1.0e-12)
        losses.append(-(int(label) * math.log(p) + (1 - int(label)) * math.log(1.0 - p)))
    return sum(losses) / len(losses)


def _write_deciles(path: Path, model_reports: dict) -> None:
    rows = []
    for model_id, report in model_reports.items():
        for score_column, deciles in report.get("deciles", {}).items():
            for row in deciles:
                rows.append({"model_id": model_id, "score_column": score_column, **row})
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
