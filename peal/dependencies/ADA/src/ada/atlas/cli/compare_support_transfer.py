from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from ada.atlas.metrics.error_detection import average_precision_score, brier_score, roc_auc_score
from ada.atlas.predictions.join_support import (
    _fit_logistic_gd,
    _log_loss,
    _sigmoid,
    _standardize_train_test,
    _stratified_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare whether DINO and VAE support predict errors from the same "
            "prediction CSVs, including confidence-controlled nested models."
        )
    )
    parser.add_argument("--prediction-csv", required=True, action="append", type=Path)
    parser.add_argument("--dino-support-csv", required=True, type=Path)
    parser.add_argument("--vae-support-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vae-support-name", default="vae")
    parser.add_argument("--dino-support-name", default="dino")
    parser.add_argument("--global-support-column", default="support_k50_kth_distance")
    parser.add_argument("--class-support-column", default="class_support_k50_kth_distance")
    parser.add_argument("--folds", default=5, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = compare_support_transfer(
        prediction_csvs=args.prediction_csv,
        dino_support_csv=args.dino_support_csv,
        vae_support_csv=args.vae_support_csv,
        output_dir=args.output_dir,
        dino_support_name=args.dino_support_name,
        vae_support_name=args.vae_support_name,
        global_support_column=args.global_support_column,
        class_support_column=args.class_support_column,
        folds=args.folds,
    )
    print(json.dumps({"metrics_json": str(output)}, indent=2, sort_keys=True))


def compare_support_transfer(
    *,
    prediction_csvs: Sequence[str | Path],
    dino_support_csv: str | Path,
    vae_support_csv: str | Path,
    output_dir: str | Path,
    dino_support_name: str,
    vae_support_name: str,
    global_support_column: str,
    class_support_column: str,
    folds: int,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dino_support = _read_support(Path(dino_support_csv), global_support_column, class_support_column)
    vae_support = _read_support(Path(vae_support_csv), global_support_column, class_support_column)

    rows = []
    for prediction_csv in prediction_csvs:
        with Path(prediction_csv).open("r", newline="") as f:
            for row in csv.DictReader(f):
                sample_id = row["sample_id"]
                if sample_id not in dino_support:
                    raise KeyError(f"sample_id missing from DINO support: {sample_id}")
                if sample_id not in vae_support:
                    raise KeyError(f"sample_id missing from VAE support: {sample_id}")
                dino_global, dino_class = dino_support[sample_id]
                vae_global, vae_class = vae_support[sample_id]
                out = dict(row)
                out["is_error"] = str(1 - int(row["correct"]))
                out[f"{dino_support_name}_global_support"] = str(dino_global)
                out[f"{dino_support_name}_class_support"] = str(dino_class)
                out[f"{vae_support_name}_global_support"] = str(vae_global)
                out[f"{vae_support_name}_class_support"] = str(vae_class)
                rows.append(out)

    joined_csv = output / "joined_support_transfer.csv"
    _write_rows(joined_csv, rows)
    metrics = _summarize(
        rows,
        dino_support_name=dino_support_name,
        vae_support_name=vae_support_name,
        folds=folds,
    )
    metrics["prediction_csvs"] = [str(path) for path in prediction_csvs]
    metrics["dino_support_csv"] = str(dino_support_csv)
    metrics["vae_support_csv"] = str(vae_support_csv)
    metrics["joined_csv"] = str(joined_csv)
    metrics_json = output / "metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    _write_summary_csv(output / "summary.csv", metrics)
    return metrics_json


def _read_support(path: Path, global_column: str, class_column: str) -> dict[str, tuple[float, float]]:
    with path.open("r", newline="") as f:
        rows = {
            row["sample_id"]: (float(row[global_column]), float(row[class_column]))
            for row in csv.DictReader(f)
        }
    if not rows:
        raise ValueError(f"support table is empty: {path}")
    return rows


def _write_rows(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    rows: Sequence[dict[str, str]],
    *,
    dino_support_name: str,
    vae_support_name: str,
    folds: int,
) -> dict:
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row)

    out = {
        "rows": len(rows),
        "support_columns": {
            "dino_global": f"{dino_support_name}_global_support",
            "dino_class": f"{dino_support_name}_class_support",
            "vae_global": f"{vae_support_name}_global_support",
            "vae_class": f"{vae_support_name}_class_support",
        },
        "models": {},
    }
    for model_id, model_rows in sorted(by_model.items()):
        errors = [int(row["is_error"]) for row in model_rows]
        confidence_risk = [_confidence_risk(row) for row in model_rows]
        negative_margin = [-float(row["logit_margin"]) for row in model_rows]
        dino_global = [float(row[f"{dino_support_name}_global_support"]) for row in model_rows]
        dino_class = [float(row[f"{dino_support_name}_class_support"]) for row in model_rows]
        vae_global = [float(row[f"{vae_support_name}_global_support"]) for row in model_rows]
        vae_class = [float(row[f"{vae_support_name}_class_support"]) for row in model_rows]
        model_report = {
            "rows": len(model_rows),
            "errors": int(sum(errors)),
            "accuracy": 1.0 - (sum(errors) / len(errors)),
            "univariate": {
                "confidence_risk": _metrics(confidence_risk, errors, probability_like=True),
                "negative_logit_margin": _metrics(negative_margin, errors),
                f"{dino_support_name}_global_support": _metrics(dino_global, errors),
                f"{dino_support_name}_class_support": _metrics(dino_class, errors),
                f"{vae_support_name}_global_support": _metrics(vae_global, errors),
                f"{vae_support_name}_class_support": _metrics(vae_class, errors),
            },
        }
        model_report["crossfit_logistic"] = _crossfit_logistic(
            errors=errors,
            feature_sets={
                "M0_confidence_margin": [confidence_risk, negative_margin],
                f"M1_M0_plus_{vae_support_name}": [
                    confidence_risk,
                    negative_margin,
                    vae_global,
                    vae_class,
                ],
                f"M2_M0_plus_{dino_support_name}": [
                    confidence_risk,
                    negative_margin,
                    dino_global,
                    dino_class,
                ],
                f"M3_M0_plus_{vae_support_name}_plus_{dino_support_name}": [
                    confidence_risk,
                    negative_margin,
                    vae_global,
                    vae_class,
                    dino_global,
                    dino_class,
                ],
            },
            folds=folds,
        )
        _add_nested_deltas(model_report["crossfit_logistic"], dino_support_name, vae_support_name)
        out["models"][model_id] = model_report
    return out


def _confidence_risk(row: dict[str, str]) -> float:
    value = row.get("max_probability_calibrated") or row.get("max_probability_raw")
    if value in {"", "nan", "NaN", "None"}:
        value = row["max_probability_raw"]
    return 1.0 - float(value)


def _metrics(scores: Sequence[float], errors: Sequence[int], *, probability_like: bool = False) -> dict:
    return {
        "auroc": roc_auc_score(scores, errors),
        "auprc": average_precision_score(scores, errors),
        "brier": brier_score(scores, errors) if probability_like else None,
    }


def _crossfit_logistic(
    *,
    errors: Sequence[int],
    feature_sets: dict[str, Sequence[Sequence[float]]],
    folds: int,
) -> dict:
    if sum(errors) < 2 or (len(errors) - sum(errors)) < 2:
        return {"available": False, "reason": "not enough positive or negative examples"}

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except Exception as exc:  # pragma: no cover - environment-dependent
        return _crossfit_logistic_pure_python(errors=errors, feature_sets=feature_sets, folds=folds, reason=repr(exc))

    y = np.asarray(errors, dtype=np.int64)
    report = {"available": True, "backend": "sklearn", "folds": int(folds), "models": {}}
    splits = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=0)
    for name, columns in feature_sets.items():
        x = np.stack([np.asarray(column, dtype=np.float64) for column in columns], axis=1)
        pred = np.zeros(len(y), dtype=np.float64)
        for train_idx, test_idx in splits.split(x, y):
            mean = x[train_idx].mean(axis=0, keepdims=True)
            std = x[train_idx].std(axis=0, keepdims=True)
            std[std < 1.0e-8] = 1.0
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
            clf.fit((x[train_idx] - mean) / std, y[train_idx])
            pred[test_idx] = clf.predict_proba((x[test_idx] - mean) / std)[:, 1]
        scores = pred.tolist()
        report["models"][name] = {
            **_metrics(scores, list(errors), probability_like=True),
            "log_loss": _log_loss(scores, list(errors)),
        }
    return report


def _crossfit_logistic_pure_python(
    *,
    errors: Sequence[int],
    feature_sets: dict[str, Sequence[Sequence[float]]],
    folds: int,
    reason: str,
) -> dict:
    split_indices = _stratified_folds(list(errors), n_splits=int(folds), seed=0)
    report = {
        "available": True,
        "backend": "pure_python",
        "sklearn_unavailable_reason": reason,
        "folds": len(split_indices),
        "models": {},
    }
    for name, columns in feature_sets.items():
        matrix = [list(row) for row in zip(*columns)]
        pred = [0.0] * len(errors)
        for test_idx in split_indices:
            test_set = set(test_idx)
            train_idx = [idx for idx in range(len(errors)) if idx not in test_set]
            x_train_raw = [matrix[idx] for idx in train_idx]
            x_test_raw = [matrix[idx] for idx in test_idx]
            x_train, x_test = _standardize_train_test(x_train_raw, x_test_raw)
            y_train = [int(errors[idx]) for idx in train_idx]
            weights = _fit_logistic_gd(x_train, y_train, epochs=250, lr=0.1, l2=1.0e-3)
            for idx, x_vec in zip(test_idx, x_test):
                pred[idx] = _sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], x_vec)))
        report["models"][name] = {
            **_metrics(pred, list(errors), probability_like=True),
            "log_loss": _log_loss(pred, list(errors)),
        }
    return report


def _add_nested_deltas(report: dict, dino_support_name: str, vae_support_name: str) -> None:
    if not report.get("available"):
        return
    models = report.get("models", {})
    m0 = models.get("M0_confidence_margin")
    m1 = models.get(f"M1_M0_plus_{vae_support_name}")
    m2 = models.get(f"M2_M0_plus_{dino_support_name}")
    m3 = models.get(f"M3_M0_plus_{vae_support_name}_plus_{dino_support_name}")
    if not all([m0, m1, m2, m3]):
        return
    report["deltas"] = {
        f"{vae_support_name}_over_M0_auroc": m1["auroc"] - m0["auroc"],
        f"{dino_support_name}_over_M0_auroc": m2["auroc"] - m0["auroc"],
        f"{dino_support_name}_over_M0_plus_{vae_support_name}_auroc": m3["auroc"] - m1["auroc"],
        f"{vae_support_name}_over_M0_plus_{dino_support_name}_auroc": m3["auroc"] - m2["auroc"],
        f"{vae_support_name}_over_M0_log_loss": m0["log_loss"] - m1["log_loss"],
        f"{dino_support_name}_over_M0_log_loss": m0["log_loss"] - m2["log_loss"],
        f"{dino_support_name}_over_M0_plus_{vae_support_name}_log_loss": m1["log_loss"] - m3["log_loss"],
        f"{vae_support_name}_over_M0_plus_{dino_support_name}_log_loss": m2["log_loss"] - m3["log_loss"],
    }


def _write_summary_csv(path: Path, metrics: dict) -> None:
    rows = []
    for model_id, report in metrics["models"].items():
        for name, values in report["univariate"].items():
            rows.append(
                {
                    "model_id": model_id,
                    "section": "univariate",
                    "name": name,
                    "auroc": values["auroc"],
                    "auprc": values["auprc"],
                    "brier": values["brier"],
                    "log_loss": "",
                }
            )
        for name, values in report["crossfit_logistic"].get("models", {}).items():
            rows.append(
                {
                    "model_id": model_id,
                    "section": "crossfit_logistic",
                    "name": name,
                    "auroc": values["auroc"],
                    "auprc": values["auprc"],
                    "brier": values["brier"],
                    "log_loss": values["log_loss"],
                }
            )
        for name, value in report["crossfit_logistic"].get("deltas", {}).items():
            rows.append(
                {
                    "model_id": model_id,
                    "section": "crossfit_delta",
                    "name": name,
                    "auroc": value if name.endswith("_auroc") else "",
                    "auprc": "",
                    "brier": "",
                    "log_loss": value if name.endswith("_log_loss") else "",
                }
            )
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
