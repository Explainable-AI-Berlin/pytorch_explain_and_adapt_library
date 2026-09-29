from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ada.atlas.metrics.deciles import quantile_summary
from ada.atlas.metrics.error_detection import average_precision_score, confidence_to_error_risk, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate error-detection scores for an ADA atlas score table.")
    parser.add_argument("--scores-csv", required=True, type=Path)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--error-column", default="is_error")
    parser.add_argument("--confidence-column", default="confidence_raw")
    parser.add_argument("--high-score-is-risky", action="store_true")
    parser.add_argument("--bins", default=10, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.scores_csv.open("r", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    labels = [int(row[args.error_column]) for row in rows]
    scores = [float(row[args.score_column]) for row in rows]
    confidence = [float(row[args.confidence_column]) for row in rows]
    confidence_risk = confidence_to_error_risk(confidence)
    report = {
        "rows": len(rows),
        "score_column": args.score_column,
        "error_column": args.error_column,
        "confidence_column": args.confidence_column,
        "score_auroc": roc_auc_score(scores, labels),
        "score_auprc": average_precision_score(scores, labels),
        "confidence_risk_auroc": roc_auc_score(confidence_risk, labels),
        "confidence_risk_auprc": average_precision_score(confidence_risk, labels),
        "deciles": quantile_summary(
            rows,
            score_column=args.score_column,
            error_column=args.error_column,
            confidence_column=args.confidence_column,
            bins=args.bins,
            high_score_is_risky=args.high_score_is_risky,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
