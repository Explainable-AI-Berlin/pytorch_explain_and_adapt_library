from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.predictions.join_support import summarize_predictions_with_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join prediction rows to atlas support scores and summarize error signals.")
    parser.add_argument("--support-csv", required=True, type=Path)
    parser.add_argument("--prediction-csv", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--support-columns",
        default="support_k5_kth_distance,support_k10_kth_distance,support_k50_kth_distance,"
        "class_support_k5_kth_distance,class_support_k10_kth_distance,class_support_k50_kth_distance",
        help="Comma-separated support/sidecar score columns to copy from --support-csv and evaluate.",
    )
    parser.add_argument("--primary-support-column", default="support_k50_kth_distance")
    parser.add_argument("--primary-class-support-column", default="class_support_k50_kth_distance")
    parser.add_argument("--bins", default=10, type=int)
    parser.add_argument("--skip-crossfit", action="store_true", help="Skip cross-fit logistic add-on summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = summarize_predictions_with_support(
        support_csv=args.support_csv,
        prediction_csvs=args.prediction_csv,
        output_dir=args.output_dir,
        support_columns=[column.strip() for column in args.support_columns.split(",") if column.strip()],
        primary_support_column=args.primary_support_column,
        primary_class_support_column=args.primary_class_support_column,
        bins=args.bins,
        include_crossfit=not args.skip_crossfit,
    )
    print(json.dumps({"metrics_json": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
