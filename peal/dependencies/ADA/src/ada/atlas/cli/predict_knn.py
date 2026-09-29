from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.predictions.knn_predict import predict_knn_from_caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DINO kNN prediction rows from atlas caches.")
    parser.add_argument("--query-cache", required=True, type=Path)
    parser.add_argument("--reference-cache", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--k", default="1:5:10")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = [int(item) for item in args.k.replace(",", ":").split(":") if item]
    output = predict_knn_from_caches(
        query_cache=args.query_cache,
        reference_cache=args.reference_cache,
        output_csv=args.output_csv,
        k_values=k_values,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps({"output_csv": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
