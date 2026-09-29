from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from ada.atlas.support.score_cache import score_cache_knn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score query CLS cache against a reference cache with exact kNN support.")
    parser.add_argument("--query-cache", required=True, type=Path)
    parser.add_argument("--reference-cache", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--k", default="5,10,50", help="Comma- or colon-separated k values, for example 5:10:50.")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--class-conditional", action="store_true")
    parser.add_argument("--mmap", action="store_true", help="Load embedding .npy files with numpy mmap_mode='r'.")
    parser.add_argument(
        "--reference-shard-size",
        default=None,
        type=int,
        help="Optional exact-search reference shard size. Keeps global top-k exact while reducing score-matrix memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = [int(item.strip()) for item in re.split(r"[,:]", args.k) if item.strip()]
    output = score_cache_knn(
        query_cache=args.query_cache,
        reference_cache=args.reference_cache,
        output_csv=args.output_csv,
        k_values=k_values,
        batch_size=args.batch_size,
        device=args.device,
        leave_one_out=args.leave_one_out,
        class_conditional=args.class_conditional,
        mmap=args.mmap,
        reference_shard_size=args.reference_shard_size,
    )
    print(json.dumps({"output_csv": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
