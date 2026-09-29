from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.cache.combine import combine_embedding_caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine completed ADA atlas embedding cache shards.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-dir", action="append", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = combine_embedding_caches(args.input_dir, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({"output_dir": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
