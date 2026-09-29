from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.cache.subset_by_manifest import subset_cache_by_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subset/reorder an embedding cache to match another cache manifest.")
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--target-manifest-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-key", default="relative_path")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = subset_cache_by_manifest(
        source_cache=args.source_cache,
        target_manifest_cache=args.target_manifest_cache,
        output_dir=args.output_dir,
        match_key=str(args.match_key),
        batch_size=int(args.batch_size),
        output_dtype=str(args.output_dtype),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
