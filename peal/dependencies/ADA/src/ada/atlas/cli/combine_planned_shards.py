from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.cache.combine import combine_embedding_caches
from ada.atlas.cache.dinov2_cls import safe_path_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine completed embedding cache shards discovered by metadata.")
    parser.add_argument("--cache-root", default=Path("artifacts/ada/atlas/embeddings"), type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--encoder", default="facebook/dinov2-with-registers-base")
    parser.add_argument("--feature", default="cls")
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.cache_root) / args.dataset / args.split / safe_path_name(args.encoder) / args.feature
    if not base.exists():
        raise FileNotFoundError(f"cache base does not exist: {base}")
    shards: dict[int, Path] = {}
    for candidate in sorted(path for path in base.iterdir() if path.is_dir()):
        completed = candidate / "COMPLETED"
        metadata_path = candidate / "metadata.json"
        if not completed.exists() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("dataset") != args.dataset or metadata.get("split") != args.split:
            continue
        if metadata.get("encoder") != args.encoder or metadata.get("feature") != args.feature:
            continue
        if metadata.get("num_shards") != int(args.num_shards):
            continue
        shard_index = metadata.get("shard_index")
        if shard_index is None:
            continue
        shard = int(shard_index)
        previous = shards.get(shard)
        if previous is not None:
            raise ValueError(f"multiple completed caches found for shard {shard}: {previous} and {candidate}")
        shards[shard] = candidate

    missing = [idx for idx in range(int(args.num_shards)) if idx not in shards]
    if missing:
        raise FileNotFoundError(f"missing completed shards for {args.dataset}/{args.split}: {missing}")

    input_dirs = [shards[idx] for idx in range(int(args.num_shards))]
    output = combine_embedding_caches(input_dirs, args.output_dir, overwrite=bool(args.overwrite))
    print(json.dumps({"output_dir": str(output), "input_dirs": [str(path) for path in input_dirs]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
