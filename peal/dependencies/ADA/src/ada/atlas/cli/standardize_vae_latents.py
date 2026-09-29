from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.cache.vae_latents import standardize_vae_latent_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply train-only coordinate standardization to a VAE posterior-mean cache.")
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--standardization-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--storage-dtype", default="float32", choices=("float16", "float32"))
    parser.add_argument(
        "--no-scale-by-sqrt-dim",
        action="store_true",
        help="Do not divide standardized vectors by sqrt(D). This changes distance scale but not neighbour ordering.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = standardize_vae_latent_cache(
        input_cache=args.input_cache,
        standardization_dir=args.standardization_dir,
        output_dir=args.output_dir,
        batch_size=int(args.batch_size),
        mmap=bool(args.mmap),
        storage_dtype=str(args.storage_dtype),
        scale_by_sqrt_dim=not bool(args.no_scale_by_sqrt_dim),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
