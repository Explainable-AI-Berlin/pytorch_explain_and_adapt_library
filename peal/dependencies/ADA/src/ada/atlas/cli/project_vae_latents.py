from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.cache.vae_latents import project_vae_latent_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project a raw VAE posterior-mean cache with a fitted train-only PCA.")
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--projection-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = project_vae_latent_cache(
        input_cache=args.input_cache,
        projection_dir=args.projection_dir,
        output_dir=args.output_dir,
        batch_size=int(args.batch_size),
        mmap=bool(args.mmap),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
