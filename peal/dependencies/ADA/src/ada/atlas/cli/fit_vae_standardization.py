from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.cache.vae_latents import fit_vae_standardization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit train-only coordinate standardization for VAE posterior-mean caches.")
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epsilon", default=1.0e-6, type=float)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = fit_vae_standardization(
        train_cache=args.train_cache,
        output_dir=args.output_dir,
        epsilon=float(args.epsilon),
        batch_size=int(args.batch_size),
        mmap=bool(args.mmap),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
