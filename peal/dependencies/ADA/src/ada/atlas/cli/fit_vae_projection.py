from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.cache.vae_latents import fit_vae_pca_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a train-only PCA projection for VAE posterior-mean caches.")
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pca-dim", default=512, type=int)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--epsilon", default=1.0e-6, type=float)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = fit_vae_pca_projection(
        train_cache=args.train_cache,
        output_dir=args.output_dir,
        pca_dim=int(args.pca_dim),
        standardize=not bool(args.no_standardize),
        whiten=bool(args.whiten),
        epsilon=float(args.epsilon),
        batch_size=int(args.batch_size),
        mmap=bool(args.mmap),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
