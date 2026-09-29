from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.cache.vae_latents import cache_vae_latents, plan_vae_latent_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache deterministic VAE posterior-mean latents for an ImageFolder dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vae-type", default="sdvae-ema", choices=["sdvae-ema", "sdvae-mse", "sdxl-vae"])
    parser.add_argument("--output-root", default=Path("artifacts/ada/atlas/vae_latents"), type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--storage-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--no-logvar", action="store_true", help="Do not store full posterior log-variance tensor.")
    parser.add_argument("--no-raw-mean", action="store_true", help="Do not store a separate raw posterior-mean tensor.")
    parser.add_argument("--save-reconstruction-mse", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow diffusers to download the requested VAE.")
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--end-index", default=None, type=int)
    parser.add_argument("--shard-index", default=None, type=int)
    parser.add_argument("--num-shards", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Return an existing completed cache instead of failing.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, plan = plan_vae_latent_cache(
        dataset=args.dataset,
        split=args.split,
        root=args.root,
        vae_type=args.vae_type,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        precision=args.precision,
        storage_dtype=args.storage_dtype,
        save_logvar=not bool(args.no_logvar),
        save_raw_mean=not bool(args.no_raw_mean),
        save_reconstruction_mse=bool(args.save_reconstruction_mse),
        allow_download=bool(args.allow_download),
        max_samples=args.max_samples,
        start_index=args.start_index,
        end_index=args.end_index,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    plan["dry_run"] = bool(args.dry_run)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    planned_output = Path(plan["output_dir"])
    if args.reuse_existing and (planned_output / "COMPLETED").exists():
        plan["output_dir"] = str(planned_output)
        plan["reused_existing"] = True
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    output_dir = cache_vae_latents(manifest=manifest, plan=plan, overwrite=bool(args.overwrite))
    plan["output_dir"] = str(output_dir)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
