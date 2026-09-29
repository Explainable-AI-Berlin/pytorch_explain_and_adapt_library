from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.cache.dinov2_cls import cache_dinov2_cls, plan_dinov2_cls_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache DINOv2 CLS embeddings for an ImageFolder dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--encoder", default="facebook/dinov2-with-registers-base")
    parser.add_argument("--feature", default="cls")
    parser.add_argument("--output-root", default=Path("artifacts/ada/atlas/embeddings"), type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--encoder-input-size", default=224, type=int)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
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
    manifest, plan = plan_dinov2_cls_cache(
        dataset=args.dataset,
        split=args.split,
        root=args.root,
        encoder=args.encoder,
        feature=args.feature,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        encoder_input_size=args.encoder_input_size,
        precision=args.precision,
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
    output_dir = cache_dinov2_cls(manifest=manifest, plan=plan, overwrite=args.overwrite)
    plan["output_dir"] = str(output_dir)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
