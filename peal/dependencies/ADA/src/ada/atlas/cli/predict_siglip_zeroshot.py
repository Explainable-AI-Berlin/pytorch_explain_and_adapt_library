from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

from ada.atlas.predictions.siglip_zeroshot import siglip_zero_shot_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline SigLIP zero-shot prediction rows.")
    parser.add_argument("--query-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="google/siglip2-base-patch16-256")
    parser.add_argument("--imagenet-meta", default=os.path.join(os.environ.get("PEAL_DATA", "datasets"), "imagenet_torchvision/data/meta.bin"))
    parser.add_argument("--template", default="a photo of a {}")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = siglip_zero_shot_predictions(
        query_cache=args.query_cache,
        output_dir=args.output_dir,
        model_name=args.model_name,
        imagenet_meta=args.imagenet_meta,
        template=args.template,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
    )
    print(json.dumps({"predictions_csv": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
