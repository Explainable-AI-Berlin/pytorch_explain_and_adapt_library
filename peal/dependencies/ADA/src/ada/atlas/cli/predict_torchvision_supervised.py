from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

from ada.atlas.predictions.torchvision_supervised import torchvision_imagenet_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate restricted ImageNet-100 predictions from an ImageNet-1K torchvision model.")
    parser.add_argument("--query-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument("--imagenet-meta", default=os.path.join(os.environ.get("PEAL_DATA", "datasets"), "imagenet_torchvision/data/meta.bin"))
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = torchvision_imagenet_predictions(
        query_cache=args.query_cache,
        output_dir=args.output_dir,
        model_name=args.model_name,
        imagenet_meta=args.imagenet_meta,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
    )
    print(json.dumps({"predictions_csv": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
