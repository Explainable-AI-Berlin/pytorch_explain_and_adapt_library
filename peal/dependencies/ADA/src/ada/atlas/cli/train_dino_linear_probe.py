from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.predictions.dino_linear_probe import train_dino_linear_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DINO CLS linear probe and emit validation predictions.")
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--query-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--lr", default=1.0e-2, type=float)
    parser.add_argument("--weight-decay", default=1.0e-4, type=float)
    parser.add_argument("--calibration-fraction", default=0.1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--model-id", default="dino_linear_probe")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = train_dino_linear_probe(
        train_cache=args.train_cache,
        query_cache=args.query_cache,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        model_id=args.model_id,
        device=args.device,
    )
    print(json.dumps({"predictions_csv": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
