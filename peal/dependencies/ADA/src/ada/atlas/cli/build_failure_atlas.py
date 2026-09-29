from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.failure.signatures import build_failure_atlas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-model failure signatures from joined ADA atlas predictions.")
    parser.add_argument("--joined-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-csv", default=None, type=Path)
    parser.add_argument("--cache-metadata-json", default=None, type=Path)
    parser.add_argument("--task-model", default="dino_linear_probe")
    parser.add_argument("--independent-model", default="resnet18_imagenet1k_restricted100")
    parser.add_argument("--purity-model", default="dino_knn_k10")
    parser.add_argument("--support-column", default="class_support_k50_kth_distance")
    parser.add_argument("--weak-support-fraction", default=0.10, type=float)
    parser.add_argument("--max-exemplars", default=80, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_failure_atlas(
        joined_csv=args.joined_csv,
        output_dir=args.output_dir,
        manifest_csv=args.manifest_csv,
        cache_metadata_json=args.cache_metadata_json,
        task_model=args.task_model,
        independent_model=args.independent_model,
        purity_model=args.purity_model,
        support_column=args.support_column,
        weak_support_fraction=args.weak_support_fraction,
        max_exemplars=args.max_exemplars,
    )
    print(json.dumps({"failure_atlas_summary": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
