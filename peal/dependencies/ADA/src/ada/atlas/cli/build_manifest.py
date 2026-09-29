from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada.atlas.data.manifests import build_imagefolder_manifest, manifest_summary, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an immutable ImageFolder manifest for ADA atlas experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("artifacts/ada/atlas/manifests"), type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_imagefolder_manifest(args.root, dataset=args.dataset, split=args.split)
    summary = manifest_summary(manifest)
    summary["dry_run"] = bool(args.dry_run)
    summary["planned_output_dir"] = str(args.output_root / args.dataset / args.split / manifest.manifest_hash)
    if not args.dry_run:
        output_dir = write_manifest(manifest, args.output_root)
        summary["output_dir"] = str(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
