#!/usr/bin/env python3
"""Download and verify the pinned RAEv2 DINOv2-L decoder and statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "nyu-visionx/RAEv2-models"
REVISION = "9770b7b980fa1875c8e6d65f226c615c0ce908a8"
RELATIVE_ROOT = Path("stage1/imagenet/dinov2l-k1")
EXPECTED = {
    "decoder.pt": {
        "bytes": 1662766063,
        "sha256": "12e40ca9d74b7c45441256d00ada5e6a6b109c1af13f520faf838f13387c861b",
    },
    "stats.pt": {
        "bytes": 2098901,
        "sha256": "b82fe80fae9b27f07e324ecf4222c3aa1a190dba8536343d1a500a718a21b90d",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        repo_type="model",
        local_dir=args.output_root,
        allow_patterns=[f"{RELATIVE_ROOT}/**"],
    )

    artifact_root = args.output_root / RELATIVE_ROOT
    observed = {}
    for name, expected in EXPECTED.items():
        path = artifact_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {actual} != {expected}")
        observed[name] = actual

    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "artifact_path": str(RELATIVE_ROOT),
        "files": observed,
    }
    manifest_path = artifact_root / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(artifact_root.resolve())


if __name__ == "__main__":
    main()
