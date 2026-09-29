from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


DEFAULT_MODEL_ID = "google/siglip2-base-patch16-256"
DEFAULT_REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
DEFAULT_OUTPUT_ROOT = Path("external/models/google__siglip2-base-patch16-256")
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a pinned complete SigLIP2 shared image/text snapshot.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.output_root) / "snapshots" / str(args.revision)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=str(args.model_id),
        revision=str(args.revision),
        local_dir=snapshot_dir,
        allow_patterns=list(REQUIRED_FILES),
        local_files_only=bool(args.local_files_only),
    )
    missing = [name for name in REQUIRED_FILES if not (snapshot_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Incomplete SigLIP2 snapshot at {snapshot_dir}: missing {missing}")

    manifest = build_asset_manifest(
        model_id=str(args.model_id),
        revision=str(args.revision),
        snapshot_dir=snapshot_dir,
        required_files=REQUIRED_FILES,
    )
    manifest_path = snapshot_dir / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"snapshot_dir": str(snapshot_dir.resolve()), "asset_manifest": str(manifest_path.resolve())}, indent=2, sort_keys=True))


def build_asset_manifest(
    *,
    model_id: str,
    revision: str,
    snapshot_dir: Path,
    required_files: Sequence[str],
) -> dict[str, object]:
    transformers_version, tokenizer_class, image_processor_class = _transformers_metadata(snapshot_dir)
    return {
        "model_id": model_id,
        "revision": revision,
        "snapshot_dir": str(snapshot_dir),
        "required_files": list(required_files),
        "transformers_version": transformers_version,
        "huggingface_hub_version": _package_version("huggingface_hub"),
        "sentencepiece_version": _package_version("sentencepiece"),
        "tokenizer_class": tokenizer_class,
        "image_processor_class": image_processor_class,
        "file_sha256": {name: sha256_file(snapshot_dir / name) for name in required_files},
    }


def _transformers_metadata(snapshot_dir: Path) -> tuple[str, str, str]:
    try:
        import transformers
        from transformers import AutoImageProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True, use_fast=True)
        image_processor = AutoImageProcessor.from_pretrained(snapshot_dir, local_files_only=True)
        return transformers.__version__, tokenizer.__class__.__name__, image_processor.__class__.__name__
    except Exception as exc:
        return _package_version("transformers"), f"UNRESOLVED:{exc.__class__.__name__}", f"UNRESOLVED:{exc.__class__.__name__}"


def _package_version(package: str) -> str:
    try:
        from importlib.metadata import version

        return str(version(package))
    except Exception:
        return "unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
