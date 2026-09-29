from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a pinned Qwen3-VL snapshot for ADA interpretation.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    parser.add_argument("--output-root", type=Path, default=Path("external/models"))
    parser.add_argument("--max-hashed-bytes", type=int, default=64 * 1024 * 1024 * 1024)
    return parser.parse_args()


def main() -> None:
    from huggingface_hub import snapshot_download
    import transformers

    args = parse_args()
    safe_name = args.model_id.replace("/", "__")
    output_dir = args.output_root / safe_name / "snapshots" / str(args.revision)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=output_dir,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "*.py",
                "merges.txt",
                "vocab.json",
                "tokenizer.*",
                "preprocessor_config.json",
                "processor_config.json",
                "chat_template.json",
            ],
        )
    )
    files = sorted(path for path in snapshot_path.rglob("*") if path.is_file())
    file_rows = []
    for path in files:
        rel = path.relative_to(snapshot_path).as_posix()
        size = path.stat().st_size
        row = {"path": rel, "size": size}
        if size <= int(args.max_hashed_bytes):
            row["sha256"] = _sha256(path)
        else:
            row["sha256"] = ""
            row["sha256_skipped_reason"] = f"larger than {int(args.max_hashed_bytes)} bytes"
        file_rows.append(row)
    manifest = {
        "model_id": args.model_id,
        "revision": args.revision,
        "snapshot_dir": str(snapshot_path.resolve()),
        "transformers_version": transformers.__version__,
        "files": file_rows,
        "file_count": len(file_rows),
        "total_bytes": sum(int(row["size"]) for row in file_rows),
    }
    (snapshot_path / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"snapshot_dir": str(snapshot_path.resolve()), "asset_manifest": str(snapshot_path / "asset_manifest.json")}, indent=2))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
