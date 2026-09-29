from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ada.atlas.data.manifests import ImageFolderManifest, load_manifest_csv, write_manifest_files
from ada.atlas.hashing import stable_hash


def combine_embedding_caches(input_dirs: Sequence[str | Path], output_dir: str | Path, *, overwrite: bool = False) -> Path:
    if not input_dirs:
        raise ValueError("input_dirs must not be empty")
    output = Path(output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"Combined cache already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    import numpy as np

    arrays = []
    all_rows = []
    input_metadata = []
    first_meta = None

    for raw_dir in input_dirs:
        cache_dir = Path(raw_dir)
        if not (cache_dir / "COMPLETED").exists():
            raise FileNotFoundError(f"Input cache is not completed: {cache_dir}")
        embeddings_path = cache_dir / "embeddings.npy"
        manifest_path = cache_dir / "manifest.csv"
        metadata_path = cache_dir / "metadata.json"
        if not embeddings_path.exists() or not manifest_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Input cache is missing embeddings, manifest, or metadata: {cache_dir}")

        embeddings = np.load(embeddings_path)
        rows = load_manifest_csv(manifest_path)
        if embeddings.shape[0] != len(rows):
            raise ValueError(f"Row mismatch in {cache_dir}: embeddings={embeddings.shape[0]} manifest={len(rows)}")
        arrays.append(embeddings)
        all_rows.extend(rows)
        meta = json.loads(metadata_path.read_text())
        input_metadata.append(
            {
                "cache_dir": str(cache_dir),
                "cache_id": meta.get("cache_id"),
                "manifest_hash": meta.get("cached_manifest_hash") or meta.get("manifest", {}).get("manifest_hash"),
                "sample_count": int(embeddings.shape[0]),
            }
        )
        if first_meta is None:
            first_meta = meta

    sample_ids = [row.sample_id for row in all_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Input caches contain duplicate sample IDs")

    combined_embeddings = np.concatenate(arrays, axis=0)
    root = str(first_meta.get("source_manifest", first_meta.get("manifest", {})).get("root", "")) if first_meta else ""
    dataset = str(first_meta.get("dataset", "combined")) if first_meta else "combined"
    split = str(first_meta.get("split", "combined")) if first_meta else "combined"
    class_names = tuple(first_meta.get("source_manifest", first_meta.get("manifest", {})).get("class_names", ())) if first_meta else ()
    if not class_names:
        class_names = tuple(sorted({row.class_name for row in all_rows}))
    manifest = ImageFolderManifest(
        dataset=dataset,
        split=split,
        root=root,
        rows=tuple(all_rows),
        class_names=tuple(class_names),
        skipped_dirs=(),
        skipped_files=(),
        manifest_hash=stable_hash([asdict(row) for row in all_rows], prefix="manifest"),
    )

    np.save(output / "embeddings.npy", combined_embeddings)
    write_manifest_files(manifest, output)
    combined_meta = {
        "cache_id": stable_hash(input_metadata, prefix="combined-cache"),
        "dataset": dataset,
        "split": split,
        "embedding_shape": list(combined_embeddings.shape),
        "embedding_dtype": str(combined_embeddings.dtype),
        "cached_manifest_hash": manifest.manifest_hash,
        "inputs": input_metadata,
    }
    (output / "metadata.json").write_text(json.dumps(combined_meta, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return output
