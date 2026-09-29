from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ada.atlas.data.manifests import ManifestRow, load_manifest_csv
from ada.atlas.hashing import hash_rows, stable_hash


def subset_cache_by_manifest(
    *,
    source_cache: str | Path,
    target_manifest_cache: str | Path,
    output_dir: str | Path,
    match_key: str = "relative_path",
    batch_size: int = 8192,
    output_dtype: str = "float32",
    overwrite: bool = False,
) -> dict[str, object]:
    """Copy source embeddings into the target manifest order.

    This is useful when a full ImageNet-1K cache has the right image features but
    incompatible sample IDs. The output preserves the target manifest rows
    exactly, so downstream exposure manifests can still index by the original
    ImageNet-100 sample IDs.
    """

    source_dir = Path(source_cache)
    target_dir = Path(target_manifest_cache)
    output = Path(output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not overwrite:
        return json.loads((output / "metadata.json").read_text())
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output_dir exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_embeddings = np.load(source_dir / "embeddings.npy", mmap_mode="r")
    source_rows = load_manifest_csv(source_dir / "manifest.csv")
    target_rows = load_manifest_csv(target_dir / "manifest.csv")
    if len(source_rows) != int(source_embeddings.shape[0]):
        raise ValueError("source manifest row count does not match source embeddings")

    source_index = _index_rows(source_rows, match_key)
    missing = [str(getattr(row, match_key)) for row in target_rows if str(getattr(row, match_key)) not in source_index]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} target rows missing from source cache by {match_key}: {preview}")

    dtype = np.dtype(output_dtype)
    out_shape = (len(target_rows), int(source_embeddings.shape[1]))
    output_embeddings = np.lib.format.open_memmap(output / "embeddings.npy", mode="w+", dtype=dtype, shape=out_shape)
    indices = np.asarray([source_index[str(getattr(row, match_key))] for row in target_rows], dtype=np.int64)
    for start in range(0, len(target_rows), int(batch_size)):
        end = min(start + int(batch_size), len(target_rows))
        output_embeddings[start:end] = np.asarray(source_embeddings[indices[start:end]], dtype=dtype)
    output_embeddings.flush()

    _write_manifest(output / "manifest.csv", target_rows)
    source_manifest_hash = hash_rows((asdict(row) for row in source_rows), prefix="manifest")
    target_manifest_hash = hash_rows((asdict(row) for row in target_rows), prefix="manifest")
    source_metadata = _read_json(source_dir / "metadata.json")
    target_metadata = _read_json(target_dir / "metadata.json")
    metadata = {
        "artifact_id": stable_hash(
            {
                "source_cache": str(source_dir),
                "target_manifest_cache": str(target_dir),
                "source_manifest_hash": source_manifest_hash,
                "target_manifest_hash": target_manifest_hash,
                "match_key": match_key,
                "embedding_shape": list(out_shape),
                "dtype": str(dtype),
            },
            prefix="cache-subset",
        ),
        "cache_id": stable_hash(
            {
                "source_cache_id": source_metadata.get("cache_id", ""),
                "target_cache_id": target_metadata.get("cache_id", ""),
                "target_manifest_hash": target_manifest_hash,
                "match_key": match_key,
                "embedding_shape": list(out_shape),
            },
            prefix="cache",
        ),
        "cache_format": "ada_embedding_cache_manifest_subset_v1",
        "source_cache": str(source_dir),
        "target_manifest_cache": str(target_dir),
        "source_cache_id": source_metadata.get("cache_id", ""),
        "target_cache_id": target_metadata.get("cache_id", ""),
        "source_manifest_hash": source_manifest_hash,
        "target_manifest_hash": target_manifest_hash,
        "match_key": match_key,
        "sample_count": len(target_rows),
        "embedding_shape": list(out_shape),
        "source_embedding_shape": list(source_embeddings.shape),
        "dtype": str(dtype),
        "source_dtype": str(source_embeddings.dtype),
        "source_dataset": source_rows[0].source_dataset if source_rows else "",
        "target_dataset": target_rows[0].source_dataset if target_rows else "",
        "split": target_rows[0].split if target_rows else "",
        "class_count": len({row.class_id for row in target_rows}),
        "preserves_target_sample_ids": True,
        "completed": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _index_rows(rows: Sequence[ManifestRow], match_key: str) -> dict[str, int]:
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for row_idx, row in enumerate(rows):
        if not hasattr(row, match_key):
            raise ValueError(f"ManifestRow has no match key: {match_key}")
        value = str(getattr(row, match_key))
        if value in index:
            duplicates.append(value)
        index[value] = row_idx
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"source cache has duplicate {match_key} values: {preview}")
    return index


def _write_manifest(path: Path, rows: Sequence[ManifestRow]) -> None:
    fieldnames = list(ManifestRow.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}
