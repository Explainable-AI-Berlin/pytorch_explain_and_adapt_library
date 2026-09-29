#!/usr/bin/env python3
"""Export a compact, memory-mappable CLS-only statistics bundle.

Stage-2 cache shards also contain very large patch latents. This exporter reads
each shard once and writes only the CLS vectors and row identifiers as ordinary
NumPy arrays suitable for statistical analysis and transfer to collaborators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(output_dir: Path, metadata: dict) -> None:
    text = f"""# DINOv2-L CLS Statistics Bundle

This directory contains the ImageNet-1K training CLS vectors used by the
RAEv2 Stage-2 experiments. Patch latents and images are intentionally omitted.

## Representation

- Encoder: `{metadata['encoder_class']}`
- Source: `{metadata['condition_source']}`
- Selected layers: `{metadata['selected_layers']}`
- Rows: `{metadata['num_rows']:,}`
- Dimensions: `{metadata['cls_dim']}`
- Stored analysis dtype: `float32`
- Original cache dtype/precision: `{metadata['source_storage_precision']}`
- Row order: premixed physical cache order; use `source_index.int32.npy` as the
  stable sample identity.

`cls.float32.npy` contains raw encoder coordinates. It is not PCA projected,
L2-normalized, or standardized. Train mean and standard deviation are provided
separately.

## Files

- `cls.float32.npy`: `[N, 1024]` raw CLS vectors in a memory-mappable array.
- `y.int16.npy`: ImageNet class index for each row.
- `source_index.int32.npy`: stable ImageFolder source index.
- `view.uint8.npy`: extraction view (`0` means the original deterministic crop).
- `cls_norm.float32.npy`: raw row L2 norm, provided as a convenience sidecar.
- `cls_mean.float32.npy`, `cls_std.float32.npy`: train coordinate statistics.
- `class_to_idx.json`: WNID to integer-label mapping.
- `metadata.json`: provenance and schema.
- `sha256sums.txt`: integrity hashes.

## Python

```python
from pathlib import Path
import numpy as np

root = Path("in1k_dinov2l_cls_statistics")
x = np.load(root / "cls.float32.npy", mmap_mode="r")
y = np.load(root / "y.int16.npy", mmap_mode="r")
source_id = np.load(root / "source_index.int32.npy", mmap_mode="r")
mean = np.load(root / "cls_mean.float32.npy")
std = np.load(root / "cls_std.float32.npy")

# Work in chunks; this does not materialize all N x 1024 values.
rows = np.flatnonzero(y == 682)
chunk = np.asarray(x[rows[:1000]], dtype=np.float32)
standardized = (chunk - mean) / np.maximum(std, 1e-6)
```

For class-bootstrap analyses, the independent grouping unit should generally be
the ImageNet class or a prespecified region, not each embedding coordinate or
image row. Fit PCA, regressions, density models, and normalizers on training rows
only when later joining to validation outcomes.

## Transfer

Prefer resumable transfer because the CLS matrix is several GiB:

```bash
rsync -ah --partial --info=progress2 \\
  hydra:{output_dir.resolve()}/ \\
  ./in1k_dinov2l_cls_statistics/

cd in1k_dinov2l_cls_statistics
sha256sum -c sha256sums.txt
```

Share only with collaborators who are authorized to use the underlying
ImageNet-derived representation artifacts.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.cache_root / "metadata.json"
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cls_metadata = source_metadata.get("cls_condition", {})
    if not bool(cls_metadata.get("included", False)):
        raise ValueError(f"Cache does not declare CLS tensors: {args.cache_root}")

    num_rows = int(source_metadata["num_samples"])
    cls_shape = cls_metadata.get("shape", [])
    if len(cls_shape) != 1:
        raise ValueError(f"Expected one-dimensional CLS rows, got {cls_shape}")
    cls_dim = int(cls_shape[0])

    outputs = {
        "cls": args.output_dir / "cls.float32.npy",
        "y": args.output_dir / "y.int16.npy",
        "source_index": args.output_dir / "source_index.int32.npy",
        "view": args.output_dir / "view.uint8.npy",
        "cls_norm": args.output_dir / "cls_norm.float32.npy",
    }
    if not args.overwrite:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Output arrays already exist: {existing}")

    cls_out = np.lib.format.open_memmap(outputs["cls"], mode="w+", dtype=np.float32, shape=(num_rows, cls_dim))
    y_out = np.lib.format.open_memmap(outputs["y"], mode="w+", dtype=np.int16, shape=(num_rows,))
    source_out = np.lib.format.open_memmap(outputs["source_index"], mode="w+", dtype=np.int32, shape=(num_rows,))
    view_out = np.lib.format.open_memmap(outputs["view"], mode="w+", dtype=np.uint8, shape=(num_rows,))
    norm_out = np.lib.format.open_memmap(outputs["cls_norm"], mode="w+", dtype=np.float32, shape=(num_rows,))

    offset = 0
    for shard_number, entry in enumerate(source_metadata["shards"]):
        shard_path = args.cache_root / entry["file"]
        payload = torch.load(shard_path, map_location="cpu")
        for key in ("cls", "y", "source_index", "view"):
            if key not in payload:
                raise KeyError(f"Missing {key!r} in {shard_path}")
        cls = payload["cls"].float()
        count = int(cls.shape[0])
        if tuple(cls.shape[1:]) != (cls_dim,):
            raise ValueError(f"Unexpected CLS shape {tuple(cls.shape)} in {shard_path}")
        stop = offset + count
        cls_np = cls.numpy()
        cls_out[offset:stop] = cls_np
        y_out[offset:stop] = payload["y"].numpy().astype(np.int16, copy=False)
        source_out[offset:stop] = payload["source_index"].numpy().astype(np.int32, copy=False)
        view_out[offset:stop] = payload["view"].numpy().astype(np.uint8, copy=False)
        norm_out[offset:stop] = np.linalg.norm(cls_np, axis=1)
        offset = stop
        del payload, cls, cls_np
        if shard_number % 100 == 0:
            print(f"[export] shard={shard_number:,} rows={offset:,}/{num_rows:,}", flush=True)

    if offset != num_rows:
        raise RuntimeError(f"Exported {offset} rows, expected {num_rows}")
    for array in (cls_out, y_out, source_out, view_out, norm_out):
        array.flush()
    del cls_out, y_out, source_out, view_out, norm_out

    stats_path = args.cache_root / str(cls_metadata.get("stats_file", "cls_stats.pt"))
    stats = torch.load(stats_path, map_location="cpu")
    np.save(args.output_dir / "cls_mean.float32.npy", stats["mean"].float().numpy())
    np.save(args.output_dir / "cls_std.float32.npy", stats["std"].float().numpy())
    (args.output_dir / "class_to_idx.json").write_text(
        json.dumps(source_metadata.get("class_to_idx", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_ids = np.load(outputs["source_index"], mmap_mode="r")
    views = np.load(outputs["view"], mmap_mode="r")
    keys = np.asarray(source_ids, dtype=np.int64) * 256 + np.asarray(views, dtype=np.int64)
    unique_rows = int(np.unique(keys).size)
    del keys
    if unique_rows != num_rows:
        raise RuntimeError(f"Expected {num_rows} unique source/view keys, found {unique_rows}")

    bundle_metadata = {
        "schema_version": 1,
        "source_cache_root": str(args.cache_root.resolve()),
        "source_cache_metadata": str(metadata_path.resolve()),
        "source_cache_metadata_sha256": sha256(metadata_path),
        "num_rows": num_rows,
        "cls_dim": cls_dim,
        "analysis_dtype": "float32",
        "source_storage_precision": str(source_metadata.get("output_dtype", "unknown")),
        "encoder_class": cls_metadata.get("encoder_class"),
        "condition_source": cls_metadata.get("condition_source"),
        "selected_layers": cls_metadata.get("selected_layers"),
        "token": cls_metadata.get("token"),
        "split": source_metadata.get("split"),
        "views": source_metadata.get("views"),
        "row_order": "premixed physical cache order",
        "stable_identity": ["source_index", "view"],
        "unique_source_view_rows": unique_rows,
        "files": {
            "cls": {"path": outputs["cls"].name, "shape": [num_rows, cls_dim], "dtype": "float32"},
            "y": {"path": outputs["y"].name, "shape": [num_rows], "dtype": "int16"},
            "source_index": {"path": outputs["source_index"].name, "shape": [num_rows], "dtype": "int32"},
            "view": {"path": outputs["view"].name, "shape": [num_rows], "dtype": "uint8"},
            "cls_norm": {"path": outputs["cls_norm"].name, "shape": [num_rows], "dtype": "float32"},
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(bundle_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(args.output_dir, bundle_metadata)

    hash_targets = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "sha256sums.txt"
    )
    with (args.output_dir / "sha256sums.txt").open("w", encoding="utf-8") as handle:
        for path in hash_targets:
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
