from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ada.atlas.cache.subset_by_manifest import subset_cache_by_manifest
from ada.atlas.data.manifests import ImageFolderManifest, ManifestRow, write_manifest_files


class SubsetCacheByManifestTests(unittest.TestCase):
    def test_subset_reorders_embeddings_and_preserves_target_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            source_rows = [
                ManifestRow("in1k-a", "n0001/a.JPEG", "imagenet1k", "train", 7, "n0001"),
                ManifestRow("in1k-b", "n0001/b.JPEG", "imagenet1k", "train", 7, "n0001"),
                ManifestRow("in1k-c", "n0002/c.JPEG", "imagenet1k", "train", 9, "n0002"),
            ]
            target_rows = [
                ManifestRow("in100-c", "n0002/c.JPEG", "imagenet100", "train", 1, "n0002"),
                ManifestRow("in100-a", "n0001/a.JPEG", "imagenet100", "train", 0, "n0001"),
            ]
            source = _write_cache(tmp, "source", np.asarray([[1, 2], [3, 4], [5, 6]], dtype="float32"), source_rows)
            target = _write_cache(tmp, "target", np.zeros((2, 2), dtype="float32"), target_rows)
            output = tmp / "subset"

            metadata = subset_cache_by_manifest(source_cache=source, target_manifest_cache=target, output_dir=output)
            subset = np.load(output / "embeddings.npy")
            out_rows = (output / "manifest.csv").read_text()

        np.testing.assert_array_equal(subset, np.asarray([[5, 6], [1, 2]], dtype="float32"))
        self.assertIn("in100-c", out_rows)
        self.assertIn("in100-a", out_rows)
        self.assertNotIn("in1k-c", out_rows)
        self.assertTrue(metadata["preserves_target_sample_ids"])
        self.assertEqual(metadata["target_dataset"], "imagenet100")

    def test_missing_target_relative_path_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            source = _write_cache(
                tmp,
                "source",
                np.asarray([[1, 2]], dtype="float32"),
                [ManifestRow("in1k-a", "n0001/a.JPEG", "imagenet1k", "val", 0, "n0001")],
            )
            target = _write_cache(
                tmp,
                "target",
                np.zeros((1, 2), dtype="float32"),
                [ManifestRow("in100-z", "n0001/z.JPEG", "imagenet100", "val", 0, "n0001")],
            )
            with self.assertRaisesRegex(ValueError, "missing from source cache"):
                subset_cache_by_manifest(source_cache=source, target_manifest_cache=target, output_dir=tmp / "subset")


def _write_cache(root: Path, name: str, embeddings: np.ndarray, rows: list[ManifestRow]) -> Path:
    cache = root / name
    cache.mkdir(parents=True)
    np.save(cache / "embeddings.npy", embeddings.astype("float32"))
    manifest = ImageFolderManifest(
        dataset=rows[0].source_dataset,
        split=rows[0].split,
        root=str(root),
        rows=tuple(rows),
        class_names=tuple(sorted({row.class_name for row in rows})),
        skipped_dirs=(),
        skipped_files=(),
        manifest_hash=f"manifest-{name}",
    )
    write_manifest_files(manifest, cache)
    metadata = json.loads((cache / "metadata.json").read_text())
    metadata["cache_id"] = f"cache-{name}"
    (cache / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return cache


if __name__ == "__main__":
    unittest.main()
