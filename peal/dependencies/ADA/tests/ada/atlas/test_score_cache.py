from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from ada.atlas.data.manifests import ImageFolderManifest, ManifestRow, write_manifest_files
from ada.atlas.support.score_cache import score_cache_knn


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if NUMPY_AVAILABLE:
    import numpy as np
else:
    np = None


def _write_cache(root: Path, name: str, embeddings: np.ndarray, rows: list[ManifestRow]) -> Path:
    cache = root / name
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "embeddings.npy", embeddings.astype("float32"))
    manifest = ImageFolderManifest(
        dataset="toy",
        split=name,
        root=str(root),
        rows=tuple(rows),
        class_names=("zero", "one"),
        skipped_dirs=(),
        skipped_files=(),
        manifest_hash=f"manifest-{name}",
    )
    write_manifest_files(manifest, cache)
    return cache


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for score_cache_knn")
class ScoreCacheKNNTests(unittest.TestCase):
    def test_sharded_exact_matches_dense_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            reference_rows = [
                ManifestRow("r0", "r0.jpg", "toy", "ref", 0, "zero"),
                ManifestRow("r1", "r1.jpg", "toy", "ref", 0, "zero"),
                ManifestRow("r2", "r2.jpg", "toy", "ref", 1, "one"),
                ManifestRow("r3", "r3.jpg", "toy", "ref", 1, "one"),
            ]
            query_rows = [
                ManifestRow("q0", "q0.jpg", "toy", "query", 0, "zero"),
                ManifestRow("q1", "q1.jpg", "toy", "query", 1, "one"),
            ]
            reference = np.asarray(
                [
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.0, 1.0],
                    [0.2, 0.8],
                ],
                dtype="float32",
            )
            query = np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype="float32",
            )
            reference_cache = _write_cache(tmp, "ref", reference, reference_rows)
            query_cache = _write_cache(tmp, "query", query, query_rows)
            dense_csv = tmp / "dense.csv"
            sharded_csv = tmp / "sharded.csv"

            score_cache_knn(
                query_cache=query_cache,
                reference_cache=reference_cache,
                output_csv=dense_csv,
                k_values=(1, 2),
                batch_size=2,
                device="cpu",
                class_conditional=True,
            )
            score_cache_knn(
                query_cache=query_cache,
                reference_cache=reference_cache,
                output_csv=sharded_csv,
                k_values=(1, 2),
                batch_size=2,
                device="cpu",
                class_conditional=True,
                mmap=True,
                reference_shard_size=2,
            )

            dense_rows = _read_rows(dense_csv)
            sharded_rows = _read_rows(sharded_csv)
            self.assertEqual([row["nearest_sample_id"] for row in dense_rows], ["r0", "r2"])
            self.assertEqual(dense_rows, sharded_rows)

    def test_class_conditional_leave_one_out_removes_identical_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            rows = [
                ManifestRow("same", "same.jpg", "toy", "train", 0, "zero"),
                ManifestRow("near", "near.jpg", "toy", "train", 0, "zero"),
                ManifestRow("other", "other.jpg", "toy", "train", 1, "one"),
            ]
            embeddings = np.asarray(
                [
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.0, 1.0],
                ],
                dtype="float32",
            )
            cache = _write_cache(tmp, "train", embeddings, rows)
            output_csv = tmp / "loo.csv"

            score_cache_knn(
                query_cache=cache,
                reference_cache=cache,
                output_csv=output_csv,
                k_values=(1,),
                batch_size=3,
                device="cpu",
                leave_one_out=True,
                class_conditional=True,
                reference_shard_size=2,
            )

            output_rows = _read_rows(output_csv)
            self.assertEqual(output_rows[0]["sample_id"], "same")
            self.assertGreater(float(output_rows[0]["class_support_k1_kth_distance"]), 0.0)
            self.assertEqual(output_rows[2]["class_support_k1_kth_distance"], "")


if __name__ == "__main__":
    unittest.main()
