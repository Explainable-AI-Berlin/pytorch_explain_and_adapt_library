from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ada.actionability.region_filters import EligibilityThresholds
from ada.actionability.regions import RegionBuildConfig, build_region_artifact


class RegionBuildTests(unittest.TestCase):
    def test_train_only_regions_and_validation_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = _write_cache(
                root / "train_cache",
                "train",
                np.asarray(
                    [
                        [1.0, 0.0],
                        [0.9, 0.1],
                        [0.0, 1.0],
                        [0.1, 0.9],
                        [-1.0, 0.0],
                        [-0.9, -0.1],
                        [0.0, -1.0],
                        [-0.1, -0.9],
                    ],
                    dtype=np.float32,
                ),
                [0, 0, 0, 0, 1, 1, 1, 1],
            )
            val_cache = _write_cache(
                root / "val_cache",
                "val",
                np.asarray([[1.0, 0.05], [-1.0, -0.05]], dtype=np.float32),
                [0, 1],
            )
            out = root / "regions"
            metadata = build_region_artifact(
                RegionBuildConfig(
                    train_cache=train_cache,
                    validation_cache=val_cache,
                    output_dir=out,
                    k_values=(1, 2),
                    seed=7,
                    residual_mode="raw",
                    local_purity_k=2,
                    duplicate_distance_threshold=-1.0,
                    thresholds=EligibilityThresholds(
                        min_train_count=1,
                        min_val_count=0,
                        same_class_purity=0.5,
                        min_positive_class_margin=-10.0,
                        max_duplicate_fraction=1.0,
                    ),
                )
            )

            regions = _read_csv(out / "regions.csv")
            memberships = _read_csv(out / "region_membership.csv")
            val_assignments = _read_csv(out / "validation_assignments.csv")

            self.assertEqual(metadata["summary"]["region_count"], 6)
            self.assertEqual(len(memberships), 16)
            self.assertEqual(len(val_assignments), 4)
            self.assertEqual(
                {(row["sample_id"], row["k_regions"]) for row in val_assignments},
                {("val-0", "1"), ("val-0", "2"), ("val-1", "1"), ("val-1", "2")},
            )
            self.assertEqual({row["split"] for row in _read_csv(train_cache / "manifest.csv")}, {"train"})
            self.assertTrue(all(row["region_id"] for row in regions))
            self.assertEqual(len({row["sample_id"] for row in memberships}), 8)
            val_counts = {}
            for row in val_assignments:
                val_counts[row["region_id"]] = val_counts.get(row["region_id"], 0) + 1
            for row in regions:
                self.assertEqual(int(row["validation_count"]), val_counts.get(row["region_id"], 0))

    def test_region_ids_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = _write_cache(
                root / "train_cache",
                "train",
                np.asarray([[1, 0], [0.8, 0.2], [0, 1], [-1, 0], [-0.8, -0.2], [0, -1]], dtype=np.float32),
                [0, 0, 0, 1, 1, 1],
            )
            kwargs = dict(
                train_cache=train_cache,
                k_values=(1,),
                seed=3,
                residual_mode="raw",
                local_purity_k=1,
                thresholds=EligibilityThresholds(min_train_count=1, min_val_count=0, same_class_purity=0.0, min_positive_class_margin=-10.0),
            )
            build_region_artifact(RegionBuildConfig(output_dir=root / "a", **kwargs))
            build_region_artifact(RegionBuildConfig(output_dir=root / "b", **kwargs))
            ids_a = [row["region_id"] for row in _read_csv(root / "a" / "regions.csv")]
            ids_b = [row["region_id"] for row in _read_csv(root / "b" / "regions.csv")]
            self.assertEqual(ids_a, ids_b)

    def test_validation_sample_id_overlap_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = _write_cache(
                root / "train_cache",
                "shared",
                np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32),
                [0, 0, 1, 1],
            )
            val_cache = _write_cache(
                root / "val_cache",
                "shared",
                np.asarray([[1, 0], [-1, 0]], dtype=np.float32),
                [0, 1],
            )
            with self.assertRaisesRegex(ValueError, "train/validation sample_id overlap"):
                build_region_artifact(
                    RegionBuildConfig(
                        train_cache=train_cache,
                        validation_cache=val_cache,
                        output_dir=root / "regions",
                        k_values=(1,),
                    )
                )


def _write_cache(path: Path, split: str, embeddings: np.ndarray, labels: list[int]) -> Path:
    path.mkdir(parents=True)
    np.save(path / "embeddings.npy", embeddings.astype("float32"))
    rows = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "sample_id": f"{split}-{idx}",
                "relative_path": f"n{label:04d}/{idx}.JPEG",
                "source_dataset": "toy",
                "split": split,
                "class_id": label,
                "class_name": f"n{label:04d}",
                "patient_id": "",
                "slide_id": "",
                "site_id": "",
                "hidden_group_id": "",
            }
        )
    with (path / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "cache_id": f"cache-{split}",
                "cached_manifest_hash": f"manifest-{split}",
                "dataset": "toy",
                "split": split,
                "embedding_shape": list(embeddings.shape),
                "embedding_dtype": "float32",
            }
        )
    )
    (path / "COMPLETED").write_text("ok\n")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


if __name__ == "__main__":
    unittest.main()
