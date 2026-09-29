from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ada.actionability.enrich_regions import RegionEnrichmentConfig, enrich_regions
from ada.actionability.region_filters import EligibilityThresholds
from ada.actionability.regions import RegionBuildConfig, build_region_artifact


class RegionEnrichmentTests(unittest.TestCase):
    def test_enrichment_is_k10_derived_and_parent_mapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = _write_cache(
                root / "train_cache",
                "train",
                np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.95, 0.05, 0.0],
                        [0.9, 0.1, 0.0],
                        [0.85, 0.15, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.95, 0.05],
                        [0.0, 0.9, 0.1],
                        [0.0, 0.85, 0.15],
                    ],
                    dtype=np.float32,
                ),
                [0, 0, 0, 0, 1, 1, 1, 1],
            )
            val_cache = _write_cache(
                root / "val_cache",
                "val",
                np.asarray([[1.0, 0.02, 0.0], [0.0, 1.0, 0.02]], dtype=np.float32),
                [0, 1],
            )
            regions_dir = root / "regions"
            build_region_artifact(
                RegionBuildConfig(
                    train_cache=train_cache,
                    validation_cache=val_cache,
                    output_dir=regions_dir,
                    k_values=(1, 2),
                    seed=0,
                    residual_mode="raw",
                    local_purity_k=2,
                    thresholds=EligibilityThresholds(
                        min_train_count=1,
                        min_val_count=0,
                        same_class_purity=0.0,
                        min_positive_class_margin=-10.0,
                    ),
                )
            )

            out = root / "enriched"
            metadata = enrich_regions(
                RegionEnrichmentConfig(
                    train_cache=train_cache,
                    regions_dir=regions_dir,
                    output_dir=out,
                    primary_k=2,
                    parent_k=1,
                    support_k=1,
                    global_neighbor_k=2,
                    margin_core_k=1,
                )
            )
            rows = _read_csv(out / "regions_enriched.csv")
            membership = _read_csv(out / "region_membership_k10.csv")
            val = _read_csv(out / "validation_assignments_k10.csv")

            self.assertTrue(str(metadata["source_region_artifact_id"]).startswith("regions-artifact-"))
            self.assertEqual({row["k_regions"] for row in rows}, {"2"})
            self.assertEqual(len(membership), 8)
            self.assertEqual(len(val), 2)
            self.assertTrue(all(row["parent_k5_region_id"] for row in rows))
            for row in rows:
                self.assertIn("member_support_pct_median", row)
                self.assertIn("global_neighbor_purity", row)
                self.assertIn("local_label_entropy", row)
                self.assertIn("robust_class_margin", row)


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
