from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ada.actionability.deletion_controls import DeletionControlConfig, build_deletion_controls
from ada.actionability.restoration import RestorationConfig, build_restoration_manifests


class RestorationManifestTests(unittest.TestCase):
    def test_restoration_manifests_are_nested_and_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = _write_cache(root / "train_cache", labels=[0] * 24 + [1] * 4)
            enriched = root / "enriched"
            selected = root / "selected"
            enriched.mkdir()
            selected.mkdir()
            target_ids = [f"train-{idx}" for idx in range(12)]
            membership_rows = []
            for idx in range(24):
                membership_rows.append(
                    {
                        "region_id": "region-a" if idx < 12 else "region-b",
                        "sample_id": f"train-{idx}",
                        "class_id": 0,
                        "class_name": "n0000",
                        "k_regions": 10,
                    }
                )
            for idx in range(24, 28):
                membership_rows.append(
                    {
                        "region_id": "region-c",
                        "sample_id": f"train-{idx}",
                        "class_id": 1,
                        "class_name": "n0001",
                        "k_regions": 10,
                    }
                )
            _write_csv(enriched / "region_membership_k10.csv", membership_rows)
            _write_csv(enriched / "validation_assignments_k10.csv", [{"region_id": "region-a", "sample_id": "val-0"}])
            (enriched / "metadata.json").write_text(
                json.dumps(
                    {
                        "artifact_id": "enrich-1",
                        "source_region_artifact_id": "regions-1",
                        "source_validation_assignment_hash": "val-1",
                    }
                )
            )
            _write_csv(
                selected / "selected_pilot_regions.csv",
                [{"region_id": "region-a", "class_id": 0, "class_name": "n0000", "pilot_type": "sparse_interior"}],
            )
            (selected / "metadata.json").write_text(json.dumps({"artifact_id": "pilot-1"}))

            deletion_dir = root / "deletion"
            build_deletion_controls(
                DeletionControlConfig(
                    train_cache=train_cache,
                    enriched_regions_dir=enriched,
                    pilot_regions_dir=selected,
                    output_dir=deletion_dir,
                    retention_levels=(1.0, 0.25, 0.0),
                    seeds=(0,),
                )
            )

            restore_dir = root / "restore"
            metadata = build_restoration_manifests(
                RestorationConfig(
                    train_cache=train_cache,
                    enriched_regions_dir=enriched,
                    pilot_regions_dir=selected,
                    deletion_controls_dir=deletion_dir,
                    output_dir=restore_dir,
                    budget_fractions=(0.10, 0.25, 0.50, 1.00),
                    seeds=(0,),
                )
            )
            self.assertEqual(metadata["summary"]["manifest_count"], 30)
            index = _read_csv(restore_dir / "restoration_manifests.csv")
            self.assertTrue((restore_dir / "deletion_control_manifests.csv").exists())
            self.assertEqual(len(index), 30)

            q0_conditions = {row["restoration_condition"] for row in index if float(row["retention_level"]) == 0.0}
            self.assertNotIn("target_anchor_oversampling", q0_conditions)
            self.assertNotIn("target_anchor_augmented_oversampling", q0_conditions)
            self.assertNotIn("region_weighted_loss", q0_conditions)

            q25_target = [
                row
                for row in index
                if float(row["retention_level"]) == 0.25 and row["restoration_condition"] == "unique_target_region_restoration"
            ]
            q25_target = sorted(q25_target, key=lambda row: float(row["restoration_budget_fraction"]))
            restored_sets = []
            for row in q25_target:
                manifest = json.loads(Path(row["manifest_json"]).read_text())
                restored_sets.append(set(manifest["restored_target_sample_count"] and _read_lines(Path(row["manifest_json"]).parent / "restored_target_sample_ids.txt")))
            for smaller, larger in zip(restored_sets, restored_sets[1:]):
                self.assertTrue(smaller.issubset(larger))

            full_rows = [row for row in index if row["restoration_condition"] == "full_real_restoration"]
            self.assertEqual(len(full_rows), 2)
            for row in full_rows:
                manifest = json.loads(Path(row["manifest_json"]).read_text())
                self.assertTrue(manifest["full_restoration_matches_baseline_manifest"])

            weighted = [
                row
                for row in index
                if float(row["retention_level"]) == 0.25
                and row["restoration_condition"] == "region_weighted_loss"
                and float(row["restoration_budget_fraction"]) == 1.0
            ][0]
            weighted_rows = _read_csv(Path(weighted["exposure_manifest_csv"]))
            target_weights = [
                float(row["loss_weight"])
                for row in weighted_rows
                if row["sample_id"] in target_ids and int(row["multiplicity"]) > 0
            ]
            self.assertTrue(target_weights)
            self.assertGreater(min(target_weights), 1.0)
            self.assertEqual(len(weighted_rows), len({row["sample_id"] for row in weighted_rows}))


def _write_cache(path: Path, labels: list[int]) -> Path:
    path.mkdir(parents=True)
    np.save(path / "embeddings.npy", np.eye(len(labels), 4, dtype=np.float32))
    rows = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "sample_id": f"train-{idx}",
                "relative_path": f"n{label:04d}/{idx}.JPEG",
                "source_dataset": "toy",
                "split": "train",
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
    (path / "metadata.json").write_text(json.dumps({"cache_id": "cache-train", "cached_manifest_hash": "manifest-train"}))
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
