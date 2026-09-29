from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ada.actionability.deletion_controls import DeletionControlConfig, build_deletion_controls


class DeletionControlTests(unittest.TestCase):
    def test_count_preserving_replacement_uses_multiplicity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = _write_cache(root / "train_cache", labels=[0, 0, 0, 0, 1, 1])
            enriched = root / "enriched"
            selected = root / "selected"
            superseded = root / "old_deletion"
            enriched.mkdir()
            selected.mkdir()
            superseded.mkdir()
            _write_csv(
                enriched / "region_membership_k10.csv",
                [
                    {"region_id": "region-a", "sample_id": "train-0", "class_id": 0, "class_name": "n0000", "k_regions": 10},
                    {"region_id": "region-a", "sample_id": "train-1", "class_id": 0, "class_name": "n0000", "k_regions": 10},
                    {"region_id": "region-b", "sample_id": "train-2", "class_id": 0, "class_name": "n0000", "k_regions": 10},
                    {"region_id": "region-b", "sample_id": "train-3", "class_id": 0, "class_name": "n0000", "k_regions": 10},
                    {"region_id": "region-c", "sample_id": "train-4", "class_id": 1, "class_name": "n0001", "k_regions": 10},
                    {"region_id": "region-c", "sample_id": "train-5", "class_id": 1, "class_name": "n0001", "k_regions": 10},
                ],
            )
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
                [
                    {
                        "region_id": "region-a",
                        "class_id": 0,
                        "class_name": "n0000",
                        "pilot_type": "sparse_interior",
                    }
                ],
            )
            (selected / "metadata.json").write_text(json.dumps({"artifact_id": "pilot-1"}))

            out = root / "controls"
            metadata = build_deletion_controls(
                DeletionControlConfig(
                    train_cache=train_cache,
                    enriched_regions_dir=enriched,
                    pilot_regions_dir=selected,
                    output_dir=out,
                    superseded_deletion_dir=superseded,
                    retention_levels=(1.0, 0.0),
                    seeds=(0,),
                )
            )
            index = _read_csv(out / "deletion_control_manifests.csv")
            self.assertEqual(len(index), 5)
            self.assertEqual(metadata["summary"]["by_control_family"]["baseline"], 1)
            repl = [row for row in index if row["control_family"] == "same_class_count_preserving_replacement"][0]
            manifest = json.loads(Path(repl["manifest_json"]).read_text())
            self.assertEqual(manifest["totals"]["total_exposure_count"], 6)
            exposure = _read_csv(Path(repl["exposure_manifest_csv"]))
            self.assertEqual(len(exposure), len({row["sample_id"] for row in exposure}))
            self.assertEqual(sum(int(row["multiplicity"]) for row in exposure), 6)
            self.assertTrue((superseded / "SUPERSEDED_FOR_PRIMARY_CAUSAL_ANALYSIS.md").exists())


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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


if __name__ == "__main__":
    unittest.main()
