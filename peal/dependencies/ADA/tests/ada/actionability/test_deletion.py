from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ada.actionability.deletion import DeletionBuildConfig, build_deletion_artifact
from ada.actionability.region_filters import EligibilityThresholds


class DeletionManifestTests(unittest.TestCase):
    def test_deletion_counts_and_no_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = _write_train_cache(root / "train_cache")
            regions_dir = _write_regions(root / "regions")
            out = root / "deletion"
            metadata = build_deletion_artifact(
                DeletionBuildConfig(
                    train_cache=train_cache,
                    regions_dir=regions_dir,
                    output_dir=out,
                    retention_levels=(1.0, 0.5, 0.0),
                    seed=11,
                    max_regions=1,
                    thresholds=EligibilityThresholds(
                        min_train_count=1,
                        min_val_count=0,
                        same_class_purity=0.0,
                        min_positive_class_margin=-10.0,
                        max_duplicate_fraction=1.0,
                    ),
                )
            )
            self.assertEqual(metadata["manifest_count"], 3)
            rows = _read_csv(out / "deletion_manifests.csv")
            by_level = {row["retention_label"]: row for row in rows}
            self.assertEqual(int(by_level["retain_100"]["deleted_region_count"]), 0)
            self.assertEqual(int(by_level["retain_050"]["retained_region_count"]), 2)
            self.assertEqual(int(by_level["retain_050"]["deleted_region_count"]), 2)
            self.assertEqual(int(by_level["retain_000"]["deleted_region_count"]), 4)

            manifest = json.loads(Path(by_level["retain_050"]["manifest_json"]).read_text())
            retained = _read_lines(Path(manifest["retained_region_sample_ids_path"]))
            deleted = _read_lines(Path(manifest["deleted_region_sample_ids_path"]))
            self.assertFalse(set(retained).intersection(deleted))
            self.assertEqual(len(retained), len(set(retained)))
            self.assertEqual(len(deleted), len(set(deleted)))

    def test_duplicate_train_sample_ids_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = _write_train_cache(root / "train_cache")
            rows = _read_csv(train_cache / "manifest.csv")
            rows[1]["sample_id"] = rows[0]["sample_id"]
            _write_csv(train_cache / "manifest.csv", rows)
            regions_dir = _write_regions(root / "regions")
            with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
                build_deletion_artifact(
                    DeletionBuildConfig(
                        train_cache=train_cache,
                        regions_dir=regions_dir,
                        output_dir=root / "deletion",
                        thresholds=EligibilityThresholds(
                            min_train_count=1,
                            min_val_count=0,
                            same_class_purity=0.0,
                            min_positive_class_margin=-10.0,
                            max_duplicate_fraction=1.0,
                        ),
                    )
                )


def _write_train_cache(path: Path) -> Path:
    path.mkdir(parents=True)
    rows = []
    for idx in range(6):
        label = 0 if idx < 4 else 1
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
    (path / "metadata.json").write_text(json.dumps({"cache_id": "cache-train"}))
    return path


def _write_regions(path: Path) -> Path:
    path.mkdir(parents=True)
    regions = [
        {
            "region_id": "region-a",
            "k_regions": 1,
            "class_id": 0,
            "class_name": "n0000",
            "cluster_index": 0,
            "prototype_index": 0,
            "train_count": 4,
            "validation_count": 2,
            "same_class_purity": 1.0,
            "class_margin": 1.0,
            "duplicate_fraction": 0.0,
            "eligible_primary": 1,
        }
    ]
    membership = [
        {"region_id": "region-a", "sample_id": f"train-{idx}", "relative_path": f"n0000/{idx}.JPEG", "class_id": 0, "class_name": "n0000", "k_regions": 1, "cluster_index": 0}
        for idx in range(4)
    ]
    _write_csv(path / "regions.csv", regions)
    _write_csv(path / "region_membership.csv", membership)
    (path / "metadata.json").write_text(json.dumps({"artifact_id": "regions-a", "region_hash": "regions-hash"}))
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
