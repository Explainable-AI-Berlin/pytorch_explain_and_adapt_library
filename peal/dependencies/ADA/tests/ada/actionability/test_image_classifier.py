from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ada.actionability.train_image_classifier import _resolve_image_root, write_image_probe_index


class ImageClassifierProbeTests(unittest.TestCase):
    def test_resolve_image_root_prefers_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            (cache / "metadata.json").write_text(json.dumps({"root": "/wrong"}))

            resolved = _resolve_image_root(root / "images", cache)

            self.assertEqual(resolved, root / "images")

    def test_resolve_image_root_falls_back_to_cache_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            (cache / "metadata.json").write_text(json.dumps({"root": str(root / "images")}))

            resolved = _resolve_image_root(None, cache)

            self.assertEqual(resolved, root / "images")

    def test_write_image_probe_index_matches_evaluator_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run = output / "manifest_0000_manifest-a"
            run.mkdir()
            (run / "metadata.json").write_text(
                json.dumps(
                    {
                        "artifact_id": "image-probe-1",
                        "manifest_index": 0,
                        "experiment_id": "e8-test",
                        "model_id": "resnet18_scratch_imageclf",
                        "architecture": "torchvision_resnet18",
                        "query_accuracy": 0.75,
                        "predictions_csv": str(run / "predictions.csv"),
                        "manifest_row": {
                            "manifest_id": "manifest-a",
                            "region_id": "region-a",
                            "class_id": 3,
                            "pilot_type": "sparse_interior",
                            "control_family": "regional_drop",
                            "retention_level": 0.25,
                            "seed": 1,
                        },
                    }
                )
            )

            index = write_image_probe_index(output)

            rows = _read_csv(index)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["manifest_id"], "manifest-a")
            self.assertEqual(rows[0]["region_id"], "region-a")
            self.assertEqual(rows[0]["model_id"], "resnet18_scratch_imageclf")
            self.assertEqual(rows[0]["feature_id"], "torchvision_resnet18")
            self.assertEqual(rows[0]["predictions_csv"], str(run / "predictions.csv"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


if __name__ == "__main__":
    unittest.main()
