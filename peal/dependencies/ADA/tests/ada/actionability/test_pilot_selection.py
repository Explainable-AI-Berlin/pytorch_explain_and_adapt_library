from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ada.actionability.select_pilot_regions import PilotSelectionConfig, select_pilot_regions


class PilotSelectionTests(unittest.TestCase):
    def test_selects_requested_categories_with_distinct_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enriched = root / "enriched"
            enriched.mkdir()
            rows = []
            for i in range(4):
                rows.append(_row(i, "sparse", support=0.9, purity=1.0, entropy=0.0, margin=0.4))
            for i in range(4, 6):
                rows.append(_row(i, "dense", support=0.1, purity=1.0, entropy=0.0, margin=0.4))
            for i in range(6, 8):
                rows.append(_row(i, "boundary", support=0.9, purity=0.6, entropy=0.6, margin=-0.1))
            _write_csv(enriched / "regions_enriched.csv", rows)
            (enriched / "metadata.json").write_text(json.dumps({"artifact_id": "enrich-1"}))

            out = root / "selected"
            metadata = select_pilot_regions(
                PilotSelectionConfig(
                    enriched_regions_dir=enriched,
                    output_dir=out,
                    sparse_interior_count=4,
                    dense_interior_count=2,
                    sparse_boundary_count=2,
                    min_train_count=100,
                    min_validation_count=15,
                )
            )
            selected = _read_csv(out / "selected_pilot_regions.csv")
            self.assertEqual(len(selected), 8)
            self.assertEqual(len({row["class_id"] for row in selected}), 8)
            self.assertEqual(metadata["summary"]["by_type"]["sparse_interior"], 4)
            self.assertEqual(metadata["summary"]["by_type"]["dense_interior"], 2)
            self.assertEqual(metadata["summary"]["by_type"]["sparse_boundary"], 2)


def _row(class_id: int, prefix: str, *, support: float, purity: float, entropy: float, margin: float) -> dict[str, object]:
    return {
        "region_id": f"region-{prefix}-{class_id}",
        "k_regions": 10,
        "class_id": class_id,
        "class_name": f"n{class_id:04d}",
        "cluster_index": 0,
        "prototype_index": class_id,
        "train_count": 120,
        "validation_count": 16,
        "member_support_pct_median": support,
        "member_support_pct_q75": support,
        "member_support_pct_q90": support,
        "prototype_support_pct": support,
        "global_neighbor_purity": purity,
        "local_label_entropy": entropy,
        "robust_class_margin": margin,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


if __name__ == "__main__":
    unittest.main()
