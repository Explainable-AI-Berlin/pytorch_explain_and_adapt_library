from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ada.interpretability.contrastive_concepts import concept_scores
from ada.interpretability.phrase_bank import PhraseBankConfig, build_phrase_bank
from ada.interpretability.region_cards import RegionLanguageCardConfig, build_region_language_cards
from ada.interpretability.region_manifest import RegionInterpretabilityManifestConfig, build_region_interpretability_manifest
from ada.interpretability.sparse_concepts import sparse_positive_decomposition
from ada.interpretability.text_ambiguity import class_prompt_matrix, text_class_metrics
from ada.interpretability.vlm_shared_cache import _load_cached_image_features


class RegionLanguageCardTests(unittest.TestCase):
    def test_contrastive_concept_score_finds_region_phrase(self) -> None:
        phrases = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        region = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
        control = np.asarray([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32)

        scores = concept_scores(region, phrases, control_embeddings=control)

        self.assertGreater(scores["region_contrast_score"][0], scores["region_contrast_score"][1])

    def test_text_ambiguity_outputs_margin_and_entropy(self) -> None:
        phrase_rows = [
            {"phrase_type": "class_prompt", "class_id": "0", "display_name": "class zero"},
            {"phrase_type": "class_prompt", "class_id": "1", "display_name": "class one"},
        ]
        phrase_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        class_embeddings, class_ids, names = class_prompt_matrix(phrase_rows, phrase_embeddings)

        metrics = text_class_metrics(np.asarray([[1.0, 0.0], [0.1, 0.9]], dtype=np.float32), [0, 1], class_embeddings, class_ids)

        self.assertEqual(class_ids, [0, 1])
        self.assertEqual(names, ["class zero", "class one"])
        self.assertTrue(np.all(metrics["text_class_margin"] > 0.0))
        self.assertEqual(metrics["text_competing_class_id"], [1, 0])

    def test_sparse_positive_decomposition_is_nonnegative(self) -> None:
        image = np.asarray([[1.0, 0.2]], dtype=np.float32)
        concepts = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        coeffs = sparse_positive_decomposition(image, concepts, max_terms=2)

        self.assertEqual(coeffs.shape, (1, 2))
        self.assertTrue(np.all(coeffs >= 0.0))
        self.assertGreater(coeffs[0, 0], coeffs[0, 1])

    def test_load_cached_image_features_preserves_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = _write_cache(root / "cache", [0, 1, 0])
            features = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
            np.save(cache / "embeddings.npy", features)

            rows = _read_manifest_rows(cache / "manifest.csv")
            selected = [rows[2], rows[0]]
            loaded = _load_cached_image_features(cache, selected)

            self.assertTrue(np.allclose(loaded, np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype=np.float32)))

    def test_build_phrase_bank_and_region_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = _write_cache(root / "train_cache", [0, 0, 0, 1])
            val_cache = _write_cache(root / "val_cache", [0, 0, 1], prefix="val")

            phrase_meta = build_phrase_bank(
                PhraseBankConfig(
                    train_cache=train_cache,
                    output_dir=root / "phrase_bank",
                    imagenet_meta=None,
                    include_general_phrases=True,
                    include_class_prompts=True,
                )
            )
            self.assertGreater(phrase_meta["summary"]["phrases"], 0)

            shared = root / "shared"
            shared.mkdir()
            _copy(train_cache / "manifest.csv", shared / "train_manifest.csv")
            _copy(val_cache / "manifest.csv", shared / "val_manifest.csv")
            _copy(root / "phrase_bank" / "phrase_bank.csv", shared / "phrase_bank.csv")
            phrases = _read_csv(shared / "phrase_bank.csv")
            phrase_embeddings = np.zeros((len(phrases), 3), dtype=np.float32)
            for idx, row in enumerate(phrases):
                if row["phrase"] == "a side view":
                    phrase_embeddings[idx, 0] = 1.0
                elif row["phrase_type"] == "class_prompt" and row["class_id"] == "0":
                    phrase_embeddings[idx, 1] = 1.0
                elif row["phrase_type"] == "class_prompt" and row["class_id"] == "1":
                    phrase_embeddings[idx, 2] = 1.0
                else:
                    phrase_embeddings[idx, 0] = 0.2
            np.save(shared / "phrase_embeddings.npy", _l2(phrase_embeddings))
            np.save(shared / "train_image_embeddings.npy", _l2(np.asarray([[1.0, 1.0, 0.0], [1.0, 0.9, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)))
            np.save(shared / "val_image_embeddings.npy", _l2(np.asarray([[1.0, 1.0, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)))

            enriched = root / "enriched"
            selected = root / "selected"
            enriched.mkdir()
            selected.mkdir()
            _write_csv(
                enriched / "region_membership_k10.csv",
                [
                    {"region_id": "region-a", "sample_id": "train-0", "class_id": 0, "class_name": "class0", "k_regions": 10, "cluster_index": 0},
                    {"region_id": "region-a", "sample_id": "train-1", "class_id": 0, "class_name": "class0", "k_regions": 10, "cluster_index": 0},
                    {"region_id": "region-b", "sample_id": "train-2", "class_id": 0, "class_name": "class0", "k_regions": 10, "cluster_index": 1},
                    {"region_id": "region-c", "sample_id": "train-3", "class_id": 1, "class_name": "class1", "k_regions": 10, "cluster_index": 0},
                ],
            )
            _write_csv(
                enriched / "validation_assignments_k10.csv",
                [
                    {"region_id": "region-a", "sample_id": "val-0"},
                    {"region_id": "region-a", "sample_id": "val-1"},
                ],
            )
            _write_csv(
                selected / "selected_pilot_regions.csv",
                [
                    {
                        "region_id": "region-a",
                        "class_id": 0,
                        "class_name": "class0",
                        "pilot_type": "sparse_interior",
                        "train_count": 2,
                        "validation_count": 2,
                    }
                ],
            )
            deletion_controls = root / "deletion_controls"
            manifest_dir = deletion_controls / "region-a" / "regional_drop" / "retain_025" / "seed_000"
            manifest_dir.mkdir(parents=True)
            _write_csv(
                manifest_dir / "exposure_manifest.csv",
                [
                    {"sample_id": "train-0", "relative_path": "class0/0.JPEG", "class_id": 0, "class_name": "class0", "region_id": "region-a", "multiplicity": 1, "source_condition": "regional_drop", "seed": 0},
                    {"sample_id": "train-1", "relative_path": "class0/1.JPEG", "class_id": 0, "class_name": "class0", "region_id": "region-a", "multiplicity": 0, "source_condition": "regional_drop", "seed": 0},
                    {"sample_id": "train-2", "relative_path": "class0/2.JPEG", "class_id": 0, "class_name": "class0", "region_id": "region-b", "multiplicity": 1, "source_condition": "regional_drop", "seed": 0},
                    {"sample_id": "train-3", "relative_path": "class1/3.JPEG", "class_id": 1, "class_name": "class1", "region_id": "region-c", "multiplicity": 1, "source_condition": "regional_drop", "seed": 0},
                ],
            )
            _write_csv(
                deletion_controls / "deletion_control_manifests.csv",
                [
                    {
                        "manifest_id": "manifest-q025",
                        "region_id": "region-a",
                        "class_id": 0,
                        "class_name": "class0",
                        "pilot_type": "sparse_interior",
                        "control_family": "regional_drop",
                        "retention_level": 0.25,
                        "retention_label": "retain_025",
                        "seed": 0,
                        "deleted_sample_count": 1,
                        "replacement_exposure_count": 0,
                        "unique_active_count": 3,
                        "total_exposure_count": 3,
                        "zero_multiplicity_count": 1,
                        "manifest_json": str(manifest_dir / "manifest.json"),
                        "exposure_manifest_csv": str(manifest_dir / "exposure_manifest.csv"),
                    }
                ],
            )
            manifest_meta = build_region_interpretability_manifest(
                RegionInterpretabilityManifestConfig(
                    train_cache=train_cache,
                    validation_cache=val_cache,
                    enriched_regions_dir=enriched,
                    pilot_regions_dir=selected,
                    deletion_controls_dir=deletion_controls,
                    output_dir=root / "interp_manifest",
                    control_count=2,
                    seed=0,
                )
            )
            self.assertGreater(manifest_meta["summary"]["groups"]["q025_deleted_real"], 0)

            meta = build_region_language_cards(
                RegionLanguageCardConfig(
                    vlm_shared_dir=shared,
                    enriched_regions_dir=enriched,
                    pilot_regions_dir=selected,
                    output_dir=root / "cards",
                    interpretability_manifest_dir=root / "interp_manifest",
                    bootstrap_samples=5,
                    permutation_samples=5,
                    min_supporting_images=1,
                    top_k_phrases=5,
                    sparse_enabled=True,
                )
            )

            rows = _read_csv(root / "cards" / "region_language_cards.csv")
            concept_rows = _read_csv(root / "cards" / "region_concept_scores.csv")
            deleted_rows = _read_csv(root / "cards" / "region_deleted_vs_retained_concepts.csv")
            self.assertEqual(meta["summary"]["regions"], 1)
            side_view = [row for row in concept_rows if row["phrase"] == "a side view"][0]
            self.assertGreater(float(side_view["region_contrast_score"]), 0.0)
            self.assertGreater(len(deleted_rows), 0)
            self.assertGreater(float(rows[0]["mean_text_class_margin"]), 0.0)


def _write_cache(path: Path, labels: list[int], *, prefix: str = "train") -> Path:
    path.mkdir(parents=True)
    rows = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "sample_id": f"{prefix}-{idx}",
                "relative_path": f"class{label}/{idx}.JPEG",
                "source_dataset": "toy",
                "split": prefix,
                "class_id": label,
                "class_name": f"class{label}",
                "patient_id": "",
                "slide_id": "",
                "site_id": "",
                "hidden_group_id": "",
            }
        )
    _write_csv(path / "manifest.csv", rows)
    (path / "metadata.json").write_text(json.dumps({"root": str(path / "images")}))
    return path


def _copy(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text())


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1.0e-12)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_manifest_rows(path: Path):
    from ada.atlas.data.manifests import load_manifest_csv

    return load_manifest_csv(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
