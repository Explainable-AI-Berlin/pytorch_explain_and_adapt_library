from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ada.atlas.cli.join_vae_support_metrics import join_vae_support_metrics
from ada.atlas.cache.vae_latents import (
    fit_vae_pca_projection,
    fit_vae_standardization,
    plan_vae_latent_cache,
    project_vae_latent_cache,
    standardize_vae_latent_cache,
)
from ada.atlas.data.manifests import ImageFolderManifest, ManifestRow, write_manifest_files


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
    metadata = json.loads((cache / "metadata.json").read_text())
    metadata.update(
        {
            "cache_id": f"cache-{name}",
            "cache_format": "ada_vae_posterior_cache_v1",
            "coordinate": "scaled_posterior_mean_flat",
            "cached_manifest_hash": manifest.manifest_hash,
        }
    )
    (cache / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return cache


class VAELatentCacheTests(unittest.TestCase):
    def test_plan_uses_deterministic_posterior_mean_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            (tmp / "n0001").mkdir()
            (tmp / "n0001" / "a.jpg").write_bytes(b"not-opened")
            manifest, plan = plan_vae_latent_cache(
                dataset="toy",
                split="train",
                root=tmp,
                vae_type="sdvae-ema",
                output_root=tmp / "out",
                batch_size=2,
                num_workers=0,
                image_size=256,
                precision="bf16",
            )

        self.assertEqual(manifest.sample_count, 1)
        self.assertEqual(plan["coordinate"], "scaled_posterior_mean_flat")
        self.assertEqual(plan["posterior_logvar_coordinate"], "unscaled_vae_posterior_logvar_flat")
        self.assertTrue(plan["save_raw_mean"])
        self.assertFalse(plan["allow_download"])
        self.assertTrue(plan["local_files_only"])

    def test_train_projection_is_fit_once_and_applied_to_val(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            train_rows = [
                ManifestRow("t0", "t0.jpg", "toy", "train", 0, "zero"),
                ManifestRow("t1", "t1.jpg", "toy", "train", 0, "zero"),
                ManifestRow("t2", "t2.jpg", "toy", "train", 1, "one"),
                ManifestRow("t3", "t3.jpg", "toy", "train", 1, "one"),
            ]
            val_rows = [
                ManifestRow("v0", "v0.jpg", "toy", "val", 0, "zero"),
                ManifestRow("v1", "v1.jpg", "toy", "val", 1, "one"),
            ]
            train = np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 2.0],
                    [1.0, 1.0, 2.0],
                ],
                dtype="float32",
            )
            val = np.asarray(
                [
                    [0.5, 0.0, 1.0],
                    [0.5, 1.0, 2.0],
                ],
                dtype="float32",
            )
            train_cache = _write_cache(tmp, "train", train, train_rows)
            val_cache = _write_cache(tmp, "val", val, val_rows)
            projection_dir = tmp / "projection"
            train_out = tmp / "train_projected"
            val_out = tmp / "val_projected"

            projection_meta = fit_vae_pca_projection(
                train_cache=train_cache,
                output_dir=projection_dir,
                pca_dim=2,
                batch_size=2,
            )
            train_meta = project_vae_latent_cache(
                input_cache=train_cache,
                projection_dir=projection_dir,
                output_dir=train_out,
                batch_size=2,
            )
            val_meta = project_vae_latent_cache(
                input_cache=val_cache,
                projection_dir=projection_dir,
                output_dir=val_out,
                batch_size=2,
            )

            train_projected = np.load(train_out / "embeddings.npy")
            val_projected = np.load(val_out / "embeddings.npy")
            self.assertEqual(projection_meta["input_embedding_shape"], [4, 3])
            self.assertEqual(train_meta["embedding_shape"], [4, 2])
            self.assertEqual(val_meta["embedding_shape"], [2, 2])
            self.assertEqual(train_projected.shape, (4, 2))
            self.assertEqual(val_projected.shape, (2, 2))
            self.assertTrue((train_out / "manifest.csv").exists())
            self.assertTrue((val_out / "manifest.csv").exists())

    def test_join_vae_support_metrics_requires_complete_metrics_when_strict(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            support = tmp / "support.csv"
            metrics = tmp / "metrics.csv"
            output = tmp / "joined.csv"
            support.write_text(
                "sample_id,class_id,class_name,support_k5_kth_distance\n"
                "s0,0,zero,1.25\n"
            )
            metrics.write_text(
                "sample_id,class_id,class_name,posterior_kl_raw,posterior_kl,latent_norm_raw,"
                "latent_norm_scaled_mean,latent_norm_unscaled_mean,latent_mean_scalar,latent_std_scalar,reconstruction_mse\n"
                "s0,0,zero,3.5,3.5,7.0,1.2,7.0,0.0,0.4,\n"
            )
            join_vae_support_metrics(support_csv=support, metrics_csv=metrics, output_csv=output, strict=True)
            joined = output.read_text()

        self.assertIn("posterior_kl_raw", joined)
        self.assertIn("3.5", joined)

    def test_standardized_flat_cache_keeps_full_dimensionality(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            train_rows = [
                ManifestRow("t0", "t0.jpg", "toy", "train", 0, "zero"),
                ManifestRow("t1", "t1.jpg", "toy", "train", 0, "zero"),
                ManifestRow("t2", "t2.jpg", "toy", "train", 1, "one"),
                ManifestRow("t3", "t3.jpg", "toy", "train", 1, "one"),
            ]
            val_rows = [
                ManifestRow("v0", "v0.jpg", "toy", "val", 0, "zero"),
                ManifestRow("v1", "v1.jpg", "toy", "val", 1, "one"),
            ]
            train = np.asarray(
                [
                    [0.0, 1.0, 2.0],
                    [2.0, 3.0, 4.0],
                    [4.0, 5.0, 6.0],
                    [6.0, 7.0, 8.0],
                ],
                dtype="float32",
            )
            val = np.asarray([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype="float32")
            train_cache = _write_cache(tmp, "train", train, train_rows)
            val_cache = _write_cache(tmp, "val", val, val_rows)
            standardization_dir = tmp / "standardization"
            train_out = tmp / "train_standardized"
            val_out = tmp / "val_standardized"

            std_meta = fit_vae_standardization(train_cache=train_cache, output_dir=standardization_dir, batch_size=2)
            train_meta = standardize_vae_latent_cache(
                input_cache=train_cache,
                standardization_dir=standardization_dir,
                output_dir=train_out,
                batch_size=2,
            )
            val_meta = standardize_vae_latent_cache(
                input_cache=val_cache,
                standardization_dir=standardization_dir,
                output_dir=val_out,
                batch_size=2,
            )

            train_std = np.load(train_out / "embeddings.npy")
            val_std = np.load(val_out / "embeddings.npy")

        self.assertEqual(std_meta["input_embedding_shape"], [4, 3])
        self.assertEqual(train_meta["embedding_shape"], [4, 3])
        self.assertEqual(val_meta["embedding_shape"], [2, 3])
        self.assertEqual(train_std.shape, (4, 3))
        self.assertEqual(val_std.shape, (2, 3))
        self.assertEqual(train_meta["coordinate"], "train_standardized_flat_vae_posterior_mean")
        self.assertTrue(train_meta["scale_by_sqrt_dim"])


if __name__ == "__main__":
    unittest.main()
