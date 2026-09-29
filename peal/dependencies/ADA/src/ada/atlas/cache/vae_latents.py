from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ada.atlas.cache.dinov2_cls import safe_path_name, select_manifest_rows
from ada.atlas.data.manifests import ImageFolderManifest, build_imagefolder_manifest, manifest_summary
from ada.atlas.hashing import file_sha1, hash_rows, stable_hash


VAE_PRESETS: dict[str, dict[str, object]] = {
    "sdvae-ema": {
        "pretrained_path": "stabilityai/sd-vae-ft-ema",
        "subfolder": "",
        "latent_channels": 4,
        "downsample_factor": 8,
        "scaling_factor": 0.18215,
        "shift_factor": 0.0,
    },
    "sdvae-mse": {
        "pretrained_path": "stabilityai/sd-vae-ft-mse",
        "subfolder": "",
        "latent_channels": 4,
        "downsample_factor": 8,
        "scaling_factor": 0.18215,
        "shift_factor": 0.0,
    },
    "sdxl-vae": {
        "pretrained_path": "madebyollin/sdxl-vae-fp16-fix",
        "subfolder": "",
        "latent_channels": 4,
        "downsample_factor": 8,
        "scaling_factor": 0.13025,
        "shift_factor": 0.0,
    },
}


def plan_vae_latent_cache(
    *,
    dataset: str,
    split: str,
    root: str | Path,
    vae_type: str,
    output_root: str | Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    precision: str,
    storage_dtype: str = "float16",
    save_logvar: bool = True,
    save_raw_mean: bool = True,
    save_reconstruction_mse: bool = False,
    allow_download: bool = False,
    max_samples: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> tuple[ImageFolderManifest, dict[str, Any]]:
    if vae_type not in VAE_PRESETS:
        raise ValueError(f"unsupported vae_type={vae_type!r}; expected one of {sorted(VAE_PRESETS)}")
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be one of bf16, fp16, fp32")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")

    manifest = build_imagefolder_manifest(root, dataset=dataset, split=split)
    selected_manifest = select_manifest_rows(
        manifest,
        max_samples=max_samples,
        start_index=start_index,
        end_index=end_index,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    preset = dict(VAE_PRESETS[vae_type])
    cache_id = stable_hash(
        {
            "dataset": dataset,
            "split": split,
            "source_manifest_hash": manifest.manifest_hash,
            "selected_manifest_hash": selected_manifest.manifest_hash,
            "vae_type": vae_type,
            "pretrained_path": preset["pretrained_path"],
            "subfolder": preset["subfolder"],
            "image_size": int(image_size),
            "precision": precision,
            "coordinate": "scaled_posterior_mean_flat",
            "storage_dtype": storage_dtype,
            "save_logvar": bool(save_logvar),
            "save_raw_mean": bool(save_raw_mean),
            "save_reconstruction_mse": bool(save_reconstruction_mse),
            "max_samples": max_samples,
            "start_index": int(start_index),
            "end_index": end_index,
            "shard_index": shard_index,
            "num_shards": num_shards,
        },
        prefix="vae-cache",
    )
    output_dir = Path(output_root) / dataset / split / safe_path_name(vae_type) / "posterior_mean_flat" / cache_id
    latent_side = int(image_size) // int(preset["downsample_factor"])
    latent_dim = int(preset["latent_channels"]) * latent_side * latent_side
    plan = {
        "cache_id": cache_id,
        "cache_format": "ada_vae_posterior_cache_v1",
        "dataset": dataset,
        "split": split,
        "root": str(Path(root).expanduser().resolve()),
        "vae_type": vae_type,
        "pretrained_path": preset["pretrained_path"],
        "subfolder": preset["subfolder"],
        "latent_channels": int(preset["latent_channels"]),
        "downsample_factor": int(preset["downsample_factor"]),
        "scaling_factor": preset.get("scaling_factor"),
        "shift_factor": preset.get("shift_factor"),
        "expected_latent_shape": [int(preset["latent_channels"]), latent_side, latent_side],
        "expected_flat_dim": latent_dim,
        "coordinate": "scaled_posterior_mean_flat",
        "posterior_logvar_coordinate": "unscaled_vae_posterior_logvar_flat",
        "output_dir": str(output_dir),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "image_size": int(image_size),
        "precision": precision,
        "storage_dtype": storage_dtype,
        "save_logvar": bool(save_logvar),
        "save_raw_mean": bool(save_raw_mean),
        "save_reconstruction_mse": bool(save_reconstruction_mse),
        "allow_download": bool(allow_download),
        "local_files_only": not bool(allow_download),
        "max_samples": max_samples,
        "start_index": int(start_index),
        "end_index": end_index,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "effective_sample_count": selected_manifest.sample_count,
        "source_manifest": manifest_summary(manifest),
        "manifest": manifest_summary(selected_manifest),
        "estimated_mean_bytes": selected_manifest.sample_count * latent_dim * np.dtype(storage_dtype).itemsize,
        "estimated_raw_mean_bytes": (
            selected_manifest.sample_count * latent_dim * np.dtype(storage_dtype).itemsize if save_raw_mean else 0
        ),
        "estimated_logvar_bytes": (
            selected_manifest.sample_count * latent_dim * np.dtype(storage_dtype).itemsize if save_logvar else 0
        ),
    }
    return selected_manifest, plan


def cache_vae_latents(
    *,
    manifest: ImageFolderManifest,
    plan: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    output_dir = Path(str(plan["output_dir"]))
    completed = output_dir / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"Cache already completed: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import AutoencoderKL
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class ManifestImageDataset(Dataset):
        def __init__(self) -> None:
            self.root = Path(manifest.root)
            self.rows = list(manifest.rows)
            self.transform = transforms.Compose(
                [
                    transforms.Resize(int(plan["image_size"]), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(int(plan["image_size"])),
                    transforms.ToTensor(),
                ]
            )

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            image = Image.open(self.root / row.relative_path).convert("RGB")
            return self.transform(image), int(index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(plan.get("precision", "bf16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.float32 if precision == "fp32" else autocast_dtype

    vae = AutoencoderKL.from_pretrained(
        str(plan["pretrained_path"]),
        subfolder=str(plan.get("subfolder", "")),
        local_files_only=bool(plan.get("local_files_only", True)),
    ).to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)

    scaling_factor = _vae_scaling_factor(vae, plan, device=device, dtype=torch.float32)
    shift_factor = _vae_shift_factor(vae, plan, device=device, dtype=torch.float32)

    dataset = ManifestImageDataset()
    loader = DataLoader(
        dataset,
        batch_size=int(plan["batch_size"]),
        shuffle=False,
        num_workers=int(plan.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    embeddings = None
    raw_means = None
    logvars = None
    storage_dtype = np.dtype(str(plan.get("storage_dtype", "float16")))
    metric_rows: list[dict[str, object]] = []
    latent_shape: list[int] | None = None
    n_rows = len(dataset)

    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device=device, dtype=dtype, non_blocking=True)
            vae_input = images.mul(2.0).sub(1.0)
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    posterior = vae.encode(vae_input).latent_dist
            else:
                posterior = vae.encode(vae_input).latent_dist
            mean = _posterior_mean(posterior).float()
            logvar = _posterior_logvar(posterior)
            scaled_mean = (mean - shift_factor) * scaling_factor
            batch_flat = scaled_mean.flatten(1).detach().cpu().numpy()
            if embeddings is None:
                latent_shape = [int(v) for v in scaled_mean.shape[1:]]
                embeddings = np.lib.format.open_memmap(
                    output_dir / "embeddings.npy",
                    mode="w+",
                    dtype=storage_dtype,
                    shape=(n_rows, int(batch_flat.shape[1])),
                )
                if bool(plan.get("save_logvar", True)) and logvar is not None:
                    logvars = np.lib.format.open_memmap(
                        output_dir / "posterior_logvar.npy",
                        mode="w+",
                        dtype=storage_dtype,
                        shape=(n_rows, int(batch_flat.shape[1])),
                    )
                if bool(plan.get("save_raw_mean", True)):
                    raw_means = np.lib.format.open_memmap(
                        output_dir / "posterior_mean_raw.npy",
                        mode="w+",
                        dtype=storage_dtype,
                        shape=(n_rows, int(batch_flat.shape[1])),
                    )
            indices_np = indices.detach().cpu().numpy().astype("int64", copy=False)
            embeddings[indices_np] = batch_flat.astype(storage_dtype, copy=False)
            if raw_means is not None:
                raw_means[indices_np] = mean.flatten(1).detach().cpu().numpy().astype(storage_dtype, copy=False)
            if logvars is not None and logvar is not None:
                logvars[indices_np] = logvar.float().flatten(1).detach().cpu().numpy().astype(storage_dtype, copy=False)
            reconstruction_mse = None
            if bool(plan.get("save_reconstruction_mse", False)):
                decoded = vae.decode(scaled_mean / scaling_factor + shift_factor).sample
                decoded = decoded.add(1.0).mul(0.5).clamp(0.0, 1.0)
                reconstruction_mse = (decoded.float() - images.float()).square().flatten(1).mean(dim=1)
            kl = _posterior_kl_diagonal(mean, logvar).detach().cpu().numpy() if logvar is not None else None
            norm_scaled = scaled_mean.flatten(1).norm(dim=1).detach().cpu().numpy()
            norm_unscaled = mean.flatten(1).norm(dim=1).detach().cpu().numpy()
            mean_scalar = scaled_mean.flatten(1).mean(dim=1).detach().cpu().numpy()
            std_scalar = scaled_mean.flatten(1).std(dim=1, unbiased=False).detach().cpu().numpy()
            rec_mse_np = reconstruction_mse.detach().cpu().numpy() if reconstruction_mse is not None else None
            for local_offset, global_index in enumerate(indices_np):
                row = manifest.rows[int(global_index)]
                metric = {
                    "sample_id": row.sample_id,
                    "class_id": int(row.class_id),
                    "class_name": row.class_name,
                    "posterior_kl_raw": "" if kl is None else float(kl[local_offset]),
                    "posterior_kl": "" if kl is None else float(kl[local_offset]),
                    "latent_norm_raw": float(norm_unscaled[local_offset]),
                    "latent_norm_scaled_mean": float(norm_scaled[local_offset]),
                    "latent_norm_unscaled_mean": float(norm_unscaled[local_offset]),
                    "latent_mean_scalar": float(mean_scalar[local_offset]),
                    "latent_std_scalar": float(std_scalar[local_offset]),
                    "reconstruction_mse": "" if rec_mse_np is None else float(rec_mse_np[local_offset]),
                }
                metric_rows.append(metric)

    if embeddings is None:
        np.save(output_dir / "embeddings.npy", np.zeros((0, 0), dtype=storage_dtype))
        latent_shape = []
    else:
        embeddings.flush()
        if logvars is not None:
            logvars.flush()
        if raw_means is not None:
            raw_means.flush()
    _link_posterior_mean_alias(output_dir)
    _write_manifest_files(manifest, output_dir)
    _write_metric_rows(output_dir / "posterior_metrics.csv", metric_rows)

    metadata = dict(plan)
    metadata["embedding_shape"] = list(np.load(output_dir / "embeddings.npy", mmap_mode="r").shape)
    metadata["embedding_dtype"] = str(np.load(output_dir / "embeddings.npy", mmap_mode="r").dtype)
    metadata["latent_shape"] = latent_shape
    metadata["cached_manifest_hash"] = manifest.manifest_hash
    metadata["torch_device"] = str(device)
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    metadata["resolved_scaling_factor"] = _factor_to_json(scaling_factor)
    metadata["resolved_shift_factor"] = _factor_to_json(shift_factor)
    metadata["coordinate_warning"] = (
        "Primary embeddings are deterministic scaled posterior means. They are not posterior samples. "
        "Generic ADA kNN/region utilities L2-normalize embeddings internally; use VAE-specific Euclidean "
        "scores when radial prior/aggregate-posterior information is important."
    )
    metadata["outputs"] = {
        "embeddings_npy": str(output_dir / "embeddings.npy"),
        "mu_flat_npy": str(output_dir / "mu_flat.npy"),
        "posterior_mean_raw_npy": str(output_dir / "posterior_mean_raw.npy")
        if (output_dir / "posterior_mean_raw.npy").exists()
        else "",
        "posterior_logvar_npy": str(output_dir / "posterior_logvar.npy")
        if (output_dir / "posterior_logvar.npy").exists()
        else "",
        "posterior_metrics_csv": str(output_dir / "posterior_metrics.csv"),
        "manifest_csv": str(output_dir / "manifest.csv"),
    }
    metadata["rows"] = [asdict(row) for row in manifest.rows[:5]]
    metadata["artifact_id"] = stable_hash(
        {
            "cache_id": metadata["cache_id"],
            "manifest_hash": manifest.manifest_hash,
            "embedding_sha1": file_sha1(output_dir / "embeddings.npy"),
            "metrics_hash": hash_rows(metric_rows, prefix="vae-metrics"),
        },
        prefix="vae-cache-artifact",
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return output_dir


def fit_vae_pca_projection(
    *,
    train_cache: str | Path,
    output_dir: str | Path,
    pca_dim: int = 512,
    standardize: bool = True,
    whiten: bool = False,
    epsilon: float = 1.0e-6,
    batch_size: int = 4096,
    mmap: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    train_dir = Path(train_cache)
    output = Path(output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"Projection already completed: {output}")
    output.mkdir(parents=True, exist_ok=True)

    x = _load_feature_matrix(train_dir, mmap=mmap)
    if x.ndim != 2:
        raise ValueError("VAE feature matrix must be 2-dimensional")
    n_rows, dim = int(x.shape[0]), int(x.shape[1])
    if n_rows < 2:
        raise ValueError("At least two rows are required to fit PCA")
    if pca_dim < 1 or pca_dim > dim:
        raise ValueError(f"pca_dim must satisfy 1 <= pca_dim <= {dim}")

    mean, std = _feature_mean_std(x, batch_size=batch_size, standardize=standardize, epsilon=epsilon)
    cov = _feature_covariance(x, mean=mean, std=std, batch_size=batch_size)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].astype("float32", copy=False)
    eigvecs = eigvecs[:, order].astype("float32", copy=False)
    components = eigvecs[:, : int(pca_dim)].T.astype("float32", copy=False)
    explained = eigvals[: int(pca_dim)]
    projection_id = stable_hash(
        {
            "train_cache": str(train_dir),
            "train_cache_id": _metadata_value(train_dir, "cache_id"),
            "train_manifest_hash": _metadata_value(train_dir, "cached_manifest_hash"),
            "embedding_sha1": file_sha1(train_dir / "embeddings.npy"),
            "pca_dim": int(pca_dim),
            "standardize": bool(standardize),
            "whiten": bool(whiten),
            "epsilon": float(epsilon),
        },
        prefix="vae-proj",
    )
    np.savez_compressed(
        output / "projection.npz",
        mean=mean.astype("float32", copy=False),
        std=std.astype("float32", copy=False),
        components=components,
        explained_variance=explained.astype("float32", copy=False),
        all_explained_variance=eigvals.astype("float32", copy=False),
        whiten=np.asarray([bool(whiten)]),
        epsilon=np.asarray([float(epsilon)], dtype="float32"),
    )
    total_variance = float(np.maximum(eigvals.sum(), float(epsilon)))
    explained_ratio = (eigvals / total_variance).astype("float32", copy=False)
    np.save(output / "feature_mean.npy", mean.astype("float32", copy=False))
    np.save(output / "feature_std.npy", std.astype("float32", copy=False))
    np.save(output / "pca_components.npy", components)
    np.save(output / "pca_mean.npy", np.zeros((int(pca_dim),), dtype="float32"))
    np.save(output / "explained_variance.npy", explained.astype("float32", copy=False))
    np.save(output / "explained_variance_ratio.npy", explained_ratio[: int(pca_dim)])
    metadata = {
        "projection_id": projection_id,
        "projection_format": "ada_vae_pca_projection_v1",
        "train_cache": str(train_dir),
        "train_cache_id": _metadata_value(train_dir, "cache_id"),
        "train_manifest_hash": _metadata_value(train_dir, "cached_manifest_hash"),
        "input_embedding_shape": [n_rows, dim],
        "pca_dim": int(pca_dim),
        "standardize": bool(standardize),
        "whiten": bool(whiten),
        "epsilon": float(epsilon),
        "embedding_sha1": file_sha1(train_dir / "embeddings.npy"),
        "pca_explained_variance_total": float(explained_ratio[: int(pca_dim)].sum()),
        "outputs": {
            "projection_npz": str(output / "projection.npz"),
            "feature_mean_npy": str(output / "feature_mean.npy"),
            "feature_std_npy": str(output / "feature_std.npy"),
            "pca_components_npy": str(output / "pca_components.npy"),
            "pca_mean_npy": str(output / "pca_mean.npy"),
            "explained_variance_npy": str(output / "explained_variance.npy"),
            "explained_variance_ratio_npy": str(output / "explained_variance_ratio.npy"),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def project_vae_latent_cache(
    *,
    input_cache: str | Path,
    projection_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 4096,
    mmap: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    input_dir = Path(input_cache)
    projection_path = Path(projection_dir)
    output = Path(output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"Projected cache already completed: {output}")
    output.mkdir(parents=True, exist_ok=True)

    x = _load_feature_matrix(input_dir, mmap=mmap)
    projection = np.load(projection_path / "projection.npz")
    mean = projection["mean"].astype("float32", copy=False)
    std = projection["std"].astype("float32", copy=False)
    components = projection["components"].astype("float32", copy=False)
    explained = projection["explained_variance"].astype("float32", copy=False)
    whiten = bool(projection["whiten"][0])
    epsilon = float(projection["epsilon"][0])
    if x.shape[1] != mean.shape[0] or components.shape[1] != mean.shape[0]:
        raise ValueError("Projection dimensionality does not match input cache")

    n_rows = int(x.shape[0])
    projected_dim = int(components.shape[0])
    out = np.lib.format.open_memmap(
        output / "embeddings.npy",
        mode="w+",
        dtype="float32",
        shape=(n_rows, projected_dim),
    )
    denom = np.sqrt(np.maximum(explained, epsilon)).astype("float32", copy=False) if whiten else None
    for start in range(0, n_rows, int(batch_size)):
        end = min(start + int(batch_size), n_rows)
        chunk = np.asarray(x[start:end], dtype="float32")
        chunk = (chunk - mean) / std
        values = chunk @ components.T
        if denom is not None:
            values = values / denom
        out[start:end] = values.astype("float32", copy=False)
    out.flush()

    _copy_manifest_files(input_dir, output)
    input_meta = _load_metadata(input_dir)
    projection_meta = _load_metadata(projection_path)
    metadata = {
        "cache_id": stable_hash(
            {
                "input_cache_id": input_meta.get("cache_id"),
                "input_manifest_hash": input_meta.get("cached_manifest_hash"),
                "projection_id": projection_meta.get("projection_id"),
                "projection_sha1": file_sha1(projection_path / "projection.npz"),
            },
            prefix="vae-projected-cache",
        ),
        "cache_format": "ada_projected_vae_latent_cache_v1",
        "dataset": input_meta.get("dataset"),
        "split": input_meta.get("split"),
        "source_input_cache": str(input_dir),
        "source_input_cache_id": input_meta.get("cache_id"),
        "source_input_coordinate": input_meta.get("coordinate"),
        "projection_dir": str(projection_path),
        "projection_id": projection_meta.get("projection_id"),
        "coordinate": "train_standardized_pca_vae_posterior_mean",
        "whiten": whiten,
        "embedding_shape": [n_rows, projected_dim],
        "embedding_dtype": "float32",
        "cached_manifest_hash": input_meta.get("cached_manifest_hash"),
        "warning": (
            "This projected cache is compatible with generic ADA support/region utilities. Those utilities "
            "currently L2-normalize features internally, so this is a compatibility atlas rather than the "
            "final VAE Euclidean/prior-aware geometry."
        ),
        "outputs": {
            "embeddings_npy": str(output / "embeddings.npy"),
            "manifest_csv": str(output / "manifest.csv"),
            "source_posterior_metrics_csv": input_meta.get("outputs", {}).get("posterior_metrics_csv", ""),
        },
    }
    metadata["embedding_sha1"] = file_sha1(output / "embeddings.npy")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def fit_vae_standardization(
    *,
    train_cache: str | Path,
    output_dir: str | Path,
    epsilon: float = 1.0e-6,
    batch_size: int = 4096,
    mmap: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    train_dir = Path(train_cache)
    output = Path(output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"Standardization already completed: {output}")
    output.mkdir(parents=True, exist_ok=True)

    x = _load_feature_matrix(train_dir, mmap=mmap)
    if x.ndim != 2:
        raise ValueError("VAE feature matrix must be 2-dimensional")
    n_rows, dim = int(x.shape[0]), int(x.shape[1])
    mean, std = _feature_mean_std(x, batch_size=batch_size, standardize=True, epsilon=epsilon)
    standardization_id = stable_hash(
        {
            "train_cache": str(train_dir),
            "train_cache_id": _metadata_value(train_dir, "cache_id"),
            "train_manifest_hash": _metadata_value(train_dir, "cached_manifest_hash"),
            "embedding_sha1": file_sha1(train_dir / "embeddings.npy"),
            "epsilon": float(epsilon),
        },
        prefix="vae-std",
    )
    np.savez_compressed(
        output / "standardization.npz",
        mean=mean.astype("float32", copy=False),
        std=std.astype("float32", copy=False),
        epsilon=np.asarray([float(epsilon)], dtype="float32"),
    )
    np.save(output / "feature_mean.npy", mean.astype("float32", copy=False))
    np.save(output / "feature_std.npy", std.astype("float32", copy=False))
    metadata = {
        "standardization_id": standardization_id,
        "standardization_format": "ada_vae_flat_standardization_v1",
        "train_cache": str(train_dir),
        "train_cache_id": _metadata_value(train_dir, "cache_id"),
        "train_manifest_hash": _metadata_value(train_dir, "cached_manifest_hash"),
        "input_embedding_shape": [n_rows, dim],
        "epsilon": float(epsilon),
        "embedding_sha1": file_sha1(train_dir / "embeddings.npy"),
        "outputs": {
            "standardization_npz": str(output / "standardization.npz"),
            "feature_mean_npy": str(output / "feature_mean.npy"),
            "feature_std_npy": str(output / "feature_std.npy"),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def standardize_vae_latent_cache(
    *,
    input_cache: str | Path,
    standardization_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 4096,
    mmap: bool = True,
    storage_dtype: str = "float32",
    scale_by_sqrt_dim: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    input_dir = Path(input_cache)
    standardization_path = Path(standardization_dir)
    output = Path(output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"Standardized cache already completed: {output}")
    output.mkdir(parents=True, exist_ok=True)

    x = _load_feature_matrix(input_dir, mmap=mmap)
    standardization = np.load(standardization_path / "standardization.npz")
    mean = standardization["mean"].astype("float32", copy=False)
    std = standardization["std"].astype("float32", copy=False)
    if x.ndim != 2:
        raise ValueError("VAE feature matrix must be 2-dimensional")
    if x.shape[1] != mean.shape[0]:
        raise ValueError("Standardization dimensionality does not match input cache")

    n_rows, dim = int(x.shape[0]), int(x.shape[1])
    dtype = np.dtype(storage_dtype)
    out = np.lib.format.open_memmap(
        output / "embeddings.npy",
        mode="w+",
        dtype=dtype,
        shape=(n_rows, dim),
    )
    scale = math.sqrt(float(dim)) if scale_by_sqrt_dim else 1.0
    for start in range(0, n_rows, int(batch_size)):
        end = min(start + int(batch_size), n_rows)
        chunk = np.asarray(x[start:end], dtype="float32")
        values = (chunk - mean) / std
        values = values / scale
        out[start:end] = values.astype(dtype, copy=False)
    out.flush()

    _copy_manifest_files(input_dir, output)
    input_meta = _load_metadata(input_dir)
    standardization_meta = _load_metadata(standardization_path)
    metadata = {
        "cache_id": stable_hash(
            {
                "input_cache_id": input_meta.get("cache_id"),
                "input_manifest_hash": input_meta.get("cached_manifest_hash"),
                "standardization_id": standardization_meta.get("standardization_id"),
                "standardization_sha1": file_sha1(standardization_path / "standardization.npz"),
                "scale_by_sqrt_dim": bool(scale_by_sqrt_dim),
                "storage_dtype": storage_dtype,
            },
            prefix="vae-standardized-cache",
        ),
        "cache_format": "ada_standardized_vae_latent_cache_v1",
        "dataset": input_meta.get("dataset"),
        "split": input_meta.get("split"),
        "source_input_cache": str(input_dir),
        "source_input_cache_id": input_meta.get("cache_id"),
        "source_input_coordinate": input_meta.get("coordinate"),
        "standardization_dir": str(standardization_path),
        "standardization_id": standardization_meta.get("standardization_id"),
        "coordinate": "train_standardized_flat_vae_posterior_mean",
        "scale_by_sqrt_dim": bool(scale_by_sqrt_dim),
        "embedding_shape": [n_rows, dim],
        "embedding_dtype": storage_dtype,
        "cached_manifest_hash": input_meta.get("cached_manifest_hash"),
        "warning": (
            "This cache keeps the full flattened VAE posterior-mean coordinate. It standardizes each "
            "coordinate using train-only statistics and does not rotate, truncate, or L2-normalize features."
        ),
        "outputs": {
            "embeddings_npy": str(output / "embeddings.npy"),
            "manifest_csv": str(output / "manifest.csv"),
            "source_posterior_metrics_csv": input_meta.get("outputs", {}).get("posterior_metrics_csv", ""),
        },
    }
    metadata["embedding_sha1"] = file_sha1(output / "embeddings.npy")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _posterior_mean(posterior) -> Any:
    mean = getattr(posterior, "mean", None)
    if mean is not None:
        return mean
    return posterior.mode()


def _posterior_logvar(posterior) -> Any | None:
    logvar = getattr(posterior, "logvar", None)
    if logvar is not None:
        return logvar
    parameters = getattr(posterior, "parameters", None)
    if parameters is None:
        return None
    channels = parameters.shape[1] // 2
    return parameters[:, channels:]


def _posterior_kl_diagonal(mean, logvar):
    if logvar is None:
        raise ValueError("logvar is required to compute diagonal Gaussian KL")
    return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).flatten(1).sum(dim=1)


def _vae_scaling_factor(vae, plan: Mapping[str, Any], *, device, dtype):
    import torch

    config = getattr(vae, "config", object())
    if hasattr(config, "latents_std") and getattr(config, "latents_std") is not None:
        values = torch.tensor(getattr(config, "latents_std"), device=device, dtype=dtype).reshape(1, -1, 1, 1)
        return 1.0 / values
    value = plan.get("scaling_factor", getattr(config, "scaling_factor", None))
    if value is None:
        value = VAE_PRESETS[str(plan["vae_type"])].get("scaling_factor", 1.0)
    return torch.as_tensor(value, device=device, dtype=dtype).reshape(1, -1, 1, 1) if isinstance(value, list) else torch.tensor(
        float(value), device=device, dtype=dtype
    )


def _vae_shift_factor(vae, plan: Mapping[str, Any], *, device, dtype):
    import torch

    config = getattr(vae, "config", object())
    if hasattr(config, "latents_mean") and getattr(config, "latents_mean") is not None:
        return torch.tensor(getattr(config, "latents_mean"), device=device, dtype=dtype).reshape(1, -1, 1, 1)
    value = plan.get("shift_factor", getattr(config, "shift_factor", None))
    if value is None:
        value = VAE_PRESETS[str(plan["vae_type"])].get("shift_factor", 0.0)
    return torch.as_tensor(value, device=device, dtype=dtype).reshape(1, -1, 1, 1) if isinstance(value, list) else torch.tensor(
        float(value), device=device, dtype=dtype
    )


def _feature_mean_std(
    x: np.ndarray,
    *,
    batch_size: int,
    standardize: bool,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    dim = int(x.shape[1])
    if not standardize:
        return np.zeros((dim,), dtype="float32"), np.ones((dim,), dtype="float32")
    total = np.zeros((dim,), dtype="float64")
    total_sq = np.zeros((dim,), dtype="float64")
    n_rows = int(x.shape[0])
    for start in range(0, n_rows, int(batch_size)):
        end = min(start + int(batch_size), n_rows)
        chunk = np.asarray(x[start:end], dtype="float64")
        total += chunk.sum(axis=0)
        total_sq += np.square(chunk).sum(axis=0)
    mean = total / float(n_rows)
    var = np.maximum(total_sq / float(n_rows) - np.square(mean), float(epsilon))
    std = np.sqrt(var)
    return mean.astype("float32"), std.astype("float32")


def _feature_covariance(
    x: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    n_rows, dim = int(x.shape[0]), int(x.shape[1])
    cov = np.zeros((dim, dim), dtype="float64")
    for start in range(0, n_rows, int(batch_size)):
        end = min(start + int(batch_size), n_rows)
        chunk = np.asarray(x[start:end], dtype="float32")
        chunk = (chunk - mean) / std
        cov += chunk.T.astype("float64") @ chunk.astype("float64")
    return (cov / float(max(n_rows - 1, 1))).astype("float32")


def _load_feature_matrix(cache_dir: Path, *, mmap: bool) -> np.ndarray:
    path = cache_dir / "embeddings.npy"
    if not path.exists():
        path = cache_dir / "mu_flat.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing embeddings.npy or mu_flat.npy in {cache_dir}")
    return np.load(path, mmap_mode="r" if mmap else None)


def _metadata_value(cache_dir: Path, key: str) -> object:
    return _load_metadata(cache_dir).get(key, "")


def _load_metadata(path: Path) -> dict[str, object]:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def _factor_to_json(value) -> object:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().float()
            if int(cpu.numel()) == 1:
                return float(cpu.item())
            return [float(item) for item in cpu.flatten().tolist()]
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return str(value)


def _copy_manifest_files(source: Path, output: Path) -> None:
    for name in ("manifest.csv",):
        shutil.copyfile(source / name, output / name)
    if (source / "metadata.json").exists():
        source_meta = _load_metadata(source)
        if "rows" in source_meta:
            (output / "source_rows_preview.json").write_text(json.dumps(source_meta["rows"], indent=2, sort_keys=True))


def _write_manifest_files(manifest: ImageFolderManifest, output: Path) -> None:
    from ada.atlas.data.manifests import write_manifest_files

    write_manifest_files(manifest, output)


def _write_metric_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id",
        "class_id",
        "class_name",
        "posterior_kl_raw",
        "posterior_kl",
        "latent_norm_raw",
        "latent_norm_scaled_mean",
        "latent_norm_unscaled_mean",
        "latent_mean_scalar",
        "latent_std_scalar",
        "reconstruction_mse",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _link_posterior_mean_alias(output_dir: Path) -> None:
    alias = output_dir / "mu_flat.npy"
    if alias.exists():
        return
    try:
        os.link(output_dir / "embeddings.npy", alias)
    except OSError:
        try:
            alias.symlink_to("embeddings.npy")
        except OSError:
            pass
