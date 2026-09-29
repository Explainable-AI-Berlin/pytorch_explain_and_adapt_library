from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ada.atlas.data.manifests import (
    ImageFolderManifest,
    build_imagefolder_manifest,
    manifest_summary,
    shard_manifest,
    slice_manifest,
    subset_manifest,
    write_manifest_files,
)
from ada.atlas.hashing import stable_hash


def plan_dinov2_cls_cache(
    *,
    dataset: str,
    split: str,
    root: str | Path,
    encoder: str,
    feature: str,
    output_root: str | Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    encoder_input_size: int,
    precision: str,
    max_samples: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> tuple[ImageFolderManifest, dict[str, Any]]:
    if feature not in {"cls", "pooler"}:
        raise ValueError("atlas cache supports feature='cls' or feature='pooler'")
    manifest = build_imagefolder_manifest(root, dataset=dataset, split=split)
    selected_manifest = select_manifest_rows(
        manifest,
        max_samples=max_samples,
        start_index=start_index,
        end_index=end_index,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    effective_count = selected_manifest.sample_count
    cache_id = stable_hash(
        {
            "dataset": dataset,
            "split": split,
            "source_manifest_hash": manifest.manifest_hash,
            "selected_manifest_hash": selected_manifest.manifest_hash,
            "encoder": encoder,
            "feature": feature,
            "image_size": int(image_size),
            "encoder_input_size": int(encoder_input_size),
            "precision": precision,
            "max_samples": max_samples,
            "start_index": int(start_index),
            "end_index": end_index,
            "shard_index": shard_index,
            "num_shards": num_shards,
        },
        prefix="cache",
    )
    output_dir = Path(output_root) / dataset / split / safe_path_name(encoder) / feature / cache_id
    plan = {
        "cache_id": cache_id,
        "dataset": dataset,
        "split": split,
        "root": str(Path(root).expanduser().resolve()),
        "encoder": encoder,
        "feature": feature,
        "output_dir": str(output_dir),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "image_size": int(image_size),
        "encoder_input_size": int(encoder_input_size),
        "precision": precision,
        "max_samples": max_samples,
        "start_index": int(start_index),
        "end_index": end_index,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "effective_sample_count": effective_count,
        "source_manifest": manifest_summary(manifest),
        "manifest": manifest_summary(selected_manifest),
        "estimated_cls_bytes_fp32": effective_count * 768 * 4,
        "estimated_cls_bytes_fp16": effective_count * 768 * 2,
    }
    return selected_manifest, plan


def select_manifest_rows(
    manifest: ImageFolderManifest,
    *,
    max_samples: int | None,
    start_index: int,
    end_index: int | None,
    shard_index: int | None,
    num_shards: int | None,
) -> ImageFolderManifest:
    selected = subset_manifest(manifest, max_samples)
    if (shard_index is None) != (num_shards is None):
        raise ValueError("shard_index and num_shards must be provided together")
    if shard_index is not None and num_shards is not None:
        if start_index != 0 or end_index is not None:
            raise ValueError("Use either shard selection or start/end selection, not both")
        return shard_manifest(selected, int(shard_index), int(num_shards))
    return slice_manifest(selected, int(start_index), end_index)


def safe_path_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(" ", "_")


def cache_dinov2_cls(
    *,
    manifest: ImageFolderManifest,
    plan: dict[str, Any],
    overwrite: bool = False,
) -> Path:
    output_dir = Path(plan["output_dir"])
    completed = output_dir / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"Cache already completed: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports stay inside the non-dry-run path so manifests can be
    # planned in bare Python shells.
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from stage1_semantic_retention_dino_utils import FrozenDINOFeatureExtractor, build_transform

    if str(plan.get("feature", "cls")) == "pooler":
        return _cache_hf_pooler(manifest=manifest, plan=plan, output_dir=output_dir, overwrite=overwrite)

    class ManifestImageDataset(Dataset):
        def __init__(self) -> None:
            self.root = Path(manifest.root)
            self.rows = list(manifest.rows)
            self.transform = build_transform(int(plan["image_size"]))

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            image = Image.open(self.root / row.relative_path).convert("RGB")
            return self.transform(image), int(index)

    dataset = ManifestImageDataset()
    extractor = FrozenDINOFeatureExtractor(
        encoder_path=str(plan["encoder"]),
        encoder_input_size=int(plan["encoder_input_size"]),
        precision=str(plan["precision"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(plan["batch_size"]),
        shuffle=False,
        num_workers=int(plan.get("num_workers", 4)),
        pin_memory=getattr(extractor, "device", torch.device("cpu")).type == "cuda",
        drop_last=False,
    )

    chunks: list[np.ndarray] = []
    for images, _indices in loader:
        views = extractor.encode_views(images)
        cls = views["cls"].numpy().astype("float32", copy=False)
        norms = np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1.0e-12)
        chunks.append(cls / norms)

    embeddings = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 768), dtype="float32")
    np.save(output_dir / "embeddings.npy", embeddings)

    write_manifest_files(manifest, output_dir)

    metadata = dict(plan)
    metadata["embedding_shape"] = list(embeddings.shape)
    metadata["embedding_dtype"] = str(embeddings.dtype)
    metadata["cached_manifest_hash"] = manifest.manifest_hash
    metadata["torch_device"] = str(getattr(extractor, "device", "unknown"))
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    metadata["rows"] = [asdict(row) for row in manifest.rows[:5]]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return output_dir


def _cache_hf_pooler(
    *,
    manifest: ImageFolderManifest,
    plan: dict[str, Any],
    output_dir: Path,
    overwrite: bool,
) -> Path:
    # Heavy imports stay scoped to the SigLIP/AutoModel path.
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoImageProcessor, AutoModel

    class ManifestPILDataset(Dataset):
        def __init__(self) -> None:
            self.root = Path(manifest.root)
            self.rows = list(manifest.rows)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            image = Image.open(self.root / row.relative_path).convert("RGB")
            return image, int(index)

    def collate_pil(batch):
        images, indices = zip(*batch)
        return list(images), list(indices)

    completed = output_dir / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"Cache already completed: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(plan.get("precision", "bf16"))
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    processor = AutoImageProcessor.from_pretrained(str(plan["encoder"]), local_files_only=True)
    model = AutoModel.from_pretrained(str(plan["encoder"]), local_files_only=True).to(device).eval()
    model.requires_grad_(False)

    dataset = ManifestPILDataset()
    loader = DataLoader(
        dataset,
        batch_size=int(plan["batch_size"]),
        shuffle=False,
        num_workers=int(plan.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_pil,
    )

    chunks: list[np.ndarray] = []
    for images, _indices in loader:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    pooler = _forward_pooler(model, inputs)
            else:
                pooler = _forward_pooler(model, inputs)
        values = pooler.float().detach().cpu().numpy().astype("float32", copy=False)
        norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)
        chunks.append(values / norms)

    embeddings = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype="float32")
    np.save(output_dir / "embeddings.npy", embeddings)

    write_manifest_files(manifest, output_dir)

    metadata = dict(plan)
    metadata["embedding_shape"] = list(embeddings.shape)
    metadata["embedding_dtype"] = str(embeddings.dtype)
    metadata["cached_manifest_hash"] = manifest.manifest_hash
    metadata["torch_device"] = str(device)
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    metadata["pooler_backend"] = type(model).__name__
    metadata["rows"] = [asdict(row) for row in manifest.rows[:5]]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return output_dir


def _forward_pooler(model, inputs):
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        raise ValueError("AutoProcessor did not return pixel_values")

    if hasattr(model, "vision_model"):
        outputs = model.vision_model(pixel_values=pixel_values)
    else:
        outputs = model(pixel_values=pixel_values)

    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    image_embeds = getattr(outputs, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=pixel_values)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise ValueError("model output has no pooler_output, image_embeds, or last_hidden_state")
    return hidden[:, 0]
