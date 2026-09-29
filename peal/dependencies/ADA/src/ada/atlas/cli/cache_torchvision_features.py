from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ada.atlas.cache.dinov2_cls import safe_path_name, select_manifest_rows
from ada.atlas.data.manifests import (
    ImageFolderManifest,
    build_imagefolder_manifest,
    manifest_summary,
    write_manifest_files,
)
from ada.atlas.hashing import stable_hash


SUPPORTED_MODELS = {"resnet18", "resnet50"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache torchvision supervised penultimate features for an ImageFolder dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model-name", default="resnet18", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--feature", default="penultimate", choices=["penultimate"])
    parser.add_argument("--output-root", default=Path("artifacts/ada/atlas/embeddings"), type=Path)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--end-index", default=None, type=int)
    parser.add_argument("--shard-index", default=None, type=int)
    parser.add_argument("--num-shards", default=None, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, plan = plan_torchvision_feature_cache(
        dataset=args.dataset,
        split=args.split,
        root=args.root,
        model_name=args.model_name,
        feature=args.feature,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        start_index=args.start_index,
        end_index=args.end_index,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        device=args.device,
    )
    plan["dry_run"] = bool(args.dry_run)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    planned_output = Path(plan["output_dir"])
    if args.reuse_existing and (planned_output / "COMPLETED").exists():
        plan["output_dir"] = str(planned_output)
        plan["reused_existing"] = True
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    output = cache_torchvision_features(manifest=manifest, plan=plan, overwrite=bool(args.overwrite))
    plan["output_dir"] = str(output)
    print(json.dumps(plan, indent=2, sort_keys=True))


def plan_torchvision_feature_cache(
    *,
    dataset: str,
    split: str,
    root: str | Path,
    model_name: str,
    feature: str,
    output_root: str | Path,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    start_index: int,
    end_index: int | None,
    shard_index: int | None,
    num_shards: int | None,
    device: str,
) -> tuple[ImageFolderManifest, dict[str, Any]]:
    model_name = str(model_name).lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported torchvision model: {model_name}")
    if feature != "penultimate":
        raise ValueError("only feature='penultimate' is supported")
    manifest = build_imagefolder_manifest(root, dataset=dataset, split=split)
    selected_manifest = select_manifest_rows(
        manifest,
        max_samples=max_samples,
        start_index=start_index,
        end_index=end_index,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    encoder, weights_name, feature_dim = _model_metadata(model_name)
    cache_id = stable_hash(
        {
            "dataset": dataset,
            "split": split,
            "source_manifest_hash": manifest.manifest_hash,
            "selected_manifest_hash": selected_manifest.manifest_hash,
            "encoder": encoder,
            "model_name": model_name,
            "weights": weights_name,
            "feature": feature,
            "batch_size": int(batch_size),
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
        "model_name": model_name,
        "weights": weights_name,
        "feature_dim": int(feature_dim),
        "output_dir": str(output_dir),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_samples": max_samples,
        "start_index": int(start_index),
        "end_index": end_index,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "device": str(device),
        "effective_sample_count": selected_manifest.sample_count,
        "source_manifest": manifest_summary(manifest),
        "manifest": manifest_summary(selected_manifest),
        "estimated_embedding_bytes_fp32": selected_manifest.sample_count * int(feature_dim) * 4,
    }
    return selected_manifest, plan


def cache_torchvision_features(
    *,
    manifest: ImageFolderManifest,
    plan: dict[str, Any],
    overwrite: bool,
) -> Path:
    output_dir = Path(plan["output_dir"])
    completed = output_dir / "COMPLETED"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"Cache already completed: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    model, weights, feature_dim = _build_feature_model(str(plan["model_name"]))
    preprocess = weights.transforms()
    torch_device = torch.device("cuda" if plan.get("device") == "auto" and torch.cuda.is_available() else ("cpu" if plan.get("device") == "auto" else str(plan.get("device", "cpu"))))
    model = model.to(torch_device).eval()
    model.requires_grad_(False)

    class ManifestImageDataset(Dataset):
        def __init__(self) -> None:
            self.root = Path(manifest.root)
            self.rows = list(manifest.rows)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            image = Image.open(self.root / row.relative_path).convert("RGB")
            return preprocess(image), int(index)

    loader = DataLoader(
        ManifestImageDataset(),
        batch_size=int(plan["batch_size"]),
        shuffle=False,
        num_workers=int(plan.get("num_workers", 8)),
        pin_memory=torch_device.type == "cuda",
        drop_last=False,
    )
    chunks: list[np.ndarray] = []
    for images, _indices in loader:
        xb = images.to(torch_device)
        with torch.inference_mode():
            features = model(xb)
            features = features.flatten(1)
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        chunks.append(features.detach().cpu().numpy().astype("float32", copy=False))
    embeddings = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, int(feature_dim)), dtype="float32")
    np.save(output_dir / "embeddings.npy", embeddings)
    write_manifest_files(manifest, output_dir)

    metadata = dict(plan)
    metadata["embedding_shape"] = list(embeddings.shape)
    metadata["embedding_dtype"] = str(embeddings.dtype)
    metadata["cached_manifest_hash"] = manifest.manifest_hash
    metadata["torch_device"] = str(torch_device)
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    metadata["rows"] = [asdict(row) for row in manifest.rows[:5]]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return output_dir


def _model_metadata(model_name: str) -> tuple[str, str, int]:
    model_name = str(model_name).lower()
    if model_name == "resnet18":
        return "torchvision/resnet18-imagenet1k-v1", "ResNet18_Weights.IMAGENET1K_V1", 512
    if model_name == "resnet50":
        return "torchvision/resnet50-imagenet1k-v2", "ResNet50_Weights.IMAGENET1K_V2", 2048
    raise ValueError(f"unsupported torchvision model: {model_name}")


def _build_feature_model(model_name: str):
    import torch
    import torchvision.models as models

    model_name = str(model_name).lower()
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        feature_dim = 512
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        feature_dim = 2048
    else:
        raise ValueError(f"unsupported torchvision model: {model_name}")
    model.fc = torch.nn.Identity()
    return model, weights, feature_dim


if __name__ == "__main__":
    main()
