from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

import numpy as np

from ada.atlas.data.manifests import ManifestRow, load_manifest_csv
from ada.atlas.hashing import stable_hash
from ada.atlas.predictions.common import PREDICTION_COLUMNS


EXTRA_PREDICTION_COLUMNS = [
    "true_class_logit",
    "max_nontrue_logit",
    "true_class_logit_margin",
    "true_class_probability",
]

SUPPORTED_ARCHITECTURES = (
    "torchvision_resnet18",
    "torchvision_resnet50",
    "torchvision_convnext_tiny",
    "torchvision_vit_b_16",
)


@dataclass(frozen=True)
class ImageClassifierConfig:
    train_cache: Path
    validation_cache: Path
    deletion_controls_dir: Path
    output_dir: Path
    train_image_root: Path | None = None
    validation_image_root: Path | None = None
    experiment_id: str = "e8a_in100_image_classifier_causal_screen"
    model_id: str = "resnet18_scratch_image_classifier"
    architecture: str = "torchvision_resnet18"
    pretrained: bool = False
    freeze_backbone: bool = False
    allow_weight_download: bool = False
    fixed_steps: int = 2000
    batch_size: int = 128
    eval_batch_size: int = 256
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    image_size: int = 224
    train_augmentation: str = "standard"
    precision: str = "fp32"
    num_workers: int = 8
    device: str = "auto"
    overwrite: bool = False


def train_image_classifier_for_index(config: ImageClassifierConfig, manifest_index: int) -> dict[str, object]:
    manifest_rows = _read_csv(Path(config.deletion_controls_dir) / "deletion_control_manifests.csv")
    if manifest_index < 0 or manifest_index >= len(manifest_rows):
        raise IndexError(f"manifest_index={manifest_index} outside [0, {len(manifest_rows)})")
    return train_image_classifier(config, manifest_rows[int(manifest_index)], manifest_index=int(manifest_index))


def train_image_classifier(config: ImageClassifierConfig, manifest_row: Mapping[str, object], *, manifest_index: int) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    output = Path(config.output_dir) / f"manifest_{int(manifest_index):04d}_{manifest_row['manifest_id']}"
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        return json.loads((output / "metadata.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    train_dir = Path(config.train_cache)
    val_dir = Path(config.validation_cache)
    train_rows = load_manifest_csv(train_dir / "manifest.csv")
    val_rows = load_manifest_csv(val_dir / "manifest.csv")
    sample_to_index = {row.sample_id: idx for idx, row in enumerate(train_rows)}
    if len(sample_to_index) != len(train_rows):
        raise ValueError("train manifest contains duplicate sample IDs")

    exposure_rows = _read_csv(Path(str(manifest_row["exposure_manifest_csv"])))
    if len(exposure_rows) != len(train_rows):
        raise ValueError("exposure manifest must contain one row per train sample")
    multiplicity = np.zeros(len(train_rows), dtype=np.float64)
    loss_weight = np.ones(len(train_rows), dtype=np.float32)
    seen: set[str] = set()
    for row in exposure_rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in exposure manifest: {sample_id}")
        seen.add(sample_id)
        if sample_id not in sample_to_index:
            raise KeyError(f"sample_id in exposure manifest not found in train cache: {sample_id}")
        idx = sample_to_index[sample_id]
        multiplicity[idx] = float(row["multiplicity"])
        raw_loss_weight = row.get("loss_weight", "")
        if raw_loss_weight not in ("", None):
            value = float(raw_loss_weight)
            if value <= 0.0:
                raise ValueError(f"loss_weight must be positive for sample_id={sample_id}")
            loss_weight[idx] = value
    if float(multiplicity.sum()) <= 0.0:
        raise ValueError("exposure manifest has zero total multiplicity")

    train_root = _resolve_image_root(config.train_image_root, train_dir)
    val_root = _resolve_image_root(config.validation_image_root, val_dir)
    num_classes = int(max(max(row.class_id for row in train_rows), max(row.class_id for row in val_rows)) + 1)
    seed = int(manifest_row["seed"])
    _seed_everything(seed)

    torch_device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    model, weights_name = _build_model(
        architecture=str(config.architecture),
        num_classes=num_classes,
        pretrained=bool(config.pretrained),
        freeze_backbone=bool(config.freeze_backbone),
        allow_weight_download=bool(config.allow_weight_download),
    )
    model = model.to(torch_device)

    train_transform, eval_transform = _build_transforms(
        image_size=int(config.image_size),
        train_augmentation=str(config.train_augmentation),
    )
    train_dataset = _ManifestImageDataset(train_rows, train_root, train_transform, loss_weight)
    val_dataset = _ManifestImageDataset(val_rows, val_root, eval_transform, None)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(multiplicity, dtype=torch.double),
        num_samples=int(config.fixed_steps) * int(config.batch_size),
        replacement=True,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.batch_size),
        sampler=sampler,
        num_workers=int(config.num_workers),
        pin_memory=torch_device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=torch_device.type == "cuda",
        drop_last=False,
    )

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    opt = torch.optim.AdamW(trainable, lr=float(config.lr), weight_decay=float(config.weight_decay))
    amp_enabled = torch_device.type == "cuda" and str(config.precision).lower() in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if str(config.precision).lower() == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and autocast_dtype == torch.float16)

    model.train()
    last_loss = math.nan
    for step, batch in enumerate(train_loader):
        if step >= int(config.fixed_steps):
            break
        images, labels, weights = batch
        images = images.to(torch_device, non_blocking=True)
        labels = labels.to(torch_device, non_blocking=True)
        weights = weights.to(torch_device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=torch_device.type, dtype=autocast_dtype, enabled=amp_enabled):
            logits = model(images)
            per_sample_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = (per_sample_loss * weights).mean()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        last_loss = float(loss.detach().cpu().item())

    pred_csv = output / "predictions.csv"
    correct = 0
    nll_sum = 0.0
    prediction_columns = list(PREDICTION_COLUMNS)
    for column in EXTRA_PREDICTION_COLUMNS:
        if column not in prediction_columns:
            prediction_columns.append(column)
    model.eval()
    with pred_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_columns)
        writer.writeheader()
        val_offset = 0
        for images, labels, _weights in val_loader:
            images = images.to(torch_device, non_blocking=True)
            labels = labels.to(torch_device, non_blocking=True)
            with torch.no_grad():
                with torch.autocast(device_type=torch_device.type, dtype=autocast_dtype, enabled=amp_enabled):
                    logits = model(images)
                probs = torch.softmax(logits.float(), dim=1)
                top_vals, top_idx = torch.topk(logits.float(), k=2, dim=1)
            for offset in range(images.shape[0]):
                row = val_rows[val_offset + offset]
                true_label = int(labels[offset].detach().cpu().item())
                pred = int(top_idx[offset, 0].item())
                prob = probs[offset].detach().cpu()
                logit_row = logits[offset].float().detach().cpu()
                true_logit = float(logit_row[true_label].item())
                nontrue_logits = logit_row.clone()
                nontrue_logits[true_label] = -float("inf")
                max_nontrue_logit = float(nontrue_logits.max().item())
                nll = -math.log(max(float(prob[true_label]), 1.0e-12))
                is_correct = int(pred == true_label)
                correct += is_correct
                nll_sum += nll
                writer.writerow(
                    {
                        "sample_id": row.sample_id,
                        "true_label": true_label,
                        "model_id": config.model_id,
                        "predicted_label": pred,
                        "correct": is_correct,
                        "top1_logit": float(top_vals[offset, 0].item()),
                        "top2_logit": float(top_vals[offset, 1].item()),
                        "logit_margin": float((top_vals[offset, 0] - top_vals[offset, 1]).item()),
                        "max_probability_raw": float(prob.max().item()),
                        "entropy_raw": float((-(prob * prob.clamp_min(1.0e-12).log()).sum()).item()),
                        "max_probability_calibrated": float(prob.max().item()),
                        "nll": nll,
                        "true_class_logit": true_logit,
                        "max_nontrue_logit": max_nontrue_logit,
                        "true_class_logit_margin": true_logit - max_nontrue_logit,
                        "true_class_probability": float(prob[true_label].item()),
                    }
                )
            val_offset += images.shape[0]

    metadata = {
        "artifact_id": stable_hash(
            {
                "manifest_id": manifest_row["manifest_id"],
                "manifest_index": int(manifest_index),
                "experiment_id": config.experiment_id,
                "model_id": config.model_id,
                "architecture": config.architecture,
                "pretrained": bool(config.pretrained),
                "freeze_backbone": bool(config.freeze_backbone),
                "fixed_steps": int(config.fixed_steps),
                "batch_size": int(config.batch_size),
                "lr": float(config.lr),
                "weight_decay": float(config.weight_decay),
                "train_cache": str(train_dir),
                "validation_cache": str(val_dir),
                "train_image_root": str(train_root),
                "validation_image_root": str(val_root),
            },
            prefix="image-probe",
        ),
        "manifest_index": int(manifest_index),
        "manifest_row": dict(manifest_row),
        "experiment_id": config.experiment_id,
        "train_cache": str(train_dir),
        "validation_cache": str(val_dir),
        "train_image_root": str(train_root),
        "validation_image_root": str(val_root),
        "model_id": config.model_id,
        "architecture": config.architecture,
        "weights": weights_name,
        "pretrained": bool(config.pretrained),
        "freeze_backbone": bool(config.freeze_backbone),
        "fixed_steps": int(config.fixed_steps),
        "batch_size": int(config.batch_size),
        "eval_batch_size": int(config.eval_batch_size),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "image_size": int(config.image_size),
        "train_augmentation": str(config.train_augmentation),
        "precision": str(config.precision),
        "device": str(torch_device),
        "query_accuracy": correct / len(val_rows),
        "query_nll": nll_sum / len(val_rows),
        "last_train_loss": last_loss,
        "total_exposure_count": int(multiplicity.sum()),
        "active_unique_count": int(np.sum(multiplicity > 0)),
        "loss_weight_min": float(loss_weight.min()),
        "loss_weight_max": float(loss_weight.max()),
        "loss_weight_mean": float(loss_weight.mean()),
        "uses_loss_weight": bool(np.any(np.abs(loss_weight - 1.0) > 1.0e-12)),
        "predictions_csv": str(pred_csv),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, output / "model.pt")
    completed.write_text("ok\n")
    return metadata


def write_image_probe_index(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(output.glob("manifest_*/metadata.json")):
        meta = json.loads(metadata_path.read_text())
        manifest = dict(meta["manifest_row"])
        rows.append(
            {
                "probe_artifact_id": meta["artifact_id"],
                "manifest_index": int(meta["manifest_index"]),
                "manifest_id": manifest["manifest_id"],
                "region_id": manifest["region_id"],
                "class_id": int(manifest["class_id"]),
                "pilot_type": manifest.get("pilot_type", ""),
                "control_family": manifest["control_family"],
                "retention_level": float(manifest["retention_level"]),
                "seed": int(manifest["seed"]),
                "experiment_id": str(meta.get("experiment_id", "")),
                "model_id": str(meta.get("model_id", "")),
                "feature_id": str(meta.get("architecture", "")),
                "query_accuracy": float(meta["query_accuracy"]),
                "predictions_csv": meta["predictions_csv"],
                "metadata_json": str(metadata_path),
            }
        )
    index = output / "probe_runs.csv"
    _write_csv(index, rows)
    return index


class _ManifestImageDataset:
    def __init__(self, rows: Sequence[ManifestRow], root: Path, transform, loss_weight: np.ndarray | None) -> None:
        self.rows = list(rows)
        self.root = Path(root)
        self.transform = transform
        self.loss_weight = loss_weight

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image
        import torch

        row = self.rows[int(index)]
        image = Image.open(self.root / row.relative_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        weight = 1.0 if self.loss_weight is None else float(self.loss_weight[int(index)])
        return image, torch.as_tensor(int(row.class_id), dtype=torch.long), torch.as_tensor(weight, dtype=torch.float32)


def _build_model(
    *,
    architecture: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    allow_weight_download: bool,
):
    import torch
    import torchvision.models as models

    arch = architecture.lower()
    if arch == "torchvision_resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        _require_cached_weight(weights, allow_weight_download=allow_weight_download)
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, int(num_classes))
        head_names = ("fc.",)
    elif arch == "torchvision_resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        _require_cached_weight(weights, allow_weight_download=allow_weight_download)
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, int(num_classes))
        head_names = ("fc.",)
    elif arch == "torchvision_convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        _require_cached_weight(weights, allow_weight_download=allow_weight_download)
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, int(num_classes))
        head_names = ("classifier.2.",)
    elif arch == "torchvision_vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        _require_cached_weight(weights, allow_weight_download=allow_weight_download)
        model = models.vit_b_16(weights=weights)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, int(num_classes))
        head_names = ("heads.head.",)
    else:
        raise ValueError(f"unsupported architecture={architecture!r}; expected one of {SUPPORTED_ARCHITECTURES}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in head_names)
    return model, str(weights) if weights is not None else "random_initialization"


def _build_transforms(*, image_size: int, train_augmentation: str):
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(round(image_size / 0.875))),
            transforms.CenterCrop(int(image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    policy = train_augmentation.lower()
    if policy == "none":
        train_transform = eval_transform
    elif policy == "standard":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(int(image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError("train_augmentation must be one of: none, standard")
    return train_transform, eval_transform


def _require_cached_weight(weights, *, allow_weight_download: bool) -> None:
    if weights is None or allow_weight_download:
        return
    path = _local_weight_path(str(weights.url))
    if not path.exists():
        raise FileNotFoundError(
            f"pretrained weight is not cached locally: {path}. "
            "Set allow_weight_download=true only if a network download is intentional."
        )


def _local_weight_path(url: str) -> Path:
    filename = Path(urlparse(url).path).name
    hub_dir = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
    return hub_dir / filename


def _resolve_image_root(config_root: Path | None, cache_dir: Path) -> Path:
    if config_root is not None and str(config_root):
        return Path(config_root)
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"image root was not configured and cache metadata is missing: {metadata_path}")
    root = json.loads(metadata_path.read_text()).get("root", "")
    if not root:
        raise ValueError(f"image root was not configured and cache metadata has no root field: {metadata_path}")
    return Path(root)


def _seed_everything(seed: int) -> None:
    import random
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
