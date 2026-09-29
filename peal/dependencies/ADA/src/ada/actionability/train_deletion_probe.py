from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.hashing import hash_rows, stable_hash
from ada.atlas.predictions.common import PREDICTION_COLUMNS


EXTRA_PREDICTION_COLUMNS = [
    "true_class_logit",
    "max_nontrue_logit",
    "true_class_logit_margin",
    "true_class_probability",
]


@dataclass(frozen=True)
class DeletionProbeConfig:
    train_cache: Path
    validation_cache: Path
    deletion_controls_dir: Path
    output_dir: Path
    experiment_id: str = "e4a_in100_k10_deletion_probe"
    model_id: str = "dino_cls_deletion_probe"
    feature_id: str = ""
    fixed_steps: int = 1000
    batch_size: int = 2048
    lr: float = 1.0e-2
    weight_decay: float = 1.0e-4
    device: str = "auto"
    overwrite: bool = False


def train_deletion_probe_for_index(config: DeletionProbeConfig, manifest_index: int) -> dict[str, object]:
    manifest_rows = _read_csv(Path(config.deletion_controls_dir) / "deletion_control_manifests.csv")
    if manifest_index < 0 or manifest_index >= len(manifest_rows):
        raise IndexError(f"manifest_index={manifest_index} outside [0, {len(manifest_rows)})")
    return train_deletion_probe(config, manifest_rows[int(manifest_index)], manifest_index=int(manifest_index))


def train_deletion_probe(config: DeletionProbeConfig, manifest_row: Mapping[str, object], *, manifest_index: int) -> dict[str, object]:
    import torch

    output = Path(config.output_dir) / f"manifest_{int(manifest_index):04d}_{manifest_row['manifest_id']}"
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        return json.loads((output / "metadata.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    train_dir = Path(config.train_cache)
    val_dir = Path(config.validation_cache)
    x_train_np = np.load(train_dir / "embeddings.npy", mmap_mode="r").astype("float32", copy=False)
    x_val_np = np.load(val_dir / "embeddings.npy", mmap_mode="r").astype("float32", copy=False)
    train_rows = load_manifest_csv(train_dir / "manifest.csv")
    val_rows = load_manifest_csv(val_dir / "manifest.csv")
    y_train_np = np.asarray([row.class_id for row in train_rows], dtype=np.int64)
    y_val_np = np.asarray([row.class_id for row in val_rows], dtype=np.int64)
    sample_to_index = {row.sample_id: idx for idx, row in enumerate(train_rows)}

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
        multiplicity[sample_to_index[sample_id]] = float(row["multiplicity"])
        raw_loss_weight = row.get("loss_weight", "")
        if raw_loss_weight not in ("", None):
            value = float(raw_loss_weight)
            if value <= 0.0:
                raise ValueError(f"loss_weight must be positive for sample_id={sample_id}")
            loss_weight[sample_to_index[sample_id]] = value
    if float(multiplicity.sum()) <= 0.0:
        raise ValueError("exposure manifest has zero total multiplicity")

    num_classes = int(max(y_train_np.max(), y_val_np.max()) + 1)
    torch_device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    seed = int(manifest_row["seed"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.from_numpy(multiplicity.astype("float32", copy=False))
    sample_loss_weight = torch.from_numpy(loss_weight.astype("float32", copy=False))
    x_train = torch.from_numpy(np.asarray(x_train_np, dtype=np.float32).copy())
    y_train = torch.from_numpy(np.asarray(y_train_np).copy())

    model = torch.nn.Linear(x_train.shape[1], num_classes).to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    for _step in range(int(config.fixed_steps)):
        idx = torch.multinomial(weights, num_samples=int(config.batch_size), replacement=True, generator=gen)
        xb = x_train.index_select(0, idx).to(torch_device)
        yb = y_train.index_select(0, idx).to(torch_device)
        wb = sample_loss_weight.index_select(0, idx).to(torch_device)
        opt.zero_grad(set_to_none=True)
        per_sample_loss = torch.nn.functional.cross_entropy(model(xb), yb, reduction="none")
        loss = (per_sample_loss * wb).mean()
        loss.backward()
        opt.step()

    pred_csv = output / "predictions.csv"
    x_val = torch.from_numpy(np.asarray(x_val_np, dtype=np.float32).copy())
    correct = 0
    nll_sum = 0.0
    prediction_columns = list(PREDICTION_COLUMNS)
    for column in EXTRA_PREDICTION_COLUMNS:
        if column not in prediction_columns:
            prediction_columns.append(column)
    with pred_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_columns)
        writer.writeheader()
        for start in range(0, x_val.shape[0], int(config.batch_size)):
            end = min(start + int(config.batch_size), x_val.shape[0])
            xb = x_val[start:end].to(torch_device)
            with torch.no_grad():
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                top_vals, top_idx = torch.topk(logits, k=2, dim=1)
            for offset, row in enumerate(val_rows[start:end]):
                true_label = int(y_val_np[start + offset])
                pred = int(top_idx[offset, 0].item())
                prob = probs[offset].detach().cpu()
                logit_row = logits[offset].detach().cpu()
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

    metadata = {
        "artifact_id": stable_hash(
            {
                "manifest_id": manifest_row["manifest_id"],
                "manifest_index": int(manifest_index),
                "fixed_steps": int(config.fixed_steps),
                "batch_size": int(config.batch_size),
                "lr": float(config.lr),
                "weight_decay": float(config.weight_decay),
                "experiment_id": config.experiment_id,
                "model_id": config.model_id,
                "feature_id": config.feature_id,
                "train_cache": str(train_dir),
                "validation_cache": str(val_dir),
            },
            prefix="deletion-probe",
        ),
        "manifest_index": int(manifest_index),
        "manifest_row": dict(manifest_row),
        "experiment_id": config.experiment_id,
        "train_cache": str(train_dir),
        "validation_cache": str(val_dir),
        "model_id": config.model_id,
        "feature_id": config.feature_id,
        "fixed_steps": int(config.fixed_steps),
        "batch_size": int(config.batch_size),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "device": str(torch_device),
        "query_accuracy": correct / len(val_rows),
        "query_nll": nll_sum / len(val_rows),
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


def write_probe_index(output_dir: str | Path) -> Path:
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
                "feature_id": str(meta.get("feature_id", "")),
                "query_accuracy": float(meta["query_accuracy"]),
                "predictions_csv": meta["predictions_csv"],
                "metadata_json": str(metadata_path),
            }
        )
    index = output / "probe_runs.csv"
    _write_csv(index, rows)
    return index


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
