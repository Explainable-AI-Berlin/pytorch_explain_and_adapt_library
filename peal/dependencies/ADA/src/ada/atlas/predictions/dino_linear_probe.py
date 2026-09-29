from __future__ import annotations

import json
from pathlib import Path

from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.predictions.common import PREDICTION_COLUMNS


def train_dino_linear_probe(
    *,
    train_cache: str | Path,
    query_cache: str | Path,
    output_dir: str | Path,
    epochs: int = 40,
    batch_size: int = 2048,
    lr: float = 1.0e-2,
    weight_decay: float = 1.0e-4,
    calibration_fraction: float = 0.1,
    seed: int = 0,
    model_id: str = "dino_linear_probe",
    device: str = "auto",
) -> Path:
    import csv
    import math
    import numpy as np
    import torch

    train_dir = Path(train_cache)
    query_dir = Path(query_cache)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_train_np = np.load(train_dir / "embeddings.npy").astype("float32", copy=False)
    x_query_np = np.load(query_dir / "embeddings.npy").astype("float32", copy=False)
    train_rows = load_manifest_csv(train_dir / "manifest.csv")
    query_rows = load_manifest_csv(query_dir / "manifest.csv")
    y_train_np = np.asarray([row.class_id for row in train_rows], dtype="int64")
    y_query_np = np.asarray([row.class_id for row in query_rows], dtype="int64")
    num_classes = int(max(y_train_np.max(), y_query_np.max()) + 1)
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))

    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(x_train_np.shape[0], generator=gen)
    calib_count = max(1, int(round(float(calibration_fraction) * x_train_np.shape[0])))
    calib_idx = perm[:calib_count]
    fit_idx = perm[calib_count:]

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    model = torch.nn.Linear(x_train.shape[1], num_classes).to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    fit_x = x_train.index_select(0, fit_idx)
    fit_y = y_train.index_select(0, fit_idx)
    for _epoch in range(int(epochs)):
        order = fit_idx[torch.randperm(fit_idx.numel(), generator=gen)]
        for start in range(0, order.numel(), int(batch_size)):
            idx = order[start:start + int(batch_size)]
            xb = x_train.index_select(0, idx).to(torch_device)
            yb = y_train.index_select(0, idx).to(torch_device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        calib_logits = model(x_train.index_select(0, calib_idx).to(torch_device))
        calib_y = y_train.index_select(0, calib_idx).to(torch_device)
    temperature = _fit_temperature(calib_logits, calib_y, torch=torch)

    pred_csv = output / "predictions.csv"
    x_query = torch.from_numpy(x_query_np)
    with pred_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        correct = 0
        nll_sum = 0.0
        for start in range(0, x_query.shape[0], int(batch_size)):
            end = min(start + int(batch_size), x_query.shape[0])
            xb = x_query[start:end].to(torch_device)
            with torch.no_grad():
                logits = model(xb)
                probs_raw = torch.softmax(logits, dim=1)
                probs_cal = torch.softmax(logits / temperature, dim=1)
                top_vals, top_idx = torch.topk(logits, k=2, dim=1)
            for offset, row in enumerate(query_rows[start:end]):
                true_label = int(y_query_np[start + offset])
                pred = int(top_idx[offset, 0].item())
                prob_raw = probs_raw[offset].detach().cpu()
                prob_cal = probs_cal[offset].detach().cpu()
                nll = -math.log(max(float(prob_cal[true_label]), 1.0e-12))
                is_correct = int(pred == true_label)
                correct += is_correct
                nll_sum += nll
                writer.writerow(
                    {
                        "sample_id": row.sample_id,
                        "true_label": true_label,
                        "model_id": str(model_id),
                        "predicted_label": pred,
                        "correct": is_correct,
                        "top1_logit": float(top_vals[offset, 0].item()),
                        "top2_logit": float(top_vals[offset, 1].item()),
                        "logit_margin": float((top_vals[offset, 0] - top_vals[offset, 1]).item()),
                        "max_probability_raw": float(prob_raw.max().item()),
                        "entropy_raw": float((-(prob_raw * prob_raw.clamp_min(1.0e-12).log()).sum()).item()),
                        "max_probability_calibrated": float(prob_cal.max().item()),
                        "nll": nll,
                    }
                )

    metadata = {
        "model_id": "dino_linear_probe",
        "train_cache": str(train_dir),
        "query_cache": str(query_dir),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "calibration_fraction": float(calibration_fraction),
        "seed": int(seed),
        "device": str(torch_device),
        "model_id": str(model_id),
        "temperature": float(temperature.item()),
        "query_accuracy": correct / len(query_rows),
        "query_nll": nll_sum / len(query_rows),
        "fit_count": int(fit_idx.numel()),
        "calibration_count": int(calib_idx.numel()),
        "num_classes": int(num_classes),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, output / "model.pt")
    return pred_csv


def _fit_temperature(logits, labels, *, torch):
    log_temperature = torch.nn.Parameter(torch.zeros((), device=logits.device))
    opt = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp_min(1.0e-3)
        loss = torch.nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_temperature.detach().exp().clamp_min(1.0e-3)
