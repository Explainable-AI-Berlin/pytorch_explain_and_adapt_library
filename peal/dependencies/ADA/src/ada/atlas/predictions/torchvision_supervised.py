from __future__ import annotations

import os

import json
from pathlib import Path

from ada.atlas.data.imagenet_meta import load_imagenet_wnid_to_index
from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.predictions.common import PREDICTION_COLUMNS


def torchvision_imagenet_predictions(
    *,
    query_cache: str | Path,
    output_dir: str | Path,
    model_name: str = "resnet18",
    imagenet_meta: str | Path = os.path.join(os.environ.get("PEAL_DATA", "datasets"), "imagenet_torchvision/data/meta.bin"),
    batch_size: int = 128,
    max_samples: int | None = None,
    device: str = "auto",
) -> Path:
    import csv
    import math
    import torch
    from PIL import Image
    import torchvision.models as models

    query_dir = Path(query_cache)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_csv(query_dir / "manifest.csv")
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    root = Path(json.loads((query_dir / "metadata.json").read_text()).get("root", ""))
    local_class_names = [None] * (max(row.class_id for row in rows) + 1)
    for row in rows:
        local_class_names[row.class_id] = row.class_name
    wnid_to_index = load_imagenet_wnid_to_index(imagenet_meta)
    restricted_indices = [wnid_to_index[str(wnid)] for wnid in local_class_names]
    model_id = f"{model_name}_imagenet1k_restricted{len(local_class_names)}"

    model_name = model_name.lower()
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    elif model_name == "alexnet":
        weights = models.AlexNet_Weights.IMAGENET1K_V1
        model = models.alexnet(weights=weights)
    elif model_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    else:
        raise ValueError(f"unsupported torchvision model: {model_name}")

    preprocess = weights.transforms()
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    model = model.to(torch_device).eval()
    index_tensor = torch.as_tensor(restricted_indices, dtype=torch.long, device=torch_device)

    pred_csv = output / "predictions.csv"
    correct = 0
    nll_sum = 0.0
    with pred_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            images = [preprocess(Image.open(root / row.relative_path).convert("RGB")) for row in chunk]
            xb = torch.stack(images, dim=0).to(torch_device)
            with torch.no_grad():
                logits_1000 = model(xb)
                logits = logits_1000.index_select(1, index_tensor)
                probs = torch.softmax(logits, dim=1)
                top_vals, top_idx = torch.topk(logits, k=2, dim=1)
            for offset, row in enumerate(chunk):
                true_label = int(row.class_id)
                pred = int(top_idx[offset, 0].item())
                prob = probs[offset].detach().cpu()
                nll = -math.log(max(float(prob[true_label]), 1.0e-12))
                is_correct = int(pred == true_label)
                correct += is_correct
                nll_sum += nll
                writer.writerow(
                    {
                        "sample_id": row.sample_id,
                        "true_label": true_label,
                        "model_id": model_id,
                        "predicted_label": pred,
                        "correct": is_correct,
                        "top1_logit": float(top_vals[offset, 0].item()),
                        "top2_logit": float(top_vals[offset, 1].item()),
                        "logit_margin": float((top_vals[offset, 0] - top_vals[offset, 1]).item()),
                        "max_probability_raw": float(prob.max().item()),
                        "entropy_raw": float((-(prob * prob.clamp_min(1.0e-12).log()).sum()).item()),
                        "max_probability_calibrated": "",
                        "nll": nll,
                    }
                )

    metadata = {
        "model_id": model_id,
        "query_cache": str(query_dir),
        "model_name": model_name,
        "weights": str(weights),
        "imagenet_meta": str(imagenet_meta),
        "batch_size": int(batch_size),
        "max_samples": max_samples,
        "device": str(torch_device),
        "query_accuracy": correct / len(rows),
        "query_nll": nll_sum / len(rows),
        "restricted_indices": restricted_indices,
        "local_class_names": [str(name) for name in local_class_names],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return pred_csv
