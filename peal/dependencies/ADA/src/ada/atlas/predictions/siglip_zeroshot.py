from __future__ import annotations

import os

import json
from pathlib import Path

from ada.atlas.data.imagenet_meta import class_display_names
from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.predictions.common import PREDICTION_COLUMNS


def siglip_zero_shot_predictions(
    *,
    query_cache: str | Path,
    output_dir: str | Path,
    model_name: str = "google/siglip2-base-patch16-256",
    imagenet_meta: str | Path | None = os.path.join(os.environ.get("PEAL_DATA", "datasets"), "imagenet_torchvision/data/meta.bin"),
    template: str = "a photo of a {}",
    batch_size: int = 64,
    max_samples: int | None = None,
    device: str = "auto",
) -> Path:
    import csv
    import math
    import torch
    from PIL import Image
    from transformers import AutoProcessor, SiglipModel

    query_dir = Path(query_cache)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_csv(query_dir / "manifest.csv")
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    root = Path(json.loads((query_dir / "metadata.json").read_text()).get("root", ""))
    class_names_by_id = [None] * (max(row.class_id for row in rows) + 1)
    for row in rows:
        class_names_by_id[row.class_id] = row.class_name
    class_names = [str(name) for name in class_names_by_id]
    display_names = class_display_names(class_names, imagenet_meta)
    prompts = [template.format(name) for name in display_names]

    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model = SiglipModel.from_pretrained(model_name, local_files_only=True).to(torch_device).eval()

    text_inputs = processor(text=prompts, padding="max_length", return_tensors="pt")
    text_inputs = {key: value.to(torch_device) for key, value in text_inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True).clamp_min(1.0e-12)

    pred_csv = output / "predictions.csv"
    correct = 0
    nll_sum = 0.0
    with pred_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            images = [Image.open(root / row.relative_path).convert("RGB") for row in chunk]
            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {key: value.to(torch_device) for key, value in image_inputs.items()}
            with torch.no_grad():
                image_features = model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
                logits = image_features @ text_features.T
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
                        "model_id": "siglip2_zeroshot_photo",
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
        "model_id": "siglip2_zeroshot_photo",
        "query_cache": str(query_dir),
        "model_name": model_name,
        "imagenet_meta": None if imagenet_meta is None else str(imagenet_meta),
        "template": template,
        "batch_size": int(batch_size),
        "max_samples": max_samples,
        "device": str(torch_device),
        "query_accuracy": correct / len(rows),
        "query_nll": nll_sum / len(rows),
        "class_names": class_names,
        "display_names": display_names,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return pred_csv
