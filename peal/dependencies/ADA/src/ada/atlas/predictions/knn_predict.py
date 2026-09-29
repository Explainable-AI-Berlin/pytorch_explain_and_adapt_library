from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Sequence

from ada.atlas.data.manifests import load_manifest_csv
from ada.atlas.predictions.common import entropy_from_probs, write_prediction_rows


def predict_knn_from_caches(
    *,
    query_cache: str | Path,
    reference_cache: str | Path,
    output_csv: str | Path,
    k_values: Sequence[int] = (1, 5, 10),
    batch_size: int = 256,
    device: str = "auto",
) -> Path:
    import numpy as np
    import torch

    query_dir = Path(query_cache)
    reference_dir = Path(reference_cache)
    query_embeddings = np.load(query_dir / "embeddings.npy").astype("float32", copy=False)
    reference_embeddings = np.load(reference_dir / "embeddings.npy").astype("float32", copy=False)
    query_rows = load_manifest_csv(query_dir / "manifest.csv")
    reference_rows = load_manifest_csv(reference_dir / "manifest.csv")
    if query_embeddings.shape[0] != len(query_rows):
        raise ValueError("query embeddings and manifest row counts differ")
    if reference_embeddings.shape[0] != len(reference_rows):
        raise ValueError("reference embeddings and manifest row counts differ")

    k_values = tuple(sorted({int(k) for k in k_values}))
    max_k = max(k_values)
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    ref = torch.from_numpy(reference_embeddings).to(torch_device)
    ref = ref / ref.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    query = torch.from_numpy(query_embeddings)
    ref_labels = [int(row.class_id) for row in reference_rows]

    out_rows = []
    num_classes = max(max(ref_labels), max(int(row.class_id) for row in query_rows)) + 1
    for start in range(0, query.shape[0], int(batch_size)):
        end = min(start + int(batch_size), query.shape[0])
        q = query[start:end].to(torch_device)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        sims = q @ ref.T
        values, indices = torch.topk(sims, k=max_k, dim=1, largest=True, sorted=True)
        values_cpu = values.detach().cpu().numpy()
        indices_cpu = indices.detach().cpu().numpy()
        for offset, row in enumerate(query_rows[start:end]):
            neighbour_labels = [ref_labels[int(idx)] for idx in indices_cpu[offset]]
            neighbour_sims = [float(x) for x in values_cpu[offset]]
            for k in k_values:
                pred, score_by_class = _majority_prediction(neighbour_labels[:k], neighbour_sims[:k], num_classes)
                scores = [score_by_class.get(cls, 0.0) for cls in range(num_classes)]
                total = sum(scores) or 1.0
                probs = [score / total for score in scores]
                top_scores = sorted(scores, reverse=True)
                top1 = top_scores[0]
                top2 = top_scores[1] if len(top_scores) > 1 else 0.0
                true_label = int(row.class_id)
                out_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "true_label": true_label,
                        "model_id": f"dino_knn_k{k}",
                        "predicted_label": pred,
                        "correct": int(pred == true_label),
                        "top1_logit": top1,
                        "top2_logit": top2,
                        "logit_margin": top1 - top2,
                        "max_probability_raw": max(probs),
                        "entropy_raw": entropy_from_probs(probs),
                        "max_probability_calibrated": "",
                        "nll": -1.0 if probs[true_label] <= 0 else -math.log(probs[true_label]),
                    }
                )
    return write_prediction_rows(out_rows, output_csv)


def _majority_prediction(labels: Sequence[int], sims: Sequence[float], num_classes: int) -> tuple[int, dict[int, float]]:
    if len(labels) == 1:
        return int(labels[0]), {int(labels[0]): 1.0}
    counts = Counter(int(label) for label in labels)
    max_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == max_count}
    weights: dict[int, float] = {cls: 0.0 for cls in range(num_classes)}
    for label, sim in zip(labels, sims):
        weights[int(label)] = weights.get(int(label), 0.0) + max(float(sim), 0.0)
    pred = max(tied, key=lambda label: (weights.get(label, 0.0), -label))
    return int(pred), weights
