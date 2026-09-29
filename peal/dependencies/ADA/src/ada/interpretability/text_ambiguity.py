from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def class_prompt_matrix(
    phrase_rows: Sequence[Mapping[str, str]],
    phrase_embeddings: np.ndarray,
) -> tuple[np.ndarray, list[int], list[str]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    names: dict[int, str] = {}
    for idx, row in enumerate(phrase_rows):
        if str(row.get("phrase_type", "")) != "class_prompt":
            continue
        raw_class = row.get("class_id", "")
        if raw_class == "":
            continue
        class_id = int(raw_class)
        by_class[class_id].append(idx)
        names[class_id] = str(row.get("display_name") or row.get("class_name") or class_id)
    class_ids = sorted(by_class)
    if not class_ids:
        raise ValueError("phrase bank contains no class_prompt rows")
    vectors = []
    display_names = []
    for class_id in class_ids:
        emb = np.asarray(phrase_embeddings[by_class[class_id]], dtype=np.float32)
        mean = emb.mean(axis=0)
        norm = np.linalg.norm(mean)
        vectors.append(mean / max(float(norm), 1.0e-12))
        display_names.append(names[class_id])
    return np.stack(vectors, axis=0).astype(np.float32), class_ids, display_names


def text_class_metrics(
    image_embeddings: np.ndarray,
    labels: Sequence[int],
    class_embeddings: np.ndarray,
    class_ids: Sequence[int],
    *,
    temperature: float | None = None,
    logit_scale: float | None = None,
) -> dict[str, object]:
    if len(labels) != int(image_embeddings.shape[0]):
        raise ValueError("labels length must match image embeddings")
    class_to_col = {int(class_id): idx for idx, class_id in enumerate(class_ids)}
    logits = np.asarray(image_embeddings, dtype=np.float32) @ np.asarray(class_embeddings, dtype=np.float32).T
    scale = float(logit_scale) if logit_scale is not None else (1.0 / max(float(temperature or 0.05), 1.0e-6))
    scaled = logits * scale
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1.0e-12)
    true_cols = np.asarray([class_to_col[int(label)] for label in labels], dtype=np.int64)
    true_scores = logits[np.arange(logits.shape[0]), true_cols]
    masked = logits.copy()
    masked[np.arange(masked.shape[0]), true_cols] = -np.inf
    competing_cols = masked.argmax(axis=1)
    margins = true_scores - masked[np.arange(masked.shape[0]), competing_cols]
    order = np.argsort(-logits, axis=1)
    true_ranks = np.asarray([int(np.where(order[row_idx] == true_cols[row_idx])[0][0]) + 1 for row_idx in range(logits.shape[0])], dtype=np.int64)
    top_scores = logits[np.arange(logits.shape[0]), order[:, 0]]
    second_scores = logits[np.arange(logits.shape[0]), order[:, 1]] if logits.shape[1] > 1 else np.full(logits.shape[0], -np.inf, dtype=np.float32)
    entropy = -(probs * np.log(np.clip(probs, 1.0e-12, 1.0))).sum(axis=1)
    return {
        "true_class_text_similarity": true_scores.astype(np.float32),
        "true_class_text_rank": true_ranks,
        "text_class_margin": margins.astype(np.float32),
        "top_two_text_margin": (top_scores - second_scores).astype(np.float32),
        "text_class_entropy": entropy.astype(np.float32),
        "text_true_class_probability": probs[np.arange(probs.shape[0]), true_cols].astype(np.float32),
        "top_text_class_id": [int(class_ids[idx]) for idx in order[:, 0].tolist()],
        "top_text_class_score": top_scores.astype(np.float32),
        "text_competing_class_id": [int(class_ids[idx]) for idx in competing_cols.tolist()],
        "text_competing_score": masked[np.arange(masked.shape[0]), competing_cols].astype(np.float32),
    }


def calibrate_temperature(
    image_embeddings: np.ndarray,
    labels: Sequence[int],
    class_embeddings: np.ndarray,
    class_ids: Sequence[int],
    *,
    candidates: Sequence[float] = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0),
) -> float:
    if len(labels) == 0:
        return 0.05
    class_to_col = {int(class_id): idx for idx, class_id in enumerate(class_ids)}
    true_cols = np.asarray([class_to_col[int(label)] for label in labels], dtype=np.int64)
    logits = np.asarray(image_embeddings, dtype=np.float32) @ np.asarray(class_embeddings, dtype=np.float32).T
    best_temp = float(candidates[0])
    best_loss = float("inf")
    for temp in candidates:
        scaled = logits / max(float(temp), 1.0e-6)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1.0e-12)
        loss = float(-np.log(np.clip(probs[np.arange(probs.shape[0]), true_cols], 1.0e-12, 1.0)).mean())
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp


def within_class_standardize(values: Sequence[float], labels: Sequence[int], *, eps: float = 1.0e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    out = np.zeros_like(arr, dtype=np.float32)
    for label in sorted(set(labels_arr.tolist())):
        mask = labels_arr == int(label)
        if not mask.any():
            continue
        mean = float(arr[mask].mean())
        std = float(arr[mask].std())
        out[mask] = (arr[mask] - mean) / max(std, float(eps))
    return out
