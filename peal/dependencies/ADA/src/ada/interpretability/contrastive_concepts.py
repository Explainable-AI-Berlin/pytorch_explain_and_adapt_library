from __future__ import annotations

from typing import Sequence

import numpy as np


def concept_scores(
    region_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
    *,
    control_embeddings: np.ndarray | None = None,
    failure_embeddings: np.ndarray | None = None,
    correct_region_embeddings: np.ndarray | None = None,
    deleted_embeddings: np.ndarray | None = None,
    retained_embeddings: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    phrases = np.asarray(phrase_embeddings, dtype=np.float32)
    region = np.asarray(region_embeddings, dtype=np.float32)
    absolute = region.mean(axis=0) @ phrases.T
    out = {"absolute_score": absolute.astype(np.float32)}
    if control_embeddings is not None and len(control_embeddings):
        control = np.asarray(control_embeddings, dtype=np.float32)
        out["region_contrast_score"] = (region.mean(axis=0) - control.mean(axis=0)) @ phrases.T
        out["region_contrast_standardized_effect"] = standardized_phrase_effect(region, control, phrases)
    if failure_embeddings is not None and correct_region_embeddings is not None and len(failure_embeddings) and len(correct_region_embeddings):
        fail = np.asarray(failure_embeddings, dtype=np.float32)
        correct = np.asarray(correct_region_embeddings, dtype=np.float32)
        out["failure_contrast_score"] = (fail.mean(axis=0) - correct.mean(axis=0)) @ phrases.T
        out["failure_contrast_standardized_effect"] = standardized_phrase_effect(fail, correct, phrases)
    if deleted_embeddings is not None and retained_embeddings is not None and len(deleted_embeddings) and len(retained_embeddings):
        deleted = np.asarray(deleted_embeddings, dtype=np.float32)
        retained = np.asarray(retained_embeddings, dtype=np.float32)
        out["deleted_vs_retained_score"] = (deleted.mean(axis=0) - retained.mean(axis=0)) @ phrases.T
        out["deleted_vs_retained_standardized_effect"] = standardized_phrase_effect(deleted, retained, phrases)
    return {key: np.asarray(value, dtype=np.float32) for key, value in out.items()}


def standardized_phrase_effect(group_a_embeddings: np.ndarray, group_b_embeddings: np.ndarray, phrase_embeddings: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    a_scores = np.asarray(group_a_embeddings, dtype=np.float32) @ np.asarray(phrase_embeddings, dtype=np.float32).T
    b_scores = np.asarray(group_b_embeddings, dtype=np.float32) @ np.asarray(phrase_embeddings, dtype=np.float32).T
    pooled = np.sqrt(0.5 * (a_scores.var(axis=0) + b_scores.var(axis=0)) + float(eps))
    return ((a_scores.mean(axis=0) - b_scores.mean(axis=0)) / pooled).astype(np.float32)


def top_phrase_indices(scores: np.ndarray, *, top_k: int) -> list[int]:
    if scores.size == 0:
        return []
    k_eff = min(int(top_k), int(scores.shape[0]))
    order = np.argsort(-np.asarray(scores, dtype=np.float32))[:k_eff]
    return [int(idx) for idx in order.tolist()]


def bootstrap_top_frequency(
    region_embeddings: np.ndarray,
    control_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
    *,
    top_k: int = 20,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> np.ndarray:
    if len(region_embeddings) == 0 or len(control_embeddings) == 0 or n_bootstrap <= 0:
        return np.zeros(int(phrase_embeddings.shape[0]), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    region = np.asarray(region_embeddings, dtype=np.float32)
    control = np.asarray(control_embeddings, dtype=np.float32)
    phrases = np.asarray(phrase_embeddings, dtype=np.float32)
    counts = np.zeros(int(phrases.shape[0]), dtype=np.float32)
    for _ in range(int(n_bootstrap)):
        ridx = rng.integers(0, region.shape[0], size=region.shape[0])
        cidx = rng.integers(0, control.shape[0], size=control.shape[0])
        scores = (region[ridx].mean(axis=0) - control[cidx].mean(axis=0)) @ phrases.T
        for phrase_idx in top_phrase_indices(scores, top_k=int(top_k)):
            counts[phrase_idx] += 1.0
    return counts / float(n_bootstrap)


def bootstrap_contrast_stability(
    group_a_embeddings: np.ndarray,
    group_b_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
    *,
    top_k: int = 20,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    if len(group_a_embeddings) == 0 or len(group_b_embeddings) == 0 or n_bootstrap <= 0:
        n_phrases = int(phrase_embeddings.shape[0])
        return {
            "top_frequency": np.zeros(n_phrases, dtype=np.float32),
            "sign_consistency": np.zeros(n_phrases, dtype=np.float32),
            "effect_mean": np.full(n_phrases, np.nan, dtype=np.float32),
            "effect_ci_low": np.full(n_phrases, np.nan, dtype=np.float32),
            "effect_ci_high": np.full(n_phrases, np.nan, dtype=np.float32),
        }
    rng = np.random.default_rng(int(seed))
    a = np.asarray(group_a_embeddings, dtype=np.float32)
    b = np.asarray(group_b_embeddings, dtype=np.float32)
    phrases = np.asarray(phrase_embeddings, dtype=np.float32)
    all_scores = np.zeros((int(n_bootstrap), int(phrases.shape[0])), dtype=np.float32)
    top_counts = np.zeros(int(phrases.shape[0]), dtype=np.float32)
    for boot_idx in range(int(n_bootstrap)):
        aidx = rng.integers(0, a.shape[0], size=a.shape[0])
        bidx = rng.integers(0, b.shape[0], size=b.shape[0])
        scores = (a[aidx].mean(axis=0) - b[bidx].mean(axis=0)) @ phrases.T
        all_scores[boot_idx] = scores.astype(np.float32)
        for phrase_idx in top_phrase_indices(scores, top_k=int(top_k)):
            top_counts[phrase_idx] += 1.0
    sign_consistency = np.maximum((all_scores >= 0.0).mean(axis=0), (all_scores <= 0.0).mean(axis=0))
    return {
        "top_frequency": top_counts / float(n_bootstrap),
        "sign_consistency": sign_consistency.astype(np.float32),
        "effect_mean": all_scores.mean(axis=0).astype(np.float32),
        "effect_ci_low": np.quantile(all_scores, 0.025, axis=0).astype(np.float32),
        "effect_ci_high": np.quantile(all_scores, 0.975, axis=0).astype(np.float32),
    }


def permutation_null_p_values(
    group_a_embeddings: np.ndarray,
    group_b_embeddings: np.ndarray,
    phrase_embeddings: np.ndarray,
    *,
    n_permutations: int = 100,
    seed: int = 0,
) -> np.ndarray:
    if len(group_a_embeddings) == 0 or len(group_b_embeddings) == 0 or n_permutations <= 0:
        return np.full(int(phrase_embeddings.shape[0]), np.nan, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    a = np.asarray(group_a_embeddings, dtype=np.float32)
    b = np.asarray(group_b_embeddings, dtype=np.float32)
    phrases = np.asarray(phrase_embeddings, dtype=np.float32)
    observed = np.abs((a.mean(axis=0) - b.mean(axis=0)) @ phrases.T)
    pooled = np.concatenate([a, b], axis=0)
    count = np.ones(int(phrases.shape[0]), dtype=np.float32)
    for _ in range(int(n_permutations)):
        perm = rng.permutation(pooled.shape[0])
        pa = pooled[perm[: a.shape[0]]]
        pb = pooled[perm[a.shape[0] :]]
        null_score = np.abs((pa.mean(axis=0) - pb.mean(axis=0)) @ phrases.T)
        count += (null_score >= observed).astype(np.float32)
    return (count / float(int(n_permutations) + 1)).astype(np.float32)


def format_top_phrases(
    phrase_rows: Sequence[dict[str, str]],
    scores: np.ndarray,
    *,
    top_k: int,
    stability: np.ndarray | None = None,
    min_stability: float | None = None,
) -> str:
    items: list[str] = []
    for idx in top_phrase_indices(scores, top_k=max(int(top_k) * 3, int(top_k))):
        if min_stability is not None and stability is not None and float(stability[idx]) < float(min_stability):
            continue
        phrase = str(phrase_rows[idx]["phrase"])
        if stability is None:
            items.append(f"{phrase}:{float(scores[idx]):.4f}")
        else:
            items.append(f"{phrase}:{float(scores[idx]):.4f}:{float(stability[idx]):.2f}")
        if len(items) >= int(top_k):
            break
    return ";".join(items)
