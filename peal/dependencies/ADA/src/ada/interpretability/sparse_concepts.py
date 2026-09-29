from __future__ import annotations

import numpy as np


def sparse_positive_decomposition(
    image_embeddings: np.ndarray,
    concept_embeddings: np.ndarray,
    *,
    max_terms: int = 8,
    min_score: float = 0.0,
) -> np.ndarray:
    """Greedy nonnegative concept coefficients for normalized embeddings.

    This is a dependency-light approximation to nonnegative sparse coding. It is
    intended for concept ranking, not reconstruction-quality claims.
    """
    images = _l2_normalize(np.asarray(image_embeddings, dtype=np.float32))
    concepts = _l2_normalize(np.asarray(concept_embeddings, dtype=np.float32))
    coeffs = np.zeros((images.shape[0], concepts.shape[0]), dtype=np.float32)
    for row_idx, image in enumerate(images):
        residual = image.copy()
        used: set[int] = set()
        for _ in range(int(max_terms)):
            scores = concepts @ residual
            if used:
                scores[list(used)] = -np.inf
            concept_idx = int(np.argmax(scores))
            score = float(scores[concept_idx])
            if score <= float(min_score):
                break
            coeffs[row_idx, concept_idx] = score
            used.add(concept_idx)
            residual = residual - score * concepts[concept_idx]
            if float(np.linalg.norm(residual)) <= 1.0e-12:
                break
    return coeffs


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1.0e-12, None)
