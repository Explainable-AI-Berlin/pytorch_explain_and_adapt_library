from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ada.atlas.data.manifests import ManifestRow


@dataclass(frozen=True)
class AssignedRegion:
    sample_id: str
    class_id: int
    class_name: str
    region_id: str
    k_regions: int
    cluster_index: int
    similarity: float
    distance: float


def l2_normalize(values: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def normalize_vector(value: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / max(norm, eps)


def class_residuals(embeddings: np.ndarray, class_prototype: np.ndarray, *, mode: str) -> np.ndarray:
    x = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    proto = normalize_vector(np.asarray(class_prototype, dtype=np.float32).reshape(-1))
    if mode == "raw":
        return x
    if mode == "centered":
        return l2_normalize(x - proto.reshape(1, -1))
    if mode == "tangent":
        projection = (x @ proto).reshape(-1, 1) * proto.reshape(1, -1)
        return l2_normalize(x - projection)
    raise ValueError(f"unsupported residual mode: {mode!r}")


def assign_rows_to_regions(
    *,
    embeddings: np.ndarray,
    rows: Sequence[ManifestRow],
    region_rows: Sequence[Mapping[str, object]],
    class_prototypes: Mapping[int, np.ndarray],
    residual_prototypes: np.ndarray,
    residual_mode: str,
) -> list[AssignedRegion]:
    """Assign rows to the nearest already-built region from their true class.

    This function never fits or updates prototypes. It is safe for validation
    assignment after train-only region construction.
    """

    if len(rows) != int(np.asarray(embeddings).shape[0]):
        raise ValueError("embeddings and rows have different lengths")
    by_class: dict[int, dict[int, list[tuple[int, Mapping[str, object]]]]] = {}
    for proto_index, region in enumerate(region_rows):
        class_id = int(region["class_id"])
        k_regions = int(region["k_regions"])
        by_class.setdefault(class_id, {}).setdefault(k_regions, []).append((proto_index, region))

    out: list[AssignedRegion] = []
    embeddings_arr = np.asarray(embeddings, dtype=np.float32)
    for class_id, class_regions_by_k in sorted(by_class.items()):
        row_indices = [idx for idx, row in enumerate(rows) if int(row.class_id) == int(class_id)]
        if not row_indices:
            continue
        class_proto = class_prototypes.get(int(class_id))
        if class_proto is None:
            raise KeyError(f"missing class prototype for class_id={class_id}")
        local = class_residuals(embeddings_arr[row_indices], class_proto, mode=residual_mode)
        for _k_regions, class_regions in sorted(class_regions_by_k.items()):
            proto_indices = [idx for idx, _region in class_regions]
            protos = l2_normalize(residual_prototypes[proto_indices])
            sims = local @ protos.T
            best = np.argmax(sims, axis=1)
            for local_pos, row_idx in enumerate(row_indices):
                proto_pos = int(best[local_pos])
                _proto_index, region = class_regions[proto_pos]
                similarity = float(sims[local_pos, proto_pos])
                row = rows[row_idx]
                out.append(
                    AssignedRegion(
                        sample_id=str(row.sample_id),
                        class_id=int(row.class_id),
                        class_name=str(row.class_name),
                        region_id=str(region["region_id"]),
                        k_regions=int(region["k_regions"]),
                        cluster_index=int(region["cluster_index"]),
                        similarity=similarity,
                        distance=float(1.0 - similarity),
                    )
                )
    out.sort(key=lambda item: (item.sample_id, item.k_regions))
    return out
