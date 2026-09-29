from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ada.atlas.support.knn import KNNResult, Vector, exact_cosine_knn


@dataclass(frozen=True)
class ClassConditionalSupport:
    labels: list[int]
    kth_distance: list[float | None]
    mean_distance: list[float | None]
    neighbours: list[KNNResult | None]


def exact_class_conditional_support(
    query: Sequence[Vector],
    query_labels: Sequence[int],
    reference: Sequence[Vector],
    reference_labels: Sequence[int],
    *,
    k: int,
    query_ids: Sequence[str] | None = None,
    reference_ids: Sequence[str] | None = None,
    leave_one_out: bool = False,
) -> ClassConditionalSupport:
    if len(query) != len(query_labels):
        raise ValueError("query_labels length does not match query rows")
    if len(reference) != len(reference_labels):
        raise ValueError("reference_labels length does not match reference rows")

    kth: list[float | None] = []
    mean: list[float | None] = []
    neighbours: list[KNNResult | None] = []
    labels = [int(x) for x in query_labels]

    for qi, label in enumerate(labels):
        ref_indices = [idx for idx, ref_label in enumerate(reference_labels) if int(ref_label) == label]
        if not ref_indices:
            kth.append(None)
            mean.append(None)
            neighbours.append(None)
            continue

        ref_rows = [reference[idx] for idx in ref_indices]
        ref_ids = None if reference_ids is None else [reference_ids[idx] for idx in ref_indices]
        q_ids = None if query_ids is None else [query_ids[qi]]
        try:
            result = exact_cosine_knn(
                [query[qi]],
                ref_rows,
                k=k,
                query_ids=q_ids,
                reference_ids=ref_ids,
                leave_one_out=leave_one_out,
            )
        except ValueError:
            kth.append(None)
            mean.append(None)
            neighbours.append(None)
            continue
        kth.append(result.kth_distance[0])
        mean.append(result.mean_distance[0])
        neighbours.append(result)

    return ClassConditionalSupport(labels=labels, kth_distance=kth, mean_distance=mean, neighbours=neighbours)
