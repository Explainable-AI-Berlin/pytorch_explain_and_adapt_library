from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from ada.atlas.data.manifests import ManifestRow, load_manifest_csv


def score_cache_knn(
    *,
    query_cache: str | Path,
    reference_cache: str | Path,
    output_csv: str | Path,
    k_values: Sequence[int],
    batch_size: int = 256,
    device: str = "auto",
    leave_one_out: bool = False,
    class_conditional: bool = False,
    mmap: bool = False,
    reference_shard_size: int | None = None,
) -> Path:
    if not k_values:
        raise ValueError("k_values must not be empty")
    k_values = tuple(sorted({int(k) for k in k_values}))
    if k_values[0] < 1:
        raise ValueError("k values must be positive")
    max_k = k_values[-1]

    import numpy as np
    import torch

    query_dir = Path(query_cache)
    reference_dir = Path(reference_cache)
    mmap_mode = "r" if mmap else None
    query_embeddings = np.load(query_dir / "embeddings.npy", mmap_mode=mmap_mode).astype("float32", copy=False)
    reference_embeddings = np.load(reference_dir / "embeddings.npy", mmap_mode=mmap_mode).astype("float32", copy=False)
    query_rows = load_manifest_csv(query_dir / "manifest.csv")
    reference_rows = load_manifest_csv(reference_dir / "manifest.csv")
    if query_embeddings.shape[0] != len(query_rows):
        raise ValueError("query embeddings and manifest row counts differ")
    if reference_embeddings.shape[0] != len(reference_rows):
        raise ValueError("reference embeddings and manifest row counts differ")
    if query_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError("query/reference embedding dimensions differ")
    if reference_embeddings.shape[0] < max_k:
        raise ValueError("reference cache has fewer rows than max k")

    torch_device = _resolve_device(device, torch)
    shard_size = int(reference_shard_size or 0)
    use_sharded_reference = shard_size > 0 and shard_size < int(reference_embeddings.shape[0])
    ref = None
    if not use_sharded_reference:
        ref = _torch_from_numpy(reference_embeddings, torch=torch, copy_readonly=False).to(torch_device)
        ref = _l2_normalize_tensor(ref)
    query_ids = [row.sample_id for row in query_rows]
    reference_ids = [row.sample_id for row in reference_rows]
    reference_position = {sample_id: idx for idx, sample_id in enumerate(reference_ids)}
    class_indices = _build_class_indices(reference_rows) if class_conditional else {}
    class_index_tensors = {
        label: torch.tensor(indices, device=torch_device, dtype=torch.long)
        for label, indices in class_indices.items()
    } if class_conditional and ref is not None else {}

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "class_id", "class_name"]
    for k in k_values:
        columns.extend([f"support_k{k}_kth_distance", f"support_k{k}_mean_distance"])
    columns.extend(["nearest_sample_id", "nearest_class_id", "nearest_distance"])
    if class_conditional:
        for k in k_values:
            columns.extend([f"class_support_k{k}_kth_distance", f"class_support_k{k}_mean_distance"])

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for start in range(0, int(query_embeddings.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(query_embeddings.shape[0]))
            q = _torch_from_numpy(query_embeddings[start:end], torch=torch, copy_readonly=True).to(torch_device)
            q = _l2_normalize_tensor(q)
            if use_sharded_reference:
                values, indices = _topk_sharded_exact(
                    q,
                    reference_embeddings,
                    k=max_k,
                    shard_size=shard_size,
                    query_ids=query_ids[start:end],
                    reference_position=reference_position,
                    leave_one_out=leave_one_out,
                    torch=torch,
                )
            else:
                if ref is None:
                    raise RuntimeError("reference tensor was not initialized")
                sims = q @ ref.T
                if leave_one_out:
                    _mask_self_matches(sims, query_ids[start:end], reference_position)
                values, indices = torch.topk(sims, k=max_k, dim=1, largest=True, sorted=True)
            values_cpu = values.detach().cpu().numpy()
            indices_cpu = indices.detach().cpu().numpy()
            class_conditional_rows = (
                _score_batch_class_conditional(
                    q,
                    query_rows[start:end],
                    reference_embeddings=reference_embeddings,
                    reference_rows=reference_rows,
                    ref=ref,
                    class_indices=class_indices,
                    class_index_tensors=class_index_tensors,
                    reference_position=reference_position,
                    k_values=k_values,
                    leave_one_out=leave_one_out,
                    torch=torch,
                )
                if class_conditional
                else None
            )
            for offset, row in enumerate(query_rows[start:end]):
                distances = 1.0 - values_cpu[offset]
                out = {
                    "sample_id": row.sample_id,
                    "class_id": row.class_id,
                    "class_name": row.class_name,
                    "nearest_sample_id": reference_rows[int(indices_cpu[offset, 0])].sample_id,
                    "nearest_class_id": reference_rows[int(indices_cpu[offset, 0])].class_id,
                    "nearest_distance": float(distances[0]),
                }
                for k in k_values:
                    kd = distances[:k]
                    out[f"support_k{k}_kth_distance"] = float(kd[-1])
                    out[f"support_k{k}_mean_distance"] = float(kd.mean())
                if class_conditional:
                    if class_conditional_rows is None:
                        raise RuntimeError("class-conditional rows were not computed")
                    out.update(class_conditional_rows[offset])
                writer.writerow(out)

    return output


def _l2_normalize_tensor(values):
    return values / values.norm(dim=1, keepdim=True).clamp_min(1.0e-12)


def _torch_from_numpy(values, *, torch, copy_readonly: bool):
    if copy_readonly and hasattr(values, "flags") and not values.flags.writeable:
        import numpy as np

        values = np.array(values, copy=True)
    return torch.from_numpy(values)


def _resolve_device(device: str, torch_module):
    if device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device)


def _mask_self_matches(sims, query_ids: Sequence[str], reference_position: dict[str, int]) -> None:
    for row_idx, sample_id in enumerate(query_ids):
        ref_idx = reference_position.get(sample_id)
        if ref_idx is not None:
            sims[row_idx, ref_idx] = -math.inf


def _topk_sharded_exact(
    q,
    reference_embeddings,
    *,
    k: int,
    shard_size: int,
    query_ids: Sequence[str],
    reference_position: dict[str, int],
    leave_one_out: bool,
    torch,
):
    top_values = None
    top_indices = None
    n_reference = int(reference_embeddings.shape[0])
    for ref_start in range(0, n_reference, int(shard_size)):
        ref_end = min(ref_start + int(shard_size), n_reference)
        ref_shard = _torch_from_numpy(
            reference_embeddings[ref_start:ref_end],
            torch=torch,
            copy_readonly=True,
        ).to(q.device)
        ref_shard = _l2_normalize_tensor(ref_shard)
        sims = q @ ref_shard.T
        if leave_one_out:
            for row_idx, sample_id in enumerate(query_ids):
                ref_idx = reference_position.get(sample_id)
                if ref_idx is not None and ref_start <= ref_idx < ref_end:
                    sims[row_idx, ref_idx - ref_start] = -math.inf
        local_k = min(int(k), int(ref_end - ref_start))
        local_values, local_indices = torch.topk(sims, k=local_k, dim=1, largest=True, sorted=True)
        local_indices = local_indices + int(ref_start)
        if top_values is None:
            top_values = local_values
            top_indices = local_indices
            continue
        merged_values = torch.cat([top_values, local_values], dim=1)
        merged_indices = torch.cat([top_indices, local_indices], dim=1)
        top_values, merge_positions = torch.topk(merged_values, k=int(k), dim=1, largest=True, sorted=True)
        top_indices = torch.gather(merged_indices, 1, merge_positions)
    if top_values is None or top_indices is None:
        raise ValueError("reference cache has no rows")
    return top_values, top_indices


def _build_class_indices(reference_rows: Sequence[ManifestRow]) -> dict[int, list[int]]:
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(reference_rows):
        class_indices[int(row.class_id)].append(int(idx))
    return dict(class_indices)


def _blank_class_scores(k_values: Sequence[int]) -> dict[str, object]:
    out = {}
    for k in k_values:
        out[f"class_support_k{k}_kth_distance"] = ""
        out[f"class_support_k{k}_mean_distance"] = ""
    return out


def _score_batch_class_conditional(
    q,
    query_rows: Sequence[ManifestRow],
    *,
    reference_embeddings,
    reference_rows: Sequence[ManifestRow],
    ref,
    class_indices: dict[int, list[int]],
    class_index_tensors: dict[int, object],
    reference_position: dict[str, int],
    k_values: Sequence[int],
    leave_one_out: bool,
    torch,
) -> list[dict[str, object]]:
    out_rows = [_blank_class_scores(k_values) for _ in query_rows]
    query_offsets_by_class: dict[int, list[int]] = defaultdict(list)
    for offset, row in enumerate(query_rows):
        query_offsets_by_class[int(row.class_id)].append(int(offset))

    for label, query_offsets in query_offsets_by_class.items():
        indices = class_indices.get(int(label), [])
        if not indices:
            continue
        if ref is not None:
            index_tensor = class_index_tensors[int(label)]
            class_ref = ref.index_select(0, index_tensor)
        else:
            class_ref = _torch_from_numpy(reference_embeddings[indices], torch=torch, copy_readonly=True).to(q.device)
            class_ref = _l2_normalize_tensor(class_ref)
        offset_tensor = torch.tensor(query_offsets, device=q.device, dtype=torch.long)
        q_class = q.index_select(0, offset_tensor)
        sims = q_class @ class_ref.T
        if leave_one_out:
            local_position = {int(global_idx): pos for pos, global_idx in enumerate(indices)}
            for local_row_idx, query_offset in enumerate(query_offsets):
                ref_idx = reference_position.get(query_rows[query_offset].sample_id)
                if ref_idx is not None and ref_idx in local_position:
                    sims[local_row_idx, local_position[ref_idx]] = -math.inf
        available_k = min(max(k_values), int(class_ref.shape[0]))
        values, _indices = torch.topk(sims, k=available_k, dim=1, largest=True, sorted=True)
        distances_cpu = 1.0 - values.detach().cpu().numpy()
        for local_row_idx, query_offset in enumerate(query_offsets):
            finite_distances = [float(distance) for distance in distances_cpu[local_row_idx] if math.isfinite(float(distance))]
            row_scores = {}
            for k in k_values:
                if len(finite_distances) < int(k):
                    row_scores[f"class_support_k{k}_kth_distance"] = ""
                    row_scores[f"class_support_k{k}_mean_distance"] = ""
                else:
                    kd = finite_distances[: int(k)]
                    row_scores[f"class_support_k{k}_kth_distance"] = float(kd[-1])
                    row_scores[f"class_support_k{k}_mean_distance"] = float(sum(kd) / len(kd))
            out_rows[query_offset] = row_scores
    return out_rows
