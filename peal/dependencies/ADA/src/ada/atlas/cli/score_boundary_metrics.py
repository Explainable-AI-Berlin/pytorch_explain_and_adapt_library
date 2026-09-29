from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from ada.atlas.data.manifests import ManifestRow, load_manifest_csv
from ada.atlas.metrics.error_detection import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute support, competing-class margin, local label entropy, and "
            "optional prediction/error summaries for CLS atlas queries."
        )
    )
    parser.add_argument("--query-cache", required=True, type=Path)
    parser.add_argument("--reference-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", default="1,5,10,50")
    parser.add_argument("--entropy-k", default="10,50")
    parser.add_argument("--global-search-k", default=512, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--reference-shard-size", default=100000, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument(
        "--class-key",
        choices=["class_id", "class_name"],
        default="class_name",
        help="Use class_name for cross-dataset scoring such as ImageNet-R vs ImageNet-1K.",
    )
    parser.add_argument(
        "--skip-reference-percentiles",
        action="store_true",
        help="Skip within-reference same-class percentile calibration.",
    )
    parser.add_argument(
        "--reference-percentile-cache",
        default=None,
        type=Path,
        help="Optional .npz cache for reference leave-one-out class-support percentile distributions.",
    )
    parser.add_argument("--prediction-csv", action="append", type=Path, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = _parse_ints(args.k)
    entropy_k_values = _parse_ints(args.entropy_k)
    if 1 not in k_values:
        k_values = [1, *k_values]
    k_values = sorted(set(k_values))
    entropy_k_values = sorted(set(entropy_k_values))
    search_k = max([int(args.global_search_k), *k_values, *entropy_k_values])

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_csv = output / "boundary_metrics.csv"
    joined_csv = output / "joined_predictions_boundary.csv"
    summary_json = output / "summary.json"

    metrics_rows = compute_boundary_metrics(
        query_cache=args.query_cache,
        reference_cache=args.reference_cache,
        k_values=k_values,
        entropy_k_values=entropy_k_values,
        global_search_k=search_k,
        batch_size=int(args.batch_size),
        reference_shard_size=int(args.reference_shard_size),
        device=str(args.device),
        mmap=bool(args.mmap),
        leave_one_out=bool(args.leave_one_out),
        class_key=str(args.class_key),
        skip_reference_percentiles=bool(args.skip_reference_percentiles),
        reference_percentile_cache=args.reference_percentile_cache,
    )
    _write_csv(metrics_csv, metrics_rows)

    summary = {
        "query_cache": str(args.query_cache),
        "reference_cache": str(args.reference_cache),
        "output_dir": str(output),
        "metrics_csv": str(metrics_csv),
        "k_values": k_values,
        "entropy_k_values": entropy_k_values,
        "global_search_k": search_k,
        "batch_size": int(args.batch_size),
        "reference_shard_size": int(args.reference_shard_size),
        "class_key": str(args.class_key),
        "row_count": len(metrics_rows),
        "prediction_csvs": [str(path) for path in args.prediction_csv],
    }
    if args.prediction_csv:
        joined_rows, prediction_summary = join_predictions(metrics_rows, args.prediction_csv)
        _write_csv(joined_csv, joined_rows)
        summary["joined_predictions_csv"] = str(joined_csv)
        summary["prediction_summary"] = prediction_summary

    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"metrics_csv": str(metrics_csv), "summary_json": str(summary_json)}, indent=2, sort_keys=True))


def compute_boundary_metrics(
    *,
    query_cache: Path,
    reference_cache: Path,
    k_values: Sequence[int],
    entropy_k_values: Sequence[int],
    global_search_k: int,
    batch_size: int,
    reference_shard_size: int,
    device: str,
    mmap: bool,
    leave_one_out: bool,
    class_key: str,
    skip_reference_percentiles: bool,
    reference_percentile_cache: Path | None,
) -> list[dict[str, object]]:
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

    torch_device = _resolve_device(device, torch)
    class_indices = _build_class_indices(reference_rows, class_key)
    reference_position = {row.sample_id: idx for idx, row in enumerate(reference_rows)}
    ref_percentiles = (
        {}
        if skip_reference_percentiles
        else _load_or_compute_reference_percentiles(
            reference_embeddings=reference_embeddings,
            reference_rows=reference_rows,
            class_indices=class_indices,
            class_key=class_key,
            k_values=k_values,
            batch_size=batch_size,
            device=torch_device,
            torch=torch,
            cache_path=reference_percentile_cache,
        )
    )

    output_rows: list[dict[str, object]] = []
    for start in range(0, int(query_embeddings.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(query_embeddings.shape[0]))
        q = _to_torch(query_embeddings[start:end], torch=torch, device=torch_device)
        q = _l2_normalize(q)
        query_chunk = query_rows[start:end]
        global_values, global_indices = _topk_global_sharded(
            q=q,
            reference_embeddings=reference_embeddings,
            k=int(global_search_k),
            shard_size=int(reference_shard_size),
            query_ids=[row.sample_id for row in query_chunk],
            reference_position=reference_position,
            leave_one_out=leave_one_out,
            torch=torch,
        )
        global_values_cpu = global_values.detach().cpu().numpy()
        global_indices_cpu = global_indices.detach().cpu().numpy()
        class_rows = _score_class_support(
            q=q,
            query_rows=query_chunk,
            reference_embeddings=reference_embeddings,
            reference_rows=reference_rows,
            class_indices=class_indices,
            class_key=class_key,
            k_values=k_values,
            reference_position=reference_position,
            leave_one_out=leave_one_out,
            torch=torch,
        )
        for offset, row in enumerate(query_chunk):
            out = {
                "sample_id": row.sample_id,
                "class_id": int(row.class_id),
                "class_name": row.class_name,
                "class_key": _row_class_key(row, class_key),
            }
            _add_global_neighbour_columns(
                out=out,
                row=row,
                reference_rows=reference_rows,
                global_values=global_values_cpu[offset],
                global_indices=global_indices_cpu[offset],
                k_values=k_values,
                entropy_k_values=entropy_k_values,
                class_key=class_key,
            )
            out.update(class_rows[offset])
            _add_margin_columns(out, k_values)
            if ref_percentiles:
                key = str(_row_class_key(row, class_key))
                for k in k_values:
                    value = _float_or_none(out.get(f"class_support_k{k}_kth_distance"))
                    if value is None:
                        out[f"class_support_k{k}_percentile"] = ""
                    else:
                        out[f"class_support_k{k}_percentile"] = _percentile(value, ref_percentiles, key, k)
            if k_values:
                pct_values = [
                    float(out[f"class_support_k{k}_percentile"])
                    for k in k_values
                    if out.get(f"class_support_k{k}_percentile") not in ("", None)
                ]
                out["class_support_multiscale_percentile"] = sum(pct_values) / len(pct_values) if pct_values else ""
            output_rows.append(out)
    return output_rows


def _parse_ints(raw: str) -> list[int]:
    out = [int(item.strip()) for item in raw.replace(":", ",").split(",") if item.strip()]
    if not out or any(k < 1 for k in out):
        raise ValueError(f"invalid positive integer list: {raw!r}")
    return out


def _row_class_key(row: ManifestRow, class_key: str) -> str | int:
    if class_key == "class_id":
        return int(row.class_id)
    if class_key == "class_name":
        return str(row.class_name)
    raise ValueError(f"unsupported class_key: {class_key}")


def _resolve_device(device: str, torch_module):
    if device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device)


def _to_torch(values, *, torch, device):
    return torch.from_numpy(values).to(device=device, dtype=torch.float32)


def _l2_normalize(values):
    return values / values.norm(dim=1, keepdim=True).clamp_min(1.0e-12)


def _build_class_indices(rows: Sequence[ManifestRow], class_key: str) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(_row_class_key(row, class_key))].append(int(idx))
    return dict(grouped)


def _topk_global_sharded(
    *,
    q,
    reference_embeddings,
    k: int,
    shard_size: int,
    query_ids: Sequence[str],
    reference_position: dict[str, int],
    leave_one_out: bool,
    torch,
):
    if int(reference_embeddings.shape[0]) < k:
        k = int(reference_embeddings.shape[0])
    top_values = None
    top_indices = None
    n_reference = int(reference_embeddings.shape[0])
    shard = int(shard_size) if int(shard_size) > 0 else n_reference
    for ref_start in range(0, n_reference, shard):
        ref_end = min(ref_start + shard, n_reference)
        ref = _to_torch(reference_embeddings[ref_start:ref_end], torch=torch, device=q.device)
        ref = _l2_normalize(ref)
        sims = q @ ref.T
        if leave_one_out:
            for row_idx, sample_id in enumerate(query_ids):
                ref_idx = reference_position.get(sample_id)
                if ref_idx is not None and ref_start <= ref_idx < ref_end:
                    sims[row_idx, ref_idx - ref_start] = -math.inf
        local_k = min(k, int(ref_end - ref_start))
        values, indices = torch.topk(sims, k=local_k, dim=1, largest=True, sorted=True)
        indices = indices + int(ref_start)
        if top_values is None:
            top_values = values
            top_indices = indices
        else:
            merged_values = torch.cat([top_values, values], dim=1)
            merged_indices = torch.cat([top_indices, indices], dim=1)
            merge_k = min(k, int(merged_values.shape[1]))
            top_values, positions = torch.topk(merged_values, k=merge_k, dim=1, largest=True, sorted=True)
            top_indices = torch.gather(merged_indices, 1, positions)
    if top_values is None or top_indices is None:
        raise ValueError("reference cache has no rows")
    return top_values, top_indices


def _score_class_support(
    *,
    q,
    query_rows: Sequence[ManifestRow],
    reference_embeddings,
    reference_rows: Sequence[ManifestRow],
    class_indices: dict[str, list[int]],
    class_key: str,
    k_values: Sequence[int],
    reference_position: dict[str, int],
    leave_one_out: bool,
    torch,
) -> list[dict[str, object]]:
    out_rows = [_blank_class_row(k_values) for _ in query_rows]
    offsets_by_key: dict[str, list[int]] = defaultdict(list)
    for offset, row in enumerate(query_rows):
        offsets_by_key[str(_row_class_key(row, class_key))].append(offset)

    max_k = int(max(k_values))
    for key, offsets in offsets_by_key.items():
        ref_indices = class_indices.get(key, [])
        if not ref_indices:
            continue
        ref = _to_torch(reference_embeddings[ref_indices], torch=torch, device=q.device)
        ref = _l2_normalize(ref)
        offset_tensor = torch.tensor(offsets, device=q.device, dtype=torch.long)
        q_class = q.index_select(0, offset_tensor)
        sims = q_class @ ref.T
        if leave_one_out:
            local_pos = {reference_rows[global_idx].sample_id: local_idx for local_idx, global_idx in enumerate(ref_indices)}
            for local_query_idx, query_offset in enumerate(offsets):
                ref_local_idx = local_pos.get(query_rows[query_offset].sample_id)
                if ref_local_idx is not None:
                    sims[local_query_idx, ref_local_idx] = -math.inf
        usable = int(torch.isfinite(sims).sum(dim=1).min().item()) if sims.numel() else 0
        local_k = min(max_k, usable)
        if local_k < 1:
            continue
        values, indices = torch.topk(sims, k=local_k, dim=1, largest=True, sorted=True)
        values_cpu = values.detach().cpu().numpy()
        indices_cpu = indices.detach().cpu().numpy()
        for local_query_idx, query_offset in enumerate(offsets):
            distances = 1.0 - values_cpu[local_query_idx]
            if len(distances) > 0:
                nearest_ref = reference_rows[ref_indices[int(indices_cpu[local_query_idx, 0])]]
                out_rows[query_offset]["same_class_nearest_sample_id"] = nearest_ref.sample_id
                out_rows[query_offset]["same_class_nearest_distance"] = float(distances[0])
            for k in k_values:
                if len(distances) < int(k):
                    continue
                kd = distances[: int(k)]
                out_rows[query_offset][f"class_support_k{k}_kth_distance"] = float(kd[-1])
                out_rows[query_offset][f"class_support_k{k}_mean_distance"] = float(kd.mean())
    return out_rows


def _blank_class_row(k_values: Sequence[int]) -> dict[str, object]:
    out: dict[str, object] = {
        "same_class_nearest_sample_id": "",
        "same_class_nearest_distance": "",
    }
    for k in k_values:
        out[f"class_support_k{k}_kth_distance"] = ""
        out[f"class_support_k{k}_mean_distance"] = ""
        out[f"class_support_k{k}_percentile"] = ""
    return out


def _add_global_neighbour_columns(
    *,
    out: dict[str, object],
    row: ManifestRow,
    reference_rows: Sequence[ManifestRow],
    global_values,
    global_indices,
    k_values: Sequence[int],
    entropy_k_values: Sequence[int],
    class_key: str,
) -> None:
    distances = 1.0 - global_values
    out["nearest_sample_id"] = ""
    out["nearest_class_id"] = ""
    out["nearest_class_name"] = ""
    out["nearest_distance"] = ""
    if len(global_indices) > 0:
        nearest = reference_rows[int(global_indices[0])]
        out["nearest_sample_id"] = nearest.sample_id
        out["nearest_class_id"] = int(nearest.class_id)
        out["nearest_class_name"] = nearest.class_name
        out["nearest_distance"] = float(distances[0])
    for k in k_values:
        if len(distances) >= int(k):
            kd = distances[: int(k)]
            out[f"support_k{k}_kth_distance"] = float(kd[-1])
            out[f"support_k{k}_mean_distance"] = float(kd.mean())
        else:
            out[f"support_k{k}_kth_distance"] = ""
            out[f"support_k{k}_mean_distance"] = ""

    query_key = str(_row_class_key(row, class_key))
    other = None
    for rank, ref_idx in enumerate(global_indices):
        ref_row = reference_rows[int(ref_idx)]
        if str(_row_class_key(ref_row, class_key)) != query_key:
            other = (rank, ref_row, float(distances[rank]))
            break
    if other is None:
        out["nearest_other_rank"] = ""
        out["nearest_other_sample_id"] = ""
        out["nearest_other_class_id"] = ""
        out["nearest_other_class_name"] = ""
        out["nearest_other_distance"] = ""
    else:
        rank, ref_row, distance = other
        out["nearest_other_rank"] = int(rank + 1)
        out["nearest_other_sample_id"] = ref_row.sample_id
        out["nearest_other_class_id"] = int(ref_row.class_id)
        out["nearest_other_class_name"] = ref_row.class_name
        out["nearest_other_distance"] = distance

    labels = [str(_row_class_key(reference_rows[int(idx)], class_key)) for idx in global_indices]
    for k in entropy_k_values:
        chunk = labels[: int(k)]
        counts = Counter(chunk)
        denom = max(len(chunk), 1)
        entropy = -sum((count / denom) * math.log(max(count / denom, 1.0e-12)) for count in counts.values())
        norm_entropy = entropy / math.log(max(len(counts), 2))
        true_frac = counts.get(query_key, 0) / denom
        top_key, top_count = counts.most_common(1)[0]
        out[f"local_label_entropy_k{k}"] = float(entropy)
        out[f"local_label_entropy_norm_k{k}"] = float(norm_entropy)
        out[f"local_true_label_fraction_k{k}"] = float(true_frac)
        out[f"local_top_label_key_k{k}"] = top_key
        out[f"local_top_label_fraction_k{k}"] = float(top_count / denom)


def _add_margin_columns(out: dict[str, object], k_values: Sequence[int]) -> None:
    nearest_other = _float_or_none(out.get("nearest_other_distance"))
    same_nearest = _float_or_none(out.get("same_class_nearest_distance"))
    if nearest_other is None or same_nearest is None:
        out["class_margin_nearest"] = ""
        out["trust_ratio_nearest"] = ""
    else:
        out["class_margin_nearest"] = nearest_other - same_nearest
        out["trust_ratio_nearest"] = nearest_other / max(same_nearest, 1.0e-12)
    for k in k_values:
        same_k = _float_or_none(out.get(f"class_support_k{k}_kth_distance"))
        if nearest_other is None or same_k is None:
            out[f"class_margin_vs_k{k}"] = ""
            out[f"trust_ratio_vs_k{k}"] = ""
        else:
            out[f"class_margin_vs_k{k}"] = nearest_other - same_k
            out[f"trust_ratio_vs_k{k}"] = nearest_other / max(same_k, 1.0e-12)


def _load_or_compute_reference_percentiles(
    *,
    reference_embeddings,
    reference_rows: Sequence[ManifestRow],
    class_indices: dict[str, list[int]],
    class_key: str,
    k_values: Sequence[int],
    batch_size: int,
    device,
    torch,
    cache_path: Path | None,
) -> dict[str, dict[int, object]]:
    import numpy as np

    if cache_path is not None and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)["payload"].item()
        return {str(key): {int(k): value for k, value in inner.items()} for key, inner in payload.items()}

    payload: dict[str, dict[int, object]] = {}
    max_k = int(max(k_values)) + 1
    for key, indices in sorted(class_indices.items(), key=lambda kv: str(kv[0])):
        if len(indices) < 2:
            continue
        class_ref = _to_torch(reference_embeddings[indices], torch=torch, device=device)
        class_ref = _l2_normalize(class_ref)
        per_k: dict[int, list[float]] = {int(k): [] for k in k_values if len(indices) > int(k)}
        if not per_k:
            continue
        for start in range(0, len(indices), int(batch_size)):
            end = min(start + int(batch_size), len(indices))
            q = class_ref[start:end]
            sims = q @ class_ref.T
            diag = torch.arange(start, end, device=device)
            sims[torch.arange(end - start, device=device), diag] = -math.inf
            values, _ = torch.topk(sims, k=min(max_k, len(indices) - 1), dim=1, largest=True, sorted=True)
            distances = (1.0 - values.detach().cpu().numpy()).astype("float32", copy=False)
            for k in per_k:
                per_k[int(k)].extend(float(v) for v in distances[:, int(k) - 1])
        payload[str(key)] = {int(k): np.sort(np.asarray(values, dtype="float32")) for k, values in per_k.items()}

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, payload=np.array(payload, dtype=object))
    return payload


def _percentile(value: float, payload: dict[str, dict[int, object]], key: str, k: int) -> object:
    import numpy as np

    values = payload.get(str(key), {}).get(int(k))
    if values is None or len(values) == 0:
        return ""
    return float(np.searchsorted(values, float(value), side="right") / len(values))


def _float_or_none(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def join_predictions(metrics_rows: Sequence[dict[str, object]], prediction_csvs: Sequence[Path]):
    by_sample = {str(row["sample_id"]): row for row in metrics_rows}
    joined_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for path in prediction_csvs:
        with Path(path).open("r", newline="") as f:
            reader = csv.DictReader(f)
            for pred in reader:
                metrics = by_sample.get(str(pred.get("sample_id", "")))
                if metrics is None:
                    continue
                row = dict(metrics)
                row.update(pred)
                row["is_error"] = int(str(pred.get("correct", "0")) == "0")
                joined_rows.append(row)

    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in joined_rows:
        by_model[str(row.get("model_id", ""))].append(row)
    for model_id, rows in sorted(by_model.items()):
        summary[model_id] = _summarize_model_rows(rows)
    return joined_rows, summary


def _summarize_model_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    labels = [int(row["is_error"]) for row in rows]
    out: dict[str, object] = {
        "n": len(rows),
        "error_count": int(sum(labels)),
        "error_rate": float(sum(labels) / max(len(labels), 1)),
        "scores": {},
    }
    score_specs = {
        "confidence_risk": _confidence_risk,
        "negative_logit_margin": _negative_logit_margin,
        "class_support_pct_k5": lambda row: _float_or_none(row.get("class_support_k5_percentile")),
        "class_support_pct_k10": lambda row: _float_or_none(row.get("class_support_k10_percentile")),
        "class_support_pct_k50": lambda row: _float_or_none(row.get("class_support_k50_percentile")),
        "class_support_multiscale": lambda row: _float_or_none(row.get("class_support_multiscale_percentile")),
        "negative_class_margin": _negative_class_margin,
        "local_entropy_k10": lambda row: _float_or_none(row.get("local_label_entropy_norm_k10")),
        "local_entropy_k50": lambda row: _float_or_none(row.get("local_label_entropy_norm_k50")),
    }
    for score_name, fn in score_specs.items():
        parsed: list[tuple[float, int]] = []
        for row, label in zip(rows, labels):
            value = fn(row)
            if value is not None and math.isfinite(float(value)):
                parsed.append((float(value), label))
        if not parsed or len({label for _, label in parsed}) < 2:
            continue
        scores = [score for score, _ in parsed]
        sub_labels = [label for _, label in parsed]
        out["scores"][score_name] = {
            "n": len(parsed),
            "auroc": roc_auc_score(scores, sub_labels),
            "auprc": average_precision_score(scores, sub_labels),
        }
    return out


def _confidence_risk(row: dict[str, object]) -> float | None:
    confidence = _float_or_none(row.get("max_probability_calibrated"))
    if confidence is None:
        confidence = _float_or_none(row.get("max_probability_raw"))
    if confidence is None:
        return None
    return 1.0 - confidence


def _negative_logit_margin(row: dict[str, object]) -> float | None:
    margin = _float_or_none(row.get("logit_margin"))
    return None if margin is None else -margin


def _negative_class_margin(row: dict[str, object]) -> float | None:
    margin = _float_or_none(row.get("class_margin_nearest"))
    return None if margin is None else -margin


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
