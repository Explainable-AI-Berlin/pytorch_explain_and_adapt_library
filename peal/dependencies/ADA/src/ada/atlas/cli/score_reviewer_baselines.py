from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from ada.atlas.cli.score_boundary_metrics import (
    _build_class_indices,
    _float_or_none,
    _l2_normalize,
    _row_class_key,
    _to_torch,
    _topk_global_sharded,
)
from ada.atlas.data.manifests import ManifestRow, load_manifest_csv
from ada.atlas.metrics.error_detection import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute reviewer-facing proximity baselines: ProCal-style global proximity, "
            "class-conditional KDE, Trust Score, and diagonal class-conditional "
            "Mahalanobis/GDA scores."
        )
    )
    parser.add_argument("--query-cache", required=True, type=Path)
    parser.add_argument("--reference-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-csv", action="append", type=Path, default=[])
    parser.add_argument("--class-key", choices=["class_id", "class_name"], default="class_name")
    parser.add_argument("--procal-k", default=10, type=int)
    parser.add_argument("--trust-k", default=10, type=int)
    parser.add_argument("--trust-alpha", default=0.10, type=float)
    parser.add_argument("--trust-search-k", default=4096, type=int)
    parser.add_argument("--kde-bandwidth-scale", default=1.0, type=float)
    parser.add_argument("--kde-min-bandwidth", default=0.01, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--reference-shard-size", default=100000, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--mahalanobis-shrinkage", default=0.10, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores_csv = output / "reviewer_baseline_scores.csv"
    joined_csv = output / "joined_reviewer_baselines.csv"
    summary_json = output / "summary.json"

    rows, query_class_lookup, keyed_scores = compute_scores(
        query_cache=args.query_cache,
        reference_cache=args.reference_cache,
        prediction_csvs=args.prediction_csv,
        class_key=str(args.class_key),
        procal_k=int(args.procal_k),
        trust_k=int(args.trust_k),
        trust_alpha=float(args.trust_alpha),
        trust_search_k=int(args.trust_search_k),
        kde_bandwidth_scale=float(args.kde_bandwidth_scale),
        kde_min_bandwidth=float(args.kde_min_bandwidth),
        batch_size=int(args.batch_size),
        reference_shard_size=int(args.reference_shard_size),
        device=str(args.device),
        mmap=bool(args.mmap),
        mahalanobis_shrinkage=float(args.mahalanobis_shrinkage),
    )
    _write_csv(scores_csv, rows)

    summary: dict[str, object] = {
        "query_cache": str(args.query_cache),
        "reference_cache": str(args.reference_cache),
        "output_dir": str(output),
        "scores_csv": str(scores_csv),
        "class_key": str(args.class_key),
        "procal_k": int(args.procal_k),
        "trust_k": int(args.trust_k),
        "trust_alpha": float(args.trust_alpha),
        "trust_search_k": int(args.trust_search_k),
        "kde_bandwidth_scale": float(args.kde_bandwidth_scale),
        "kde_min_bandwidth": float(args.kde_min_bandwidth),
        "batch_size": int(args.batch_size),
        "reference_shard_size": int(args.reference_shard_size),
        "mahalanobis_shrinkage": float(args.mahalanobis_shrinkage),
        "row_count": len(rows),
        "prediction_csvs": [str(path) for path in args.prediction_csv],
    }
    if args.prediction_csv:
        joined_rows, prediction_summary = join_predictions(rows, query_class_lookup, keyed_scores, args.prediction_csv)
        _write_csv(joined_csv, joined_rows)
        summary["joined_csv"] = str(joined_csv)
        summary["prediction_summary"] = prediction_summary
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"scores_csv": str(scores_csv), "summary_json": str(summary_json)}, indent=2, sort_keys=True))


def compute_scores(
    *,
    query_cache: Path,
    reference_cache: Path,
    prediction_csvs: Sequence[Path],
    class_key: str,
    procal_k: int,
    trust_k: int,
    trust_alpha: float,
    trust_search_k: int,
    kde_bandwidth_scale: float,
    kde_min_bandwidth: float,
    batch_size: int,
    reference_shard_size: int,
    device: str,
    mmap: bool,
    mahalanobis_shrinkage: float,
) -> tuple[list[dict[str, object]], dict[int, str], dict[tuple[str, str], dict[str, object]]]:
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
    reference_position = {row.sample_id: idx for idx, row in enumerate(reference_rows)}
    class_indices = _build_class_indices(reference_rows, class_key)
    query_class_lookup = _query_class_lookup(query_rows, class_key)
    prediction_keys = _prediction_keys_by_sample(prediction_csvs, query_class_lookup)

    gda = _fit_diag_gda(
        reference_embeddings=reference_embeddings,
        reference_rows=reference_rows,
        class_key=class_key,
        shrinkage=float(mahalanobis_shrinkage),
    )
    core_indices, kde_bandwidths = _build_trust_core_indices(
        reference_embeddings=reference_embeddings,
        reference_rows=reference_rows,
        class_indices=class_indices,
        class_key=class_key,
        trust_k=int(trust_k),
        alpha=float(trust_alpha),
        kde_bandwidth_scale=float(kde_bandwidth_scale),
        kde_min_bandwidth=float(kde_min_bandwidth),
        batch_size=int(batch_size),
        device=torch_device,
        torch=torch,
    )
    core_flat_indices = [idx for indices in core_indices.values() for idx in indices]
    core_reference_rows = [reference_rows[idx] for idx in core_flat_indices]
    core_embeddings = reference_embeddings[core_flat_indices] if core_flat_indices else reference_embeddings[:0]
    core_position = {row.sample_id: idx for idx, row in enumerate(core_reference_rows)}
    core_positions_by_key: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(core_reference_rows):
        core_positions_by_key[str(_row_class_key(row, class_key))].append(idx)
    max_core_class_size = max((len(indices) for indices in core_positions_by_key.values()), default=0)

    output_rows: list[dict[str, object]] = []
    keyed_scores: dict[tuple[str, str], dict[str, object]] = {}
    core_search_k = max(int(trust_search_k), int(max_core_class_size) + 1)
    for start in range(0, int(query_embeddings.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(query_embeddings.shape[0]))
        query_chunk = query_rows[start:end]
        q = _to_torch(query_embeddings[start:end], torch=torch, device=torch_device)
        q = _l2_normalize(q)
        global_values, global_indices = _topk_global_sharded(
            q=q,
            reference_embeddings=reference_embeddings,
            k=int(procal_k),
            shard_size=int(reference_shard_size),
            query_ids=[row.sample_id for row in query_chunk],
            reference_position=reference_position,
            leave_one_out=False,
            torch=torch,
        )
        if len(core_flat_indices) > 0:
            core_values, core_indices_local = _topk_global_sharded(
                q=q,
                reference_embeddings=core_embeddings,
                k=min(int(core_search_k), int(core_embeddings.shape[0])),
                shard_size=int(reference_shard_size),
                query_ids=[row.sample_id for row in query_chunk],
                reference_position=core_position,
                leave_one_out=False,
                torch=torch,
            )
            core_values_cpu = core_values.detach().cpu().numpy()
            core_indices_cpu = core_indices_local.detach().cpu().numpy()
        else:
            core_values_cpu = None
            core_indices_cpu = None
        global_values_cpu = global_values.detach().cpu().numpy()
        global_indices_cpu = global_indices.detach().cpu().numpy()
        q_np = query_embeddings[start:end]
        needed_keys_by_offset = {
            offset: {str(_row_class_key(row, class_key)), *prediction_keys.get(row.sample_id, set())}
            for offset, row in enumerate(query_chunk)
        }
        kde_scores = _score_kde_for_query_keys(
            q=q,
            query_rows=query_chunk,
            keys_by_offset=needed_keys_by_offset,
            reference_embeddings=reference_embeddings,
            class_indices=class_indices,
            bandwidths=kde_bandwidths,
            torch=torch,
        )
        trust_same_distances = _score_core_same_distances_for_query_keys(
            q=q,
            query_rows=query_chunk,
            keys_by_offset=needed_keys_by_offset,
            core_embeddings=core_embeddings,
            core_positions_by_key=core_positions_by_key,
            torch=torch,
        )
        for offset, row in enumerate(query_chunk):
            distances = 1.0 - global_values_cpu[offset]
            out: dict[str, object] = {
                "sample_id": row.sample_id,
                "class_id": int(row.class_id),
                "class_name": row.class_name,
                "class_key": _row_class_key(row, class_key),
            }
            if len(distances) >= int(procal_k):
                kd = distances[: int(procal_k)]
                out[f"procal_global_k{procal_k}_mean_distance"] = float(kd.mean())
                out[f"procal_global_k{procal_k}_kth_distance"] = float(kd[-1])
            else:
                out[f"procal_global_k{procal_k}_mean_distance"] = ""
                out[f"procal_global_k{procal_k}_kth_distance"] = ""

            true_key = str(_row_class_key(row, class_key))
            needed_keys = needed_keys_by_offset[offset]
            for key in sorted(key for key in needed_keys if key):
                key_scores = _gda_scores_for_key(gda, q_np[offset], key, prefix="")
                key_scores.update(kde_scores.get((row.sample_id, key), {}))
                if core_values_cpu is not None and core_indices_cpu is not None:
                    key_scores.update(
                        _trust_distances_from_core_topk(
                            core_values=core_values_cpu[offset],
                            core_indices=core_indices_cpu[offset],
                            core_rows=core_reference_rows,
                            query_key=key,
                            class_key=class_key,
                            same_distance_override=trust_same_distances.get((row.sample_id, key)),
                            prefix="",
                        )
                    )
                keyed_scores[(row.sample_id, key)] = key_scores
            out.update(_prefix_key_scores(keyed_scores.get((row.sample_id, true_key), {}), prefix="true_class"))
            if core_values_cpu is not None and core_indices_cpu is not None:
                # True-class trust scores were inserted from keyed_scores above.
                pass
            output_rows.append(out)
    return output_rows, query_class_lookup, keyed_scores


def _resolve_device(device: str, torch_module):
    if device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device)


def _query_class_lookup(rows: Sequence[ManifestRow], class_key: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in rows:
        out[int(row.class_id)] = str(_row_class_key(row, class_key))
    return out


def _prediction_keys_by_sample(
    prediction_csvs: Sequence[Path],
    query_class_lookup: dict[int, str],
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for path in prediction_csvs:
        with Path(path).open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pred = int(row.get("predicted_label", -1))
                except (TypeError, ValueError):
                    continue
                key = query_class_lookup.get(pred, "")
                if key:
                    out[str(row.get("sample_id", ""))].add(str(key))
    return out


def _fit_diag_gda(
    *,
    reference_embeddings,
    reference_rows: Sequence[ManifestRow],
    class_key: str,
    shrinkage: float,
) -> dict[str, object]:
    import numpy as np

    keys = sorted({str(_row_class_key(row, class_key)) for row in reference_rows})
    key_to_idx = {key: idx for idx, key in enumerate(keys)}
    dim = int(reference_embeddings.shape[1])
    counts = np.zeros(len(keys), dtype=np.float64)
    sums = np.zeros((len(keys), dim), dtype=np.float64)
    sums_sq = np.zeros((len(keys), dim), dtype=np.float64)
    for idx, row in enumerate(reference_rows):
        key_idx = key_to_idx[str(_row_class_key(row, class_key))]
        values = np.asarray(reference_embeddings[idx], dtype=np.float64)
        counts[key_idx] += 1.0
        sums[key_idx] += values
        sums_sq[key_idx] += values * values
    means = sums / np.maximum(counts[:, None], 1.0)
    variances = sums_sq / np.maximum(counts[:, None], 1.0) - means * means
    total_count = max(float(counts.sum()), 1.0)
    global_mean = sums.sum(axis=0) / total_count
    global_var = sums_sq.sum(axis=0) / total_count - global_mean * global_mean
    shrink = min(max(float(shrinkage), 0.0), 1.0)
    variances = (1.0 - shrink) * variances + shrink * global_var[None, :]
    variances = np.maximum(variances, 1.0e-6)
    log_det = np.log(variances).sum(axis=1)
    return {
        "keys": keys,
        "key_to_idx": key_to_idx,
        "means": means.astype("float32"),
        "variances": variances.astype("float32"),
        "log_det": log_det.astype("float64"),
        "dim": dim,
    }


def _gda_scores_for_key(gda: dict[str, object], query, key: str, prefix: str) -> dict[str, object]:
    import numpy as np

    key_to_idx = gda["key_to_idx"]
    if key not in key_to_idx:
        return {
            _join_prefix(prefix, "mahalanobis_diag"): "",
            _join_prefix(prefix, "gda_diag_log_density"): "",
        }
    idx = int(key_to_idx[key])
    means = gda["means"]
    variances = gda["variances"]
    diff = np.asarray(query, dtype=np.float32) - means[idx]
    mahal = float(np.sum((diff * diff) / variances[idx]))
    log_density = -0.5 * (mahal + float(gda["log_det"][idx]) + int(gda["dim"]) * math.log(2.0 * math.pi))
    return {
        _join_prefix(prefix, "mahalanobis_diag"): mahal,
        _join_prefix(prefix, "gda_diag_log_density"): log_density,
    }


def _build_trust_core_indices(
    *,
    reference_embeddings,
    reference_rows: Sequence[ManifestRow],
    class_indices: dict[str, list[int]],
    class_key: str,
    trust_k: int,
    alpha: float,
    kde_bandwidth_scale: float,
    kde_min_bandwidth: float,
    batch_size: int,
    device,
    torch,
) -> tuple[dict[str, list[int]], dict[str, float]]:
    del class_key
    alpha = min(max(float(alpha), 0.0), 0.95)
    keep_by_class: dict[str, list[int]] = {}
    bandwidths: dict[str, float] = {}
    for key, indices in sorted(class_indices.items(), key=lambda kv: str(kv[0])):
        if len(indices) <= 1:
            keep_by_class[key] = list(indices)
            bandwidths[key] = float(max(kde_min_bandwidth, 1.0))
            continue
        if alpha <= 0.0:
            keep_by_class[key] = list(indices)
        class_ref = _to_torch(reference_embeddings[indices], torch=torch, device=device)
        class_ref = _l2_normalize(class_ref)
        local_k = min(int(trust_k) + 1, len(indices))
        density_scores: list[tuple[float, int]] = []
        for start in range(0, len(indices), int(batch_size)):
            end = min(start + int(batch_size), len(indices))
            q = class_ref[start:end]
            sims = q @ class_ref.T
            diag = torch.arange(start, end, device=device)
            sims[torch.arange(end - start, device=device), diag] = -math.inf
            values, _ = torch.topk(sims, k=min(local_k, len(indices) - 1), dim=1, largest=True, sorted=True)
            distances = (1.0 - values.detach().cpu().numpy()).astype("float32", copy=False)
            kth = distances[:, min(int(trust_k), distances.shape[1]) - 1]
            density_scores.extend((float(distance), indices[start + offset]) for offset, distance in enumerate(kth))
        if density_scores:
            kth_distances = sorted(distance for distance, _idx in density_scores)
            median_cosine_distance = kth_distances[len(kth_distances) // 2]
            # Embeddings are L2-normalized. Squared Euclidean distance is
            # 2 * cosine_distance, so h lives on the Euclidean scale.
            bandwidths[key] = float(
                max(
                    float(kde_min_bandwidth),
                    float(kde_bandwidth_scale) * math.sqrt(max(2.0 * median_cosine_distance, 1.0e-12)),
                )
            )
        density_scores.sort(key=lambda item: (item[0], item[1]))
        keep_count = max(1, int(math.ceil((1.0 - alpha) * len(density_scores))))
        keep_by_class[key] = [idx for _distance, idx in density_scores[:keep_count]]
    return keep_by_class, bandwidths


def _score_kde_for_query_keys(
    *,
    q,
    query_rows: Sequence[ManifestRow],
    keys_by_offset: dict[int, set[str]],
    reference_embeddings,
    class_indices: dict[str, list[int]],
    bandwidths: dict[str, float],
    torch,
) -> dict[tuple[str, str], dict[str, object]]:
    dim = int(q.shape[1])
    offsets_by_key: dict[str, list[int]] = defaultdict(list)
    for offset, keys in keys_by_offset.items():
        for key in keys:
            if key:
                offsets_by_key[str(key)].append(int(offset))

    out: dict[tuple[str, str], dict[str, object]] = {}
    constant = -0.5 * dim * math.log(2.0 * math.pi)
    for key, offsets in offsets_by_key.items():
        ref_indices = class_indices.get(key, [])
        if not ref_indices:
            for offset in offsets:
                out[(query_rows[offset].sample_id, key)] = {
                    "kde_log_density": "",
                    "kde_bandwidth": "",
                    "kde_reference_count": 0,
                }
            continue
        bandwidth = float(bandwidths.get(key, 1.0))
        bandwidth = max(bandwidth, 1.0e-6)
        ref = _to_torch(reference_embeddings[ref_indices], torch=torch, device=q.device)
        ref = _l2_normalize(ref)
        offset_tensor = torch.tensor(offsets, device=q.device, dtype=torch.long)
        q_key = q.index_select(0, offset_tensor)
        sims = q_key @ ref.T
        sq_euclidean = (2.0 * (1.0 - sims)).clamp_min(0.0)
        log_kernel = -0.5 * sq_euclidean / (bandwidth * bandwidth)
        log_density = (
            torch.logsumexp(log_kernel, dim=1)
            - math.log(max(len(ref_indices), 1))
            - dim * math.log(bandwidth)
            + constant
        )
        values = log_density.detach().cpu().numpy()
        for local_idx, offset in enumerate(offsets):
            out[(query_rows[offset].sample_id, key)] = {
                "kde_log_density": float(values[local_idx]),
                "kde_bandwidth": bandwidth,
                "kde_reference_count": int(len(ref_indices)),
            }
    return out


def _score_core_same_distances_for_query_keys(
    *,
    q,
    query_rows: Sequence[ManifestRow],
    keys_by_offset: dict[int, set[str]],
    core_embeddings,
    core_positions_by_key: dict[str, list[int]],
    torch,
) -> dict[tuple[str, str], float]:
    offsets_by_key: dict[str, list[int]] = defaultdict(list)
    for offset, keys in keys_by_offset.items():
        for key in keys:
            if key:
                offsets_by_key[str(key)].append(int(offset))

    out: dict[tuple[str, str], float] = {}
    for key, offsets in offsets_by_key.items():
        positions = core_positions_by_key.get(str(key), [])
        if not positions:
            continue
        ref = _to_torch(core_embeddings[positions], torch=torch, device=q.device)
        ref = _l2_normalize(ref)
        offset_tensor = torch.tensor(offsets, device=q.device, dtype=torch.long)
        q_key = q.index_select(0, offset_tensor)
        sims = q_key @ ref.T
        distances = 1.0 - sims.max(dim=1).values.detach().cpu().numpy()
        for local_idx, offset in enumerate(offsets):
            out[(query_rows[offset].sample_id, key)] = float(distances[local_idx])
    return out


def _trust_distances_from_core_topk(
    *,
    core_values,
    core_indices,
    core_rows: Sequence[ManifestRow],
    query_key: str,
    class_key: str,
    same_distance_override: float | None,
    prefix: str,
) -> dict[str, object]:
    distances = 1.0 - core_values
    same_distance = same_distance_override
    other_distance = None
    other_key = ""
    for rank, ref_idx in enumerate(core_indices):
        ref_row = core_rows[int(ref_idx)]
        ref_key = str(_row_class_key(ref_row, class_key))
        if ref_key == str(query_key) and same_distance is None:
            same_distance = float(distances[rank])
        if ref_key != str(query_key) and other_distance is None:
            other_distance = float(distances[rank])
            other_key = ref_key
        if same_distance is not None and other_distance is not None:
            break
    if same_distance is None or other_distance is None:
        return {
            _join_prefix(prefix, "trust_same_core_distance"): "" if same_distance is None else same_distance,
            _join_prefix(prefix, "trust_other_core_distance"): "" if other_distance is None else other_distance,
            _join_prefix(prefix, "trust_other_key"): other_key,
            _join_prefix(prefix, "trust_score"): "",
            _join_prefix(prefix, "trust_log_score"): "",
        }
    trust_score = other_distance / max(same_distance, 1.0e-12)
    return {
        _join_prefix(prefix, "trust_same_core_distance"): same_distance,
        _join_prefix(prefix, "trust_other_core_distance"): other_distance,
        _join_prefix(prefix, "trust_other_key"): other_key,
        _join_prefix(prefix, "trust_score"): trust_score,
        _join_prefix(prefix, "trust_log_score"): math.log(max(trust_score, 1.0e-12)),
    }


def _join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _prefix_key_scores(scores: dict[str, object], prefix: str) -> dict[str, object]:
    return {_join_prefix(prefix, key): value for key, value in scores.items()}


def join_predictions(
    score_rows: Sequence[dict[str, object]],
    query_class_lookup: dict[int, str],
    keyed_scores: dict[tuple[str, str], dict[str, object]],
    prediction_csvs: Sequence[Path],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_sample = {str(row["sample_id"]): row for row in score_rows}
    joined_rows: list[dict[str, object]] = []
    for path in prediction_csvs:
        with Path(path).open("r", newline="") as f:
            reader = csv.DictReader(f)
            for pred in reader:
                base = by_sample.get(str(pred.get("sample_id", "")))
                if base is None:
                    continue
                row = dict(base)
                row.update(pred)
                row["is_error"] = int(str(pred.get("correct", "0")) == "0")
                pred_key = query_class_lookup.get(int(pred.get("predicted_label", -1)), "")
                row["predicted_class_key"] = pred_key
                if pred_key:
                    row.update(
                        _prefix_key_scores(
                            keyed_scores.get((str(pred.get("sample_id", "")), pred_key), {}),
                            prefix="predicted_class",
                        )
                    )
                joined_rows.append(row)

    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in joined_rows:
        by_model[str(row.get("model_id", ""))].append(row)
    summary = {model_id: _summarize_model_rows(rows) for model_id, rows in sorted(by_model.items())}
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
        "procal_global_mean": lambda row: _float_or_none(row.get("procal_global_k10_mean_distance")),
        "procal_global_kth": lambda row: _float_or_none(row.get("procal_global_k10_kth_distance")),
        "true_class_mahalanobis_diag": lambda row: _float_or_none(row.get("true_class_mahalanobis_diag")),
        "negative_true_class_gda_log_density": _negative_true_gda_log_density,
        "negative_true_class_kde_log_density": _negative_true_kde_log_density,
        "negative_true_class_trust": _negative_true_trust,
        "negative_true_class_log_trust": _negative_true_log_trust,
        "predicted_class_mahalanobis_diag": lambda row: _float_or_none(row.get("predicted_class_mahalanobis_diag")),
        "negative_predicted_class_gda_log_density": _negative_predicted_gda_log_density,
        "negative_predicted_class_kde_log_density": _negative_predicted_kde_log_density,
        "negative_predicted_class_trust": _negative_predicted_trust,
        "negative_predicted_class_log_trust": _negative_predicted_log_trust,
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


def _negative_true_gda_log_density(row: dict[str, object]) -> float | None:
    value = _float_or_none(row.get("true_class_gda_diag_log_density"))
    return None if value is None else -value


def _negative_true_kde_log_density(row: dict[str, object]) -> float | None:
    value = _float_or_none(row.get("true_class_kde_log_density"))
    return None if value is None else -value


def _negative_true_trust(row: dict[str, object]) -> float | None:
    value = _float_or_none(row.get("true_class_trust_score"))
    return None if value is None else -value


def _negative_true_log_trust(row: dict[str, object]) -> float | None:
    value = _float_or_none(row.get("true_class_trust_log_score"))
    return None if value is None else -value


def _negative_predicted_gda_log_density(row: dict[str, object]) -> float | None:
    value = _float_or_none(row.get("predicted_class_gda_diag_log_density"))
    return None if value is None else -value


def _negative_predicted_kde_log_density(row: dict[str, object]) -> float | None:
    value = _float_or_none(row.get("predicted_class_kde_log_density"))
    return None if value is None else -value


def _negative_predicted_trust(row: dict[str, object]) -> float | None:
    value = _float_or_none(row.get("predicted_class_trust_score"))
    return None if value is None else -value


def _negative_predicted_log_trust(row: dict[str, object]) -> float | None:
    value = _float_or_none(row.get("predicted_class_trust_log_score"))
    return None if value is None else -value


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
