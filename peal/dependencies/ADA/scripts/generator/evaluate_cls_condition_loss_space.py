#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_SRC = Path(__file__).resolve().parent
if str(REPO_SRC) not in sys.path:
    sys.path.append(str(REPO_SRC))

from evaluate_cls_condition_fidelity import (  # noqa: E402
    ClsStats,
    _as_dict,
    _choose_base_indices,
    _choose_within_and_cross,
    _load_stage2_weights,
    infer_cls_input_normalization,
    prepare_cls_for_checkpoint,
)
from stage2.transport import create_transport  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import parse_configs  # noqa: E402


def _load_cache(cache_root: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], ClsStats]:
    metadata_path = cache_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {metadata_path}")
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    cls_meta = metadata.get("cls_condition", {})
    if not bool(cls_meta.get("included", False)):
        raise ValueError(f"Cache does not declare paired CLS conditions: {cache_root}")

    stats_path = cache_root / str(cls_meta.get("stats_file", "cls_stats.pt"))
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing CLS stats: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"].float()
    std = stats["std"].float().clamp_min(1.0e-6)

    z_parts: List[torch.Tensor] = []
    cls_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    seen = set()
    for entry in metadata.get("shards", []):
        shard_path = cache_root / entry["file"]
        payload = torch.load(shard_path, map_location="cpu")
        for key in ("z", "cls", "y", "source_index", "view"):
            if key not in payload:
                raise ValueError(f"Paired cache shard missing {key!r}: {shard_path}")
        source = payload["source_index"].long()
        view = payload["view"].long()
        for src, vv in zip(source.tolist(), view.tolist()):
            pair = (int(src), int(vv))
            if pair in seen:
                raise ValueError(f"Duplicate source_index/view pair in paired cache: {pair}")
            seen.add(pair)
        z_parts.append(payload["z"].float())
        cls_parts.append(payload["cls"].float())
        y_parts.append(payload["y"].long())

    z = torch.cat(z_parts, dim=0)
    raw_cls = torch.cat(cls_parts, dim=0)
    labels = torch.cat(y_parts, dim=0)
    if z.ndim != 4:
        raise ValueError(f"Expected patch latent z with shape [N,C,H,W], got {tuple(z.shape)}")
    if raw_cls.ndim != 2:
        raise ValueError(f"Expected CLS with shape [N,D], got {tuple(raw_cls.shape)}")
    if z.shape[0] != raw_cls.shape[0] or z.shape[0] != labels.shape[0]:
        raise ValueError("Cache z, cls, and labels have inconsistent row counts.")
    return z, raw_cls, labels, metadata, ClsStats(mean, std, str(stats_path))


def _state_l2(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).norm(dim=1)


def _loss_and_pred(
    *,
    transport,
    model: torch.nn.Module,
    z: torch.Tensor,
    y: torch.Tensor,
    cls: torch.Tensor | None,
    seed: int,
    use_bf16: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs: Dict[str, Any] = {"y": y}
    if cls is not None:
        kwargs["cls"] = cls
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            terms = transport.training_losses(model, z, kwargs)
    return terms["loss"].detach().float(), terms["pred"].detach().float()


def _loss_and_pred_at_t(
    *,
    transport,
    model: torch.nn.Module,
    z: torch.Tensor,
    y: torch.Tensor,
    cls: torch.Tensor | None,
    x0: torch.Tensor,
    t: torch.Tensor,
    use_bf16: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if getattr(transport.model_type, "name", "") != "VELOCITY":
        raise NotImplementedError("Fixed-timestep CLS diagnostics currently support velocity models only.")
    kwargs: Dict[str, Any] = {"y": y}
    if cls is not None:
        kwargs["cls"] = cls
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            _t, xt, ut, time_info = transport._plan_state(t, x0, z)
            model_t = transport._model_time_input(z, time_info)
            pred = model(xt, model_t, **kwargs)
            loss = (pred.detach().float() - ut.detach().float()).flatten(1).pow(2).mean(dim=1)
    return loss.detach().float(), pred.detach().float()


def _condition_norms(model: torch.nn.Module, *, z: torch.Tensor, y: torch.Tensor, cls: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        t = torch.full((z.shape[0],), 0.5, device=z.device, dtype=torch.float32)
        t_emb = model.t_embedder(t).float()
        y_emb = model.y_embedder(y, False).float()
        if getattr(model, "cls_conditioner", None) is None:
            cls_emb = torch.zeros_like(y_emb)
            null_cls_emb = torch.zeros_like(y_emb)
        else:
            cls_emb = model.cls_conditioner(
                cls,
                False,
                batch_size=z.shape[0],
                device=z.device,
                dtype=t_emb.dtype,
                noise_std=0.0,
            ).float()
            null_cls_emb = model.cls_conditioner(
                None,
                False,
                batch_size=z.shape[0],
                device=z.device,
                dtype=t_emb.dtype,
                noise_std=0.0,
            ).float()
    return {
        "h_t_norm_mean": float(t_emb.norm(dim=1).mean().item()),
        "h_y_norm_mean": float(y_emb.norm(dim=1).mean().item()),
        "h_cls_norm_mean": float(cls_emb.norm(dim=1).mean().item()),
        "h_null_cls_norm_mean": float(null_cls_emb.norm(dim=1).mean().item()),
        "h_cls_minus_null_norm_mean": float((cls_emb - null_cls_emb).norm(dim=1).mean().item()),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, summary: Dict[str, Any]) -> None:
    metrics = summary["aggregate"]
    rows = "\n".join(
        f"<tr><td>{key}</td><td>{value:.6g}</td></tr>"
        for key, value in sorted(metrics.items())
        if isinstance(value, (int, float))
    )
    timestep_rows = "\n".join(
        "<tr>"
        f"<td>{row['t_bin']}</td>"
        f"<td>{row['count']}</td>"
        f"<td>{row['loss_paired_mean']:.6g}</td>"
        f"<td>{row['delta_loss_within_minus_paired_mean']:.6g}</td>"
        f"<td>{row['relative_delta_loss_within_minus_paired_mean']:.6g}</td>"
        f"<td>{row['velocity_delta_paired_vs_within_ratio_mean']:.6g}</td>"
        f"<td>{row['positive_within_delta_rate']:.3f}</td>"
        "</tr>"
        for row in summary.get("timestep_bins", [])
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CLS Loss-Space Diagnostic</title>
  <style>
    body {{ margin: 24px; background: #f7f9fb; color: #18211f; font-family: system-ui, sans-serif; }}
    table {{ border-collapse: collapse; background: white; min-width: 560px; }}
    td, th {{ border: 1px solid #d7dfdd; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef3f2; }}
    code {{ background: #eef3f2; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>CLS Loss-Space Diagnostic</h1>
  <p>Checkpoint: <code>{summary["checkpoint"]}</code></p>
  <p>Cache: <code>{summary["cache_root"]}</code></p>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Timestep bins</h2>
  <p><code>t=0</code> is the noise endpoint and <code>t=1</code> is the data endpoint for the repository's linear velocity path.</p>
  <table>
    <thead>
      <tr>
        <th>Bin</th><th>N</th><th>Paired loss</th><th>ΔL within-paired</th>
        <th>Relative ΔL</th><th>Velocity Δ ratio</th><th>Positive Δ rate</th>
      </tr>
    </thead>
    <tbody>{timestep_rows}</tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def _parse_time_bins(raw: str) -> list[tuple[float, float]]:
    edges = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(edges) < 2:
        raise ValueError("--time-bins must contain at least two comma-separated edges")
    if edges[0] < 0.0 or edges[-1] > 1.0:
        raise ValueError("--time-bins must stay within [0, 1]")
    for left, right in zip(edges, edges[1:]):
        if right <= left:
            raise ValueError("--time-bins edges must be strictly increasing")
    return [(float(left), float(right)) for left, right in zip(edges, edges[1:])]


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def _median(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows])) if rows else float("nan")


def _positive_rate(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) > 0.0 for row in rows])) if rows else float("nan")


def _summarize_timestep_bins(rows: Sequence[Dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["t_bin"]), []).append(row)
    summaries = []
    for name in sorted(grouped, key=lambda value: float(value.split(",")[0].strip("["))):
        bin_rows = grouped[name]
        summaries.append(
            {
                "t_bin": name,
                "count": int(len(bin_rows)),
                "t_mid": _mean(bin_rows, "t_value"),
                "loss_paired_mean": _mean(bin_rows, "loss_paired"),
                "loss_within_shuffle_mean": _mean(bin_rows, "loss_within_shuffle"),
                "delta_loss_within_minus_paired_mean": _mean(bin_rows, "delta_loss_within_minus_paired"),
                "delta_loss_within_minus_paired_median": _median(bin_rows, "delta_loss_within_minus_paired"),
                "relative_delta_loss_within_minus_paired_mean": _mean(
                    bin_rows,
                    "relative_delta_loss_within_minus_paired",
                ),
                "relative_delta_loss_within_minus_paired_median": _median(
                    bin_rows,
                    "relative_delta_loss_within_minus_paired",
                ),
                "velocity_delta_paired_vs_within_ratio_mean": _mean(
                    bin_rows,
                    "velocity_delta_paired_vs_within_ratio",
                ),
                "positive_within_delta_rate": _positive_rate(bin_rows, "delta_loss_within_minus_paired"),
                "positive_null_delta_rate": _positive_rate(bin_rows, "delta_loss_null_minus_paired"),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure whether CLS changes flow loss and velocity predictions.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--time-bins",
        default="0,0.1,0.25,0.5,0.75,0.9,1",
        help="Comma-separated path-time bin edges for fixed-timestep diagnostics. t=0 is noise; t=1 is data.",
    )
    parser.add_argument(
        "--cls-input-normalization",
        choices=("auto", "raw", "train_standardize"),
        default="auto",
        help="CLS preprocessing before the model conditioner. auto resolves imagefolder->raw and latent cache->data.cls_normalization.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CLS loss-space diagnostic requires a CUDA device.")
    device = torch.device("cuda")
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested bf16, but this CUDA device does not support bf16.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time()

    config = OmegaConf.load(args.config)
    (
        _data_config,
        _rae_config,
        model_config,
        transport_config,
        _sampler_config,
        _guidance_config,
        misc_config,
        _training_config,
        _eval_config,
    ) = parse_configs(config)
    if model_config is None:
        raise ValueError("Config must resolve a stage_2 model section.")

    misc = _as_dict(misc_config)
    transport_cfg = _as_dict(transport_config)
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = int(misc.get("time_dist_shift_dim", math.prod(latent_size)))
    shift_base = int(misc.get("time_dist_shift_base", 4096))
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    z_cpu, raw_cls_cpu, labels_cpu, cache_metadata, cls_stats = _load_cache(args.cache_root)
    cls_input_normalization = infer_cls_input_normalization(config, args.cls_input_normalization)
    cls_cpu = prepare_cls_for_checkpoint(raw_cls_cpu, cls_stats, cls_input_normalization)
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 for within-class shuffled CLS controls.")
    num_rows = min(int(args.batch_size) * int(args.num_batches), int(z_cpu.shape[0]))
    base_indices = _choose_base_indices(labels_cpu, num_rows, args.seed)
    within_indices, cross_indices = _choose_within_and_cross(labels_cpu, base_indices, args.seed)

    model = instantiate_from_config(model_config).to(device).eval()
    _load_stage2_weights(model, args.ckpt, use_ema=args.use_ema)
    if getattr(model, "cls_conditioner", None) is None:
        raise ValueError("Model config did not instantiate a CLS conditioner.")

    transport_params = dict(transport_cfg.get("params", {}))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)

    rows: List[Dict[str, Any]] = []
    eps = 1.0e-8
    time_bins = _parse_time_bins(args.time_bins)
    timestep_rows: List[Dict[str, Any]] = []
    for batch_id in range(int(args.num_batches)):
        lo = batch_id * int(args.batch_size)
        hi = min((batch_id + 1) * int(args.batch_size), len(base_indices))
        if hi <= lo:
            break
        rows_i = base_indices[lo:hi]
        within_i = within_indices[lo:hi]
        cross_i = cross_indices[lo:hi]
        z = z_cpu[rows_i].to(device)
        y = labels_cpu[rows_i].to(device)
        cls_paired = cls_cpu[rows_i].to(device)
        cls_within = cls_cpu[within_i].to(device)
        cls_cross = cls_cpu[cross_i].to(device)
        seed = int(args.seed + 10000 + batch_id)

        loss_paired, pred_paired = _loss_and_pred(
            transport=transport,
            model=model,
            z=z,
            y=y,
            cls=cls_paired,
            seed=seed,
            use_bf16=use_bf16,
        )
        loss_within, pred_within = _loss_and_pred(
            transport=transport,
            model=model,
            z=z,
            y=y,
            cls=cls_within,
            seed=seed,
            use_bf16=use_bf16,
        )
        loss_cross, pred_cross = _loss_and_pred(
            transport=transport,
            model=model,
            z=z,
            y=y,
            cls=cls_cross,
            seed=seed,
            use_bf16=use_bf16,
        )
        loss_null, pred_null = _loss_and_pred(
            transport=transport,
            model=model,
            z=z,
            y=y,
            cls=None,
            seed=seed,
            use_bf16=use_bf16,
        )

        norm_null = _state_l2(pred_null).clamp_min(eps)
        norm_paired = _state_l2(pred_paired).clamp_min(eps)
        branch_norms = _condition_norms(model, z=z, y=y, cls=cls_paired)

        for local, row_id in enumerate(rows_i):
            rows.append(
                {
                    "batch_id": batch_id,
                    "row": int(row_id),
                    "class_id": int(labels_cpu[row_id].item()),
                    "within_shuffle_row": int(within_i[local]),
                    "cross_shuffle_row": int(cross_i[local]),
                    "loss_paired": float(loss_paired[local].item()),
                    "loss_within_shuffle": float(loss_within[local].item()),
                    "loss_cross_shuffle": float(loss_cross[local].item()),
                    "loss_null_cls": float(loss_null[local].item()),
                    "delta_loss_within_minus_paired": float((loss_within[local] - loss_paired[local]).item()),
                    "delta_loss_cross_minus_paired": float((loss_cross[local] - loss_paired[local]).item()),
                    "delta_loss_null_minus_paired": float((loss_null[local] - loss_paired[local]).item()),
                    "relative_delta_loss_within_minus_paired": float(
                        ((loss_within[local] - loss_paired[local]) / loss_paired[local].clamp_min(eps)).item()
                    ),
                    "relative_delta_loss_null_minus_paired": float(
                        ((loss_null[local] - loss_paired[local]) / loss_paired[local].clamp_min(eps)).item()
                    ),
                    "velocity_delta_paired_vs_null_ratio": float(
                        (_state_l2((pred_paired - pred_null)[local : local + 1]) / norm_null[local : local + 1]).item()
                    ),
                    "velocity_delta_paired_vs_within_ratio": float(
                        (_state_l2((pred_paired - pred_within)[local : local + 1]) / norm_paired[local : local + 1]).item()
                    ),
                    "velocity_delta_paired_vs_cross_ratio": float(
                        (_state_l2((pred_paired - pred_cross)[local : local + 1]) / norm_paired[local : local + 1]).item()
                    ),
                    **branch_norms,
                }
            )

        torch.manual_seed(seed + 500000)
        torch.cuda.manual_seed_all(seed + 500000)
        x0 = torch.randn_like(z)
        for bin_lo, bin_hi in time_bins:
            t_value = float((bin_lo + bin_hi) / 2.0)
            t = torch.full((z.shape[0],), t_value, device=device, dtype=torch.float32)
            bin_name = f"[{bin_lo:g},{bin_hi:g})" if bin_hi < 1.0 else f"[{bin_lo:g},{bin_hi:g}]"
            loss_paired_t, pred_paired_t = _loss_and_pred_at_t(
                transport=transport,
                model=model,
                z=z,
                y=y,
                cls=cls_paired,
                x0=x0,
                t=t,
                use_bf16=use_bf16,
            )
            loss_within_t, pred_within_t = _loss_and_pred_at_t(
                transport=transport,
                model=model,
                z=z,
                y=y,
                cls=cls_within,
                x0=x0,
                t=t,
                use_bf16=use_bf16,
            )
            loss_cross_t, pred_cross_t = _loss_and_pred_at_t(
                transport=transport,
                model=model,
                z=z,
                y=y,
                cls=cls_cross,
                x0=x0,
                t=t,
                use_bf16=use_bf16,
            )
            loss_null_t, pred_null_t = _loss_and_pred_at_t(
                transport=transport,
                model=model,
                z=z,
                y=y,
                cls=None,
                x0=x0,
                t=t,
                use_bf16=use_bf16,
            )
            norm_null_t = _state_l2(pred_null_t).clamp_min(eps)
            norm_paired_t = _state_l2(pred_paired_t).clamp_min(eps)
            for local, row_id in enumerate(rows_i):
                paired_loss_value = loss_paired_t[local].clamp_min(eps)
                timestep_rows.append(
                    {
                        "batch_id": batch_id,
                        "row": int(row_id),
                        "class_id": int(labels_cpu[row_id].item()),
                        "t_bin": bin_name,
                        "t_bin_lo": float(bin_lo),
                        "t_bin_hi": float(bin_hi),
                        "t_value": t_value,
                        "loss_paired": float(loss_paired_t[local].item()),
                        "loss_within_shuffle": float(loss_within_t[local].item()),
                        "loss_cross_shuffle": float(loss_cross_t[local].item()),
                        "loss_null_cls": float(loss_null_t[local].item()),
                        "delta_loss_within_minus_paired": float(
                            (loss_within_t[local] - loss_paired_t[local]).item()
                        ),
                        "delta_loss_cross_minus_paired": float((loss_cross_t[local] - loss_paired_t[local]).item()),
                        "delta_loss_null_minus_paired": float((loss_null_t[local] - loss_paired_t[local]).item()),
                        "relative_delta_loss_within_minus_paired": float(
                            ((loss_within_t[local] - loss_paired_t[local]) / paired_loss_value).item()
                        ),
                        "relative_delta_loss_null_minus_paired": float(
                            ((loss_null_t[local] - loss_paired_t[local]) / paired_loss_value).item()
                        ),
                        "velocity_delta_paired_vs_null_ratio": float(
                            (_state_l2((pred_paired_t - pred_null_t)[local : local + 1]) / norm_null_t[local : local + 1]).item()
                        ),
                        "velocity_delta_paired_vs_within_ratio": float(
                            (_state_l2((pred_paired_t - pred_within_t)[local : local + 1]) / norm_paired_t[local : local + 1]).item()
                        ),
                        "velocity_delta_paired_vs_cross_ratio": float(
                            (_state_l2((pred_paired_t - pred_cross_t)[local : local + 1]) / norm_paired_t[local : local + 1]).item()
                        ),
                    }
                )

    if not rows:
        raise RuntimeError("No diagnostic rows were produced.")

    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, float)]
    aggregate = {f"{key}_mean": float(np.mean([row[key] for row in rows])) for key in numeric_keys}
    aggregate.update({f"{key}_median": float(np.median([row[key] for row in rows])) for key in numeric_keys})
    aggregate["num_rows"] = int(len(rows))
    aggregate["positive_within_delta_rate"] = float(
        np.mean([row["delta_loss_within_minus_paired"] > 0.0 for row in rows])
    )
    aggregate["positive_null_delta_rate"] = float(np.mean([row["delta_loss_null_minus_paired"] > 0.0 for row in rows]))
    timestep_bin_summary = _summarize_timestep_bins(timestep_rows)

    summary = {
        "experiment_id": "cls_condition_loss_space",
        "config": str(args.config),
        "checkpoint": str(args.ckpt),
        "use_ema": bool(args.use_ema),
        "cache_root": str(args.cache_root),
        "cache_rows": int(z_cpu.shape[0]),
        "cls_input_normalization_requested": args.cls_input_normalization,
        "cls_input_normalization": cls_input_normalization,
        "raw_cls_stats_source": cls_stats.stats_path,
        "shared_noise_across_modes": True,
        "within_class_permutation_seed": int(args.seed) + 17,
        "guidance_method": "teacher_forced_training_losses_same_rng_and_fixed_timestep_x0",
        "precision": args.precision,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "elapsed_seconds": float(time() - started),
        "cache_metadata_cls_condition": cache_metadata.get("cls_condition", {}),
        "aggregate": aggregate,
        "timestep_convention": "For the repository's linear velocity path, t=0 is the noise endpoint and t=1 is the data endpoint. Timestep-bin diagnostics use fixed bin midpoints.",
        "time_bins": [{"lo": lo, "hi": hi, "mid": (lo + hi) / 2.0} for lo, hi in time_bins],
        "timestep_bins": timestep_bin_summary,
        "interpretation": {
            "delta_loss_within_minus_paired": "Positive means the paired CLS gives lower flow loss than within-class shuffled CLS.",
            "relative_delta_loss_within_minus_paired": "Raw paired-vs-within loss advantage divided by paired loss.",
            "delta_loss_null_minus_paired": "Positive means the paired CLS gives lower flow loss than null CLS.",
            "velocity_delta_paired_vs_null_ratio": "Mean relative change in predicted velocity induced by replacing null CLS with paired CLS.",
        },
    }

    _write_csv(args.out_dir / "loss_space_rows.csv", rows)
    _write_csv(args.out_dir / "loss_space_timestep_rows.csv", timestep_rows)
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    _write_html(args.out_dir / "report.html", summary)
    print(f"[loss-space] wrote {args.out_dir / 'summary.json'}")
    print(f"[loss-space] wrote {args.out_dir / 'loss_space_rows.csv'}")
    print(f"[loss-space] wrote {args.out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
