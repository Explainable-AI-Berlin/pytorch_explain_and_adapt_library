#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.utils import save_image

REPO_SRC = Path(__file__).resolve().parent
if str(REPO_SRC) not in sys.path:
    sys.path.append(str(REPO_SRC))

from stage1 import RAE  # noqa: E402
from stage2.models import Stage2ModelProtocol  # noqa: E402
from stage2.state_utils import (  # noqa: E402
    Stage2State,
    decode_stage2_state,
    duplicate_state_for_guidance,
    final_state_from_trajectory,
    infer_aux_state_spec,
    make_initial_sample_state,
    split_guided_state,
)
from stage2.transport import Sampler, create_transport  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import parse_configs  # noqa: E402


COUNTERFACTUAL_MODES = {
    "paired_cls",
    "within_class_shuffled_cls",
    "cross_class_shuffled_cls",
    "null_cls",
}


@dataclass(frozen=True)
class ClsStats:
    mean: torch.Tensor
    std: torch.Tensor
    stats_path: str


@dataclass(frozen=True)
class SamplingCall:
    state: Stage2State
    model_fwd: Any
    model_kwargs: Dict[str, Any]
    guidance_method: str
    split_after_guidance: bool


def _as_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(OmegaConf.to_container(cfg, resolve=True))


def _load_stage2_weights(
    model: torch.nn.Module,
    ckpt_path: Path,
    *,
    use_ema: bool,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    key = "ema" if use_ema else "model"
    if key not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain key {key!r}.")
    missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
    print(
        f"[ckpt] loaded {ckpt_path} key={key} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"[ckpt] missing first 10: {missing[:10]}")
    if unexpected:
        print(f"[ckpt] unexpected first 10: {unexpected[:10]}")
    model.eval()
    return ckpt


def infer_cls_input_normalization(config: Any, requested: str = "auto") -> str:
    requested = str(requested)
    if requested in {"raw", "train_standardize"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported cls input normalization: {requested!r}")

    data_kind = OmegaConf.select(config, "data.kind")
    if str(data_kind) == "imagefolder":
        return "raw"
    if str(data_kind) == "stage2_latent_cache":
        mode = OmegaConf.select(config, "data.cls_normalization")
        if mode is None or str(mode) in {"none", "raw"}:
            return "raw"
        if str(mode) == "train_standardize":
            return "train_standardize"
        raise ValueError(f"Unsupported data.cls_normalization for CLS evaluator auto mode: {mode!r}")
    return "raw"


def prepare_cls_for_checkpoint(raw_cls: torch.Tensor, stats: ClsStats, normalization: str) -> torch.Tensor:
    if normalization == "raw":
        return raw_cls.float()
    if normalization == "train_standardize":
        return (raw_cls.float() - stats.mean.float()) / stats.std.float().clamp_min(1.0e-6)
    raise ValueError(f"Unsupported cls input normalization: {normalization!r}")


def _load_cache_tensors(cache_root: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], ClsStats]:
    metadata_path = cache_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {metadata_path}")
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    cls_meta = metadata.get("cls_condition", {})
    if not bool(cls_meta.get("included", False)):
        raise ValueError(f"Cache does not declare paired CLS conditions: {cache_root}")

    cls_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    source_parts: List[torch.Tensor] = []
    view_parts: List[torch.Tensor] = []
    seen = set()
    for entry in metadata.get("shards", []):
        shard_path = cache_root / entry["file"]
        payload = torch.load(shard_path, map_location="cpu")
        for key in ("cls", "y", "source_index", "view"):
            if key not in payload:
                raise ValueError(f"Paired cache shard missing {key!r}: {shard_path}")
        cls = payload["cls"].float()
        y = payload["y"].long()
        source = payload["source_index"].long()
        view = payload["view"].long()
        if cls.ndim != 2 or cls.shape[1] != int(cls_meta.get("shape", [0])[0]):
            raise ValueError(f"Unexpected CLS shape {tuple(cls.shape)} in {shard_path}")
        for src, vv in zip(source.tolist(), view.tolist()):
            pair = (int(src), int(vv))
            if pair in seen:
                raise ValueError(f"Duplicate source_index/view pair in paired cache: {pair}")
            seen.add(pair)
        cls_parts.append(cls)
        y_parts.append(y)
        source_parts.append(source)
        view_parts.append(view)

    raw_cls = torch.cat(cls_parts, dim=0)
    labels = torch.cat(y_parts, dim=0)
    source_index = torch.cat(source_parts, dim=0)
    view = torch.cat(view_parts, dim=0)

    stats_file = str(cls_meta.get("stats_file", "cls_stats.pt"))
    stats_path = cache_root / stats_file
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing CLS stats: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"].float()
    std = stats["std"].float().clamp_min(1.0e-6)
    return raw_cls, prepare_cls_for_checkpoint(raw_cls, ClsStats(mean, std, str(stats_path)), "train_standardize"), labels, source_index, metadata, ClsStats(mean, std, str(stats_path))


def _class_index(labels: torch.Tensor) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for i, y in enumerate(labels.tolist()):
        out.setdefault(int(y), []).append(i)
    return out


def _choose_base_indices(labels: torch.Tensor, n: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    by_class = _class_index(labels)
    classes = [y for y, rows in sorted(by_class.items()) if len(rows) >= 2]
    if not classes:
        raise ValueError("Need at least one class with two examples for shuffled-CLS controls.")

    selected: List[int] = []
    offset = int(rng.integers(0, len(classes))) if classes else 0
    for j in range(len(classes)):
        if len(selected) >= n:
            break
        cls = classes[(offset + j) % len(classes)]
        selected.append(int(rng.choice(by_class[cls])))
    while len(selected) < n:
        cls = int(rng.choice(classes))
        selected.append(int(rng.choice(by_class[cls])))
    return selected


def _choose_within_and_cross(labels: torch.Tensor, base_indices: List[int], seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed + 17)
    by_class = _class_index(labels)
    all_classes = [y for y, rows in by_class.items() if len(rows) > 0]
    within: List[int] = []
    cross: List[int] = []
    for idx in base_indices:
        y = int(labels[idx].item())
        same = [j for j in by_class[y] if j != idx]
        if not same:
            raise ValueError(f"Class {y} has no within-class shuffle partner.")
        within.append(int(rng.choice(same)))
        other_classes = [c for c in all_classes if c != y]
        other = int(rng.choice(other_classes))
        cross.append(int(rng.choice(by_class[other])))
    return within, cross


def _best_fixed_noise_class(labels: torch.Tensor, n: int) -> List[int]:
    by_class = _class_index(labels)
    cls, rows = max(by_class.items(), key=lambda item: len(item[1]))
    rows = list(rows)
    return rows[: min(n, len(rows))]


def _cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.normalize(a.float(), dim=1) @ F.normalize(b.float(), dim=1).T


def _retrieval_rank(
    query_cls: torch.Tensor,
    raw_cls: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_idx: int,
    candidate_label: int,
) -> int:
    candidates = torch.nonzero(labels == int(candidate_label), as_tuple=False).flatten()
    if candidates.numel() == 0:
        return -1
    sims = _cosine_matrix(query_cls.view(1, -1), raw_cls[candidates]).flatten()
    order = torch.argsort(sims, descending=True)
    target_positions = torch.nonzero(candidates[order] == int(target_idx), as_tuple=False).flatten()
    if target_positions.numel() == 0:
        return -1
    return int(target_positions[0].item() + 1)


def _pairwise_mean_distance(x: torch.Tensor) -> float:
    if x.shape[0] < 2:
        return float("nan")
    sims = _cosine_matrix(x, x)
    mask = ~torch.eye(x.shape[0], dtype=torch.bool, device=sims.device)
    return float((1.0 - sims[mask]).mean().item())


def _pairwise_distance_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Pairwise correlation inputs must have same row count, got {a.shape[0]} and {b.shape[0]}")
    if a.shape[0] < 3:
        return float("nan")
    mask = ~torch.eye(a.shape[0], dtype=torch.bool, device=a.device)
    a_dist = (1.0 - _cosine_matrix(a, a))[mask].detach().cpu().numpy().astype("float64")
    b_dist = (1.0 - _cosine_matrix(b, b))[mask].detach().cpu().numpy().astype("float64")
    if float(np.std(a_dist)) <= 1.0e-12 or float(np.std(b_dist)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a_dist, b_dist)[0, 1])


def clone_sample_state(state: Stage2State) -> Stage2State:
    if torch.is_tensor(state):
        return state.clone()
    z, aux = state
    return z.clone(), aux.clone()


def repeat_first_sample_state(state: Stage2State, n: int) -> Stage2State:
    if torch.is_tensor(state):
        return state[:1].repeat(int(n), *([1] * (state.ndim - 1)))
    z, aux = state
    return (
        z[:1].repeat(int(n), *([1] * (z.ndim - 1))),
        aux[:1].repeat(int(n), *([1] * (aux.ndim - 1))),
    )


def make_seeded_initial_state(
    *,
    n: int,
    latent_size: Tuple[int, ...],
    aux_state_spec,
    device: torch.device,
    seed: int,
    fixed_noise: bool = False,
) -> Stage2State:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    state = make_initial_sample_state(
        n=int(n),
        latent_size=latent_size,
        aux_state_spec=aux_state_spec,
        device=device,
    )
    if fixed_noise and int(n) > 1:
        state = repeat_first_sample_state(state, int(n))
    return state


def build_mode_initial_states(
    grouped_cases: Dict[str, List[Dict[str, Any]]],
    *,
    latent_size: Tuple[int, ...],
    aux_state_spec,
    device: torch.device,
    seed: int,
) -> Dict[str, Stage2State]:
    states: Dict[str, Stage2State] = {}
    shared_n = len(next(iter(grouped_cases.values()))) if grouped_cases else 0
    if shared_n > 0:
        shared = make_seeded_initial_state(
            n=shared_n,
            latent_size=latent_size,
            aux_state_spec=aux_state_spec,
            device=device,
            seed=int(seed) + 910000,
            fixed_noise=False,
        )
        for mode in COUNTERFACTUAL_MODES:
            if mode in grouped_cases:
                states[mode] = clone_sample_state(shared)

    for mode, mode_cases in grouped_cases.items():
        if mode in states:
            continue
        states[mode] = make_seeded_initial_state(
            n=len(mode_cases),
            latent_size=latent_size,
            aux_state_spec=aux_state_spec,
            device=device,
            seed=int(seed) + 920000 + len(states),
            fixed_noise=bool(mode_cases[0].get("fixed_noise", False)),
        )
    return states


def prepare_sampling_call(
    *,
    model: Stage2ModelProtocol,
    state: Stage2State,
    y: torch.Tensor,
    cls: Optional[torch.Tensor],
    device: torch.device,
    cfg_scale_class: float,
    cfg_scale_cls: float,
    cfg_interval: Tuple[float, float],
) -> SamplingCall:
    y_cond = y.to(device)
    model_kwargs: Dict[str, Any] = {"y": y_cond}
    model_fwd = model.forward
    guidance_method = "none"
    split_after_guidance = False

    if cls is not None:
        model_kwargs["cls"] = cls.to(device)

    if cls is not None and (float(cfg_scale_class) != 1.0 or float(cfg_scale_cls) != 1.0):
        model_fwd = model.forward_with_separated_cfg
        model_kwargs["cfg_scale_class"] = float(cfg_scale_class)
        model_kwargs["cfg_scale_cls"] = float(cfg_scale_cls)
        model_kwargs["cfg_interval"] = cfg_interval
        guidance_method = "separated_class_cls_cfg"
    elif cls is None and float(cfg_scale_class) != 1.0:
        null_label = int(getattr(model.y_embedder, "num_classes"))
        y_null = torch.full_like(y_cond, null_label)
        model_kwargs = {
            "y": torch.cat([y_cond, y_null], dim=0),
            "cfg_scale": float(cfg_scale_class),
            "cfg_interval": cfg_interval,
        }
        state = duplicate_state_for_guidance(state)
        model_fwd = model.forward_with_cfg
        guidance_method = "class_cfg_null_cls"
        split_after_guidance = True

    return SamplingCall(
        state=state,
        model_fwd=model_fwd,
        model_kwargs=model_kwargs,
        guidance_method=guidance_method,
        split_after_guidance=split_after_guidance,
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(out_dir: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    mode_rows = []
    for mode, values in summary["mode_summary"].items():
        mode_rows.append(
            "<tr>"
            f"<td>{html.escape(mode)}</td>"
            f"<td>{values['count']}</td>"
            f"<td>{values['target_cosine_mean']:.4f}</td>"
            f"<td>{values['comparison_cosine_mean']:.4f}</td>"
            f"<td>{values['target_minus_comparison_mean']:.4f}</td>"
            f"<td>{values['nearest_train_distance_mean']:.4f}</td>"
            f"<td>{values['nn_class_match_rate']:.3f}</td>"
            f"<td>{values['pairwise_dino_distance_mean']:.4f}</td>"
            f"<td>{values['condition_generated_pairwise_distance_corr']:.4f}</td>"
            "</tr>"
        )
    cards = []
    for row in rows:
        cards.append(
            "<figure>"
            f"<img src='{html.escape(row['image_path'])}' loading='lazy' />"
            "<figcaption>"
            f"{html.escape(row['mode'])}<br>"
            f"y={row['class_id']} cond_y={row['condition_class_id']} "
            f"cos={row['target_cosine']:.3f} "
            f"nn_y={row['nearest_train_class_id']}"
            "</figcaption>"
            "</figure>"
        )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CLS Condition Fidelity</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; background: #f7f9fb; }}
    h1 {{ margin-bottom: 4px; }}
    table {{ border-collapse: collapse; background: white; width: 100%; margin: 18px 0; }}
    th, td {{ border: 1px solid #d8dee8; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #edf2f7; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d8dee8; padding: 8px; border-radius: 6px; }}
    img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
    figcaption {{ font-size: 12px; line-height: 1.35; margin-top: 6px; }}
    code {{ background: #edf2f7; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>CLS Condition Fidelity</h1>
  <p>Checkpoint: <code>{html.escape(str(summary['checkpoint']))}</code></p>
  <p>Cache: <code>{html.escape(str(summary['cache_root']))}</code></p>
  <table>
    <thead>
      <tr>
        <th>Mode</th><th>N</th><th>Target Cosine</th><th>Comparison Cosine</th>
        <th>Delta</th><th>Nearest Train Dist</th><th>NN Class Match</th><th>Pairwise Dist</th><th>Cond-Gen Geometry Corr</th>
      </tr>
    </thead>
    <tbody>{''.join(mode_rows)}</tbody>
  </table>
  <section class="grid">{''.join(cards)}</section>
</body>
</html>
"""
    (out_dir / "report.html").write_text(doc)


def _sample_batch(
    *,
    model: Stage2ModelProtocol,
    sample_fn,
    rae: RAE,
    initial_state: Stage2State,
    y: torch.Tensor,
    cls: Optional[torch.Tensor],
    device: torch.device,
    use_bf16: bool,
    cfg_scale_class: float,
    cfg_scale_cls: float,
    cfg_interval: Tuple[float, float],
) -> Tuple[torch.Tensor, str]:
    call = prepare_sampling_call(
        model=model,
        state=clone_sample_state(initial_state),
        y=y,
        cls=cls,
        device=device,
        cfg_scale_class=float(cfg_scale_class),
        cfg_scale_cls=float(cfg_scale_cls),
        cfg_interval=cfg_interval,
    )

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            sampled_state = final_state_from_trajectory(sample_fn(call.state, call.model_fwd, **call.model_kwargs))
        if call.split_after_guidance:
            sampled_state = split_guided_state(sampled_state)
        sampled_state = sampled_state.float() if torch.is_tensor(sampled_state) else tuple(part.float() for part in sampled_state)
        images = decode_stage2_state(rae, sampled_state).clamp(0, 1).float()
    return images, call.guidance_method


def _encode_cls(rae: RAE, images: torch.Tensor, *, use_bf16: bool) -> torch.Tensor:
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            _patch, global_tokens = rae._encode_tokens(images, need_aux=True)  # noqa: SLF001
    if global_tokens is None or global_tokens.ndim != 3:
        raise ValueError("RAE did not return global tokens when re-encoding generated images.")
    return global_tokens[:, 0].detach().float()


def _mode_cases(
    raw_cls: torch.Tensor,
    std_cls: torch.Tensor,
    labels: torch.Tensor,
    source_index: torch.Tensor,
    *,
    num_per_mode: int,
    seed: int,
) -> List[Dict[str, Any]]:
    base = _choose_base_indices(labels, num_per_mode, seed)
    within, cross = _choose_within_and_cross(labels, base, seed)
    cases: List[Dict[str, Any]] = []

    def add_mode(mode: str, src_rows: Iterable[int], cond_rows: Iterable[Optional[int]], *, fixed_noise: bool = False) -> None:
        for local_id, (src, cond) in enumerate(zip(src_rows, cond_rows)):
            cond_idx = -1 if cond is None else int(cond)
            cases.append(
                {
                    "mode": mode,
                    "local_id": local_id,
                    "source_row": int(src),
                    "condition_row": cond_idx,
                    "source_index": int(source_index[src].item()),
                    "condition_source_index": -1 if cond is None else int(source_index[cond].item()),
                    "class_id": int(labels[src].item()),
                    "condition_class_id": -1 if cond is None else int(labels[cond].item()),
                    "fixed_noise": bool(fixed_noise),
                }
            )

    add_mode("paired_cls", base, base)
    add_mode("within_class_shuffled_cls", base, within)
    add_mode("cross_class_shuffled_cls", base, cross)
    add_mode("null_cls", base, [None] * len(base))

    repeated = [base[0]] * min(num_per_mode, max(2, min(num_per_mode, len(base))))
    add_mode("fixed_cls_multi_seed", repeated, repeated)

    fixed_rows = _best_fixed_noise_class(labels, num_per_mode)
    add_mode("fixed_noise_multi_cls", fixed_rows, fixed_rows, fixed_noise=True)

    return cases


def _group_cases(cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(case["mode"], []).append(case)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether a CLS-conditioned Stage-2 model obeys CLS controls.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-per-mode", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cfg-scale-class", type=float, default=1.0)
    parser.add_argument("--cfg-scale-cls", type=float, default=1.0)
    parser.add_argument(
        "--cls-input-normalization",
        choices=("auto", "raw", "train_standardize"),
        default="auto",
        help="CLS preprocessing before the model conditioner. auto resolves imagefolder->raw and latent cache->data.cls_normalization.",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CLS fidelity evaluation requires a CUDA device.")
    device = torch.device("cuda")
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested bf16, but this CUDA device does not support bf16.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    (
        _data_config,
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        _training_config,
        _eval_config,
    ) = parse_configs(config)
    if rae_config is None or model_config is None:
        raise ValueError("Config must resolve both stage_1 and stage_2 sections.")

    misc = _as_dict(misc_config)
    transport_cfg = _as_dict(transport_config)
    sampler_cfg = _as_dict(sampler_config)
    guidance_cfg = _as_dict(guidance_config)
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = int(misc.get("time_dist_shift_dim", math.prod(latent_size)))
    shift_base = int(misc.get("time_dist_shift_base", 4096))
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    raw_cls, _std_cls, labels, source_index, cache_metadata, cls_stats = _load_cache_tensors(args.cache_root)
    cls_input_normalization = infer_cls_input_normalization(config, args.cls_input_normalization)
    cls_condition_cpu = prepare_cls_for_checkpoint(raw_cls, cls_stats, cls_input_normalization)
    cases = _mode_cases(
        raw_cls,
        cls_condition_cpu,
        labels,
        source_index,
        num_per_mode=args.num_per_mode,
        seed=args.seed,
    )

    print(f"[eval] cache rows={raw_cls.shape[0]} cls_dim={raw_cls.shape[1]} modes={len(_group_cases(cases))}")
    print(f"[eval] config={args.config}")
    print(f"[eval] ckpt={args.ckpt} use_ema={args.use_ema}")
    print(f"[eval] out_dir={args.out_dir}")

    rae: RAE = instantiate_from_config(rae_config).to(device).eval()
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device).eval()
    if getattr(model, "cls_conditioner", None) is None:
        raise ValueError("Model config did not instantiate a CLS conditioner.")
    _load_stage2_weights(model, args.ckpt, use_ema=args.use_ema)

    aux_state_spec = infer_aux_state_spec(rae, latent_size=latent_size)
    transport_params = dict(transport_cfg.get("params", {}))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    sampler_mode = str(sampler_cfg.get("mode", "ODE")).upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    if args.steps is not None:
        sampler_params["num_steps"] = int(args.steps)
    if sampler_mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampler mode: {sampler_mode}")

    cfg_interval = (
        float(guidance_cfg.get("t_min", 0.0)),
        float(guidance_cfg.get("t_max", 1.0)),
    )
    raw_cls_norm = F.normalize(raw_cls.float(), dim=1)
    all_rows: List[Dict[str, Any]] = []
    generated_by_mode: Dict[str, torch.Tensor] = {}
    grouped_cases = _group_cases(cases)
    initial_states = build_mode_initial_states(
        grouped_cases,
        latent_size=latent_size,
        aux_state_spec=aux_state_spec,
        device=device,
        seed=int(args.seed),
    )
    guidance_method_by_mode: Dict[str, str] = {}
    started = time()

    for mode, mode_cases in grouped_cases.items():
        y = torch.tensor([case["class_id"] for case in mode_cases], dtype=torch.long, device=device)
        cond_rows = [int(case["condition_row"]) for case in mode_cases]
        has_cls = all(idx >= 0 for idx in cond_rows)
        cls_batch = cls_condition_cpu[cond_rows].to(device) if has_cls else None
        images, guidance_method = _sample_batch(
            model=model,
            sample_fn=sample_fn,
            rae=rae,
            initial_state=initial_states[mode],
            y=y,
            cls=cls_batch,
            device=device,
            use_bf16=use_bf16,
            cfg_scale_class=float(args.cfg_scale_class),
            cfg_scale_cls=float(args.cfg_scale_cls),
            cfg_interval=cfg_interval,
        )
        guidance_method_by_mode[mode] = guidance_method
        gen_cls = _encode_cls(rae, images, use_bf16=use_bf16).cpu()
        generated_by_mode[mode] = gen_cls
        gen_norm = F.normalize(gen_cls, dim=1)
        sims_to_train = gen_norm @ raw_cls_norm.T
        nearest_sim, nearest_idx = torch.max(sims_to_train, dim=1)
        for row_i, case in enumerate(mode_cases):
            condition_row = int(case["condition_row"])
            source_row = int(case["source_row"])
            target_idx = condition_row if condition_row >= 0 else source_row
            target_cos = float(F.cosine_similarity(gen_cls[row_i], raw_cls[target_idx], dim=0).item())
            comparison_row = source_row
            if mode == "paired_cls":
                same_rows = [j for j in _class_index(labels)[int(labels[source_row].item())] if j != source_row]
                comparison_row = same_rows[0] if same_rows else source_row
            comparison_cos = float(F.cosine_similarity(gen_cls[row_i], raw_cls[comparison_row], dim=0).item())
            nearest_row = int(nearest_idx[row_i].item())
            top10 = torch.topk(sims_to_train[row_i], k=min(10, sims_to_train.shape[1])).indices
            nn_purity = float((labels[top10] == int(case["class_id"])).float().mean().item())
            rel_image = f"images/{mode}_{row_i:02d}_src{case['source_index']}_cond{case['condition_source_index']}.png"
            save_image(images[row_i].detach().cpu(), args.out_dir / rel_image)
            all_rows.append(
                {
                    "mode": mode,
                    "row_in_mode": row_i,
                    "source_row": source_row,
                    "condition_row": condition_row,
                    "source_index": int(case["source_index"]),
                    "condition_source_index": int(case["condition_source_index"]),
                    "class_id": int(case["class_id"]),
                    "condition_class_id": int(case["condition_class_id"]),
                    "target_cosine": target_cos,
                    "comparison_cosine": comparison_cos,
                    "target_minus_comparison": target_cos - comparison_cos,
                    "nearest_train_row": nearest_row,
                    "nearest_train_source_index": int(source_index[nearest_row].item()),
                    "nearest_train_class_id": int(labels[nearest_row].item()),
                    "nearest_train_cosine": float(nearest_sim[row_i].item()),
                    "nearest_train_distance": float(1.0 - nearest_sim[row_i].item()),
                    "nearest_train_class_matches_y": int(labels[nearest_row].item() == int(case["class_id"])),
                    "top10_train_label_purity_for_y": nn_purity,
                    "target_retrieval_rank_within_condition_class": _retrieval_rank(
                        gen_cls[row_i],
                        raw_cls,
                        labels,
                        target_idx=target_idx,
                        candidate_label=int(labels[target_idx].item()),
                    ),
                    "guidance_method": guidance_method,
                    "cls_input_normalization": cls_input_normalization,
                    "shared_noise_group": "counterfactual" if mode in COUNTERFACTUAL_MODES else mode,
                    "image_path": rel_image,
                }
            )
        grid_path = image_dir / f"{mode}_grid.png"
        save_image(images.detach().cpu(), grid_path, nrow=min(4, images.shape[0]))
        print(f"[eval] mode={mode} n={images.shape[0]} grid={grid_path}")

    mode_summary: Dict[str, Any] = {}
    for mode, gen_cls in generated_by_mode.items():
        rows = [row for row in all_rows if row["mode"] == mode]
        cond_rows = [int(case["condition_row"]) for case in grouped_cases[mode]]
        valid_condition_rows = [row for row in cond_rows if row >= 0]
        geometry_corr = (
            _pairwise_distance_correlation(raw_cls[valid_condition_rows], gen_cls.cpu())
            if len(valid_condition_rows) == gen_cls.shape[0]
            else float("nan")
        )
        mode_summary[mode] = {
            "count": len(rows),
            "target_cosine_mean": float(np.mean([row["target_cosine"] for row in rows])),
            "target_cosine_std": float(np.std([row["target_cosine"] for row in rows])),
            "comparison_cosine_mean": float(np.mean([row["comparison_cosine"] for row in rows])),
            "target_minus_comparison_mean": float(np.mean([row["target_minus_comparison"] for row in rows])),
            "nearest_train_distance_mean": float(np.mean([row["nearest_train_distance"] for row in rows])),
            "nn_class_match_rate": float(np.mean([row["nearest_train_class_matches_y"] for row in rows])),
            "top10_train_label_purity_mean": float(np.mean([row["top10_train_label_purity_for_y"] for row in rows])),
            "median_target_retrieval_rank": float(np.median([row["target_retrieval_rank_within_condition_class"] for row in rows])),
            "pairwise_dino_distance_mean": _pairwise_mean_distance(gen_cls),
            "condition_generated_pairwise_distance_corr": geometry_corr,
        }

    summary = {
        "experiment_id": "g1_cls_condition_fidelity",
        "config": str(args.config),
        "checkpoint": str(args.ckpt),
        "use_ema": bool(args.use_ema),
        "cache_root": str(args.cache_root),
        "cache_num_rows": int(raw_cls.shape[0]),
        "cache_cls_dim": int(raw_cls.shape[1]),
        "cache_metadata_cls_condition": cache_metadata.get("cls_condition", {}),
        "cls_input_normalization_requested": args.cls_input_normalization,
        "cls_input_normalization": cls_input_normalization,
        "raw_cls_stats_source": cls_stats.stats_path,
        "shared_noise_across_modes": {
            "counterfactual_modes": sorted(mode for mode in COUNTERFACTUAL_MODES if mode in grouped_cases),
            "enabled": True,
            "seed": int(args.seed) + 910000,
        },
        "within_class_permutation_seed": int(args.seed) + 17,
        "guidance_method_by_mode": guidance_method_by_mode,
        "precision": args.precision,
        "seed": int(args.seed),
        "sampler_mode": sampler_mode,
        "sampler_params": sampler_params,
        "cfg_scale_class": float(args.cfg_scale_class),
        "cfg_scale_cls": float(args.cfg_scale_cls),
        "latent_size": list(latent_size),
        "aux_state_spec": {"mode": aux_state_spec.mode, "shape": None if aux_state_spec.shape is None else list(aux_state_spec.shape)},
        "elapsed_seconds": time() - started,
        "mode_summary": mode_summary,
        "leakage_declaration": "Uses only ImageNet-100 training examples from the paired tiny cache; no validation labels or validation CLS tokens are used.",
        "decoder_path": "decode_stage2_state(rae, sampled_state); target CLS is not passed directly to the decoder.",
    }
    _write_csv(args.out_dir / "metrics.csv", all_rows)
    _write_json(args.out_dir / "summary.json", summary)
    _write_html(args.out_dir, summary, all_rows)
    print(f"[eval] wrote {args.out_dir / 'summary.json'}")
    print(f"[eval] wrote {args.out_dir / 'metrics.csv'}")
    print(f"[eval] wrote {args.out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
