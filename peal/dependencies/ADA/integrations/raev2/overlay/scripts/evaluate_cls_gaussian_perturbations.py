#!/usr/bin/env python3
"""Sample a CLS-only RAEv2 model under controlled Gaussian CLS perturbations."""

from __future__ import annotations

import argparse
import bisect
import csv
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from configs.stage2 import Stage2Config
from stage2.state_utils import infer_aux_state_spec, make_initial_sample_state
from stage2.transport import create_sampler, create_transport
from stage2.utils import sample_and_decode, validate_stage2_config
from utils.guidance_utils import get_model_forward_fn
from utils.model_utils import instantiate_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sigmas", default="0,0.05,0.1,0.2,0.4")
    parser.add_argument("--num-sources", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--norm-preserving", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def center_crop_arr(image: Image.Image, image_size: int) -> Image.Image:
    image = image.convert("RGB")
    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.Resampling.BOX)
    scale = image_size / min(*image.size)
    image = image.resize(
        tuple(round(x * scale) for x in image.size),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(array[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def repeat_state(state, count: int):
    if torch.is_tensor(state):
        return state.repeat(int(count), *([1] * (state.ndim - 1)))
    return tuple(part.repeat(int(count), *([1] * (part.ndim - 1))) for part in state)


def load_manifest_rows(path: Path, count: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"][:count]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def load_cls_rows(cache_root: Path, rows: list[dict]) -> torch.Tensor:
    metadata = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
    shards = metadata["shards"]
    cumulative = []
    total = 0
    for entry in shards:
        total += int(entry["num_samples"])
        cumulative.append(total)
    loaded: dict[int, dict] = {}
    cls_rows = []
    for row in rows:
        cache_index = int(row["cache_index"])
        shard_idx = bisect.bisect_right(cumulative, cache_index)
        previous = 0 if shard_idx == 0 else cumulative[shard_idx - 1]
        local_idx = cache_index - previous
        if shard_idx not in loaded:
            loaded[shard_idx] = torch.load(cache_root / shards[shard_idx]["file"], map_location="cpu")
        payload = loaded[shard_idx]
        if int(payload["source_index"][local_idx]) != int(row["source_index"]):
            raise RuntimeError(f"Source identity mismatch at cache row {cache_index}")
        cls_rows.append(payload["cls"][local_idx].float())
    return torch.stack(cls_rows)


def perturb_cls(base: torch.Tensor, direction: torch.Tensor, sigmas: list[float], preserve_norm: bool) -> torch.Tensor:
    candidates = torch.stack([base + float(sigma) * direction for sigma in sigmas])
    if preserve_norm:
        base_norm = base.norm().clamp_min(1.0e-8)
        candidates = candidates * (base_norm / candidates.norm(dim=1).clamp_min(1.0e-8)).unsqueeze(1)
    return candidates


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def build_panel(
    references: list[Image.Image],
    generated: list[list[Image.Image]],
    rows: list[dict],
    sigmas: list[float],
    output_path: Path,
) -> None:
    tile = references[0].width
    gap = 8
    header = 58
    caption = 38
    columns = 1 + len(sigmas)
    width = columns * tile + (columns + 1) * gap
    height = header + len(rows) * (tile + caption + gap) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 7), "CLS-only EMA: norm-preserving Gaussian perturbations", fill="black")
    draw.text((gap, 26), "Fixed diffusion noise within each row; one fixed Gaussian direction per source", fill="black")
    headings = ["REFERENCE"] + [f"sigma={sigma:g}" for sigma in sigmas]
    for column, heading in enumerate(headings):
        draw.text((gap + column * (tile + gap), 43), heading, fill="black")
    for row_index, row_data in enumerate(rows):
        y = header + row_index * (tile + caption + gap)
        images = [references[row_index]] + generated[row_index]
        for column, image in enumerate(images):
            x = gap + column * (tile + gap)
            canvas.paste(image, (x, y))
        draw.text(
            (gap, y + tile + 4),
            f"#{row_index} source={row_data['source_index']} class={row_data['class_index']} "
            f"{row_data['wnid']}",
            fill="black",
        )
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sigmas = [float(value) for value in args.sigmas.split(",")]
    if not sigmas or sigmas[0] != 0.0:
        raise ValueError("The first sigma must be 0 for the unperturbed control.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for RAEv2 sampling.")

    device = torch.device("cuda")
    use_bf16 = args.precision == "bf16"
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config))
    )
    config.post_process()
    validate_stage2_config(config)
    if config.conditioning.type != "cls":
        raise ValueError(f"Expected a CLS-only config, got conditioning.type={config.conditioning.type!r}")
    config.prepare_model_params()

    rows = load_manifest_rows(args.source_manifest, args.num_sources)
    raw_cls = load_cls_rows(args.cache_root, rows)
    references = [
        center_crop_arr(Image.open(args.train_image_root / row["relative_path"]), config.training.image_size)
        for row in rows
    ]

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["ema"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    checkpoint_step = int(checkpoint.get("step", -1))
    del checkpoint

    latent_size = tuple(int(value) for value in config.misc.latent_size)
    aux_state_spec = infer_aux_state_spec(rae, latent_size=latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sampler_config = dataclasses.asdict(config.sampler)
    sampler_config["num_steps"] = int(args.steps)
    sample_fn = sampler.sample_ode(**sampler_config)
    model_fn, sample_model_kwargs = get_model_forward_fn(model, config.guidance)
    use_guidance = config.guidance.any_guidance_active
    autocast_kwargs = {"enabled": use_bf16, "dtype": torch.bfloat16} if use_bf16 else {"enabled": False}

    generated_by_source: list[list[Image.Image]] = []
    metric_rows = []
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for source_slot, (row, base_cpu) in enumerate(zip(rows, raw_cls)):
        direction_generator = torch.Generator(device="cpu")
        direction_generator.manual_seed(int(args.seed) + 10000 + source_slot)
        direction = torch.randn(base_cpu.shape, generator=direction_generator)
        candidates_cpu = perturb_cls(base_cpu, direction, sigmas, args.norm_preserving)

        torch.manual_seed(int(args.seed) + source_slot)
        torch.cuda.manual_seed_all(int(args.seed) + source_slot)
        initial = make_initial_sample_state(1, latent_size, aux_state_spec, device)
        initial = repeat_state(initial, len(sigmas))
        images = sample_and_decode(
            initial,
            candidates_cpu.to(device),
            None,
            eval_sampler=sample_fn,
            model_fn=model_fn,
            sample_model_kwargs=sample_model_kwargs,
            rae=rae,
            decode_rae=None,
            use_guidance=use_guidance,
            condition_type=config.conditioning.type,
            text_encoder=None,
            num_classes=config.misc.num_classes,
            device=device,
            autocast_kwargs=autocast_kwargs,
            cls_null=getattr(model, "null_cls", None),
        ).clamp(0, 1)

        with torch.no_grad(), torch.cuda.amp.autocast(**autocast_kwargs):
            _patch, reencoded_cls = rae.encode_with_cls(images.to(device))
        reencoded_cls = reencoded_cls.detach().float().cpu()
        source_images = []
        for sigma_index, (sigma, image) in enumerate(zip(sigmas, images)):
            pil_image = tensor_to_pil(image)
            image_path = image_dir / f"source{source_slot:02d}_sigma{sigma:g}.png"
            pil_image.save(image_path)
            source_images.append(pil_image)
            candidate = candidates_cpu[sigma_index]
            metric_rows.append(
                {
                    "source_slot": source_slot,
                    "source_index": int(row["source_index"]),
                    "class_index": int(row["class_index"]),
                    "wnid": row["wnid"],
                    "sigma": sigma,
                    "input_cosine_to_original": float(F.cosine_similarity(candidate[None], base_cpu[None]).item()),
                    "input_l2_to_original": float((candidate - base_cpu).norm().item()),
                    "input_norm": float(candidate.norm().item()),
                    "generated_cls_cosine_to_condition": float(
                        F.cosine_similarity(reencoded_cls[sigma_index][None], candidate[None]).item()
                    ),
                    "generated_cls_cosine_to_original": float(
                        F.cosine_similarity(reencoded_cls[sigma_index][None], base_cpu[None]).item()
                    ),
                    "image_path": str(image_path.resolve()),
                }
            )
        generated_by_source.append(source_images)
        print(f"[sample] source={source_slot + 1}/{len(rows)}", flush=True)

    panel_path = args.output_dir / "cls_only_gaussian_perturbation_panel.png"
    build_panel(references, generated_by_source, rows, sigmas, panel_path)
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    metadata = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_step": checkpoint_step,
        "weights": "ema",
        "cache_root": str(args.cache_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "sigmas": sigmas,
        "norm_preserving": args.norm_preserving,
        "fixed_diffusion_noise_within_source": True,
        "gaussian_direction_fixed_within_source": True,
        "sampler_steps": int(args.steps),
        "precision": args.precision,
        "panel": str(panel_path.resolve()),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(panel_path.resolve())


if __name__ == "__main__":
    main()
