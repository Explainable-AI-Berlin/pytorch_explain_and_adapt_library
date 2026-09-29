#!/usr/bin/env python3
"""Sample class-only, CLS-only, or class+CLS ADA RAEv2 checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cls-array", type=Path)
    parser.add_argument("--label-array", type=Path)
    parser.add_argument("--indices", default="")
    parser.add_argument("--class-ids", default="")
    parser.add_argument("--num-conditions", type=int, default=4)
    parser.add_argument("--samples-per-condition", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--cls-noise-sigma", type=float, default=0.0)
    parser.add_argument("--cls-noise-seed", type=int, default=20260817)
    parser.add_argument(
        "--cls-noise-norm-preserving",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_config(path: Path) -> Stage2Config:
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    config.prepare_model_params()
    return config


def detect_variant(config: Stage2Config) -> str:
    has_cls = bool(config.stage_2.params.get("cls_conditioning", False))
    if config.conditioning.type == "cls" and has_cls:
        return "cls_only"
    if config.conditioning.type == "label" and has_cls:
        return "cls_class"
    if config.conditioning.type == "label" and not has_cls:
        return "class_only"
    raise ValueError(
        "Unsupported conditioning combination: "
        f"type={config.conditioning.type!r}, cls_conditioning={has_cls}"
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, mmap=True, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_conditions(
    args: argparse.Namespace,
    variant: str,
) -> tuple[list[int], torch.Tensor | None, torch.Tensor]:
    indices = parse_int_list(args.indices)
    if not indices:
        indices = list(range(args.num_conditions))
    if len(indices) > args.num_conditions:
        indices = indices[: args.num_conditions]

    class_ids = parse_int_list(args.class_ids)
    labels_array = None
    if args.label_array is not None:
        labels_array = np.load(args.label_array, mmap_mode="r")

    if class_ids:
        if len(class_ids) == 1 and len(indices) > 1:
            class_ids = class_ids * len(indices)
        if len(class_ids) != len(indices):
            raise ValueError("--class-ids must contain one ID or one ID per condition")
        labels = torch.tensor(class_ids, dtype=torch.long)
    elif labels_array is not None:
        labels = torch.from_numpy(np.asarray(labels_array[indices], dtype=np.int64).copy())
    else:
        if variant == "cls_only":
            labels = torch.full((len(indices),), -1, dtype=torch.long)
        else:
            raise ValueError("Class-conditioned variants require --class-ids or --label-array")

    cls_rows = None
    if variant in {"cls_only", "cls_class"}:
        if args.cls_array is None:
            raise ValueError(f"{variant} requires --cls-array")
        cls_array = np.load(args.cls_array, mmap_mode="r")
        cls_rows = torch.from_numpy(np.asarray(cls_array[indices], dtype=np.float32).copy())
        if cls_rows.ndim != 2:
            raise ValueError(f"Expected a 2D CLS array, got {tuple(cls_rows.shape)}")
    return indices, cls_rows, labels


def perturb_cls_rows(
    rows: torch.Tensor,
    *,
    sigma: float,
    seed: int,
    row_ids: list[int],
    preserve_norm: bool,
) -> torch.Tensor:
    """Apply one reproducible Gaussian direction per CLS condition."""
    if sigma < 0:
        raise ValueError("--cls-noise-sigma must be non-negative")
    if sigma == 0:
        return rows.clone()

    perturbed = []
    if len(row_ids) != len(rows):
        raise ValueError("row_ids must align with CLS rows")
    for source_index, row in zip(row_ids, rows, strict=True):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + source_index)
        candidate = row + sigma * torch.randn(
            row.shape,
            dtype=row.dtype,
            generator=generator,
        )
        if preserve_norm:
            candidate = candidate * (
                row.norm().clamp_min(1e-8) / candidate.norm().clamp_min(1e-8)
            )
        perturbed.append(candidate)
    return torch.stack(perturbed)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def build_grid(
    rows: list[dict[str, Any]],
    images: list[Image.Image],
    *,
    samples_per_condition: int,
    output_path: Path,
    variant: str,
) -> None:
    tile = images[0].width
    gap = 8
    header = 34
    caption = 24
    columns = samples_per_condition
    row_count = len(rows) // samples_per_condition
    canvas = Image.new(
        "RGB",
        (
            columns * tile + (columns + 1) * gap,
            header + row_count * (tile + caption + gap) + gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 9), f"ADA RAEv2 EMA samples: {variant}", fill="black")
    for condition_slot in range(row_count):
        y = header + condition_slot * (tile + caption + gap)
        for repeat in range(samples_per_condition):
            flat = condition_slot * samples_per_condition + repeat
            x = gap + repeat * (tile + gap)
            canvas.paste(images[flat], (x, y))
        first = rows[condition_slot * samples_per_condition]
        draw.text(
            (gap, y + tile + 4),
            f"condition={condition_slot} index={first['source_index']} class={first['class_id']}",
            fill="black",
        )
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for sampling")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    use_bf16 = args.precision == "bf16"
    os.environ.setdefault("ADA_STAGE2_CACHE", "/tmp/unused-stage2-cache")

    config = load_config(args.config)
    variant = detect_variant(config)
    indices, original_cls_rows, labels = load_conditions(args, variant)
    if variant == "class_only" and args.cls_noise_sigma != 0:
        raise ValueError("class_only has no CLS condition to perturb")
    cls_rows = (
        perturb_cls_rows(
            original_cls_rows,
            sigma=args.cls_noise_sigma,
            seed=args.cls_noise_seed,
            row_ids=indices,
            preserve_norm=args.cls_noise_norm_preserving,
        )
        if original_cls_rows is not None
        else None
    )

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    checkpoint = load_checkpoint(args.checkpoint)
    state = checkpoint.get("ema", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    checkpoint_meta = {
        "format": checkpoint.get("format", "raev2-training-checkpoint"),
        "epoch": int(checkpoint.get("epoch", -1)),
        "step": int(checkpoint.get("step", -1)),
        "weights": "ema",
    }
    del state, checkpoint

    latent_size = tuple(int(value) for value in config.misc.latent_size)
    aux_state_spec = infer_aux_state_spec(rae, latent_size=latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sampler_config = dataclasses.asdict(config.sampler)
    sampler_config["num_steps"] = int(args.steps)
    sample_fn = sampler.sample_ode(**sampler_config)
    model_fn, sample_model_kwargs = get_model_forward_fn(model, config.guidance)
    use_guidance = config.guidance.any_guidance_active
    autocast_kwargs = (
        {"enabled": True, "dtype": torch.bfloat16}
        if use_bf16
        else {"enabled": False}
    )

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_images: list[Image.Image] = []
    output_rows: list[dict[str, Any]] = []

    for condition_slot, source_index in enumerate(indices):
        for repeat in range(args.samples_per_condition):
            sample_seed = args.seed + condition_slot * 10000 + repeat
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            initial = make_initial_sample_state(1, latent_size, aux_state_spec, device)

            cls = (
                cls_rows[condition_slot : condition_slot + 1].to(device)
                if cls_rows is not None
                else None
            )
            label = labels[condition_slot : condition_slot + 1].to(device)
            if variant == "cls_only":
                context = cls
                cls_kwarg = None
            elif variant == "cls_class":
                context = label
                cls_kwarg = cls
            else:
                context = label
                cls_kwarg = None

            sampled = sample_and_decode(
                initial,
                context,
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
                cls=cls_kwarg,
                cls_null=getattr(model, "null_cls", None),
            ).clamp(0, 1)

            cosine = None
            cosine_to_original = None
            if cls is not None:
                with torch.no_grad(), torch.cuda.amp.autocast(**autocast_kwargs):
                    _patch, generated_cls = rae.encode_with_cls(sampled.to(device))
                cosine = float(
                    F.cosine_similarity(
                        generated_cls.detach().float(),
                        cls.detach().float(),
                    ).item()
                )
                original_cls = original_cls_rows[
                    condition_slot : condition_slot + 1
                ].to(device)
                cosine_to_original = float(
                    F.cosine_similarity(
                        generated_cls.detach().float(),
                        original_cls.detach().float(),
                    ).item()
                )

            input_cls_cosine = None
            input_cls_l2 = None
            input_cls_original_norm = None
            input_cls_condition_norm = None
            if cls_rows is not None:
                original_row = original_cls_rows[condition_slot].float()
                condition_row = cls_rows[condition_slot].float()
                input_cls_cosine = float(
                    F.cosine_similarity(
                        condition_row.unsqueeze(0), original_row.unsqueeze(0)
                    ).item()
                )
                input_cls_l2 = float((condition_row - original_row).norm().item())
                input_cls_original_norm = float(original_row.norm().item())
                input_cls_condition_norm = float(condition_row.norm().item())

            image = tensor_to_pil(sampled[0])
            image_path = image_dir / (
                f"condition{condition_slot:03d}_sample{repeat:02d}.png"
            )
            image.save(image_path)
            output_images.append(image)
            output_rows.append(
                {
                    "condition_slot": condition_slot,
                    "source_index": int(source_index),
                    "class_id": int(labels[condition_slot]),
                    "repeat": repeat,
                    "seed": sample_seed,
                    "input_cls_cosine_to_original": input_cls_cosine,
                    "input_cls_l2_to_original": input_cls_l2,
                    "input_cls_original_norm": input_cls_original_norm,
                    "input_cls_condition_norm": input_cls_condition_norm,
                    "generated_cls_cosine_to_condition": cosine,
                    "generated_cls_cosine_to_original": cosine_to_original,
                    "image": str(image_path.resolve()),
                }
            )
            print(
                f"[sample] variant={variant} condition={condition_slot + 1}/{len(indices)} "
                f"repeat={repeat + 1}/{args.samples_per_condition}",
                flush=True,
            )

    grid_path = args.output_dir / "samples.png"
    build_grid(
        output_rows,
        output_images,
        samples_per_condition=args.samples_per_condition,
        output_path=grid_path,
        variant=variant,
    )
    metadata = {
        "variant": variant,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata": checkpoint_meta,
        "indices": indices,
        "class_ids": [int(value) for value in labels],
        "samples_per_condition": args.samples_per_condition,
        "steps": args.steps,
        "seed": args.seed,
        "cls_noise_sigma": args.cls_noise_sigma,
        "cls_noise_seed": args.cls_noise_seed,
        "cls_noise_norm_preserving": args.cls_noise_norm_preserving,
        "precision": args.precision,
        "cls_array": str(args.cls_array.resolve()) if args.cls_array else None,
        "label_array": str(args.label_array.resolve()) if args.label_array else None,
        "samples": output_rows,
        "grid": str(grid_path.resolve()),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(grid_path.resolve())


if __name__ == "__main__":
    main()
