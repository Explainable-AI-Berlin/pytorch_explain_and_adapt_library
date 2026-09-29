#!/usr/bin/env python3
"""Compare CLS-only and class+CLS EMA models under identical diffusion noise."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from configs.stage2 import Stage2Config
from evaluate_cls_gaussian_perturbations import (
    center_crop_arr,
    load_cls_rows,
    load_manifest_rows,
    tensor_to_pil,
)
from stage2.state_utils import infer_aux_state_spec, make_initial_sample_state
from stage2.transport import create_sampler, create_transport
from stage2.utils import sample_and_decode, validate_stage2_config
from utils.guidance_utils import get_model_forward_fn
from utils.model_utils import instantiate_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cls-only-config", type=Path, required=True)
    parser.add_argument("--cls-only-checkpoint", type=Path, required=True)
    parser.add_argument("--cls-class-config", type=Path, required=True)
    parser.add_argument("--cls-class-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-sources", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def load_config(path: Path) -> Stage2Config:
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path))
    )
    config.post_process()
    validate_stage2_config(config)
    config.prepare_model_params()
    return config


def sample_model(
    *,
    config: Stage2Config,
    checkpoint_path: Path,
    rae,
    cls_rows: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    steps: int,
    seed: int,
    use_bf16: bool,
) -> tuple[list[Image.Image], torch.Tensor, dict]:
    model = instantiate_from_config(config.stage_2).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["ema"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    checkpoint_meta = {
        "path": str(checkpoint_path.resolve()),
        "epoch": int(checkpoint.get("epoch", -1)),
        "step": int(checkpoint.get("step", -1)),
    }
    del checkpoint

    latent_size = tuple(int(value) for value in config.misc.latent_size)
    aux_state_spec = infer_aux_state_spec(rae, latent_size=latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sampler_config = dataclasses.asdict(config.sampler)
    sampler_config["num_steps"] = int(steps)
    sample_fn = sampler.sample_ode(**sampler_config)
    model_fn, sample_model_kwargs = get_model_forward_fn(model, config.guidance)
    use_guidance = config.guidance.any_guidance_active
    autocast_kwargs = (
        {"enabled": True, "dtype": torch.bfloat16}
        if use_bf16
        else {"enabled": False}
    )

    images = []
    reencoded = []
    for source_slot, (cls_cpu, label) in enumerate(zip(cls_rows, labels)):
        # Reset immediately before constructing z_0. The same seed schedule is
        # used for both models, so their initial diffusion states are identical.
        torch.manual_seed(int(seed) + source_slot)
        torch.cuda.manual_seed_all(int(seed) + source_slot)
        initial = make_initial_sample_state(1, latent_size, aux_state_spec, device)
        cls = cls_cpu[None].to(device)
        if config.conditioning.type == "cls":
            context = cls
            cls_kwarg = None
        elif config.conditioning.type == "label":
            context = label[None].to(device=device, dtype=torch.long)
            cls_kwarg = cls
        else:
            raise ValueError(
                f"Expected conditioning.type in {{'cls', 'label'}}, got "
                f"{config.conditioning.type!r}"
            )
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
        with torch.no_grad(), torch.cuda.amp.autocast(**autocast_kwargs):
            _patch, sampled_cls = rae.encode_with_cls(sampled.to(device))
        images.append(tensor_to_pil(sampled[0]))
        reencoded.append(sampled_cls[0].detach().float().cpu())
        print(
            f"[sample] condition={config.conditioning.type} "
            f"source={source_slot + 1}/{len(cls_rows)}",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()
    return images, torch.stack(reencoded), checkpoint_meta


def build_panel(
    references: list[Image.Image],
    cls_only: list[Image.Image],
    cls_class: list[Image.Image],
    rows: list[dict],
    output_path: Path,
) -> None:
    tile = references[0].width
    gap = 8
    header = 58
    caption = 38
    width = 3 * tile + 4 * gap
    height = header + len(rows) * (tile + caption + gap) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 7), "Matched EMA comparison: same source, CLS token, and diffusion noise", fill="black")
    draw.text((gap, 26), "Only the learned conditioner differs: CLS-only versus class + CLS", fill="black")
    headings = ("REFERENCE", "CLS ONLY", "CLS + CLASS")
    for column, heading in enumerate(headings):
        draw.text((gap + column * (tile + gap), 43), heading, fill="black")
    for row_index, row_data in enumerate(rows):
        y = header + row_index * (tile + caption + gap)
        for column, image in enumerate(
            (references[row_index], cls_only[row_index], cls_class[row_index])
        ):
            x = gap + column * (tile + gap)
            canvas.paste(image, (x, y))
        draw.text(
            (gap, y + tile + 4),
            f"#{row_index} source={row_data['source_index']} "
            f"class={row_data['class_index']} {row_data['wnid']}",
            fill="black",
        )
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for RAEv2 sampling.")
    device = torch.device("cuda")
    use_bf16 = args.precision == "bf16"
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cls_only_config = load_config(args.cls_only_config)
    cls_class_config = load_config(args.cls_class_config)
    if cls_only_config.conditioning.type != "cls":
        raise ValueError("The CLS-only config must use conditioning.type='cls'.")
    if cls_class_config.conditioning.type != "label":
        raise ValueError("The CLS+class config must use conditioning.type='label'.")
    if cls_only_config.misc.latent_size != cls_class_config.misc.latent_size:
        raise ValueError("The paired models must have identical latent shapes.")

    rows = load_manifest_rows(args.source_manifest, args.num_sources)
    cls_rows = load_cls_rows(args.cache_root, rows)
    labels = torch.tensor([int(row["class_index"]) for row in rows], dtype=torch.long)
    references = [
        center_crop_arr(
            Image.open(args.train_image_root / row["relative_path"]),
            cls_only_config.training.image_size,
        )
        for row in rows
    ]

    rae = instantiate_from_config(cls_only_config.stage_1).to(device).eval()
    cls_only_images, cls_only_reencoded, cls_only_checkpoint = sample_model(
        config=cls_only_config,
        checkpoint_path=args.cls_only_checkpoint,
        rae=rae,
        cls_rows=cls_rows,
        labels=labels,
        device=device,
        steps=args.steps,
        seed=args.seed,
        use_bf16=use_bf16,
    )
    cls_class_images, cls_class_reencoded, cls_class_checkpoint = sample_model(
        config=cls_class_config,
        checkpoint_path=args.cls_class_checkpoint,
        rae=rae,
        cls_rows=cls_rows,
        labels=labels,
        device=device,
        steps=args.steps,
        seed=args.seed,
        use_bf16=use_bf16,
    )

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for source_slot, (row, base) in enumerate(zip(rows, cls_rows)):
        only_path = image_dir / f"source{source_slot:02d}_cls_only.png"
        class_path = image_dir / f"source{source_slot:02d}_cls_class.png"
        cls_only_images[source_slot].save(only_path)
        cls_class_images[source_slot].save(class_path)
        metric_rows.append(
            {
                "source_slot": source_slot,
                "source_index": int(row["source_index"]),
                "class_index": int(row["class_index"]),
                "wnid": row["wnid"],
                "cls_only_reencoded_cosine": float(
                    F.cosine_similarity(cls_only_reencoded[source_slot][None], base[None]).item()
                ),
                "cls_class_reencoded_cosine": float(
                    F.cosine_similarity(cls_class_reencoded[source_slot][None], base[None]).item()
                ),
                "cls_only_image": str(only_path.resolve()),
                "cls_class_image": str(class_path.resolve()),
            }
        )

    panel_path = args.output_dir / "cls_only_vs_cls_class_same_noise_panel.png"
    build_panel(references, cls_only_images, cls_class_images, rows, panel_path)
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    metadata = {
        "cls_only_config": str(args.cls_only_config.resolve()),
        "cls_class_config": str(args.cls_class_config.resolve()),
        "cls_only_checkpoint": cls_only_checkpoint,
        "cls_class_checkpoint": cls_class_checkpoint,
        "cache_root": str(args.cache_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "same_source_identity": True,
        "same_cls_token": True,
        "same_diffusion_noise": True,
        "weights": "ema",
        "sampler_steps": int(args.steps),
        "seed": int(args.seed),
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
