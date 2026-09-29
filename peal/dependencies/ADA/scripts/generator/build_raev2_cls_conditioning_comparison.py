#!/usr/bin/env python3
"""Build reference | CLS-only | class+CLS panels for matched conditions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cls-only-sample-dir", type=Path, required=True)
    parser.add_argument("--cls-class-sample-dir", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--grid-padding", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
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


def load_manifest(sample_dir: Path) -> dict:
    path = sample_dir / "fixed_reference_pairs" / "fixed_condition_sources.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def extract_tiles(grid_path: Path, count: int, tile_size: int, padding: int) -> list[Image.Image]:
    grid = Image.open(grid_path).convert("RGB")
    nrow = round(math.sqrt(count))
    expected_width = nrow * tile_size + (nrow + 1) * padding
    if grid.width != expected_width:
        raise ValueError(f"Unexpected grid width {grid.width} for {grid_path}; expected {expected_width}")
    tiles = []
    for index in range(count):
        row, col = divmod(index, nrow)
        left = padding + col * (tile_size + padding)
        top = padding + row * (tile_size + padding)
        tiles.append(grid.crop((left, top, left + tile_size, top + tile_size)))
    return tiles


def build_panel(
    references: list[Image.Image],
    cls_only: list[Image.Image],
    cls_class: list[Image.Image],
    rows: list[dict],
    output_path: Path,
) -> None:
    tile = references[0].width
    gap = 8
    header = 54
    caption = 40
    examples_per_row = 2
    group_width = 3 * tile + 2 * gap
    group_height = tile + caption + gap
    panel_rows = math.ceil(len(rows) / examples_per_row)
    width = examples_per_row * group_width + (examples_per_row + 1) * gap
    height = header + panel_rows * group_height + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 7), "Same source image and same DINOv2-L CLS token", fill="black")
    draw.text(
        (gap, 25),
        "REF | CLS-only | CLS+class   (each run uses its own fixed diffusion-noise realization)",
        fill="black",
    )
    headings = ("REF", "CLS ONLY", "CLS + CLASS")
    for index, row_data in enumerate(rows):
        panel_row, panel_col = divmod(index, examples_per_row)
        x0 = gap + panel_col * (group_width + gap)
        y0 = header + panel_row * group_height
        images = (references[index], cls_only[index], cls_class[index])
        for column, (heading, image) in enumerate(zip(headings, images)):
            x = x0 + column * (tile + gap)
            canvas.paste(image, (x, y0))
            draw.text((x + 4, y0 + 4), heading, fill="white", stroke_width=2, stroke_fill="black")
        draw.text(
            (x0, y0 + tile + 4),
            f"#{index} source={row_data['source_index']} class={row_data['class_index']} "
            f"{row_data['wnid']} view={row_data['view']}",
            fill="black",
        )
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cls_only_manifest = load_manifest(args.cls_only_sample_dir)
    cls_class_manifest = load_manifest(args.cls_class_sample_dir)
    rows = cls_only_manifest["rows"]
    identity = [(row["source_index"], row["view"], row["class_index"]) for row in rows]
    other_identity = [
        (row["source_index"], row["view"], row["class_index"])
        for row in cls_class_manifest["rows"]
    ]
    if identity != other_identity:
        raise RuntimeError("CLS-only and class+CLS fixed-condition manifests are not identical.")

    references = [
        center_crop_arr(Image.open(args.train_image_root / row["relative_path"]), args.image_size)
        for row in rows
    ]
    cls_only_grids = {path.name: path for path in args.cls_only_sample_dir.glob("epoch*_samples_fixed.png")}
    cls_class_grids = {path.name: path for path in args.cls_class_sample_dir.glob("epoch*_samples_fixed.png")}
    common = sorted(set(cls_only_grids) & set(cls_class_grids))
    if not common:
        raise FileNotFoundError("No matched fixed EMA grid names were found.")

    outputs = []
    for name in common:
        output_path = args.output_dir / name.replace("_samples_fixed.png", "_cls_conditioning_comparison.png")
        if output_path.exists() and not args.overwrite:
            outputs.append(output_path)
            continue
        cls_only = extract_tiles(cls_only_grids[name], len(rows), args.image_size, args.grid_padding)
        cls_class = extract_tiles(cls_class_grids[name], len(rows), args.image_size, args.grid_padding)
        build_panel(references, cls_only, cls_class, rows, output_path)
        outputs.append(output_path)

    metadata = {
        "same_source_identity": True,
        "same_cls_token": True,
        "same_diffusion_noise_across_runs": False,
        "noise_note": "Each training process froze its own run-local initial diffusion noise.",
        "source_manifest": str(
            (args.cls_only_sample_dir / "fixed_reference_pairs" / "fixed_condition_sources.json").resolve()
        ),
        "outputs": [str(path.resolve()) for path in outputs],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
