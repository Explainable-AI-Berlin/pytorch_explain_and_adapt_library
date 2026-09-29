#!/usr/bin/env python3
"""Pair fixed-condition RAEv2 EMA samples with their source images.

The Stage-2 trainer freezes its visualization condition from rank 0's first
microbatch in epoch 0. This script reproduces that sampler order, recovers the
cache rows' source indices, and builds side-by-side source/generated panels for
existing fixed EMA grids.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.datasets import ImageFolder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, action="append", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--grid-padding", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def center_crop_arr(image: Image.Image, image_size: int) -> Image.Image:
    """Match the ADM-style crop used by the Stage-2 cache extractor."""
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


def assign_shards_to_ranks(shard_sizes: list[int], world_size: int, batch_size: int) -> list[list[int]]:
    rank_shards = [[] for _ in range(world_size)]
    rank_batch_counts = [0] * world_size
    shard_order = sorted(
        range(len(shard_sizes)),
        key=lambda shard_idx: (-(shard_sizes[shard_idx] // batch_size), shard_idx),
    )
    for shard_idx in shard_order:
        rank = min(range(world_size), key=lambda candidate: (rank_batch_counts[candidate], candidate))
        rank_shards[rank].append(shard_idx)
        rank_batch_counts[rank] += shard_sizes[shard_idx] // batch_size
    return rank_shards


def fixed_cache_indices(
    shard_sizes: list[int],
    world_size: int,
    rank: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[int]:
    """Reproduce the first batch from ShardBatchDistributedSampler."""
    rank_shards = assign_shards_to_ranks(shard_sizes, world_size, batch_size)[rank]
    generator = torch.Generator()
    generator.manual_seed(seed + epoch * world_size + rank)
    permutation = torch.randperm(len(rank_shards), generator=generator).tolist()
    shuffled_shards = [rank_shards[index] for index in permutation]
    if not shuffled_shards:
        raise RuntimeError("Rank has no assigned cache shards.")

    first_shard = shuffled_shards[0]
    local_order = torch.randperm(shard_sizes[first_shard], generator=generator).tolist()
    if len(local_order) < batch_size:
        raise RuntimeError("First cache shard does not contain one complete microbatch.")
    offset = sum(shard_sizes[:first_shard])
    return [offset + local_idx for local_idx in local_order[:batch_size]]


def load_fixed_rows(cache_root: Path, cache_indices: list[int]) -> list[dict]:
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shards = metadata["shards"]
    cumulative: list[int] = []
    total = 0
    for entry in shards:
        total += int(entry["num_samples"])
        cumulative.append(total)

    loaded: dict[int, dict] = {}
    rows = []
    for cache_index in cache_indices:
        shard_idx = bisect.bisect_right(cumulative, cache_index)
        previous = 0 if shard_idx == 0 else cumulative[shard_idx - 1]
        local_idx = cache_index - previous
        if shard_idx not in loaded:
            loaded[shard_idx] = torch.load(cache_root / shards[shard_idx]["file"], map_location="cpu")
        payload = loaded[shard_idx]
        if "source_index" not in payload:
            raise KeyError(f"Cache shard {shards[shard_idx]['file']} has no source_index tensor.")
        cls = payload.get("cls")
        rows.append(
            {
                "cache_index": cache_index,
                "cache_shard": shards[shard_idx]["file"],
                "cache_local_index": local_idx,
                "source_index": int(payload["source_index"][local_idx]),
                "view": int(payload.get("view", torch.zeros(len(payload["y"]), dtype=torch.long))[local_idx]),
                "class_index": int(payload["y"][local_idx]),
                "cls_l2": float(cls[local_idx].float().norm()) if cls is not None else None,
            }
        )
    return rows


def extract_grid_tiles(grid_path: Path, count: int, tile_size: int, padding: int) -> list[Image.Image]:
    grid = Image.open(grid_path).convert("RGB")
    nrow = round(math.sqrt(count))
    expected_width = nrow * tile_size + (nrow + 1) * padding
    if grid.width != expected_width:
        raise ValueError(
            f"Unexpected grid width for {grid_path}: got {grid.width}, expected {expected_width}."
        )
    tiles = []
    for index in range(count):
        row, col = divmod(index, nrow)
        left = padding + col * (tile_size + padding)
        top = padding + row * (tile_size + padding)
        tiles.append(grid.crop((left, top, left + tile_size, top + tile_size)))
    return tiles


def build_reference_grid(reference_images: list[Image.Image], output_path: Path, padding: int = 2) -> None:
    count = len(reference_images)
    tile_size = reference_images[0].width
    nrow = round(math.sqrt(count))
    nrows = math.ceil(count / nrow)
    canvas = Image.new(
        "RGB",
        (nrow * tile_size + (nrow + 1) * padding, nrows * tile_size + (nrows + 1) * padding),
        "white",
    )
    for index, image in enumerate(reference_images):
        row, col = divmod(index, nrow)
        canvas.paste(image, (padding + col * (tile_size + padding), padding + row * (tile_size + padding)))
    canvas.save(output_path)


def build_pair_panel(
    reference_images: list[Image.Image],
    generated_images: list[Image.Image],
    rows: list[dict],
    output_path: Path,
) -> None:
    tile_size = reference_images[0].width
    pairs_per_row = 2
    label_height = 42
    margin = 8
    header_height = 30
    pair_width = 2 * tile_size + margin
    pair_height = tile_size + label_height + margin
    num_rows = math.ceil(len(rows) / pairs_per_row)
    canvas = Image.new(
        "RGB",
        (2 * pair_width + 3 * margin, header_height + num_rows * pair_height + margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 8), "Reference crop (CLS source) | EMA generation", fill="black")
    for index, (reference, generated, row_data) in enumerate(zip(reference_images, generated_images, rows)):
        panel_row, panel_col = divmod(index, pairs_per_row)
        x = margin + panel_col * (pair_width + margin)
        y = header_height + panel_row * pair_height
        canvas.paste(reference, (x, y))
        canvas.paste(generated, (x + tile_size + margin, y))
        caption = (
            f"#{index} source={row_data['source_index']} view={row_data['view']} "
            f"class={row_data['class_index']} {row_data['wnid']}"
        )
        draw.text((x, y + tile_size + 4), caption, fill="black")
        draw.text((x + 4, y + 4), "REF", fill="white", stroke_width=2, stroke_fill="black")
        draw.text((x + tile_size + margin + 4, y + 4), "EMA", fill="white", stroke_width=2, stroke_fill="black")
    canvas.save(output_path)


def write_manifest(rows: list[dict], output_dir: Path, settings: dict) -> None:
    manifest_json = output_dir / "fixed_condition_sources.json"
    manifest_csv = output_dir / "fixed_condition_sources.csv"
    manifest_json.write_text(
        json.dumps({"settings": settings, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.cache_root / "metadata.json").read_text(encoding="utf-8"))
    shard_sizes = [int(entry["num_samples"]) for entry in metadata["shards"]]
    cache_indices = fixed_cache_indices(
        shard_sizes,
        args.world_size,
        args.rank,
        args.micro_batch_size,
        args.sampler_seed,
        args.epoch,
    )
    rows = load_fixed_rows(args.cache_root, cache_indices)

    image_folder = ImageFolder(str(args.train_image_root))
    reference_images = []
    for row in rows:
        source_path, imagefolder_class = image_folder.samples[row["source_index"]]
        if int(imagefolder_class) != row["class_index"]:
            raise RuntimeError(
                f"Class mismatch for source {row['source_index']}: cache={row['class_index']}, "
                f"ImageFolder={imagefolder_class}."
            )
        row["relative_path"] = Path(source_path).relative_to(args.train_image_root).as_posix()
        row["wnid"] = image_folder.classes[imagefolder_class]
        reference_images.append(center_crop_arr(Image.open(source_path), args.image_size))

    settings = {
        "cache_root": str(args.cache_root.resolve()),
        "train_image_root": str(args.train_image_root.resolve()),
        "world_size": args.world_size,
        "rank": args.rank,
        "micro_batch_size": args.micro_batch_size,
        "sampler_seed": args.sampler_seed,
        "epoch": args.epoch,
        "image_size": args.image_size,
        "selection": "rank-0 first microbatch of epoch 0",
        "reference_transform": "ADM center crop; no horizontal flip",
    }

    for experiment_dir in args.experiment_dir:
        sample_dir = experiment_dir / "ema_samples"
        output_dir = sample_dir / "fixed_reference_pairs"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(rows, output_dir, settings)
        reference_grid_path = output_dir / "fixed_condition_reference_grid.png"
        if args.overwrite or not reference_grid_path.exists():
            build_reference_grid(reference_images, reference_grid_path, padding=args.grid_padding)

        fixed_grids = sorted(sample_dir.glob("epoch*_samples_fixed.png"))
        if not fixed_grids:
            raise FileNotFoundError(f"No fixed EMA grids found in {sample_dir}")
        for grid_path in fixed_grids:
            output_path = output_dir / grid_path.name.replace("_samples_fixed.png", "_reference_pairs.png")
            if output_path.exists() and not args.overwrite:
                continue
            generated_images = extract_grid_tiles(
                grid_path,
                count=len(rows),
                tile_size=args.image_size,
                padding=args.grid_padding,
            )
            build_pair_panel(reference_images, generated_images, rows, output_path)
        print(output_dir.resolve())


if __name__ == "__main__":
    main()
