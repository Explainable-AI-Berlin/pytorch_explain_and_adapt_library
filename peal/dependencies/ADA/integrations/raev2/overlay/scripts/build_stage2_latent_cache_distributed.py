#!/usr/bin/env python3
"""Build native RAEv2 Stage-2 latent shards, optionally under torchrun.

The cache stores patch latents and, when requested, the exact public
``RAE.encode_with_cls`` global condition used by the online Stage-2 trainer.
For DINOv3 MLS K7 this is the final selected-layer patch mean packed in the
first global-token slot, not a raw DINO CLS token.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence

os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from configs.stage2 import Stage2Config
from data.cache_index_order import CacheIndexSelection, select_cache_indices
from data.imagenet_hf_dataset import ImageNetHFDataset
from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.train_utils import center_crop_arr


class IndexedViewDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Sequence[int], view: str):
        self.base = base_dataset
        self.indices = list(indices)
        self.view = str(view)
        if self.view not in {"original", "hflip"}:
            raise ValueError(f"Unsupported view={self.view!r}")

    @property
    def class_to_idx(self):
        inner = getattr(self.base, "dataset", self.base)
        return getattr(inner, "class_to_idx", getattr(self.base, "class_to_idx", {}))

    @property
    def y_vocab(self):
        class_to_idx = self.class_to_idx
        return class_to_idx if isinstance(class_to_idx, dict) else {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        source_index = int(self.indices[idx])
        image, label = self.base[source_index]
        if self.view == "hflip":
            image = TF.hflip(image)
        view_id = 0 if self.view == "original" else 1
        return image, int(label), source_index, view_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build distributed RAEv2 Stage-2 latent cache.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=0, help="<=0 means all source images.")
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive source-index start after dataset ordering.")
    parser.add_argument("--end-index", type=int, default=0, help="Exclusive source-index end; <=0 means dataset length.")
    parser.add_argument("--index-shard", type=int, default=-1, help="Contiguous source-index shard id for array jobs.")
    parser.add_argument("--num-index-shards", type=int, default=1, help="Number of contiguous source-index shards.")
    parser.add_argument("--views", default="original", help="Comma-separated: original,hflip")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--output-dtype", default="bf16", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--include-cls", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _distributed_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, world_size, local_rank, device




def _cast_tensor(tensor: torch.Tensor, output_dtype: str) -> torch.Tensor:
    output_dtype = str(output_dtype).lower()
    if output_dtype in {"float32", "fp32"}:
        return tensor.float().cpu()
    if output_dtype in {"bfloat16", "bf16"}:
        return tensor.bfloat16().cpu()
    if output_dtype in {"float16", "fp16"}:
        return tensor.half().cpu()
    raise ValueError(f"Unsupported output dtype: {output_dtype}")


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _flush_shard(
    out_dir: Path,
    rank: int,
    shard_id: int,
    z_parts: List[torch.Tensor],
    y_parts: List[torch.Tensor],
    source_parts: List[torch.Tensor],
    view_parts: List[torch.Tensor],
    cache_position_parts: List[torch.Tensor],
    relative_path_parts: List[List[str]],
    cls_parts: List[torch.Tensor],
) -> dict:
    labels = torch.cat(y_parts, dim=0).long()
    payload = {
        "z": torch.cat(z_parts, dim=0),
        "y": labels,
        "source_index": torch.cat(source_parts, dim=0).long(),
        "view": torch.cat(view_parts, dim=0).long(),
        "cache_position": torch.cat(cache_position_parts, dim=0).long(),
        "relative_path": [path for part in relative_path_parts for path in part],
    }
    if cls_parts:
        payload["cls"] = torch.cat(cls_parts, dim=0)
    filename = f"rank{rank:03d}_shard_{shard_id:06d}.pt"
    _atomic_torch_save(payload, out_dir / filename)
    class_counts = torch.bincount(labels)
    nonzero = class_counts[class_counts > 0]
    return {
        "file": filename,
        "num_samples": int(payload["z"].shape[0]),
        "num_classes": int(nonzero.numel()),
        "dominant_class_fraction": float(nonzero.max().item() / labels.numel()),
        "cache_position_min": int(payload["cache_position"].min().item()),
        "cache_position_max": int(payload["cache_position"].max().item()),
    }


def _build_base_dataset(config: Stage2Config, args: argparse.Namespace):
    data_dir = args.data_dir or Path(config.dataset.data_dir)
    split = args.split or str(config.dataset.split)
    transform = transforms.Compose([
        transforms.Lambda(partial(center_crop_arr, image_size=args.image_size)),
        transforms.ToTensor(),
    ])
    return ImageNetHFDataset(
        data_dir=str(data_dir),
        split=split,
        transform=transform,
        condition_type=config.conditioning.type,
    )


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, _local_rank, device = _distributed_setup()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "metadata.json"
    if rank == 0:
        existing_shards = sorted(args.out.glob("rank*_shard_*.pt"))
        if metadata_path.exists() or existing_shards:
            if not args.overwrite:
                raise FileExistsError(f"{args.out} already has cache outputs; pass --overwrite to rebuild.")
            for path in [
                metadata_path,
                args.out / "cls_stats.pt",
                args.out / "cls_stats.json",
                args.out / "selection_manifest.pt",
                *existing_shards,
            ]:
                if path.exists():
                    path.unlink()
            for path in args.out.glob("rank*_metadata.json"):
                path.unlink()
            for path in args.out.glob("rank*_cls_stats.pt"):
                path.unlink()
    if distributed:
        dist.barrier()

    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config))
    )
    config.post_process()

    base = _build_base_dataset(config, args)
    permutation_seed = (
        int(args.global_permutation_seed) if int(args.global_permutation_seed) >= 0 else None
    )
    selection: CacheIndexSelection = select_cache_indices(
        num_samples=len(base),
        max_images=int(args.max_images),
        subset_seed=int(args.subset_seed),
        start_index=int(args.start_index),
        end_index=int(args.end_index),
        index_shard=int(args.index_shard),
        num_index_shards=int(args.num_index_shards),
        global_permutation_seed=permutation_seed,
    )
    selected_indices = selection.source_indices
    selected_cache_positions = selection.cache_positions
    rank_indices = selected_indices[rank::world_size]
    rank_cache_positions = selected_cache_positions[rank::world_size]
    if rank == 0:
        _atomic_torch_save(
            {
                "source_index": torch.tensor(selected_indices, dtype=torch.long),
                "cache_position": torch.tensor(selected_cache_positions, dtype=torch.long),
                "global_permutation_seed": permutation_seed,
            },
            args.out / "selection_manifest.pt",
        )
    views = [view.strip() for view in str(args.views).split(",") if view.strip()]
    if not views:
        raise ValueError("--views must contain at least one view")

    rae: RAE = instantiate_from_config(config.stage_1).to(device)
    rae.eval()

    z_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    source_parts: List[torch.Tensor] = []
    view_parts: List[torch.Tensor] = []
    cache_position_parts: List[torch.Tensor] = []
    relative_path_parts: List[List[str]] = []
    cls_parts: List[torch.Tensor] = []
    shard_entries: List[dict] = []
    shard_id = 0
    pending = 0
    rank_written_samples = 0
    encode_start_time = time.monotonic()
    seen_keys = set()
    cls_sum: Optional[torch.Tensor] = None
    cls_sumsq: Optional[torch.Tensor] = None
    cls_count = 0

    with torch.inference_mode():
        for view in views:
            dataset = IndexedViewDataset(base, rank_indices, rank_cache_positions, view)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=args.num_workers > 0,
                multiprocessing_context="spawn" if args.num_workers > 0 else None,
            )
            for images, labels, source_index, view_id, cache_position, relative_path in loader:
                images = images.to(device, non_blocking=True)
                if args.include_cls:
                    z, cls = rae.encode_with_cls(images)
                else:
                    z, cls = rae.encode(images), None

                z_cpu = _cast_tensor(z, args.output_dtype)
                y_cpu = labels.cpu().long()
                source_cpu = source_index.cpu().long()
                view_cpu = view_id.cpu().long()
                cache_position_cpu = cache_position.cpu().long()
                relative_path_batch = [str(path) for path in relative_path]
                cls_cpu = _cast_tensor(cls, args.output_dtype) if cls is not None else None

                if args.include_cls:
                    if cls is None:
                        raise RuntimeError("--include-cls was set but RAE returned no global condition.")
                    cls_stats = cls.detach().float().cpu()
                    batch_sum = cls_stats.sum(dim=0, dtype=torch.float64)
                    batch_sumsq = (cls_stats.to(dtype=torch.float64) ** 2).sum(dim=0)
                    cls_sum = batch_sum if cls_sum is None else cls_sum + batch_sum
                    cls_sumsq = batch_sumsq if cls_sumsq is None else cls_sumsq + batch_sumsq
                    cls_count += int(cls_stats.shape[0])

                for source_value, view_value in zip(source_cpu.tolist(), view_cpu.tolist()):
                    key = (int(source_value), int(view_value))
                    if key in seen_keys:
                        raise ValueError(f"Duplicate source_index/view key on rank {rank}: {key}")
                    seen_keys.add(key)

                start = 0
                while start < z_cpu.shape[0]:
                    room = args.shard_size - pending
                    end = min(z_cpu.shape[0], start + room)
                    z_parts.append(z_cpu[start:end])
                    y_parts.append(y_cpu[start:end])
                    source_parts.append(source_cpu[start:end])
                    view_parts.append(view_cpu[start:end])
                    cache_position_parts.append(cache_position_cpu[start:end])
                    relative_path_parts.append(relative_path_batch[start:end])
                    if cls_cpu is not None:
                        cls_parts.append(cls_cpu[start:end])
                    pending += end - start
                    start = end
                    if pending >= args.shard_size:
                        entry = _flush_shard(
                            args.out,
                            rank,
                            shard_id,
                            z_parts,
                            y_parts,
                            source_parts,
                            view_parts,
                            cache_position_parts,
                            relative_path_parts,
                            cls_parts,
                        )
                        shard_entries.append(entry)
                        rank_written_samples += int(entry["num_samples"])
                        if rank == 0 or shard_id == 0 or shard_id % 25 == 0:
                            elapsed = max(time.monotonic() - encode_start_time, 1.0e-6)
                            print(
                                f"[rank {rank}] wrote shard {shard_id:05d}; "
                                f"rank_samples={rank_written_samples}; "
                                f"elapsed={elapsed / 60.0:.2f}m; "
                                f"rate={rank_written_samples / elapsed:.2f} samples/s",
                                flush=True,
                            )
                        shard_id += 1
                        z_parts, y_parts, source_parts, view_parts = [], [], [], []
                        cache_position_parts, relative_path_parts = [], []
                        cls_parts = []
                        pending = 0

    if pending > 0:
        entry = _flush_shard(
            args.out,
            rank,
            shard_id,
            z_parts,
            y_parts,
            source_parts,
            view_parts,
            cache_position_parts,
            relative_path_parts,
            cls_parts,
        )
        shard_entries.append(entry)
        rank_written_samples += int(entry["num_samples"])
        elapsed = max(time.monotonic() - encode_start_time, 1.0e-6)
        print(
            f"[rank {rank}] wrote final shard {shard_id:05d}; "
            f"rank_samples={rank_written_samples}; "
            f"elapsed={elapsed / 60.0:.2f}m; "
            f"rate={rank_written_samples / elapsed:.2f} samples/s",
            flush=True,
        )

    rank_meta_path = args.out / f"rank{rank:03d}_metadata.json"
    _atomic_json_save({"rank": rank, "num_samples": sum(e["num_samples"] for e in shard_entries), "shards": shard_entries}, rank_meta_path)
    if args.include_cls:
        if cls_sum is None or cls_sumsq is None or cls_count <= 0:
            raise RuntimeError(f"Rank {rank} accumulated no CLS/global rows.")
        _atomic_torch_save(
            {"sum": cls_sum, "sumsq": cls_sumsq, "count": torch.tensor(cls_count, dtype=torch.long)},
            args.out / f"rank{rank:03d}_cls_stats.pt",
        )

    if distributed:
        dist.barrier()

    if rank == 0:
        rank_metas = []
        for r in range(world_size):
            with (args.out / f"rank{r:03d}_metadata.json").open("r", encoding="utf-8") as handle:
                rank_metas.append(json.load(handle))
        all_shards = []
        for meta in rank_metas:
            all_shards.extend(meta["shards"])
        all_shards = sorted(all_shards, key=lambda entry: entry["file"])

        cls_condition = {"included": bool(args.include_cls)}
        if args.include_cls:
            total_sum = None
            total_sumsq = None
            total_count = 0
            for r in range(world_size):
                stats = torch.load(args.out / f"rank{r:03d}_cls_stats.pt", map_location="cpu")
                total_sum = stats["sum"] if total_sum is None else total_sum + stats["sum"]
                total_sumsq = stats["sumsq"] if total_sumsq is None else total_sumsq + stats["sumsq"]
                total_count += int(stats["count"].item())
            mean = (total_sum / float(total_count)).to(dtype=torch.float32)
            var = (total_sumsq / float(total_count) - mean.to(dtype=torch.float64) ** 2).clamp_min(0).to(dtype=torch.float32)
            std = torch.sqrt(var + 1.0e-6)
            _atomic_torch_save(
                {
                    "mean": mean,
                    "var": var,
                    "std": std,
                    "count": torch.tensor(total_count, dtype=torch.long),
                    "normalization": "train_standardize",
                    "source": "public_rae_encode_with_cls",
                },
                args.out / "cls_stats.pt",
            )
            _atomic_json_save(
                {
                    "count": int(total_count),
                    "dim": int(mean.numel()),
                    "mean_abs_avg": float(mean.abs().mean().item()),
                    "std_avg": float(std.mean().item()),
                    "std_min": float(std.min().item()),
                    "std_max": float(std.max().item()),
                    "normalization": "train_standardize",
                },
                args.out / "cls_stats.json",
            )
            encoder = getattr(rae, "encoder", None)
            encoder_class = encoder.__class__.__name__ if encoder is not None else "unknown"
            cls_condition = {
                "included": True,
                "shape": [int(mean.numel())],
                "raw": True,
                "token": "RAE.encode_with_cls()[1]",
                "normalization": "raw_for_training; train_standardize_stats_recorded",
                "stats_file": "cls_stats.pt",
                "stats_json": "cls_stats.json",
                "count": int(total_count),
                "condition_source": (
                    "final_selected_layer_patch_mean"
                    if encoder_class == "DINOv3MultiLayerSimpleAddEncoder"
                    else "encoder_forward_with_global_slot0"
                ),
                "encoder_class": encoder_class,
                "selected_layers": getattr(encoder, "layer_indices", None),
            }

        class_to_idx = getattr(getattr(base, "dataset", base), "class_to_idx", {})
        metadata = {
            "builder": "external/RAEv2/scripts/build_stage2_latent_cache_distributed.py",
            "config": str(args.config.expanduser().resolve()),
            "data_dir": str((args.data_dir or Path(config.dataset.data_dir)).expanduser().resolve()),
            "split": args.split or str(config.dataset.split),
            "image_size": int(args.image_size),
            "num_source_images": int(len(selected_indices)),
            "num_samples": int(sum(entry["num_samples"] for entry in all_shards)),
            "views": views,
            "selection_mode": (
                "global_permutation"
                if selection.globally_permuted and args.max_images <= 0
                else "global_permutation_random_subset"
                if selection.globally_permuted
                else "all" if args.max_images <= 0 else "random_subset"
            ),
            "selection": {
                "dataset_size": int(len(base)),
                "start_index": int(args.start_index),
                "end_index": int(args.end_index) if int(args.end_index) > 0 else int(len(base)),
                "index_shard": int(args.index_shard),
                "num_index_shards": int(args.num_index_shards),
                "selected_min_index": int(min(selected_indices)) if selected_indices else None,
                "selected_max_index": int(max(selected_indices)) if selected_indices else None,
                "eligible_size": int(selection.eligible_size),
                "cache_position_start": int(selection.shard_start),
                "cache_position_end": int(selection.shard_end),
                "global_permutation_seed": selection.global_permutation_seed,
                "permutation_applied_before_index_partition": bool(selection.globally_permuted),
                "selection_manifest": "selection_manifest.pt",
            },
            "subset_seed": int(args.subset_seed),
            "max_images": int(args.max_images),
            "row_provenance": {
                "source_index": "payload tensor",
                "relative_path": "payload string list",
                "label": "payload y tensor",
                "view": "payload tensor",
                "cache_position": "payload tensor",
            },
            "output_dtype": str(args.output_dtype),
            "distributed_world_size": int(world_size),
            "latent_size": list(config.misc.latent_size),
            "physical_mixing": {
                "globally_permuted_before_extraction": bool(selection.globally_permuted),
                "mean_classes_per_shard": float(sum(entry["num_classes"] for entry in all_shards) / len(all_shards)),
                "mean_dominant_class_fraction": float(sum(entry["dominant_class_fraction"] for entry in all_shards) / len(all_shards)),
            },
            "cls_condition": cls_condition,
            "class_to_idx": {str(k): int(v) for k, v in class_to_idx.items()},
            "y_vocab": {str(k): int(v) for k, v in class_to_idx.items()},
            "shards": all_shards,
        }
        _atomic_json_save(metadata, metadata_path)
        print(
            f"Wrote {metadata['num_samples']} cached RAEv2 Stage-2 samples "
            f"from {metadata['num_source_images']} source images to {args.out}",
            flush=True,
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
