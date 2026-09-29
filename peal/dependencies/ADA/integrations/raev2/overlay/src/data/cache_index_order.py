"""Deterministic source ordering for offline Stage-2 cache extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheIndexSelection:
    """Source indices and their positions in the extraction-wide order."""

    source_indices: list[int]
    cache_positions: list[int]
    eligible_size: int
    shard_start: int
    shard_end: int
    globally_permuted: bool
    global_permutation_seed: int | None


def select_cache_indices(
    num_samples: int,
    max_images: int,
    subset_seed: int,
    start_index: int,
    end_index: int,
    index_shard: int,
    num_index_shards: int,
    global_permutation_seed: int | None = None,
) -> CacheIndexSelection:
    """Select one deterministic extraction slice.

    Legacy mode preserves the original behavior: split the class-ordered source
    range into contiguous array-task slices, then optionally select a sorted
    random subset within each slice.

    Premixed mode first globally permutes the full eligible source range and
    only then partitions that sequence across array tasks. Consecutive output
    rows are therefore drawn from the full dataset rather than adjacent
    ImageFolder classes.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    start_index = max(0, int(start_index))
    end_index = int(end_index)
    if end_index <= 0 or end_index > num_samples:
        end_index = num_samples
    if start_index >= end_index:
        raise ValueError(
            f"Invalid source-index range [{start_index}, {end_index}) "
            f"for dataset of size {num_samples}."
        )

    num_index_shards = max(1, int(num_index_shards))
    index_shard = int(index_shard)
    if num_index_shards > 1 and not 0 <= index_shard < num_index_shards:
        raise ValueError(
            f"index_shard must be in [0, {num_index_shards}) when "
            f"num_index_shards={num_index_shards}; got {index_shard}."
        )
    if num_index_shards == 1:
        index_shard = 0

    eligible_size = end_index - start_index
    ordered = torch.arange(start_index, end_index, dtype=torch.long)
    globally_permuted = global_permutation_seed is not None
    if globally_permuted:
        generator = torch.Generator().manual_seed(int(global_permutation_seed))
        ordered = ordered[torch.randperm(eligible_size, generator=generator)]

    shard_start = (eligible_size * index_shard) // num_index_shards
    shard_end = (eligible_size * (index_shard + 1)) // num_index_shards
    source_indices = ordered[shard_start:shard_end]
    cache_positions = torch.arange(shard_start, shard_end, dtype=torch.long)

    if 0 < int(max_images) < len(source_indices):
        generator = torch.Generator().manual_seed(int(subset_seed))
        chosen = torch.randperm(len(source_indices), generator=generator)[: int(max_images)]
        if globally_permuted:
            # Preserve the globally permuted stream order after selecting rows.
            chosen = torch.sort(chosen).values
        else:
            # Preserve the legacy source-sorted subset behavior.
            chosen = torch.sort(chosen).values
            source_indices = source_indices.index_select(0, chosen)
            source_sort = torch.argsort(source_indices)
            source_indices = source_indices.index_select(0, source_sort)
            cache_positions = cache_positions.index_select(0, chosen).index_select(0, source_sort)
            return CacheIndexSelection(
                source_indices=source_indices.tolist(),
                cache_positions=cache_positions.tolist(),
                eligible_size=eligible_size,
                shard_start=shard_start,
                shard_end=shard_end,
                globally_permuted=False,
                global_permutation_seed=None,
            )
        source_indices = source_indices.index_select(0, chosen)
        cache_positions = cache_positions.index_select(0, chosen)

    return CacheIndexSelection(
        source_indices=source_indices.tolist(),
        cache_positions=cache_positions.tolist(),
        eligible_size=eligible_size,
        shard_start=shard_start,
        shard_end=shard_end,
        globally_permuted=globally_permuted,
        global_permutation_seed=(int(global_permutation_seed) if globally_permuted else None),
    )
