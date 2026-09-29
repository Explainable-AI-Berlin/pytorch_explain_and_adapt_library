from bisect import bisect_right

from data.latent_cache_dataset import ShardBatchDistributedSampler
from data.unified_dataloader import DataloaderResult


class _Dataset:
    def __init__(self, shard_sizes):
        self.shards = [{"num_samples": size} for size in shard_sizes]
        self.cumulative = []
        total = 0
        for size in shard_sizes:
            total += size
            self.cumulative.append(total)


def _batches(sampler):
    indices = list(sampler)
    size = sampler.batch_size
    return [indices[offset : offset + size] for offset in range(0, len(indices), size)]


def _batch_shard(dataset, batch):
    shard_ids = {bisect_right(dataset.cumulative, index) for index in batch}
    assert len(shard_ids) == 1
    return shard_ids.pop()


def test_shards_are_rank_local_balanced_and_contiguous():
    dataset = _Dataset([64, 64, 64, 64, 40, 40, 40, 40])
    samplers = [
        ShardBatchDistributedSampler(
            dataset,
            num_replicas=4,
            rank=rank,
            batch_size=8,
            shuffle=True,
            seed=7,
            drop_last=True,
        )
        for rank in range(4)
    ]

    assert len({len(sampler) for sampler in samplers}) == 1
    rank_shard_sets = []
    for sampler in samplers:
        sampler.set_epoch(3)
        shard_sequence = [_batch_shard(dataset, batch) for batch in _batches(sampler)]
        rank_shard_sets.append(set(shard_sequence))

        completed = set()
        previous = None
        for shard_idx in shard_sequence:
            if previous is not None and shard_idx != previous:
                completed.add(previous)
            assert shard_idx not in completed
            previous = shard_idx

    for left in range(len(rank_shard_sets)):
        for right in range(left + 1, len(rank_shard_sets)):
            assert rank_shard_sets[left].isdisjoint(rank_shard_sets[right])


def test_epoch_changes_order_without_changing_rank_assignment():
    dataset = _Dataset([64] * 12)
    sampler = ShardBatchDistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        batch_size=8,
        shuffle=True,
        seed=11,
        drop_last=True,
    )

    sampler.set_epoch(0)
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_zero != epoch_one
    assert len(epoch_zero) == len(epoch_one) == len(sampler)
    assert set(epoch_zero) == set(epoch_one)


def test_interleave_window_spreads_consecutive_batches_across_shards():
    dataset = _Dataset([64] * 8)
    sampler = ShardBatchDistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        batch_size=8,
        shuffle=False,
        seed=11,
        drop_last=True,
        shard_interleave_window=4,
    )

    shard_sequence = [_batch_shard(dataset, batch) for batch in _batches(sampler)]
    assert shard_sequence[:8] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert shard_sequence[32:40] == [4, 5, 6, 7, 4, 5, 6, 7]


def test_virtual_epoch_steps_bound_map_iteration():
    result = DataloaderResult(
        loader=list(range(100)),
        sampler=None,
        dataset_size=100,
        is_iterable=False,
        virtual_epoch_steps=7,
    )

    assert len(result) == 7
    assert list(result) == list(range(7))
