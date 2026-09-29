from data.cache_index_order import select_cache_indices


def _selection(shard: int, *, seed: int | None):
    return select_cache_indices(
        num_samples=1000,
        max_images=0,
        subset_seed=42,
        start_index=0,
        end_index=0,
        index_shard=shard,
        num_index_shards=8,
        global_permutation_seed=seed,
    )


def test_global_permutation_is_deterministic_disjoint_and_complete():
    selections = [_selection(shard, seed=20260813) for shard in range(8)]
    source_indices = [index for selection in selections for index in selection.source_indices]
    cache_positions = [position for selection in selections for position in selection.cache_positions]

    assert source_indices == [
        index
        for shard in range(8)
        for index in _selection(shard, seed=20260813).source_indices
    ]
    assert sorted(source_indices) == list(range(1000))
    assert cache_positions == list(range(1000))
    assert source_indices != list(range(1000))


def test_permutation_happens_before_array_partitioning():
    first = _selection(0, seed=7)
    second = _selection(1, seed=7)

    assert first.cache_positions == list(range(0, 125))
    assert second.cache_positions == list(range(125, 250))
    assert max(first.source_indices) > 125
    assert min(second.source_indices) < 125
    assert set(first.source_indices).isdisjoint(second.source_indices)


def test_legacy_mode_remains_contiguous():
    selection = _selection(3, seed=None)

    assert selection.source_indices == list(range(375, 500))
    assert selection.cache_positions == list(range(375, 500))
    assert not selection.globally_permuted
