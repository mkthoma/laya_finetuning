import random

import pytest

from laya_poc.sampler import bucketed_batches, padding_ratio


def _lengths(n=1000, seed=0):
    r = random.Random(seed)
    return [r.randint(10, 300) for _ in range(n)]


def test_every_index_appears_exactly_once():
    lengths = _lengths()
    batches = bucketed_batches(lengths, micro_batch=8, seed=11)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))
    assert all(1 <= len(b) <= 8 for b in batches)


def test_same_seed_same_order_different_seed_different_order():
    lengths = _lengths()
    assert bucketed_batches(lengths, 8, 11) == bucketed_batches(lengths, 8, 11)
    assert bucketed_batches(lengths, 8, 11) != bucketed_batches(lengths, 8, 12)


def test_bucketing_reduces_padding_versus_random_batches():
    lengths = _lengths()
    bucketed = bucketed_batches(lengths, 8, 11)
    idx = list(range(len(lengths)))
    random.Random(0).shuffle(idx)
    unbucketed = [idx[i:i + 8] for i in range(0, len(idx), 8)]
    assert padding_ratio(lengths, bucketed) < 1.3 < padding_ratio(lengths, unbucketed)


def test_invalid_micro_batch():
    with pytest.raises(ValueError):
        bucketed_batches([1, 2], 0, 1)


def test_empty_input():
    assert bucketed_batches([], 8, 1) == []
    assert padding_ratio([], []) == 1.0
