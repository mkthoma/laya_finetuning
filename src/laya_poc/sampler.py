"""Length-bucketed batching (design doc §7.6.4).

Sort by token length within chunks of `chunk_mult * micro_batch`, then shuffle the batches.
Deterministic for a given seed, so a resumed run replays exactly the same order.
"""
from __future__ import annotations

import random
from typing import Sequence


def bucketed_batches(lengths: Sequence[int], micro_batch: int, seed: int, chunk_mult: int = 50) -> list[list[int]]:
    """Return batches of row indices. Every index appears exactly once."""
    if micro_batch < 1:
        raise ValueError("micro_batch must be >= 1")
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    chunk = micro_batch * chunk_mult
    batches: list[list[int]] = []
    for start in range(0, len(idx), chunk):
        ordered = sorted(idx[start:start + chunk], key=lambda i: (lengths[i], i))
        batches.extend(ordered[j:j + micro_batch] for j in range(0, len(ordered), micro_batch))
    rng.shuffle(batches)
    return batches


def padding_ratio(lengths: Sequence[int], batches: Sequence[Sequence[int]]) -> float:
    """Padded tokens / real tokens; the doc's troubleshooting target is < 1.3."""
    real = sum(lengths[i] for b in batches for i in b)
    padded = sum(max(lengths[i] for i in b) * len(b) for b in batches if b)
    return padded / real if real else 1.0
