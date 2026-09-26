"""Record serialisation and the token-budget check (design doc §5.5, §5.8, §7.5).

A record is a flat dict of cleaned FSQ fields. It reaches the model as a compact JSON *string*
(never the dict: Laya would re-serialise a dict with ", " / ": " separators, see items.py).
Field order is alphabetical for evaluation and shuffled per epoch for training.
"""
from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence

from .normalise import clean

_WEBSITE_PREFIX = re.compile(r"^(https?://)?(www\.)?", re.IGNORECASE)
_TOKEN_CACHE_SIZE = 1 << 17
NOT_EVIDENCE = frozenset({"country"})


def _field_value(field: str, value: Any) -> str | None:
    if field == "facebook_id" and isinstance(value, float) and value == value:
        # int64 ids above 2**53 lose digits as float64; extract.py casts them to VARCHAR.
        raise TypeError(f"facebook_id arrived as float ({value!r}); read it as a string (CAST AS VARCHAR)")
    v = clean(value)
    if v is not None and field == "website":
        v = _WEBSITE_PREFIX.sub("", v) or None
    return v


def order_fields(rec: Mapping[str, str], *, rng: random.Random | None = None, shuffle: bool = False) -> dict:
    """New dict with alphabetical keys, or keys shuffled with `rng` (training)."""
    items = list(rec.items())
    if shuffle:
        if rng is None:
            raise ValueError("shuffle=True needs an rng (seeded per epoch for reproducibility)")
        rng.shuffle(items)
    else:
        items.sort()
    return dict(items)


def make_record(row: Mapping[str, Any], keep_fields: Iterable[str], *, rng: random.Random | None = None,
                shuffle: bool = False) -> dict:
    """Clean record from a pool/split row (dict or pandas Series); missing and empty values dropped."""
    rec = {}
    for field in keep_fields:
        v = _field_value(field, row.get(field))
        if v is not None:
            rec[field] = v
    return order_fields(rec, rng=rng, shuffle=shuffle)


def has_evidence(rec: Mapping[str, Any], evidence_fields: Iterable[str]) -> bool:
    """True if any evidence field has a value. `country` alone is never evidence (§5.8)."""
    return any(rec.get(f) for f in evidence_fields if f not in NOT_EVIDENCE)


def dumps(rec: Mapping[str, Any]) -> str:
    """Compact JSON in the record's own key order."""
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"))


def fit_state(rec: Mapping[str, str], fits: Callable[[str], bool], compress_order: Sequence[str],
              address_max_chars: int) -> tuple[str | None, bool]:
    """(state string or None if it cannot be made to fit, whether anything was compressed).

    Compression order (§5.5): drop each compress_order field in turn, re-checking after each,
    then truncate `address`. Never mutates `rec`.
    """
    cur = dict(rec)
    state = dumps(cur)
    if fits(state):
        return state, False
    changed = False
    for field in compress_order:
        if field not in cur:
            continue
        cur = {k: v for k, v in cur.items() if k != field}
        changed = True
        state = dumps(cur)
        if fits(state):
            return state, True
    address = cur.get("address")
    if address and len(address) > address_max_chars:
        cur = {**cur, "address": address[:address_max_chars].rstrip()}
        changed = True
        state = dumps(cur)
        if fits(state):
            return state, True
    return None, changed


def token_counter(tok: Any) -> Callable[[str], int]:
    """Cached `items.count_state_tokens` for one tokenizer (fit check and stats share it)."""
    from .items import count_state_tokens  # imports laya; keep the pure helpers laya-free

    @lru_cache(maxsize=_TOKEN_CACHE_SIZE)
    def count(state: str) -> int:
        return count_state_tokens(tok, state)

    return count


def _check_budgets(budgets: Iterable[int]) -> None:
    budgets = list(budgets)
    if not budgets:
        raise ValueError("need at least one (tokenizer, budget) pair")
    bad = [b for b in budgets if int(b) <= 0]
    if bad:
        raise ValueError(f"state token budget must be positive, got {bad}")


def fits_from_counters(counters_and_budgets: Sequence[tuple[Callable[[str], int], int]]) -> Callable[[str], bool]:
    """fits(state) is True when the state is within EVERY model's budget."""
    _check_budgets(b for _, b in counters_and_budgets)
    pairs = [(count, int(b)) for count, b in counters_and_budgets]
    return lambda state: all(count(state) <= b for count, b in pairs)


def make_fits(tokenizers_and_budgets: Sequence[tuple[Any, int]]) -> Callable[[str], bool]:
    """fits(state) for a list of (tokenizer, state-token budget) pairs, one per model."""
    _check_budgets(b for _, b in tokenizers_and_budgets)
    return fits_from_counters([(token_counter(tok), b) for tok, b in tokenizers_and_budgets])


def state_budget(tok: Any, scheme: str, max_len: int, head_max_len: int, margin: int) -> int:
    """Exact state room for the scheme's question (items.state_room) minus a safety margin."""
    from . import labels
    from .items import internal_question, state_room

    qdef = labels.question(scheme)[labels.QUESTION_NAME]
    q = internal_question(labels.QUESTION_NAME, qdef)
    return state_room(tok, q, max_len, head_max_len) - int(margin)
