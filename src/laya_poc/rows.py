"""Typed-decisions JSONL rows from split records (design doc §5.4-§5.5, §7.5; spec §1.1).

A row carries the state, question and gold as JSON *strings*, exactly what `items.build_items`
and `Agent.predict_batch` consume, plus bookkeeping (label key, country, aug tag, evidence flag).
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import labels
from .serialise import dumps, fit_state, has_evidence, make_record, order_fields

STRIPPED_SPLIT = "stripped_test"
TRAIN_SPLIT = "train"
EMPTIED_TAG = "emptied"


def records_from_split(df: pd.DataFrame, keep_fields: Iterable[str], scheme: str) -> list[tuple[dict, str]]:
    """(clean record with alphabetical keys, label key) per split row; `label` is the level-1 name."""
    keep = list(keep_fields)
    return [(make_record(row, keep), labels.l1_to_key(row["label"], scheme)) for row in df.to_dict("records")]


def _token_stats(tokens: Sequence[int]) -> dict[str, float]:
    if not tokens:
        return {"tokens_p50": 0, "tokens_p95": 0, "tokens_max": 0}
    arr = np.asarray(tokens)
    return {"tokens_p50": round(float(np.percentile(arr, 50)), 1),
            "tokens_p95": round(float(np.percentile(arr, 95)), 1), "tokens_max": int(arr.max())}


def _gold_json(cache: dict, label: str | None, scheme: str, smoothing: float) -> str:
    if label not in cache:
        cache[label] = json.dumps(labels.gold(label, scheme, smoothing), ensure_ascii=False)
    return cache[label]


def make_rows(split: str, records_with_labels: Sequence[tuple[Mapping[str, str], str | None]], *, scheme: str,
              smoothing: float, fits: Callable[[str], bool], compress_order: Sequence[str],
              address_max_chars: int, evidence_fields: Iterable[str], count_tokens: Callable[[str], int],
              rng: random.Random | None = None, shuffle: bool = False,
              aug_tags: Sequence[str] | None = None) -> tuple[list[dict], dict[str, Any]]:
    """Serialise, budget-check and wrap records as rows; returns (rows, per-split stats).

    A train row without evidence (e.g. compression removed it all, counted as `evidence_lost`) gets
    label None (uniform target) and tag `emptied` (§5.7/§5.8), so the model is never taught a class
    from `country` alone. Eval rows keep their true label, like stripped_test.
    """
    if aug_tags is not None and len(aug_tags) != len(records_with_labels):
        raise ValueError(f"aug_tags has {len(aug_tags)} entries for {len(records_with_labels)} records")
    if shuffle and rng is None:
        raise ValueError("shuffle=True needs an rng (seeded per epoch for reproducibility)")
    evidence_fields = list(evidence_fields)
    questions = json.dumps(labels.question(scheme), ensure_ascii=False)
    tags = list(aug_tags) if aug_tags is not None else ["none"] * len(records_with_labels)
    gold_cache: dict = {}
    rows, tokens, counts, aug = [], [], Counter(), Counter()
    for (rec, label), tag in zip(records_with_labels, tags, strict=True):
        ordered = order_fields(rec, rng=rng, shuffle=shuffle)
        state, compressed = fit_state(ordered, fits, compress_order, address_max_chars)
        if state is None:
            counts["rejected"] += 1
            continue
        final = json.loads(state)
        evidence = has_evidence(final, evidence_fields)
        lost = compressed and not evidence and has_evidence(ordered, evidence_fields)
        if split == TRAIN_SPLIT and not evidence and label is not None:
            label, tag = None, EMPTIED_TAG
        counts["compressed"] += int(compressed)
        counts["no_evidence"] += int(not evidence)
        counts["evidence_lost"] += int(lost)
        aug[tag] += 1
        tokens.append(count_tokens(state))
        rows.append({"id": f"{split}-{len(rows):06d}", "split": split, "state": state, "questions": questions,
                     "gold": _gold_json(gold_cache, label, scheme, smoothing), "label": label,
                     "country": final.get("country"), "aug": tag, "has_evidence": evidence})
    stats = {"split": split, "total": len(records_with_labels), "written": len(rows),
             "compressed": counts["compressed"], "rejected": counts["rejected"],
             "no_evidence": counts["no_evidence"], "evidence_lost": counts["evidence_lost"],
             **_token_stats(tokens), "aug": dict(sorted(aug.items()))}
    return rows, stats


def _substring_finder(needles: Iterable[str]) -> Callable[[str], str | None]:
    """First needle occurring in a text. Only runs of needle-alphabet characters at least as long
    as the shortest needle are scanned, so 24-char hex category ids cost almost nothing."""
    by_len: dict[int, set[str]] = {}
    for n in needles:
        if n:
            by_len.setdefault(len(n), set()).add(n)
    if not by_len:
        return lambda text: None
    alphabet = "".join(sorted({ch for group in by_len.values() for n in group for ch in n}))
    runs = re.compile(f"[{re.escape(alphabet)}]{{{min(by_len)},}}")

    def find(text: str) -> str | None:
        for m in runs.finditer(text):
            run = m.group()
            for length, group in by_len.items():
                for i in range(len(run) - length + 1):
                    if run[i:i + length] in group:
                        return run[i:i + length]
        return None

    return find


def assert_no_leakage(rows: Iterable[Mapping[str, Any]], keep_fields: Iterable[str], category_ids: set[str],
                      category_names: set[str]) -> None:
    """§5.4: states hold only keep_fields keys, no category id anywhere, no category name as a key."""
    keep, names = set(keep_fields), set(category_names)
    find_id = _substring_finder(category_ids)
    for row in rows:
        rec = json.loads(row["state"])
        if not isinstance(rec, dict):
            raise AssertionError(f"{row['id']}: state is not a JSON object")
        extra = set(rec) - keep
        if extra:
            raise AssertionError(f"{row['id']}: state keys outside keep_fields: {sorted(extra)}")
        as_key = set(rec) & names
        if as_key:
            raise AssertionError(f"{row['id']}: category name used as a JSON key: {sorted(as_key)}")
        hit = find_id(row["state"])
        if hit:
            raise AssertionError(f"{row['id']}: category id {hit} appears in the state")


def check_reject_rate(stats: Mapping[str, Any], max_rate: float) -> None:
    """§5.5: a rejection rate above max_rate blocks training until the budgets are reviewed."""
    total = stats["written"] + stats["rejected"]
    if total and stats["rejected"] / total > max_rate:
        raise ValueError(f"split {stats.get('split', '?')}: {stats['rejected']}/{total} records rejected "
                         f"({stats['rejected'] / total:.2%}) > max_reject_rate {max_rate:.2%}; review "
                         "serialise.compress_order / address_max_chars / token budgets")


def _scheme_of(option_keys: list[str]) -> str:
    for scheme in labels.SCHEMES:
        if labels.option_keys(scheme) == option_keys:
            return scheme
    raise ValueError(f"gold options {option_keys} match no label scheme")


def _uniform_gold(gold_json: str) -> str:
    probs = json.loads(gold_json)[labels.QUESTION_NAME]["probabilities"]
    return json.dumps(labels.gold(None, _scheme_of(list(probs)), 0.0), ensure_ascii=False)


def stripped_rows(test_rows: Sequence[Mapping[str, Any]], n: int, seed: int) -> list[dict]:
    """Copies of n test rows with every field but `country` removed (§5.3 stripped_test). The TRUE
    label is kept for the no-evidence metrics; the gold becomes the uniform no-evidence target."""
    n = min(int(n), len(test_rows))
    picks = sorted(random.Random(seed).sample(range(len(test_rows)), n))
    out = []
    for i, src_idx in enumerate(picks):
        src = test_rows[src_idx]
        country = json.loads(src["state"]).get("country")
        out.append({**src, "id": f"{STRIPPED_SPLIT}-{i:06d}", "split": STRIPPED_SPLIT,
                    "state": dumps({"country": country} if country else {}), "gold": _uniform_gold(src["gold"]),
                    "country": country, "aug": "stripped", "has_evidence": False})
    return out
