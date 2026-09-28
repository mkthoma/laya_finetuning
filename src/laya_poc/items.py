"""Turn JSONL rows into Laya training items, token-for-token identical to the runtime path.

The state is always the compact JSON *string* stored in the row. Passing the parsed dict
instead would make Laya re-serialise it with ", " / ": " separators (common.py:33-36): about
20% more tokens and different ids from what the model was trained on.
"""
from __future__ import annotations

import json
from typing import Any

from laya.agent import Agent
from laya.common import QTYPES, build_sequence, encode_text, render_options, serialize_state

# build_sequence trims every option once the options leave fewer than this many tokens of the
# head budget for the instructions (common.py:127-131).
_MIN_HEAD_ROOM = 16
_MAX_OPTION_TOKENS = 48


def internal_question(qid: str, qdef: dict) -> dict:
    """Validate a public question and convert it exactly as Agent.predict_batch does."""
    Agent._check_question(qid, qdef)
    return Agent._to_internal(qdef)


def count_state_tokens(tok: Any, state: str) -> int:
    """Tokens the model sees for this state (before any truncation)."""
    text = serialize_state(state).replace(tok.mask_token, " ")
    return len(encode_text(tok, text, add_special_tokens=False)["input_ids"])


def state_room(tok: Any, q_internal: dict, max_len: int, head_max_len: int) -> int:
    """Exact number of state tokens kept for this question; the rest is silently dropped."""
    ref, _ = build_sequence(tok, "", q_internal, max_len, head_max_len)
    return max_len - len(ref)


def option_token_lengths(tok: Any, q_internal: dict) -> list[int]:
    """Per-option token counts including the [MASK] marker, as build_sequence computes them."""
    lengths = []
    for opt in render_options(q_internal):
        ids = encode_text(tok, " " + opt.replace(tok.mask_token, " "), add_special_tokens=False,
                          truncation=True, max_length=_MAX_OPTION_TOKENS)["input_ids"]
        lengths.append(1 + len(ids))
    return lengths


def assert_options_untrimmed(tok: Any, q_internal: dict, head_max_len: int) -> int:
    """Raise if build_sequence would trim option descriptions. Returns the total option tokens."""
    total = sum(option_token_lengths(tok, q_internal))
    if head_max_len - total < _MIN_HEAD_ROOM:
        raise ValueError(f"options need {total} tokens; head_max_len={head_max_len} leaves "
                         f"{head_max_len - total} < {_MIN_HEAD_ROOM}, so Laya would trim them")
    return total


def build_items(tok: Any, row: dict, max_len: int, head_max_len: int) -> list[dict]:
    """One training item per question in the row (recipe verified against agent._encode_state)."""
    state = row["state"]
    if not isinstance(state, str):
        raise TypeError(f"row {row.get('id')}: state must be the serialised JSON string")
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    items = []
    for qid, qdef in questions.items():
        q = internal_question(qid, qdef)
        keys = list(q["crit"].keys())
        probs = gold[qid]["probabilities"]
        if set(keys) != set(probs):
            raise ValueError(f"row {row.get('id')}: gold keys differ from criteria: {set(keys) ^ set(probs)}")
        target = [float(probs[k]) for k in keys]
        total = sum(target)
        if total <= 0:
            raise ValueError(f"row {row.get('id')}: gold probabilities sum to {total}")
        target = [v / total for v in target]
        seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
        if len(markers) != len(render_options(q)):
            raise ValueError(f"row {row.get('id')}: options exceed head_max_len={head_max_len}")
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": target,
                      "label": max(range(len(target)), key=target.__getitem__),
                      "row_id": row.get("id")})
    return items
