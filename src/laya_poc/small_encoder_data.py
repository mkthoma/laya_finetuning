"""B4 inputs and forward passes: soft targets, tokenised STATE strings, padded batches and raw logits.

Shared by the trainer (small_encoder_train.py) and the evaluation CLI (small_encoder.py). States are always the
rows' compact JSON STRINGS (never parsed dicts), encoded by the model's own tokenizer with its special tokens and
truncated at max_length; targets are the rows' gold probabilities in option-key order (label-smoothed one-hot, or
uniform for the no-evidence rows). fp16 autocast is used on CUDA only; logits always come back as float64 numpy.
"""
from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .evaluate import label_indices
from .io_utils import iter_jsonl
from .labels import QUESTION_NAME

GOLD_SUM_TOL = 1e-3


# ---------------------------------------------------------------- data

def soft_targets(rows: Sequence[dict], keys: Sequence[str]) -> np.ndarray:
    """[N, K] gold probabilities of each row's question, in option-key order (each row sums to 1)."""
    out = np.empty((len(rows), len(keys)), dtype=float)
    for i, r in enumerate(rows):
        probs = json.loads(r["gold"])[QUESTION_NAME]["probabilities"]
        if set(probs) != set(keys):
            raise ValueError(f"row {r.get('id')}: gold options {sorted(probs)} differ from the scheme's options "
                             f"{list(keys)} (pass the rows' --scheme)")
        out[i] = [probs[k] for k in keys]
    if len(out) and not np.allclose(out.sum(1), 1.0, atol=GOLD_SUM_TOL):
        raise ValueError("gold probabilities must sum to 1 per row")
    return out


def encode(tok: Any, states: Sequence[str], max_length: int) -> tuple[list[list[int]], int]:
    """(token ids per STATE string with special tokens, truncated to max_length; how many were cut)."""
    if any(not isinstance(s, str) for s in states):
        raise TypeError("states must be the compact JSON strings from the rows, never parsed dicts")
    if not states:
        return [], 0
    ids = [list(x) for x in tok(list(states))["input_ids"]]
    long = [i for i, x in enumerate(ids) if len(x) > max_length]
    if long:  # re-encode only the long rows: truncation keeps the closing special token
        cut = dict(zip(long, tok([states[i] for i in long], truncation=True, max_length=max_length)["input_ids"]))
        ids = [list(cut[i]) if i in cut else x for i, x in enumerate(ids)]
    return ids, len(long)


def collate(batch: Sequence[Sequence[int]], pad_id: int) -> tuple[Any, Any]:
    """(input_ids, attention_mask) LongTensors, right-padded to the longest row of the batch."""
    import torch

    width = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, x in enumerate(batch):
        input_ids[i, :len(x)] = torch.tensor(list(x), dtype=torch.long)
        mask[i, :len(x)] = 1
    return input_ids, mask


@dataclass(frozen=True)
class EpochData:
    ids: list[list[int]]
    targets: np.ndarray
    truncated: int


def load_epoch(path: Path, tok: Any, keys: Sequence[str], max_length: int) -> EpochData:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"no rows in {path}")
    ids, truncated = encode(tok, [r["state"] for r in rows], max_length)
    return EpochData(ids=ids, targets=soft_targets(rows, keys), truncated=truncated)


@dataclass(frozen=True)
class ValSet:
    """The labelled val rows the epoch is chosen on."""
    keys: tuple[str, ...]
    states: tuple[str, ...]
    y: np.ndarray
    targets: np.ndarray


def val_set(rows: Sequence[dict], keys: Sequence[str]) -> ValSet:
    lab = [r for r in rows if r.get("label") is not None]
    if not lab:
        raise ValueError("val has no labelled rows: B4 picks its epoch on val macro-F1")
    return ValSet(keys=tuple(keys), states=tuple(r["state"] for r in lab),
                  y=np.array(label_indices(lab, keys), dtype=int), targets=soft_targets(lab, keys))


# ---------------------------------------------------------------- forward passes

def autocast(device: str, fp16: bool) -> Any:
    import torch

    return torch.autocast("cuda", dtype=torch.float16) if fp16 and device == "cuda" else nullcontext()


def soft_ce(logits: Any, targets: Any) -> Any:
    """Mean over rows of -sum_k target_k log softmax(logits)_k, computed in fp32."""
    import torch

    return -(targets.float() * torch.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


@contextmanager
def eval_mode(model: Any) -> Iterator[None]:
    was = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was)


def score_states(model: Any, tok: Any, states: Sequence[str], *, max_length: int, batch_size: int, device: str,
                 fp16: bool) -> np.ndarray:
    """[N, K] raw logits for STATE strings in input order (length-sorted batches, eval mode, no grad)."""
    import torch

    ids, _ = encode(tok, states, max_length)
    order = sorted(range(len(ids)), key=lambda i: (len(ids[i]), i))
    out = np.zeros((len(ids), int(model.config.num_labels)), dtype=float)
    with eval_mode(model), torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            input_ids, mask = collate([ids[i] for i in idx], tok.pad_token_id)
            with autocast(device, fp16):
                logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
            out[idx] = logits.float().cpu().numpy()
    return out
