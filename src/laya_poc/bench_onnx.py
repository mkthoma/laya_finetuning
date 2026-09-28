"""Laya on ONNX Runtime for the CPU benchmark (design doc §7.11): sessions with a fixed thread count, and BATCHED
inference through the PyTorch agent's own collate path (laya's ONNXAgent has no predict_batch and no thread control).

- `load_onnx_agent(ckpt_dir, onnx_path, threads)` builds laya's own `laya.onnx_agent.ONNXAgent` (tokenizer, config and
  clamped temperatures exactly as upstream loads them); only its SessionOptions get `intra_op_num_threads = threads`
  (one session, so cold start and peak RSS are those of a single session). CPU provider only.
- `onnx_predict_batch(agent, states, questions, ...)` mirrors `Agent.predict_batch` step for step (question checks,
  `_to_internal`, per-state `_encode_state`, windows of 8 batches sorted by encoded length when `sort_by_length`,
  `collate_items`, `_decode_answers` for the answers) with the forward pass on the ONNX session; batch-1 latency is the
  same function with one state. Output rows have the same shape as `Agent.predict_batch`'s.
- `onnx_forward(session, batch)` feeds a collated batch exactly as `ONNXAgent._infer` does (int64 ids / mask / marker
  positions / qtype, bool marker mask) and returns (logits, softmaxed action probabilities) as numpy.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

ONNX_MODEL_NAME = "laya-rl-agent-onnx"
CPU_PROVIDER = "CPUExecutionProvider"
OUTPUT_NAMES = ("logits", "act_logits")


# ---------------------------------------------------------------- sessions

def session_options(threads: int | None) -> Any:
    """ORT_ENABLE_ALL (as ONNXAgent sets it) and, when given, intra-op threads = `threads`, one inter-op thread."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
    return so


def make_session(onnx_path: str | Path, threads: int | None = None) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), sess_options=session_options(threads), providers=[CPU_PROVIDER])


@contextmanager
def _threaded_session_options(threads: int) -> Iterator[None]:
    """While active, `onnxruntime.SessionOptions()` returns options pinned to `threads` (ONNXAgent builds its own)."""
    import onnxruntime as ort

    original = ort.SessionOptions

    def factory() -> Any:
        so = original()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        return so

    ort.SessionOptions = factory
    try:
        yield
    finally:
        ort.SessionOptions = original


def load_onnx_agent(ckpt_dir: str | Path, onnx_path: str | Path, threads: int) -> Any:
    """laya's ONNXAgent on a local checkpoint dir (config + tokenizer) and an exported .onnx, at `threads` threads."""
    from laya.onnx_agent import ONNXAgent

    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_absolute():
        raise ValueError(f"checkpoint paths must be absolute, got {str(ckpt_dir)!r}")
    with _threaded_session_options(threads):
        agent = ONNXAgent(str(ckpt_dir), onnx_path=str(onnx_path))
    opts, providers = agent.session.get_session_options(), agent.session.get_providers()
    if opts.intra_op_num_threads != threads or providers[:1] != [CPU_PROVIDER]:
        raise RuntimeError(f"ONNX session not pinned as asked: intra_op_num_threads {opts.intra_op_num_threads} "
                           f"(want {threads}), providers {providers} (want {CPU_PROVIDER} first)")
    return agent


# ---------------------------------------------------------------- forward and batched predict

def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def onnx_inputs(batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """A collate_items batch as ONNXAgent._infer feeds it to the session."""
    return {"input_ids": batch["input_ids"].numpy().astype(np.int64),
            "attention_mask": batch["attention_mask"].numpy().astype(np.int64),
            "marker_pos": batch["marker_pos"].numpy().astype(np.int64),
            "marker_mask": batch["marker_mask"].numpy().astype(bool),
            "qtype": batch["qtype"].numpy().astype(np.int64)}


def onnx_forward(session: Any, batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(logits [rows, markers], action probabilities [rows, A]) for one collated batch."""
    logits, act_logits = session.run(list(OUTPUT_NAMES), onnx_inputs(batch))
    return logits, softmax(act_logits)


def prepare_questions(questions: Mapping[str, dict]) -> tuple[list[str], dict[str, dict]]:
    """(question ids in order, internal form), validated exactly as Agent.predict_batch does."""
    from laya.agent import Agent

    ids = list(questions.keys())
    for qid in ids:
        Agent._check_question(qid, questions[qid])
    return ids, {qid: Agent._to_internal(questions[qid]) for qid in ids}


def encode_states(agent: Any, states: Sequence[str], ids: list[str], internal: dict[str, dict],
                  **budget: int) -> list[list[dict]]:
    """Per state, its question items via the PyTorch Agent's own _encode_state (works on any object with cfg + tok)."""
    from laya.agent import Agent

    return [Agent._encode_state(agent, st, ids, internal, **budget) for st in states]


def _run_window(agent: Any, encoded: list[list[dict]], ids: list[str], internal: dict[str, dict], chunk: int,
                reorder: bool) -> list[dict[str, Any]]:
    from laya.agent import Agent
    from laya.common import collate_items

    order = list(range(len(encoded)))
    if reorder:
        order.sort(key=lambda i: max(len(item["ids"]) for item in encoded[i]))
    out: list[Any] = [None] * len(encoded)
    for offset in range(0, len(order), chunk):
        indices = order[offset:offset + chunk]
        per_state = [encoded[i] for i in indices]
        b = collate_items(per_state, agent.tok.pad_token_id)
        logits, act = onnx_forward(agent.session, b)
        row = 0
        for index, items in zip(indices, per_state):
            answers = Agent._decode_answers(agent, logits, act, items, ids, internal, row)
            n_tokens = int(b["attention_mask"][row:row + len(items)].sum())
            out[index] = {"model": ONNX_MODEL_NAME, "answers": answers,
                          "usage": {"input_tokens": n_tokens, "output_tokens": 0}}
            row += len(items)
    return out


def onnx_predict_batch(agent: Any, states: Sequence[str], questions: Mapping[str, dict], *,
                       batch_size: int | None = None, sort_by_length: bool = False, max_len: int | None = None,
                       head_max_len: int | None = None) -> list[dict[str, Any]]:
    """Agent.predict_batch's algorithm (no hooks, no lang) with the forward pass on `agent.session`."""
    if isinstance(states, (str, bytes, dict)):
        raise TypeError("onnx_predict_batch expects a list of states")
    states = list(states)
    if not states:
        return []
    ids, internal = prepare_questions(questions)
    budget = {k: v for k, v in (("max_len", max_len), ("head_max_len", head_max_len)) if v is not None}
    chunk = batch_size if (batch_size and batch_size > 0) else len(states)
    reorder = sort_by_length and 1 < chunk < len(states)
    window = chunk * 8 if reorder else chunk
    results: list[dict[str, Any]] = []
    for start in range(0, len(states), window):
        encoded = encode_states(agent, states[start:start + window], ids, internal, **budget)
        results.extend(_run_window(agent, encoded, ids, internal, chunk, reorder))
    return results
