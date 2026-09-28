"""Evaluate a Laya checkpoint on one split, before and after temperature (design doc §7.9 steps 1-4).

Per split: (1) the no-evidence gate (§5.8.1): a row whose state has no evidence field (country alone is
not evidence) is abstained WITHOUT a forward pass; (2) `predict_batch` on the compact JSON *strings*
(length-sorted batches, the per-model max_len/head_max_len the rows were built with) under a
FallbackGuard, since Laya answers a CUDA-OOM batch on CPU and then reports cuda again; (3) per-row
predictions (post-T); (4) metrics.report on the labelled, answered rows twice: POST with the
checkpoint's own temperature as loaded, then PRE with the temperatures neutralised (raw logits).
The 4-dp rounded probabilities are renormalised in option-key order and feed every metric except NLL:
NLL comes from the raw choice logits captured inside Laya's decode, as -log_softmax(z/T)[y] (critique
§7.9), so a true class that rounds to 0 costs its real -log p, not the 1e-12 clip (`nll_source` says
which; "probs" only for an agent without `_decode_answers`). `n_p_true_zero` still counts rows whose true
class rounded to 0. macro_f1 averages over every option (labels=range(K)); macro_f1_9 excludes `event`
(§5.2 headline rule).
Multi-split mode (--splits, Phase 3; evaluate_splits.py) runs every split on ONE loaded agent and adds the
val-chosen abstention threshold, the stripped-test no-evidence metrics and field-order invariance
(robustness.py, steps 5-6); OOD gaps, seed statistics and bootstrap CIs are computed from its JSON and preds.

    python -m laya_poc.evaluate --ckpt hub|<abs Laya dir> [--model laya|laya_ml] [--scheme c10|c7]
                                (--rows <split.jsonl> | --split <name> [--data-dir <dir>]) --out <json>
                                [--preds <jsonl>] [--device cuda|cpu] [--batch-size N] [--n N] [--config yaml]
    python -m laya_poc.evaluate --ckpt ... --splits <split> ... --data-dir <dir> --out-dir <dir> --preds-dir <dir>
                                [--extra name=<abs jsonl> ...] [--optional <split> ...] [--order-invariance-out ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .config import load_config, project_root
from .env_check import run_cli, write_json
from .fallback_guard import FallbackGuard
from .io_utils import iter_jsonl, write_jsonl
from .labels import SCHEMES
from .metrics import macro_f1, report
from .serialise import has_evidence

EVENT_KEY = "event"
DETAIL_KEYS = ("per_class", "cm")
DEFAULT_BATCH = 64  # design doc §7.9; config eval.batch_size overrides
RAW_LOGITS = "raw_logits"  # added to each choice answer by capture_choice_logits: {option: logit before T}


# ---------------------------------------------------------------- rows

def load_rows(path: str | Path, n: int | None = None) -> list[dict]:
    rows = list(itertools.islice(iter_jsonl(path), n))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def check_question(rows: Sequence[dict], question: dict) -> None:
    """Rows must carry the one question object of labels.question(scheme), option order included
    (a reordered criteria dict compares equal as a dict but changes every option index)."""
    order = [list(q["criteria"]) for q in question.values()]
    first_id = {}
    for r in rows:
        if "questions" in r:
            first_id.setdefault(r["questions"], r.get("id"))
    for text, row_id in first_id.items():
        got = json.loads(text)
        if got != question or [list(q["criteria"]) for q in got.values()] != order:
            raise ValueError(f"row {row_id} carries a different question (options or their order) than "
                             f"labels.question(scheme); pass the rows' --scheme, rebuild the data or fix --config")


def evidence_mask(rows: Sequence[dict], evidence_fields: Sequence[str]) -> np.ndarray:
    """True where the state has at least one evidence field (serialise.has_evidence; country is not)."""
    def one(state: str) -> bool:
        rec = json.loads(state)
        return has_evidence(rec, evidence_fields) if isinstance(rec, dict) else bool(str(rec).strip())
    return np.array([one(r["state"]) for r in rows], dtype=bool)


def label_indices(rows: Sequence[dict], keys: Sequence[str]) -> list[int | None]:
    """Option index of each row's label; None for no-evidence (null-label) rows."""
    index = {k: i for i, k in enumerate(keys)}
    bad = sorted({r["label"] for r in rows if r.get("label") is not None and r["label"] not in index})
    if bad:
        raise ValueError(f"labels {bad} are not options of the question {list(keys)}")
    return [None if r.get("label") is None else index[r["label"]] for r in rows]


# ---------------------------------------------------------------- probabilities and metrics

def probs_from_outputs(out: Sequence[dict], qid: str, keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """([N, K] renormalised probabilities in option-key order, answer_confidence [N]) from predict_batch."""
    ans = [o["answers"][qid] for o in out]
    raw = np.array([[a["probabilities"][k] for k in keys] for a in ans], dtype=float).reshape(len(ans), len(keys))
    s = raw.sum(1, keepdims=True)
    P = np.divide(raw, s, out=np.full_like(raw, np.nan), where=s > 0)
    return P, np.array([float(a["answer_confidence"]) for a in ans], dtype=float)


def logits_from_outputs(out: Sequence[dict], qid: str, keys: Sequence[str]) -> np.ndarray | None:
    """[N, K] raw choice logits in option-key order, or None when any answer lacks them (no capture)."""
    ans = [o["answers"][qid] for o in out]
    if not ans or any(RAW_LOGITS not in a for a in ans):
        return None
    return np.array([[a[RAW_LOGITS][k] for k in keys] for a in ans], dtype=float).reshape(len(ans), len(keys))


def log_softmax(Z: np.ndarray, T: float) -> np.ndarray:
    """log softmax(Z / T) over the last axis, numerically stable (NaN rows stay NaN)."""
    z = np.asarray(Z, dtype=float) / T
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def split_report(P: np.ndarray, y: np.ndarray, keys: Sequence[str], conf: np.ndarray | None = None,
                 logp: np.ndarray | None = None) -> dict:
    """metrics.report with per_class/cm moved under "detail"; macro_f1 over every option (an option absent
    from truth and predictions scores 0 rather than dropping out), macro_f1_9 (None without an event
    option), n_p_true_zero, and the NLL from `logp` ([N, K] log-probabilities) when given, else from P."""
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    rep = report(P, y, list(keys), conf)
    head = {k: v for k, v in rep.items() if k not in DETAIL_KEYS}
    yhat = P.argmax(1)
    nine = [i for i, k in enumerate(keys) if k != EVENT_KEY]
    f1_9 = macro_f1(y, yhat, labels=nine) if len(nine) < len(keys) else None
    return {**head, "macro_f1": macro_f1(y, yhat, labels=list(range(len(keys)))), "macro_f1_9": f1_9,
            **_nll_fields(head["nll"], logp, y, P.shape),
            "n_p_true_zero": int((P[np.arange(len(y)), y] <= 0).sum()), "detail": {k: rep[k] for k in DETAIL_KEYS}}


def _nll_fields(nll_probs: float, logp: np.ndarray | None, y: np.ndarray, shape: tuple) -> dict:
    """{"nll", "nll_source"}: from the log-probabilities when given ("logits"), else metrics.nll on P."""
    if logp is None:
        return {"nll": nll_probs, "nll_source": "probs"}
    logp = np.asarray(logp, dtype=float)
    if logp.shape != shape:
        raise ValueError(f"logp shape {logp.shape} does not match P {shape}")
    return {"nll": float(-logp[np.arange(len(y)), y].mean()), "nll_source": "logits"}


def applied_choice_temperature(agent: Any, k: int) -> float:
    """The T Laya divides this question's logits by: temperature_by_options[bucket] else temperature[0]
    (agent.py _decode_answers; no lang override is used here)."""
    from laya.common import QTYPES, temp_bucket

    qt = QTYPES["choice"]
    return float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt]))


# ---------------------------------------------------------------- inference

@contextmanager
def capture_choice_logits(agent: Any) -> Iterator[None]:
    """Wrap Laya's per-state `_decode_answers` so each choice answer also carries RAW_LOGITS: the logit row it
    decoded (before the temperature), keyed by option name. The answer dict lands in predict_batch's output
    at the state's input index, so no batch sort order has to be undone. No-op for an agent without it."""
    if not hasattr(agent, "_decode_answers"):
        yield
        return
    decode, had_own = agent._decode_answers, "_decode_answers" in vars(agent)

    def with_logits(logits: Any, act: Any, items: Any, ids: Any, internal: Any, offset: int,
                    *args: Any, **kwargs: Any) -> dict:
        answers = decode(logits, act, items, ids, internal, offset, *args, **kwargs)
        extra = {}
        for j, qid in enumerate(ids):
            if internal[qid]["t"] == "choice" and qid in answers:
                z = logits[offset + j, :len(items[j]["markers"])]
                extra[qid] = {**answers[qid], RAW_LOGITS: {k: float(v) for k, v in zip(internal[qid]["crit"], z)}}
        return {**answers, **extra}

    agent._decode_answers = with_logits
    try:
        yield
    finally:
        if had_own:
            agent._decode_answers = decode
        else:
            del agent._decode_answers


def predict_pass(agent: Any, states: Sequence[str], question: dict, keys: Sequence[str], *, batch_size: int,
                 max_len: int, head_max_len: int,
                 label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """(P, answer_confidence, raw logits or None, seconds) for STRING states, length-sorted batches, sparse
    progress."""
    from .parity import batch_progress

    if any(not isinstance(s, str) for s in states):
        raise TypeError("states must be the compact JSON strings from the rows, never parsed dicts")
    t0 = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(batch_progress(agent, len(states), label) if hasattr(agent, "_forward")
                            else nullcontext())
        stack.enter_context(capture_choice_logits(agent))
        out = agent.predict_batch(list(states), question, batch_size=batch_size, max_len=max_len,
                                  head_max_len=head_max_len, sort_by_length=True)
    qid = next(iter(question))
    P, conf = probs_from_outputs(out, qid, keys)
    return P, conf, logits_from_outputs(out, qid, keys), round(time.perf_counter() - t0, 2)


def _scatter(values: np.ndarray, mask: np.ndarray, width: int | None) -> np.ndarray:
    """Full-length array with `values` at the masked rows and NaN elsewhere (abstained rows)."""
    shape = (len(mask),) if width is None else (len(mask), width)
    full = np.full(shape, np.nan)
    full[mask] = values
    return full


def _score(P: np.ndarray, conf: np.ndarray, Z: np.ndarray | None, T: float, y: list[int | None],
           scored: np.ndarray, keys: Sequence[str]) -> dict | None:
    """split_report on the scored rows; the NLL from log_softmax(Z / T) when raw logits were captured."""
    if not scored.any():
        return None
    yy = np.array([v for v, s in zip(y, scored) if s], dtype=int)
    logp = None if Z is None else log_softmax(Z[scored], T)
    return split_report(P[scored], yy, keys, conf[scored], logp=logp)


def evaluate_rows(agent: Any, rows: Sequence[dict], question: dict, *, evidence_fields: Sequence[str],
                  batch_size: int, max_len: int, head_max_len: int, guard: FallbackGuard | None = None,
                  label: str = "evaluate") -> tuple[dict[str, Any], list[dict]]:
    """Both passes on one loaded agent: POST (temperatures as loaded) then PRE (neutralised; this mutates the
    agent, so it runs last). Returns (result without file metadata, post-T prediction records)."""
    from .export import neutralise_temperatures

    keys = list(question[next(iter(question))]["criteria"])
    guard = guard or FallbackGuard()
    answered, y = evidence_mask(rows, evidence_fields), label_indices(rows, keys)
    scored = answered & np.array([v is not None for v in y], dtype=bool)
    states = [r["state"] for r, a in zip(rows, answered) if a]
    temps = {"post": applied_choice_temperature(agent, len(keys)), "pre": 1.0}
    sides: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, float]] = {}
    for side in ("post", "pre"):
        if side == "pre":
            neutralise_temperatures(agent)
        if states:
            with guard:
                P, conf, Z, secs = predict_pass(agent, states, question, keys, batch_size=batch_size,
                                                max_len=max_len, head_max_len=head_max_len, label=f"{label} {side}-T")
        else:
            P, conf, Z, secs = np.zeros((0, len(keys))), np.zeros(0), None, 0.0
        sides[side] = (_scatter(P, answered, len(keys)), _scatter(conf, answered, None),
                       None if Z is None else _scatter(Z, answered, len(keys)), secs)
    res = {"n": len(rows), "n_unlabelled": int(sum(v is None for v in y)),
           "n_abstained_no_evidence": int((~answered).sum()), "n_scored": int(scored.sum()),
           "temperature": temps["post"], "temperature_pre": temps["pre"],
           **{side: _score(P, conf, Z, temps[side], y, scored, keys) for side, (P, conf, Z, _) in sides.items()},
           "seconds_post": sides["post"][3], "seconds_pre": sides["pre"][3]}
    return res, pred_records(rows, y, sides["post"][0], sides["post"][1], ~answered)


def pred_records(rows: Sequence[dict], y: Sequence[int | None], P: np.ndarray, conf: np.ndarray,
                 abstained: np.ndarray) -> list[dict]:
    """preds JSONL rows {id, y, p, answer_confidence, abstained}; abstained rows have no p / confidence."""
    return [{"id": r["id"], "y": yi, "p": None if a else [float(v) for v in p],
             "answer_confidence": None if a else float(c), "abstained": bool(a)}
            for r, yi, p, c, a in zip(rows, y, P, conf, abstained, strict=True)]


# ---------------------------------------------------------------- CLI

def rows_and_split(args: argparse.Namespace) -> tuple[Path, str]:
    """--rows wins; else <data-dir or project data>/<split>.jsonl. The split name defaults to the file stem."""
    if args.rows:
        path = Path(args.rows)
        return path, args.split or path.stem
    data_dir = Path(args.data_dir) if args.data_dir else project_root() / "data"
    return data_dir / f"{args.split}.jsonl", args.split


def load_checked(source: Any, device: str) -> Any:
    from .hub import load_agent

    agent = load_agent(source, device=device)
    if device == "cuda" and agent.device.type != "cuda":
        raise RuntimeError(f"agent landed on {agent.device.type}, not cuda (OOM at load?)")
    return agent


def eval_settings(cfg: dict, args: argparse.Namespace) -> dict[str, Any]:
    """Question, option keys, evidence fields and batching, shared by both CLI modes."""
    from .labels import question as make_question

    scheme = args.scheme or cfg["labels"]["scheme"]
    question = make_question(scheme)
    mc = cfg["model"][args.model]
    return {"scheme": scheme, "question": question, "keys": list(question[next(iter(question))]["criteria"]),
            "evidence_fields": cfg["serialise"]["evidence_fields"],
            "batch_size": args.batch_size or int(cfg.get("eval", {}).get("batch_size", DEFAULT_BATCH)),
            "max_len": int(mc["max_len"]), "head_max_len": int(mc["head_max_len"])}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from .parity import free_memory, resolve_source

    cfg = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but no CUDA device (use --device cpu for local runs)")
    source = resolve_source(cfg, args.model, args.ckpt)  # validates the path before any load
    path, split = rows_and_split(args)
    rows = load_rows(path, args.n)
    s = eval_settings(cfg, args)
    check_question(rows, s["question"])
    budget = {k: s[k] for k in ("batch_size", "max_len", "head_max_len")}
    t0, guard = time.perf_counter(), FallbackGuard()
    print(f"evaluate {args.model} ({args.ckpt}) on {split}: {len(rows)} rows, {args.device}", flush=True)
    agent = load_checked(source, args.device)
    device = agent.device.type
    res, preds = evaluate_rows(agent, rows, s["question"], evidence_fields=s["evidence_fields"], guard=guard,
                               label=f"evaluate {split}", **budget)
    del agent
    free_memory()
    if args.preds:
        write_jsonl(args.preds, preds)
    return {"model": args.model, "ckpt": args.ckpt, "split": split, "rows": str(path), "scheme": s["scheme"],
            "labels": s["keys"], **res, "cpu_fallback": guard.cpu_fallback, "device": device, **budget,
            "preds": args.preds, "seconds": round(time.perf_counter() - t0, 2)}


def _line(res: dict[str, Any]) -> str:
    def f(side: str, k: str) -> str:
        v = (res.get(side) or {}).get(k)
        return "n/a" if v is None else f"{v:.4f}"
    return (f"evaluate {res['model']} {res['split']}: macro-F1 {f('post', 'macro_f1')} (9-class "
            f"{f('post', 'macro_f1_9')}), acc {f('post', 'acc')}, ECE {f('pre', 'ece')}->{f('post', 'ece')} "
            f"at T={res['temperature']:.4f}; {res['n_scored']} scored, {res['n_unlabelled']} unlabelled, "
            f"{res['n_abstained_no_evidence']} abstained (no evidence)"
            f"{', CPU fallback during inference' if res['cpu_fallback'] else ''} ({res['seconds']:.0f}s)")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    from .evaluate_splits import add_multi_args, check_mode

    p = argparse.ArgumentParser(prog="laya_poc.evaluate", description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="'hub' (pinned revision) or an absolute Laya checkpoint dir")
    p.add_argument("--model", choices=("laya", "laya_ml"), default="laya",
                   help="config.model entry: max_len/head_max_len (and the Hub checkpoint for --ckpt hub)")
    p.add_argument("--scheme", choices=sorted(SCHEMES), default=None,
                   help="label scheme of the rows' question (default: config labels.scheme)")
    p.add_argument("--rows", default=None, help="split JSONL (e.g. <data>/val.jsonl)")
    p.add_argument("--split", default=None, help="split name: reads <data-dir>/<split>.jsonl when --rows is absent")
    p.add_argument("--data-dir", default=None, help="data dir for --split / --splits (default: <project root>/data)")
    p.add_argument("--out", default=None, help="report JSON (single-split mode, required there)")
    p.add_argument("--preds", default=None, help="per-row predictions JSONL (post-T)")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=None, help="default: config eval.batch_size")
    p.add_argument("--n", type=int, default=None, help="evaluate the first N rows only (default: all)")
    p.add_argument("--config", default=None)
    add_multi_args(p)
    args = p.parse_args(argv)
    check_mode(p, args)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.splits is not None:
        from .evaluate_splits import main_multi
        return main_multi(args)

    def body() -> int:
        res = run(args)
        write_json(args.out, res)
        print(_line(res))
        print(f"evaluate: wrote {args.out}")
        return 0

    return run_cli("evaluate", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
