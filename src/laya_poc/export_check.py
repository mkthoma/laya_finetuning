"""Calibration + export round trip on a fine-tuned checkpoint (design doc §5.10/§7.8, fixed per critique).

Fixes over the doc: T is fitted on log_softmax of the RAW logits from the exact predict_batch code path
(predict probabilities are rounded to 4 dp, so the confidently-wrong rows that set T would become log(1e-12));
T is written to temperature[0] with temperature_by_options removed (keys like "choice:10" are dead;
the runtime bucket is "choice:6-10"); and the reloaded agent is checked to apply exactly that T.
Cross-checks against training (critique §C smoke assertions): the saved max_len/head_max_len equal the
build values in config.model, and the checkpoint's val accuracy matches train_single's final_eval
(<run>/summary.json beside <run>/final). Either is skipped, not failed, when there is nothing to compare.
Every inference call runs under a FallbackGuard: a batch Laya answered on CPU after a CUDA OOM fails the check.

    python -m laya_poc.export_check --ckpt <abs final dir> --rows <val.jsonl> --out <json>
                                    [--device cuda|cpu] [--n N] [--config yaml] [--model laya|laya_ml]
                                    [--train-summary <summary.json>]

Exit code is 0 even when a check fails: the smoke report decides. It prints PASS/FAIL.
NOTE: this rewrites <ckpt>/rl_agent_config.json in place (that is the export step being tested).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calibrate import fit_temperature, is_clamped, log_softmax, softmax
from .config import load_config
from .env_check import run_cli, write_json
from .fallback_guard import FallbackGuard
from .io_utils import iter_jsonl
from .metrics import report
from .parity import checkpoint_dir, free_memory, probs_matrix
from .zeroshot import select_rows, shared_question

ROUNDTRIP_N = 20
ROUNDTRIP_MAX_DP = 1e-3   # predict_batch rounds to 4 dp; anything larger means T is not what we wrote
NLL_TOLERANCE = 1e-6
BUDGET_KEYS = ("max_len", "head_max_len")
SUMMARY_NAME = "summary.json"  # train_single writes <run>/summary.json beside <run>/final
TRAIN_ACC_TOL_ROWS = 2         # fp16 export and other padding can flip a near-tied argmax or two
TRAIN_ACC_TOL_FRAC = 0.005     # ... and proportionally more on a large val set
_MODEL_NAME_RE = re.compile(r"^laya-poc-(?P<model>.+)-s\d+$")  # train_single's model_name


def raw_logits(agent: Any, states: Sequence[str], question: dict, batch_size: int = 32,
               max_len: int | None = None, head_max_len: int | None = None) -> np.ndarray:
    """[N, K] pre-temperature logits through the same private path predict_batch uses (laya-api §6b)."""
    import torch
    from laya.common import collate_items

    if any(not isinstance(s, str) for s in states):
        raise TypeError("states must be the compact JSON strings from the rows, never parsed dicts")
    ids = list(question)
    if len(ids) != 1:
        raise ValueError(f"expected a single question, got {ids}")
    for qid in ids:
        agent._check_question(qid, question[qid])
    internal = {qid: agent._to_internal(question[qid]) for qid in ids}
    rows = []
    with torch.no_grad():
        for s in range(0, len(states), batch_size):
            enc = [agent._encode_state(st, ids, internal, max_len=max_len, head_max_len=head_max_len)
                   for st in states[s:s + batch_size]]
            logits, _ = agent._forward(collate_items(enc, agent.tok.pad_token_id))
            rows.extend(logits[r, :len(items[0]["markers"])] for r, items in enumerate(enc))
    return np.asarray(rows, dtype=np.float64)


def fit_logit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """T for raw logits, fitted on their log_softmax: shift-invariant like softmax itself. calibrate floors
    its input at log(1e-12), a floor meant for log-probabilities that would otherwise make T depend on the
    head's arbitrary logit offset."""
    return fit_temperature(log_softmax(logits), y)


def label_indices(rows: Sequence[dict], keys: Sequence[str]) -> np.ndarray:
    index = {k: i for i, k in enumerate(keys)}
    bad = sorted({r["label"] for r in rows if r["label"] not in index})
    if bad:
        raise ValueError(f"labels {bad} are not options of the question {list(keys)}")
    return np.array([index[r["label"]] for r in rows])


def write_temperature(ckpt_dir: Path, T: float) -> dict:
    """The export step under test (owned by laya_poc.export)."""
    from .export import write_choice_temperature
    return write_choice_temperature(ckpt_dir, T)


def headline(P: np.ndarray, y: np.ndarray, keys: Sequence[str]) -> dict[str, Any]:
    return {k: v for k, v in report(P, y, list(keys)).items() if k not in ("per_class", "cm")}


def load_on(ckpt: Path, device: str) -> Any:
    from .hub import load_agent

    agent = load_agent(ckpt, device=device)
    if device == "cuda" and agent.device.type != "cuda":
        raise RuntimeError(f"agent landed on {agent.device.type}, not cuda (OOM at load?)")
    return agent


def fit_on_checkpoint(ckpt: Path, rows: list[dict], question: dict, keys: list[str], device: str,
                      batch_size: int, *, guard: FallbackGuard) -> tuple[np.ndarray, np.ndarray, float]:
    """(raw logits, labels, fitted T) with the checkpoint's own max_len/head_max_len."""
    agent = load_on(ckpt, device)
    with guard:
        logits = raw_logits(agent, [r["state"] for r in rows], question, batch_size)
    del agent
    free_memory()
    y = label_indices(rows, keys)
    return logits, y, fit_logit_temperature(logits, y)


def check_reload(ckpt: Path, rows: list[dict], question: dict, keys: list[str], T: float,
                 device: str, *, guard: FallbackGuard) -> dict[str, Any]:
    """Reload after the write-back: T is in the config verbatim and predict_batch applies it."""
    agent = load_on(ckpt, device)
    config_ok = bool(math.isclose(float(agent.temperature_raw[0]), T, rel_tol=1e-9, abs_tol=0.0)
                     and agent.temperature_by_options_raw == {})
    T_applied = float(agent.temperature[0])  # what the runtime uses after its [0.5, 5] clamp
    states = [r["state"] for r in rows[:ROUNDTRIP_N]]
    budget = (agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192))
    with guard:
        z = raw_logits(agent, states, question, batch_size=len(states))  # same batch as predict_batch below
        p_pred = probs_matrix(agent, states, question, keys, len(states), *budget)
    res = {"config_ok": config_ok, "T_applied": T_applied,
           "temperature_raw": list(agent.temperature_raw),
           "temperature_by_options_raw": dict(agent.temperature_by_options_raw),
           "roundtrip_n": len(states),
           "roundtrip_max_dp": float(np.abs(softmax(z / T_applied) - p_pred).max()),
           "saved_budgets": {k: agent.cfg.get(k) for k in BUDGET_KEYS}, "model_name": agent.cfg.get("model_name")}
    del agent
    free_memory()
    return res


def _real(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def read_summary(path: Path) -> Any:
    """train_single's summary.json: None when absent, the parse error text when unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"unreadable {path}: {exc}"


def resolve_model(flag: str | None, summary: Any, model_name: Any) -> str | None:
    """--model, else summary.json's "model", else train_single's model_name "laya-poc-<model>-s<seed>"."""
    if flag:
        return flag
    if isinstance(summary, dict) and isinstance(summary.get("model"), str):
        return summary["model"]
    m = _MODEL_NAME_RE.match(model_name) if isinstance(model_name, str) else None
    return m.group("model") if m else None


def check_budgets(saved: dict[str, Any], cfg: dict[str, Any], model: str | None) -> dict[str, Any]:
    """critique §C: the saved max_len/head_max_len equal the values the training items were built with."""
    base = {"saved": saved, "model": model}
    if model is None:
        return {**base, "ok": None, "note": "model unknown (no summary.json model, no laya-poc model_name); "
                                             "pass --model"}
    if model not in cfg["model"]:
        return {**base, "ok": False, "note": f"model {model!r} is not in config.model {sorted(cfg['model'])}"}
    build = {k: int(cfg["model"][model][k]) for k in BUDGET_KEYS}
    note = ", ".join(f"{k} {saved.get(k)}" + ("" if saved.get(k) == build[k] else f" != build {build[k]}")
                     for k in BUDGET_KEYS)
    return {**base, "build": build, "ok": saved == build, "note": f"{model}: {note}"}


def check_train_eval(pre: dict[str, Any], summary: Any, *, subset: bool) -> dict[str, Any]:
    """critique §C: the exported checkpoint scores the labelled val rows like the training model did at the
    end (train_single's final_eval: the same rows, raw-logit argmax), within a couple of rows."""
    if summary is None:
        return {"ok": None, "note": f"no {SUMMARY_NAME} beside the checkpoint: nothing to compare with"}
    fe = summary.get("final_eval") if isinstance(summary, dict) else f"{SUMMARY_NAME} is {summary!r:.100}"
    if fe is None:
        return {"ok": None, "note": f"{SUMMARY_NAME} has no final_eval (trained without --final-eval)"}
    acc, n = (fe.get("val_acc"), fe.get("n")) if isinstance(fe, dict) else (None, None)
    if not (_real(acc) and isinstance(n, int)):
        return {"ok": False, "note": f"malformed final_eval: {fe!r:.120}"}
    if n != pre["n"]:
        note = f"training evaluated {n} rows, this check {pre['n']}"
        return {"ok": None, "note": note + " (--n selects a subset)"} if subset else {"ok": False, "note": note}
    off, tol = abs(pre["acc"] - acc) * n, max(TRAIN_ACC_TOL_ROWS, TRAIN_ACC_TOL_FRAC * n)
    return {"ok": off <= tol + 1e-9, "train_val_acc": acc, "train_n": n, "rows_off": round(off, 3),
            "note": f"acc {pre['acc']:.4f} vs training {acc:.4f} on {n} rows ({off:.1f} rows off, max {tol:g})"}


def cross_checks(args: argparse.Namespace, ckpt: Path, pre: dict[str, Any], rt: dict[str, Any]) -> dict[str, Any]:
    """train_eval and budgets; pops the reload facts they consume from `rt`."""
    summary = read_summary(Path(args.train_summary) if args.train_summary else ckpt.parent / SUMMARY_NAME)
    model = resolve_model(args.model, summary, rt.pop("model_name"))
    return {"train_eval": check_train_eval(pre, summary, subset=args.n is not None),
            "budgets": check_budgets(rt.pop("saved_budgets"), load_config(args.config), model)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ckpt = checkpoint_dir(args.ckpt)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but no CUDA device (use --device cpu for local runs)")
    t0 = time.perf_counter()
    rows = select_rows(iter_jsonl(args.rows), args.n)
    question = shared_question(rows)
    keys = list(question[next(iter(question))]["criteria"])
    guard = FallbackGuard()
    logits, y, T = fit_on_checkpoint(ckpt, rows, question, keys, args.device, args.batch_size, guard=guard)
    write_temperature(ckpt, T)
    rt = check_reload(ckpt, rows, question, keys, T, args.device, guard=guard)
    pre, post = headline(softmax(logits), y, keys), headline(softmax(logits / rt["T_applied"]), y, keys)
    xc = cross_checks(args, ckpt, pre, rt)
    ok = (rt["config_ok"] and rt["roundtrip_max_dp"] <= ROUNDTRIP_MAX_DP and math.isfinite(T)
          and post["nll"] <= pre["nll"] + NLL_TOLERANCE and all(c["ok"] is not False for c in xc.values())
          and not guard.cpu_fallback)
    # passed: the export round trip and the cross-checks. calibration_ok: the fitted T is the one applied
    # (Laya clamps to [0.5, 5]; a clamp signals a mis-specified model, informational for a 250-step run).
    return {"ckpt": str(ckpt), "n": len(rows), "T": T, "clamped": is_clamped(T), "calibration_ok": not is_clamped(T),
            **rt, **xc, "pre": pre, "post": post, "passed": bool(ok), "device": args.device,
            "cpu_fallback": guard.cpu_fallback,
            "seconds": round(time.perf_counter() - t0, 2)}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.export_check", description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="absolute Laya checkpoint dir (e.g. runs/smoke/control/final)")
    p.add_argument("--rows", required=True, help="val.jsonl; rows with a null label are skipped")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--n", type=int, default=None, help="use the first N labelled rows (default: all)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--config", default=None, help="config.yaml with the build max_len/head_max_len per model")
    p.add_argument("--model", choices=("laya", "laya_ml"), default=None,
                   help="config.model entry the checkpoint was trained from (default: from summary.json)")
    p.add_argument("--train-summary", default=None, help=f"train_single summary (default: <ckpt>/../{SUMMARY_NAME})")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        res = run(args)
        write_json(args.out, res)
        print(f"export_check: T={res['T']:.4f} (applied {res['T_applied']:.4f}{', CLAMPED' if res['clamped'] else ''}) "
              f"ECE {res['pre']['ece']:.4f}->{res['post']['ece']:.4f} NLL {res['pre']['nll']:.4f}->"
              f"{res['post']['nll']:.4f} roundtrip max|dp|={res['roundtrip_max_dp']:.5f} "
              f"-> {'PASS' if res['passed'] else 'FAIL'} ({res['n']} rows"
              f"{', CPU fallback during inference' if res['cpu_fallback'] else ''})")
        for name in ("train_eval", "budgets"):
            status = {True: "ok", False: "FAIL"}.get(res[name]["ok"], "skipped")
            print(f"export_check: {name} {status}: {res[name]['note']}")
        return 0

    return run_cli("export_check", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
