"""fp32-CPU vs fp16-GPU parity of a Laya checkpoint (design doc §7.7, fixed per critique §A §7.7).

Fixes over the doc's snippet: states are passed as the compact JSON *strings* the model is trained
on (a dict would be re-serialised with ", " / ": " separators); the GPU agent is asserted to really
be on CUDA (Laya silently falls back to CPU, which would compare CPU with CPU and pass trivially);
rounded 4-dp probabilities are renormalised; and a padded-vs-unpadded check on the test agent
catches NaN/drift from ModernBERT sliding-window masks under fp16 SDPA (unverified on sm_75).
The GPU calls run under a FallbackGuard: Laya answers an OOM batch on CPU and then reports cuda
again, so a fallback fails parity. The CPU fp32 reference (10-25 min on Colab's single core) uses
every CPU thread and length-sorted batches (fp32 is padding-invariant) and prints a line per ~25%
of rows; the test side stays unsorted so the padding check still bites.

    python -m laya_poc.parity --model laya|laya_ml [--init hub|<abs dir>] --rows <val.jsonl> --n 200
                              --out <json> [--allow-cpu] [--attn eager] [--config yaml]

Exit code is 0 even when parity fails: the smoke report decides. It prints PASS/FAIL.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .config import load_config
from .env_check import run_cli, write_json
from .fallback_guard import FallbackGuard
from .io_utils import iter_jsonl

PADDED_N = 16
PROGRESS_PARTS = 4  # one reference progress line per ~25% of rows


def compare(p_ref: np.ndarray, p_test: np.ndarray) -> dict[str, Any]:
    """max |Δp| over rows where both are finite, argmax agreement (non-finite rows disagree), NaN rows."""
    p_ref, p_test = np.asarray(p_ref, dtype=float), np.asarray(p_test, dtype=float)
    if p_ref.shape != p_test.shape or p_ref.ndim != 2 or len(p_ref) == 0:
        raise ValueError(f"need two non-empty [N, K] matrices of the same shape; got {p_ref.shape} vs {p_test.shape}")
    bad_ref, bad_test = ~np.isfinite(p_ref).all(1), ~np.isfinite(p_test).all(1)
    ok = ~(bad_ref | bad_test)
    max_dp = float(np.abs(p_ref[ok] - p_test[ok]).max()) if ok.any() else None
    agree = (p_ref.argmax(1) == p_test.argmax(1)) & ok
    return {"max_dp": max_dp, "argmax_agree": float(agree.mean()),
            "nan_ref": int(bad_ref.sum()), "nan_test": int(bad_test.sum())}


def passed(cmp: dict[str, Any], *, padded_max_dp: float | None, padded_nan: bool,
           thresholds: dict[str, float]) -> bool:
    """Doc §6.2 parity criterion, plus the padded-vs-unpadded check at the same |Δp| bound."""
    max_dp_ok = cmp["max_dp"] is not None and cmp["max_dp"] <= thresholds["parity_max_dp"]
    padded_ok = padded_max_dp is not None and padded_max_dp <= thresholds["parity_max_dp"] and not padded_nan
    return bool(max_dp_ok and cmp["argmax_agree"] >= thresholds["parity_min_agree"]
                and not (cmp["nan_ref"] or cmp["nan_test"]) and padded_ok)


def renormalise(p: np.ndarray) -> np.ndarray:
    """Rows of 4-dp-rounded probabilities rescaled to sum to 1; an all-zero row becomes NaN."""
    p = np.asarray(p, dtype=float)
    s = p.sum(1, keepdims=True)
    return np.divide(p, s, out=np.full_like(p, np.nan), where=s > 0)


def mixed_length_indices(states: Sequence[str], k: int) -> list[int]:
    """k row indices spread evenly over the length distribution (shortest and longest included)."""
    n = len(states)
    if n <= k:
        return list(range(n))
    by_len = sorted(range(n), key=lambda i: len(states[i]))
    picks = np.linspace(0, n - 1, k).round().astype(int)
    return sorted(by_len[p] for p in picks)


def _question_id(question: dict) -> str:
    if len(question) != 1:
        raise ValueError(f"expected a single question, got {list(question)}")
    return next(iter(question))


def probs_matrix(agent: Any, states: Sequence[str], question: dict, keys: Sequence[str], batch_size: int,
                 max_len: int, head_max_len: int, *, sort_by_length: bool = False) -> np.ndarray:
    """[N, K] renormalised probabilities from predict_batch on STRING states, columns in `keys` order
    (rows keep their order with sort_by_length too: Laya only regroups the batches internally)."""
    if any(not isinstance(s, str) for s in states):
        raise TypeError("states must be the compact JSON strings from the rows, never parsed dicts")
    qid = _question_id(question)
    out = agent.predict_batch(list(states), question, batch_size=batch_size, max_len=max_len,
                              head_max_len=head_max_len, sort_by_length=sort_by_length)
    raw = np.array([[o["answers"][qid]["probabilities"][k] for k in keys] for o in out], dtype=float)
    return renormalise(raw)


def padded_vs_unpadded(agent: Any, states: Sequence[str], question: dict, keys: Sequence[str], max_len: int,
                       head_max_len: int, n: int = PADDED_N) -> dict[str, Any]:
    """Same rows in one padded batch vs one at a time on the same agent: max |Δp| and NaN."""
    sub = [states[i] for i in mixed_length_indices(states, n)]
    batched = probs_matrix(agent, sub, question, keys, len(sub), max_len, head_max_len)
    single = probs_matrix(agent, sub, question, keys, 1, max_len, head_max_len)
    cmp = compare(single, batched)
    return {"padded_max_dp": cmp["max_dp"], "padded_argmax_agree": cmp["argmax_agree"],
            "padded_nan": bool(cmp["nan_ref"] or cmp["nan_test"]), "padded_n": len(sub)}


def resolve_source(cfg: dict[str, Any], model: str, init: str) -> Any:
    """The pinned Hub ModelSpec for --init hub, else an absolute existing checkpoint directory."""
    from .hub import model_spec

    return model_spec(cfg, model) if init == "hub" else checkpoint_dir(init)


def checkpoint_dir(path_str: str | Path) -> Path:
    """An absolute, existing Laya checkpoint directory (a missing relative path would be sent to the Hub)."""
    path = Path(path_str)
    if not path.is_absolute():
        raise ValueError(f"checkpoint paths must be absolute ('hub' for the pinned Hub model), got {str(path_str)!r}")
    if not (path / "rl_agent_config.json").exists():
        raise FileNotFoundError(f"not a Laya checkpoint directory: {path}")
    return path


def load_reference(source: Any) -> Any:
    """CPU agent computing in true fp32 (LAYA_CPU_AMP could otherwise enable bf16 autocast)."""
    import torch
    from .hub import load_agent

    agent = load_agent(source, device="cpu")
    agent.amp_enabled, agent.dtype = False, torch.float32
    if next(agent.model.encoder.parameters()).dtype != torch.float32:
        agent.model.float()
    return agent


def load_test(source: Any, *, allow_cpu: bool, attn: str | None) -> Any:
    """GPU agent with fp16 autocast; refuses a silent CPU fallback unless --allow-cpu."""
    import torch
    from .hub import load_agent

    agent = load_agent(source, device="cuda" if torch.cuda.is_available() else "cpu")
    if agent.device.type == "cuda":
        agent.dtype, agent.amp_enabled = torch.float16, True
    elif not allow_cpu:
        raise RuntimeError(f"test agent landed on {agent.device.type}, not cuda (OOM at load?)")
    if attn:
        agent.model.encoder.set_attn_implementation(attn)
    return agent


def free_memory() -> None:
    """Release a dropped agent before loading the next one (Colab RAM ~12.7 GB, T4 VRAM ~15 GB)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@contextmanager
def all_cpu_threads() -> Iterator[int]:
    """Every logical CPU for torch (it defaults to physical cores: 1 on Colab's 2 vCPUs); restored after."""
    import torch

    before, n = torch.get_num_threads(), os.cpu_count() or 1
    torch.set_num_threads(n)
    try:
        yield n
    finally:
        torch.set_num_threads(before)


@contextmanager
def batch_progress(agent: Any, total: int, label: str, parts: int = PROGRESS_PARTS) -> Iterator[None]:
    """Print '<label> k/total rows, Ns' each time another ~1/parts of the rows is done. predict_batch is
    silent, so count the rows of every batch it sends through the agent's per-batch `_forward`."""
    forward, had_own, t0 = agent._forward, "_forward" in vars(agent), time.perf_counter()
    done, mark = 0, 1

    def counted(b: dict) -> Any:
        nonlocal done, mark
        out = forward(b)
        done += int(b["input_ids"].shape[0])  # one row per state: a single question is enforced
        if done * parts >= mark * total:
            print(f"{label} {min(done, total)}/{total} rows, {time.perf_counter() - t0:.0f}s", flush=True)
            mark = done * parts // total + 1
        return out

    agent._forward = counted
    try:
        yield
    finally:
        if had_own:
            agent._forward = forward
        else:
            del agent._forward


def _meta(agent: Any, t0: float) -> dict[str, Any]:
    return {"device": agent.device.type, "amp": bool(agent.amp_enabled),
            "dtype": str(agent.dtype).replace("torch.", "") if agent.amp_enabled else "float32",
            "seconds": round(time.perf_counter() - t0, 2)}


def _reference_side(source: Any, states: Sequence[str], question: dict, keys: Sequence[str], batch_size: int,
                    budget: tuple[int, int], label: str) -> tuple[np.ndarray, dict[str, Any]]:
    """CPU fp32 probabilities: all CPU threads, length-sorted batches, sparse progress; the agent is freed."""
    agent = load_reference(source)
    t0 = time.perf_counter()
    with all_cpu_threads() as threads, batch_progress(agent, len(states), label):
        p = probs_matrix(agent, states, question, keys, batch_size, *budget, sort_by_length=True)
    meta = {**_meta(agent, t0), "threads": threads}
    del agent
    free_memory()
    return p, meta


def _test_side(source: Any, states: Sequence[str], question: dict, keys: Sequence[str], batch_size: int,
               budget: tuple[int, int], *, allow_cpu: bool, attn: str | None
               ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """GPU fp16 probabilities in UNSORTED batches plus the padded check, all under the CPU-fallback guard."""
    agent = load_test(source, allow_cpu=allow_cpu, attn=attn)
    guard = FallbackGuard()
    t0 = time.perf_counter()
    with guard:
        p = probs_matrix(agent, states, question, keys, batch_size, *budget)
        padded = padded_vs_unpadded(agent, states, question, keys, *budget)
    meta = {**_meta(agent, t0), "cpu_fallback": guard.cpu_fallback}
    del agent
    free_memory()
    return p, meta, padded


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from .labels import question as make_question

    cfg = load_config(args.config)
    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: the fp16 side of parity needs a GPU (--allow-cpu is for local tests)")
    source = resolve_source(cfg, args.model, args.init)
    rows = list(itertools.islice(iter_jsonl(args.rows), args.n))
    if not rows:
        raise ValueError(f"no rows in {args.rows}")
    states = [r["state"] for r in rows]
    question = make_question(cfg["labels"]["scheme"])
    keys = list(question[_question_id(question)]["criteria"])
    budget = (int(cfg["model"][args.model]["max_len"]), int(cfg["model"][args.model]["head_max_len"]))
    thresholds = {k: float(cfg["smoke"]["exit"][k]) for k in ("parity_max_dp", "parity_min_agree")}

    t0, label = time.perf_counter(), f"parity {args.model}: reference"
    print(f"{label} on cpu fp32 ({os.cpu_count() or 1} threads, length-sorted), {len(states)} rows ...", flush=True)
    p_ref, ref_meta = _reference_side(source, states, question, keys, args.batch_size, budget, label)
    p_test, test_meta, padded = _test_side(source, states, question, keys, args.batch_size, budget,
                                           allow_cpu=args.allow_cpu, attn=args.attn)
    cmp = compare(p_ref, p_test)
    ok = passed(cmp, padded_max_dp=padded["padded_max_dp"], padded_nan=padded["padded_nan"], thresholds=thresholds)
    return {"model": args.model, "init": args.init, "n": len(states), **cmp,
            "nan": bool(cmp["nan_ref"] or cmp["nan_test"]), **padded, "cpu_fallback": test_meta["cpu_fallback"],
            "passed": ok and not test_meta["cpu_fallback"], "thresholds": thresholds,
            "device_ref": ref_meta["device"], "device_test": test_meta["device"], "dtype_test": test_meta["dtype"],
            "attn_test": args.attn or "sdpa", "max_len": budget[0], "head_max_len": budget[1],
            "threads_ref": ref_meta["threads"], "seconds_ref": ref_meta["seconds"],
            "seconds_test": test_meta["seconds"], "seconds": round(time.perf_counter() - t0, 2)}


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.parity", description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=("laya", "laya_ml"))
    p.add_argument("--init", default="hub", help="'hub' (pinned revision) or an absolute checkpoint dir")
    p.add_argument("--rows", required=True, help="JSONL rows (val.jsonl); the first --n are used")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--allow-cpu", action="store_true", help="let the test agent run on CPU (local tests only)")
    p.add_argument("--attn", choices=("sdpa", "eager"), default=None, help="attention for the test agent")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        res = run(args)
        write_json(args.out, res)
        print(f"parity {res['model']}: max|dp|={_fmt(res['max_dp'])} argmax={res['argmax_agree']:.3f} "
              f"padded={_fmt(res['padded_max_dp'])} nan={res['nan']} -> {'PASS' if res['passed'] else 'FAIL'} "
              f"(test {res['device_test']} {res['dtype_test']}{', CPU fallback' if res['cpu_fallback'] else ''}, "
              f"{res['n']} rows, {res['seconds']:.0f}s; ref {res['seconds_ref']:.0f}s, "
              f"test {res['seconds_test']:.0f}s)")
        return 0

    return run_cli("parity", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
