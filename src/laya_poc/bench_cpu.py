"""CPU inference benchmark of a Laya checkpoint, PyTorch backend (design doc §7.11, fixed per critique §7.11).

One FRESH subprocess per thread count, so torch.set_num_threads, the BLAS pools and the cold start are clean
(Laya's Agent ignores LAYA_THREADS; only laya.serve reads it). Each worker (GPU hidden, fp32: LAYA_CPU_AMP
unset) loads the agent (cold start = load seconds), warms up, times batch-1 `agent.predict` over n STRING
states (p50/p95/mean ms), times `predict_batch(sort_by_length=True, batch_size)` (records/s) and reports
its peak RSS in GiB (2**30 bytes; the true peak: psutil peak_wset on Windows, resource ru_maxrss on POSIX;
else psutil's current RSS, else tracemalloc's Python heap), then prints one marker-prefixed JSON line that
the parent collects. The config cpu_budget (p95_ms, min_rps) is compared as INFORMATION only: the
decision-rule budget is judged in Phase 5 on 4 vCPUs, with the PyTorch-vs-ONNX, 1/2/4/8-thread sweep.

    python -m laya_poc.bench_cpu --ckpt <abs Laya dir>|hub [--model laya|laya_ml] --rows <jsonl> --out <json>
                                 [--n 200] [--threads 1 2] [--warmup 20] [--batch-size 32]
                                 [--backend torch|onnx] [--timeout-min 60] [--config yaml]

Exit code 0 when at least one thread setting produced a result (failures are listed under "errors").
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import load_config
from .env_check import run_cli, write_json
from .io_utils import iter_jsonl

MARKER = "BENCH_JSON "
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LAYA_THREADS")
_PMC_SIZE_FIELDS = ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
BUDGET_NOTE = ("information only: the decision-rule CPU budget is judged in Phase 5 on "
               "{vcpus} vCPUs (fp32, the faster of PyTorch and ONNX)")
ONNX_TODO = ("the ONNX backend is Phase 5 (design doc §7.11): export with laya's scripts/export_onnx.py --model "
             "<ckpt> --output <file.onnx>, then load laya.onnx_agent.ONNXAgent (not laya.ONNXAgent; it has no "
             "predict_batch and sets threads via SessionOptions.intra_op_num_threads)")


# ---------------------------------------------------------------- pure helpers

def latency_stats(ms: Sequence[float]) -> dict[str, float]:
    a = np.asarray(ms, dtype=float)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "max_ms": float(a.max())}


def budget_flags(result: Mapping[str, Any], budget: Mapping[str, Any]) -> dict[str, bool]:
    return {"p95_ok": bool(result["p95_ms"] <= budget["p95_ms"]),
            "rps_ok": bool(result["batch_rps"] >= budget["min_rps"])}


def parse_worker_output(stdout: str) -> dict[str, Any] | None:
    """The worker's result: the last line starting with MARKER (Laya may print warnings around it)."""
    lines = [ln[len(MARKER):] for ln in stdout.splitlines() if ln.startswith(MARKER)]
    return json.loads(lines[-1]) if lines else None


def worker_env(threads: int, base: Mapping[str, str]) -> dict[str, str]:
    """A copy of `base` pinning every thread pool to `threads`, hiding CUDA and dropping CPU autocast."""
    env = {k: v for k, v in base.items() if k != "LAYA_CPU_AMP"}
    env.update({k: str(threads) for k in THREAD_VARS})
    env.update(CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    return env


def _windows_peak_bytes() -> int | None:
    """PeakWorkingSetSize via GetProcessMemoryInfo (psutil's peak_wset without psutil)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    *((name, ctypes.c_size_t) for name in _PMC_SIZE_FIELDS)]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    ok = get_info(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return int(c.PeakWorkingSetSize) if ok else None


def peak_rss_gb() -> tuple[float | None, str]:
    """(peak resident set size in GiB, how it was measured) for this process."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        if getattr(info, "peak_wset", None):
            return info.peak_wset / 2**30, "psutil.peak_wset"
    except ImportError:
        info = None
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (2**30 if sys.platform == "darwin" else 2**20), "resource.ru_maxrss"
    except ImportError:
        pass
    peak = _windows_peak_bytes()
    if peak:
        return peak / 2**30, "GetProcessMemoryInfo.PeakWorkingSetSize"
    if info is not None:
        return info.rss / 2**30, "psutil.rss (current, not peak)"
    import tracemalloc
    if tracemalloc.is_tracing():
        return tracemalloc.get_traced_memory()[1] / 2**30, "tracemalloc peak (Python heap only)"
    return None, "unavailable"


# ---------------------------------------------------------------- worker (one thread setting, in-process)

def _timed_ms(fn: Callable[[], Any]) -> float:
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000.0


def measure(source: Any, states: Sequence[str], question: dict, *, threads: int, warmup: int, batch_size: int,
            max_len: int, head_max_len: int) -> dict[str, Any]:
    """Cold start, batch-1 latency and batched throughput of one agent at `threads` torch threads."""
    import torch
    from .hub import load_agent

    if any(not isinstance(s, str) for s in states):
        raise TypeError("states must be the compact JSON strings from the rows, never parsed dicts")
    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    agent = load_agent(source, device="cpu")
    cold = time.perf_counter() - t0
    budget = {"max_len": max_len, "head_max_len": head_max_len}
    first = _timed_ms(lambda: agent.predict(states[0], question, **budget))
    for s in itertools.islice(itertools.cycle(states), warmup):
        agent.predict(s, question, **budget)
    lat = [_timed_ms(lambda s=s: agent.predict(s, question, **budget)) for s in states]
    batch_ms = _timed_ms(lambda: agent.predict_batch(list(states), question, batch_size=batch_size,
                                                     sort_by_length=True, **budget))
    rss, source_name = peak_rss_gb()
    return {"threads": threads, "torch_threads": torch.get_num_threads(), "n": len(states), "warmup": warmup,
            "batch_size": batch_size, "cold_s": round(cold, 3), "first_predict_ms": round(first, 2),
            **{k: round(v, 2) for k, v in latency_stats(lat).items()},
            "batch_rps": round(len(states) / (batch_ms / 1000.0), 3), "batch_seconds": round(batch_ms / 1000.0, 3),
            "peak_rss_gb": None if rss is None else round(rss, 3), "peak_rss_source": source_name,
            "device": agent.device.type, "amp": bool(agent.amp_enabled),
            "dtype": str(agent.dtype).replace("torch.", ""), "torch": torch.__version__}


def worker_main(args: argparse.Namespace) -> int:
    from .labels import question as make_question
    from .parity import resolve_source

    cfg = load_config(args.config)
    mc = cfg["model"][args.model]
    states = load_states(args.rows, args.n)
    res = measure(resolve_source(cfg, args.model, args.ckpt), states, make_question(cfg["labels"]["scheme"]),
                  threads=args.worker_threads, warmup=args.warmup, batch_size=args.batch_size,
                  max_len=int(mc["max_len"]), head_max_len=int(mc["head_max_len"]))
    print(MARKER + json.dumps(res), flush=True)
    return 0


# ---------------------------------------------------------------- parent

def load_states(path: str | Path, n: int) -> list[str]:
    states = [r["state"] for r in itertools.islice(iter_jsonl(path), n)]
    if not states:
        raise ValueError(f"no rows in {path}")
    return states


def worker_cmd(args: argparse.Namespace, threads: int) -> list[str]:
    cmd = [sys.executable, "-m", "laya_poc.bench_cpu", "--worker-threads", str(threads), "--ckpt", args.ckpt,
           "--model", args.model, "--rows", str(args.rows), "--out", str(args.out), "--n", str(args.n),
           "--warmup", str(args.warmup), "--batch-size", str(args.batch_size)]
    return cmd + (["--config", str(args.config)] if args.config else [])


def run_worker(cmd: list[str], env: dict[str, str], timeout: float) -> Any:
    return subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", stdin=subprocess.DEVNULL)


def prefetch(source: Any) -> None:
    """Download a Hub checkpoint before timing, so no worker's cold start includes the download."""
    from .hub import ModelSpec, snapshot

    if isinstance(source, ModelSpec):
        snapshot(source)


def run_setting(args: argparse.Namespace, threads: int, budget: Mapping[str, Any]) -> dict[str, Any]:
    """One worker process: its result (plus budget flags), or {threads, returncode, error}."""
    try:
        proc = run_worker(worker_cmd(args, threads), worker_env(threads, os.environ), args.timeout_min * 60)
    except subprocess.TimeoutExpired:
        return {"threads": threads, "returncode": None, "error": f"timed out after {args.timeout_min} min"}
    res = parse_worker_output(proc.stdout or "")
    if proc.returncode != 0 or res is None:
        tail = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
        why = tail[-1] if tail else f"no result line (exit code {proc.returncode})"
        return {"threads": threads, "returncode": proc.returncode,
                "error": why if proc.returncode != 0 else f"no result line: {why}"}
    return {**res, **budget_flags(res, budget)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from .parity import resolve_source

    if args.backend == "onnx":
        raise NotImplementedError(ONNX_TODO)
    cfg, t0 = load_config(args.config), time.perf_counter()
    source = resolve_source(cfg, args.model, args.ckpt)  # validates the path before any worker starts
    n = len(load_states(args.rows, args.n))
    budget = dict(cfg["cpu_budget"])
    prefetch(source)
    results, errors = [], []
    for threads in args.threads:
        print(f"bench_cpu: {threads} thread(s), {n} states, fresh process ...", flush=True)
        r = run_setting(args, threads, budget)
        (errors if "error" in r else results).append(r)
        print(_line(r), flush=True)
    return {"model": args.model, "ckpt": args.ckpt, "backend": args.backend, "rows": str(args.rows), "n": n,
            "warmup": args.warmup, "batch_size": args.batch_size, "thread_settings": list(args.threads),
            "cpu_count": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version(),
            "results": results, "errors": errors,
            "budget": {**budget, "note": BUDGET_NOTE.format(vcpus=budget.get("vcpus", 4))},
            "seconds": round(time.perf_counter() - t0, 2)}


def _line(r: dict[str, Any]) -> str:
    if "error" in r:
        return f"bench_cpu: {r['threads']} thread(s): FAILED: {r['error']}"
    return (f"bench_cpu: {r['threads']} thread(s): cold {r['cold_s']:.1f}s, p50 {r['p50_ms']:.0f} ms, p95 "
            f"{r['p95_ms']:.0f} ms, batch {r['batch_rps']:.1f} rec/s, peak RSS {r['peak_rss_gb']} GiB "
            f"(budget info: p95 {'ok' if r['p95_ok'] else 'over'}, rps {'ok' if r['rps_ok'] else 'under'})")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    cfg_bench = load_config(_config_flag(argv)).get("bench", {})
    p = argparse.ArgumentParser(prog="laya_poc.bench_cpu", description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="absolute Laya checkpoint dir, or 'hub' (pinned revision)")
    p.add_argument("--model", choices=("laya", "laya_ml"), default="laya")
    p.add_argument("--rows", required=True, help="JSONL rows (e.g. test_id.jsonl); the first --n states are used")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=int(cfg_bench.get("n_records", 200)))
    p.add_argument("--threads", type=int, nargs="+", default=list(cfg_bench.get("threads", [1, 2])))
    p.add_argument("--warmup", type=int, default=int(cfg_bench.get("warmup", 20)))
    p.add_argument("--batch-size", type=int, default=int(cfg_bench.get("batch_size", 32)))
    p.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    p.add_argument("--timeout-min", type=float, default=60.0, help="per thread setting")
    p.add_argument("--config", default=None)
    p.add_argument("--worker-threads", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _config_flag(argv: list[str] | None) -> str | None:
    """--config before full parsing, so the config's bench section can supply the defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    return pre.parse_known_args(argv)[0].config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_threads is not None:
        return run_cli("bench_cpu worker", lambda: worker_main(args), None)

    def body() -> int:
        res = run(args)
        write_json(args.out, res)
        print(f"bench_cpu: wrote {args.out} ({len(res['results'])}/{len(args.threads)} thread settings)")
        return 0 if res["results"] else 1

    return run_cli("bench_cpu", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
