"""The CPU-benchmark worker: ONE (model, backend, thread count) setting in a fresh process (design doc §7.11).

Every backend goes through the same timing core (`time_runtime`): cold start = seconds to a ready-to-predict runtime
(weights on local disk, no network), one untimed-but-recorded first call, `warmup` batch-1 calls, then batch-1
latency over the `latency` STRING states (p50/p95/mean/max ms), then ONE batched call over the `batch` states
(length-sorted batches of `batch_size`) for records/s, then the process's peak RSS. Its two halves, `time_latency` and
`time_batch`, are also the GPU benchmark's timing core (bench_gpu_worker: several batch sizes per process). Backends:
- `torch`: the Laya PyTorch Agent in true fp32 (parity.load_reference; amp off): batch-1 `agent.predict`, batched
  `agent.predict_batch(sort_by_length=True, batch_size)`;
- `onnx`: laya's ONNXAgent with intra_op_num_threads = threads (bench_onnx): batch-1 and batched both through
  `bench_onnx.onnx_predict_batch` (Agent.predict_batch's collate path on the ONNX session);
- `hf`: a small encoder (bench_hf), fp32, max_length 256, batch-1 and length-sorted batches of `batch_size`.
Times are end to end from the state string to probabilities (tokenisation and decoding included) for every backend.
"""
from __future__ import annotations

import itertools
import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .bench_machine import peak_rss_gb, physical_cores

MARKER = "BENCH_JSON "


@dataclass(frozen=True)
class Runtime:
    predict_one: Callable[[str], Any]
    predict_many: Callable[[list[str]], Any]
    info: dict[str, Any] = field(default_factory=dict)


def latency_stats(ms: Sequence[float]) -> dict[str, float]:
    a = np.asarray(ms, dtype=float)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "max_ms": float(a.max())}


def _timed_ms(fn: Callable[[], Any]) -> float:
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000.0


def _check_states(states: Sequence[Any]) -> None:
    if not states:
        raise ValueError("no states to time")
    if any(not isinstance(s, str) for s in states):
        raise TypeError("states must be the compact JSON strings from the rows, never parsed dicts")


def time_latency(load: Callable[[], Any], latency: Sequence[str], warmup: int) -> tuple[Any, dict[str, Any]]:
    """(the runtime `load()` returned, its cold start, first call and batch-1 latency). The runtime only needs a
    `predict_one`: the GPU benchmark (bench_gpu_worker) times its batched calls per batch size with time_batch."""
    _check_states(latency)
    t0 = time.perf_counter()
    rt = load()
    cold = time.perf_counter() - t0
    first = _timed_ms(lambda: rt.predict_one(latency[0]))
    for s in itertools.islice(itertools.cycle(latency), warmup):
        rt.predict_one(s)
    lat = [_timed_ms(lambda s=s: rt.predict_one(s)) for s in latency]
    return rt, {"n": len(latency), "n_latency": len(latency), "warmup": warmup, "cold_s": round(cold, 3),
                "first_predict_ms": round(first, 2), **{k: round(v, 2) for k, v in latency_stats(lat).items()}}


def time_batch(predict_many: Callable[[list[str]], Any], batch: Sequence[str]) -> dict[str, Any]:
    """ONE call of `predict_many` over every batch state: records/s and its seconds."""
    _check_states(batch)
    batch_ms = _timed_ms(lambda: predict_many(list(batch)))
    return {"n_batch": len(batch), "batch_rps": round(len(batch) / (batch_ms / 1000.0), 3),
            "batch_seconds": round(batch_ms / 1000.0, 3)}


def time_runtime(load: Callable[[], Runtime], latency: Sequence[str], batch: Sequence[str],
                 warmup: int) -> dict[str, Any]:
    """Cold start, batch-1 latency and batched throughput of the runtime `load()` returns."""
    _check_states(latency)
    _check_states(batch)
    rt, lat = time_latency(load, latency, warmup)
    b = time_batch(rt.predict_many, batch)
    rss, how = peak_rss_gb()
    return {"n": lat["n"], "n_latency": lat["n_latency"], "n_batch": b["n_batch"], "warmup": warmup,
            **{k: v for k, v in lat.items() if k not in ("n", "n_latency", "warmup")},
            "batch_rps": b["batch_rps"], "batch_seconds": b["batch_seconds"],
            "peak_rss_gb": None if rss is None else round(rss, 3), "peak_rss_source": how, **rt.info}


# ---------------------------------------------------------------- Laya runtimes

def laya_torch_loader(source: Any, question: dict, *, batch_size: int, max_len: int,
                      head_max_len: int) -> Callable[[], Runtime]:
    def load() -> Runtime:
        import torch
        from .parity import load_reference

        agent = load_reference(source)
        budget = {"max_len": max_len, "head_max_len": head_max_len}
        return Runtime(
            predict_one=lambda s: agent.predict(s, question, **budget),
            predict_many=lambda ss: agent.predict_batch(ss, question, batch_size=batch_size, sort_by_length=True,
                                                        **budget),
            info={"framework": "pytorch", "device": agent.device.type, "amp": bool(agent.amp_enabled),
                  "dtype": str(agent.dtype).replace("torch.", ""),
                  "param_dtype": str(next(agent.model.parameters()).dtype).replace("torch.", ""),
                  "torch": torch.__version__})
    return load


def laya_onnx_loader(ckpt_dir: Path, onnx_path: Path, question: dict, *, threads: int, batch_size: int,
                     max_len: int, head_max_len: int) -> Callable[[], Runtime]:
    def load() -> Runtime:
        import onnxruntime as ort
        from .bench_onnx import load_onnx_agent, onnx_predict_batch

        agent = load_onnx_agent(ckpt_dir, onnx_path, threads)
        budget = {"max_len": max_len, "head_max_len": head_max_len}
        return Runtime(
            predict_one=lambda s: onnx_predict_batch(agent, [s], question, batch_size=1, **budget),
            predict_many=lambda ss: onnx_predict_batch(agent, ss, question, batch_size=batch_size,
                                                       sort_by_length=True, **budget),
            info={"framework": "onnxruntime", "device": "cpu", "amp": False, "dtype": "float32",
                  "ort_intra_op_threads": agent.session.get_session_options().intra_op_num_threads,
                  "onnxruntime": ort.__version__, "onnx_path": str(onnx_path)})
    return load


def measure(source: Any, states: Sequence[str], question: dict, *, threads: int, warmup: int, batch_size: int,
            max_len: int, head_max_len: int, batch_states: Sequence[str] | None = None) -> dict[str, Any]:
    """Phase 2 entry point: the Laya PyTorch agent at `threads` torch threads (batched over `batch_states`, default
    the latency states)."""
    import torch

    torch.set_num_threads(threads)
    loader = laya_torch_loader(source, question, batch_size=batch_size, max_len=max_len, head_max_len=head_max_len)
    res = time_runtime(loader, states, list(batch_states or states), warmup)
    return {"threads": threads, "torch_threads": torch.get_num_threads(), "batch_size": batch_size, **res}


# ---------------------------------------------------------------- worker entry (called by bench_cpu.main)

def _loader(args: Any, cfg: dict, question: dict, source: Any) -> Callable[[], Runtime]:
    if args.worker_backend == "hf":
        from .bench_hf import hf_loader
        return hf_loader(cfg, args.model, source.path, batch_size=args.batch_size)
    mc = cfg["model"][args.model]
    budget = {"batch_size": args.batch_size, "max_len": int(mc["max_len"]), "head_max_len": int(mc["head_max_len"])}
    if args.worker_backend == "onnx":
        if not args.onnx_path:
            raise ValueError("the onnx backend needs --onnx-path (the parent exports it first)")
        return laya_onnx_loader(source.path, Path(args.onnx_path), question, threads=args.worker_threads, **budget)
    return laya_torch_loader(source.path, question, **budget)


def worker_main(args: Any, cfg: dict, load_states: Callable[[Any, int], list[str]]) -> int:
    """Measure one setting and print its result as one MARKER-prefixed JSON line."""
    import torch
    from .bench_sources import backends_for, resolve_source
    from .labels import question as make_question

    threads, backend = args.worker_threads, args.worker_backend
    if backend not in backends_for(cfg, args.model, [backend]):
        raise ValueError(f"backend {backend!r} does not apply to model {args.model!r}")
    torch.set_num_threads(threads)
    source = resolve_source(cfg, args.model, args.ckpt)
    states = load_states(args.rows, max(args.latency_n, args.batch_n))
    q = make_question(cfg["labels"]["scheme"])
    res = time_runtime(_loader(args, cfg, q, source), states[:args.latency_n], states[:args.batch_n], args.warmup)
    cores, _ = physical_cores()
    row = {"model": args.model, "backend": backend, "threads": threads, "torch_threads": torch.get_num_threads(),
           "batch_size": args.batch_size, **res, "cpu_count": os.cpu_count(), "physical_cores": cores,
           "platform": platform.platform(), "pid": os.getpid()}
    print(MARKER + json.dumps(row), flush=True)
    return 0
