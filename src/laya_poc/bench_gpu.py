"""GPU inference benchmark (Phase 7): batch-1 latency and batched throughput of Laya and the small encoders on a GPU.

    python -m laya_poc.bench_gpu [--models laya laya_ml modernbert_base mmbert_small] --rows <test_id.jsonl>
        --out <json> [--ckpt hub | MODEL=/abs/dir ...] [--latency-n 500] [--batch-n 2000] [--batch-sizes 32 64]
        [--warmup 20] [--device cuda|cpu] [--timeout-min 60] [--config yaml]

Models default to config phase5.bench.models (the CPU benchmark's four): `laya`/`laya_ml` run on the PyTorch Agent
(backend `torch`, the Agent's CUDA autocast), the B4 small encoders as Phase 4 scored them (backend `hf`, fp16
autocast on CUDA). One FRESH subprocess per model, so its cold start and peak VRAM are its own; every source is
resolved to a local directory (downloading) in this parent BEFORE any timing, and workers run offline. What a worker
times: bench_gpu_worker.py (batch-1 latency once per model, then per batch size a warm-up and one timed call).
Latency does not depend on fine-tuned weights, so `--ckpt hub` (the pinned base checkpoints, the default) is valid.
`--device cuda` fails at once without a CUDA device; `--device cpu` is a local plumbing check, not a GPU number.

Output JSON (schema_version 1):
  schema_version, plan {models, rows, latency_n, batch_n, batch_sizes, warmup, batch_warmup_batches, device, ckpt},
  sources {model: {kind, path, label}}, machine {gpu, capability, vram_total_gb, cuda, cudnn, multiprocessors,
  driver, cpu_count, physical_cores, cpu_model, ram_gb, platform, python, versions{torch, transformers, laya, ...}},
  hardware (one-line label), seconds,
  results: one row per (model, batch size):
    {model, backend (torch|hf), device, gpu, capability, amp, dtype (the autocast dtype), param_dtype, batch_size,
     cold_s, first_predict_ms, p50_ms, p95_ms, mean_ms, max_ms (batch-1 latency, ms: measured once per model, the
     same on each of its rows), n_latency, n_batch, warmup, batch_warmup_rows, batch_rps, batch_seconds (one timed
     batched call over n_batch rows), peak_vram_gb, peak_vram_reserved_gb (GiB, this batch size),
     b1_peak_vram_gb, b1_peak_vram_reserved_gb (load + batch-1 phase), peak_rss_gb, cpu_fallback, source,
     hardware, pid, framework, torch, + laya_cuda_amp/max_len/head_max_len (torch) or max_length/... (hf)}
  errors: [{model, backend, batch_size (null: the whole model), returncode, error}]: a failed model or batch size
  (e.g. CUDA OOM) is recorded and the others still run. Exit code 0 when at least one row was measured.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .bench_cpu import _last_line, load_states, parse_worker_output, run_worker
from .bench_gpu_worker import BATCH_WARMUP_BATCHES, gpu_facts
from .bench_machine import machine_info
from .bench_sources import HF, LAYA, Source, model_kind, parse_ckpt, resolve_source
from .config import load_config
from .env_check import run_cli, write_json

SCHEMA_VERSION = 1
LATENCY_N, BATCH_N, BATCH_SIZES, WARMUP = 500, 2000, (32, 64), 20
TIMEOUT_MIN = 60.0
BACKEND = {LAYA: "torch", HF: "hf"}
PROG = "bench_gpu"


@dataclass(frozen=True)
class GpuPlan:
    models: tuple[str, ...]
    backends: dict[str, str]            # model -> torch | hf
    ckpt: dict[str, str]                # model -> 'hub' | absolute dir
    latency_n: int
    batch_n: int
    batch_sizes: tuple[int, ...]
    warmup: int
    device: str                         # cuda | cpu

    def as_dict(self, rows: str) -> dict[str, Any]:
        return {"models": list(self.models), "rows": rows, "latency_n": self.latency_n, "batch_n": self.batch_n,
                "batch_sizes": list(self.batch_sizes), "warmup": self.warmup,
                "batch_warmup_batches": BATCH_WARMUP_BATCHES, "device": self.device, "ckpt": self.ckpt}


def make_plan(args: argparse.Namespace, cfg: Mapping[str, Any]) -> GpuPlan:
    models = list(dict.fromkeys(args.models or ((cfg.get("phase5") or {}).get("bench") or {}).get("models") or []))
    if not models:
        raise ValueError("no models: pass --models or set config phase5.bench.models")
    backends = {m: BACKEND[model_kind(cfg, m)] for m in models}  # unknown models fail here, before anything runs
    sizes = tuple(dict.fromkeys(args.batch_sizes))
    nums = {"latency_n": args.latency_n, "batch_n": args.batch_n, "warmup": args.warmup}
    bad = [f"{k}={v}" for k, v in nums.items() if v < (0 if k == "warmup" else 1)]
    bad += [f"batch_size={b}" for b in sizes if b < 1]
    if bad or not sizes:
        raise ValueError(f"invalid settings: {', '.join(bad) or 'no batch size'} (row counts and batch sizes >= 1, "
                         "warmup >= 0)")
    return GpuPlan(models=tuple(models), backends=backends, ckpt=parse_ckpt(args.ckpt, models), batch_sizes=sizes,
                   device=args.device, **nums)


def cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def check_device(device: str) -> None:
    if device == "cuda" and not cuda_available():
        raise RuntimeError("--device cuda but no CUDA device: switch to a GPU runtime (Colab: Change runtime type -> "
                           "T4 GPU or G4), or pass --device cpu for a local plumbing check")


def resolve_sources(cfg: Mapping[str, Any], plan: GpuPlan) -> dict[str, Source]:
    """Every model's weights as an absolute local directory (downloads happen here, before any timing)."""
    return {m: resolve_source(cfg, m, plan.ckpt[m]) for m in plan.models}


# ---------------------------------------------------------------- one worker per model

def worker_env(device: str, base: Mapping[str, str]) -> dict[str, str]:
    """A copy of `base` for a worker: offline Hub, unbuffered UTF-8 output; CUDA hidden for a --device cpu run."""
    env = dict(base)
    env.update(TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def worker_cmd(args: argparse.Namespace, plan: GpuPlan, source: Source) -> list[str]:
    cmd = [sys.executable, "-m", f"laya_poc.{PROG}", "--worker", "--models", source.model, "--ckpt", str(source.path),
           "--rows", str(args.rows), "--out", str(args.out), "--latency-n", str(plan.latency_n),
           "--batch-n", str(plan.batch_n), "--batch-sizes", *map(str, plan.batch_sizes), "--warmup", str(plan.warmup),
           "--device", plan.device]
    return cmd + (["--config", str(args.config)] if args.config else [])


def run_model(args: argparse.Namespace, plan: GpuPlan, source: Source) -> tuple[list[dict], list[dict]]:
    """One worker process: (its result rows, its error rows)."""
    who = {"model": source.model, "backend": plan.backends[source.model]}
    try:
        proc = run_worker(worker_cmd(args, plan, source), worker_env(plan.device, os.environ), args.timeout_min * 60)
    except subprocess.TimeoutExpired:
        return [], [{**who, "batch_size": None, "returncode": None, "error": f"timed out after {args.timeout_min} min"}]
    res = parse_worker_output(proc.stdout or "")
    if proc.returncode != 0 or res is None:
        why = _last_line(proc.stderr, f"no result line (exit code {proc.returncode})")
        return [], [{**who, "batch_size": None, "returncode": proc.returncode,
                     "error": why if proc.returncode != 0 else f"no result line: {why}"}]
    rows = [{**r, **who, "source": source.label} for r in res.get("results") or []]
    errors = [{**who, "returncode": 0, **e} for e in res.get("errors") or []]
    return rows, errors


def _line(r: dict[str, Any]) -> str:
    bs = f" bs {r['batch_size']}" if r.get("batch_size") is not None else ""
    who = f"{PROG}: {r['model']}/{r['backend']}{bs}"
    if "error" in r:
        return f"{who}: FAILED: {r['error']}"
    vram = "n/a" if r.get("peak_vram_gb") is None else f"{r['peak_vram_gb']:.2f} GiB"
    prec = f"{r.get('dtype')} autocast" if r.get("amp") else str(r.get("dtype"))
    return (f"{who} on {r.get('gpu') or r.get('device')}: cold {r['cold_s']:.1f} s, first "
            f"{r['first_predict_ms']:.0f} ms, p50 {r['p50_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms, batch "
            f"{r['batch_rps']:.1f} rec/s, peak VRAM {vram} ({prec})")


def run_models(args: argparse.Namespace, plan: GpuPlan, sources: dict[str, Source]) -> tuple[list[dict], list[dict]]:
    results, errors = [], []
    for model in plan.models:
        print(f"{PROG}: {model}: timing in a fresh process ...", flush=True)
        rows, errs = run_model(args, plan, sources[model])
        results += rows
        errors += errs
        for r in [*rows, *errs]:
            print(_line(r), flush=True)
    return results, errors


# ---------------------------------------------------------------- parent

def hardware_label(m: Mapping[str, Any]) -> str:
    """One line for a results table: the GPU, its CUDA stack and the host CPU."""
    gpu = (f"{m['gpu']} (cc {m.get('capability')}, {m.get('vram_total_gb'):g} GiB)" if m.get("gpu")
           else "no CUDA GPU")
    stack = "".join(f", {k.upper() if k == 'cuda' else k} {m[k]}" for k in ("cuda", "driver") if m.get(k))
    return f"{gpu}{stack}; host {m.get('cpu_model') or 'unknown CPU'} ({m.get('cpu_count')} vCPUs)"


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, t0 = load_config(args.config), time.perf_counter()
    plan = make_plan(args, cfg)
    check_device(plan.device)
    load_states(args.rows, 1)  # the rows file exists and has a state before anything is downloaded
    sources = resolve_sources(cfg, plan)
    print(f"{PROG}: {', '.join(f'{m} ({plan.backends[m]})' for m in plan.models)} on {plan.device}; "
          f"{plan.latency_n} latency / {plan.batch_n} batch rows, batch sizes {list(plan.batch_sizes)}; one fresh "
          "process per model", flush=True)
    results, errors = run_models(args, plan, sources)
    machine = {**gpu_facts(), **machine_info()}  # after the workers: this parent never holds a CUDA context early
    hardware = hardware_label(machine)
    return {"schema_version": SCHEMA_VERSION, "plan": plan.as_dict(str(args.rows)),
            "sources": {m: s.as_dict() for m, s in sources.items()}, "machine": machine, "hardware": hardware,
            "results": [{**r, "hardware": hardware} for r in results], "errors": errors,
            "seconds": round(time.perf_counter() - t0, 2)}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=f"laya_poc.{PROG}", description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=None, help="default: config phase5.bench.models")
    p.add_argument("--rows", required=True, help="JSONL rows (test_id.jsonl); the first N states are used")
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", nargs="+", default=None, help="'hub' (default), one absolute dir, or MODEL=/abs/dir")
    p.add_argument("--latency-n", type=int, default=LATENCY_N, help="batch-1 latency rows")
    p.add_argument("--batch-n", type=int, default=BATCH_N, help="rows of the one timed batched call")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    p.add_argument("--warmup", type=int, default=WARMUP, help="untimed batch-1 calls before the latency rows")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--timeout-min", type=float, default=TIMEOUT_MIN, help="per model")
    p.add_argument("--config", default=None)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        from .bench_gpu_worker import worker_main

        return run_cli(f"{PROG} worker", lambda: worker_main(args, load_config(args.config), load_states), None)

    def body() -> int:
        res = run(args)
        write_json(args.out, res)
        print(f"{PROG}: wrote {args.out} ({len(res['results'])} result rows, {len(res['errors'])} errors)")
        return 0 if res["results"] else 1

    return run_cli(PROG, body, args.out)


if __name__ == "__main__":
    sys.exit(main())
