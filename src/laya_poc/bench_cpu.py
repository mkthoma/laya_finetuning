"""CPU inference benchmark (design doc §7.11, §5.11 CPU row, §5.12 criterion 5): Laya on PyTorch and on ONNX Runtime,
and the small encoders (B4) for the relative-throughput row, over a thread sweep.

One FRESH subprocess per (model, backend, thread count) setting, so thread pools, cold start and peak RSS are clean
(Laya's Agent ignores LAYA_THREADS; the worker sets torch.set_num_threads / ORT intra_op_num_threads and pins
OMP/MKL/OpenBLAS; GPU hidden; fp32: LAYA_CPU_AMP dropped). Every source is resolved to a local directory and every
ONNX file exported (and checked, §7.11 acceptance) BEFORE any timing; workers run offline. What a worker times:
bench_worker.py. Latency does not depend on fine-tuned weights, so the pinned base checkpoints (`--ckpt hub`) are valid.

Phase 5 sweep form (one JSON for every model):
    python -m laya_poc.bench_cpu --models laya laya_ml modernbert_base mmbert_small --rows <test_id.jsonl> --out <json>
        [--threads 1 2 4 8] [--backends torch onnx hf] [--quick] [--ckpt hub | MODEL=/abs/dir ...]
        [--latency-n 500] [--batch-n 1000] [--warmup 20] [--batch-size 32] [--check-n 1000] [--opset 18]
        [--onnx-dir <dir; default runs/onnx>] [--thread-cap physical|logical|none] [--reexport]
        [--timeout-min 240] [--config yaml]
  defaults: config phase5.bench (backends per model: laya/laya_ml torch+onnx, small encoders hf; `torch` is an alias
  of `hf` for them); threads above the physical cores are skipped with a note; --quick = QUICK row counts.
Phase 2 single-model form (E1 notebook; flags unchanged; config `bench` defaults; no thread cap):
    python -m laya_poc.bench_cpu --ckpt <abs Laya dir>|hub [--model laya] --rows <jsonl> --out <json> [--n 200]
        [--threads 1 2] [--warmup 20] [--batch-size 32] [--backend torch|onnx|hf] [--timeout-min] [--config]

Output JSON (both forms; `form` says which; schema_version 2):
  form, schema_version, models, backends {model: [backend]}, sources {model: {kind, path, label}}, rows (input JSONL
  path), quick, latency_n, batch_n, warmup, batch_size, check_n, threads_requested, threads_run,
  threads_skipped [{threads, reason}], thread_cap {basis, limit, note}, skipped_models [{model, reason}],
  machine {cpu_count, physical_cores, physical_cores_source, cpu_model, ram_gb, platform, python, versions{...}},
  hardware (one-line label), budget {p95_ms, min_rps, vcpus, note}, seconds,
  results: one row per successful setting:
    {model, backend (torch|onnx|hf), framework (pytorch|onnxruntime), threads, torch_threads, batch_size,
     n_latency, n_batch, n (= n_latency), warmup, cold_s, first_predict_ms, p50_ms, p95_ms, mean_ms, max_ms (batch-1
     latency, ms), batch_rps, batch_seconds (one batched call over n_batch rows), peak_rss_gb, peak_rss_source,
     device, dtype, amp, cpu_count, physical_cores, platform, pid, source (checkpoint label), hardware,
     p95_ok, rps_ok (config cpu_budget, INFORMATION only), onnx_accepted (onnx rows: the model's acceptance result),
     + torch/param_dtype (torch), ort_intra_op_threads/onnxruntime/onnx_path (onnx), max_length/... (hf)}
  errors: [{model, backend, threads, returncode, error}] (a failed/timed-out setting or a failed ONNX export),
  onnx: {model: {path, ckpt, reused, export {exporter, dynamo_arg, dummy_batch, attempt, upstream_verbatim,
         failed_attempts, opset, warnings, probe, bytes, seconds}, acceptance {n, argmax_agree, n_disagree, max_dp,
         max_dlogit, temperature, seq_len_range, min_argmax_agree, max_dp_threshold, passed, ...} | null,
         accepted (true|false|null = not checked), error?}} for every Laya model run with the onnx backend.
  The single form also carries the Phase 2 keys: model, ckpt, backend, n, thread_settings, cpu_count, platform, python.
Criterion 5 reads the 4-thread rows (fp32, the faster accepted backend); the stop check the 8-thread ONNX row, which a
4-physical-core host skips unless --thread-cap logical. Exit code 0 when at least one setting produced a result.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .bench_machine import _windows_peak_bytes, hardware_label, machine_info, peak_rss_gb  # noqa: F401 (re-export)
from .bench_plan import SINGLE, Plan, cap_threads, make_plan, resolve_sources
from .bench_sources import Source
from .bench_worker import MARKER, latency_stats, measure  # noqa: F401 (re-exported Phase 2 API)
from .config import load_config
from .env_check import run_cli, write_json
from .io_utils import iter_jsonl

SCHEMA_VERSION = 2
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LAYA_THREADS")
BUDGET_NOTE = ("information only: the decision rule (Phase 5, design §5.12 criterion 5) judges the CPU budget on "
               "{vcpus} threads/vCPUs, fp32, with the faster of PyTorch and ONNX")
PHASE2_KEYS = ("cpu_count", "platform", "python")


# ---------------------------------------------------------------- pure helpers

def budget_flags(result: Mapping[str, Any], budget: Mapping[str, Any]) -> dict[str, bool]:
    return {"p95_ok": bool(result["p95_ms"] <= budget["p95_ms"]),
            "rps_ok": bool(result["batch_rps"] >= budget["min_rps"])}


def parse_worker_output(stdout: str) -> dict[str, Any] | None:
    """The worker's result: the last line starting with MARKER (Laya may print warnings around it)."""
    lines = [ln[len(MARKER):] for ln in stdout.splitlines() if ln.startswith(MARKER)]
    return json.loads(lines[-1]) if lines else None


def worker_env(threads: int, base: Mapping[str, str]) -> dict[str, str]:
    """A copy of `base` pinning every thread pool to `threads`, hiding CUDA, dropping CPU autocast, offline Hub."""
    env = {k: v for k, v in base.items() if k != "LAYA_CPU_AMP"}
    env.update({k: str(threads) for k in THREAD_VARS})
    env.update(CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
    return env


def load_states(path: str | Path, n: int) -> list[str]:
    states = [r["state"] for r in itertools.islice(iter_jsonl(path), n)]
    if not states:
        raise ValueError(f"no rows in {path}")
    return states


def _last_line(text: str | None, fallback: str) -> str:
    tail = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return tail[-1] if tail else fallback


# ---------------------------------------------------------------- subprocesses

def run_worker(cmd: list[str], env: dict[str, str], timeout: float) -> Any:
    return subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", stdin=subprocess.DEVNULL)


def worker_cmd(args: argparse.Namespace, plan: Plan, source: Source, backend: str, threads: int,
               onnx_path: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "laya_poc.bench_cpu", "--worker-threads", str(threads), "--worker-backend", backend,
           "--model", source.model, "--ckpt", str(source.path), "--rows", str(args.rows), "--out", str(args.out),
           "--latency-n", str(plan.latency_n), "--batch-n", str(plan.batch_n), "--warmup", str(plan.warmup),
           "--batch-size", str(plan.batch_size)]
    cmd += ["--onnx-path", onnx_path] if onnx_path else []
    return cmd + (["--config", str(args.config)] if args.config else [])


def onnx_cmd(args: argparse.Namespace, plan: Plan, source: Source) -> list[str]:
    cmd = [sys.executable, "-m", "laya_poc.onnx_export", "--model", source.model, "--ckpt", str(source.path),
           "--out", str(plan.onnx_dir / f"{source.model}.onnx"), "--opset", str(plan.opset),
           "--check-rows", str(Path(args.rows).resolve()), "--check-n", str(plan.check_n),
           "--batch-size", str(plan.batch_size)]
    cmd += [] if args.reexport else ["--reuse"]
    return cmd + (["--config", str(args.config)] if args.config else [])


def _mtime(path: Path) -> int | None:
    return path.stat().st_mtime_ns if path.is_file() else None


def prepare_onnx(args: argparse.Namespace, plan: Plan, source: Source, threads: int) -> dict[str, Any]:
    """Export + acceptance check in a fresh process (onnx_export CLI); the record for the JSON's `onnx` block.
    The <model>.onnx.json counts only when this run wrote it or explicitly reused it (never a stale one)."""
    path, meta_path = plan.onnx_dir / f"{source.model}.onnx", plan.onnx_dir / f"{source.model}.onnx.json"
    print(f"bench_cpu: {source.model}: ONNX export + acceptance on {plan.check_n} rows -> {path} ...", flush=True)
    before = _mtime(meta_path)
    try:
        proc = run_worker(onnx_cmd(args, plan, source), worker_env(threads, os.environ), args.timeout_min * 60)
    except subprocess.TimeoutExpired:
        return {"path": str(path), "error": f"timed out after {args.timeout_min} min", "accepted": None}
    reused = "reusing" in (proc.stdout or "")
    fresh = _mtime(meta_path) is not None and (reused or _mtime(meta_path) != before)
    if not fresh:
        why = _last_line(proc.stderr, f"no new {meta_path.name} (exit code {proc.returncode})")
        print(f"bench_cpu: {source.model}: ONNX export FAILED: {why}", flush=True)
        return {"path": str(path), "error": why, "accepted": None}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    acc = meta.get("acceptance")
    rec = {"path": str(path), "ckpt": meta.get("ckpt"), "reused": reused, "export": meta.get("export"),
           "acceptance": acc, "accepted": None if acc is None else bool(acc["passed"])}
    print(f"bench_cpu: {source.model}: {_last_line(proc.stdout, 'ONNX ready')}", flush=True)
    return rec


def run_setting(args: argparse.Namespace, plan: Plan, source: Source, backend: str, threads: int,
                budget: Mapping[str, Any], onnx_path: str | None = None) -> dict[str, Any]:
    """One worker process: its result row (plus budget flags), or {model, backend, threads, returncode, error}."""
    who = {"model": source.model, "backend": backend, "threads": threads}
    cmd = worker_cmd(args, plan, source, backend, threads, onnx_path)
    try:
        proc = run_worker(cmd, worker_env(threads, os.environ), args.timeout_min * 60)
    except subprocess.TimeoutExpired:
        return {**who, "returncode": None, "error": f"timed out after {args.timeout_min} min"}
    res = parse_worker_output(proc.stdout or "")
    if proc.returncode != 0 or res is None:
        why = _last_line(proc.stderr, f"no result line (exit code {proc.returncode})")
        return {**who, "returncode": proc.returncode,
                "error": why if proc.returncode != 0 else f"no result line: {why}"}
    return {**res, **who, **budget_flags(res, budget), "source": source.label}


def _line(r: dict[str, Any]) -> str:
    who = f"bench_cpu: {r['model']}/{r['backend']} {r['threads']} thread(s)"
    if "error" in r:
        return f"{who}: FAILED: {r['error']}"
    return (f"{who}: cold {r['cold_s']:.1f}s, p50 {r['p50_ms']:.0f} ms, p95 {r['p95_ms']:.0f} ms, batch "
            f"{r['batch_rps']:.1f} rec/s, peak RSS {r['peak_rss_gb']} GiB (budget info: p95 "
            f"{'ok' if r['p95_ok'] else 'over'}, rps {'ok' if r['rps_ok'] else 'under'})")


def run_settings(args: argparse.Namespace, plan: Plan, sources: dict[str, Source], onnx: dict[str, dict],
                 threads: list[int], budget: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    results, errors = [], []
    for model, backend in ((m, b) for m in plan.models for b in plan.backends[m]):
        rec = onnx.get(model) if backend == "onnx" else None
        if rec is not None and "error" in rec:
            errors.append({"model": model, "backend": backend, "threads": None, "returncode": None,
                           "error": f"ONNX export failed: {rec['error']}"})
            continue
        for t in threads:
            r = run_setting(args, plan, sources[model], backend, t, budget, rec["path"] if rec else None)
            if rec is not None and "error" not in r:
                r = {**r, "onnx_accepted": rec["accepted"]}
            (errors if "error" in r else results).append(r)
            print(_line(r), flush=True)
    return results, errors


# ---------------------------------------------------------------- parent

def _header(plan: Plan, threads: list[int], skipped: list[dict]) -> str:
    what = ", ".join(f"{m} ({'+'.join(plan.backends[m])})" for m in plan.models)
    skip = f" (skipped {[s['threads'] for s in skipped]}: {skipped[0]['reason']})" if skipped else ""
    return (f"bench_cpu: {what}; threads {threads}{skip}; {plan.latency_n} latency / {plan.batch_n} batch rows"
            f"{' (quick)' if plan.quick else ''}; one fresh process per setting")


def _single_keys(plan: Plan, machine: dict, threads: list[int]) -> dict[str, Any]:
    """The Phase 2 JSON keys (gate.py and the E1 notebook read them)."""
    model = plan.models[0]
    return {"model": model, "ckpt": plan.ckpt[model], "backend": plan.backends[model][0], "n": plan.latency_n,
            "thread_settings": threads, **{k: machine[k] for k in PHASE2_KEYS}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, t0 = load_config(args.config), time.perf_counter()
    plan = make_plan(args, cfg)
    load_states(args.rows, 1)  # the rows file exists and has a state before anything is downloaded
    machine = machine_info()
    threads, skipped, cap = cap_threads(plan.threads, plan.thread_cap, machine)
    if not threads:
        raise ValueError(f"no thread setting fits this machine: {skipped[0]['reason']}")
    sources = resolve_sources(cfg, plan)
    budget = dict(cfg["cpu_budget"])
    print(_header(plan, threads, skipped), flush=True)
    build = machine.get("physical_cores") or machine.get("cpu_count") or 1
    onnx = {m: prepare_onnx(args, plan, sources[m], build) for m in plan.needs_onnx()}
    results, errors = run_settings(args, plan, sources, onnx, threads, budget)
    hardware = hardware_label(machine)
    out = {"form": plan.form, "schema_version": SCHEMA_VERSION, "models": list(plan.models),
           "backends": plan.backends, "sources": {m: s.as_dict() for m, s in sources.items()}, "rows": str(args.rows),
           "quick": plan.quick, "latency_n": plan.latency_n, "batch_n": plan.batch_n, "warmup": plan.warmup,
           "batch_size": plan.batch_size, "check_n": plan.check_n, "threads_requested": list(plan.threads),
           "threads_run": threads, "threads_skipped": skipped, "thread_cap": cap,
           "skipped_models": list(plan.skipped_models), "machine": machine, "hardware": hardware,
           "results": [{**r, "hardware": hardware} for r in results], "errors": errors, "onnx": onnx,
           "budget": {**budget, "note": BUDGET_NOTE.format(vcpus=budget.get("vcpus", 4))},
           "seconds": round(time.perf_counter() - t0, 2)}
    return {**out, **_single_keys(plan, machine, threads)} if plan.form == SINGLE else out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.bench_cpu", description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=None, help="Phase 5 sweep: laya laya_ml modernbert_base ...")
    p.add_argument("--model", default=None, help="Phase 2 single-model form (default laya)")
    p.add_argument("--ckpt", nargs="+", default=None, help="'hub' (default), one absolute dir, or MODEL=/abs/dir")
    p.add_argument("--rows", required=True, help="JSONL rows (test_id.jsonl); the first N states are used")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, nargs="+", default=None)
    p.add_argument("--backends", nargs="+", choices=("torch", "onnx", "hf"), default=None, help="sweep form")
    p.add_argument("--backend", choices=("torch", "onnx", "hf"), default="torch", help="single form")
    for flag in ("--n", "--latency-n", "--batch-n", "--warmup", "--batch-size", "--check-n", "--opset"):
        p.add_argument(flag, type=int, default=None)
    p.add_argument("--onnx-dir", default=None, help="exported .onnx files (default <repo>/runs/onnx, gitignored)")
    p.add_argument("--thread-cap", choices=("physical", "logical", "none"), default=None)
    p.add_argument("--quick", action="store_true", help="smoke run: few rows (bench_plan.QUICK)")
    p.add_argument("--reexport", action="store_true", help="export again even when a matching .onnx exists")
    p.add_argument("--timeout-min", type=float, default=240.0, help="per setting (and per ONNX export)")
    p.add_argument("--config", default=None)
    p.add_argument("--worker-threads", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-backend", default="torch", help=argparse.SUPPRESS)
    p.add_argument("--onnx-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _worker(args: argparse.Namespace) -> int:
    from .bench_worker import worker_main

    one = argparse.Namespace(**{**vars(args), "ckpt": (args.ckpt or ["hub"])[0]})  # a worker gets ONE local dir
    return worker_main(one, load_config(args.config), load_states)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_threads is not None:
        return run_cli("bench_cpu worker", lambda: _worker(args), None)

    def body() -> int:
        res = run(args)
        write_json(args.out, res)
        print(f"bench_cpu: wrote {args.out} ({len(res['results'])} result rows, {len(res['errors'])} errors)")
        return 0 if res["results"] else 1

    return run_cli("bench_cpu", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
