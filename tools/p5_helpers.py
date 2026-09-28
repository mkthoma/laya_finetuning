"""Helper functions tools/build_p5_bench_notebook.py embeds verbatim in the Phase 5 CPU notebook's Step 1, after the
Phase 1, E1 and Phase 3 helpers (tools/notebook_helpers.py, e1_helpers.py, p3_helpers.py).

Same rules as there: each is copied with inspect.getsource, so it may use only json, os, shutil, subprocess,
time, Path and the other helpers (Step 1 imports and defines those); module constants here do not exist in the
notebook.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from e1_helpers import trainer_pids  # an E1 helper: Step 1 defines it before these


def cpuinfo_facts(text) -> dict:
    """CPU model, physical cores (distinct physical id + core id pairs) and the ISA extensions that matter for CPU
    inference speed, from the text of /proc/cpuinfo ('' elsewhere)."""
    model, phys, cores, flags = None, None, set(), set()
    for line in str(text).splitlines():
        key, _, value = (part.strip() for part in line.partition(":"))
        if key == "model name" and model is None:
            model = value
        elif key == "physical id":
            phys = value
        elif key == "core id":
            cores.add((phys, value))
        elif key == "flags" and not flags:
            flags = set(value.split())
    isa = [f for f in ("avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_tile") if f in flags]
    return {"cpu_model": model, "physical_cores": len(cores) or None, "isa": isa}


def cgroup_cpus(base="/sys/fs/cgroup"):
    """The container's CPU quota in CPUs (cgroup v2 cpu.max, else v1 CFS), None when unlimited or unknown: a quota
    below the core count throttles multi-threaded timings."""
    base = Path(base)
    try:
        if (base / "cpu.max").exists():
            quota, period = (base / "cpu.max").read_text(encoding="utf-8").split()[:2]
            return None if quota == "max" else round(int(quota) / int(period), 2)
        quota_file, period_file = base / "cpu" / "cpu.cfs_quota_us", base / "cpu" / "cpu.cfs_period_us"
        if quota_file.exists() and period_file.exists():
            quota = int(quota_file.read_text(encoding="utf-8"))
            return None if quota < 0 else round(quota / int(period_file.read_text(encoding="utf-8")), 2)
    except (OSError, ValueError):
        pass
    return None


def cpu_facts() -> dict:
    """What the benchmark runs on (Phase 5 spec §5): CPU model, logical CPUs, physical cores, the CPUs this process
    may use (affinity, cgroup quota), ISA extensions, RAM. Linux /proc on Colab; best effort elsewhere."""
    import platform

    path = Path("/proc/cpuinfo")
    facts = cpuinfo_facts(path.read_text(encoding="utf-8", errors="replace") if path.exists() else "")
    ram = None
    try:
        import psutil  # the benchmark caps threads at psutil's physical core count: use the same number
        facts["physical_cores"] = psutil.cpu_count(logical=False) or facts["physical_cores"]
        ram = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {**facts, "cpu_model": facts["cpu_model"] or platform.processor() or None, "logical_cpus": os.cpu_count(),
            "affinity_cpus": affinity, "cgroup_cpus": cgroup_cpus(), "ram_gb": ram, "platform": platform.platform()}


def cpu_notes(cpu, env, threads, min_cores=4) -> list:
    """Step 4's warnings: a GPU is attached (the benchmark is CPU-only), too few cores for criterion 5 (judged at
    min_cores threads), thread settings the benchmark will skip (capped at physical cores), a CPU quota."""
    notes = []
    if env.get("cuda_available") or env.get("gpu_name"):
        notes.append(f"a GPU is attached ({env.get('gpu_name') or 'CUDA'}): the benchmark runs on the CPU only (every "
                     "worker hides CUDA), on this GPU host's CPU. Prefer Remove Server and a CPU runtime with "
                     "High-RAM (8 vCPUs), which is what criterion 5 describes")
    cores = cpu.get("physical_cores")
    if not cores:
        return notes + ["physical core count unknown: the benchmark caps the thread settings itself"]
    if cores < min_cores:
        notes.append(f"only {cores} physical core(s): decision criterion 5 is judged at {min_cores} threads, and "
                     f"that row will be missing (settings above {cores} are skipped). Remove Server and connect "
                     "CPU + High-RAM (8 vCPUs)")
    skipped = [t for t in threads if int(t) > cores]
    if skipped:
        notes.append(f"thread settings {skipped} exceed the {cores} physical cores and will be skipped (spec: "
                     "capped at physical cores); the report shows them as not measured on this machine")
    quota = cpu.get("cgroup_cpus")
    if quota and quota < min(cores, max(int(t) for t in threads)):
        notes.append(f"the container's CPU quota is {quota} CPUs: multi-threaded timings above it are throttled")
    return notes


def bench_pids() -> list:
    """A benchmark (or its worker, export or check) still running, e.g. from before a kernel restart: it would
    compete for the cores and distort every timing, so the benchmark step refuses to start."""
    modules = ("laya_poc.bench_cpu", "laya_poc.bench_hf", "laya_poc.onnx_export")
    return sorted({pid for module in modules for pid in trainer_pids(module)})


def bench_row_line(row) -> str:
    """One benchmark row (model, backend, threads) for the cell output."""
    def num(key, nd=0):
        value = row.get(key)
        return f"{value:.{nd}f}" if isinstance(value, (int, float)) else "n/a"

    head = f"{row.get('model', '?')} {row.get('backend', '?')} {row.get('threads', '?')}t"
    if row.get("error"):
        return f"{head}: FAILED: {row['error']}"
    return (f"{head}: cold {num('cold_s', 1)} s, p50 {num('p50_ms')} ms, p95 {num('p95_ms')} ms, batch "
            f"{num('batch_rps', 2)} rec/s, peak RSS {num('peak_rss_gb', 2)} GiB")


def onnx_line(model, entry) -> str:
    """One Laya model's ONNX export and acceptance check (bench_cpu JSON `onnx`: {model: {accepted, acceptance,
    export, error}}) for the cell output."""
    entry = entry if isinstance(entry, dict) else {"accepted": entry}
    acc, export = entry.get("acceptance") or {}, entry.get("export") or {}
    brief = {"accepted": entry.get("accepted", acc.get("passed")), "exporter": export.get("exporter"),
             **{k: acc[k] for k in ("n", "argmax_agree", "max_dp") if k in acc}}
    if entry.get("error"):
        brief["error"] = str(entry["error"])[:120]
    return f"ONNX {model}: {json.dumps(brief, default=str)}"


def bench_summary(path) -> dict:
    """Print the benchmark JSON briefly: the machine, one line per measured row, the failed and skipped settings,
    the ONNX export and acceptance check per Laya model; return the whole object."""
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not obj:
        print(f"no benchmark results in {path}")
        return obj
    if obj.get("hardware"):
        print(f"  machine: {obj['hardware']}")
    for row in [*(obj.get("results") or []), *(obj.get("errors") or [])]:
        print("  " + bench_row_line(row))
    for row in [*(obj.get("threads_skipped") or []), *(obj.get("skipped_models") or [])]:
        print(f"  skipped: {json.dumps(row, ensure_ascii=False)[:160]}")
    for model, entry in (obj.get("onnx") or {}).items():
        print("  " + onnx_line(model, entry))
    return obj


P5_HELPERS = (cpuinfo_facts, cgroup_cpus, cpu_facts, cpu_notes, bench_pids, bench_row_line, onnx_line,
              bench_summary)
