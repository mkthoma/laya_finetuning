"""Helper functions tools/build_p7_gpu_notebook.py embeds verbatim in the Phase 7 GPU notebook's Step 1, after the
Phase 1, E1 and Phase 3 helpers (tools/notebook_helpers.py, e1_helpers.py, p3_helpers.py).

Same rules as there: each is copied with inspect.getsource, so it may use only json, os, shutil, subprocess,
time, Path and the other helpers (Step 1 imports and defines those); module constants here do not exist in the
notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

from e1_helpers import trainer_pids  # an E1 helper: Step 1 defines it before these


def gpu_bench_pids() -> list:
    """A GPU benchmark (or its worker) still running, e.g. from before a kernel restart: two would share the GPU and
    distort every timing, so the benchmark step refuses to start."""
    return sorted(set(trainer_pids("laya_poc.bench_gpu")))


def gpu_row_line(row) -> str:
    """One benchmark row (model, backend, batch size) or error for the cell output."""
    def num(key, nd=0):
        value = row.get(key)
        return f"{value:.{nd}f}" if isinstance(value, (int, float)) else "n/a"

    size = row.get("batch_size")
    head = f"{row.get('model', '?')} {row.get('backend', '?')}" + (f" bs {size}" if size is not None else "")
    if row.get("error"):
        return f"{head}: FAILED: {row['error']}"
    return (f"{head}: cold {num('cold_s', 1)} s, first {num('first_predict_ms')} ms, p50 {num('p50_ms', 1)} ms, p95 "
            f"{num('p95_ms', 1)} ms, batch {num('batch_rps', 1)} rec/s, peak VRAM {num('peak_vram_gb', 2)} GiB")


def gpu_bench_summary(path) -> dict:
    """Print the GPU benchmark JSON briefly: the machine, one line per measured row and per error; return it all."""
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not obj:
        print(f"no benchmark results in {path}")
        return obj
    if obj.get("hardware"):
        print(f"  machine: {obj['hardware']}")
    for row in [*(obj.get("results") or []), *(obj.get("errors") or [])]:
        print("  " + gpu_row_line(row))
    return obj


P7_HELPERS = (gpu_bench_pids, gpu_row_line, gpu_bench_summary)
