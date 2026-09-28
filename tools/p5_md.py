"""Markdown of the Phase 5 CPU benchmark notebook: the intro (what, connection, recovery, estimates), one note per
step and the closing notes. The estimates come from config.yaml phase5.bench and the assumptions stated below."""
from __future__ import annotations

import math

from p5_cells import ARCHIVE_NAME
from p5_commands import BENCH_OUT, ONNX_PINS, P3_ROOT, P4_ROOT, P5_COLAB, bench_models, runbook_commands, shell

RUNTIME = "4-6 h"
# Estimate assumptions: seconds per record at 1 thread on one Colab CPU core, fp32, as (batch-1, batched). laya: the
# Phase 2 Colab host measured the fine-tuned laya at p50 1.5 s and 0.6 rec/s batched on 1 thread
# (docs/results/phase2_gate_report_2026-09-26.md); laya_ml ran ~3x faster than laya on the laptop
# (docs/results/early_cpu_bench_2026-09-26.md); the small encoders read the record only (<= 256 tokens, no question
# or options), assumed 10-25x faster than laya. ONNX is assumed no faster than PyTorch (an upper bound).
SEC_PER_RECORD = {"laya": (1.5, 1.65), "laya_ml": (0.5, 0.55), "modernbert_base": (0.12, 0.08),
                  "mmbert_small": (0.06, 0.04)}
SPEEDUP = {1: 1.0, 2: 1.65, 4: 2.6, 8: 3.5}  # thread scaling (Phase 2: 1 -> 2 threads gave 1.66x)
CORES = 4          # physical cores behind a Colab CPU High-RAM runtime's 8 vCPUs (hyperthreads): 8 threads skipped
COLD_S = 30        # load + first call per worker process
EXPORT_MIN = 2.0   # ONNX export of one Laya model
SETUP_MIN = "10-25"  # Steps 1-5: install, token, env, the frozen data build (CPU-bound on any runtime)


def setting_seconds(cfg: dict, model: str, threads: int) -> float:
    """One (model, backend, threads) worker: cold start, warm-up + batch-1 latency rows, then the batched rows."""
    b, (one, batched) = cfg["phase5"]["bench"], SEC_PER_RECORD.get(model, SEC_PER_RECORD["laya"])
    s = SPEEDUP.get(threads, SPEEDUP[max(SPEEDUP)])
    return COLD_S + ((b["warmup"] + b["latency_n"]) * one + b["batch_n"] * batched) / s


def model_minutes(cfg: dict, model: str, cores: int = CORES) -> float:
    """Every backend x measured thread setting of one model, plus the ONNX export and acceptance check (check_n rows
    through both backends) when it has an ONNX backend."""
    b = cfg["phase5"]["bench"]
    threads = [int(t) for t in b["threads"] if int(t) <= cores]
    backends = b["backends"].get(model, ["torch"])
    total = sum(setting_seconds(cfg, model, t) for t in threads) * len(backends) / 60
    if "onnx" in backends:
        batched = SEC_PER_RECORD.get(model, SEC_PER_RECORD["laya"])[1]
        total += EXPORT_MIN + 2 * b["onnx"]["check_n"] * batched / SPEEDUP[min(cores, 4)] / 60
    return total


def span(minutes: float) -> str:
    """'~lo-hi min' (5-minute steps; hi is 1.5x: nothing here is measured on the Colab CPU yet)."""
    lo = max(5, 5 * round(minutes / 5))
    return f"~{lo}-{max(lo + 5, 5 * math.ceil(minutes * 1.5 / 5))} min"


def _table(cfg: dict) -> str:
    b = cfg["phase5"]["bench"]
    lines = ["| Model | Backends | Threads measured (4 cores) | Estimate |", "|---|---|---|---|"]
    for m in bench_models(cfg):
        threads = [str(t) for t in b["threads"] if int(t) <= CORES]
        lines.append(f"| `{m}` | {', '.join(b['backends'].get(m, ['torch']))} | {', '.join(threads)} | "
                     f"{span(model_minutes(cfg, m))} |")
    total = sum(model_minutes(cfg, m) for m in bench_models(cfg)) / 60
    return "\n".join(lines) + f"\n\nBenchmark total: about {total:.1f} h (estimate)."


def intro_md(cfg: dict, sha256: str, n_files: int) -> str:
    b, budget = cfg["phase5"]["bench"], cfg["cpu_budget"]
    return f"""# Laya PoC - Phase 5: CPU and ONNX benchmark (Colab CPU High-RAM from VS Code)

Design doc §7.11 and §5.11 "CPU", for decision criterion 5 (§5.12): on {budget['vcpus']} vCPUs, fp32, the faster of \
PyTorch and ONNX, a single record's p95 at batch 1 must be <= {budget['p95_ms']} ms and batched throughput >= \
{budget['min_rps']} records/s; stop if p95 exceeds {cfg['phase5']['decision']['stop_p95_ms']} ms even with ONNX on 8 \
threads. This notebook measures every model of `config.yaml` → `phase5.bench` on a 4-8-core cloud CPU: \
{', '.join(f'`{m}`' for m in bench_models(cfg))}, threads {', '.join(map(str, b['threads']))} (capped at the \
physical cores), a fresh process per setting. Latency does not depend on the fine-tuned weights, so it uses the \
pinned Hub checkpoints. Runbook: `docs/phase5_runbook.md`.

{_table(cfg)}

**Expected runtime:** about {RUNTIME} in total: setup and the frozen data {SETUP_MIN} min, then the benchmark (above; \
assumptions in `tools/p5_md.py`). Set `QUICK = True` in Step 6 first for a plumbing check in minutes.

**How to connect:** **Select Kernel → Colab → New Colab Server → CPU → High-RAM** (8 vCPUs, Colab Pro), then \
the Python 3 kernel. Do **not** use *Auto Connect* (a standard CPU has 2 vCPUs: no 4-thread row, so criterion 5 could \
not be judged). No GPU is needed; Step 4 warns if one is attached. Keep VS Code open and the laptop awake: a running \
cell keeps the server alive. Run nothing else on the server while Step 6 runs. The Hugging Face token (Step 3, masked \
input box, never stored in this notebook) is needed for the frozen data build only.

**After a kernel restart or a Colab disconnect:** re-run **Step 1**, then the interrupted step (or **Run All**). \
Finished steps skip themselves; an interrupted benchmark starts again from its first setting (partial timings are \
never mixed). If the server itself was removed, `/content` is gone: start again from Step 1.

Code bundle: {n_files} files, sha256 `{sha256}` (verified when Step 1 unpacks it).
"""


def step_notes(cfg: dict) -> dict[str, tuple[str, str]]:
    """(title, note) per cell key, in notebook order."""
    b, onnx = cfg["phase5"]["bench"], cfg["phase5"]["bench"]["onnx"]
    pins = ", ".join(f"`{p}` {v}" for p, v in ONNX_PINS.items())
    notes = {
        "setup": ("Setup", "Non-interactive environment, work dir, code bundle, helpers. Seconds. **Re-run this cell "
                  "first after any kernel restart or Colab disconnect.**"),
        "install": ("Install", f"Plain `laya` at the pinned commit, DuckDB pinned to {cfg['data']['duckdb_version']}, "
                    f"this project, and {pins} (the tested versions), constrained to the runtime's own torch, "
                    "transformers, protobuf and numpy. Fails if pip changed any of those four. 1-3 min."),
        "token": ("Hugging Face token", "Masked input box at the top of VS Code; never printed. Only the data build "
                  "needs it. Unattended: `python tools/build_p5_bench_notebook.py --with-token` and open the "
                  "gitignored `notebooks/phase5_cpu_bench.local.ipynb`."),
        "env": ("Environment check", "Versions, Laya commit, RAM, disk -> `runs/p5/env.json`; then the CPU itself "
                "(model, logical CPUs, physical cores, affinity, cgroup quota, AVX-512/AMX flags) -> "
                "`runs/p5/cpu.json`. Warns when a GPU is attached, when there are fewer than 4 physical cores and "
                "which thread settings will be skipped. ~30 s."),
        "data": ("Full data (frozen)", "The E1 build, identical: FULL extraction over `hf://`, then `--verify-frozen` "
                 "(the fingerprint must equal `data.frozen_manifest`). The benchmark reads the frozen "
                 f"`test_id.jsonl`. {SETUP_MIN} min with Steps 1-4."),
        "bench": ("CPU benchmark", f"`python -m laya_poc.bench_cpu --models ...`: per model, backend and thread count "
                  f"a fresh worker: cold start, warm-up ({b['warmup']}), batch-1 latency p50/p95/mean over "
                  f"{b['latency_n']} `test_id` records, batched throughput over {b['batch_n']} (batch "
                  f"{b['batch_size']}, length-sorted), peak RSS. For each Laya model it exports ONNX (opset "
                  f"{onnx['opset']}, to `runs/p5/onnx/`; the graphs are never archived) and checks it against "
                  f"PyTorch fp32 on {onnx['check_n']} records: argmax agreement >= {onnx['min_argmax_agree']}, max "
                  "|Δp| <= "
                  f"{onnx['max_dp']}. Writes `runs/p5/{BENCH_OUT}`. {RUNTIME} (see the table above); `QUICK = "
                  "True` is a few-minute plumbing check with its own output file."),
        "archive": ("Archive", "---\nZip `env.json`, `cpu.json`, the benchmark JSON, the ONNX export records, logs, "
                    f"the data report and the FSQ NOTICE into `{ARCHIVE_NAME}` (no ONNX graphs, no weights, no FSQ "
                    "rows) for **Download...**. Seconds."),
    }
    numbers = {k: str(i) for i, k in enumerate(notes, start=1)}
    return {k: (f"Step {numbers[k]} - {title}", note) for k, (title, note) in notes.items()}


def finish_md(cfg: dict) -> str:
    report = shell(*runbook_commands(cfg)["report_colab"], py="python")
    return f"""### Finish
- Download `{ARCHIVE_NAME}`: Colab view in the activity bar > Contents > right-click > **Download...**
- Locally, unzip it to `runs/p5_colab` (gitignored: `{P5_COLAB}/{BENCH_OUT}`) and pass the benchmark to the
  Phase 5 report, whose metrics come from the Phase 3 and Phase 4 archives (`{P3_ROOT}`, `{P4_ROOT}`; runbook):
  `{report}`
- Run **Colab: Remove Server** when you are done. It stops the compute-unit spend and deletes `/content`.
"""
