"""Markdown of the Phase 7 GPU benchmark notebook: the intro (what, connection, recovery, estimates), one note per
step and the closing notes. The estimates come from the bench_gpu defaults and the assumptions stated below."""
from __future__ import annotations

import math

from p7_cells import ARCHIVE_NAME
from p7_commands import BENCH_OUT, P7_COLAB, bench_models

from laya_poc.bench_gpu import BATCH_N, BATCH_SIZES, LATENCY_N, WARMUP
from laya_poc.bench_gpu_worker import BATCH_WARMUP_BATCHES

RUNTIME = "15-45 min"
SETUP_MIN = "10-25"  # Steps 1-5: install, token, env, the frozen data build (CPU-bound on any runtime)
DOWNLOAD_MIN = 2.0   # the four pinned checkpoints (~3 GB) into the Hub cache, before any timing
COLD_S = 30          # per worker process: imports, weights onto the GPU, the CUDA context
# Estimate assumptions on a Colab T4 with fp16 autocast, as (batch-1 ms per record, batched records/s). Nothing is
# measured on a GPU yet: laya is ModernBERT-large-sized and reads the record plus the question and its 10 options
# (~250 tokens); laya_ml ran ~3x faster than laya on the laptop CPU (docs/results/early_cpu_bench_2026-09-26.md);
# the small encoders read the record only (<= 256 tokens, no question). A G4 is several times faster than a T4.
T4_ASSUMED = {"laya": (35, 60), "laya_ml": (20, 150), "modernbert_base": (12, 800), "mmbert_small": (10, 1500)}


def model_minutes(model: str) -> float:
    """One worker: cold start, the first call + warm-up + batch-1 latency rows, then per batch size its warm-up
    batches and the timed call over BATCH_N rows."""
    one_ms, rps = T4_ASSUMED.get(model, T4_ASSUMED["laya"])
    batch_one = (1 + WARMUP + LATENCY_N) * one_ms / 1000
    batched = sum(BATCH_N + bs * BATCH_WARMUP_BATCHES for bs in BATCH_SIZES) / rps
    return (COLD_S + batch_one + batched) / 60


def span(minutes: float) -> str:
    """'~lo-hi min' (whole minutes; hi is 2x: nothing here is measured on a GPU yet)."""
    lo = max(1, round(minutes))
    return f"~{lo}-{max(lo + 1, math.ceil(minutes * 2))} min"


def _table(cfg: dict) -> str:
    lines = ["| Model | Backend | Assumed on a T4 (batch 1 / batched) | Estimate |", "|---|---|---|---|"]
    for m in bench_models(cfg):
        one_ms, rps = T4_ASSUMED.get(m, T4_ASSUMED["laya"])
        backend = "torch (Laya Agent)" if m in cfg["model"] else "hf (small encoder)"
        lines.append(f"| `{m}` | {backend} | {one_ms} ms / {rps} rec/s | {span(model_minutes(m))} |")
    total = sum(model_minutes(m) for m in bench_models(cfg)) + DOWNLOAD_MIN
    return "\n".join(lines) + f"\n\nBenchmark total on a T4: about {total:.0f} min with the downloads (estimate)."


def intro_md(cfg: dict, sha256: str, n_files: int) -> str:
    sizes = " and ".join(map(str, BATCH_SIZES))
    return f"""# Laya PoC - Phase 7: GPU inference benchmark (Colab GPU from VS Code)

Inference time on a GPU for the models of the CPU benchmark (`config.yaml` → `phase5.bench.models`): \
{', '.join(f'`{m}`' for m in bench_models(cfg))}, each in a fresh process: cold start (load, weights onto the GPU), \
batch-1 latency p50/p95/mean/max over {LATENCY_N} `test_id` records after {WARMUP} warm-up calls (measured once per \
model), and batched throughput in records/s over {BATCH_N} records at batch sizes {sizes} (length-sorted, after \
{BATCH_WARMUP_BATCHES} untimed warm-up batches), with the peak VRAM of each. Times run end to end, from the state \
string to probabilities on the host. Precision as the models were evaluated: Laya with its CUDA autocast in fp16 \
(Step 1 sets `LAYA_CUDA_AMP=fp16`, as every earlier notebook), the small encoders with fp16 autocast as Phase 4 \
scored them. Latency does not depend on the fine-tuned weights, so the pinned Hub checkpoints are used. Runbook: \
`docs/phase7_gpu_runbook.md`.

{_table(cfg)}

**Expected runtime:** about {RUNTIME} on a T4: setup and the frozen data {SETUP_MIN} min, then the benchmark (above; \
assumptions in `tools/p7_md.py`). A G4 runs the benchmark several times faster.

**How to connect:** **Select Kernel → Colab → New Colab Server → GPU → T4** (the default), then the Python 3 kernel. \
**G4** (RTX PRO 6000 Blackwell) works too and is faster: every result row records which GPU ran, so name the GPU \
whenever you compare numbers. Do **not** use *Auto Connect*: it provisions a CPU server, and Step 4 stops without a \
GPU. Keep VS Code open and the laptop awake: a running cell keeps the server alive. Run nothing else on the GPU \
while Step 6 runs. The Hugging Face token (Step 3, masked input box, never stored in this notebook) is needed for \
the frozen data build only.

**After a kernel restart or a Colab disconnect:** re-run **Step 1**, then the interrupted step (or **Run All**). \
Finished steps skip themselves; an interrupted benchmark starts again from its first model (partial timings are \
never mixed). If the server itself was removed, `/content` is gone: start again from Step 1.

Code bundle: {n_files} files, sha256 `{sha256}` (verified when Step 1 unpacks it).
"""


def step_notes(cfg: dict) -> dict[str, tuple[str, str]]:
    """(title, note) per cell key, in notebook order."""
    notes = {
        "setup": ("Setup", "Non-interactive environment (`LAYA_CUDA_AMP=fp16`), work dir, code bundle, helpers. "
                  "Seconds. **Re-run this cell first after any kernel restart or Colab disconnect.**"),
        "install": ("Install", f"Plain `laya` at the pinned commit, DuckDB pinned to {cfg['data']['duckdb_version']}, "
                    "this project; the runtime's torch, transformers, protobuf and numpy stay as they are (the step "
                    "fails if pip changed any of them). 1-3 min."),
        "token": ("Hugging Face token", "Masked input box at the top of VS Code; never printed. Only the data build "
                  "needs it. Unattended: `python tools/build_p7_gpu_notebook.py --with-token` and open the "
                  "gitignored `notebooks/phase7_gpu_bench.local.ipynb`."),
        "env": ("Environment check", "GPU (required: any card), compute capability, driver, VRAM, versions, Laya "
                "commit, disk -> `runs/p7/env.json`, then the GPU the benchmark will run on. ~30 s."),
        "data": ("Full data (frozen)", "The E1 build, identical: FULL extraction over `hf://`, then `--verify-frozen` "
                 "(the fingerprint must equal `data.frozen_manifest`). The benchmark reads the frozen "
                 f"`test_id.jsonl`. {SETUP_MIN} min with Steps 1-4."),
        "bench": ("GPU benchmark", "`python -m laya_poc.bench_gpu --models ...`: the four pinned checkpoints are "
                  "downloaded first, then per model a fresh worker process: cold start, batch-1 latency, then per "
                  "batch size a warm-up and one timed batched call, peak VRAM. Writes "
                  f"`runs/p7/{BENCH_OUT}` (log: `runs/p7/logs/06_bench_gpu.log`). A failed model or batch size (e.g. "
                  "CUDA out of memory) is listed and the rest still runs; lower `--batch-sizes` in `cmd` after an "
                  "OOM. About 5-15 min on a T4 (the table above)."),
        "archive": ("Archive", "---\nZip `env.json`, the benchmark JSON, the logs, the data report and the FSQ NOTICE "
                    f"into `{ARCHIVE_NAME}` (no weights, no FSQ rows) for **Download...**. Seconds."),
    }
    return {k: (f"Step {i} - {title}", note) for i, (k, (title, note)) in enumerate(notes.items(), start=1)}


def finish_md(cfg: dict) -> str:
    return f"""### Finish
- Download `{ARCHIVE_NAME}`: Colab view in the activity bar > Contents > right-click > **Download...**
- Locally, from the repo root: `mkdir -p runs/p7_colab && unzip -q {ARCHIVE_NAME} -d runs/p7_colab` (gitignored),
  which gives `{P7_COLAB}/{BENCH_OUT}` (timings and machine facts, no FSQ rows). Keep a dated copy in
  `docs/results/` (runbook).
- Run **Colab: Remove Server** when you are done. It stops the compute-unit spend and deletes `/content`.
"""
