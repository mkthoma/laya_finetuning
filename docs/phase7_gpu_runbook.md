# Phase 7 runbook: the GPU inference benchmark

The CPU benchmark (Phase 5, `docs/phase5_runbook.md`) answers the decision rule's CPU budget. This one answers the
other deployment question: **how fast are the models on a GPU?** It measures, for the same four models
(`config.yaml` → `phase5.bench.models`: `laya`, `laya_ml`, `modernbert_base`, `mmbert_small`), on a Colab GPU:

- **cold start**: seconds from a fresh process to a runtime ready to predict (weights loaded onto the GPU);
- **batch-1 latency**: p50 / p95 / mean / max ms over 500 `test_id` records after 20 warm-up calls, one record per
  call (measured once per model; the same numbers appear on each of its rows);
- **batched throughput**: rec/s of one timed call over 2,000 records at batch size 32 and at 64 (length-sorted
  batches, after 3 untimed warm-up batches at that size);
- **peak VRAM**: max allocated and reserved GiB, per batch size and for the load + batch-1 phase.

Times run end to end, from the state string to probabilities on the host (tokenisation, the forward pass and
decoding), with `torch.cuda.synchronize()` around each timed call. Each model runs in a fresh process, so its cold
start and peak VRAM are its own. Precision is what evaluation used: Laya with its CUDA autocast in fp16 (the notebook
sets `LAYA_CUDA_AMP=fp16`, as every earlier notebook; below compute capability 8, e.g. a T4, the Agent uses fp16
anyway), the small encoders with fp16 autocast as Phase 4 scored them. Latency does not depend on the fine-tuned
weights, so the pinned Hub checkpoints are used. Laya's CPU fallback on a CUDA out-of-memory error is caught: that
batch size becomes an error, never a CPU number passed off as a GPU one.

## 1. Which runtime, how long

| Runtime | When | Estimate (whole notebook) |
|---|---|---|
| **T4** (default) | The reference: the cheapest Colab GPU, what the notebook's metadata asks for | 15-45 min: setup and the frozen data 10-25 min, the benchmark 5-15 min |
| **G4** (RTX PRO 6000 Blackwell) | Faster, and closer to a modern serving GPU | The same setup time; the benchmark several times faster |

Every result row records the GPU (`gpu`, `capability`) and the JSON's `hardware` line names it, so numbers from a T4
and a G4 are never confused. Say which GPU a number came from whenever you quote it. The estimates are assumptions
(`tools/p7_md.py`: e.g. `laya` at 35 ms per record at batch 1 and 60 rec/s batched on a T4); nothing has been
measured on a GPU yet, so record the real step times after the first run:

| Step | Estimate | Actual (T4) | Actual (G4) |
|---|---|---|---|
| 1-5 Setup and data | 10-25 min | | |
| 6 Benchmark | 5-15 min (T4) | | |

## 2. How to run it

```bash
.venv/Scripts/python tools/build_p7_gpu_notebook.py              # writes notebooks/phase7_gpu_bench.ipynb (token prompt in Step 3)
.venv/Scripts/python tools/build_p7_gpu_notebook.py --with-token # unattended: gitignored notebooks/phase7_gpu_bench.local.ipynb
```

The notebook embeds the project code and the frozen data manifest as a checksummed bundle. **After changing any code
or `config.yaml`, rebuild it** (`tests/test_build_p7_notebook.py` fails while the committed notebook is stale). The
unattended variant reads `HF_TOKEN=hf_...` from the repo's `.env` and only writes a gitignored `*.local.ipynb`: open
it instead of the committed notebook and nothing prompts.

Connect with **Select Kernel → Colab → New Colab Server → GPU → T4** (or **G4**), then the Python 3 kernel. Do
**not** use Auto Connect: it provisions a CPU server and Step 4 stops. Then **Run All**.

| Step | What it does | Output (under `/content/laya_poc`) | Estimate |
|---|---|---|---|
| 1 Setup | Non-interactive environment (`LAYA_CUDA_AMP=fp16`), bundle unpacked and checked, helpers | `src/`, `config.yaml` | seconds |
| 2 Install | Pinned `laya` and DuckDB, this project; fails if pip changed the runtime's torch, transformers, protobuf or numpy | `runs/p7/logs/02_*.log` | 1-3 min |
| 3 HF token | Masked prompt (or the preset token), validated, never printed | token file on the server | seconds |
| 4 Environment | `env_check --require-gpu`: GPU, compute capability, driver, VRAM, versions; prints the GPU the benchmark runs on | `runs/p7/env.json` | ~30 s |
| 5 Full data | The E1 build with `--verify-frozen` (the benchmark reads its `test_id.jsonl`) | `data/` | 10-25 min with Steps 1-4 |
| 6 Benchmark | `bench_gpu` over the four models (below) | `runs/p7/bench_gpu.json`, `runs/p7/logs/06_bench_gpu.log` | 5-15 min (T4) |
| 7 Archive | Zip of the small artefacts | `phase7_gpu_artifacts.zip` | seconds |

Step 6 runs (paths on Colab):

```bash
python -m laya_poc.bench_gpu --models laya laya_ml modernbert_base mmbert_small --rows /content/laya_poc/data/test_id.jsonl --out /content/laya_poc/runs/p7/bench_gpu.json --latency-n 500 --batch-n 2000 --batch-sizes 32 64 --warmup 20 --device cuda
```

It downloads the four pinned checkpoints first (about 3 GB), then starts one worker process per model and prints one
line per (model, batch size): cold start, first call, p50, p95, rec/s and peak VRAM. A failed model or batch size is
listed as FAILED and the others still run.

**After a disconnect or a kernel restart:** re-run Step 1, then the interrupted step (or Run All). Finished steps
skip themselves (`REDO = True` in Step 1 redoes them). An interrupted benchmark starts again from its first model.
Step 6 refuses to start while a benchmark process from before the restart still runs (two would share the GPU): wait
for it, or stop it with `os.kill(pid, 9)`.

**Optional local plumbing check** (CPU, one small model, a few rows; it proves the CLI and the rows file work, its
numbers are not GPU numbers):

```bash
.venv/Scripts/python -m laya_poc.bench_gpu --models mmbert_small --rows data/test_id.jsonl --out runs/p7_local/bench_gpu_cpu_check.json --device cpu --latency-n 10 --batch-n 64 --batch-sizes 16 32 --warmup 2
```

## 3. Where the outputs go, and bringing them back

`runs/p7/bench_gpu.json` holds the plan (models, rows, row counts, batch sizes, warm-up), the sources (pinned
checkpoint per model), the machine (`gpu`, `capability`, `vram_total_gb`, `cuda`, `driver`, host CPU, library
versions), one `results` row per (model, batch size) (`model`, `backend`, `device`, `gpu`, `amp`, `dtype`,
`batch_size`, `cold_s`, `first_predict_ms`, `p50_ms`, `p95_ms`, `mean_ms`, `max_ms`, `batch_rps`, `batch_seconds`,
`peak_vram_gb`, ...), the `errors` and the total seconds. Timings and machine facts only: no FSQ rows.

Step 7 zips `env.json`, the benchmark JSON and its `_ok` marker, the logs, `data/data_report.json`, `SHA256SUMS` and
`NOTICE_FSQ.txt` into `phase7_gpu_artifacts.zip` (no weights, no FSQ rows). Download it (Colab view: Contents →
right-click → **Download...**), run **Colab: Remove Server**, then locally from the repo root:

```bash
mkdir -p runs/p7_colab
unzip -q phase7_gpu_artifacts.zip -d runs/p7_colab   # -> runs/p7_colab/runs/p7/bench_gpu.json (gitignored)
```

Keep a dated copy of the JSON in `docs/results/` (for example `docs/results/bench_gpu_t4_<date>.json`, with the GPU
in the name), as for the CPU benchmark JSONs.

## 4. When something fails

| Symptom | Likely cause and fix |
|---|---|
| Step 4: `environment check failed: no CUDA GPU (Auto Connect gives a CPU) ...` / `bench_gpu: error: ... no CUDA device` | A CPU runtime (e.g. Auto Connect). **Colab: Remove Server**, then New Colab Server → GPU → T4 (in a browser: Runtime → Change runtime type → T4 GPU), and Run All |
| A row FAILED with `OutOfMemoryError` / `CUDA out of memory`, or `fell back to the CPU at batch size N` (a CUDA OOM) | Not enough VRAM for that batch size (the other rows are fine). Lower `--batch-sizes` in Step 6's `cmd` (e.g. `"16", "32"`), set `REDO = True` in Step 1 and re-run Steps 1 and 6. A G4 (95 GB) does not run out |
| A model FAILED with `timed out after 60 min` | Far slower than expected: check `nvidia-smi` shows the GPU busy (not the CPU), then raise `--timeout-min` |
| A model FAILED with `fell back to the CPU during the batch-1 latency` | The GPU ran out of memory at batch 1: something else holds the GPU (a process from before a restart). Remove Server and start again |
| Step 6: `still running (pid ...)` | A benchmark from before a kernel restart: wait, or `os.kill(pid, 9)`, then re-run the step |
| Step 5: data build errors | As in the Phase 2 runbook §7 (`docs/phase2_e1_runbook.md`): 401/403 on the gated data, 429 rate limits, frozen-manifest mismatch |
| Step 2: `pip changed ...` | Colab's image changed under the pinned packages: Remove Server and start again on the *Latest* runtime |

## 5. What only the real run can confirm

- The real latencies and throughputs on a T4 (and a G4), the cold starts, and the peak VRAM per batch size: none of
  this has run on a GPU yet (the laptop has none; the unit tests fake the GPU and run the real code on CPU).
- Whether batch 64 fits every model on a T4 (16 GB) without an OOM, and how much batching gains over batch 1.
- The autocast dtype each GPU ends up with (`dtype` in each row: fp16 on both with `LAYA_CUDA_AMP=fp16`).
