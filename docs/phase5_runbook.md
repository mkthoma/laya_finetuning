# Phase 5 runbook: the full evaluation, the CPU/ONNX benchmark and the decision

This runbook covers Phase 5 of the design doc (`Laya Record-Normalisation PoC.md` §6.2): the full metric suite on
every split (§5.11), the ONNX export and the CPU benchmark on a 4-8-core cloud CPU and the laptop (§7.11), trap
accuracy once the annotation lands (§5.3), and the decision rule (§5.12). The Phase 5 exit is: **every results
template (§7.13) is filled in and the decision rule is evaluated.**

Most of Phase 5 runs locally from the saved predictions of Phases 3 and 4: no GPU and no retraining. Only the cloud
CPU benchmark runs on Colab, in `notebooks/phase5_cpu_bench.ipynb` (a CPU runtime; about **4-6 h**).

## 1. The steps

| Step | Where | What | Output (gitignored unless noted) | Time |
|---|---|---|---|---|
| 1 Metrics and report | laptop | `phase5_metrics` over the Phase 3 + 4 archives, then `phase5_report` | `results/phase5_metrics.json`, `results/phase5_report.md` (+ `.json`) | minutes |
| 2 Laptop benchmark | laptop, nothing else running | `bench_cpu --models ...`, PyTorch and ONNX | `runs/p5_laptop/bench_cpu.json` | several hours |
| 3 Cloud CPU benchmark | Colab CPU High-RAM | `notebooks/phase5_cpu_bench.ipynb` | `phase5_bench_artifacts.zip` -> `runs/p5_colab/` | 4-6 h |
| 4 Trap annotation | two annotators + the lead | copies of the candidate CSV -> check -> `traps merge` -> metrics and report again | `data/trap.jsonl` | about a day per annotator |
| 5 Verdict | the lead | the final report; Phase 7 memo inputs | `docs/results/` (committed, metrics only) | - |

Steps 2-4 are independent: run them in any order or in parallel (but never run the laptop benchmark while anything
else uses the laptop's CPU). Re-run Step 1's report whenever an input arrives. Until then the report marks the
decision inputs it lacks as PENDING: criterion 4 without the annotated traps, criterion 5 and stop check (c) without a
benchmark JSON.

**What is known already.** `docs/results/phase34_report_2026-09-28.md` has an early read of criterion 1 against the
best trained baseline of *either* language (E2 `laya` -3.1 points, E3 `laya_ml` -0.4 points, both behind the
mmBERT-small B4). Phase 5 judges criterion 1 against the **language-matched** small encoder (`config.yaml` →
`phase5.small_encoder_match`: `laya` ↔ ModernBERT-base, `laya_ml` ↔ mmBERT-small) and reports the stricter "best
of either language" as a sensitivity check, with bootstrap 95% CIs. The thresholds are in `config.yaml` →
`phase5.decision`, fixed before any Phase 5 result: do not change them after seeing one.

## 2. Before you start

| Need | Detail |
|---|---|
| Phase 3 and 4 archives | Unzipped as in `docs/phase4_baselines_runbook.md` §7: `runs/p3_colab/runs/p3/<run>/...` and `runs/p4_colab/runs/p4/<run>/...` (each also has `data_eval/variant.json`) |
| The frozen data | `data/` from the frozen build: `test_id.jsonl` (benchmark input), `trap_candidates.csv`, `pool.parquet`, `split_*.parquet`. Never committed (FSQ rows) |
| ONNX in the venv | `onnx` 1.23.0, `onnxruntime` 1.30.0 and `onnxscript` 0.7.2 with `onnx-ir` 1.0.0 (torch's default ONNX exporter needs onnxscript; `tools/p5_commands.py` pins the same versions for Colab): `.venv/Scripts/python -m pip install onnx==1.23.0 onnxruntime==1.30.0 onnxscript==0.7.2 onnx-ir==1.0.0` if missing |
| Disk | About 3 GB for the ONNX exports of `laya` and `laya_ml` (fp32) per machine, plus about 3 GB of Hub cache for the four pinned models |
| Colab (Step 3) | An account that can start a **CPU High-RAM** runtime (Colab Pro), and the Hugging Face read token for the gated FSQ dataset (the notebook rebuilds the frozen data) |

Run every command from the repo root. `results/` is **not** gitignored: do not commit it by accident; copy what you
keep to `docs/results/` with a date (Step 5).

## 3. Step 1: metrics and report (local)

```bash
.venv/Scripts/python -m laya_poc.phase5_metrics --runs-root runs/p3_colab/runs/p3 --runs-root runs/p4_colab/runs/p4 --data-dir data --out results/phase5_metrics.json
.venv/Scripts/python -m laya_poc.phase5_report --metrics results/phase5_metrics.json --out results/phase5_report.md
```

`phase5_metrics` loads every finished run of both archives, groups them like the matrix report (seeds pooled) and
computes, per group and split: per-class P/R/F1 and the confusion matrix recomputed from the per-row predictions (with
the look-alike pairs of `phase5.lookalike_pairs`), the seed flip rate, and the paired bootstrap (1,000 resamples,
seeded) of the macro-F1 difference between each candidate and each baseline on every pool and on the OOD average of
criterion 1. ECE, Brier and NLL before and after T, accuracy at 80% and 90% coverage, tau and abstention, the
stripped-test metrics, order invariance and the OOD gaps come from the eval JSONs. It exits 1 when a recomputed
macro-F1 differs from the eval JSON of the same run and split (`recomputed_check`), naming them.

`phase5_report` renders the §7.13 templates: the main results per pool (every arm), traps / abstention / stability,
the CPU table (once a benchmark exists), the decision checklist per candidate (E2 `laya`, E3 `laya_ml`) with the stop
and investigate checks and the verdict, the bootstrap CIs, and the per-class and look-alike tables. Metrics only: no
FSQ row is read or written.

## 4. Step 2: the laptop benchmark

The laptop is the second machine of §7.11 ("an 8/16 GB laptop"). Timings are only meaningful on a quiet machine: on
mains power, the power plan on *Best performance*, every other program closed (browsers, IDE indexing, sync clients,
other jobs), and nobody using it. A plumbing check first (a few rows per setting, minutes; its numbers are not the
benchmark), then the benchmark:

```bash
.venv/Scripts/python -m laya_poc.bench_cpu --models laya laya_ml modernbert_base mmbert_small --rows data/test_id.jsonl --out runs/p5_laptop/bench_cpu_quick.json --threads 1 2 4 8 --onnx-dir runs/p5_laptop/onnx --quick
.venv/Scripts/python -m laya_poc.bench_cpu --models laya laya_ml modernbert_base mmbert_small --rows data/test_id.jsonl --out runs/p5_laptop/bench_cpu.json --threads 1 2 4 8 --onnx-dir runs/p5_laptop/onnx
```

Per model, backend (`laya` and `laya_ml`: PyTorch and ONNX; the small encoders: PyTorch) and thread count, a fresh
worker process measures the cold start, batch-1 latency p50/p95/mean over `phase5.bench.latency_n` (500) `test_id`
records after a warm-up, batched throughput over `batch_n` (1,000; batch 32, length-sorted) and the peak RSS. Thread
settings above the machine's physical cores are skipped with a note. Latency does not depend on the fine-tuned
weights, so the pinned Hub checkpoints are used. Each Laya model is first exported to ONNX (opset 18, into
`runs/p5_laptop/onnx/`) and checked against PyTorch fp32 on 1,000 `test_id` records (§7.11: argmax agreement >=
0.999, max |Δp| <= 1e-3); the result is in the benchmark JSON.

**Duration:** several hours. The early laptop run measured `laya` at about 1 s per record on 4 threads
(`docs/results/early_cpu_bench_2026-09-26.md`), and the benchmark times 1,500 records per setting: start it when the
laptop can be left alone, e.g. overnight. Delete `runs/p5_laptop/onnx/` afterwards if you need the disk.

## 5. Step 3: the cloud CPU benchmark on Colab

The decision's CPU budget (criterion 5) is judged "on 4 vCPUs" of a "4-8-core cloud CPU" (§5.11-§5.12): a Colab CPU
runtime with High-RAM (8 vCPUs). The notebook rebuilds the frozen data exactly as E1 did (the benchmark reads its
`test_id.jsonl`) and runs the same `bench_cpu` command as Step 2 with its paths on Colab.

```bash
.venv/Scripts/python tools/build_p5_bench_notebook.py              # writes notebooks/phase5_cpu_bench.ipynb (token prompt in Step 3)
.venv/Scripts/python tools/build_p5_bench_notebook.py --with-token # unattended: gitignored notebooks/phase5_cpu_bench.local.ipynb
```

The notebook embeds the project code and the frozen data manifest as a checksummed bundle. **After changing any code
or `config.yaml`, rebuild it** (`tests/test_build_p5_notebook.py` fails while the committed notebook is stale). The
unattended variant reads `HF_TOKEN=hf_...` from the repo's `.env` and only writes a gitignored `*.local.ipynb`.

Connect with **Select Kernel → Colab → New Colab Server → CPU → High-RAM**, then the Python 3 kernel. Do **not** use
Auto Connect: a standard CPU runtime has 2 vCPUs (one physical core), so the 4-thread row that criterion 5 needs
would be missing. Then **Run All**.

| Step | What it does | Output (under `/content/laya_poc`) | Estimate |
|---|---|---|---|
| 1 Setup | Non-interactive environment, bundle unpacked and checked, helpers | `src/`, `config.yaml` | seconds |
| 2 Install | Pinned `laya` and DuckDB, this project, and the pinned ONNX packages (`onnx`, `onnxruntime`, `onnxscript`, `onnx-ir`) under a constraints file that holds the runtime's torch, transformers, protobuf and numpy; fails if any of those four changed | `runs/p5/logs/02_*.log` | 1-3 min |
| 3 HF token | Masked prompt (or the preset token), validated, never printed | token file on the server | seconds |
| 4 Environment | `env_check` without a GPU, then the CPU: model, logical CPUs, physical cores, affinity, cgroup quota, AVX-512/AMX flags. Warns when a GPU is attached, when there are fewer than 4 physical cores, and which thread settings will be skipped | `runs/p5/env.json`, `runs/p5/cpu.json` | ~30 s |
| 5 Full data | The E1 build with `--verify-frozen` | `data/` | 10-25 min with Steps 1-4 |
| 6 Benchmark | `bench_cpu --models laya laya_ml modernbert_base mmbert_small ...` (`QUICK = True`: the plumbing check, own output file) | `runs/p5/bench_cpu.json`, `runs/p5/onnx/` | 4-6 h |
| 7 Archive | Zip of the small artefacts (below) | `phase5_bench_artifacts.zip` | seconds |

**Where the estimate comes from** (`tools/p5_md.py`; the intro table of the notebook is the authoritative version):
`laya` at 1.5 s per record at batch 1 and 0.6 records/s batched on one thread (the Phase 2 Colab host,
`docs/results/phase2_gate_report_2026-09-26.md`), `laya_ml` about 3x faster (the laptop), the small encoders 10-25x
faster (they read the record only), ONNX no faster than PyTorch, and 4 physical cores behind the 8 vCPUs, so the
8-thread setting is skipped. `laya` dominates: about 3 h for both backends at 1, 2 and 4 threads. Nothing here is
measured on a Colab CPU yet: record the real step times after the first run.

| Step | Estimate | Actual |
|---|---|---|
| 1-5 Setup and data | 10-25 min | |
| 6 Benchmark | 4-6 h | |

**The 8-thread setting.** Threads are capped at the physical cores. On 4 physical cores there is no 8-thread row, so
stop check (c) ("p95 > 1,000 ms even with ONNX on 8 threads") uses the most threads measured: at or below 1,000 ms
there already means no stop; above it, the check stays open and the report says it was not measured at 8 threads.

**After a disconnect or a kernel restart:** re-run Step 1, then the interrupted step (or Run All). Finished steps
skip themselves (`REDO = True` in Step 1 redoes them). An interrupted benchmark starts again from its first setting:
partial timings are never mixed with a later run's. Step 6 refuses to start while a benchmark process from before
the restart is still running (it would compete for the cores): wait for it, or stop it with `os.kill(pid, 9)`. If
the server itself was removed, `/content` is gone: start again from Step 1.

**Archive and report.** Step 7 zips `env.json`, `cpu.json`, the benchmark JSON(s), the ONNX export records
(`onnx/*.onnx.json`), the logs, `data/data_report.json`, `SHA256SUMS` and `NOTICE_FSQ.txt`: no ONNX graphs, no
weights, no FSQ rows. Download it (Colab view: Contents → right-click → **Download...**), run **Colab: Remove Server**,
then locally:

```bash
mkdir -p runs/p5_colab
unzip -q phase5_bench_artifacts.zip -d runs/p5_colab   # -> runs/p5_colab/runs/p5/bench_cpu.json
.venv/Scripts/python -m laya_poc.phase5_report --metrics results/phase5_metrics.json --bench runs/p5_colab/runs/p5/bench_cpu.json --out results/phase5_report.md
```

The **first** `--bench` is the machine criterion 5 and stop check (c) are judged on (the cloud CPU); a later one (the
laptop, Step 5) is reported alongside it.

## 6. Step 4: trap annotation, merge, and the metrics again

Two annotators and the lead follow [`docs/trap_annotation_guide.md`](trap_annotation_guide.md): each annotator marks
a copy of `data/trap_candidates.csv` (700 candidates, 100 per pattern), the check lists the rows the lead must
decide, and the merge writes the kept items (target 150-300) to `data/trap.jsonl`:

```bash
cp data/trap_candidates.csv data/trap_candidates_a1.csv   # annotator 1 (and the lead's keep_lead)
cp data/trap_candidates.csv data/trap_candidates_a2.csv   # annotator 2
.venv/Scripts/python tools/p5_trap_check.py data/trap_candidates_a1.csv data/trap_candidates_a2.csv
.venv/Scripts/python -m laya_poc.traps merge --candidates data/trap_candidates_a1.csv data/trap_candidates_a2.csv --out data/trap.jsonl --data-dir data
.venv/Scripts/python -m laya_poc.phase5_metrics --runs-root runs/p3_colab/runs/p3 --runs-root runs/p4_colab/runs/p4 --data-dir data --traps data/trap.jsonl --out results/phase5_metrics.json
```

No run is repeated: every Phase 3 and 4 run already scored all 700 candidates (`preds/trap_candidates.jsonl`), and
trap accuracy (all, and `multi` = 1 separately) is their accuracy on the kept items. `phase5_metrics` takes the kept
places from the trap rows' `fsq_place_id`, and maps each scored candidate to its place by the row order of the
original `data/trap_candidates.csv`, checked against the archive's `data_eval/variant.json`. That is why the original
file must stay exactly as the build wrote it: annotate the copies only. Then re-render the report (§7).

## 7. Step 5: the verdict and the Phase 7 memo inputs

With every input in place, render the final report (cloud CPU first: it is the judged machine):

```bash
.venv/Scripts/python -m laya_poc.phase5_report --metrics results/phase5_metrics.json --bench runs/p5_colab/runs/p5/bench_cpu.json --bench runs/p5_laptop/bench_cpu.json --out results/phase5_report.md
```

The decision checklist marks criteria 1-5 PASS, FAIL or PENDING for each candidate (seed means; `laya` without
`ood_script`), with the stop checks (a)-(c), the investigate checks and the verdict: **PASS** (all five hold for at
least one candidate), **STOP**, **INVESTIGATE** (one bounded iteration of at most 2 days, then re-decide), or **NO
PASS** with no formal stop condition met (the lead's judgement: the doc has no rule for it). While an input is
pending the report says whether it can still change the verdict (with criterion 1 failed for every candidate,
pending traps or CPU numbers cannot produce a PASS).

Record it (metrics only, as for Phases 2-4): copy `results/phase5_report.md` and `.json` to
`docs/results/phase5_report_<date>.md` / `.json`, and the benchmark JSONs (timings and machine facts, no rows) to
`docs/results/`. The Phase 7 verdict memo (design §6.2: two pages plus the results workbook) takes from here:

- the verdict and the decision checklist, with each criterion's number, and the pending inputs if any;
- the main results tables, the bootstrap CIs of criterion 1 (primary: the language-matched small encoder; the strict
  variant as sensitivity) and the ID→OOD gaps;
- calibration (ECE before and after T), abstention and the stripped-test false-confident rate, stability (seed range,
  flip rate, order invariance);
- trap accuracy (all / multi) with the annotation statistics from `tools/p5_trap_check.py` (agreement, Cohen's kappa,
  kept count);
- the CPU table with the hardware of each machine (`cpu.json`, the benchmark's machine block) and the ONNX acceptance;
- §5.12 "What a pass does not show" (a pass on public place data is not a pass on the target data);
- provenance: `config.yaml` (thresholds fixed before the results), the Laya commit, the frozen data fingerprint
  (`docs/results/data_fingerprint_2026-09-15.json`), seeds 11/22/33 and the bundle sha256 of every notebook run.

## 8. When something fails

| Symptom | Likely cause and fix |
|---|---|
| `phase5_metrics: error: ... no finished run` | A `--runs-root` does not point at `<archive>/runs/p3` or `runs/p4`: check the unzip layout (§2) |
| `phase5_metrics: ... recomputed macro-F1 differs` (exit 1) | A run's predictions and its eval JSON disagree: the message names the run-splits. Check the archive is complete and unmodified (unzip it again); do not edit it |
| `cannot map trap_candidates ids to places by CSV order ... differs from the one the build scored` | `data/trap_candidates.csv` was edited or re-saved. Restore the original: rebuild the frozen data into a new directory (`python -m laya_poc.build_data --out data_rebuild --verify-frozen`) and copy its `trap_candidates.csv` back |
| `traps: error: ... annotation incomplete` | Unmarked rows or disagreements without `keep_lead`: run `tools/p5_trap_check.py`, which lists them, and complete them (guide §4) |
| `traps: error: ... do not hold the same candidates` or `... not in the pool` | A copy lost, gained or changed a row, or a spreadsheet changed an `fsq_place_id`: make a fresh copy and move the marks over (guide §6) |
| `bench_cpu`: a setting FAILED | Read the setting's error in the JSON and the log: the other settings still ran. Fix the cause, then re-run (the whole benchmark: timings of different runs are not mixed) |
| ONNX acceptance fails for a model | Criterion 5 then ignores that model's ONNX rows (the report says so) and judges its PyTorch rows; the export record (`onnx/<model>.onnx.json`: exporter, attempts, warnings) says how it was exported |
| Colab Step 2: pip cannot install the ONNX packages | The runtime's protobuf is older than onnx 1.23 needs (e.g. the 2026.07 runtime has protobuf 5.29): Remove Server and start the *Latest* runtime |
| Colab Step 4: GPU attached, or fewer than 4 physical cores | Remove Server and start **CPU → High-RAM** (§5) |
| Colab Step 5: data build errors | As in the Phase 2 runbook §7 (`docs/phase2_e1_runbook.md`): 401/403 on the gated data, 429 rate limits, frozen-manifest mismatch |
| Colab Step 6: `still running (pid ...)` | A benchmark from before a kernel restart: wait, or `os.kill(pid, 9)`, then re-run the step |

## 9. What only the real runs can confirm

- The benchmark's real duration on the Colab CPU and on the laptop, and the physical core count behind Colab's 8
  vCPUs (Step 4 records it).
- Whether ONNX Runtime is faster than PyTorch for these models, and whether batching recovers any throughput on CPU
  (the early laptop run saw none: `docs/results/early_cpu_bench_2026-09-26.md`).
- That the pinned ONNX packages install cleanly next to the current Colab image's protobuf and numpy, and which
  exporter the export record shows there (the laptop venv used torch's default dynamo exporter).
