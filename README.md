# Laya record-normalisation PoC

This PoC tests whether [Laya](https://github.com/NandhaKishorM/laya), a small non-autoregressive decision model with
calibrated probabilities, can be fine-tuned to normalise messy records into a fixed taxonomy. The proxy task is
Foursquare OS Places (FSQ): predict a place's 10-class category from its name, address and contact fields. The full
design, plan and decision rule are in [`Laya Record-Normalisation PoC.md`](Laya%20Record-Normalisation%20PoC.md).

**Status (2026-09-28): done.** Every planned phase has run. Phase 6, the optional GLEIF backup, was not needed.

**Verdict:** under the design's decision rule (§5.12), the verdict is **INVESTIGATE (final)**. The
[Phase 7 verdict memo](docs/results/phase7_verdict_memo_2026-09-28.md) recommends not taking Laya to an internal
trial. Fine-tuning Laya works, but a plainly fine-tuned small encoder (mmBERT-small) matches its accuracy, is better
calibrated, and is about 10x faster on CPU.

**Where the detail is:**
- the [results workbook](docs/results/phase7_results_workbook_2026-09-28.xlsx) (metrics only, 12 sheets);
- the per-phase reports in [`docs/results/`](docs/results/);
- the [Results](#results) section below, which summarises them.

## Results

Macro-F1 is the unweighted mean of the per-class F1 over 10 classes. Fine-tuned rows show the mean of 3 seeds.

The test set is in-distribution: countries and chains seen in training. The three OOD (out-of-distribution) pools are
unseen data: countries held out of training, Thai-script records, and chains held out of training.

ECE is the expected calibration error after temperature scaling. Full tables, with seed ranges, bootstrap CIs and
per-class scores, are in the [Phase 5 report](docs/results/phase5_report_2026-09-28.md).

| Model | Test macro-F1 | Test top-1 acc | New countries | Thai script | New brands | ECE test / OOD |
|---|---|---|---|---|---|---|
| Laya multilingual (`laya_ml`), fine-tuned | 0.589 | 60.6% | 0.494 | **0.506** | 0.551 | 0.044 / 0.070 |
| Laya English (`laya`), fine-tuned | 0.564 | 58.2% | 0.430 | 0.324 ¹ | **0.588** | 0.062 / 0.096 |
| mmBERT-small, fine-tuned | **0.596** | **61.5%** | **0.502** | 0.483 | 0.578 | 0.026 / 0.065 |
| ModernBERT-base, fine-tuned | 0.560 | 57.5% | 0.412 | 0.327 | 0.521 | 0.028 / 0.102 |
| Qwen3-4B, zero-shot (evaluation subsets) | 0.502 | 50.6% | 0.449 | 0.463 | 0.401 | 0.061 / 0.088 |
| Char TF-IDF + logistic regression | 0.487 | 49.7% | 0.343 | 0.279 | 0.449 | 0.022 / 0.079 |
| Laya zero-shot (`laya` / `laya_ml`) | 0.313 / 0.282 | 31.5% / 28.5% | | | | |

¹ English `laya` cannot read Thai by design, so this pool is excluded from its criteria.

**What fine-tuning bought.** It roughly doubles Laya's zero-shot score, from 0.28–0.31 to 0.56–0.59.

**How it does on unseen data.**
- **Against TF-IDF:** averaged over the three OOD pools, `laya_ml` leads TF-IDF by 16 points.
- **Against mmBERT-small:** it is level (−0.4 points, 95% CI [−1.2, +0.5]). It wins on Thai script (+2.3) and loses on
  new brands (−2.7).
- **The ID→OOD drop fails the limit for both models.** Going from test data to new countries, `laya_ml` drops 9.4
  points and mmBERT-small 9.4. The design's limit is 5, so this fails for both.
- **English `laya`'s scores on new brands are unstable across seeds**, ranging over 7 points.

**Decision checklist** for `laya_ml`, the better checkpoint:
- C1, the OOD lead over the language-matched encoder, fails: −0.4 against the required +3.
- C2, the ID→OOD gap, fails: 9.4 against a limit of 5.
- C3, calibration, fails narrowly: OOD ECE is 0.070 against a limit of 0.05.
- C4, trap accuracy, is pending annotation, and it cannot change the verdict.
- C5, the CPU budget, fails on throughput.

### Sample inferences

Real example records cannot be published here. The repository is public, and FSQ rows may not leave the team (design
doc Appendix D). [`python -m laya_poc.phase7_samples`](#phase-7-the-write-up) writes about 100 real records with every
model's prediction to the gitignored `runs/phase7/sample_inferences.md`. The records cover the test set, each OOD pool
and the no-evidence set, grouped by which model was right.

Typical cases from the seed-11 runs, described rather than quoted (✓ right, ✗ wrong, probability in brackets):

| Record (described) | True label | Laya multilingual | mmBERT-small |
|---|---|---|---|
| A hotel's swimming pool, name only | sports | sports (0.87) ✓ | sports (0.81) ✓ |
| A clinic name written only in Thai | health | health (0.94) ✓ | community (0.77) ✗ |
| An orthopaedic clinic in Poland, name only | health | health (0.89) ✓ | outdoors (0.27) ✗ |
| A national postal-service branch with full address | community | services (0.72) ✗ | community (0.42) ✓ |
| An EV charging station in the US | travel | services (0.58) ✗ | travel (0.57) ✓ |
| A company whose name starts with "Event" | services | event (0.90) ✗ | event (0.87) ✗ |
| A bare street name filed as a place | community | travel (0.74) ✗ | travel (0.61) ✗ |

The last two rows show a common pattern. Many records that both models "get wrong" look mislabelled in FSQ itself,
which probably caps accuracy around 60%. This is not measured; a label audit would quantify it.

When every field except the country is removed, every model abstains on all 1,000 records. Without that evidence gate,
Laya answers near-uniformly (mean confidence 0.10), which is the right behaviour, while TF-IDF is confidently wrong
(0.45).

Right/wrong agreement between Laya multilingual and mmBERT-small, seed-11 runs
([`phase7_sample_agreement`](docs/results/phase7_sample_agreement_2026-09-28.json)):

| Split | Both right | Only Laya right | Only mmBERT right | Both wrong |
|---|---|---|---|---|
| Test (3,000) | 1,578 | 212 | 234 | 976 |
| New countries (1,998) | 855 | 155 | 203 | 785 |
| Thai script (2,000) | 837 | 245 | 178 | 740 |
| New brands (2,000) | 974 | 135 | 298 | 593 |

### Inference time

End to end, from the record string to the probabilities, tokenisation included.

**CPU:** the maintainer's laptop, an i5-1135G7 with 4 cores / 8 threads, fp32, at 4 threads. Each cell is the faster
of PyTorch and ONNX. The design's budget is p95 ≤ 500 ms and ≥ 8 records/s.

**GPU:** a Colab G4 (RTX PRO 6000 Blackwell), fp16 autocast. The per-record time is batch 1, over 500 records; the
throughput is at batch 64.

| Model | CPU p95 per record | CPU records/s | GPU p50 / p95 per record | GPU records/s | GPU memory | Train + evaluate, one run (G4) |
|---|---|---|---|---|---|---|
| Laya English | 1,131 ms ❌ | 1.1 ❌ | 9.6 / 9.7 ms | 668 | 2.9 GiB | 9.7 min |
| Laya multilingual | 408 ms ✅ | 3.2 ❌ | 7.8 / 8.0 ms | 1,305 | 1.9 GiB | 5.2 min |
| ModernBERT-base | 167 ms | 13.6 | 6.6 / 7.3 ms | 4,745 | 1.0 GiB | 1.5 min |
| mmBERT-small | 75 ms | 30.9 | 6.9 / 7.5 ms | 6,426 | 0.7 GiB | 1.5 min |

- **ONNX export is exact.** Both Laya models agree with PyTorch on 1,000 of 1,000 checked records.
- **On GPU, speed does not block Laya.** Every model answers one record in under 10 ms, but batched the encoders are
  5–10x faster.
- **On CPU, Laya fails the budget.** `laya_ml` meets the latency limit but reaches only 3.2 of the 8 records/s floor.
  English `laya` also hits a stop condition: p95 above 1 s even with ONNX on 8 threads.
- **Details:**
  - CPU: the [Phase 5 report](docs/results/phase5_report_2026-09-28.md);
  - GPU: the [GPU timing](docs/results/phase7_gpu_timing_2026-09-28.md) and the workbook's "GPU latency" sheet.

**Earlier phases:**
- Phase 1, the smoke test, passed 8/8 on a Colab T4 on 2026-09-26
  ([`docs/results/phase1_smoke_report_2026-09-26.md`](docs/results/phase1_smoke_report_2026-09-26.md)).
- The Phase 2 gate passed 4/4.
- The Phase 3 and Phase 4 exit checks passed 14/14 and 13/13 (see [`docs/results/`](docs/results/)).

## Layout

```
config.yaml                 every parameter (pinned Laya commit and Hub revision, data, training, smoke exit criteria,
                            the E1 plan and the gate thresholds)
src/laya_poc/               the package; every step is a CLI: python -m laya_poc.<module> --help
  config, labels, normalise, splits, sampler, io_utils, metrics, calibrate, hub, items, notice   shared building blocks
  extract, serialise, augment, rows, build_data, build_report, traps, freeze
                            FSQ extraction (DuckDB over hf://) -> trap candidates set aside -> splits -> JSONL rows;
                            the JSONL fingerprint and the frozen-data manifest
  loss, schedule, ckpt, export, train_single        single-GPU port of the upstream trainer, resumable, best/ export
  env_check, zeroshot, parity, export_check, smoke_report, smoke_render   Phase 1 checks and the report
  evaluate, baselines, bench_cpu, gate              Phase 2: val evaluation (pre/post T), B1/B3, CPU benchmark, gate
  matrix, matrix_*, variants                        Phases 3-4: the run matrix (plan, run, results, report)
  baseline_runs, baseline_eval, small_encoder*, llm_baseline   Phase 4 baselines B1/B3/B4/B5
  phase5_*, decision, decision_eval, bench_onnx, onnx_export, bench_worker, bench_hf
                            Phase 5: metrics, bootstrap, decision rule, report, CPU/ONNX benchmark
  phase7_samples(_md), phase7_timing(_md), phase7_workbook, phase7_xlsx, phase7_provenance, bench_gpu(_worker)
                            Phase 7: sample inferences, GPU timing, GPU latency benchmark, results workbook
  bundle                    packs the code into the notebook (checksummed, deterministic)
tools/build_notebook.py     generates notebooks/smoke_test.ipynb (code bundle embedded; no git/Drive needed on Colab)
  notebook_helpers, notebook_commands, notebook_cells   its parts: Step 1 helpers (embedded verbatim), the CLI
                            commands, the code cell sources
tools/build_e1_notebook.py  generates notebooks/e1_first_run.ipynb from the same parts plus e1_helpers,
                            e1_commands, e1_cells
tools/build_p3_notebook.py, build_p4_notebook.py, build_p5_bench_notebook.py, build_p7_gpu_notebook.py
                            the Phase 3/4/5/7 notebooks, each with its p*_commands, p*_cells, p*_helpers, p*_md parts
tools/dry_run_local.py, dry_run_e1_local.py, dry_run_p3_local.py, dry_run_p4_local.py
                            run a notebook's cells locally on CPU with a tiny checkpoint
notebooks/*.ipynb           generated; do not edit by hand
docs/*_runbook.md           how to run each notebook and read its report; docs/trap_annotation_guide.md
docs/results/               committed results: reports, the verdict memo, the results workbook and the frozen data
                            fingerprint (metrics, hashes and counts only; no FSQ rows)
tests/                      pytest suite (CPU only; fixtures build a tiny random Laya checkpoint)
```

Data, runs and checkpoints are never committed (`.gitignore`). FSQ rows may not be redistributed (design doc Appendix D).
Every data directory gets `NOTICE_FSQ.txt` (the FSQ NOTICE verbatim plus a "modified" statement, `laya_poc.notice`),
and every fine-tuned checkpoint gets `NOTICE.md`, `NOTICE_FSQ.txt` and Laya's `LICENSE`.

## Local development

Python 3.10+. On Windows use `.venv/Scripts/python`; elsewhere use `.venv/bin/python`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,data]"
# Laya at the commit pinned in config.yaml (laya.commit); pulls torch and transformers
.venv/Scripts/python -m pip install "laya @ git+https://github.com/NandhaKishorM/laya.git@4066d5d5fbf08b66c6757ddeedbd797bd7655bc0"
.venv/Scripts/python -m pytest -p no:warnings -q
```

Tests marked `torch` need Laya and torch. The tiny-checkpoint fixtures download the public Laya tokenizer (about 4 MB,
cached) and skip when the Hub is unreachable. The full local dry runs of the notebooks take several minutes each, so
they are opt-in: `LAYA_DRY_RUN_TEST=1 .venv/Scripts/python -m pytest -q tests/test_build_notebook.py
tests/test_build_e1_notebook.py`.

Never put the Hugging Face token in code, config or the committed notebook. The CLIs read it only from the `HF_TOKEN`
environment variable. It is needed only for the gated FSQ dataset. The notebook asks for it with a hidden prompt, and
`tools/build_notebook.py` refuses to embed anything that looks like a token, except in the opt-in, gitignored
unattended variant below.

## Build the smoke-test notebook

The notebook embeds the code bundle (`config.yaml`, `pyproject.toml`, `src/laya_poc/**/*.py`). Regenerate it after
every code or config change. A test fails while the committed notebook is stale.

```bash
.venv/Scripts/python tools/build_notebook.py          # writes notebooks/smoke_test.ipynb, prints the bundle sha256
.venv/Scripts/python tools/dry_run_local.py           # optional: run the same cells on CPU first
```

Unattended variant (no token prompt): add `HF_TOKEN=hf_...` to `.env` (gitignored), then run
`.venv/Scripts/python tools/build_notebook.py --with-token`. It writes `notebooks/smoke_test.local.ipynb` (gitignored)
with the token in Step 3. Open that file instead, and keep it local.

Then open the notebook in VS Code and connect with **Select Kernel → Colab → New Colab Server → GPU → T4** (not Auto
Connect). Follow [`docs/smoke_test_runbook.md`](docs/smoke_test_runbook.md). The smoke test takes about 60-90 min. The
longest step is the CPU fp32 parity reference (Step 7), at about 25-35 min.

## Phase 2: E1 first real run and the gate

Design doc §6.2: E1 (`laya`, seed 11, the full 25k train split x 4 epochs, resumable) on a free Colab T4, then the
temperature fit, the evaluation on `val`, a CPU benchmark on 200 records, the B1 (majority/prior) and B3 (char TF-IDF +
logistic regression) baselines, zero-shot `laya`, and the **gate** (`python -m laya_poc.gate`): end-to-end, numerics,
beats trivial (zero-shot + 10 points, majority + 20), near TF-IDF+LR (at most 5 points below). A gate FAIL means stop
and debug before any spend. Expected runtime about 2.5-3.5 h, E1 itself about 2-2.3 h.

The data is frozen: `config.yaml` → `data.frozen_manifest` names the fingerprint of the first FULL build
(`docs/results/`, hashes only). The notebook rebuilds the data on Colab with DuckDB pinned to `data.duckdb_version` and
stops unless every JSONL file matches. E1 checkpoints every `e1.ckpt_every_micro_steps` micro-steps and is hard-killed
once at `e1.crash_at_micro_step` to force the resume the gate asks for; after a Colab disconnect, re-run Step 1 and
Step 9 and training continues from its newest checkpoint.

```bash
.venv/Scripts/python tools/build_e1_notebook.py          # writes notebooks/e1_first_run.ipynb
.venv/Scripts/python tools/build_e1_notebook.py --with-token   # unattended: gitignored e1_first_run.local.ipynb
.venv/Scripts/python tools/dry_run_e1_local.py           # optional: the same cells on CPU (tiny model, synthetic pool, 2 full epochs)
```

Open the notebook, connect with **Select Kernel → Colab → New Colab Server → GPU → T4**, and **Run All**. Follow
[`docs/phase2_e1_runbook.md`](docs/phase2_e1_runbook.md): what each step proves, the forced resume, what to do after a
disconnect, how to read `runs/e1/gate_report.md`, and the pending human task (trap annotation:
`data/trap_candidates.csv` → two annotators → `python -m laya_poc.traps merge`).

## Phase 3: the seed matrix on a Colab G4

Design doc §6.2 Phase 3 and §5.14. Phase 2's gate passed 4/4 (`docs/results/phase2_gate_report_2026-09-26.md`), so
every arm of the matrix now runs in one unattended session on a Colab **G4** (RTX PRO 6000 Blackwell). The arms are:

- E2: `laya`, seeds 11/22/33.
- E3: `laya_ml`, seeds 11/22/33.
- E4: head-only `laya`.
- E5: 7-class, both checkpoints.
- E6: learning curve at 1k/3k/10k train rows.
- B2: zero-shot, both checkpoints.

That is 12 trained runs plus 2 zero-shot runs, about 2-3.5 h in total. `config.yaml` → `phase3` defines the matrix,
and `python -m laya_poc.matrix {plan,run,report}` runs it. Each run goes through these stages:

1. Training on the G4 profile (`--card G4`: micro-batch 32 x 1, no gradient checkpointing). A run resumes from its
   checkpoint after a disconnect.
2. A temperature fit on `val`.
3. Evaluation, pre- and post-T, on every split: `val`, `test_id`, the three OOD pools, `stripped_test` and the 700 trap
   candidates. Per-row predictions and an order-invariance check are saved with it.
4. Cleanup: only `best/` is kept.
5. A row in `results/runs.csv`.

`matrix report` writes `results/phase3_report.md`. It contains per-arm means and ranges over seeds, ID→OOD gaps, ECE
before and after T, the seed-variance flag, the E6 learning curve and B2. It also runs the **Phase 3 exit check**:
every configured run must have `best/`, a temperature and `val` metrics.

The notebook first rebuilds the frozen data exactly as E1 did (`--verify-frozen`). It then builds the data variants
with `python -m laya_poc.variants {c7,subset,traps}` and re-checks CPU-vs-GPU parity on the new card for both
checkpoints. Only after that does it run the matrix, one cell per arm.

```bash
.venv/Scripts/python tools/build_p3_notebook.py          # writes notebooks/phase3_matrix.ipynb
.venv/Scripts/python tools/build_p3_notebook.py --with-token   # unattended: gitignored phase3_matrix.local.ipynb
.venv/Scripts/python tools/dry_run_p3_local.py           # optional: the same cells on CPU (tiny model, tiny matrix)
```

The notebook's parts are `tools/p3_commands.py`, `p3_cells.py`, `p3_helpers.py` and `p3_md.py`. They reuse the
Phase 1/E1 notebook infrastructure.

To run it:

1. Open the notebook and connect with **Select Kernel → Colab → New Colab Server → GPU → G4**. The G4 uses paid
   compute units, so check Colab's Resources panel.
2. Choose **Run All**.
3. After a disconnect, re-run Step 1 and then the interrupted arm's cell. Finished runs are skipped, and an
   interrupted run resumes.

[`docs/phase3_matrix_runbook.md`](docs/phase3_matrix_runbook.md) covers step durations, disk, reading the report and
`runs.csv`, failures, and what Phases 4-5 do with the saved predictions. Trap accuracy is computed after annotation.
ONNX and the CPU benchmark come in Phase 5, from a `best/` copied to Drive.

## Phase 4: the baselines on a Colab G4

Design doc §6.2 Phase 4, §5.13 and §7.10. Phase 3 passed its exit check 14/14
(`docs/results/phase3_report_2026-09-27.md`), and B2 (zero-shot Laya) ran there. Phase 4 gives every other baseline
predictions for every split, in the same run layout, `results/runs.csv` and report as the Laya runs (Phase 4 exit):

- B1: majority class and class prior, c10 and c7 (CPU).
- B3: char TF-IDF + logistic regression, C and T on `val`, c10 and c7 (CPU).
- B4: fine-tuned small encoders, ModernBERT-base and mmBERT-small, seeds 11/22/33, on the same augmented epochs
  Laya saw.
- B5 (optional, `RUN_B5`): Qwen3-4B zero-shot, scored on evaluation subsets, as a reference only.

B3 and B4 are the bars of decision criterion 1: fine-tuned Laya must beat both by at least 3 macro-F1 points on OOD.
That is 13 runs, about 1-1.5 h on a G4. `config.yaml` → `phase4` defines the arms and pins the models. The matrix
runs them (`python -m laya_poc.matrix run --only B1|B3|B4|B5`) into `runs/p4/<run_name>/`, dispatching each run to
`python -m laya_poc.baseline_runs`, `small_encoder` or `llm_baseline`. `matrix report` adds the Phase 4 exit check
(every baseline run has `preds/` for every eval split) and an early read of decision criterion 1 once the Phase 3
runs are merged in.

The notebook rebuilds the frozen data exactly as E1 did (`--verify-frozen`) and builds the c7 and trap-candidate
variants. It then runs one cell per arm group, the report and a small archive (no weights, no FSQ rows).

```bash
.venv/Scripts/python tools/build_p4_notebook.py          # writes notebooks/phase4_baselines.ipynb
.venv/Scripts/python tools/build_p4_notebook.py --with-token   # unattended: gitignored phase4_baselines.local.ipynb
.venv/Scripts/python tools/dry_run_p4_local.py           # optional: the same cells on CPU (tiny stand-in models)
```

The notebook's parts are `tools/p4_commands.py`, `p4_cells.py`, `p4_helpers.py` and `p4_md.py`, on top of the
Phase 1/E1/Phase 3 notebook infrastructure. Connect with **Select Kernel → Colab → New Colab Server → GPU → G4** and
choose **Run All**; after a disconnect, re-run Step 1 and then the interrupted arm's cell (finished runs are skipped).
To read decision criterion 1, unzip the Phase 3 and Phase 4 archives locally and merge them:
`python -m laya_poc.matrix report --runs-root runs/p3_colab/runs/p3 --runs-root runs/p4_colab/runs/p4 --archived`.
[`docs/phase4_baselines_runbook.md`](docs/phase4_baselines_runbook.md) covers the steps, durations, recovery,
reading the report, the merge and what Phase 5 does next.

## Phase 5: the full evaluation, the CPU/ONNX benchmark and the decision

Design doc §6.2 Phase 5, §5.11, §5.12 and §7.11. Phase 4 passed its exit check 13/13, and the merged Phase 3 + 4
report (`docs/results/phase34_report_2026-09-28.md`) has an early read of decision criterion 1. Phase 5 works mostly
locally, from the saved predictions of both archives; nothing is retrained. Its exit is that every results template
(§7.13) is filled in and the decision rule is evaluated:

1. Metrics and report: `python -m laya_poc.phase5_metrics` computes the full metric suite with seed ranges, flip
   rates and bootstrap CIs, and `python -m laya_poc.phase5_report` renders the templates and the decision checklist.
2. The CPU/ONNX benchmark on the laptop: `python -m laya_poc.bench_cpu --models laya laya_ml modernbert_base
   mmbert_small ...` runs PyTorch and ONNX (with the ONNX export and its acceptance check) at threads 1/2/4/8, capped
   at the physical cores.
3. The same benchmark on a 4-8-core cloud CPU: `notebooks/phase5_cpu_bench.ipynb`, on a Colab **CPU High-RAM**
   runtime (8 vCPUs), about 4-6 h. This is the machine criterion 5 is judged on.
4. Trap annotation by two annotators and the lead, following
   [`docs/trap_annotation_guide.md`](docs/trap_annotation_guide.md). `tools/p5_trap_check.py` lists the
   disagreements for the lead, `python -m laya_poc.traps merge` writes `data/trap.jsonl`, and the metrics are re-run
   with `--traps`.
5. The verdict (PASS, STOP, INVESTIGATE or no pass) and the inputs for the Phase 7 memo.

```bash
.venv/Scripts/python tools/build_p5_bench_notebook.py          # writes notebooks/phase5_cpu_bench.ipynb
.venv/Scripts/python tools/build_p5_bench_notebook.py --with-token   # unattended: gitignored phase5_cpu_bench.local.ipynb
```

The notebook's parts are `tools/p5_commands.py`, `p5_cells.py`, `p5_helpers.py` and `p5_md.py`, on top of the
Phase 1/E1/Phase 3 notebook infrastructure. It rebuilds the frozen data (`--verify-frozen`) for `test_id.jsonl`.
Step 2 installs the pinned ONNX packages (`onnx`, `onnxruntime`, `onnxscript`) under a constraints file, so the
runtime's torch, transformers, protobuf and numpy stay unchanged. Step 4 records the CPU model, the physical cores and the ISA flags, and warns if a
GPU is attached. Connect with **Select Kernel → Colab → New Colab Server → CPU → High-RAM** (not Auto Connect: 2
vCPUs) and choose **Run All**; `QUICK = True` in Step 6 is a few-minute plumbing check. The archive holds no ONNX
graphs, no weights and no FSQ rows. [`docs/phase5_runbook.md`](docs/phase5_runbook.md) covers the order of the
steps, every command, the estimates, recovery, the merge and the verdict.

## Phase 7: the write-up

Design doc §6.2 Phase 7 asks for a two-page verdict memo and the results workbook, archived with configs, hashes and
seeds. They are [`docs/results/phase7_verdict_memo_2026-09-28.md`](docs/results/phase7_verdict_memo_2026-09-28.md)
and [`phase7_results_workbook_2026-09-28.xlsx`](docs/results/phase7_results_workbook_2026-09-28.xlsx).

Everything runs locally from the Phase 3-5 outputs, except the GPU latency benchmark. The workbook needs openpyxl,
which is in the `dev` and `report` extras.

```bash
# Sample inferences: real records with every model's prediction.
# LOCAL ONLY (FSQ rows); the CLI refuses a git-tracked output path.
.venv/Scripts/python -m laya_poc.phase7_samples --data-dir data \
  --run laya_ml=runs/p3_colab/runs/p3/fsq-c10-E3-laya_ml-s11 --run laya=runs/p3_colab/runs/p3/fsq-c10-E2-laya-s11 \
  --run mmbert_small=runs/p4_colab/runs/p4/fsq-c10-B4-mmbert_small-s11 --run tfidf_lr=runs/p4_colab/runs/p4/fsq-c10-B3-tfidf_lr \
  --focus laya_ml --versus mmbert_small --splits test_id ood_country ood_script ood_brand stripped_test \
  --per-cell 3 --seed 20260928 --notice data/NOTICE_FSQ.txt \
  --out runs/phase7/sample_inferences.md --stats-out runs/phase7/sample_agreement.json

# GPU scoring throughput and training time, read from the Phase 3/4 archives.
.venv/Scripts/python -m laya_poc.phase7_timing --runs-root runs/p3_colab/runs/p3 --runs-root runs/p4_colab/runs/p4 \
  --out runs/phase7/gpu_timing.md

# The results workbook: metrics only; refuses row ids and absolute local paths.
.venv/Scripts/python -m laya_poc.phase7_workbook --report runs/phase5/phase5_report.json \
  --bench-cpu runs/phase5/bench_laptop_merged.json --gpu-timing runs/phase7/gpu_timing.json \
  --bench-gpu runs/p7_colab/runs/p7/bench_gpu.json --samples-stats runs/phase7/sample_agreement.json \
  --notice data/NOTICE_FSQ.txt --out docs/results/phase7_results_workbook_<date>.xlsx
```

**GPU per-record latency** comes from `notebooks/phase7_gpu_bench.ipynb`. It runs `python -m laya_poc.bench_gpu` for
the four models, one fresh process each, and records:
- batch-1 p50 and p95 over 500 records;
- batched throughput at batch sizes 32 and 64;
- peak VRAM.

To run it:
1. Build the notebook: `.venv/Scripts/python tools/build_p7_gpu_notebook.py`. Add `--with-token` for the gitignored
   unattended copy.
2. Connect with **Select Kernel → Colab → New Colab Server → GPU → T4** (or G4) and choose **Run All**. It takes about
   15-45 min, most of it rebuilding the frozen data.
3. Download `phase7_gpu_artifacts.zip` into `runs/`, then unzip it into `runs/p7_colab/`.
4. Pass `runs/p7_colab/runs/p7/bench_gpu.json` to the workbook with `--bench-gpu`.

[`docs/phase7_gpu_runbook.md`](docs/phase7_gpu_runbook.md) has the details.

## References

- Design doc: [`Laya Record-Normalisation PoC.md`](Laya%20Record-Normalisation%20PoC.md). Phases 1-2 and the gate
  are §6.2, the smoke-test template §7.13, troubleshooting §9.
- Laya: https://github.com/NandhaKishorM/laya (pinned commit in `config.yaml`); checkpoints
  https://huggingface.co/convaiinnovations/laya
- FSQ OS Places: https://huggingface.co/datasets/foursquare/fsq-os-places (gated; accept the terms first)
