# Phase 3 runbook: the seed matrix on a Colab G4 (VS Code)

This runbook covers `notebooks/phase3_matrix.ipynb`. The notebook runs Phase 3 of the design doc
(`Laya Record-Normalisation PoC.md` §6.2, matrix §5.14): every arm of the experiment matrix in one unattended session
on a Colab **G4** (NVIDIA RTX PRO 6000 Blackwell, 95 GB). Expected runtime: about **2-3.5 h**.

## 1. What Phase 3 is and why

Phase 2 (E1) proved the pipeline on one run and passed the gate 4/4
(`docs/results/phase2_gate_report_2026-09-26.md`). Phase 3 answers the questions one run cannot:

| Arm | Runs | Question |
|---|---|---|
| E2 | `laya`, c10, seeds 11, 22, 33 | How much does the fine-tuned model vary with the seed? (E1's weights were not kept, so seed 11 is re-run under the same card profile.) |
| E3 | `laya_ml` (laya-multilingual), c10, seeds 11, 22, 33 | Does the multilingual checkpoint do better, above all on the OOD script pool? |
| E4 | `laya`, c10, seed 11, head only | The lower bound: how much of the gain needs the encoder at all (`--freeze-encoder`)? |
| E5 | `laya` and `laya_ml`, 7-class (c7), seed 11 | The coarser taxonomy |
| E6 | `laya`, c10, seed 11, 1000/3000/10000 train rows | The learning curve (25k is the E2 seed-11 run) |
| B2 | zero-shot `laya`, `laya_ml` (shipped temperatures) | The reference every fine-tuned arm must beat |

Every trained run is temperature-fitted on `val` and evaluated **in the same session** on every eval split
(`val`, `test_id`, `ood_country`, `ood_script`, `ood_brand`, `stripped_test`, and the 700 unannotated
`trap_candidates`), with per-row predictions and an order-invariance check. The later phases work from those saved
predictions, so no checkpoint has to outlive the session. The whole matrix is configured in `config.yaml` →
`phase3`. Change the arms, seeds, splits or card there, then rebuild the notebook.

## 2. Before you start

| Need | Detail |
|---|---|
| VS Code | Extensions **Google Colab** and **Jupyter**, as for the smoke test (`docs/smoke_test_runbook.md` §1) |
| Colab | An account that can provision a **G4**. G4 is a paid accelerator: it uses compute units (Colab Pro / Pay As You Go) |
| Hugging Face | A read token for the gated dataset `foursquare/fsq-os-places` (terms accepted), as in Phase 2 |
| Time | About 2-3.5 h with VS Code open, the laptop awake and on mains power |
| Disk | About 65 GB free on the G4 runtime is plenty. Step 4 requires 20 GiB. A running arm holds up to 2 resumable checkpoints (about 5 GB each for `laya`), deleted when the run finishes. Each finished run keeps only `train/best/` (about 0.9 GB for `laya`, fp16): about 11 GB for the 12 trained runs, plus about 1 GB of data |

**Cost.** Check the Resources panel in Colab (web: the RAM/Disk widget → *View resources*; it shows the compute-unit
balance and burn rate) before you start and after Step 9. On any other card the matrix takes longer and costs a
different amount. Run **Colab: Remove Server** as soon as you have downloaded the archive: an idle G4 keeps spending
units.

Optional: run the plumbing locally first (no GPU; about 30-45 min on a laptop). See §9.

## 3. Build and connect

```bash
.venv/Scripts/python tools/build_p3_notebook.py            # writes notebooks/phase3_matrix.ipynb (token prompt in Step 3)
.venv/Scripts/python tools/build_p3_notebook.py --with-token   # unattended: gitignored notebooks/phase3_matrix.local.ipynb
```

The notebook embeds the project code and the frozen data manifest as a checksummed bundle. **After changing any code
or `config.yaml`, rebuild it.** `tests/test_build_p3_notebook.py` fails while the committed notebook is stale. The
unattended variant reads `HF_TOKEN=hf_...` from the repo's `.env` (gitignored) and fills in Step 3 of the gitignored
`.local.ipynb` copy. Never share, upload or commit that file. With a token, `--out` must be a gitignored
`notebooks/*.local.ipynb` (or a `*.local.ipynb` outside the repo); any other path is refused.

Connect with **Select Kernel → Colab → New Colab Server → GPU → G4**, then pick the Python 3 kernel. Do **not** use
Auto Connect: it gives a CPU server, and Step 4 fails.

If the extension assigns another GPU, Step 4 prints a warning naming it. Every run trains with the card profile in
`CARD` (Step 1; default `config.yaml` → `phase3.card` = `G4`: micro-batch 32 x accumulation 1, no gradient
checkpointing). You then have two choices:

- **Remove Server** and ask for a G4 again (preferred: the estimates and the report's `card` column assume it).
- Set `CARD` in Step 1 to that card's profile from `train.micro_batch` (`T4`, `L4`, `A10` or `G4`), re-run Step 1,
  and continue. Do this **before the first training step (Step 10)**. A run resumes only with the settings it started
  with, so changing `CARD` later makes an interrupted run refuse to resume. Finished runs are unaffected. To restart
  an interrupted run under the new card, delete its directory `runs/p3/<run_name>`.

## 4. Run it (unattended)

Click **Run All** and leave it. Every step is idempotent. A finished step prints `skip: ... exists`; the matrix skips
finished runs (`done.json`) and, inside a run, finished stages. So Run All is also how you continue after a restart
(§5). Full logs: `/content/laya_poc/runs/p3/logs/` (notebook steps) and `runs/p3/<run_name>/logs/` (each run's
train, export_check and evaluate). Cell output is sparse on purpose: about one training line a minute. Every step ends
with its wall-clock time (`Step 10 - E2: 41.3 min`). Copy those times into §6 after the first real run.

| Step | What it does | Output (under `/content/laya_poc`) | Estimate on a G4 |
|---|---|---|---|
| 1 Setup | Non-interactive environment, bundle unpacked and sha256-checked, helpers, `CARD`, `RESULTS`; restores a saved HF token | `src/`, `config.yaml`, `sentinel.json` | seconds |
| 2 Install | Pinned `laya`, DuckDB pinned to `data.duckdb_version`, this project; fails if pip changed torch, transformers, protobuf or numpy | `runs/p3/logs/02_*.log` | 1-3 min |
| 3 HF token | Masked prompt (or the preset token), validated with whoami, never printed | token file on the server | seconds |
| 4 Environment | GPU, driver, VRAM, disk, versions, Laya commit; warns if the card is not `CARD` | `runs/p3/env.json` | ~30 s |
| 5 Full data | The E1 build, unchanged: FULL extraction, trap candidates held out, `--verify-frozen` against `data.frozen_manifest` | `data/`, `build_ok.json` | 3-20 min (E1 on this card: 63 s extraction) |
| 6 Variants | `variants c7` (7-class labels, same rows), `variants subset --n 1000/3000/10000` (stratified, same augmentation), `variants traps` (trap candidates as eval rows, c10 and c7) | `data_c7/`, `data_lc<N>/`, `data_eval/` (each with `build_ok.json`) | 2-10 min |
| 7a/7b Parity | Design doc §7.7 "again on each new card": CPU fp32 reference vs GPU fp16 on 200 `val` rows plus padded vs unpadded, for `laya` then `laya_ml`. **Stops the notebook on FAIL** | `runs/p3/parity_*.json` | 2-15 min each |
| 8 Plan | `matrix plan`: every configured run with status done / partial / todo | log only | seconds |
| 9 B2 | `matrix run --only B2`: zero-shot evaluation of both shipped checkpoints | `runs/p3/fsq-c10-B2-*-zs/` | ~5-10 min |
| 10 E2 | `matrix run --only E2` | 3 run dirs | ~35-55 min |
| 11 E3 | `matrix run --only E3` | 3 run dirs | ~25-40 min |
| 12 E4 | `matrix run --only E4` (head only) | 1 run dir | ~5-10 min |
| 13 E5 | `matrix run --only E5` (c7, both models) | 2 run dirs | ~20-35 min |
| 14 E6 | `matrix run --only E6` (learning curve) | 3 run dirs | ~15-25 min |
| 15 Report | `matrix report` and the Phase 3 exit check | `results/phase3_report.md`, `.json`, `runs.csv` | seconds |
| 16a Archive | Zip of the small artefacts (below) | `phase3_artifacts.zip` | seconds |
| 16b Drive (off) | Optional copy of the archive and/or every `best/` to Drive | `MyDrive/laya_poc/phase3/` | minutes |

**Where the estimates come from.** E1 on this card ran the T4 profile at 0.064 s per micro-step of 8 items with
gradient checkpointing (8 ms per item; 17 min for 25k x 4 epochs including 14 evals). The G4 profile (micro-batch 32,
no checkpointing) is assumed at about 5 ms per item for `laya` and about 3 ms for `laya_ml` (mmBERT-base), with
head-only training at 40% of that. Each run adds about 3 min for the temperature fit, every eval split before and
after T (about 14k rows), order invariance and three process starts. A full-size `laya` run is 836 micro-steps per
epoch x 4 epochs, with a `val` eval every 250 optimiser steps. The estimates are generated from `config.yaml`
(`tools/p3_md.py`). The table in the notebook's first cell is the authoritative version.

**What one matrix run does** (`python -m laya_poc.matrix run`, spec §5):

1. `train_single` with `--card CARD --epochs 4 --save-best --final-eval`, plus `--freeze-encoder` for E4. The E6
   subsets evaluate every max(10, total_opt // 12) optimiser steps (cap 250). Training resumes by itself from
   `train/ckpt/` if the directory exists.
2. `export_check` fits the temperature T on `val`, writes it into `train/best/` and checks the reload
   (`export_check.json`).
3. `evaluate` scores every eval split post-T and pre-T, writing `eval/<split>.json`, `preds/<split>.jsonl`, the
   abstention threshold chosen on `val`, and the stripped false-confident rate. It then checks order invariance on
   `test_id` (1000 rows x 5 field-order permutations → `order_invariance.json`).
4. Everything under `train/` except `best/` and the small files is deleted: `ckpt/` and `final/`.
5. `done.json` is written, then `results/runs.csv` is rewritten from every finished run.

A failing stage stops only its own run, with one line naming the stage log. The other runs of the arm still run, and
the cell then raises. Fix the cause and re-run the cell: finished runs and stages are skipped.

## 5. After a disconnect or a kernel restart

Every run is resumable at the level of a stage. Training itself resumes from its newest checkpoint: one every
`train.ckpt_every_min` minutes (15), and at most 2 kept.

1. Reconnect VS Code to the **same** server (Select Kernel → Colab → the existing alias). If the cell still shows as
   running, the kernel survived and the matrix continues. Do nothing.
2. If the kernel restarted, re-run **Step 1**, then the interrupted step (or **Run All**: Steps 2-9 and the finished
   arms skip themselves in seconds). The interrupted run resumes from its newest checkpoint. A run that had not yet
   written a checkpoint starts its training again.
3. An arm step refuses to start while a matrix or trainer process from before the restart is still running (Linux
   `/proc` check). Wait for it to stop, or stop it with `os.kill(pid, 9)` in a cell (checkpoints are written
   atomically), then re-run the step.
4. If Step 1 prints `HF token: not set yet`, run Step 3 too. Only Step 5 needs the token.
5. If the server itself was removed (idle timeout, the session time limit, Remove Server), `/content` is gone: start again from
   Step 1 on a new server. Everything is rebuilt, and every run is redone. Only what you copied to Drive or downloaded
   survives.

`REDO = True` in Step 1 redoes Steps 5-7 (data, variants, parity) but **never** a finished matrix run. To redo one run,
delete `runs/p3/<run_name>` and re-run its arm's step. A running cell keeps the server alive. If an `"<alias>" is idle`
toast appears, press **Cancel** within 10 s.

## 6. Reading the report and runs.csv (Step 15)

The header of `results/phase3_report.md` (and `phase3_report.json`) gives the verdict of the **Phase 3 exit check**
(design doc §6.2 Phase 3 exit). Every configured run must have `done.json`, trained runs must have `best/` weights, and
every run must have a temperature T and `val` macro-F1 in `runs.csv`. The last section, "Phase 3 exit check", has one
row per run showing what an incomplete run is missing. In between come:

- per arm (and model / scheme / subset), the mean ± range over seeds of: macro-F1 on `val`, `test_id` and each OOD pool;
  the ID→OOD gap per pool (design §5.11); ECE before → after the temperature (`val`, `test_id`); trap-candidate
  accuracy (labelled **unannotated candidates**: not the trap metric yet); the stripped false-confident rate; and order
  invariance;
- the **seed-variance flag**: an arm whose seeds differ by more than 3 macro-F1 points is marked "investigate"
  (design §5.12);
- the **E6 learning curve**: train rows → `val` macro-F1 (the 25k point is E2 seed 11);
- the B2 zero-shot rows, the reference every arm is compared with.

`results/runs.csv` has one row per finished run, sorted by run name, with a fixed header (spec §1): identity (`run_name,
arm, model, scheme, seed, subset, head_only, card`), training (`epochs_run, stop_reason, best_opt_step`), calibration
(`T, clamped`), `val_macro_f1, val_macro_f1_9, val_acc, val_ece_pre, val_ece_post`, macro-F1 on `test_id` and each OOD
pool, `trap_candidates_acc, stripped_false_confident, order_invariance`, and `train_seconds, run_seconds`. An empty
cell means that metric was not produced (for example no training row for B2).

Run directory layout (`runs/p3/<run_name>/`): `train/` (log.jsonl, summary.json, best/), `export_check.json`,
`eval/<split>.json`, `preds/<split>.jsonl` (`{id, y, p, answer_confidence, abstained}`, post-T), `order_invariance.json`,
`logs/`, `done.json`.

Record the actual step times from the cell outputs here after the first real run:

| Step | Estimate | Actual |
|---|---|---|
| 5 Data / 6 Variants / 7 Parity | 3-20 / 2-10 / 4-30 min | |
| 9 B2 / 10 E2 / 11 E3 | 5-10 / 35-55 / 25-40 min | |
| 12 E4 / 13 E5 / 14 E6 | 5-10 / 20-35 / 15-25 min | |

## 7. When something fails

| Symptom | Likely cause and fix |
|---|---|
| Step 4: warning about the card | Not a G4: §3 (Remove Server, or set `CARD` before Step 10) |
| Step 4 errors, Step 2/5 problems | As in the Phase 2 runbook §7 (`docs/phase2_e1_runbook.md`): CPU server, pinned DuckDB, 401/403 on the gated data, frozen-manifest mismatch |
| Step 6: `variants ... exited` | Read `runs/p3/logs/06_variant_<dir>.log`. The partial directory is deleted and rebuilt on the next run of the step. A budget rejection (a c7 row too long) is a real data problem: do not work around it |
| Step 7: `parity FAILED` | The card's fp16 path disagrees with the CPU fp32 reference (design §7.7). Do not train on it: Remove Server and try another server. Only if you accept the risk, set `PARITY_MUST_PASS = False` in the cell and note it in the results |
| An arm step: `matrix: <run>: <stage> failed (exit N); log: ...` | Read that log. Fix the cause and re-run the step; finished runs and stages are skipped |
| `train` failed: CUDA OOM | Only on a smaller card: use that card's profile in `CARD` (§3). Delete the run dir so it starts afresh under the new profile |
| `train` failed: refuses to resume (settings differ) | `CARD` or `config.yaml` changed since the run started. Restore them, or delete `runs/p3/<run_name>` |
| `... finished earlier but ...; delete <run dir> to redo it` | A stage's output is incomplete (for example no T in `export_check.json`). Delete the run dir named there and re-run the step |
| `export_check: WARNING: a check FAILED` | T was fitted, but a round-trip or cross-check failed. The run continues and the report lists it. Read `export_check.json` |
| `data_c7/ ... missing` (or `data_lc<N>`, `data_eval`) | Step 6 did not finish: re-run Step 6, then the arm |
| Step 15: exit check FAIL | The report lists each incomplete run and what it lacks. Re-run that arm's step |
| `still running (pid ...)` | §5 point 3 |

## 8. Archive, Drive and what comes next

Step 16a zips the small artefacts into `phase3_artifacts.zip`: every JSON, JSONL, log, markdown, YAML and CSV file under
`runs/p3/` and `results/`, excluding `ckpt/`, `best/` and `final/`; plus `data/data_report.json`, `SHA256SUMS`,
`NOTICE_FSQ.txt` and the variants' `variant.json` files. The per-row predictions carry only ids, labels and
probabilities. **No FSQ rows** go into the archive: no data JSONL, parquet or CSV. Download the zip from the Colab
view: Contents → right-click → **Download...**

Step 16b is off by default. `COPY_TO_DRIVE = True` copies the archive to `MyDrive/laya_poc/phase3/`, and
`COPY_BEST_TO_DRIVE = True` copies every run's `best/` there (with its NOTICE files). That is about 11 GB for all 12,
so copy only the runs you need. Both need an interactive Google sign-in (about 2 min). Some Workspace accounts fail.

**What the later phases do with the predictions:**

- **Trap annotation** (human task, design §5.3). `data/trap_candidates.csv` → two annotators → `python -m laya_poc.traps
  merge`. The kept `fsq_place_id`s then select the annotated subset of every run's `preds/trap_candidates.jsonl`, and
  trap accuracy is computed offline without re-running anything. The report's trap column stays "unannotated
  candidates" until then.
- **Phase 4 (baselines)**: B1-B5 need predictions for every split in the same form. B2 is already done here.
- **Phase 5 (evaluation and CPU benchmarking)**: the full metric suite (bootstrap CIs, abstention, stability) runs
  from the saved `preds/` and `eval/` files. The **ONNX export and the CPU benchmark** on a 4-8-core machine and the
  laptop need one checkpoint. Copy the chosen run's `best/` to Drive in Step 16b before you remove the server, or
  re-train that one run later.

Then run **Colab: Remove Server**. This stops the compute-unit spend and deletes `/content`, including the saved token.

## 9. Local dry run (no GPU)

```bash
.venv/Scripts/python tools/dry_run_p3_local.py [--work <dir>] [--tiny-ckpt <existing tiny checkpoint dir>]
```

The dry run executes the **same cells** on CPU: Step 1, then everything except pip and the token prompt. It uses a
tiny random Laya checkpoint (init scale 0.5; needs Hub access for the tokenizer) that stands in for both models via
`--init`. The data is a synthetic FULL-mode pool (all 15 countries, brand chains, trap-word names, mixed-level-1 rows,
so trap candidates exist), and the matrix is tiny: E2 seeds 11 and 22, E3 (`laya_ml`), E4 head-only, both E5 rows, E6
subsets 16 and 32, and B2 for both models. It runs 1 epoch, with a `val` eval every 8 optimiser steps.

It exits 1 unless all three of these hold:

- the report's **Phase 3 exit check PASSES**;
- every configured run has the spec §1 layout (`done.json`, `export_check.json`, `eval/` and `preds/` for every
  split, `order_invariance.json`, `train/best/`, with `ckpt/` and `final/` removed);
- `results/runs.csv` has exactly one row per run under the §1 header.

The metrics themselves are not meaningful on a tiny model. The test `test_dry_run_end_to_end_passes_the_phase_3_exit_check` runs the
dry run when `LAYA_DRY_RUN_TEST=1`.

## 10. What only the Colab run can confirm

- The G4 profile's real speed (micro-batch 32 without gradient checkpointing) and VRAM, and the real per-arm times
  (§6 table).
- Parity on the Blackwell card (design §7.7), for both checkpoints.
- The variants build (c7, subsets, traps) with the real tokenizers on the frozen data.
- That a multi-hour Run All survives in the VS Code Colab extension, with its idle toasts, a reconnect to a running
  kernel, and a restart in the middle of an arm.
