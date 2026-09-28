# Phase 4 runbook: the baselines on a Colab G4 (VS Code)

This runbook covers `notebooks/phase4_baselines.ipynb`. The notebook runs Phase 4 of the design doc
(`Laya Record-Normalisation PoC.md` §6.2, baselines §5.13 and §7.10): every remaining baseline in one unattended
session on a Colab **G4** (NVIDIA RTX PRO 6000 Blackwell, 95 GB). Expected runtime: about **1-1.5 h**.

## 1. What Phase 4 is and why

The Phase 4 exit (design §6.2) is: **every baseline has predictions for every split in `preds/`**. B2 (zero-shot
Laya) already ran in Phase 3 (`docs/results/phase3_report_2026-09-27.md`). Phase 4 adds the rest:

| Arm | Runs | What it is | Why |
|---|---|---|---|
| B1 | `fsq-{c10,c7}-B1-majority`, `fsq-{c10,c7}-B1-prior` | The train majority class and the train class prior (CPU) | The floor every model must clear |
| B3 | `fsq-{c10,c7}-B3-tfidf_lr` | Char TF-IDF + logistic regression, C chosen and T fitted on `val` (CPU) | Decision criterion 1: fine-tuned Laya must beat it by >= 3 macro-F1 points on OOD |
| B4 | `fsq-c10-B4-{modernbert_base,mmbert_small}-s{11,22,33}` | Fine-tuned small encoders (ModernBERT-base, mmBERT-small), 3 seeds each | Decision criterion 1: Laya must beat the small encoder too |
| B5 | `fsq-c10-B5-qwen3_4b` (optional, `RUN_B5`) | Qwen3-4B zero-shot, scored on evaluation subsets | A reference point only (§5.13 item 6) |

Every run writes the **same run layout** as a Phase 3 Laya run, under `runs/p4/<run_name>/`: `eval/<split>.json`
(the `evaluate` multi-split schema, pre- and post-T, abstention, the stripped-test extras), `preds/<split>.jsonl`
(`{id, y, p, answer_confidence, abstained}`, post-T; plus `preds/no_gate/stripped_test.jsonl`),
`calibration.json` (`T`, fitted on `val`), `order_invariance.json` (B3 and B4), `train/summary.json` (B4) and
`done.json`. The no-evidence gate (design §5.8.1) applies to every baseline exactly as in `evaluate`: rows without
evidence abstain. The eval splits are `config.yaml` → `phase3.eval_splits`: `val`, `test_id`, `ood_country`,
`ood_script`, `ood_brand`, `stripped_test` and the 700 unannotated `trap_candidates`. The arms, models (pinned Hub
revisions) and hyper-parameters are in `config.yaml` → `phase4`. Change them there, then rebuild the notebook.

## 2. Before you start

| Need | Detail |
|---|---|
| VS Code | Extensions **Google Colab** and **Jupyter**, as for the smoke test (`docs/smoke_test_runbook.md` §1) |
| Colab | An account that can provision a **G4**. G4 is a paid accelerator: it uses compute units (Colab Pro / Pay As You Go) |
| Hugging Face | A read token for the gated dataset `foursquare/fsq-os-places` (terms accepted), as in Phases 2-3. The baseline models (ModernBERT-base, mmBERT-small, Qwen3-4B) are public and ungated |
| Time | About 1-1.5 h with VS Code open, the laptop awake and on mains power |
| Disk | Step 4 requires 20 GiB free. Qwen3-4B is about 8 GB in the Hugging Face cache; each B4 run keeps its `train/best/` weights (a few hundred MB) for the session only; the data is about 1 GB |

**Cost.** Check the Resources panel in Colab (web: the RAM/Disk widget → *View resources*) before you start and
after the last arm. Run **Colab: Remove Server** as soon as you have downloaded the archive.

Optional: run the plumbing locally first (no GPU; see §10).

## 3. Build and connect

```bash
.venv/Scripts/python tools/build_p4_notebook.py              # writes notebooks/phase4_baselines.ipynb (token prompt in Step 3)
.venv/Scripts/python tools/build_p4_notebook.py --with-token # unattended: gitignored notebooks/phase4_baselines.local.ipynb
```

The notebook embeds the project code and the frozen data manifest as a checksummed bundle. **After changing any code
or `config.yaml`, rebuild it.** `tests/test_build_p4_notebook.py` fails while the committed notebook is stale. The
unattended variant reads `HF_TOKEN=hf_...` from the repo's `.env` (gitignored) and fills in Step 3 of the gitignored
`.local.ipynb` copy. Never share, upload or commit that file. With a token, `--out` must be a gitignored
`notebooks/*.local.ipynb` (or a `*.local.ipynb` outside the repo); any other path is refused.

Connect with **Select Kernel → Colab → New Colab Server → GPU → G4**, then pick the Python 3 kernel. Do **not** use
Auto Connect: it gives a CPU server, and Step 4 fails. If the extension assigns another GPU, Step 4 prints a warning:
B1 and B3 run on the CPU anyway, B4 and B5 run slower than the estimates, and B5 needs a GPU with at least 16 GB.
`CARD` (Step 1, default `config.yaml` → `phase4.card` = `G4`) is only recorded with each run (`runs.csv` `card`); set
it to the card's profile from `train.micro_batch` (`T4`, `L4`, `A10`, `G4`) if you continue on another card.

## 4. Run it (unattended)

Click **Run All** and leave it. Every step is idempotent: a finished step prints `skip: ... exists`, and the matrix
skips finished runs (`done.json`). Full logs: `/content/laya_poc/runs/p4/logs/` (notebook steps) and
`runs/p4/<run_name>/logs/` (each run's baseline process). Every step ends with its wall-clock time
(`Step 9 - B4: 24.8 min`). Copy those times into §6 after the first real run.

| Step | What it does | Output (under `/content/laya_poc`) | Estimate on a G4 |
|---|---|---|---|
| 1 Setup | Non-interactive environment, bundle unpacked and sha256-checked, helpers, `CARD`, `RESULTS`; restores a saved HF token | `src/`, `config.yaml`, `sentinel.json` | seconds |
| 2 Install | Pinned `laya`, DuckDB pinned to `data.duckdb_version`, this project; fails if pip changed torch, transformers, protobuf or numpy | `runs/p4/logs/02_*.log` | 1-3 min |
| 3 HF token | Masked prompt (or the preset token), validated with whoami, never printed | token file on the server | seconds |
| 4 Environment | GPU, driver, VRAM, disk, versions, Laya commit; warns if the card is not `CARD` | `runs/p4/env.json` | ~30 s |
| 5 Full data | The E1 build, unchanged: FULL extraction, trap candidates held out, `--verify-frozen` against `data.frozen_manifest` | `data/`, `build_ok.json` | 3-20 min |
| 6 Variants | `variants c7` (7-class labels for B1/B3 c7) and `variants traps` (trap candidates as eval rows, c10 and c7). No learning-curve subsets | `data_c7/`, `data_eval/` (each with `build_ok.json`) | 2-5 min |
| 7 B1 | `matrix run --only B1`: majority and prior, c10 and c7 | 4 run dirs | ~2-3 min |
| 8 B3 | `matrix run --only B3`: TF-IDF + LR, c10 and c7 | 2 run dirs | ~3-5 min |
| 9 B4 | `matrix run --only B4`: ModernBERT-base and mmBERT-small, seeds 11/22/33 | 6 run dirs | ~20-35 min |
| 10 B5 | `matrix run --only B5` (`RUN_B5 = True`; `False` passes `--skip-optional`) | 1 run dir | ~10-20 min |
| 11 Report | `matrix report --runs-root runs/p4` and the Phase 4 exit check | `results/phase4_report.md`, `.json`, `runs.csv` | seconds |
| 12 Archive | Zip of the small artefacts (§9) | `phase4_artifacts.zip` | seconds |

**Where the estimates come from** (`tools/p4_md.py`; the table in the notebook's first cell is the authoritative
version). B1 fits nothing (~0.4 min per run for loading and writing every split). B3 fits the char TF-IDF and one LR
per C of `baselines.lr.C_grid` (~1.5 min per scheme; Phase 2 fitted 5 C x 2 schemes on 4 CPU workers). B4 trains
25k x 1.07 rows x 3 epochs of short JSON strings at batch 32 on the G4 (assumed ~2.5 ms per row for ModernBERT-base,
~1.5 ms for mmBERT-small), plus the evaluation: ~4 and ~3 min per run. B5 downloads Qwen3-4B (~8 GB) and scores about
10k subset rows against every option key: ~12 min. None of these is measured yet.

**What one baseline run does** (`python -m laya_poc.matrix run`, spec §5): the matrix checks that the run's data
exists (c10 → `data/`, c7 → `data_c7/`, trap candidates from `data_eval/`; a missing directory names the command
that builds it), then runs one process that writes the run layout of §1:

- **B1** (`python -m laya_poc.baseline_runs majority|prior`): fits on `data/train.jsonl` (the clean train split,
  labelled rows); T = 1; no order invariance.
- **B3** (`baseline_runs tfidf_lr`): `TfidfVectorizer` + `LogisticRegression` exactly as the Phase 2 baselines, C
  chosen on `val` macro-F1, T fitted on `val`, scores = `predict_log_proba`; order invariance on `test_id`. The c10
  run must reproduce the Phase 2 number (`val` macro-F1 0.4892, C 8, T 0.973).
- **B4** (`python -m laya_poc.small_encoder`): the pinned encoder as a sequence classifier on the row's compact JSON
  state (max length 256), soft targets (the rows' gold probabilities, as Laya's CE term), trained on the same
  augmented `train_e0..2.jsonl` Laya saw: AdamW lr 3e-5, weight decay 0.01, batch 32, 6% warm-up then linear decay,
  fp16 autocast with GradScaler on CUDA, length-bucketed batches; `val` macro-F1 after each epoch, the best epoch
  kept; then T on `val`, every split evaluated, order invariance on `test_id`, `train/summary.json`. Only
  `train/best/` is kept after the run, and it is not archived.
- **B5** (`python -m laya_poc.llm_baseline`): the pinned Qwen3-4B in bf16 (on this card), a fixed plain-text prompt
  listing every option key with its description and the record's JSON; the score of an option is the sum of its
  key tokens' log-probabilities, teacher-forced (no generation, no "thinking"); softmax over the keys. Subsets
  (`phase4.llm_eval`): the first 500 labelled `val` rows (T is fitted on them), a seeded 2000-row sample of
  `test_id` and each OOD pool, every `stripped_test` row (1000) and every trap-candidate row. No order invariance.

Then `done.json` is written and `results/runs.csv` is rewritten from every finished run. A failing run stops only
itself, with one line naming its log; the other runs of the arm still run, and the cell then raises. Fix the cause
and re-run the cell: finished runs are skipped.

## 5. After a disconnect or a kernel restart

1. Reconnect VS Code to the **same** server (Select Kernel → Colab → the existing alias). If the cell still shows as
   running, the kernel survived and the matrix continues. Do nothing.
2. If the kernel restarted, re-run **Step 1**, then the interrupted step (or **Run All**: Steps 2-6 and the finished
   arms skip themselves in seconds). A run without `done.json` starts again from the beginning: the baselines are
   not resumable, and a B4 run takes a few minutes.
3. An arm step refuses to start while a matrix or baseline process from before the restart is still running (Linux
   `/proc` check over `laya_poc.matrix`, `baseline_runs`, `small_encoder`, `llm_baseline`). Wait for it to stop, or
   stop it with `os.kill(pid, 9)` in a cell, then re-run the step.
4. If Step 1 prints `HF token: not set yet`, run Step 3 too. Only Step 5 needs the token.
5. If the server itself was removed (idle timeout, the session time limit, Remove Server), `/content` is gone: start
   again from Step 1 on a new server. Everything is rebuilt and every run is redone.

`REDO = True` in Step 1 redoes Steps 4-6 (env, data, variants) but **never** a finished run. To redo one run, delete
`runs/p4/<run_name>` and re-run its arm's step. A running cell keeps the server alive. If an `"<alias>" is idle`
toast appears, press **Cancel** within 10 s.

## 6. Reading the report and runs.csv (Step 11)

`results/phase4_report.md` (and `.json`) is the Phase 3 report extended with the baselines: per group (mean ± range
over seeds for B4) macro-F1 on `val`, `test_id` and each OOD pool, the ID→OOD gaps, ECE before → after T,
trap-candidate accuracy (**unannotated candidates**), the stripped false-confident rate and order invariance.

- **Phase 4 exit check** (last section; Step 11 also prints it as one line): one row per configured baseline run,
  PASS when every run has `done.json` and `preds/<split>.jsonl` for every eval split (B5: its subsets). B5 switched
  off (`RUN_B5 = False`) shows as "optional: not run" and is not a failure.
- **The report's Phase 3 exit check reads NOT RUN in this session**: the Laya runs live in the Phase 3 archive,
  not under `runs/p4`, so they are not in these results (NOT RUN is not a failure). The Phase 3 verdict is the Colab one (PASS 14/14,
  `docs/results/phase3_report_2026-09-27.md`).
- **Decision criterion 1 (early read)** needs the fine-tuned Laya arms too: in this session it says no Laya arm has
  finished. Merge the two archives locally (§7).

`results/runs.csv` has the Phase 3 header plus a last column `kind` (`majority`, `prior`, `tfidf_lr`,
`small_encoder`, `llm`; `laya` for the Phase 3 runs). Training columns are empty where they do not apply; `T` comes
from `calibration.json`.

Record the actual step times from the cell outputs here after the first real run:

| Step | Estimate | Actual |
|---|---|---|
| 5 Data / 6 Variants | 3-20 / 2-5 min | |
| 7 B1 / 8 B3 | 2-3 / 3-5 min | |
| 9 B4 / 10 B5 | 20-35 / 10-20 min | |

## 7. Combine with the Phase 3 archive locally

Decision criterion 1 compares each fine-tuned Laya arm (seed mean) with the best trained baseline (B3 or B4) on the
OOD average (`laya` without `ood_script`). The report merges several runs roots, so unzip both archives (the paths
are gitignored, under `runs/`) and run it from the repo root:

```bash
mkdir -p runs/p3_colab runs/p4_colab
unzip -q phase3_artifacts.zip -d runs/p3_colab   # -> runs/p3_colab/runs/p3/<run>/..., runs/p3_colab/results/...
unzip -q phase4_artifacts.zip -d runs/p4_colab   # -> runs/p4_colab/runs/p4/<run>/..., runs/p4_colab/results/...
.venv/Scripts/python -m laya_poc.matrix report --runs-root runs/p3_colab/runs/p3 --runs-root runs/p4_colab/runs/p4 --archived --out results/phase34_report.md
```

Relative `--runs-root` paths are under `--work` (default: the repo root). The report and the merged `runs.csv` go to
`results/` under the repo root, which is **not** gitignored: do not commit it by accident, and copy what you keep to
`docs/results/` with a date (metrics only, as for Phases 2-3). In the merged report:

- the section **Decision criterion 1 (early read)** has one row per fine-tuned Laya arm: its OOD average, the best
  B3/B4 group, the lead in points and PASS/FAIL at 3 points. It is information only; Phase 5 applies the decision
  rule.
- `--archived` tells the exit checks that the roots are downloaded archives: they hold no weights by design, so a
  run's recorded best step (`train/summary.json`) stands in for `best/`. Without it every trained run would list
  `best/` as missing.
- The Phase 4 exit check is the same as in the session.

## 8. When something fails

| Symptom | Likely cause and fix |
|---|---|
| Step 4: warning about the card | Not a G4: §3 (Remove Server, or continue; B5 needs >= 16 GB) |
| Step 4 errors, Step 2/5 problems | As in the Phase 2 runbook §7 (`docs/phase2_e1_runbook.md`): CPU server, pinned DuckDB, 401/403 on the gated data, frozen-manifest mismatch |
| Step 6: `variants ... exited` | Read `runs/p4/logs/06_variant_<dir>.log`. The partial directory is deleted and rebuilt on the next run of the step |
| An arm step: `matrix: <run>: <step> failed (exit N); log: ...` | Read that log. Fix the cause and re-run the step; finished runs are skipped |
| `data_c7/ ... missing` (or `data_eval`) | Step 6 did not finish: re-run Step 6, then the arm |
| B4: CUDA OOM | Not expected on a G4 (batch 32, max length 256). On a smaller card, lower `phase4.small_encoder_train.batch_size`, rebuild the notebook and delete the failed run dirs |
| B4/B5: Hub download errors (429, timeouts) | The models are public; wait a few minutes and re-run the step |
| B5: CUDA OOM or too slow | Set `RUN_B5 = False` and re-run the cell: B5 is optional and its absence does not fail the exit check |
| `... finished earlier but ...` | A step's outputs are incomplete: delete the run dir named there and re-run the step |
| Step 11: Phase 4 exit check FAIL | The report lists each incomplete run and what it lacks. Re-run that arm's step |
| Step 11: Phase 3 exit check NOT RUN | Expected in this session (§6): the Laya runs are in the Phase 3 archive |
| `still running (pid ...)` | §5 point 3 |

## 9. Archive and what comes next

Step 12 zips the small artefacts into `phase4_artifacts.zip`: every JSON, JSONL, log, markdown, YAML and CSV file
under `runs/p4/` and `results/`, excluding `ckpt/`, `best/` and `final/` (so no B4 weights); plus
`data/data_report.json`, `SHA256SUMS`, `NOTICE_FSQ.txt` and the variants' `variant.json` files. The per-row
predictions carry only ids, labels and probabilities. **No FSQ rows** go into the archive: no data JSONL, parquet or
CSV. Download the zip from the Colab view: Contents → right-click → **Download...**, then run **Colab: Remove
Server**.

**Phase 5 (evaluation and the decision)** works from the saved predictions of Phases 3 and 4 together: the full
metric suite (bootstrap CIs, abstention, stability) for every Laya arm and baseline, trap accuracy once annotation
lands (the kept `fsq_place_id`s select the annotated subset of every run's `preds/trap_candidates.jsonl`), ONNX
export and the CPU benchmark on a 4-8-core machine and the laptop, and the decision rule (§5.12), whose criterion 1
is the comparison this phase makes possible.

## 10. Local dry run (no GPU)

```bash
.venv/Scripts/python tools/dry_run_p4_local.py [--work <dir>] [--tiny-ckpt <dir>] [--tiny-encoder <dir>] [--tiny-llm <dir>]
```

The dry run executes the **same cells** on CPU: Step 1, then everything except pip and the token prompt, with
`RUN_B5 = True`. It needs Hub access for configs and tokenizers only (no weights): a tiny random Laya checkpoint
lends its tokenizer to the data build and the variants; B4 gets `--init` of a tiny random ModernBERT (the pinned
ModernBERT-base config with tiny dimensions and its tokenizer, standing in for both encoders) and B5 a tiny random
causal LM (the pinned Qwen3-4B config with tiny dimensions and its tokenizer). The data is the synthetic FULL-mode
pool of the E1/Phase 3 dry runs, and the arms are tiny: B1 and B3 on c10 and c7, B4 ModernBERT seeds 11 and 22 plus
mmBERT seed 11 for 2 epochs, B5 on 16 `val` rows and up to 12 rows per pool.

It exits 1 unless all three of these hold:

- the report's **Phase 4 exit check PASSES**;
- every configured baseline run has the §1 layout (`done.json`, `calibration.json`, `eval/` and `preds/` for every
  split, `preds/no_gate/stripped_test.jsonl`, `order_invariance.json` for B3/B4, `train/summary.json` for B4);
- `results/runs.csv` has exactly one row per baseline run under the Phase 3 header plus `kind`.

The metrics are not meaningful on tiny models. The test `test_dry_run_end_to_end_passes_the_phase_4_exit_check`
runs the dry run when `LAYA_DRY_RUN_TEST=1`.

## 11. What only the Colab run can confirm

- The real per-arm times (§6 table) and B4's speed on the G4.
- That the pinned encoders and Qwen3-4B download and run at their pinned revisions on the Colab stack (transformers
  5.x), B5 in bf16 within the card's memory.
- The B3 c10 numbers against Phase 2 on the frozen data (a local check on the frozen data is part of the baseline
  implementation).
