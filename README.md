# Laya record-normalisation PoC

This PoC tests whether [Laya](https://github.com/NandhaKishorM/laya), a small non-autoregressive decision model with
calibrated probabilities, can be fine-tuned to normalise messy records into a fixed taxonomy. The proxy task is
Foursquare OS Places (FSQ): predict a place's 10-class category from its name, address and contact fields. The full
design, plan and decision rule are in [`Laya Record-Normalisation PoC.md`](Laya%20Record-Normalisation%20PoC.md).

Current phase: **Phase 2, E1 and the gate** on a free Colab T4, driven from VS Code (about 2.5-3.5 h; see
[Phase 2](#phase-2-e1-first-real-run-and-the-gate) and [`docs/phase2_e1_runbook.md`](docs/phase2_e1_runbook.md)).
Phase 1, the smoke test, passed 8/8 on a Colab T4 on 2026-09-26
([`docs/results/phase1_smoke_report_2026-09-26.md`](docs/results/phase1_smoke_report_2026-09-26.md)). It takes about
60-90 min (the longest step is the CPU fp32 parity reference, Step 7, about 25-35 min); runbook
[`docs/smoke_test_runbook.md`](docs/smoke_test_runbook.md).

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
  bundle                    packs the code into the notebook (checksummed, deterministic)
tools/build_notebook.py     generates notebooks/smoke_test.ipynb (code bundle embedded; no git/Drive needed on Colab)
  notebook_helpers, notebook_commands, notebook_cells   its parts: Step 1 helpers (embedded verbatim), the CLI
                            commands, the code cell sources
tools/build_e1_notebook.py  generates notebooks/e1_first_run.ipynb from the same parts plus e1_helpers,
                            e1_commands, e1_cells
tools/dry_run_local.py, dry_run_e1_local.py   run a notebook's cells locally on CPU with a tiny checkpoint
notebooks/*.ipynb           generated; do not edit by hand
docs/smoke_test_runbook.md, phase2_e1_runbook.md   how to run each notebook and read its report
docs/results/               committed results: reports and the frozen data fingerprint (hashes and counts only)
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
Connect). Follow [`docs/smoke_test_runbook.md`](docs/smoke_test_runbook.md).

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

## References

- Design doc: [`Laya Record-Normalisation PoC.md`](Laya%20Record-Normalisation%20PoC.md). Phases 1-2 and the gate
  are §6.2, the smoke-test template §7.13, troubleshooting §9.
- Laya: https://github.com/NandhaKishorM/laya (pinned commit in `config.yaml`); checkpoints
  https://huggingface.co/convaiinnovations/laya
- FSQ OS Places: https://huggingface.co/datasets/foursquare/fsq-os-places (gated; accept the terms first)
