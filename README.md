# Laya record-normalisation PoC

This PoC tests whether [Laya](https://github.com/NandhaKishorM/laya), a small non-autoregressive decision model with
calibrated probabilities, can be fine-tuned to normalise messy records into a fixed taxonomy. The proxy task is
Foursquare OS Places (FSQ): predict a place's 10-class category from its name, address and contact fields. The full
design, plan and decision rule are in [`Laya Record-Normalisation PoC.md`](Laya%20Record-Normalisation%20PoC.md).

Current phase: **Phase 1 smoke test** on a free Colab T4, driven from VS Code. It takes about 60-90 min; the longest
step is the CPU fp32 parity reference (Step 7, about 25-35 min). See
[`docs/smoke_test_runbook.md`](docs/smoke_test_runbook.md).

## Layout

```
config.yaml                 every parameter (pinned Laya commit and Hub revision, data, training, smoke exit criteria)
src/laya_poc/               the package; every step is a CLI: python -m laya_poc.<module> --help
  config, labels, normalise, splits, sampler, io_utils, metrics, calibrate, hub, items, notice   shared building blocks
  extract, serialise, augment, rows, build_data     FSQ extraction (DuckDB over hf://) -> splits -> JSONL rows
  loss, schedule, ckpt, export, train_single        single-GPU port of the upstream trainer, resumable
  env_check, zeroshot, parity, export_check, smoke_report   Phase 1 checks and the report
  bundle                    packs the code into the notebook (checksummed, deterministic)
tools/build_notebook.py     generates notebooks/smoke_test.ipynb (code bundle embedded; no git/Drive needed on Colab)
  notebook_helpers, notebook_commands, notebook_cells   its parts: Step 1 helpers (embedded verbatim), the CLI
                            commands, the code cell sources
tools/dry_run_local.py      runs the notebook's cells locally on CPU with a tiny checkpoint (plumbing check)
notebooks/smoke_test.ipynb  generated; do not edit by hand
docs/smoke_test_runbook.md  how to run the smoke test and read its report
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
cached) and skip when the Hub is unreachable. The full local dry run of the notebook takes about 5 minutes, so it is
opt-in: `LAYA_DRY_RUN_TEST=1 .venv/Scripts/python -m pytest -q tests/test_build_notebook.py`.

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

## References

- Design doc: [`Laya Record-Normalisation PoC.md`](Laya%20Record-Normalisation%20PoC.md). Phase 1 is §6.2, the
  smoke-test template §7.13, troubleshooting §9.
- Laya: https://github.com/NandhaKishorM/laya (pinned commit in `config.yaml`); checkpoints
  https://huggingface.co/convaiinnovations/laya
- FSQ OS Places: https://huggingface.co/datasets/foursquare/fsq-os-places (gated; accept the terms first)
