"""Markdown of the Phase 3 notebook: the intro (matrix table, connection, disconnect recovery, estimates), one note
per step and the closing notes. The estimates come from config.yaml and a few stated assumptions (below)."""
from __future__ import annotations

import math

from p3_cells import matrix_arms, step_numbers
from p3_commands import B2, run_names, subset_sizes, variant_dirs

from laya_poc.config import accumulation
from laya_poc.smoke_render import e1_micro_steps

RUNTIME = "2-3.5 h"
# Estimate assumptions. E1 on the G4 (2026-09-26) took 0.064 s per T4-profile micro-step of 8 items with gradient
# checkpointing, i.e. 8 ms per item; the G4 profile (MB 32, no checkpointing) is assumed at ~5 ms per item for
# laya (ModernBERT-large) and ~3 ms for laya_ml (mmBERT-base); head-only training at 40% of that. EVAL_MIN covers a
# run's temperature fit, every eval split pre/post T (~14k rows), order invariance and three process starts.
SEC_PER_ITEM = {"laya": 0.005, "laya_ml": 0.003}
HEAD_ONLY_FACTOR, EVAL_MIN, ZERO_SHOT_MIN = 0.4, 3.0, 2.5


def _run_minutes(cfg: dict, arm: dict, n: int | None) -> float:
    items = (n or cfg["data"]["train_size"]) * (1 + cfg["augment"]["stripped_rate"]) * cfg["train"]["epochs"]
    factor = HEAD_ONLY_FACTOR if arm.get("freeze_encoder") else 1.0
    return items * SEC_PER_ITEM[arm["model"]] * factor / 60 + EVAL_MIN


def arm_minutes(cfg: dict, arm_id: str) -> float:
    """Estimated wall-clock minutes of one matrix step on a G4 (assumptions above)."""
    p3 = cfg["phase3"]
    if arm_id == B2:
        return ZERO_SHOT_MIN * len(p3.get("zero_shot") or ())
    return sum(_run_minutes(cfg, a, n) for a in p3["arms"] if a["id"] == arm_id
               for _ in a["seeds"] for n in (a.get("train_subset") or [None]))


def _span(minutes: float) -> str:
    return f"~{5 * max(1, round(minutes / 5))}-{5 * max(1, math.ceil(minutes * 1.5 / 5))} min"


def arm_summary(cfg: dict, arm_id: str) -> str:
    """One line per arm: models, scheme(s), seeds, subsets, head-only."""
    if arm_id == B2:
        return "zero-shot " + " + ".join(f"`{m}`" for m in cfg["phase3"]["zero_shot"]) + " (shipped T), c10"
    rows = [a for a in cfg["phase3"]["arms"] if a["id"] == arm_id]
    seeds = sorted({s for a in rows for s in a["seeds"]})
    parts = [" + ".join(dict.fromkeys(f"`{a['model']}`" for a in rows)),
             "/".join(dict.fromkeys(a["scheme"] for a in rows)),
             ("seed " if len(seeds) == 1 else "seeds ") + ", ".join(map(str, seeds))]
    subsets = sorted({n for a in rows for n in a.get("train_subset") or ()})
    if subsets:
        parts.append("train subsets " + "/".join(map(str, subsets)))
    if any(a.get("freeze_encoder") for a in rows):
        parts.append("head only (frozen encoder)")
    return ", ".join(parts)


def _matrix_table(cfg: dict) -> str:
    lines = ["| Step | Arm | What | Runs | Estimate (G4) |", "|---|---|---|---|---|"]
    steps = step_numbers(cfg)
    for arm in matrix_arms(cfg):
        lines.append(f"| {steps[arm]} | {arm} | {arm_summary(cfg, arm)} | {len(run_names(cfg, arm))} | "
                     f"{_span(arm_minutes(cfg, arm))} |")
    return "\n".join(lines)


def _plan_line(cfg: dict) -> str:
    card, tc = cfg["phase3"]["card"], cfg["train"]
    mb, acc = accumulation(cfg, card)
    micro = e1_micro_steps(cfg["data"]["train_size"], cfg["augment"]["stripped_rate"], 1, mb)
    return (f"A full-size run on the {card} profile: {micro} micro-steps per epoch x {tc['epochs']} epochs "
            f"(MB {mb} x ACC {acc}, gradient checkpointing {'on' if tc['grad_ckpt'].get(card) else 'off'}), a val "
            f"eval every {tc['eval_every_opt_steps']} optimiser steps with `best/` kept.")


def intro_md(cfg: dict, sha256: str, n_files: int) -> str:
    p3, steps = cfg["phase3"], step_numbers(cfg)
    trained = sum(len(run_names(cfg, a)) for a in matrix_arms(cfg) if a != B2)
    arm_steps = [steps[a] for a in matrix_arms(cfg)]
    return f"""# Laya PoC - Phase 3: the seed matrix (Colab {p3['card']} from VS Code)

Design doc §6.2 Phase 3 and §5.14: every arm of the matrix in one unattended session on a Colab **{p3['card']}** \
(NVIDIA RTX PRO 6000 Blackwell, 95 GB). Each run trains (`--card CARD`, default `{p3['card']}`), gets its temperature \
fitted on `val`, is evaluated on every split ({', '.join(p3['eval_splits'])}) with per-row predictions and order \
invariance, then its resumable checkpoints and `final/` are deleted (only `best/` is kept) and its row is added to \
`results/runs.csv`. B2 evaluates the shipped checkpoints without training. Runbook: `docs/phase3_matrix_runbook.md`.

{_matrix_table(cfg)}

{trained} trained runs + {len(p3.get('zero_shot') or ())} zero-shot. {_plan_line(cfg)} **Expected runtime:** about \
{RUNTIME} in total, including data, variants and parity (estimates from E1 on this card; see the runbook).

**How to connect:** **Select Kernel → Colab → New Colab Server → GPU → {p3['card']}**, then the Python 3 kernel. Do
**not** use *Auto Connect* (it gives a CPU). The {p3['card']} uses paid compute units: check Colab's Resources panel.
If the extension assigns another GPU, Step 4 warns: set `CARD` in Step 1 to that card's profile before Step \
{arm_steps[0]}, or Remove Server and retry. Keep VS Code open and the laptop awake: a running cell keeps the server \
alive. The Hugging Face token is asked in a masked input box (Step 3) and never stored in this notebook.

**After a kernel restart or a Colab disconnect:** re-run **Step 1**, then the interrupted step (or **Run All**). \
Finished steps skip themselves (their outputs are on disk), the matrix skips finished runs (`done.json`) and a run \
with a resumable checkpoint resumes from it. `REDO = True` in Step 1 redoes Steps 4-7 but never a finished matrix \
run: delete `runs/p3/<run_name>` to redo one. If the server itself was removed, `/content` is gone: start again from \
Step 1.

**Phase 3 exit check** (Step {steps['report']}): every configured arm and seed has `best/`, a temperature and val \
metrics (`results/phase3_report.md`, `results/runs.csv`).

Code bundle: {n_files} files, sha256 `{sha256}` (verified when Step 1 unpacks it).
"""


def _arm_note(cfg: dict, arm: str) -> str:
    names = ", ".join(f"`{n}`" for n in run_names(cfg, arm))
    if arm == B2:
        return (f"{names}: the shipped checkpoints (shipped temperatures) on every eval split, predictions and order "
                f"invariance; no training. {_span(arm_minutes(cfg, arm))}.")
    return (f"{names}. Each run: train → temperature fit on `val` → every eval split (post-/pre-T) and predictions → "
            "order invariance → delete `ckpt/` and `final/` → `done.json` and its `results/runs.csv` row. Finished "
            "runs are skipped; an interrupted run resumes from its newest checkpoint. "
            f"{_span(arm_minutes(cfg, arm))}.")


def step_notes(cfg: dict) -> dict[str, tuple[str, str]]:
    """(title, note) per cell key, in notebook order."""
    s, card, subsets = step_numbers(cfg), cfg["phase3"]["card"], subset_sizes(cfg)
    notes = {
        "setup": ("Setup", f"Non-interactive environment, work dir, code bundle, helpers, and `CARD` (default "
                  f"`{card}`, config `phase3.card`). Seconds. **Re-run this cell first after any kernel restart or "
                  "Colab disconnect.**"),
        "install": ("Install", f"Plain `laya` at the pinned commit, DuckDB pinned to {cfg['data']['duckdb_version']} "
                    "(the frozen splits depend on its hash()), this project. Fails if pip changed Colab's torch, "
                    "transformers, protobuf or numpy. 1-3 min."),
        "token": ("Hugging Face token", "Masked input box at the top of VS Code; never printed. Unattended: "
                  "`python tools/build_p3_notebook.py --with-token` and open the gitignored "
                  "`notebooks/phase3_matrix.local.ipynb`."),
        "env": ("Environment check", f"GPU, driver, VRAM, disk, versions, Laya commit -> `runs/p3/env.json`. Warns "
                f"when the GPU is not a {card} (then set `CARD` in Step 1). ~30 s."),
        "data": ("Full data (frozen)", "The E1 build, identical: FULL extraction over `hf://`, trap candidates held "
                 "out of every split, JSONL, then `--verify-frozen` (the fingerprint must equal "
                 "`data.frozen_manifest`, else the step fails). 3-20 min."),
        "variants": ("Data variants", f"From the frozen `data/`: {', '.join(f'`{d}/`' for d in variant_dirs(cfg))}"
                     f" - the 7-class labels (E5), stratified train subsets n = {', '.join(map(str, subsets))} with "
                     "the same augmentation (E6), and the trap candidates as eval rows for c10 and c7. A partial "
                     "directory from an interrupted run is deleted and rebuilt. 2-10 min."),
        "parity_laya": ("Parity laya", "Design doc §7.7, again on this card: CPU fp32 reference vs GPU fp16 on "
                        f"{cfg['smoke']['parity_n']} val rows, plus padded vs unpadded batches. Stops the notebook on "
                        "FAIL (`PARITY_MUST_PASS`). 2-15 min: the reference runs on the host CPU."),
        "parity_laya_ml": ("Parity laya-multilingual", "The same check for `laya_ml`. 2-15 min."),
        "plan": ("Matrix plan", "Every configured run and its status (done / partial / todo). Seconds."),
    }
    notes.update({arm: (f"{arm}: {arm_summary(cfg, arm)}", _arm_note(cfg, arm)) for arm in matrix_arms(cfg)})
    notes["report"] = ("Report and the Phase 3 exit check", "`matrix report`: per arm mean ± range over seeds of "
                       "macro-F1 on val, test_id and each OOD pool, ID→OOD gaps, ECE pre→post, trap-candidate accuracy "
                       "(unannotated candidates), stripped false-confident rate, order invariance, the seed-variance "
                       "flag, the E6 learning curve, B2, and the exit check -> `results/phase3_report.md` (+ `.json`, "
                       "`runs.csv`). Seconds.")
    notes["archive"] = ("Archive", "---\nZip logs, JSON, per-row predictions, `results/` and the FSQ NOTICE (no "
                        "checkpoints, no FSQ rows) for **Download...**. Seconds.")
    notes["drive"] = ("Drive copy (optional, off)", "Copies the archive and/or every run's `best/` to Google Drive "
                      "(interactive sign-in, about 2 min; `best/` is about 0.9 GB per `laya` run).")
    return {k: (f"Step {s[k]} - {title}", note) for k, (title, note) in notes.items()}


def finish_md(cfg: dict) -> str:
    return f"""### Finish
- The exit check is at the end of Step {step_numbers(cfg)['report']}; every run's metrics are in `results/runs.csv`.
- Download `phase3_artifacts.zip`: Colab view in the activity bar > Contents > right-click > **Download...**
- Next (runbook): trap annotation turns the saved trap-candidate predictions into trap accuracy; Phase 5 computes the
  full metric suite from the saved predictions, and its ONNX export and CPU benchmark need one `best/` checkpoint:
  copy it to Drive (Step {step_numbers(cfg)['drive']}) before removing the server, or re-train that run later.
- Run **Colab: Remove Server** when you are done. It stops the compute-unit spend and deletes `/content`.
"""
