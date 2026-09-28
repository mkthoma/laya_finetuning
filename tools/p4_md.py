"""Markdown of the Phase 4 notebook: the intro (baseline table, connection, disconnect recovery, estimates), one
note per step and the closing notes. The estimates come from config.yaml and a few stated assumptions (below)."""
from __future__ import annotations

import math

from p4_cells import run_flag, step_numbers
from p4_commands import REPORT_MD, arm_ids, arm_kinds, arms, optional_arm, row_run_names, run_names, variant_dirs

RUNTIME = "1-1.5 h"
# Estimate assumptions (minutes per run on a G4, evaluation of every split included). B1 fits nothing; B3 fits the
# char TF-IDF once per scheme and one LR per C (Phase 2: 5 C x 2 schemes on 4 CPU workers); B4 trains 25k x 1.07 x 3
# epochs of short JSON strings at batch 32 (ModernBERT-base ~2.5 ms per item, mmBERT-small ~1.5 ms) plus the eval;
# B5 downloads Qwen3-4B (~8 GB) and scores ~10k subset rows x every option key without generating.
RUN_MIN = {"majority": 0.4, "prior": 0.4, "tfidf_lr": 1.5, "llm": 12.0}
SMALL_ENCODER_MIN = {"modernbert_base": 4.0, "mmbert_small": 3.0}
SETUP_MIN = "10-15"  # Steps 1-5: install, token, env, the frozen data build
VARIANTS_MIN = "2-5"


def _run_minutes(arm: dict, name: str) -> float:
    if arm["kind"] == "small_encoder":
        return SMALL_ENCODER_MIN.get(arm.get("model"), max(SMALL_ENCODER_MIN.values()))
    kind = name.rsplit("-", 1)[-1] if arm["kind"] == "majority" else arm["kind"]
    return RUN_MIN.get(kind, RUN_MIN["llm"])


def arm_minutes(cfg: dict, arm_id: str) -> float:
    """Estimated wall-clock minutes of one arm group on a G4 (assumptions above)."""
    return sum(_run_minutes(a, n) for a in arms(cfg) if a["id"] == arm_id for n in row_run_names(a))


def span(minutes: float) -> str:
    """'~lo-hi min': 1-minute steps below 10 minutes, 5-minute steps above; hi is 1.5x (G4 speed is assumed)."""
    step = 5 if minutes >= 10 else 1
    lo = max(step, step * round(minutes / step))
    return f"~{lo}-{max(lo + step, step * math.ceil(minutes * 1.5 / step))} min"


def arm_summary(cfg: dict, arm_id: str) -> str:
    """One line per arm group: what it is, models, schemes, seeds, optional."""
    rows = [a for a in arms(cfg) if a["id"] == arm_id]
    what = {"majority": "majority class + class prior (CPU)", "prior": "class prior (CPU)",
            "tfidf_lr": "char TF-IDF + logistic regression, C and T on val (CPU)",
            "small_encoder": "fine-tuned small encoder", "llm": "zero-shot LLM reference, scored subsets"}
    parts = [" + ".join(dict.fromkeys(what.get(k, k) for k in arm_kinds(cfg, arm_id)))]
    models = list(dict.fromkeys(f"`{a['model']}`" for a in rows if a.get("model")))
    if models:
        parts.append(" + ".join(models))
    parts.append("/".join(dict.fromkeys(s for a in rows for s in (a.get("schemes") or [a["scheme"]]))))
    seeds = sorted({s for a in rows for s in a.get("seeds") or ()})
    if seeds:
        parts.append(("seed " if len(seeds) == 1 else "seeds ") + ", ".join(map(str, seeds)))
    if optional_arm(cfg, arm_id):
        parts.append(f"optional (flag `{run_flag(arm_id)}`)")
    return ", ".join(parts)


def _table(cfg: dict) -> str:
    lines = ["| Step | Arm | What | Runs | Estimate (G4) |", "|---|---|---|---|---|"]
    steps = step_numbers(cfg)
    for arm in arm_ids(cfg):
        lines.append(f"| {steps[arm]} | {arm} | {arm_summary(cfg, arm)} | {len(run_names(cfg, arm))} | "
                     f"{span(arm_minutes(cfg, arm))} |")
    return "\n".join(lines)


def intro_md(cfg: dict, sha256: str, n_files: int) -> str:
    p4, steps = cfg["phase4"], step_numbers(cfg)
    n_runs = sum(len(run_names(cfg, a)) for a in arm_ids(cfg))
    return f"""# Laya PoC - Phase 4: the baselines (Colab {p4['card']} from VS Code)

Design doc §6.2 Phase 4, §5.13 and §7.10: every baseline gets predictions for every split, in the same run layout, \
`results/runs.csv` and report as the Phase 3 matrix (B2, zero-shot Laya, already ran there). B3 (TF-IDF + LR) and B4 \
(fine-tuned small encoders, 3 seeds each) are the references of decision criterion 1: fine-tuned Laya must beat both \
by at least 3 macro-F1 points out of distribution. Each run fits on the clean train split (B4 on the same augmented \
epochs Laya saw), gets its temperature on `val`, and is evaluated on every split \
({', '.join(cfg['phase3']['eval_splits'])}) with per-row predictions under `runs/p4/<run>/preds/`. Runbook: \
`docs/phase4_baselines_runbook.md`.

{_table(cfg)}

{n_runs} runs. **Expected runtime:** about {RUNTIME} in total: setup and the frozen data {SETUP_MIN} min, variants \
{VARIANTS_MIN} min, then the arms above (estimates, see the runbook).

**How to connect:** **Select Kernel → Colab → New Colab Server → GPU → {p4['card']}**, then the Python 3 kernel. Do
**not** use *Auto Connect* (it gives a CPU). The {p4['card']} uses paid compute units: check Colab's Resources panel.
If the extension assigns another GPU, Step 4 warns (B4 and B5 then take longer). Keep VS Code open and the laptop \
awake: a running cell keeps the server alive. The Hugging Face token is asked in a masked input box (Step 3) and \
never stored in this notebook.

**After a kernel restart or a Colab disconnect:** re-run **Step 1**, then the interrupted step (or **Run All**). \
Finished steps skip themselves (their outputs are on disk) and the matrix skips finished runs (`done.json`); an \
unfinished baseline run starts again (B4 runs take a few minutes each and are not resumable). `REDO = True` in Step 1 \
redoes Steps 4-6 but never a finished run: delete `runs/p4/<run_name>` to redo one. If the server itself was removed, \
`/content` is gone: start again from Step 1.

**Phase 4 exit check** (Step {steps['report']}, design §6.2): every configured baseline run has predictions for every \
eval split (B5: its subsets; B5 switched off is not a failure). The report also gives an early read of decision \
criterion 1 once it is merged with the Phase 3 runs locally (runbook).

Code bundle: {n_files} files, sha256 `{sha256}` (verified when Step 1 unpacks it).
"""


def _arm_note(cfg: dict, arm: str) -> str:
    names = ", ".join(f"`{n}`" for n in run_names(cfg, arm))
    kinds, est = set(arm_kinds(cfg, arm)), span(arm_minutes(cfg, arm))
    if "small_encoder" in kinds:
        how = ("Each run: fine-tune on the augmented `train_e*` epochs (soft targets, fp16 on CUDA), keep the best "
               "val epoch, fit T on `val`, evaluate every split with predictions, order invariance on `test_id`, "
               "then keep only `train/best/` for this session.")
    elif "llm" in kinds:
        flag = run_flag(arm)
        how = (f"The pinned model scores every option key (teacher-forced log-probabilities, no generation) on "
               f"subsets: the first {cfg['phase4']['llm_eval']['val_for_temperature']} labelled `val` rows (T), up "
               f"to {cfg['phase4']['llm_eval']['subset_per_pool']} rows of `test_id` and each OOD pool, all of "
               f"`stripped_test` and the trap candidates. Optional: `{flag} = False` in the cell skips it (recorded "
               "as optional, disabled; not an exit-check failure).")
    else:
        how = ("Fit on the clean train split on the CPU, T on `val`, every split evaluated with predictions (the "
               "no-evidence gate as in `evaluate`).")
    return f"{names}. {how} Finished runs are skipped. {est}."


def step_notes(cfg: dict) -> dict[str, tuple[str, str]]:
    """(title, note) per cell key, in notebook order."""
    s, card = step_numbers(cfg), cfg["phase4"]["card"]
    notes = {
        "setup": ("Setup", f"Non-interactive environment, work dir, code bundle, helpers, and `CARD` (default "
                  f"`{card}`, config `phase4.card`). Seconds. **Re-run this cell first after any kernel restart or "
                  "Colab disconnect.**"),
        "install": ("Install", f"Plain `laya` at the pinned commit, DuckDB pinned to {cfg['data']['duckdb_version']} "
                    "(the frozen splits depend on its hash()), this project. Fails if pip changed Colab's torch, "
                    "transformers, protobuf or numpy. 1-3 min."),
        "token": ("Hugging Face token", "Masked input box at the top of VS Code; never printed. Unattended: "
                  "`python tools/build_p4_notebook.py --with-token` and open the gitignored "
                  "`notebooks/phase4_baselines.local.ipynb`."),
        "env": ("Environment check", f"GPU, driver, VRAM, disk, versions, Laya commit -> `runs/p4/env.json`. Warns "
                f"when the GPU is not a {card}. ~30 s."),
        "data": ("Full data (frozen)", "The E1 build, identical: FULL extraction over `hf://`, trap candidates held "
                 "out of every split, JSONL, then `--verify-frozen` (the fingerprint must equal "
                 "`data.frozen_manifest`, else the step fails). 3-20 min."),
        "variants": ("Data variants", f"From the frozen `data/`: {', '.join(f'`{d}/`' for d in variant_dirs(cfg))} "
                     "- the 7-class labels (B1/B3 c7) and the trap candidates as eval rows for c10 and c7. No "
                     "learning-curve subsets. A partial directory from an interrupted run is deleted and rebuilt. "
                     f"{VARIANTS_MIN} min."),
    }
    notes.update({arm: (f"{arm}: {arm_summary(cfg, arm)}", _arm_note(cfg, arm)) for arm in arm_ids(cfg)})
    notes["report"] = ("Report and the Phase 4 exit check", f"`matrix report` over `runs/p4`: every group (mean ± "
                       "range over seeds) with macro-F1 on val, test_id and each OOD pool, ID→OOD gaps, ECE pre→post, "
                       "trap-candidate accuracy (unannotated candidates), stripped false-confident rate, order "
                       f"invariance, and the Phase 4 exit check -> `results/{REPORT_MD}` (+ `.json`, `runs.csv`); "
                       "the step prints the Phase 4 verdict in one line. The report's Phase 3 exit check reads NOT RUN "
                       "here (the Laya runs are in the Phase 3 archive), and decision criterion 1 needs them too: "
                       "merge the archives locally (runbook). Seconds.")
    notes["archive"] = ("Archive", "---\nZip logs, JSON, per-row predictions, `results/` and the FSQ NOTICE (no "
                        "weights, no FSQ rows) for **Download...**. Seconds.")
    return {k: (f"Step {s[k]} - {title}", note) for k, (title, note) in notes.items()}


def finish_md(cfg: dict) -> str:
    return f"""### Finish
- The Phase 4 exit check is at the end of Step {step_numbers(cfg)['report']}; every run's metrics are in \
`results/runs.csv`.
- Download `phase4_artifacts.zip`: Colab view in the activity bar > Contents > right-click > **Download...**
- Locally, unzip it next to the Phase 3 archive and run the merged report for decision criterion 1 (early read):
  `python -m laya_poc.matrix report --runs-root runs/p3_colab/runs/p3 --runs-root runs/p4_colab/runs/p4 --archived` (runbook).
- Next, Phase 5: the full metric suite (bootstrap CIs, abstention, stability) from every run's saved predictions,
  trap accuracy once annotation lands, ONNX and the CPU benchmark, and the decision rule.
- Run **Colab: Remove Server** when you are done. It stops the compute-unit spend and deletes `/content`.
"""
