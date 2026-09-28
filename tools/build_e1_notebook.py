"""Generate notebooks/e1_first_run.ipynb: Phase 2 of the design doc (§6.2), E1 on a Colab T4 plus the gate.

Reuses the Phase 1 infrastructure: the checksummed code bundle, Step 1-3 cells and their helpers, the token
guards and the --with-token variant (tools/build_notebook.py and its parts). The E1 cells live in e1_cells.py,
their commands in e1_commands.py, the extra Step 1 helpers in e1_helpers.py; the markdown and the assembly here.
tools/dry_run_e1_local.py runs the same cells on CPU.

    python tools/build_e1_notebook.py [--out notebooks/e1_first_run.ipynb] [--with-token]
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import sys
import zipfile
from pathlib import Path
from typing import Sequence

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
for _p in (ROOT / "src", TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_notebook as bn  # noqa: E402
from e1_cells import code_sources  # noqa: E402
from notebook_cells import Bundle  # noqa: E402
from notebook_commands import Target  # noqa: E402

from laya_poc import bundle as B  # noqa: E402
from laya_poc.config import accumulation, load_config  # noqa: E402
from laya_poc.smoke_report import expected_resume_step  # noqa: E402
from laya_poc.smoke_render import e1_micro_steps  # noqa: E402

DEFAULT_OUT = ROOT / "notebooks" / "e1_first_run.ipynb"
LOCAL_SUFFIX = ".local.ipynb"
LOCAL_OUT = ROOT / "notebooks" / f"e1_first_run{LOCAL_SUFFIX}"  # gitignored (notebooks/*.local.ipynb): holds the token
RUNTIME = "2.5-3.5 h"


def check_token_out(root: Path, out: Path) -> None:
    """A notebook holding the token may only be written where git cannot track it: the .gitignore pattern
    notebooks/*.local.ipynb inside the repo, or a *.local.ipynb outside it. Raises ValueError otherwise."""
    root, out = Path(root).resolve(), Path(out).resolve()
    in_repo = out.is_relative_to(root)
    if out.name.endswith(LOCAL_SUFFIX) and (not in_repo or out.parent == root / "notebooks"):
        return
    raise ValueError(f"refusing to write a token into {out}: use a gitignored notebooks/*{LOCAL_SUFFIX} "
                     f"(default notebooks/{LOCAL_OUT.name})")


def colab_target(cfg: dict) -> Target:
    """The Colab GPU runtime; E1 trains with the e1.card batch profile on whatever GPU the extension assigns."""
    return Target(card=cfg["e1"]["card"])


def plan(cfg: dict) -> dict[str, int]:
    """E1 on its card profile: micro/opt steps, periodic evals, resumable checkpoints (the crash run writes the first)."""
    e1, tc = cfg["e1"], cfg["train"]
    mb, acc = accumulation(cfg, e1["card"])
    per_epoch = e1_micro_steps(cfg["data"]["train_size"], cfg["augment"]["stripped_rate"], 1, mb)
    micro, opt = per_epoch * tc["epochs"], math.ceil(per_epoch / acc) * tc["epochs"]
    every = e1["ckpt_every_micro_steps"]
    return {"mb": mb, "acc": acc, "per_epoch": per_epoch, "micro": micro, "opt": opt,
            "evals": opt // tc["eval_every_opt_steps"], "ckpts": micro // every,
            "resume": expected_resume_step(e1["crash_at_micro_step"], every)}


def _steps(cfg: dict) -> dict[str, tuple[str, str]]:
    e1, p, b = cfg["e1"], plan(cfg), cfg["bench"]
    return {
        "setup": ("Step 1 - Setup", "Non-interactive environment, work dir, code bundle, helpers. Seconds. "
                  "**Re-run this cell first after any kernel restart or Colab disconnect.**"),
        "install": ("Step 2 - Install", f"Plain `laya` at the pinned commit, DuckDB pinned to "
                    f"{cfg['data']['duckdb_version']} (`data.duckdb_version`: the frozen splits depend on its hash()), "
                    "this project. Fails if pip changed Colab's torch, transformers, protobuf or numpy. 1-3 min."),
        "token": ("Step 3 - Hugging Face token", "Masked input box at the top of VS Code; never printed. Unattended: "
                  "`python tools/build_e1_notebook.py --with-token` and open the gitignored "
                  "`notebooks/e1_first_run.local.ipynb`."),
        "env": ("Step 4 - Environment check", "GPU, `sm_75`, driver, VRAM, disk, versions, Laya commit -> "
                "`runs/e1/env.json`. ~30 s."),
        "data": ("Step 5 - Full data", "FULL build over `hf://` (two passes, all 15 countries), trap candidates held "
                 "out of every split, JSONL, leakage asserts, then `--verify-frozen`: the JSONL fingerprint must equal "
                 "the frozen manifest (`data.frozen_manifest`), else the step fails. 10-20 min."),
        "baselines": ("Step 6 - Baselines B1 + B3", "Majority / prior and char TF-IDF + logistic regression (C "
                      "tuned and T fitted on val), 10- and 7-class, on the CPU. 5-15 min."),
        "zeroshot": ("Step 7 - Zero-shot laya on val", "The shipped checkpoint on the full val split (the gate's "
                     "'beats zero-shot' reference). ~2 min. `ZERO_SHOT_ML = True` adds laya-multilingual."),
        "crash": ("Step 8 - E1 crash run", f"E1 from scratch: `{e1['model']}`, seed {e1['seed']}, "
                  f"{cfg['train']['epochs']} epochs x {p['per_epoch']} micro-steps (MB {p['mb']} x ACC {p['acc']}, "
                  f"`--card {e1['card']}`), a checkpoint every {e1['ckpt_every_micro_steps']} micro-steps, "
                  f"hard-killed right after micro-step {e1['crash_at_micro_step']} (exit -9/137, `crash_exit.json`). "
                  "Skipped once the crash has happened, so Run All never restarts E1. 15-20 min."),
        "resume": ("Step 9 - E1 resume (to the end)", f"Same settings without the crash flag: resumes from micro-step "
                   f"{p['resume']} and trains to the end or an early stop, evaluating val every "
                   f"{cfg['train']['eval_every_opt_steps']} opt steps ({p['evals']} evals), exporting `best/` and "
                   "`final/`. **After a disconnect: re-run Step 1, then this step**; it resumes from the latest "
                   "checkpoint. About 1.5-2 h."),
        "export": ("Step 10 - Temperature fit", "T fitted on raw val logits of `best/`, written into it, reloaded "
                   "and checked against `best/train_eval.json`. ~2 min."),
        "evaluate": ("Step 11 - Evaluate E1 on val", "`best/` with and without its temperature: macro-F1 (10- and "
                     "9-class), ECE, selective accuracy; per-row predictions. ~2 min."),
        "bench": ("Step 12 - CPU benchmark", f"{b['n_records']} test_id records on this host's CPU, threads "
                  f"{b['threads']} (a fresh process each): cold start, p50/p95 at batch 1, batched rec/s. Information "
                  "only (the decision rule's CPU budget is Phase 5, on 4 vCPUs). 10-20 min."),
        "gate": ("Step 13 - Gate", "Design doc §6.2 gate table -> `runs/e1/gate_report.md` (+ `.json`)."),
        "archive": ("Step 14a - Archive (optional)", "---\nZip logs, JSON, the gate report and the FSQ NOTICE (no "
                    "checkpoints, no FSQ rows) for **Download...**; optional Drive copy of the zip and of `best/`."),
        "cleanup": ("Step 14b - Free disk (optional)", "Delete the resumable checkpoints once E1 has finished."),
    }


def _intro_md(cfg: dict, bundle: Bundle) -> str:
    e1, g, p = cfg["e1"], cfg["gate"], plan(cfg)
    return f"""# Laya PoC - Phase 2: E1 first real run and the gate (Colab T4 from VS Code)

E1 (design doc §6.2, §5.14): `{e1['model']}`, seed {e1['seed']}, the full train split for {cfg['train']['epochs']} \
epochs on the free T4 ({p['micro']} micro-steps, {p['opt']} optimiser steps), resumable, with one forced crash and \
resume; then the temperature fit and the evaluation on val, a CPU benchmark, the B1/B3 baselines, zero-shot `laya` and \
the gate. Runbook: `docs/phase2_e1_runbook.md`.

**How to connect** (as for the smoke test): **Select Kernel → Colab → New Colab Server → GPU → T4**, then the Python 3
kernel. Do **not** use *Auto Connect* (it gives a CPU). Keep VS Code open and the laptop awake: a running cell keeps
the server alive. The Hugging Face token is asked in a masked input box (Step 3) and never stored in this notebook.

**After a kernel restart or a Colab disconnect:** re-run **Step 1**, then continue. Finished steps are skipped (their
outputs are on disk; `REDO = True` in Step 1 redoes them). During training: re-run Step 1, then **Step 9**, which resumes
E1 from its latest checkpoint (one every {e1['ckpt_every_micro_steps']} micro-steps, about 14 min apart). Step 8 skips
itself once the forced crash has happened, so **Run All** is safe. If the server itself was removed, `/content` is gone:
start again from Step 1.

**Expected runtime:** about {RUNTIME}: data 10-20 min, baselines 5-15, zero-shot ~2, E1 about 2-2.3 h (including
{p['evals']} evals and about {p['ckpts']} checkpoints of ~72 s each), temperature fit ~2, evaluation ~2, CPU benchmark
10-20. Colab's free tier allows at most 12 h per session.

**Gate** (`config.yaml` `gate`, design doc §6.2; points are macro-F1 x 100), evaluated in Step 13:
- end-to-end: train → temperature fit → val evaluation → CPU benchmark all complete; at least one resume (forced here)
- numerics: no NaN/inf loss, no non-finite gradient applied, GradScaler scale ≥ {g['min_scale']} (skipped fp16 steps
  are expected and only reported)
- beats trivial: E1 val macro-F1 ≥ zero-shot `laya` + {100 * g['min_over_zero_shot']:.0f} points and ≥ majority + \
{100 * g['min_over_majority']:.0f} points
- near TF-IDF+LR: E1 val macro-F1 ≥ TF-IDF+LR − {100 * g['max_below_tfidf']:.0f} points; more than that below means
  **stop and debug** before any spend (the report lists the checklist)

The headline is 10-class macro-F1, or 9-class when the ID pool has fewer than {cfg['labels']['event_min_id_pool']}
labelled Event places (`data_report.json` `headline_classes`).

Code bundle: {len(bundle.members)} files, sha256 `{bundle.sha256}` (verified when Step 1 unpacks it).
"""


_FINISH_MD = """### Finish
- The verdict is at the end of Step 13; details in `runs/e1/gate_report.md` and `gate_report.json`.
- Download `e1_artifacts.zip` (Step 14a): Colab view in the activity bar > Contents > right-click > **Download...**
- Pending human task (not part of the gate): trap annotation. `data/trap_candidates.csv` holds FSQ rows: two annotators
  mark `keep_a1` / `keep_a2`, the lead resolves disagreements, then `python -m laya_poc.traps merge` writes
  `data/trap.jsonl` (runbook). Download the CSV for the annotators only if the FSQ terms allow it for your team.
- Run **Colab: Remove Server** when you are done. It stops quota use and deletes `/content`, checkpoints included.
"""


def build_cells(cfg: dict, target: Target, bundle: Bundle) -> list[bn.Cell]:
    """All notebook cells in order; code cells carry the step key tools/dry_run_e1_local.py uses."""
    code = code_sources(cfg, target, bundle)
    cells = [bn.Cell("intro", "markdown", _intro_md(cfg, bundle))]
    for key, (title, note) in _steps(cfg).items():
        cells.append(bn.Cell(f"md_{key}", "markdown", f"### {title}\n{note}"))
        cells.append(bn.Cell(key, "code", f"# {title}\n{code[key].rstrip()}\n"))
    return [*cells, bn.Cell("finish", "markdown", _FINISH_MD)]


def bundle_patterns(cfg: dict, root: Path) -> tuple[str, ...]:
    """The Phase 1 bundle plus the frozen data manifest: build_data --verify-frozen reads it on Colab."""
    rel = cfg["data"].get("frozen_manifest")
    if not rel:
        return tuple(B.DEFAULT_INCLUDE)
    if Path(rel).is_absolute() or not (Path(root) / rel).is_file():
        raise FileNotFoundError(f"config data.frozen_manifest {rel!r} must be a file under {root} (repo-relative)")
    return (*B.DEFAULT_INCLUDE, Path(rel).as_posix())


def make_bundle(root: Path, cfg: dict) -> Bundle:
    """The code bundle; refuses to embed anything that looks like an HF token."""
    b64, sha, members = B.bundle(Path(root), bundle_patterns(cfg, Path(root)))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as zf:
        for name in zf.namelist():
            if bn.TOKEN_RE.search(zf.read(name).decode("utf-8", "replace")):
                raise ValueError(f"{name} contains what looks like a Hugging Face token; refusing to embed it")
    return Bundle(b64, sha, members)


def build(root: Path = ROOT, out: Path = DEFAULT_OUT, token: str | None = None) -> tuple[Path, str]:
    """Render the E1 notebook to `out`; returns (path, bundle sha256). A token only goes to a gitignored path."""
    import nbformat

    root, out = Path(root), Path(out)
    if token is not None:
        check_token_out(root, out)
    cfg = load_config(root / "config.yaml")
    bundle = make_bundle(root, cfg)
    cells = build_cells(cfg, colab_target(cfg), bundle)
    text = nbformat.writes(bn.to_notebook(bn._with_token(cells, token) if token else cells))
    if bn.TOKEN_RE.findall(text) != ([token] if token else []):
        raise ValueError("the rendered notebook contains an unexpected Hugging Face token-like string")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
    return out, bundle.sha256


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the Colab E1 notebook (Phase 2) with the embedded code bundle.")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output .ipynb (default: {DEFAULT_OUT}, or {LOCAL_OUT} with --with-token, which only "
                         f"accepts a gitignored notebooks/*{LOCAL_SUFFIX})")
    ap.add_argument("--with-token", action="store_true",
                    help="unattended variant: embed HF_TOKEN from .env / $HF_TOKEN into a gitignored notebook")
    args = ap.parse_args(argv)
    try:
        token = bn.read_local_token(ROOT) if args.with_token else None
        path, sha = build(ROOT, args.out or (LOCAL_OUT if token else DEFAULT_OUT), token=token)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path}\nbundle sha256 {sha}")
    if token:
        print("contains your HF token: keep it local (gitignored), never share or commit it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
