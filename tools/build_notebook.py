"""Generate notebooks/smoke_test.ipynb: the Phase 1 smoke test on a Colab T4 via the VS Code Colab extension.

Every cell is rendered from config.yaml and a `Target`, so tools/dry_run_local.py can execute the SAME
cells on CPU. The project code travels inside the notebook as a checksummed bundle (laya_poc.bundle): the
runtime needs no git remote, Drive or upload. Pieces: notebook_helpers.py (functions embedded verbatim in
Step 1), notebook_commands.py (the CLI commands), notebook_cells.py (code cell sources); the markdown and
the assembly live here.

    python tools/build_notebook.py [--out notebooks/smoke_test.ipynb]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, NamedTuple, Sequence

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
for _p in (ROOT / "src", TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from notebook_cells import _TOKEN, Bundle, code_sources, resume_step  # noqa: E402
from notebook_commands import Expr, Target, commands, render_cmd  # noqa: E402,F401  (re-exported)
from notebook_helpers import HELPERS  # noqa: E402,F401  (re-exported)

from laya_poc import bundle as B  # noqa: E402
from laya_poc.config import load_config  # noqa: E402

DEFAULT_OUT = ROOT / "notebooks" / "smoke_test.ipynb"
LOCAL_OUT = ROOT / "notebooks" / "smoke_test.local.ipynb"  # gitignored: may carry the HF token
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
PRESET_SLOT = re.compile(r'^PRESET_TOKEN = ""', re.M)


class Cell(NamedTuple):
    key: str
    kind: str  # "markdown" | "code"
    source: str


def _steps(cfg: dict) -> dict[str, tuple[str, str]]:
    sm = cfg["smoke"]
    return {
        "setup": ("Step 1 - Setup", "Non-interactive environment, work dir, code bundle, helpers. Seconds. "
                  "**Re-run this cell first after any kernel restart.**"),
        "install": ("Step 2 - Install", "Plain `laya` at the pinned commit, no extras (skipped when the PEP 610 record "
                    "already matches), DuckDB, this project (editable, no deps). Fails if pip changed Colab's torch, "
                    "transformers, protobuf or numpy. 1-3 min the first time."),
        "token": ("Step 3 - Hugging Face token", "A masked input box opens at the top of VS Code. Paste a token that can "
                  "read the gated FSQ dataset. It is validated, saved in the runtime's token file and never printed. "
                  "To replace a saved token, set `ASK_AGAIN = True` in the cell. **Unattended run:** put "
                  "`HF_TOKEN=hf_...` in the repo's `.env`, run `python tools/build_notebook.py --with-token` and open "
                  "the gitignored `notebooks/smoke_test.local.ipynb` instead: no prompt."),
        "env": ("Step 4 - Environment check", "GPU, compute capability, `sm_75` in the CUDA arch list, driver, VRAM, "
                "free disk, versions and Laya commit -> `runs/smoke/env.json`. On failure it prints the errors from "
                "`env.json` (each names its fix; the runbook has one row per error). Warns if not a T4. ~30 s."),
        "data": ("Step 5 - Smoke data", "DuckDB over `hf://` (one best places file per ID country) -> splits, JSONL, "
                 "leakage asserts -> `data/`. 2-5 min."),
        "zeroshot": ("Step 6 - Zero-shot", f"Both shipped checkpoints on the first {sm['zero_shot_n']} labelled val "
                     "rows (shipped temperatures). 3-6 min including the checkpoint downloads."),
        "parity": ("Step 7 - Parity", f"CPU fp32 reference vs GPU fp16 on {sm['parity_n']} val rows, plus padded vs "
                   "unpadded batches, for each checkpoint. Prints PASS/FAIL. About 25-35 min, the longest step: the "
                   "fp32 reference runs on Colab's CPU while the T4 waits."),
        "control": ("Step 8 - Control run", f"{sm['micro_steps']} micro-steps with an initial eval (val CE at opt "
                    "step 0), a final eval and a final checkpoint; no resumable checkpoints (nothing reads them, and "
                    "the crash run measures the save time). Steps 8-10 use the T4 batch profile (`--card T4`: "
                    "micro-batch 8 x accumulation 4) on any GPU, so the plan holds. 5-10 min."),
        "crash": ("Step 9 - Crash run", "Same settings with a resumable checkpoint every "
                  f"{sm['ckpt_every_micro_steps']} micro-steps, hard-killed (SIGKILL) right after micro-step "
                  f"{sm['kill_at_micro_step']}; exit code "
                  "-9/137 goes to `crash_exit.json`, and `resumed/log.jsonl` must end with that injected crash (any "
                  "other kill, e.g. the host OOM killer, stops here). Starts the drill afresh. 3-5 min."),
        "resume": ("Step 10 - Resume", "The crash run's settings without the crash flag, and with no checkpoints "
                   f"(nothing reads them), must print `resumed from ...` (micro-step {resume_step(cfg)}) and finish. "
                   "Runs once per crash: if it was interrupted, re-run Step 9 first. 3-5 min."),
        "export": ("Step 11 - Export/calibration round trip", "Fit T on raw val logits, write it into the saved "
                   "checkpoint, reload with Laya, compare probabilities. 1-2 min."),
        "report": ("Step 12 - Report", "Exit criteria and the design doc §7.13 template -> `runs/smoke/smoke_report.md`."),
        "archive": ("Step 13a - Archive (optional)", "---\nOptional steps; each is switched by a flag in its cell. "
                    "Zip the small artefacts and the FSQ NOTICE (`data/NOTICE_FSQ.txt`) for **Download...** in the "
                    "Colab view; optional Drive copy."),
        "cleanup": ("Step 13b - Free disk (optional)", "Delete the resumable training checkpoints."),
        "kill": ("Step 13c - Kernel-kill drill (optional, off)", "Hard-kills the kernel while a child holds the GPU. "
                 "Afterwards re-run Step 1, then Step 13d."),
        "verify_kill": ("Step 13d - After the kill drill", "Checks that `/content` and site-packages survived and "
                        "that no orphan process is left on the GPU (kills only the drill's own child)."),
    }


def _intro_md(cfg: dict, bundle: Bundle) -> str:
    sm, ex = cfg["smoke"], cfg["smoke"]["exit"]
    boundary = resume_step(cfg)
    return f"""# Laya PoC - Phase 1 smoke test (Colab T4 from VS Code)

Proves on a free Colab T4 that the single-GPU fine-tuning port works end to end before any paid run (design doc
§6.2, Phase 1): pinned install and GPU check, smoke data from FSQ, zero-shot on both shipped checkpoints, CPU-vs-GPU
parity, a {sm['micro_steps']}-micro-step control run, a hard crash at micro-step {sm['kill_at_micro_step']} with a \
resume from the micro-step-{boundary} checkpoint, an export/calibration round trip and a filled-in report.

**How to connect (VS Code + Google Colab extension)**
1. **Select Kernel → Colab → New Colab Server → GPU → T4** (shape *Standard*; runtime version *Latest*, or *2026.07*
   if offered), accept the alias, sign in to Google when asked, then pick the Python 3 kernel.
2. Do **not** use *Auto Connect*: it provisions a CPU server. If no T4 is free the extension may silently assign another
   GPU (watch for a toast). The runs still use the T4 batch profile, so the crash/resume plan holds, but VRAM and speed
   then do not describe a T4: prefer **Colab: Remove Server** and retry later. Step 4 records the card.
3. **Run All**, or run the steps in order. Keep VS Code open and the laptop awake: a running cell keeps the server alive.

**Hugging Face token:** Step 3 asks for it in a masked input box at the top of VS Code. It is never printed and never
stored in this notebook, only in the runtime's own token file, which is deleted with the server.

**After any kernel restart** (crash, manual restart, the optional kill drill): re-run **Step 1 (Setup)**, then continue
where you stopped. Finished steps are skipped because their outputs are on disk; set `REDO = True` in Step 1 to redo them.
Steps 9-10 are one drill: if Step 10 was interrupted, re-run Step 9, then Step 10.
A failing step names its log file; `docs/smoke_test_runbook.md` maps errors to fixes.

**Expected runtime:** about 60-90 min. Step 7's CPU fp32 parity reference is the longest step (about 25-35 min), then
the three training runs. Everything lives under `/content/laya_poc`; full logs are in `runs/smoke/logs/`.

**Exit criteria** (`config.yaml` `smoke.exit`, evaluated in Step 12):
- parity, each checkpoint: max |Δp| ≤ {ex['parity_max_dp']}, argmax agreement ≥ {ex['parity_min_agree']}, no NaN
- resume: the crash run is killed by the injected crash and the resumed run restarts from micro-step {boundary}. Its
  mean `loss_ce` over the next {sm['resume_compare_steps']} micro-steps is within \
{ex['resume_max_rel_dev']:.0%} of the control run's, and the learning rates
  and GradScaler scale match the control run at every later optimiser step
- training: the control run's held-out val CE (eval mode) falls by ≥ {ex['min_loss_drop']:.0%} from the initial \
eval (opt step 0) to the
  final eval; no NaN/inf; GradScaler scale ≥ 1
- peak VRAM (reserved) ≤ {ex['max_vram_gb']} GB (10^9 bytes)

Code bundle: {len(bundle.members)} files, sha256 `{bundle.sha256}` (verified when Step 1 unpacks it).
"""


_FINISH_MD = """### Finish
- The verdict is at the end of Step 12; details in `runs/smoke/smoke_report.md` and `smoke_report.json`.
- Download `smoke_artifacts.zip` (Step 13a): Colab view in the activity bar > Contents > right-click > **Download...**
- Run **Colab: Remove Server** when you are done. It stops quota use and deletes `/content`, checkpoints included.
"""


def build_cells(cfg: dict, target: Target, bundle: Bundle) -> list[Cell]:
    """All notebook cells in order. Code cells carry the step key used by tools/dry_run_local.py."""
    code = code_sources(cfg, target, bundle)
    cells = [Cell("intro", "markdown", _intro_md(cfg, bundle))]
    for key, (title, note) in _steps(cfg).items():
        cells.append(Cell(f"md_{key}", "markdown", f"### {title}\n{note}"))
        cells.append(Cell(key, "code", f"# {title}\n{code[key].rstrip()}\n"))
    return [*cells, Cell("finish", "markdown", _FINISH_MD)]


def make_bundle(root: Path) -> Bundle:
    """The code bundle for the notebook; refuses to embed anything that looks like an HF token."""
    b64, sha, members = B.bundle(Path(root))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as zf:
        for name in zf.namelist():
            if TOKEN_RE.search(zf.read(name).decode("utf-8", "replace")):
                raise ValueError(f"{name} contains what looks like a Hugging Face token; refusing to embed it")
    return Bundle(b64, sha, members)


def to_notebook(cells: list[Cell]) -> Any:
    import nbformat

    nb = nbformat.v4.new_notebook(metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}})
    for i, c in enumerate(cells):
        new = nbformat.v4.new_markdown_cell if c.kind == "markdown" else nbformat.v4.new_code_cell
        nb.cells.append(new(c.source, id=f"c{i:02d}-{c.key.replace('_', '-')}"))  # stable ids: byte-identical rebuilds
    nbformat.validate(nb)
    return nb


def read_local_token(root: Path = ROOT) -> str:
    """HF token for the unattended variant: $HF_TOKEN, else HF_TOKEN=... in <root>/.env. Never echoed."""
    token = os.environ.get("HF_TOKEN", "").strip()
    env_file = Path(root) / ".env"
    if not token and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "HF_TOKEN":
                token = value.strip().strip("'\"")
    if not token:
        raise ValueError(f"no HF token: add a line HF_TOKEN=hf_... to {env_file} (gitignored) or set $HF_TOKEN")
    if not TOKEN_RE.fullmatch(token):
        raise ValueError("HF_TOKEN does not look like a Hugging Face token (hf_ + letters/digits); value not shown")
    return token


def _fill_preset(source: str, token: str | None) -> str:
    return PRESET_SLOT.sub(lambda m: f"PRESET_TOKEN = {json.dumps(token or '')}", source, count=1)


def render_token_cell(token: str | None) -> str:
    """Step 3's code with PRESET_TOKEN filled in (or left empty)."""
    return _fill_preset(_TOKEN, token)


def _with_token(cells: list[Cell], token: str) -> list[Cell]:
    return [c._replace(source=_fill_preset(c.source, token)) if c.key == "token" else c for c in cells]


def build(root: Path = ROOT, out: Path = DEFAULT_OUT, token: str | None = None) -> tuple[Path, str]:
    """Render the Colab notebook to `out`; returns (path, bundle sha256).

    With `token`, Step 3 uses it instead of prompting. That variant must go to a gitignored path, never
    to the committed notebook.
    """
    import nbformat

    root, out = Path(root), Path(out)
    if token is not None and out.resolve() == DEFAULT_OUT.resolve():
        raise ValueError(f"refusing to write a token into {DEFAULT_OUT.name}; use the gitignored {LOCAL_OUT.name}")
    bundle = make_bundle(root)
    cells = build_cells(load_config(root / "config.yaml"), Target(), bundle)
    text = nbformat.writes(to_notebook(_with_token(cells, token) if token else cells))
    found = TOKEN_RE.findall(text)
    if found != ([token] if token else []):
        raise ValueError("the rendered notebook contains an unexpected Hugging Face token-like string")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
    return out, bundle.sha256


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the Colab smoke-test notebook with the embedded code bundle.")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output .ipynb (default: {DEFAULT_OUT}, or {LOCAL_OUT} with --with-token)")
    ap.add_argument("--with-token", action="store_true",
                    help="unattended variant: embed HF_TOKEN from .env / $HF_TOKEN into a gitignored notebook")
    args = ap.parse_args(argv)
    try:
        token = read_local_token(ROOT) if args.with_token else None
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
