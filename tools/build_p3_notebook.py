"""Generate notebooks/phase3_matrix.ipynb: Phase 3 of the design doc (§6.2, matrix §5.14) on a Colab G4.

Reuses the Phase 1/E1 infrastructure: the checksummed code bundle with the frozen manifest, the Step 1-5 cells and
their helpers, the token guards and the --with-token variant (tools/build_notebook.py, build_e1_notebook.py and
their parts). The Phase 3 cells live in p3_cells.py, their commands in p3_commands.py, the extra Step 1 helpers in
p3_helpers.py; the markdown and the assembly here. tools/dry_run_p3_local.py runs the same cells on CPU.

    python tools/build_p3_notebook.py [--out notebooks/phase3_matrix.ipynb] [--with-token]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
for _p in (ROOT / "src", TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_e1_notebook as be  # noqa: E402
import build_notebook as bn  # noqa: E402
from notebook_cells import Bundle  # noqa: E402
from notebook_commands import Target  # noqa: E402
from p3_cells import code_sources, matrix_arms  # noqa: E402
from p3_md import RUNTIME, finish_md, intro_md, step_notes  # noqa: E402,F401  (RUNTIME re-exported for the tests)

from laya_poc.config import load_config  # noqa: E402

DEFAULT_OUT = ROOT / "notebooks" / "phase3_matrix.ipynb"
LOCAL_OUT = ROOT / "notebooks" / f"phase3_matrix{be.LOCAL_SUFFIX}"  # gitignored (notebooks/*.local.ipynb): the token
# Cells that print their wall-clock duration at the end (Step 1 defines elapsed(); Steps 2-3 are quick).
TIMED = ("env", "data", "variants", "parity_laya", "parity_laya_ml", "plan", "report")


def check_token_out(root: Path, out: Path) -> None:
    """A notebook holding the token may only be written where git cannot track it (the E1 rule)."""
    try:
        be.check_token_out(root, out)
    except ValueError:
        raise ValueError(f"refusing to write a token into {Path(out).resolve()}: use a gitignored notebooks/"
                         f"*{be.LOCAL_SUFFIX} (default notebooks/{LOCAL_OUT.name})") from None


def colab_target(cfg: dict) -> Target:
    """The Colab GPU runtime; CARD defaults to config phase3.card (G4)."""
    return Target(card=cfg["phase3"]["card"])


def _timed(title: str, code: str) -> str:
    """The cell's code between a start time and a closing '<title>: <duration>' line."""
    return f"T0 = time.time()\n{code.rstrip()}\nprint({json.dumps(title + ':')}, elapsed(T0))\n"


def build_cells(cfg: dict, target: Target, bundle: Bundle) -> list[bn.Cell]:
    """All notebook cells in order; code cells carry the step key tools/dry_run_p3_local.py uses."""
    code, notes, timed = code_sources(cfg, target, bundle), step_notes(cfg), {*TIMED, *matrix_arms(cfg)}
    cells = [bn.Cell("intro", "markdown", intro_md(cfg, bundle.sha256, len(bundle.members)))]
    for key, (title, note) in notes.items():
        body = _timed(title.split(":")[0], code[key]) if key in timed else code[key]
        cells.append(bn.Cell(f"md_{key}", "markdown", f"### {title}\n{note}"))
        cells.append(bn.Cell(key, "code", f"# {title}\n{body.rstrip()}\n"))
    return [*cells, bn.Cell("finish", "markdown", finish_md(cfg))]


def to_notebook(cells: list[bn.Cell], card: str):
    """The Phase 1 notebook format, with the Colab accelerator hint set to the Phase 3 card."""
    nb = bn.to_notebook(cells)
    nb.metadata["colab"]["gpuType"] = card
    return nb


def build(root: Path = ROOT, out: Path = DEFAULT_OUT, token: str | None = None) -> tuple[Path, str]:
    """Render the Phase 3 notebook to `out`; returns (path, bundle sha256). A token only goes to a gitignored path."""
    import nbformat

    root, out = Path(root), Path(out)
    if token is not None:
        check_token_out(root, out)
    cfg = load_config(root / "config.yaml")
    bundle = be.make_bundle(root, cfg)
    cells = build_cells(cfg, colab_target(cfg), bundle)
    text = nbformat.writes(to_notebook(bn._with_token(cells, token) if token else cells, cfg["phase3"]["card"]))
    if bn.TOKEN_RE.findall(text) != ([token] if token else []):
        raise ValueError("the rendered notebook contains an unexpected Hugging Face token-like string")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
    return out, bundle.sha256


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the Colab Phase 3 (seed matrix) notebook with the code bundle.")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output .ipynb (default: {DEFAULT_OUT}, or {LOCAL_OUT} with --with-token, which only "
                         f"accepts a gitignored notebooks/*{be.LOCAL_SUFFIX})")
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
