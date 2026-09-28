"""Generate notebooks/phase7_gpu_bench.ipynb: the Phase 7 GPU inference benchmark (batch-1 latency and batched
throughput of Laya and the small encoders) on a Colab GPU runtime (T4 by default; G4 works too).

Reuses the Phase 1/E1/Phase 3 infrastructure: the checksummed code bundle with the frozen manifest, the setup,
install, token and data cells and their helpers, the token guards and the --with-token variant
(tools/build_notebook.py, build_e1_notebook.py, build_p3_notebook.py and their parts). The Phase 7 cells live in
p7_cells.py, their commands in p7_commands.py, the extra Step 1 helpers in p7_helpers.py, the markdown in p7_md.py;
the assembly here.

    python tools/build_p7_gpu_notebook.py [--out notebooks/phase7_gpu_bench.ipynb] [--with-token]
"""
from __future__ import annotations

import argparse
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
import build_p3_notebook as bp3  # noqa: E402
from notebook_cells import Bundle  # noqa: E402
from notebook_commands import Target  # noqa: E402
from p7_cells import code_sources  # noqa: E402
from p7_commands import P7Target  # noqa: E402
from p7_md import RUNTIME, finish_md, intro_md, step_notes  # noqa: E402,F401  (RUNTIME re-exported for the tests)

from laya_poc.config import load_config  # noqa: E402

DEFAULT_OUT = ROOT / "notebooks" / "phase7_gpu_bench.ipynb"
LOCAL_OUT = ROOT / "notebooks" / f"phase7_gpu_bench{be.LOCAL_SUFFIX}"  # gitignored (notebooks/*.local.ipynb)
TIMED = ("env", "data", "bench")  # cells that print their wall-clock duration at the end


def check_token_out(root: Path, out: Path) -> None:
    """A notebook holding the token may only be written where git cannot track it (the E1 rule)."""
    try:
        be.check_token_out(root, out)
    except ValueError:
        raise ValueError(f"refusing to write a token into {Path(out).resolve()}: use a gitignored notebooks/"
                         f"*{be.LOCAL_SUFFIX} (default notebooks/{LOCAL_OUT.name})") from None


def colab_target(cfg: dict) -> P7Target:
    """The Colab GPU runtime: a GPU required, any card (the benchmark records which one ran)."""
    return P7Target()


def build_cells(cfg: dict, target: Target, bundle: Bundle) -> list[bn.Cell]:
    """All notebook cells in order; code cells carry their step key."""
    code, notes = code_sources(cfg, target, bundle), step_notes(cfg)
    cells = [bn.Cell("intro", "markdown", intro_md(cfg, bundle.sha256, len(bundle.members)))]
    for key, (title, note) in notes.items():
        body = bp3._timed(title.split(":")[0], code[key]) if key in TIMED else code[key]
        cells.append(bn.Cell(f"md_{key}", "markdown", f"### {title}\n{note}"))
        cells.append(bn.Cell(key, "code", f"# {title}\n{body.rstrip()}\n"))
    return [*cells, bn.Cell("finish", "markdown", finish_md(cfg))]


def build(root: Path = ROOT, out: Path = DEFAULT_OUT, token: str | None = None) -> tuple[Path, str]:
    """Render the Phase 7 GPU notebook to `out`; returns (path, bundle sha256). A token only goes to a gitignored
    path, and the rendered text may hold no token-like string but that one."""
    import nbformat

    root, out = Path(root), Path(out)
    if token is not None:
        check_token_out(root, out)
    cfg = load_config(root / "config.yaml")
    bundle = be.make_bundle(root, cfg)
    target = colab_target(cfg)
    cells = build_cells(cfg, target, bundle)
    text = nbformat.writes(bp3.to_notebook(bn._with_token(cells, token) if token else cells, target.card))
    if bn.TOKEN_RE.findall(text) != ([token] if token else []):
        raise ValueError("the rendered notebook contains an unexpected Hugging Face token-like string")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
    return out, bundle.sha256


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the Colab Phase 7 GPU inference benchmark notebook with the "
                                             "code bundle.")
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
