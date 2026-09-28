"""Run the Phase 5 CPU notebook's cells locally with tiny stand-in models: a plumbing check, not a benchmark.

It executes the SAME cells tools/build_p5_bench_notebook.py renders for Colab (Step 1, then every step but the pip
install and the token prompt) in one namespace, as a kernel would: the env check and the CPU facts, the FULL data
build with --verify-frozen on the synthetic pool of the E1/Phase 3/Phase 4 dry runs (the tiny Laya checkpoint lends
its tokenizer), the benchmark with QUICK = True (a few rows per setting) and every model mapped to a tiny stand-in
through `bench_cpu --ckpt MODEL=/abs/dir` (the tiny random Laya checkpoint for laya and laya_ml, the Phase 4 dry
run's tiny random ModernBERT for both small encoders), threads 1 and 2, and the archive. Exit code 1 unless the
benchmark JSON has a row for every configured (model, backend) at 1 thread, an ONNX export without error for each
Laya model, and the archive holds the JSONs but no ONNX graph.

    python tools/p5_dry_run.py [--work DIR] [--tiny-ckpt DIR] [--tiny-encoder DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Sequence

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
for _p in (ROOT / "src", ROOT / "tests", TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_e1_notebook as be  # noqa: E402
import build_p5_bench_notebook as bp  # noqa: E402
import dry_run_e1_local as d2  # noqa: E402
import dry_run_local as d1  # noqa: E402
from p5_cells import ARCHIVE_NAME  # noqa: E402
from p5_commands import QUICK_OUT, P5Target, bench_models  # noqa: E402

from laya_poc.config import load_config, with_overrides  # noqa: E402

DRY_OVERRIDES = {**{k: v for k, v in d2.DRY_OVERRIDES.items() if k.startswith("data.")},
                 "phase5.bench.threads": [1, 2]}
SKIP_CELLS = ("setup", "install", "token")  # Step 1 runs first; pip and the token prompt are Colab-only
QUICK_ON = ("QUICK = False", "QUICK = True")


def dry_config(cfg: dict) -> dict:
    """The project config with the E1 dry run's FULL-mode data sizes and a 1-2 thread sweep."""
    return with_overrides(cfg, DRY_OVERRIDES)


def local_target(work: Path, tiny: Path, pool: Path, encoder: Path, cfg: dict) -> P5Target:
    """CPU, the synthetic pool, the tiny Laya tokenizer for the data, and a stand-in for every benchmarked model."""
    laya_models = set(cfg["model"])
    stand_ins = tuple(f"{m}={tiny if m in laya_models else encoder}" for m in bench_models(cfg))
    return P5Target(work=str(work), init=str(tiny), pool=str(pool), data_models=("laya", "laya_ml"),
                    bench_ckpt=stand_ins)


def run_notebook_cells(dcfg: dict, target: P5Target) -> dict:
    """Execute Step 1 and every later cell but pip/token in one namespace, in notebook order (QUICK = True)."""
    import yaml

    cells = [c for c in bp.build_cells(dcfg, target, be.make_bundle(ROOT, dcfg)) if c.kind == "code"]
    ns: dict = {"__name__": "__main__"}
    work = Path(target.work)
    d1._exec_cell(cells[0], ns)
    # The bundle carries the Colab config; the CLIs read WORK/config.yaml (LAYA_POC_ROOT): swap in the tiny one.
    (work / "config.yaml").write_text(yaml.safe_dump(dcfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    # Stands in for Step 2's editable install: the CLIs import the unpacked bundle, not the repo checkout.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(work / "src"), os.environ.get("PYTHONPATH")]))
    for cell in cells:
        if cell.key == "bench":
            cell = cell._replace(source=cell.source.replace(*QUICK_ON))
        if cell.key not in SKIP_CELLS:
            d1._exec_cell(cell, ns)
    return ns


def dry_run(root: Path, tiny: Path | None = None, encoder: Path | None = None) -> Path:
    """Build the inputs and run the cells; returns the notebook's WORK directory."""
    import dry_run_p4_local as d4

    cfg = load_config(ROOT / "config.yaml")
    dcfg = dry_config(cfg)
    root.mkdir(parents=True, exist_ok=True)
    tiny = tiny.resolve() if tiny else d2.build_tiny_ckpt(root / "tiny_ckpt", cfg)
    encoder = encoder.resolve() if encoder else d4.build_tiny_encoder(root / "tiny_encoder", cfg)
    pool = d2.write_synthetic_full_pool(root / "pool.parquet", dcfg)
    with d1.preserved_environ():
        run_notebook_cells(dcfg, local_target(root / "laya_poc", tiny, pool, encoder, dcfg))
    return root / "laya_poc"


def bench_failures(dcfg: dict, bench: dict) -> list[str]:
    """What the quick benchmark JSON lacks: a 1-thread row per configured (model, backend), a clean ONNX export."""
    rows = {(r.get("model"), r.get("backend")) for r in bench.get("results") or [] if r.get("threads") == 1}
    backends = bench.get("backends") or {}
    problems = [f"no 1-thread row for {m} {b}" for m in bench_models(dcfg) for b in backends.get(m, ["?"])
                if (m, b) not in rows]
    problems += [f"setting failed: {e.get('model')} {e.get('backend')} {e.get('threads')}t: {e.get('error')}"
                 for e in bench.get("errors") or []]
    onnx = bench.get("onnx") or {}
    problems += [f"ONNX export of {m}: {(onnx.get(m) or {}).get('error') or 'missing'}" for m in dcfg["model"]
                 if m in bench_models(dcfg) and (not onnx.get(m) or onnx[m].get("error"))]
    return problems


def plumbing_failures(dcfg: dict, work: Path) -> list[str]:
    runs, archive = Path(work) / "runs" / "p5", Path(work) / ARCHIVE_NAME
    missing = [f"no {p}" for p in (runs / "cpu.json", runs / QUICK_OUT, archive) if not p.is_file()]
    if missing:
        return missing
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    problems = [f"archive lacks {n}" for n in ("runs/p5/cpu.json", "runs/p5/env.json", f"runs/p5/{QUICK_OUT}")
                if n not in names]
    problems += [f"archive holds {n}" for n in sorted(names) if n.endswith((".onnx", ".data", ".safetensors"))]
    return problems + bench_failures(dcfg, json.loads((runs / QUICK_OUT).read_text(encoding="utf-8")))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Colab Phase 5 CPU notebook's cells locally (plumbing check).")
    ap.add_argument("--work", type=Path, help="scratch directory (default: a new temp dir, kept for inspection)")
    ap.add_argument("--tiny-ckpt", type=Path, help="existing tiny Laya checkpoint dir (default: build one)")
    ap.add_argument("--tiny-encoder", type=Path, help="existing tiny ModernBERT dir (default: build one)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # child output contains non-ASCII text
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    root = (args.work or Path(tempfile.mkdtemp(prefix="laya_dry_p5_"))).resolve()
    start = time.time()
    try:
        work = dry_run(root, args.tiny_ckpt, args.tiny_encoder)
    except Exception as exc:
        print(f"error: Phase 5 dry run failed: {exc}", file=sys.stderr)
        return 1
    failed = plumbing_failures(dry_config(load_config(ROOT / "config.yaml")), work)
    print(f"\nPhase 5 CPU notebook plumbing: {'FAILED' if failed else 'OK'} in {time.time() - start:.0f}s ({work})")
    for line in failed:
        print(f"error: {line}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
