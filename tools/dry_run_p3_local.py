"""Run the Phase 3 notebook's cells locally on CPU: tiny checkpoint, synthetic FULL-mode pool, a tiny matrix.

It executes the SAME cells tools/build_p3_notebook.py renders for Colab (Step 1, then every step but the pip install
and the token prompt) in one namespace, as a kernel would: the FULL data build with --verify-frozen, the variants
(c7, the E6 subsets, the trap candidates), parity for both models, the matrix plan, B2 and every arm through
`matrix run`, the report, the archive and the (switched-off) Drive copy. Only the target differs: --init <tiny
ckpt> (init_scale 0.5) for every run, so the tiny checkpoint stands in for laya_ml too; --pool <synthetic FULL pool
with trap candidates>; --init-tokenizer with --models laya laya_ml; --device cpu; CARD "CPU"; and the tiny sizes
and matrix of DRY_OVERRIDES. Exit code 1 unless the report's Phase 3 exit check PASSES, every configured run has
the spec §1 run-directory layout, and results/runs.csv has one row per run with the §1 columns.

    python tools/dry_run_p3_local.py [--work DIR] [--tiny-ckpt DIR]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
for _p in (ROOT / "src", ROOT / "tests", TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_e1_notebook as be  # noqa: E402
import build_p3_notebook as bp  # noqa: E402
import dry_run_e1_local as d2  # noqa: E402
import dry_run_local as d1  # noqa: E402
from p3_commands import B2, all_run_names, run_names  # noqa: E402

from laya_poc.config import load_config, with_overrides  # noqa: E402

# Every config.phase3 feature once, on the smallest sizes: two seeds (the mean/range aggregation), laya_ml (the
# tiny ckpt stands in), head-only, both E5 rows under one arm id, two E6 subsets, both B2 models.
DRY_ARMS = [
    {"id": "E2", "model": "laya", "scheme": "c10", "seeds": [11, 22]},
    {"id": "E3", "model": "laya_ml", "scheme": "c10", "seeds": [11]},
    {"id": "E4", "model": "laya", "scheme": "c10", "seeds": [11], "freeze_encoder": True},
    {"id": "E5", "model": "laya", "scheme": "c7", "seeds": [11]},
    {"id": "E5", "model": "laya_ml", "scheme": "c7", "seeds": [11]},
    {"id": "E6", "model": "laya", "scheme": "c10", "seeds": [11], "train_subset": [16, 32]},
]
# The E1 dry run's FULL-mode data sizes (strict splits on the synthetic pool), 1 epoch, an eval every 8 opt steps
# (CPU trains MB 2 x ACC 2: ~16 opt steps per full-size epoch), small parity and order-invariance samples.
DRY_OVERRIDES = {
    **{k: v for k, v in d2.DRY_OVERRIDES.items() if k.startswith("data.")},
    "train.epochs": 1, "train.eval_every_opt_steps": 8, "smoke.parity_n": 8,
    "phase3.arms": DRY_ARMS, "phase3.order_invariance": {"split": "test_id", "n": 16, "perms": 2, "seed": 20260925},
}
SKIP_CELLS = ("setup", "install", "token")  # Step 1 runs first; pip and the token prompt are Colab-only
# results/runs.csv header (Phase 3 spec §1), in order
RUNS_CSV_COLUMNS = (
    "run_name", "arm", "model", "scheme", "seed", "subset", "head_only", "card", "epochs_run", "stop_reason",
    "best_opt_step", "T", "clamped", "val_macro_f1", "val_macro_f1_9", "val_acc", "val_ece_pre", "val_ece_post",
    "test_id_macro_f1", "ood_country_macro_f1", "ood_script_macro_f1", "ood_brand_macro_f1", "trap_candidates_acc",
    "stripped_false_confident", "order_invariance", "train_seconds", "run_seconds")


def dry_config(cfg: dict) -> dict:
    """A copy of the project config with FULL-mode sizes and a matrix small enough for a CPU run."""
    return with_overrides(cfg, DRY_OVERRIDES)


def local_target(work: Path, tiny: Path, pool: Path):
    """The E1 dry-run target (CPU, card CPU, tiny ckpt, synthetic pool), budgeting data for both models."""
    return dataclasses.replace(d2.local_target(work, tiny, pool), data_models=("laya", "laya_ml"))


def run_notebook_cells(dcfg: dict, target) -> dict:
    """Execute Step 1 and every later cell but pip/token in one namespace, in notebook order, as a kernel would."""
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
        if cell.key not in SKIP_CELLS:
            d1._exec_cell(cell, ns)
    return ns


def dry_run(work: Path, tiny: Path | None = None) -> Path:
    """Build the inputs and run the cells; returns the notebook's WORK directory."""
    cfg = load_config(ROOT / "config.yaml")
    dcfg = dry_config(cfg)
    work.mkdir(parents=True, exist_ok=True)
    tiny = tiny.resolve() if tiny else d2.build_tiny_ckpt(work / "tiny_ckpt", cfg)
    pool = d2.write_synthetic_full_pool(work / "pool.parquet", dcfg)
    with d1.preserved_environ():
        run_notebook_cells(dcfg, local_target(work / "laya_poc", tiny, pool))
    return work / "laya_poc"


def exit_check(report: dict) -> tuple[bool, str]:
    """(passed, what the incomplete runs lack) from phase3_report.json's exit_check (laya_poc.matrix_report)."""
    ex = report.get("exit_check") if isinstance(report.get("exit_check"), dict) else {}
    lacking = [f"{r.get('run_name')}: {', '.join(r.get('missing') or ['?'])}" for r in ex.get("runs") or []
               if not r.get("passed")]
    return ex.get("passed") is True, "; ".join(lacking)[:500] or "phase3_report.json has no passed exit_check"


def layout_failures(dcfg: dict, runs: Path) -> list[str]:
    """What the spec §1 run-directory layout is missing (or left behind) for each configured run."""
    splits, problems = dcfg["phase3"]["eval_splits"], []
    for name in all_run_names(dcfg):
        run, trained = Path(runs) / name, name not in run_names(dcfg, B2)
        need = ["done.json", "order_invariance.json", *(f"eval/{s}.json" for s in splits),
                *(f"preds/{s}.jsonl" for s in splits),
                *(["export_check.json", "train/best", "train/summary.json"] if trained else [])]
        problems += [f"{name}: no {rel}" for rel in need if not (run / rel).exists()]
        problems += [f"{name}: {rel} left after the run" for rel in ("train/ckpt", "train/final")
                     if trained and (run / rel).exists()]
    return problems


def csv_failures(dcfg: dict, csv_path: Path) -> list[str]:
    """results/runs.csv must have the §1 header and exactly one row per configured run."""
    if not Path(csv_path).is_file():
        return [f"no {csv_path}"]
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header, names = tuple(reader.fieldnames or ()), [row["run_name"] for row in reader]
    problems = [] if header == RUNS_CSV_COLUMNS else [f"runs.csv header {list(header)} != spec §1"]
    want = all_run_names(dcfg)
    if sorted(names) != sorted(want):
        problems.append(f"runs.csv rows {sorted(names)} != configured runs {sorted(want)}")
    return problems


def plumbing_failures(dcfg: dict, work: Path) -> list[str]:
    results = Path(work) / "results"
    report_path = results / "phase3_report.json"
    if not report_path.is_file():
        return [f"no {report_path}"]
    passed, note = exit_check(json.loads(report_path.read_text(encoding="utf-8")))
    failures = [] if passed else [f"Phase 3 exit check did not pass: {note}"]
    return failures + layout_failures(dcfg, Path(work) / "runs" / "p3") + csv_failures(dcfg, results / "runs.csv")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Colab Phase 3 notebook's cells locally on CPU (plumbing check).")
    ap.add_argument("--work", type=Path, help="scratch directory (default: a new temp dir, kept for inspection)")
    ap.add_argument("--tiny-ckpt", type=Path, help="existing tiny Laya checkpoint dir (default: build one)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # child output and the report contain non-ASCII text
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    root = (args.work or Path(tempfile.mkdtemp(prefix="laya_dry_p3_"))).resolve()
    start = time.time()
    try:
        work = dry_run(root, args.tiny_ckpt)
    except Exception as exc:
        print(f"error: Phase 3 dry run failed: {exc}", file=sys.stderr)
        return 1
    failed = plumbing_failures(dry_config(load_config(ROOT / "config.yaml")), work)
    print(f"\nreport: {work / 'results' / 'phase3_report.md'}\nPhase 3 exit check and run layout: "
          f"{'FAILED' if failed else 'OK'} in {time.time() - start:.0f}s")
    for line in failed:
        print(f"error: {line}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
