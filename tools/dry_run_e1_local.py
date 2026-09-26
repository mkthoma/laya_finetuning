"""Run the E1 notebook's cells locally on CPU: tiny checkpoint, synthetic FULL-mode pool, tiny sizes.

It executes the SAME cells tools/build_e1_notebook.py renders for Colab (Step 1, then Steps 4-13 and the
optional archive/cleanup cells) in one namespace, as a kernel would: the FULL data build with --verify-frozen,
the B1/B3 baselines, zero-shot evaluation, the E1 crash run (hard-killed right after e1.crash_at_micro_step) and
its resume with --save-best, the temperature fit on best/, the val evaluation, the CPU benchmark and the gate.
Only the target differs: --init <tiny ckpt> (init_scale 0.5, so it can learn a little), --pool <synthetic pool
of all 15 countries with brand chains, trap-word names and mixed-l1 rows>, --init-tokenizer, --device cpu, and
the sizes in DRY_OVERRIDES (2 epochs, early stopping off). Gate criteria 1-2 (end-to-end, numerics) must PASS
and E1 must train every planned micro-step of both epochs (stop_reason 'epochs'); 3-4 (accuracy vs zero-shot,
majority and TF-IDF+LR) are not meaningful on a tiny model. Exit code 1 when any of those plumbing checks fails.

    python tools/dry_run_e1_local.py [--work DIR] [--tiny-ckpt DIR]
"""
from __future__ import annotations

import argparse
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
import dry_run_local as d1  # noqa: E402
from laya_poc.config import load_config, with_overrides  # noqa: E402

# Tiny sizes that satisfy FULL mode's strict split sizes (config.split_spec(smoke=False)) on the synthetic pool.
# CPU trains MB 2 x ACC 2: ~64 micro-steps per epoch, a checkpoint every 8, the forced crash at 10 (resume
# from 8), a val eval every 2 opt steps. Early stopping is off (patience beyond any possible eval count): the
# tiny model's val F1 plateaus within an epoch, and the run must cross the epoch boundary (train_e1.jsonl,
# epoch 1's sigma and batch order) and end naturally (stop_reason 'epochs': final eval, final/ and best/),
# which is the new ground E1's 4 epochs cover on Colab beyond Phase 1's 1-epoch smoke test.
DRY_OVERRIDES = {
    "data.train_size": 120, "data.train_cap_per_class": 16, "data.val_size": 40, "data.test_size": 40,
    "data.ood_size": 12, "data.brand_top_n": 3, "data.trap_per_pattern": 3, "data.frozen_manifest": None,
    "train.epochs": 2, "train.eval_every_opt_steps": 2, "train.patience": 1_000_000,
    "e1.ckpt_every_micro_steps": 8, "e1.crash_at_micro_step": 10,
    "bench.n_records": 16, "bench.warmup": 2, "bench.batch_size": 8, "bench.threads": [1, 2],
}
# Code cells run locally, in notebook order. Step 2 (pip) and Step 3 (token) are Colab-only.
DRY_RUN_CELLS = ("env", "data", "baselines", "zeroshot", "crash", "resume", "export", "evaluate", "bench", "gate",
                 "archive", "cleanup")
EXTRA_COUNTRIES = ("KR",)  # the extraction's fallback for ood_script: FULL extracts it too
ID_ROWS, OOD_ROWS, BRAND_ROWS = 40, 20, 8
BRANDS = (("Mega Mart", "retail"), ("Quick Bite", "dining"), ("Fit Zone", "sports"))  # top names -> ood_brand
TRAP_NAMES = (("Station Cafe", "dining"), ("Kings Bank Bakery", "retail"), ("Park Lane Deli", "dining"),
              ("Museum Street Pharmacy", "health"), ("School Lane Garage", "services"), ("Clinic Road Cafe", "dining"))
MULTI_EVERY = 20  # every 20th ID row has two level-1 categories (n_l1 = 2): multi_category trap candidates
POOL_SCHEMA_EXTRA = ("l1s", "label", "n_l1", "first_l1")  # what extract.pool_sql adds to the kept fields


def dry_config(cfg: dict) -> dict:
    """A copy of the project config with FULL-mode sizes small enough for a CPU run in a few minutes."""
    return with_overrides(cfg, DRY_OVERRIDES)


def local_target(work: Path, tiny: Path, pool: Path):
    """The Phase 1 dry-run target: CPU, tiny checkpoint, synthetic pool, MB 2 x effective batch 4."""
    return d1.local_target(work, tiny, pool)


def _row(pid: str, rec: dict, l1: str, *, other: str | None = None) -> dict:
    """One pool row as extract.pool_sql shapes it (l1s sorted, label = l1s[0], first_l1 = first listed)."""
    l1s = sorted({l1, other} - {None})
    row = {c: rec.get(c) for c in d1.STRING_COLUMNS}
    return {**row, "fsq_place_id": pid, "l1s": l1s, "label": l1s[0], "n_l1": len(l1s), "first_l1": l1}


def _extras(ids: Sequence[str], key_to_l1: dict[str, str]) -> list[dict]:
    """Brand chains across the ID countries and trap-word names (one per country: never a brand)."""
    rows = []
    for (name, key) in BRANDS:
        for k in range(BRAND_ROWS):
            rec = {"name": name, "country": ids[k % len(ids)], "locality": f"Town{k}"}
            rows.append(_row(f"brand-{key}-{k:02d}", rec, key_to_l1[key]))
    for c in ids:
        for j, (name, key) in enumerate(TRAP_NAMES):
            rows.append(_row(f"trap-{c}-{j}", {"name": f"{name} {c}", "country": c, "locality": "Leeds"},
                             key_to_l1[key]))
    return rows


def synthetic_full_rows(dcfg: dict, seed: int = 0) -> list[dict]:
    """Pool rows for every FULL-mode country (tests/synth.py records; no gated data)."""
    from synth import synthetic_records

    from laya_poc import labels as L

    key_to_l1 = {key: name for name, key in L.L1_TO_KEY.items()}
    d = dcfg["data"]
    ids = list(d["id_countries"])
    rows = []
    for ci, country in enumerate([*ids, *d["ood_country"], d["ood_script"], *EXTRA_COUNTRIES]):
        n = ID_ROWS if country in ids else OOD_ROWS
        for i, (rec, key) in enumerate(synthetic_records(n, seed + ci)):
            rec = {**rec, "country": country, "address": f"{i} High Street" if i % 3 == 0 else None,
                   "facebook_id": str(10**14 + ci * 1000 + i) if i % 5 == 0 else None}  # VARCHAR, as in pool_sql
            multi = country in ids and i % MULTI_EVERY == MULTI_EVERY - 1
            other = key_to_l1["retail" if key != "retail" else "dining"] if multi else None
            rows.append(_row(f"{country.lower()}{i:05d}", rec, key_to_l1[key], other=other))
    return rows + _extras(ids, key_to_l1)


def write_synthetic_full_pool(path: Path, dcfg: dict, seed: int = 0) -> Path:
    """A pool parquet shaped like the FULL DuckDB extraction (build_data --pool)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([(c, pa.string()) for c in d1.STRING_COLUMNS]
                       + [("l1s", pa.list_(pa.string())), ("label", pa.string()), ("n_l1", pa.int64()),
                          ("first_l1", pa.string())])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(synthetic_full_rows(dcfg, seed), schema=schema), path)
    return path


def build_tiny_ckpt(dst: Path, cfg: dict) -> Path:
    """Tiny random Laya checkpoint (tests/conftest.py), init_scale 0.5 so it can learn a little."""
    from conftest import build_tiny_checkpoint

    from laya_poc import hub

    src = hub.snapshot(hub.model_spec(cfg, "laya"), files=("tokenizer/*", "encoder/*", "rl_agent_config.json"))
    return build_tiny_checkpoint(src, dst, init_scale=0.5)


def run_notebook_cells(dcfg: dict, target) -> dict:
    """Execute Step 1 and the pipeline cells in one namespace, as the Colab kernel would."""
    import yaml

    cells = {c.key: c for c in be.build_cells(dcfg, target, be.make_bundle(ROOT, dcfg)) if c.kind == "code"}
    ns: dict = {"__name__": "__main__"}
    work = Path(target.work)
    d1._exec_cell(cells["setup"], ns)
    # The bundle carries the Colab config; the CLIs read WORK/config.yaml (LAYA_POC_ROOT): swap in the tiny one.
    (work / "config.yaml").write_text(yaml.safe_dump(dcfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    # Stands in for Step 2's editable install: the CLIs import the unpacked bundle, not the repo checkout.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(work / "src"), os.environ.get("PYTHONPATH")]))
    for key in DRY_RUN_CELLS:
        d1._exec_cell(cells[key], ns)
    return ns


def dry_run(work: Path, tiny: Path | None = None) -> Path:
    """Build the inputs, run the cells; returns the runs/e1 directory holding the gate report."""
    cfg = load_config(ROOT / "config.yaml")
    dcfg = dry_config(cfg)
    work.mkdir(parents=True, exist_ok=True)
    tiny = tiny.resolve() if tiny else build_tiny_ckpt(work / "tiny_ckpt", cfg)
    pool = write_synthetic_full_pool(work / "pool.parquet", dcfg)
    with d1.preserved_environ():
        run_notebook_cells(dcfg, local_target(work / "laya_poc", tiny, pool))
    return work / "laya_poc" / "runs" / "e1"


def plumbing_failures(report: dict) -> list[str]:
    """Gate criteria that must pass even on a tiny model: end-to-end and numerics."""
    from laya_poc import gate as G

    crit = {c["name"]: c for c in report.get("criteria", [])}
    return [f"{name}: {crit[name]['note'] if name in crit else 'not evaluated'}"
            for name in (G.END_TO_END, G.NUMERICS) if not crit.get(name, {}).get("passed")]


def training_failures(runs: Path) -> list[str]:
    """What the E1 training run left untested: every planned micro-step of at least 2 epochs, ending naturally."""
    from e1_helpers import read_events

    train = Path(runs) / "train"
    if not (train / "summary.json").is_file():
        return [f"training: no {train / 'summary.json'}"]
    s = json.loads((train / "summary.json").read_text(encoding="utf-8"))
    planned = [e.get("total_micro_steps") for e in read_events(train / "log.jsonl") if e.get("event") == "start"]
    total = planned[-1] if planned else None
    if (s.get("epochs") or 0) < 2:
        return [f"training: {s.get('epochs')} planned epoch(s): the dry run never crosses an epoch boundary"]
    if s.get("stop_reason") != "epochs":
        return [f"training: stop_reason {s.get('stop_reason')!r} at micro-step {s.get('micro_steps')}, not 'epochs':"
                " the dry run must train every epoch to its natural end"]
    if s.get("micro_steps") != total:
        return [f"training: {s.get('micro_steps')} of the {total} planned micro-steps"]
    return []


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Colab E1 notebook's cells locally on CPU (plumbing check).")
    ap.add_argument("--work", type=Path, help="scratch directory (default: a new temp dir, kept for inspection)")
    ap.add_argument("--tiny-ckpt", type=Path, help="existing tiny Laya checkpoint dir (default: build one)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # child output and the report contain non-ASCII text
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    work = (args.work or Path(tempfile.mkdtemp(prefix="laya_dry_e1_"))).resolve()
    start = time.time()
    try:
        runs = dry_run(work, args.tiny_ckpt)
        report = json.loads((runs / "gate_report.json").read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"error: E1 dry run failed: {exc}", file=sys.stderr)
        return 1
    failed = plumbing_failures(report) + training_failures(runs)
    print(f"\ngate report: {runs / 'gate_report.md'}\nverdict: {report.get('verdict')} (criteria 3-4 are not "
          f"meaningful on a tiny model); plumbing {'FAILED' if failed else 'OK'} in {time.time() - start:.0f}s")
    for line in failed:
        print(f"error: {line}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
