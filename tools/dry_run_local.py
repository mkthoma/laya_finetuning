"""Run the smoke notebook's cells locally on CPU with a tiny checkpoint, a synthetic pool and tiny sizes.

It executes the SAME cells tools/build_notebook.py renders for Colab (Step 1, then Steps 4-12 and the
optional archive/cleanup/kill-check cells) in one namespace, as a kernel would, so the training commands are
the Colab ones: the control run with --initial-eval and no resumable checkpoints, the crash run checkpointing
every smoke.ckpt_every_micro_steps, the resume without checkpoints. Only the target differs: --init <tiny
ckpt>, --pool <synthetic pool>, --init-tokenizer, --allow-cpu, --device cpu, and the sizes in DRY_OVERRIDES.
The numeric exit criteria may FAIL on a tiny random model; the plumbing must not crash.

    python tools/dry_run_local.py [--work DIR] [--tiny-ckpt DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator, Sequence

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
for _p in (ROOT / "src", ROOT / "tests", TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_notebook as bn  # noqa: E402
from laya_poc.config import load_config, with_overrides  # noqa: E402

# Tiny sizes that still satisfy config.validate_config: crash after the micro-step-8 checkpoint (grad
# accumulation 2 on CPU, 4 on the T4 profile), compare 8 micro-steps after the resume.
DRY_OVERRIDES = {
    "smoke.train_questions": 96, "smoke.train_cap_per_class": 20, "smoke.val_size": 32, "smoke.test_size": 32,
    "smoke.brand_top_n": 2, "smoke.zero_shot_n": 8, "smoke.parity_n": 16,
    "smoke.micro_steps": 24, "smoke.ckpt_every_micro_steps": 8, "smoke.kill_at_micro_step": 10,
    "smoke.resume_compare_steps": 8,
}
# Code cells run locally, in notebook order. Step 2 (pip) and Step 3 (token) are Colab-only; Step 13c
# would kill this process.
DRY_RUN_CELLS = ("env", "data", "zeroshot", "parity", "control", "crash", "resume", "export", "report",
                 "archive", "cleanup", "verify_kill")
STRING_COLUMNS = ("fsq_place_id", "name", "address", "locality", "region", "postcode", "admin_region", "post_town",
                  "po_box", "country", "tel", "website", "email", "facebook_id", "instagram", "twitter")
POOL_COLUMNS = (*STRING_COLUMNS, "l1s", "label", "n_l1")  # what extract.pool_sql produces


def dry_config(cfg: dict) -> dict:
    """A copy of the project config with smoke sizes small enough for a CPU run in about a minute."""
    return with_overrides(cfg, DRY_OVERRIDES)


def write_synthetic_pool(path: Path, n: int = 240, seed: int = 0) -> Path:
    """A pool parquet shaped like the DuckDB extraction, built from tests/synth.py records (no gated data)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from synth import synthetic_records

    from laya_poc import labels as L

    key_to_l1 = {key: name for name, key in L.L1_TO_KEY.items()}
    rows = []
    for i, (rec, key) in enumerate(synthetic_records(n, seed)):
        row = {c: rec.get(c) for c in STRING_COLUMNS}
        row.update(fsq_place_id=f"dry{i:06d}", address=f"{i} High Street" if i % 3 == 0 else None,
                   facebook_id=str(10**14 + i) if i % 5 == 0 else None,  # VARCHAR, as CAST in pool_sql
                   l1s=[key_to_l1[key]], label=key_to_l1[key], n_l1=1)
        rows.append(row)
    schema = pa.schema([(c, pa.string()) for c in STRING_COLUMNS]
                       + [("l1s", pa.list_(pa.string())), ("label", pa.string()), ("n_l1", pa.int64())])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def build_tiny_ckpt(dst: Path, cfg: dict) -> Path:
    """Tiny random Laya checkpoint, built exactly like the tests' `tiny_ckpt_dir` fixture."""
    from conftest import build_tiny_checkpoint

    from laya_poc import hub

    src = hub.snapshot(hub.model_spec(cfg, "laya"), files=("tokenizer/*", "encoder/*", "rl_agent_config.json"))
    return build_tiny_checkpoint(src, dst)


def local_target(work: Path, tiny: Path, pool: Path) -> bn.Target:
    return bn.Target(work=str(work), device="cpu", require_gpu=False, init=str(tiny), pool=str(pool),
                     data_models=("laya",), micro_batch=2, effective_batch=4, card="CPU")


@contextlib.contextmanager
def preserved_environ() -> Iterator[None]:
    """Step 1 sets environment variables for its CLIs; undo them when the dry run ends."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _exec_cell(cell: bn.Cell, ns: dict) -> None:
    print(f"\n===== {cell.source.splitlines()[0].lstrip('# ')}", flush=True)
    start = time.time()
    try:
        exec(compile(cell.source, f"<cell {cell.key}>", "exec"), ns)
    except Exception as exc:
        raise RuntimeError(f"cell '{cell.key}' failed: {exc}") from exc
    print(f"----- {cell.key}: {time.time() - start:.0f}s", flush=True)


def run_notebook_cells(dcfg: dict, target: bn.Target) -> dict:
    """Execute Step 1 and the pipeline cells in one namespace, as the Colab kernel would."""
    import yaml

    cells = {c.key: c for c in bn.build_cells(dcfg, target, bn.make_bundle(ROOT)) if c.kind == "code"}
    ns: dict = {"__name__": "__main__"}
    work = Path(target.work)
    _exec_cell(cells["setup"], ns)
    # The bundle carries the Colab config; the CLIs read WORK/config.yaml (LAYA_POC_ROOT), so swap in the tiny one.
    (work / "config.yaml").write_text(yaml.safe_dump(dcfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    # Stands in for Step 2's editable install: the CLIs import the unpacked bundle, not the repo checkout.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(work / "src"), os.environ.get("PYTHONPATH")]))
    for key in DRY_RUN_CELLS:
        _exec_cell(cells[key], ns)
    return ns


def dry_run(work: Path, tiny: Path | None = None) -> Path:
    """Build the inputs, run the cells; returns the runs/smoke directory holding the report."""
    cfg = load_config(ROOT / "config.yaml")
    work.mkdir(parents=True, exist_ok=True)
    tiny = tiny.resolve() if tiny else build_tiny_ckpt(work / "tiny_ckpt", cfg)
    pool = write_synthetic_pool(work / "pool.parquet")
    with preserved_environ():
        run_notebook_cells(dry_config(cfg), local_target(work / "laya_poc", tiny, pool))
    return work / "laya_poc" / "runs" / "smoke"


def _failed(obj: object) -> list[str]:
    items = obj if isinstance(obj, list) else [c for v in obj.values() if isinstance(v, list) for c in v]
    return [str(c.get("name", "?")) for c in items if isinstance(c, dict) and c.get("passed") is False]


def read_verdict(runs: Path) -> tuple[str, list[str]]:
    """(overall verdict, failed criteria) from smoke_report.json, else the markdown; UNKNOWN when absent."""
    js, md = runs / "smoke_report.json", runs / "smoke_report.md"
    if js.exists():
        obj = json.loads(js.read_text(encoding="utf-8"))
        failed = _failed(obj) if isinstance(obj, (dict, list)) else []
        if isinstance(obj, dict):
            for key in ("verdict", "overall"):
                if isinstance(obj.get(key), str):
                    return obj[key].upper(), failed
            if isinstance(obj.get("passed"), bool):
                return ("PASS" if obj["passed"] else "FAIL"), failed
        elif isinstance(obj, list):
            return ("FAIL" if failed else "PASS"), failed
    if md.exists():
        m = re.search(r"verdict\W{0,12}(PASS|FAIL)", md.read_text(encoding="utf-8"), re.IGNORECASE)
        if m:
            return m.group(1).upper(), []
    return "UNKNOWN", []


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Colab smoke-test cells locally on CPU (plumbing check).")
    ap.add_argument("--work", type=Path, help="scratch directory (default: a new temp dir, kept for inspection)")
    ap.add_argument("--tiny-ckpt", type=Path, help="existing tiny Laya checkpoint dir (default: build one)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # child output and the report contain non-ASCII text
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    work = (args.work or Path(tempfile.mkdtemp(prefix="laya_dry_run_"))).resolve()
    start = time.time()
    try:
        runs = dry_run(work, args.tiny_ckpt)
    except Exception as exc:
        print(f"error: dry run failed: {exc}", file=sys.stderr)
        return 1
    verdict, failed = read_verdict(runs)
    print(f"\nreport: {runs / 'smoke_report.md'}")
    print(f"verdict: {verdict}" + (f" (failed: {', '.join(failed)})" if failed else "")
          + f"; plumbing completed in {time.time() - start:.0f}s (numeric criteria are not meaningful on a tiny model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
