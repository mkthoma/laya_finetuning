"""Run the Phase 4 notebook's cells locally on CPU: tiny stand-in models, a synthetic FULL-mode pool, tiny baselines.

It executes the SAME cells tools/build_p4_notebook.py renders for Colab (Step 1, then every step but the pip install
and the token prompt) in one namespace, as a kernel would: the FULL data build with --verify-frozen, the variants
(c7 and the trap candidates), B1, B3, B4 and B5 (RUN_B5 = True) through `matrix run`, the report and the archive.
Only the target differs: --device cpu and CARD "CPU"; --pool <synthetic FULL pool with trap candidates>; the tiny
random Laya checkpoint only lends its tokenizer to the data build and the variants (--init-tokenizer, --models laya
laya_ml); B4 gets `--init <tiny ModernBERT>` (the real pinned ModernBERT-base config with tiny dims + its tokenizer,
standing in for both encoders) and B5 `--init <tiny causal LM>` (the pinned Qwen3-4B config with tiny dims + its
tokenizer; the B4 one is tests/test_small_encoder.py's builder, the B5 one mirrors tests/test_llm_baseline.py's
fixture); and the tiny sizes of DRY_OVERRIDES. Exit code 1 unless the report's Phase 4 exit check PASSES, every
configured baseline run has the spec §1 run-directory layout, and results/runs.csv has one row per run with the
Phase 3 columns plus `kind`.

    python tools/dry_run_p4_local.py [--work DIR] [--tiny-ckpt DIR] [--tiny-encoder DIR] [--tiny-llm DIR]
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
import build_p4_notebook as bp  # noqa: E402
import dry_run_e1_local as d2  # noqa: E402
import dry_run_local as d1  # noqa: E402
import dry_run_p3_local as d3  # noqa: E402
from p4_commands import P4Target, all_run_names, arms  # noqa: E402

from laya_poc.config import load_config, with_overrides  # noqa: E402

# Every phase4 feature once, on the smallest sizes: B1 and B3 on both schemes, both B4 encoders (two seeds for the
# mean/range aggregation), the optional B5.
DRY_ARMS = [
    {"id": "B1", "kind": "majority", "schemes": ["c10", "c7"]},
    {"id": "B3", "kind": "tfidf_lr", "schemes": ["c10", "c7"]},
    {"id": "B4", "kind": "small_encoder", "model": "modernbert_base", "scheme": "c10", "seeds": [11, 22]},
    {"id": "B4", "kind": "small_encoder", "model": "mmbert_small", "scheme": "c10", "seeds": [11]},
    {"id": "B5", "kind": "llm", "model": "qwen3_4b", "scheme": "c10", "optional": True},
]
# The E1 dry run's FULL-mode data sizes (strict splits on the synthetic pool) with 2 augmented epochs, which B4
# trains on (best-epoch selection); small batches, B5 subsets and order-invariance samples.
DRY_OVERRIDES = {
    **{k: v for k, v in d2.DRY_OVERRIDES.items() if k.startswith("data.")},
    "train.epochs": 2, "phase4.arms": DRY_ARMS,
    "phase4.small_encoder_train.epochs": 2, "phase4.small_encoder_train.batch_size": 8,
    "phase4.small_encoder_train.max_length": 64,
    "phase4.llm_eval.subset_per_pool": 12, "phase4.llm_eval.val_for_temperature": 16,
    "phase3.order_invariance": {"split": "test_id", "n": 16, "perms": 2, "seed": 20260925},
    "baselines.lr.C_grid": [1, 4], "eval.batch_size": 16,
}
TINY_LLM = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                num_hidden_layers=2, max_window_layers=2, initializer_range=0.5)  # tests/test_llm_baseline.py TINY
LLM_HUB_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
SKIP_CELLS = ("setup", "install", "token")  # Step 1 runs first; pip and the token prompt are Colab-only
RUNS_CSV_COLUMNS = (*d3.RUNS_CSV_COLUMNS, "kind")  # spec P4 §5: `kind` appended at the END
ORDER_INVARIANCE_KINDS = ("tfidf_lr", "small_encoder")  # spec P4 §1: B1 and B5 skip it


def dry_config(cfg: dict) -> dict:
    """A copy of the project config with FULL-mode sizes and baselines small enough for a CPU run."""
    return with_overrides(cfg, DRY_OVERRIDES)


def build_tiny_encoder(dst: Path, cfg: dict) -> Path:
    """The B4 stand-in: tests/test_small_encoder.py's builder on the pinned ModernBERT-base config + tokenizer."""
    from test_small_encoder import build_tiny_encoder as build, hub_config_dir

    return build(hub_config_dir(cfg, "modernbert_base"), dst)


def build_tiny_llm(dst: Path, cfg: dict, key: str = "qwen3_4b") -> Path:
    """The B5 stand-in, built like tests/test_llm_baseline.py's tiny_llm_dir fixture: a random 2-layer causal LM from
    the pinned Qwen3-4B config.json with tiny dims, plus its tokenizer (config and tokenizer files only)."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    spec = cfg["phase4"]["llms"][key]
    src = Path(snapshot_download(spec["id"], revision=spec["revision"], allow_patterns=list(LLM_HUB_FILES)))
    real = json.loads((src / "config.json").read_text(encoding="utf-8"))
    kept = {k: v for k, v in real.items() if k not in ("architectures", "transformers_version", "torch_dtype")}
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(AutoConfig.for_model(**{**kept, **TINY_LLM})).save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    return dst


def local_target(work: Path, tiny: Path, pool: Path, encoder: Path, llm: Path) -> P4Target:
    """The Phase 3 dry-run target (CPU, card CPU, tiny Laya tokenizer for the data, synthetic pool) plus the B4/B5
    stand-ins."""
    base = d3.local_target(work, tiny, pool)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    return P4Target(**fields, encoder_init=str(encoder), llm_init=str(llm))


def run_notebook_cells(dcfg: dict, target: P4Target) -> dict:
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


def dry_run(root: Path, tiny: Path | None = None, encoder: Path | None = None, llm: Path | None = None) -> Path:
    """Build the inputs and run the cells; returns the notebook's WORK directory."""
    cfg = load_config(ROOT / "config.yaml")
    dcfg = dry_config(cfg)
    root.mkdir(parents=True, exist_ok=True)
    tiny = tiny.resolve() if tiny else d2.build_tiny_ckpt(root / "tiny_ckpt", cfg)
    encoder = encoder.resolve() if encoder else build_tiny_encoder(root / "tiny_encoder", cfg)
    llm = llm.resolve() if llm else build_tiny_llm(root / "tiny_llm", cfg)
    pool = d2.write_synthetic_full_pool(root / "pool.parquet", dcfg)
    with d1.preserved_environ():
        run_notebook_cells(dcfg, local_target(root / "laya_poc", tiny, pool, encoder, llm))
    return root / "laya_poc"


def phase4_exit(report: dict) -> tuple[bool, str]:
    """(passed, what the incomplete runs lack) from phase4_report.json's Phase 4 exit check."""
    ex = report.get("phase4_exit_check")
    ex = ex if isinstance(ex, dict) else {}
    lacking = [f"{r.get('run_name')}: {', '.join(r.get('missing') or ['?'])}" for r in ex.get("runs") or []
               if isinstance(r, dict) and not r.get("passed")]
    return ex.get("passed") is True, "; ".join(lacking)[:500] or "the report JSON has no passed phase4_exit_check"


def _needs(arm: dict, splits: Sequence[str]) -> list[str]:
    need = ["done.json", "calibration.json", *(f"eval/{s}.json" for s in splits), *(f"preds/{s}.jsonl" for s in splits)]
    need += ["preds/no_gate/stripped_test.jsonl"] if "stripped_test" in splits else []
    need += ["order_invariance.json"] if arm["kind"] in ORDER_INVARIANCE_KINDS else []
    return need + (["train/summary.json"] if arm["kind"] == "small_encoder" else [])


def layout_failures(dcfg: dict, runs: Path) -> list[str]:
    """What the spec §1 baseline run-directory layout is missing for each configured run."""
    from p4_commands import row_run_names

    splits, problems = dcfg["phase3"]["eval_splits"], []
    for arm in arms(dcfg):
        for name in row_run_names(arm):
            problems += [f"{name}: no {rel}" for rel in _needs(arm, splits) if not (Path(runs) / name / rel).exists()]
    return problems


def csv_failures(dcfg: dict, csv_path: Path) -> list[str]:
    """results/runs.csv must have the Phase 3 header + `kind` and exactly one row per configured baseline run."""
    if not Path(csv_path).is_file():
        return [f"no {csv_path}"]
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header, names = tuple(reader.fieldnames or ()), [row["run_name"] for row in reader]
    problems = [] if header == RUNS_CSV_COLUMNS else [f"runs.csv header {list(header)} != Phase 3 columns + kind"]
    if sorted(names) != sorted(all_run_names(dcfg)):
        problems.append(f"runs.csv rows {sorted(names)} != configured runs {sorted(all_run_names(dcfg))}")
    return problems


def plumbing_failures(dcfg: dict, work: Path) -> list[str]:
    results = Path(work) / "results"
    report_path = results / "phase4_report.json"
    if not report_path.is_file():
        return [f"no {report_path}"]
    passed, note = phase4_exit(json.loads(report_path.read_text(encoding="utf-8")))
    failures = [] if passed else [f"Phase 4 exit check did not pass: {note}"]
    return failures + layout_failures(dcfg, Path(work) / "runs" / "p4") + csv_failures(dcfg, results / "runs.csv")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Colab Phase 4 notebook's cells locally on CPU (plumbing check).")
    ap.add_argument("--work", type=Path, help="scratch directory (default: a new temp dir, kept for inspection)")
    ap.add_argument("--tiny-ckpt", type=Path, help="existing tiny Laya checkpoint dir (default: build one)")
    ap.add_argument("--tiny-encoder", type=Path, help="existing tiny ModernBERT dir for B4 (default: build one)")
    ap.add_argument("--tiny-llm", type=Path, help="existing tiny causal LM dir for B5 (default: build one)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # child output and the report contain non-ASCII text
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    root = (args.work or Path(tempfile.mkdtemp(prefix="laya_dry_p4_"))).resolve()
    start = time.time()
    try:
        work = dry_run(root, args.tiny_ckpt, args.tiny_encoder, args.tiny_llm)
    except Exception as exc:
        print(f"error: Phase 4 dry run failed: {exc}", file=sys.stderr)
        return 1
    failed = plumbing_failures(dry_config(load_config(ROOT / "config.yaml")), work)
    print(f"\nreport: {work / 'results' / 'phase4_report.md'}\nPhase 4 exit check and run layout: "
          f"{'FAILED' if failed else 'OK'} in {time.time() - start:.0f}s")
    for line in failed:
        print(f"error: {line}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
