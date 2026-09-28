"""The Phase 7 results workbook (design §6.2 Phase 7): one .xlsx of the PoC's aggregate results.

    python -m laya_poc.phase7_workbook --report runs/phase5/phase5_report.json
        --bench-cpu runs/phase5/bench_laptop_merged.json --gpu-timing runs/phase7/gpu_timing.json
        [--bench-gpu runs/p7/bench_gpu.json] [--samples-stats runs/phase7/sample_agreement.json]
        --notice data/NOTICE_FSQ.txt --out docs/results/phase7_results_workbook_<date>.xlsx
        [--config config.yaml] [--fingerprint <data manifest json>] [--archive <Colab archive dir> ...]
        [--notes-dir docs/results]

Sheets: README (what it is, the verdict, a guide to the sheets, the FSQ NOTICE verbatim), Decision (§5.12 checklist
per candidate), Main results, Robustness, Bootstrap, Per-class, Look-alikes (from the Phase 5 report JSON), CPU (the
bench_cpu JSON: rows, machine, ONNX acceptance), GPU throughput (phase7_timing JSON), GPU latency (bench_gpu JSON, or
a pending row), Samples (phase7_samples stats JSON, flattened; or a note), Provenance (phase7_provenance).
Metrics only (design Appendix D): the inputs are aggregate JSONs, and the writer refuses a cell that looks like a row
id. The NOTICE file must hold the upstream FSQ NOTICE verbatim followed by the "modified" statement. Written
atomically. Defaults: --fingerprint = config data.frozen_manifest, --archive = runs/*_colab, --notes-dir =
docs/results (all under the project root). Torch-free.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_config, project_root
from .env_check import run_cli
from .notice import FSQ_NOTICE
from .phase7_provenance import provenance_sheet, rel
from .phase7_workbook_tables import (bootstrap_sheet, cpu_sheet, decision_sheet, generic_rows_sheet, lookalike_sheet,
                                     main_sheet, per_class_sheet, robustness_sheet)
from .phase7_xlsx import (F0, F1, F2, F3, TOP_LEVEL, Sheet, Table, cols, flatten_records, kv_table,
                          message_table, records_table, write_workbook)

PROG = "phase7_workbook"
GPU_PENDING = "pending: run notebooks/phase7_gpu_bench.ipynb, then pass its JSON as --bench-gpu"
SAMPLES_PENDING = "not given: build runs/phase7/sample_agreement.json (laya_poc.phase7_samples), then pass it as " \
                  "--samples-stats"
GUIDE = {
    "Decision": "The §5.12 decision checklist per candidate: criteria C1-C5 (value, threshold, result), the C1 "
                "strict sensitivity, stop and investigate checks, candidate and overall verdicts, pending inputs, "
                "thresholds, and every detail number (leads, bootstrap CIs, gaps, backends, seed ranges).",
    "Main results": "Per pool (test_id, ood_country, ood_script, ood_brand) and arm: macro-F1 seed mean / min / max, "
                    "accuracy, ECE before and after temperature scaling, Brier, NLL, accuracy at 80/90 % coverage, "
                    "the ID->OOD gap.",
    "Robustness": "Trap accuracy (status and basis), stripped-record abstention and false-confident rates, seed flip "
                  "rate, field-order invariance, seed ranges.",
    "Bootstrap": "Paired bootstrap of candidate minus baseline macro-F1 per pool: difference, 95 % CI, P(> 0), "
                 "P(>= lead).",
    "Per-class": "Precision / recall / F1 per class (seed mean, min, max) for the candidates and the best baseline.",
    "Look-alikes": "Confusion rates between the look-alike class pairs (both directions), pooled over seeds.",
    "CPU": "CPU benchmark rows (model, backend, threads, cold start, p50 / p95 batch-1 latency, batched rec/s, peak "
           "RAM, hardware), the machine block, ONNX acceptance, Laya vs matched encoder throughput.",
    "GPU throughput": "Batched scoring throughput of every Colab run (one pass per eval split, at the eval batch "
                      "size, end to end incl. tokenisation; not batch-1 latency) and training minutes, from the run "
                      "archives (laya_poc.phase7_timing).",
    "GPU latency": "GPU batch-1 latency and batched throughput (laya_poc.bench_gpu), or the pending row.",
    "Samples": "Per-split accuracy / abstention per run and the 2x2 agreement counts of the sample inferences "
               "(laya_poc.phase7_samples stats; counts only).",
    "Provenance": "Laya commit and Hub revision, pinned baseline revisions, FSQ release and frozen data fingerprint, "
                  "seeds, decision thresholds, config.yaml and input sha256s, notebook bundle records.",
}


# ---------------------------------------------------------------- inputs

def load_json(path: str | Path, what: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{what} JSON not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{what} JSON is not an object: {p}")
    return data


def read_notice(path: str | Path) -> str:
    """The NOTICE text (trailing newlines dropped): the FSQ NOTICE verbatim, then the 'modified' statement."""
    text = Path(path).read_text(encoding="utf-8").rstrip("\n")
    if FSQ_NOTICE not in text or "modified" not in text.split(FSQ_NOTICE, 1)[1]:
        raise ValueError(f"{path}: not the FSQ NOTICE verbatim followed by the statement that the data was "
                         "modified (design Appendix D)")
    return text


def default_archives(root: Path) -> list[Path]:
    return sorted(p for p in (root / "runs").glob("*_colab") if p.is_dir())


# ---------------------------------------------------------------- sheets built here

def readme_sheet(report: Mapping, notice: str, inputs: Mapping[str, Path | None], generated_at: str,
                 names: Sequence[str]) -> Sheet:
    d = report.get("decision") or {}
    state = "final" if d.get("final") else "provisional"
    cands = "; ".join(f"{c.get('id')}: {c.get('verdict')}" for c in d.get("candidates") or [])
    pairs = [("Workbook", "Laya record-normalisation PoC, results workbook (Phase 7): aggregate results of fine-tuning "
                          "Laya to classify Foursquare OS Places records into 10 categories, against baselines B1 "
                          "(majority / prior), B3 (TF-IDF+LR), B4 (ModernBERT-base, mmBERT-small) and B5 (Qwen3-4B "
                          "zero-shot). Built by laya_poc.phase7_workbook."),
             ("Generated at (UTC)", generated_at),
             ("Verdict (§5.12)", f"{d.get('verdict')} ({state})"), ("Candidate verdicts", cands),
             ("Statement", d.get("statement")),
             ("Metrics only", "No FSQ record, place name, state or row id is in this workbook (design Appendix D); "
                              "the generator refuses to write a cell that looks like a row id."),
             *((f"Input: {k}", rel(p) or "not given") for k, p in inputs.items()),
             ("FSQ NOTICE (verbatim, with the statement that the data was modified)", notice)]
    guide = Table("Sheets", cols(("item", "Sheet"), ("value", "What it holds")),
                  [{"item": n, "value": GUIDE.get(n)} for n in names])
    return Sheet("README", [kv_table(None, pairs), guide], wrap=True)


def _joined(rows: Sequence[Mapping], key: str = "notes") -> list[dict]:
    return [{**r, key: "; ".join(r.get(key) or [])} for r in rows]


def gpu_throughput_sheet(gpu: Mapping | None) -> Sheet:
    if not gpu:
        return Sheet("GPU throughput", [message_table("not given: run laya_poc.phase7_timing, then --gpu-timing")])
    summary = Table(None, cols(
        ("group", "Group"), ("arm", "Arm"), ("model", "Model"), ("kind", "Kind"), ("card", "Card"),
        ("device", "Device"), ("precision", "Precision"), ("batch_size", "Batch"), ("n_runs", "Runs", F0),
        ("seeds", "Seeds"), ("rows_per_s_median", "Rows/s (median)", F1), ("rows_per_s_min", "Rows/s min", F1),
        ("rows_per_s_max", "Rows/s max", F1), ("n_points", "Points", F0), ("splits_used", "Splits used"),
        ("train_min_median", "Train min (median/run)", F2), ("total_min_median", "Total min (median/run)", F2),
        ("notes", "Notes")), _joined(gpu.get("summary") or []))
    splits = Table("Per group and split", cols(
        ("group", "Group"), ("split", "Split"), ("rows_forwarded", "Rows forwarded", F0), ("n_scored", "Scored", F0),
        ("n_runs", "Runs", F0), ("one_pass_s_median", "One-pass s (median)", F3),
        ("rows_per_s_median", "Rows/s (median)", F1), ("batch_size", "Batch"), ("device", "Device"),
        ("card", "Card"), ("precision", "Precision"), ("notes", "Notes")), _joined(gpu.get("throughput") or []))
    runs = Table("Per run", cols(
        ("run", "Run"), ("group", "Group"), ("seed", "Seed"), ("card", "Card"), ("device", "Device"),
        ("precision", "Precision"), ("train_precision", "Train precision"), ("train_s", "Train s", F1),
        ("scoring_s", "Scoring s (one pass, all splits)", F2), ("eval_step_s", "Eval step s", F1),
        ("export_check_s", "Export check s", F1), ("load_s", "Load s", F1), ("total_s", "Total s", F1),
        ("notes", "Notes")), _joined(gpu.get("runs") or []))
    cards = "; ".join(f"{c} = {n}" for c, n in (gpu.get("cards") or {}).items())
    defs = kv_table("Definitions", [("Generated at (UTC)", gpu.get("generated_at")), ("Cards", cards),
                                    ("Min scored rows per split (summary)", gpu.get("min_scored_rows")),
                                    *(gpu.get("definitions") or {}).items()])
    return Sheet("GPU throughput", [summary, splits, runs, defs])


def gpu_latency_sheet(bench_gpu: Mapping | None) -> Sheet:
    return generic_rows_sheet("GPU latency", bench_gpu) if bench_gpu else Sheet("GPU latency",
                                                                                [message_table(GPU_PENDING)])


def samples_sheet(samples: Mapping | None) -> Sheet:
    if not samples:
        return Sheet("Samples", [message_table(SAMPLES_PENDING)])
    records, dropped = flatten_records(samples)
    top = next((r for r in records if r["Path"] == TOP_LEVEL), {})
    nested = [r for r in records if r["Path"] != TOP_LEVEL]
    first = records_table(None, nested) if nested else message_table("no per-split numbers in the given JSON")
    context = kv_table("Context", [*((k, v) for k, v in top.items() if k != "Path"),
                                   ("Keys or values withheld (record fields or row-id-like; metrics only)", dropped)])
    return Sheet("Samples", [first, context])


# ---------------------------------------------------------------- build and CLI

def build_sheets(args: argparse.Namespace, cfg: Mapping, generated_at: str) -> list[Sheet]:
    root = project_root()
    report, bench = load_json(args.report, "report"), load_json(args.bench_cpu, "bench-cpu")
    gpu = load_json(args.gpu_timing, "gpu-timing")
    bench_gpu = load_json(args.bench_gpu, "bench-gpu") if args.bench_gpu else None
    samples = load_json(args.samples_stats, "samples-stats") if args.samples_stats else None
    notice = read_notice(args.notice)
    fp_path = Path(args.fingerprint) if args.fingerprint else root / str(cfg["data"]["frozen_manifest"])
    fingerprint = load_json(fp_path, "fingerprint") if fp_path.is_file() else None
    inputs = {k: Path(v) if v else None for k, v in (("report", args.report), ("bench_cpu", args.bench_cpu),
                                                      ("gpu_timing", args.gpu_timing), ("bench_gpu", args.bench_gpu),
                                                      ("samples_stats", args.samples_stats), ("notice", args.notice))}
    body = [decision_sheet(report), main_sheet(report), robustness_sheet(report), bootstrap_sheet(report),
            per_class_sheet(report), lookalike_sheet(report), cpu_sheet(bench, report), gpu_throughput_sheet(gpu),
            gpu_latency_sheet(bench_gpu), samples_sheet(samples),
            provenance_sheet(cfg=cfg, config_path=Path(args.config) if args.config else root / "config.yaml",
                             report=report, gpu=gpu, fingerprint=fingerprint, fp_path=fp_path, inputs=inputs,
                             archives=[Path(a) for a in args.archive] if args.archive else default_archives(root),
                             notes_dir=Path(args.notes_dir) if args.notes_dir else root / "docs" / "results")]
    return [readme_sheet(report, notice, inputs, generated_at, [s.name for s in body]), *body]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=f"laya_poc.{PROG}", description=__doc__.splitlines()[0])
    p.add_argument("--report", required=True, help="phase5_report JSON")
    p.add_argument("--bench-cpu", required=True, help="bench_cpu JSON (the judged machine's)")
    p.add_argument("--gpu-timing", required=True, help="phase7_timing JSON")
    p.add_argument("--bench-gpu", default=None, help="bench_gpu JSON (optional: else a pending row)")
    p.add_argument("--samples-stats", default=None, help="phase7_samples --stats-out JSON (optional)")
    p.add_argument("--notice", required=True, help="data/NOTICE_FSQ.txt")
    p.add_argument("--out", required=True, help="the .xlsx path")
    p.add_argument("--config", default=None)
    p.add_argument("--fingerprint", default=None, help="frozen data manifest JSON (default: config)")
    p.add_argument("--archive", action="append", default=[], help="a Colab archive dir to search for bundle sha256s")
    p.add_argument("--notes-dir", default=None, help="result reports whose header names a bundle (docs/results)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        cfg = load_config(args.config)
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sheets = build_sheets(args, cfg, generated_at)
        out = write_workbook(sheets, Path(args.out), creator=f"laya_poc.{PROG}")
        print(f"{PROG}: wrote {out} ({len(sheets)} sheets)", flush=True)
        return 0

    return run_cli(PROG, body, args.out)


if __name__ == "__main__":
    sys.exit(main())
