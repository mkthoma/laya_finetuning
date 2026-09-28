"""One record per matrix run, read back from its run directory, and results/runs.csv (spec P3 §1: one row per
finished run, fixed header, sorted by run name).

A record is a superset of the CSV row: the report also needs per-split ECE, the order-invariance verdict,
the export_check verdict and whether best/ exists. Missing files and metrics become None, never an error:
the report and the exit check decide what a gap means.

Phase 4 baselines (spec P4 §1, §5) carry `kind` in done.json (Laya runs: none, read as `laya`); their T comes
from calibration.json and their train columns are empty except for B4 (epochs and seconds from
train/summary.json). The CSV gains a `kind` column at the END of the Phase 3 header as soon as a baseline row is
present; a CSV of Laya runs only keeps the Phase 3 header exactly. Records can come from several runs roots
(runs/p3, runs/p4, archives): the first root holding a run name wins.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from .gate import wall_clock
from .matrix_exec import read_json
from .smoke_report import read_events

CSV_COLUMNS = ("run_name", "arm", "model", "scheme", "seed", "subset", "head_only", "card", "epochs_run",
               "stop_reason", "best_opt_step", "T", "clamped", "val_macro_f1", "val_macro_f1_9", "val_acc",
               "val_ece_pre", "val_ece_post", "test_id_macro_f1", "ood_country_macro_f1", "ood_script_macro_f1",
               "ood_brand_macro_f1", "trap_candidates_acc", "stripped_false_confident", "order_invariance",
               "train_seconds", "run_seconds")
OOD_POOLS = ("ood_country", "ood_script", "ood_brand")
IDENTITY = ("run_name", "arm", "model", "scheme", "seed", "subset", "head_only", "card")
BEST_WEIGHTS = "model.safetensors"
KIND_COLUMN = "kind"
LAYA_KIND = "laya"
ENCODER_KIND = "small_encoder"


def num(x: Any) -> float | None:
    """A finite real number as float, else None."""
    ok = isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    return float(x) if ok else None


def stat(values: Sequence[Any]) -> dict[str, float] | None:
    """{mean, min, max, n} over the finite values; None when there are none."""
    xs = [v for v in (num(x) for x in values) if v is not None]
    return {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "n": len(xs)} if xs else None


def split_metrics(ev: Any) -> dict[str, float | None]:
    """The headline numbers of one evaluate output (post-T unless named _pre)."""
    ev = ev if isinstance(ev, dict) else {}
    post, pre = ev.get("post") or {}, ev.get("pre") or {}
    return {"macro_f1": num(post.get("macro_f1")), "macro_f1_9": num(post.get("macro_f1_9")),
            "acc": num(post.get("acc")), "ece_pre": num(pre.get("ece")), "ece_post": num(post.get("ece")),
            "temperature": num(ev.get("temperature"))}


def epochs_run(events: list[dict], summary: dict) -> float | None:
    """Epochs trained, fractional after an early stop: micro-steps over the per-epoch micro-steps of the
    (last) start event."""
    start = next((e for e in reversed(events) if e.get("event") == "start" and e.get("n_items_per_epoch")), None)
    micro, mb = summary.get("micro_steps"), (start or {}).get("micro_batch")
    if start is None or not isinstance(micro, int) or not isinstance(mb, int) or mb < 1:
        return None
    left, done = micro, 0.0
    for n in start["n_items_per_epoch"]:
        per = math.ceil(n / mb)
        if left <= 0 or per == 0:
            break
        done, left = done + min(left, per) / per, left - min(left, per)
    return round(done, 2)


def train_seconds(events: list[dict]) -> float | None:
    """Training work over every process of the run (a resume after a disconnect adds a segment)."""
    try:
        return round(wall_clock(events)[0], 1)
    except LookupError:
        return None


def _train_facts(run_dir: Path) -> dict[str, Any]:
    train = run_dir / "train"
    summary = read_json(train / "summary.json")
    summary = summary if isinstance(summary, dict) else {}
    events = read_events(train / "log.jsonl")[0] if (train / "log.jsonl").is_file() else []
    start = next((e for e in events if e.get("event") == "start" and e.get("n_items_per_epoch")), {})
    per_epoch = start.get("n_items_per_epoch") or [None]
    return {"epochs_run": epochs_run(events, summary), "stop_reason": summary.get("stop_reason"),
            "best_opt_step": summary.get("best_opt_step"), "train_seconds": train_seconds(events),
            "n_train": per_epoch[0], "has_best": (train / "best" / BEST_WEIGHTS).is_file()}


def _encoder_facts(run_dir: Path) -> dict[str, Any]:
    """B4: train/summary.json of laya_poc.small_encoder (epochs[_run], best_epoch, best_opt_step, stop_reason,
    seconds, n_train_rows_per_epoch), each None when absent."""
    train = run_dir / "train"
    summary = read_json(train / "summary.json")
    summary = summary if isinstance(summary, dict) else {}
    per_epoch = summary.get("n_train_rows_per_epoch") or [None]
    epochs = num(summary.get("epochs_run"))
    return {"epochs_run": epochs if epochs is not None else num(summary.get("epochs")),
            "stop_reason": summary.get("stop_reason"), "best_opt_step": summary.get("best_opt_step"),
            "best_epoch": summary.get("best_epoch"), "train_seconds": num(summary.get("seconds")),
            "n_train": per_epoch[0] if isinstance(per_epoch, list) else None,
            "has_best": (train / "best" / BEST_WEIGHTS).is_file()}


NO_TRAINING = dict.fromkeys(("epochs_run", "stop_reason", "best_opt_step", "train_seconds", "n_train", "has_best"))


def _facts(run_dir: Path, kind: str, zero_shot: bool) -> dict[str, Any]:
    """Training facts: Laya runs (none for zero-shot, as in Phase 3), B4; the other baselines train nothing."""
    if kind == LAYA_KIND:
        return {} if zero_shot else _train_facts(run_dir)
    return _encoder_facts(run_dir) if kind == ENCODER_KIND else dict(NO_TRAINING)


def _baseline_calibration(run_dir: Path) -> dict[str, Any]:
    """Baselines: calibration.json {T, clamped, fitted_on, n, extra} (B1 writes T = 1)."""
    cal = read_json(run_dir / "calibration.json")
    cal = cal if isinstance(cal, dict) else {}
    clamped = cal.get("clamped")
    return {"T": num(cal.get("T")), "clamped": clamped if isinstance(clamped, bool) else None,
            "export_check_passed": None}


def _calibration(run_dir: Path, zero_shot: bool, splits: dict[str, dict], kind: str = LAYA_KIND) -> dict[str, Any]:
    """T: fitted by export_check for a trained run; the shipped (applied) one for zero-shot; calibration.json
    for a baseline."""
    if kind != LAYA_KIND:
        return _baseline_calibration(run_dir)
    if zero_shot:
        return {"T": (splits.get("val") or {}).get("temperature"), "clamped": None, "export_check_passed": None}
    ec = read_json(run_dir / "export_check.json")
    ec = ec if isinstance(ec, dict) else {}
    clamped = ec.get("clamped")
    return {"T": num(ec.get("T")), "clamped": clamped if isinstance(clamped, bool) else None,
            "export_check_passed": ec.get("passed")}


def run_record(run_dir: Path) -> dict[str, Any]:
    done = read_json(run_dir / "done.json")
    done = done if isinstance(done, dict) else {}
    name = done.get("run_name") or run_dir.name
    zero_shot = str(name).endswith("-zs")
    kind = str(done.get(KIND_COLUMN) or LAYA_KIND)
    evals = {p.stem: read_json(p) for p in sorted((run_dir / "eval").glob("*.json"))}
    splits = {k: split_metrics(v) for k, v in evals.items()}
    oi = read_json(run_dir / "order_invariance.json")
    oi = oi if isinstance(oi, dict) else {}
    stripped = evals.get("stripped_test") if isinstance(evals.get("stripped_test"), dict) else {}
    val = splits.get("val") or split_metrics(None)
    rec = {**{k: done.get(k) for k in IDENTITY}, "run_name": name, "done": bool(done), "zero_shot": zero_shot,
           "kind": kind, "eval_subset": any(isinstance(v, dict) and v.get("subset") is True for v in evals.values()),
           **_facts(run_dir, kind, zero_shot), **_calibration(run_dir, zero_shot, splits, kind),
           "val_macro_f1": val["macro_f1"], "val_macro_f1_9": val["macro_f1_9"], "val_acc": val["acc"],
           "val_ece_pre": val["ece_pre"], "val_ece_post": val["ece_post"],
           **{f"{s}_macro_f1": (splits.get(s) or {}).get("macro_f1") for s in ("test_id", *OOD_POOLS)},
           "trap_candidates_acc": (splits.get("trap_candidates") or {}).get("acc"),
           "stripped_false_confident": num(stripped.get("false_confident_rate")),
           "stripped_abstain_rate": num(stripped.get("abstain_rate_at_tau")),
           "order_invariance": num(oi.get("mean_agreement")), "order_invariance_passed": oi.get("passed_99"),
           "run_seconds": num(done.get("seconds")), "splits": splits}
    return rec


def as_roots(runs: Path | Iterable[Path]) -> tuple[Path, ...]:
    """One runs root or several (report --runs-root A --runs-root B)."""
    return (Path(runs),) if isinstance(runs, (str, Path)) else tuple(Path(r) for r in runs)


def done_run_dirs(runs: Path | Iterable[Path]) -> list[Path]:
    """Run directories with a done.json over the roots, sorted by name; a name in several roots: the first."""
    found: dict[str, Path] = {}
    for root in as_roots(runs):
        if root.is_dir():
            for p in root.iterdir():
                if (p / "done.json").is_file():
                    found.setdefault(p.name, p)
    return [found[n] for n in sorted(found)]


def locate_run(runs: Path | Iterable[Path], name: str) -> Path:
    """The directory of run `name`: the first root where it is done, else where it exists, else in the first root."""
    roots = as_roots(runs)
    for test in (lambda d: (d / "done.json").is_file(), Path.exists):
        hit = next((r / name for r in roots if test(r / name)), None)
        if hit is not None:
            return hit
    return roots[0] / name


def cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(round(value, 6))
    return str(value)


def csv_row(rec: dict[str, Any], columns: Sequence[str] = CSV_COLUMNS) -> dict[str, str]:
    return {c: cell(rec.get(c)) for c in columns}


def csv_columns(records: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    """The Phase 3 header, plus `kind` at the end once any row is a baseline."""
    baseline = any((r.get(KIND_COLUMN) or LAYA_KIND) != LAYA_KIND for r in records)
    return (*CSV_COLUMNS, KIND_COLUMN) if baseline else CSV_COLUMNS


def write_runs_csv(records: Sequence[dict[str, Any]], out: Path) -> Path:
    """results/runs.csv from the finished runs' records (stable sort by run name), written atomically."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    columns = csv_columns(records)
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for rec in sorted(records, key=lambda r: str(r["run_name"])):
            writer.writerow(csv_row(rec, columns))
    os.replace(tmp, out)
    return out


def done_records(runs: Path | Iterable[Path]) -> list[dict[str, Any]]:
    return [run_record(p) for p in done_run_dirs(runs)]


def refresh_runs_csv(runs: Path | Iterable[Path], out: Path) -> Path:
    """(Re)write runs.csv from every finished run under the runs root(s)."""
    return write_runs_csv(done_records(runs), out)
