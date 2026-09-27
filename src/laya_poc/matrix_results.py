"""One record per Phase 3 run, read back from its run directory, and results/runs.csv (spec §1: one row per
finished run, fixed header, sorted by run name).

A record is a superset of the CSV row: the report also needs per-split ECE, the order-invariance verdict,
the export_check verdict and whether best/ exists. Missing files and metrics become None, never an error:
the report and the exit check decide what a gap means.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Sequence

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


def num(x: Any) -> float | None:
    """A finite real number as float, else None."""
    ok = isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    return float(x) if ok else None


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


def _calibration(run_dir: Path, zero_shot: bool, splits: dict[str, dict]) -> dict[str, Any]:
    """T: fitted by export_check for a trained run; the shipped (applied) one for zero-shot."""
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
    evals = {p.stem: read_json(p) for p in sorted((run_dir / "eval").glob("*.json"))}
    splits = {k: split_metrics(v) for k, v in evals.items()}
    oi = read_json(run_dir / "order_invariance.json")
    oi = oi if isinstance(oi, dict) else {}
    stripped = evals.get("stripped_test") if isinstance(evals.get("stripped_test"), dict) else {}
    val = splits.get("val") or split_metrics(None)
    rec = {**{k: done.get(k) for k in IDENTITY}, "run_name": name, "done": bool(done), "zero_shot": zero_shot,
           **({} if zero_shot else _train_facts(run_dir)), **_calibration(run_dir, zero_shot, splits),
           "val_macro_f1": val["macro_f1"], "val_macro_f1_9": val["macro_f1_9"], "val_acc": val["acc"],
           "val_ece_pre": val["ece_pre"], "val_ece_post": val["ece_post"],
           **{f"{s}_macro_f1": (splits.get(s) or {}).get("macro_f1") for s in ("test_id", *OOD_POOLS)},
           "trap_candidates_acc": (splits.get("trap_candidates") or {}).get("acc"),
           "stripped_false_confident": num(stripped.get("false_confident_rate")),
           "stripped_abstain_rate": num(stripped.get("abstain_rate_at_tau")),
           "order_invariance": num(oi.get("mean_agreement")), "order_invariance_passed": oi.get("passed_99"),
           "run_seconds": num(done.get("seconds")), "splits": splits}
    return rec


def done_run_dirs(runs_dir: Path) -> list[Path]:
    """Run directories with a done.json, sorted by name."""
    if not runs_dir.is_dir():
        return []
    return sorted((p for p in runs_dir.iterdir() if (p / "done.json").is_file()), key=lambda p: p.name)


def cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(round(value, 6))
    return str(value)


def csv_row(rec: dict[str, Any]) -> dict[str, str]:
    return {c: cell(rec.get(c)) for c in CSV_COLUMNS}


def write_runs_csv(records: Sequence[dict[str, Any]], out: Path) -> Path:
    """results/runs.csv from the finished runs' records (stable sort by run name), written atomically."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for rec in sorted(records, key=lambda r: str(r["run_name"])):
            writer.writerow(csv_row(rec))
    os.replace(tmp, out)
    return out


def done_records(runs_dir: Path) -> list[dict[str, Any]]:
    return [run_record(p) for p in done_run_dirs(runs_dir)]


def refresh_runs_csv(runs_dir: Path, out: Path) -> Path:
    """(Re)write runs.csv from every finished run under runs_dir."""
    return write_runs_csv(done_records(runs_dir), out)
