"""Phase 3 report (design doc §5.11, §5.12, §6.2 Phase 3; spec §5): results/phase3_report.md and .json.

Runs are grouped by arm, model, scheme, train subset and head-only; each metric is the mean over the group's
seeds with its min..max range. Post-T numbers unless marked pre. The seed-variance flag is the range over seeds
of a macro-F1 above 3 points (design §5.12 "Investigate"). Trap accuracy is on the UNANNOTATED candidates
(annotation pending): it is not the §5.11 trap metric. The exit check is the design's Phase 3 exit: every
configured run has a best/ checkpoint (trained runs), a temperature and val metrics in results/runs.csv.
Pure functions over run records (matrix_results.run_record) and run specs; nothing here loads a model.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .matrix_exec import write_json
from .matrix_plan import RunSpec
from .matrix_report_md import render_markdown
from .matrix_results import BEST_WEIGHTS, OOD_POOLS, done_records, num

SEED_VARIANCE_POINTS = 0.03       # design §5.12: seed variance > 3 macro-F1 points -> Investigate
EPS = 1e-9                        # float slack: 0.63 - 0.60 is exactly 3 points, not above them
F1_METRICS = ("val_macro_f1", "test_id_macro_f1", *(f"{p}_macro_f1" for p in OOD_POOLS))
EXIT_CRITERION = ("every configured run has a best/ checkpoint (trained runs), a temperature and val metrics in "
                  "results/runs.csv (design §6.2 Phase 3 exit)")


def _split(split: str, key: str) -> Callable[[dict], float | None]:
    return lambda r: ((r.get("splits") or {}).get(split) or {}).get(key)


def _gap(pool: str) -> Callable[[dict], float | None]:
    def gap(r: dict) -> float | None:
        a, b = num(r.get("test_id_macro_f1")), num(r.get(f"{pool}_macro_f1"))
        return None if a is None or b is None else a - b
    return gap


METRICS: dict[str, Callable[[dict], Any]] = {
    **{k: (lambda key: lambda r: r.get(key))(k) for k in (*F1_METRICS, "val_macro_f1_9", "T", "val_ece_pre",
                                                         "val_ece_post", "trap_candidates_acc",
                                                         "stripped_false_confident", "stripped_abstain_rate",
                                                         "order_invariance", "train_seconds")},
    "test_id_ece_pre": _split("test_id", "ece_pre"), "test_id_ece_post": _split("test_id", "ece_post"),
    **{f"gap_{p}": _gap(p) for p in OOD_POOLS},
}


# ---------------------------------------------------------------- aggregation

def stat(values: Sequence[Any]) -> dict[str, float] | None:
    """{mean, min, max, n} over the finite values; None when there are none."""
    xs = [v for v in (num(x) for x in values) if v is not None]
    return {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "n": len(xs)} if xs else None


def group_key(rec: dict) -> tuple:
    return (rec.get("arm"), rec.get("model"), rec.get("scheme"), rec.get("subset"), bool(rec.get("head_only")))


def group_label(key: tuple) -> str:
    arm, model, scheme, subset, head = key
    extra = " (zero-shot, shipped T)" if arm == "B2" else (f" n={subset}" if subset else "") + \
        (" head-only" if head else "")
    return f"{arm} {model} {scheme}{extra}"


def _order(key: tuple) -> tuple:
    arm, model, scheme, subset, head = key
    return (arm != "B2", str(arm), str(model), str(scheme), subset or 0, head)


def group_rows(records: Sequence[dict]) -> list[dict]:
    keys = sorted({group_key(r) for r in records}, key=_order)
    rows = []
    for key in keys:
        recs = sorted((r for r in records if group_key(r) == key), key=lambda r: str(r["run_name"]))
        oi = [r.get("order_invariance_passed") for r in recs]
        rows.append({"label": group_label(key), "arm": key[0], "model": key[1], "scheme": key[2], "subset": key[3],
                     "head_only": key[4], "zero_shot": key[0] == "B2", "n_runs": len(recs),
                     "runs": [r["run_name"] for r in recs], "seeds": [r.get("seed") for r in recs],
                     "metrics": {m: stat([f(r) for r in recs]) for m, f in METRICS.items()},
                     "order_invariance_all_99": None if any(v is None for v in oi) else all(oi),
                     "export_check_failed": [r["run_name"] for r in recs if r.get("export_check_passed") is False]})
    return rows


def seed_variance(groups: Sequence[dict], threshold: float = SEED_VARIANCE_POINTS) -> list[dict]:
    """Range over seeds of every macro-F1 of each multi-seed group; flagged above `threshold`."""
    out = []
    for g in (g for g in groups if g["n_runs"] > 1):
        for m in F1_METRICS:
            s = g["metrics"].get(m)
            if s and s["n"] > 1:
                rng = s["max"] - s["min"]
                out.append({"group": g["label"], "metric": m, "range": rng, "n": s["n"],
                            "flagged": rng > threshold + EPS})
    return out


def _point(curve: str, r: dict, n: Any, full: bool) -> dict:
    return {"curve": curve, "n": n, "run_name": r["run_name"], "full": full,
            "val_macro_f1": r.get("val_macro_f1"), "test_id_macro_f1": r.get("test_id_macro_f1")}


def learning_curve(records: Sequence[dict]) -> list[dict]:
    """E6: the train-subset runs by n, plus the full-size trained run of the same model, scheme and seed (the
    E2 seed-11 run), whose n is its train rows per epoch."""
    points = []
    for arm, model, scheme, seed in sorted({(r["arm"], r["model"], r["scheme"], r["seed"])
                                            for r in records if r.get("subset")}, key=str):
        name = f"{arm} {model} {scheme} s{seed}"
        same = sorted((r for r in records if (r.get("model"), r.get("scheme"), r.get("seed")) == (model, scheme, seed)),
                      key=lambda r: str(r["run_name"]))
        curve = [_point(name, r, r["subset"], False) for r in same if r.get("subset") and r.get("arm") == arm]
        full = next((r for r in same if not (r.get("subset") or r.get("head_only") or r.get("zero_shot"))), None)
        curve += [_point(name, full, full.get("n_train"), True)] if full else []
        points += sorted(curve, key=lambda p: (p["n"] is None, p["n"] or 0))
    return points


# ---------------------------------------------------------------- exit check

def read_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["run_name"]: row for row in csv.DictReader(fh)}


def exit_check(specs: Sequence[RunSpec], runs_dir: Path, runs_csv: Path) -> dict[str, Any]:
    """Per configured run: done.json, best/ weights (trained runs), T and val macro-F1 in runs.csv."""
    rows, runs = read_csv_rows(runs_csv), []
    for spec in specs:
        rd, row = runs_dir / spec.name, rows.get(spec.name) or {}
        checks = (("done", "done.json", (rd / "done.json").is_file()),
                  ("best", "best/", None if spec.zero_shot else (rd / "train" / "best" / BEST_WEIGHTS).is_file()),
                  ("T", "T", bool(row.get("T"))), ("val_metrics", "val metrics", bool(row.get("val_macro_f1"))))
        missing = [label for _, label, ok in checks if ok is False]  # None: not applicable (zero-shot best/)
        runs.append({"run_name": spec.name, **{key: ok for key, _, ok in checks}, "missing": missing,
                     "passed": not missing})
    complete = sum(r["passed"] for r in runs)
    ok = bool(runs) and complete == len(runs)
    return {"verdict": "PASS" if ok else "FAIL", "passed": ok, "complete": complete, "total": len(runs),
            "criterion": EXIT_CRITERION, "runs": runs}


def build_report(cfg: dict, specs: Sequence[RunSpec], runs_dir: Path, runs_csv: Path) -> dict[str, Any]:
    records = done_records(runs_dir)
    groups = group_rows(records)
    ex = exit_check(specs, runs_dir, runs_csv)
    p3 = cfg.get("phase3") or {}
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cards": sorted({str(r.get("card")) for r in records if r.get("card")}),
            "phase3": {k: p3.get(k) for k in ("card", "eval_splits", "order_invariance", "evals_per_run")},
            "configured": [s.name for s in specs], "runs": records, "groups": groups,
            "seed_variance": seed_variance(groups), "learning_curve": learning_curve(records),
            "seed_variance_threshold": SEED_VARIANCE_POINTS, "exit_check": ex, "verdict": ex["verdict"],
            "runs_csv": str(runs_csv)}


def write_report(result: dict[str, Any], out_md: Path) -> tuple[Path, Path]:
    """<out>.md and <out>.json (the full result: records, groups, curve, exit check), each written atomically."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_md.with_name(out_md.name + ".tmp")
    tmp.write_text(render_markdown(result), encoding="utf-8")
    tmp.replace(out_md)
    return out_md, write_json(out_md.with_suffix(".json"), result)
