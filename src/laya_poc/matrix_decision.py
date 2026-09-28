"""The Phase 4 parts of the matrix report (spec P4 §5): decision criterion 1 as an early read, and the Phase 4 exit.

Decision criterion 1 (design §5.12): on OOD, a fine-tuned Laya checkpoint (mean over its seeds) must beat BOTH
char TF-IDF+LR (B3) and the fine-tuned small encoder (B4) by >= 3 macro-F1 points, averaged over the OOD pools;
`laya` cannot read Thai by design, so its ood_script pool is excluded. Each Laya arm is compared with the best
seed-mean B3/B4 group of the same label scheme, averaged over the SAME pools (beating the best beats both); a
scheme without B3 or B4 (B4 is c10 only) is marked incomplete. Candidates are the full-size,
fully fine-tuned Laya arms: zero-shot (B2), head-only (E4) and train subsets (E6) are not. It is information
only: Phase 5 applies the decision rule with every criterion and the bootstrap intervals.

Phase 4 exit (design §6.2): every configured baseline run (phase4.arms) has predictions for every eval split in
preds/; B5 scores subsets of each split, still one preds file per split; an optional run (B5) that was not run is
not a failure. B2 ran in Phase 3. Pure functions over run records, specs and the files of the run directories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from .matrix_baselines import required_splits
from .matrix_plan import LAYA, RunSpec
from .matrix_results import OOD_POOLS, as_roots, locate_run, num, stat

NO_THAI_MODELS = ("laya",)        # design §5.12 criterion 1: laya cannot read Thai; its ood_script is excluded
LEAD = 0.03                       # >= 3 macro-F1 points over the best trained baseline
EPS = 1e-9                        # float slack: 0.53 - 0.50 is exactly 3 points
TRAINED_BASELINES = {"tfidf_lr": "B3", "small_encoder": "B4"}
EXIT_CRITERION = ("every configured baseline run (phase4.arms) has predictions for every eval split in preds/ "
                  "(B5: its evaluation subsets); an optional run that was not run is not a failure (design §6.2 "
                  "Phase 4 exit). B2 (zero-shot Laya) ran in Phase 3")


# ---------------------------------------------------------------- decision criterion 1

def ood_pools(model: Any) -> tuple[str, ...]:
    return tuple(p for p in OOD_POOLS if not (p == "ood_script" and model in NO_THAI_MODELS))


def ood_average(rec: dict, pools: Sequence[str]) -> float | None:
    """Mean macro-F1 over the pools; None unless every pool has a number."""
    xs = [num(rec.get(f"{p}_macro_f1")) for p in pools]
    return None if not xs or any(x is None for x in xs) else sum(xs) / len(xs)


def is_candidate(rec: dict) -> bool:
    return ((rec.get("kind") or LAYA) == LAYA and not rec.get("zero_shot") and not rec.get("head_only")
            and not rec.get("subset"))


def _grouped(records: Iterable[dict], keep) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in sorted(records, key=lambda r: str(r["run_name"])):
        if keep(r):
            out.setdefault((str(r.get("arm")), str(r.get("model")), str(r.get("scheme"))), []).append(r)
    return dict(sorted(out.items()))


def _baselines(records: Sequence[dict], scheme: str, pools: Sequence[str]) -> list[dict]:
    groups = _grouped(records, lambda r: r.get("kind") in TRAINED_BASELINES and r.get("scheme") == scheme)
    return [{"label": " ".join(key), "arm": key[0], "model": key[1], "kind": recs[0]["kind"],
             "seeds": [r.get("seed") for r in recs], "ood_avg": stat([ood_average(r, pools) for r in recs])}
            for key, recs in groups.items()]


def _criterion_row(key: tuple, recs: Sequence[dict], records: Sequence[dict]) -> dict:
    arm, model, scheme = key
    pools = ood_pools(model)
    laya = stat([ood_average(r, pools) for r in recs])
    bases = _baselines(records, scheme, pools)
    scored = [b for b in bases if b["ood_avg"]]
    best = max(scored, key=lambda b: b["ood_avg"]["mean"], default=None)
    lead = None if laya is None or best is None else laya["mean"] - best["ood_avg"]["mean"]
    return {"label": " ".join(key), "arm": arm, "model": model, "scheme": scheme,
            "seeds": [r.get("seed") for r in recs], "pools": list(pools), "laya_ood_avg": laya, "baselines": bases,
            "best": best["label"] if best else None, "best_ood_avg": best["ood_avg"]["mean"] if best else None,
            "lead": lead, "passed": None if lead is None else lead >= LEAD - EPS,
            "missing": [TRAINED_BASELINES[k] for k in TRAINED_BASELINES if not any(b["kind"] == k for b in scored)]}


def decision_criterion_1(records: Sequence[dict]) -> list[dict]:
    """One row per fine-tuned Laya arm (arm, model, scheme): its seed-mean OOD average vs the best of B3/B4."""
    return [_criterion_row(key, recs, records) for key, recs in _grouped(records, is_candidate).items()]


# ---------------------------------------------------------------- Phase 4 exit

def _exit_row(spec: RunSpec, run_dir: Path, splits: Sequence[str]) -> dict:
    done = (run_dir / "done.json").is_file()
    absent = [s for s in required_splits(spec, splits) if not (run_dir / "preds" / f"{s}.jsonl").is_file()]
    skipped = spec.optional and not done
    missing = [] if skipped else ([] if done else ["done.json"]) + [f"preds/{s}.jsonl" for s in absent]
    return {"run_name": spec.name, "kind": spec.kind, "optional": spec.optional, "done": done, "preds": not absent,
            "skipped": skipped, "missing": missing, "passed": not missing}


def phase4_exit_check(specs: Sequence[RunSpec], runs: Path | Iterable[Path], splits: Sequence[str]) -> dict | None:
    """None when no baseline is configured; else the verdict over every configured baseline run."""
    base = [s for s in specs if s.kind != LAYA]
    if not base:
        return None
    roots = as_roots(runs)
    from .matrix_report import not_run, phase_absent  # lazy: matrix_report imports this module
    if phase_absent(base, roots):
        return {**not_run(base, EXIT_CRITERION), "skipped_optional": [], "splits": list(splits)}
    rows = [_exit_row(s, locate_run(roots, s.name), splits) for s in base]
    ran = [r for r in rows if not r["skipped"]]
    ok = bool(ran) and all(r["passed"] for r in rows)
    return {"verdict": "PASS" if ok else "FAIL", "passed": ok, "complete": sum(r["passed"] for r in ran),
            "total": len(rows), "skipped_optional": [r["run_name"] for r in rows if r["skipped"]],
            "splits": list(splits), "criterion": EXIT_CRITERION, "runs": rows}
