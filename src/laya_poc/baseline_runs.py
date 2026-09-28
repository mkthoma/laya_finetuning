"""B1 (majority / prior) and B3 (char TF-IDF + LR) as Phase 4 runs in the Phase 3 run layout (spec §1).

    python -m laya_poc.baseline_runs {majority,prior,tfidf_lr} [--scheme c10|c7] --data-dir D --run-dir R
        [--extra name=<abs jsonl> ...] [--splits s ...] [--optional s ...] [--order-invariance] [--n-jobs N]
        [--config yaml]

Fits on D/train.jsonl, the clean train split (build_data FULL mode; `variants c7` for the c7 dir), labelled rows
only; scores EVERY row of every split (config phase3.eval_splits unless --splits; --extra for split files
outside D, e.g. trap_candidates) and writes through baseline_eval, i.e. in the evaluate multi-split schema:
  R/eval/<split>.json  R/preds/<split>.jsonl  R/preds/no_gate/stripped_test.jsonl  R/calibration.json
  R/order_invariance.json   tfidf_lr with --order-invariance (config phase3.order_invariance)
The no-evidence gate applies as in evaluate; tau is chosen on val. done.json is the matrix's.
- majority / prior (B1): the train majority class (ties: first option) / the train class prior, as
  log-probabilities floored at log(1e-12); T = 1, not fitted. No order invariance (the input is ignored).
- tfidf_lr (B3): baselines.py B3 through its own functions: TfidfVectorizer on the state strings, one LR per
  config baselines.lr.C_grid value (parallel, one BLAS thread each: bit-identical whatever --n-jobs), C by
  macro-F1 on the labelled val rows (first grid value on ties), T fitted on the chosen C's val log-probabilities
  (baseline_eval.fit_T); the chosen C's model (each grid task returns it) scores every row with predict_log_proba.
  D/val.jsonl is always read (C and T). `python -m laya_poc.baselines` (Phase 2) is unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import baseline_eval as BE
from . import baselines as B
from . import labels as L
from .config import load_config
from .env_check import run_cli, write_json
from .evaluate import check_question, label_indices, load_rows
from .evaluate_splits import TAU_SPLIT, parse_extras, split_paths
from .metrics import macro_f1

KINDS = ("majority", "prior", "tfidf_lr")
B1_KINDS = ("majority", "prior")
TRAIN_FILE = "train.jsonl"
GRID_KEYS = ("C", "val_macro_f1", "n_iter", "converged", "seconds")   # baselines.py's grid report


@dataclass(frozen=True)
class Plan:
    """Everything a run needs, resolved and validated before any fit."""
    kind: str
    scheme: str
    keys: list[str]
    data_dir: Path
    run_dir: Path
    paths: dict[str, Path]          # split -> file, val first
    skipped: list[str]              # optional splits whose file is absent
    rows: dict[str, list[dict]]     # split -> rows (question checked)
    order: dict[str, Any] | None    # {split, n, perms, seed, path, rows} when order invariance runs
    evidence_fields: list[str]
    abstain: dict[str, Any]


@dataclass(frozen=True)
class Fitted:
    """A fitted baseline: scores for state strings, its temperature and calibration.json facts."""
    score: Callable[[Sequence[str]], np.ndarray]    # [N, K] log-probabilities in option-key order
    T: float
    clamped: bool
    n_calibration: int
    fitted_on: str | None
    extra: dict[str, Any]


# ---------------------------------------------------------------- inputs

def load_checked(path: Path, question: dict) -> list[dict]:
    """Rows of a split file carrying this scheme's question (evaluate.check_question)."""
    rows = load_rows(path)
    check_question(rows, question)
    return rows


def labelled(rows: Sequence[dict], keys: Sequence[str]) -> tuple[list[str], np.ndarray, int]:
    """(states, option indices, n unlabelled) of the labelled rows; a label outside `keys` is an error."""
    y = label_indices(rows, keys)
    keep = [(r["state"], v) for r, v in zip(rows, y) if v is not None]
    if not keep:
        raise ValueError(f"no labelled rows (split {rows[0].get('split')!r})")
    return [s for s, _ in keep], np.array([v for _, v in keep], dtype=int), len(rows) - len(keep)


def _order_plan(cfg: dict, rows: dict[str, list[dict]], paths: dict[str, Path], extras: dict[str, Path],
                data_dir: Path, question: dict) -> dict[str, Any]:
    """config phase3.order_invariance + its split's rows (an eval split's, or read from its file)."""
    oc = BE.order_config(cfg)
    path = paths.get(oc["split"]) or extras.get(oc["split"]) or data_dir / f"{oc['split']}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"order-invariance split {oc['split']}: {path} not found")
    return {**oc, "path": path, "rows": rows.get(oc["split"]) or load_checked(path, question)}


def make_plan(args: argparse.Namespace, cfg: dict) -> Plan:
    scheme = args.scheme or cfg["labels"]["scheme"]
    question, data_dir = L.question(scheme), Path(args.data_dir)
    extras = parse_extras(args.extra)
    splits = args.splits or list(cfg["phase3"]["eval_splits"])
    paths, skipped = split_paths(splits, data_dir, extras, args.optional or [])
    rows = {s: load_checked(p, question) for s, p in paths.items()}
    order = None
    if args.order_invariance and args.kind == "tfidf_lr":
        order = _order_plan(cfg, rows, paths, extras, data_dir, question)
    return Plan(kind=args.kind, scheme=scheme, keys=L.option_keys(scheme), data_dir=data_dir,
                run_dir=Path(args.run_dir), paths=paths, skipped=skipped, rows=rows, order=order,
                evidence_fields=list(cfg["serialise"]["evidence_fields"]), abstain=dict(cfg.get("abstain") or {}))


def train_split(plan: Plan) -> tuple[list[str], np.ndarray, int]:
    path = plan.data_dir / TRAIN_FILE
    if not path.exists():
        raise FileNotFoundError(f"no {TRAIN_FILE} in {plan.data_dir} (build_data writes it in FULL mode: the clean, "
                                f"unaugmented train split; `variants c7` writes the c7 one)")
    return labelled(load_checked(path, L.question(plan.scheme)), plan.keys)


# ---------------------------------------------------------------- B1

def fit_b1(kind: str, y_train: np.ndarray, keys: Sequence[str]) -> Fitted:
    """The same scores for every row: the one-hot majority or the class prior, floored at log(1e-12)."""
    k = len(keys)
    prior = B.class_prior(y_train, k)
    if kind == "majority":
        row = np.full(k, B.LOG_FLOOR)
        row[B.majority_index(y_train, k)] = 0.0
    else:
        row = np.log(np.maximum(prior, np.exp(B.LOG_FLOOR)))
    extra = {"kind": kind, "majority_class": keys[B.majority_index(y_train, k)],
             "prior": {c: float(p) for c, p in zip(keys, prior)}, "note": "T = 1: B1 has nothing to temper"}
    return Fitted(score=lambda states: np.tile(row, (len(states), 1)), T=1.0, clamped=False, n_calibration=0,
                  fitted_on=None, extra=extra)


# ---------------------------------------------------------------- B3

def fit_c(X_train: Any, y_train: np.ndarray, X_val: Any, y_val: np.ndarray, C: float, lr_cfg: dict[str, Any],
          k: int) -> dict[str, Any]:
    """baselines.fit_one for one C (a joblib task: one BLAS thread, convergence warnings recorded as
    converged=False), returning the fitted model too, so the chosen C scores the splits without a refit."""
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    t0 = time.perf_counter()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = B.make_model(C, lr_cfg).fit(X_train, y_train)
        logp = B.embed_log_proba(model.predict_log_proba(X_val), model.classes_, k)
    n_iter = int(np.max(model.n_iter_))
    return {"C": C, "val_macro_f1": macro_f1(y_val, logp.argmax(1)), "n_iter": n_iter,
            "converged": n_iter < model.max_iter, "seconds": round(time.perf_counter() - t0, 2),
            "logp": {B.SELECT_SPLIT: logp}, "model": model}


def fit_grid(cfg: dict, X_train: Any, y_train: np.ndarray, X_val: Any, y_val: np.ndarray, k: int,
             n_jobs: int | None) -> tuple[list[dict], int]:
    """fit_c for every C of config baselines.lr.C_grid, in grid order (processes when n_jobs > 1)."""
    from joblib import Parallel, delayed

    lr_cfg = cfg["baselines"]["lr"]
    grid = list(lr_cfg["C_grid"])
    jobs = max(1, n_jobs or min(len(grid), os.cpu_count() or 1, B.MAX_DEFAULT_JOBS))
    print(f"baseline_runs: fitting {len(grid)} LR models (C grid {grid}) on {jobs} worker(s)", flush=True)
    fits = Parallel(n_jobs=jobs)(delayed(fit_c)(X_train, y_train, X_val, y_val, C, lr_cfg, k) for C in grid)
    return fits, jobs


def vectorise(cfg: dict, train_states: list[str], val_states: list[str]) -> tuple[Any, Any, Any, dict]:
    """(vectorizer, X_train, X_val, info): baselines.make_vectorizer fitted on the train states, as B3 does."""
    t0 = time.perf_counter()
    vec = B.make_vectorizer(cfg["baselines"]["tfidf"])
    X_train = vec.fit_transform(train_states)
    X_val = vec.transform(val_states)
    info = {"n_features": len(vec.vocabulary_), "nnz_train": int(X_train.nnz),
            "seconds": round(time.perf_counter() - t0, 2)}
    print(f"baseline_runs: TF-IDF on {X_train.shape[0]} train rows -> {info['n_features']} features "
          f"({info['seconds']:.0f}s)", flush=True)
    return vec, X_train, X_val, info


def fit_b3(cfg: dict, train: tuple[list[str], np.ndarray, int], val: tuple[list[str], np.ndarray, int],
           keys: Sequence[str], n_jobs: int | None) -> Fitted:
    """C on val macro-F1 (baselines.select_best), T on its val log-probabilities; its model scores the rows."""
    (train_states, y_train, n_unl), (val_states, y_val, _) = train, val
    vec, X_train, X_val, tfidf = vectorise(cfg, train_states, val_states)
    t0, k = time.perf_counter(), len(keys)
    grid, jobs = fit_grid(cfg, X_train, y_train, X_val, y_val, k, n_jobs)
    best = B.select_best(grid)
    model = best["model"]

    def score(states: Sequence[str]) -> np.ndarray:
        return B.embed_log_proba(model.predict_log_proba(vec.transform(list(states))), model.classes_, k)

    val_logp = best["logp"][B.SELECT_SPLIT]
    T, clamped = BE.fit_T(val_logp, y_val)
    extra = {"kind": "tfidf_lr", "C": best["C"], "grid": [{g: r[g] for g in GRID_KEYS} for r in grid],
             "tfidf": tfidf, "solver": model.solver, "random_state": B.RANDOM_STATE, "n_jobs": jobs,
             "n_train": len(train_states), "n_train_unlabelled": n_unl,
             "seconds_fit": round(time.perf_counter() - t0, 2)}
    return Fitted(score=score, T=T, clamped=clamped, n_calibration=len(y_val), fitted_on=TAU_SPLIT, extra=extra)


def fit(plan: Plan, cfg: dict, n_jobs: int | None) -> Fitted:
    train = train_split(plan)
    if plan.kind in B1_KINDS:
        return fit_b1(plan.kind, train[1], plan.keys)
    val_path = plan.paths.get(TAU_SPLIT, plan.data_dir / f"{TAU_SPLIT}.jsonl")
    if not val_path.exists():
        raise FileNotFoundError(f"tfidf_lr needs {val_path} (C and T are chosen on val)")
    val_rows = plan.rows.get(TAU_SPLIT) or load_checked(val_path, L.question(plan.scheme))
    return fit_b3(cfg, train, labelled(val_rows, plan.keys), plan.keys, n_jobs)


# ---------------------------------------------------------------- outputs

def _line(kind: str, scheme: str, res: dict) -> str:
    def f(side: str, k: str) -> str:
        v = (res.get(side) or {}).get(k)
        return "n/a" if v is None else f"{v:.4f}"
    return (f"baseline_runs {kind} {scheme} {res['split']}: macro-F1 {f('post', 'macro_f1')} (9-class "
            f"{f('post', 'macro_f1_9')}), acc {f('post', 'acc')}, ECE {f('pre', 'ece')}->{f('post', 'ece')} at "
            f"T={res['temperature']:.4f}; {res['n_scored']} scored, {res['n_abstained_no_evidence']} abstained "
            f"(no evidence)")


def write_splits(plan: Plan, fitted: Fitted) -> dict[str, dict]:
    """Every split through baseline_eval (val first: its tau is applied to the others)."""
    tau, out = None, {}
    if TAU_SPLIT not in plan.paths:
        print(f"baseline_runs: WARNING: no tau ({TAU_SPLIT} not evaluated): abstention blocks carry tau null",
              flush=True)
    for split, path in plan.paths.items():
        rows = plan.rows[split]
        t0 = time.perf_counter()
        scores = fitted.score([r["state"] for r in rows])
        out[split] = BE.write_outputs(
            plan.run_dir / BE.EVAL_DIR, plan.run_dir / BE.PREDS_DIR, rows=rows, keys=plan.keys, scores=scores,
            T=fitted.T, split=split, model=plan.kind, ckpt=str(plan.data_dir / TRAIN_FILE),
            evidence_fields=plan.evidence_fields, abstain_cfg=plan.abstain, tau=tau,
            meta={"rows": str(path), "seconds_post": round(time.perf_counter() - t0, 2)})
        tau = out[split]["tau"] if split == TAU_SPLIT else tau
        print(_line(plan.kind, plan.scheme, out[split]), flush=True)
    return out


def write_order(plan: Plan, fitted: Fitted) -> dict[str, Any]:
    o = plan.order
    res = BE.order_invariance_payload(lambda states: fitted.score(states).argmax(1), o["rows"], model=plan.kind,
                                      ckpt=str(plan.data_dir / TRAIN_FILE), split=o["split"], rows_path=str(o["path"]),
                                      evidence_fields=plan.evidence_fields, n=int(o["n"]), perms=int(o["perms"]),
                                      seed=int(o["seed"]))
    write_json(plan.run_dir / BE.ORDER_FILE, res)
    print(f"baseline_runs: order invariance on {o['split']}: mean agreement {res['mean_agreement']:.4f} over "
          f"{res['perms']} orders x {res['n']} rows ({'PASS' if res['passed_99'] else 'FAIL'} >= 0.99)", flush=True)
    return res


# ---------------------------------------------------------------- CLI

def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, t0 = load_config(args.config), time.perf_counter()
    plan = make_plan(args, cfg)
    absent = f" (optional, absent: {', '.join(plan.skipped)})" if plan.skipped else ""
    print(f"baseline_runs {plan.kind} {plan.scheme}: {', '.join(plan.paths)}{absent} -> {plan.run_dir}", flush=True)
    if args.order_invariance and plan.kind in B1_KINDS:
        print(f"baseline_runs: order invariance skipped for {plan.kind} (its output ignores the input)", flush=True)
    fitted = fit(plan, cfg, args.n_jobs)
    if plan.kind == "tfidf_lr":
        print(f"baseline_runs: C {fitted.extra['C']}, T {fitted.T:.4f}{' (CLAMPED)' if fitted.clamped else ''} "
              f"on {fitted.n_calibration} labelled val rows ({fitted.extra['seconds_fit']:.0f}s)", flush=True)
    splits = write_splits(plan, fitted)
    write_json(plan.run_dir / BE.CALIBRATION_FILE,
               BE.calibration_record(T=fitted.T, clamped=fitted.clamped, n=fitted.n_calibration,
                                     fitted_on=fitted.fitted_on, extra={**fitted.extra, "scheme": plan.scheme}))
    order = write_order(plan, fitted) if plan.order is not None else None
    return {"kind": plan.kind, "scheme": plan.scheme, "splits": list(splits), "skipped": plan.skipped,
            "T": fitted.T, "order_invariance": None if order is None else order["mean_agreement"],
            "seconds": round(time.perf_counter() - t0, 2)}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.baseline_runs", description=__doc__.splitlines()[0])
    p.add_argument("kind", choices=KINDS)
    p.add_argument("--scheme", choices=sorted(L.SCHEMES), default=None, help="default: config labels.scheme")
    p.add_argument("--data-dir", required=True, help="train.jsonl, val.jsonl and the split files of the scheme")
    p.add_argument("--run-dir", required=True, help="writes eval/, preds/, calibration.json (, order_invariance.json)")
    p.add_argument("--extra", nargs="+", action="extend", default=None, metavar="NAME=ABS_PATH",
                   help="split files outside --data-dir (e.g. trap_candidates=/abs/trap_candidates.jsonl)")
    p.add_argument("--splits", nargs="+", default=None, help="default: config phase3.eval_splits")
    p.add_argument("--optional", nargs="+", action="extend", default=None, help="splits allowed to be missing")
    p.add_argument("--order-invariance", action="store_true",
                   help="tfidf_lr: field-order invariance per config phase3.order_invariance")
    p.add_argument("--n-jobs", type=int, default=None,
                   help=f"parallel LR fits (default: min(grid, CPUs, {B.MAX_DEFAULT_JOBS}))")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    if args.n_jobs is not None and args.n_jobs < 1:
        p.error("--n-jobs must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        res = run(args)
        print(f"baseline_runs: wrote {len(res['splits'])} splits to {args.run_dir} ({res['seconds']:.0f}s)")
        return 0

    return run_cli("baseline_runs", body, Path(args.run_dir) / "baseline_runs")


if __name__ == "__main__":
    sys.exit(main())
