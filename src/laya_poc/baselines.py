"""Trivial and char-TF-IDF baselines on the frozen splits (design doc §5.13 B1 and B3, §7.10).

B1: the train majority class (P one-hot) and the train class prior (P = prior for every row).
B3: TfidfVectorizer(char_wb 2-5) on the compact JSON state STRINGS the model sees, then
LogisticRegression(class_weight="balanced") with C picked from config baselines.lr.C_grid by val macro-F1
(first grid value on ties), then one temperature fitted on val (calibrate.fit_temperature on
predict_log_proba) and reported with is_clamped. Every arm is scored pre and post T on each requested
split with evaluate.split_report (macro_f1_9 excludes `event`). The 7-class variant collapses the row
labels with labels.C7_MAP and trains its own LR. Val is always read: it selects C and fits T.

Trained on <data-dir>/train.jsonl: the clean train split (alphabetical keys, no augmentation, true
labels), which build_data writes in FULL mode; null-label rows are skipped and counted.

Solver: saga (config baselines.lr.solver overrides). TfidfVectorizer rows are L2-normalised, which is the
case SAGA's fixed step size is made for: measured on 25k synthetic multi-script states (187k features,
8.6M non-zeros) it reached the same val macro-F1 as lbfgs (+-0.001) 2.6x faster at C=0.5 and 2.1x at C=8
(18/49 epochs vs 50/121 lbfgs iterations). It is seeded (RANDOM_STATE) and every fit runs with one BLAS
thread, so the output is bit-identical run to run and whatever n_jobs is. n_jobs parallelises the
independent (scheme, C) fits over processes (joblib/loky).

    python -m laya_poc.baselines --data-dir <dir> --out <json> [--preds-dir <dir>] [--splits val test_id]
                                 [--schemes c10 c7] [--n-jobs N] [--config yaml]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import labels as L
from .calibrate import apply_temperature, fit_temperature, is_clamped
from .config import load_config
from .env_check import run_cli, write_json
from .evaluate import split_report
from .io_utils import iter_jsonl, write_jsonl
from .metrics import macro_f1

RANDOM_STATE = 0
DEFAULT_SOLVER = "saga"  # see the module docstring
LOG_FLOOR = float(np.log(1e-12))  # log-probability of a class absent from train (calibrate uses the same floor)
SELECT_SPLIT = "val"
TRAIN_SPLIT = "train"
B3 = "b3_tfidf_lr"
# Cap the default worker count: each process holds its own coefficient/gradient buffers (classes x features;
# lbfgs keeps ~20 of them, ~0.8 GB at 500k features). Colab: 2 vCPUs -> 2 workers; --n-jobs overrides.
MAX_DEFAULT_JOBS = 4
_GRID_KEYS = ("C", "val_macro_f1", "n_iter", "converged", "seconds")
_HINTS = {TRAIN_SPLIT: " (build_data writes it in FULL mode: the clean, unaugmented train split)"}


# ---------------------------------------------------------------- data

def load_split(data_dir: Path, name: str) -> dict[str, Any]:
    """Labelled rows of <data_dir>/<name>.jsonl: ids, state strings and their c10 label keys."""
    path = data_dir / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no {path.name} in {data_dir}{_HINTS.get(name, '')}")
    rows = list(iter_jsonl(path))
    lab = [r for r in rows if r.get("label") is not None]
    bad = sorted({r["label"] for r in lab} - set(L.option_keys("c10")))
    if bad:
        raise ValueError(f"{path.name}: labels {bad} are not c10 keys (baselines collapse c10 labels to c7)")
    if not lab:
        raise ValueError(f"{path.name} has no labelled rows")
    return {"ids": [r["id"] for r in lab], "states": [r["state"] for r in lab],
            "labels10": [r["label"] for r in lab], "n_unlabelled": len(rows) - len(lab)}


def scheme_labels(keys10: Sequence[str], scheme: str) -> list[str]:
    """c10 label keys expressed in `scheme` (c7 collapses with labels.C7_MAP)."""
    if scheme not in L.SCHEMES:
        raise ValueError(f"unknown label scheme {scheme!r}; expected one of {sorted(L.SCHEMES)}")
    return [L.collapse_key(k, scheme) for k in keys10]


def label_vector(keys: Sequence[str], options: Sequence[str]) -> np.ndarray:
    index = {k: i for i, k in enumerate(options)}
    return np.array([index[k] for k in keys], dtype=int)


# ---------------------------------------------------------------- B1

def majority_index(y: np.ndarray, k: int) -> int:
    """Most frequent train class; ties go to the first option in canonical order."""
    return int(np.bincount(np.asarray(y, dtype=int), minlength=k).argmax())


def class_prior(y: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=k).astype(float)
    return counts / counts.sum()


def b1_reports(y_train: np.ndarray, y_by_split: dict[str, np.ndarray], keys: Sequence[str]) -> dict[str, Any]:
    k = len(keys)
    maj, prior = majority_index(y_train, k), class_prior(y_train, k)
    out: dict[str, Any] = {"majority_class": keys[maj], "prior": {c: float(p) for c, p in zip(keys, prior)},
                           "b1_majority": {}, "b1_prior": {}}
    for split, y in y_by_split.items():
        onehot = np.zeros((len(y), k))
        onehot[:, maj] = 1.0
        out["b1_majority"][split] = split_report(onehot, y, keys)
        out["b1_prior"][split] = split_report(np.tile(prior, (len(y), 1)), y, keys)
    return out


# ---------------------------------------------------------------- B3

def make_vectorizer(tfidf_cfg: dict[str, Any]) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(analyzer=tfidf_cfg["analyzer"], ngram_range=tuple(tfidf_cfg["ngram_range"]),
                           min_df=tfidf_cfg["min_df"], sublinear_tf=bool(tfidf_cfg["sublinear_tf"]),
                           max_features=tfidf_cfg["max_features"])


def make_model(C: float, lr_cfg: dict[str, Any]) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, class_weight=lr_cfg["class_weight"], max_iter=int(lr_cfg["max_iter"]),
                              solver=lr_cfg.get("solver", DEFAULT_SOLVER), random_state=RANDOM_STATE)


def embed_log_proba(logp: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    """[N, k] log-probabilities; a class the model never saw in train sits at LOG_FLOOR."""
    full = np.full((logp.shape[0], k), LOG_FLOOR)
    full[:, np.asarray(classes, dtype=int)] = np.maximum(logp, LOG_FLOOR)
    return full


def fit_one(X_train: Any, y_train: np.ndarray, X_eval: dict[str, Any], y_val: np.ndarray, C: float,
            lr_cfg: dict[str, Any], k: int) -> dict[str, Any]:
    """One LR fit (a joblib task): val macro-F1 for model selection and log-probabilities of every eval split."""
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    t0 = time.perf_counter()
    # One BLAS thread per fit: bit-identical results whatever n_jobs is (the sparse products that dominate
    # are single-threaded anyway); ConvergenceWarning is recorded as converged=False instead.
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = make_model(C, lr_cfg).fit(X_train, y_train)
        logp = {s: embed_log_proba(model.predict_log_proba(X), model.classes_, k) for s, X in X_eval.items()}
    n_iter = int(np.max(model.n_iter_))
    return {"C": C, "val_macro_f1": macro_f1(y_val, logp[SELECT_SPLIT].argmax(1)), "n_iter": n_iter,
            "converged": n_iter < model.max_iter, "seconds": round(time.perf_counter() - t0, 2), "logp": logp}


def fit_all(tasks: list[tuple], n_jobs: int) -> list[dict[str, Any]]:
    """Run fit_one over the tasks, in task order (processes when n_jobs > 1)."""
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs)(delayed(fit_one)(*t) for t in tasks)


def select_best(grid: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Highest val macro-F1; max() keeps the first (smallest-C) entry on ties."""
    return max(grid, key=lambda g: g["val_macro_f1"])


def b3_result(grid: list[dict[str, Any]], y_by_split: dict[str, np.ndarray], splits: Sequence[str],
              keys: Sequence[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """(report {C, T, is_clamped, grid, split: {pre, post}}, post-T probabilities per split)."""
    best = select_best(grid)
    T = fit_temperature(best["logp"][SELECT_SPLIT], y_by_split[SELECT_SPLIT])
    out: dict[str, Any] = {"C": best["C"], "T": T, "is_clamped": is_clamped(T),
                           "grid": [{k: g[k] for k in _GRID_KEYS} for g in grid]}
    post_p = {}
    for split in splits:
        logp, y = best["logp"][split], y_by_split[split]
        post_p[split] = apply_temperature(logp, T)
        out[split] = {"pre": split_report(apply_temperature(logp, 1.0), y, keys),
                      "post": split_report(post_p[split], y, keys)}
    return out, post_p


# ---------------------------------------------------------------- CLI

def vectorise(cfg: dict[str, Any], train: dict[str, Any], evals: dict[str, dict]) -> tuple[Any, dict[str, Any], dict]:
    t0 = time.perf_counter()
    vec = make_vectorizer(cfg["baselines"]["tfidf"])
    X_train = vec.fit_transform(train["states"])
    X_eval = {s: vec.transform(e["states"]) for s, e in evals.items()}
    info = {"n_features": len(vec.vocabulary_), "nnz_train": int(X_train.nnz),
            "seconds": round(time.perf_counter() - t0, 2)}
    print(f"baselines: TF-IDF on {X_train.shape[0]} train rows -> {info['n_features']} features "
          f"({info['seconds']:.0f}s)", flush=True)
    return X_train, X_eval, info


def scheme_targets(scheme: str, train: dict, evals: dict[str, dict]) -> dict[str, np.ndarray]:
    keys = L.option_keys(scheme)
    return {s: label_vector(scheme_labels(d["labels10"], scheme), keys)
            for s, d in {TRAIN_SPLIT: train, **evals}.items()}


def write_preds(preds_dir: Path, scheme: str, ids_by_split: dict[str, list[str]], y_by_split: dict[str, np.ndarray],
                P_by_split: dict[str, np.ndarray]) -> None:
    for split, P in P_by_split.items():
        write_jsonl(preds_dir / f"{scheme}_{B3}_{split}.jsonl",
                    ({"id": i, "y": int(y), "p": [float(v) for v in p], "answer_confidence": float(p.max())}
                     for i, y, p in zip(ids_by_split[split], y_by_split[split], P)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, t0, data_dir = load_config(args.config), time.perf_counter(), Path(args.data_dir)
    splits = list(dict.fromkeys(args.splits))
    train = load_split(data_dir, TRAIN_SPLIT)
    evals = {s: load_split(data_dir, s) for s in dict.fromkeys([SELECT_SPLIT, *splits])}
    X_train, X_eval, tfidf = vectorise(cfg, train, evals)
    targets = {scheme: scheme_targets(scheme, train, evals) for scheme in args.schemes}
    grid_cs, lr_cfg = list(cfg["baselines"]["lr"]["C_grid"]), cfg["baselines"]["lr"]
    tasks = [(X_train, targets[s][TRAIN_SPLIT], X_eval, targets[s][SELECT_SPLIT], C, lr_cfg, len(L.option_keys(s)))
             for s in args.schemes for C in grid_cs]
    n_jobs = max(1, args.n_jobs or min(len(tasks), os.cpu_count() or 1, MAX_DEFAULT_JOBS))
    print(f"baselines: fitting {len(tasks)} LR models ({len(grid_cs)} C x {len(args.schemes)} schemes) "
          f"on {n_jobs} worker(s) ...", flush=True)
    t_fit, fits = time.perf_counter(), fit_all(tasks, n_jobs)
    fit_seconds = round(time.perf_counter() - t_fit, 2)
    schemes = {}
    for i, scheme in enumerate(args.schemes):
        keys, y = L.option_keys(scheme), targets[scheme]
        b3, post_p = b3_result(fits[i * len(grid_cs):(i + 1) * len(grid_cs)], y, splits, keys)
        schemes[scheme] = {"labels": keys, **b1_reports(y[TRAIN_SPLIT], {s: y[s] for s in splits}, keys), B3: b3}
        if args.preds_dir:
            write_preds(Path(args.preds_dir), scheme, {s: evals[s]["ids"] for s in splits}, y, post_p)
        print(_line(scheme, schemes[scheme]), flush=True)
    return {"schemes": schemes, "splits": splits, "n_train": len(train["ids"]),
            "n_train_unlabelled": train["n_unlabelled"], "n_by_split": {s: len(evals[s]["ids"]) for s in splits},
            "tfidf": tfidf, "solver": make_model(1.0, lr_cfg).solver, "n_jobs": n_jobs, "random_state": RANDOM_STATE,
            "seconds_fit": fit_seconds,
            "seconds": round(time.perf_counter() - t0, 2)}


def _line(scheme: str, s: dict[str, Any]) -> str:
    def f1(node: dict | None) -> str:
        return "n/a" if not node else f"{node['macro_f1']:.4f}"
    b3, split = s[B3], SELECT_SPLIT if SELECT_SPLIT in s["b1_majority"] else next(iter(s["b1_majority"]), None)
    return (f"baselines {scheme} ({split}): majority {f1(s['b1_majority'].get(split))}, prior "
            f"{f1(s['b1_prior'].get(split))}, TF-IDF+LR {f1((b3.get(split) or {}).get('post'))} (C={b3['C']}, "
            f"T={b3['T']:.3f}{', CLAMPED' if b3['is_clamped'] else ''}) macro-F1")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.baselines", description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", required=True, help="dir with train.jsonl, val.jsonl and the --splits files")
    p.add_argument("--out", required=True)
    p.add_argument("--preds-dir", default=None, help="write per-row B3 predictions (post-T) here")
    p.add_argument("--splits", nargs="+", default=["val", "test_id"])
    p.add_argument("--schemes", nargs="+", default=["c10", "c7"], choices=sorted(L.SCHEMES))
    p.add_argument("--n-jobs", type=int, default=None,
                   help=f"parallel LR fits (default: min(fits, CPUs, {MAX_DEFAULT_JOBS}))")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        write_json(args.out, run(args))
        print(f"baselines: wrote {args.out}")
        return 0

    return run_cli("baselines", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
