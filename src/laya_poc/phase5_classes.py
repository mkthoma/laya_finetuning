"""Per-class metrics, confusion matrices, look-alike cells, metrics at the val tau and the seed flip rate, recomputed
from a run's predictions (design §5.11 "Accuracy" and "Stability", §7.9; spec P5 §2).

The rows evaluate scores are the answered (no-evidence gate) and labelled rows; the prediction is the argmax of the
renormalised post-T probabilities in preds/, so `all.macro_f1` equals the eval JSON's post macro-F1 for the same rows
(metrics.macro_f1 over every option, labels=range(K)). `answered` restricts to rows with answer_confidence >= the val
tau (the §5.8.2 abstention threshold); its coverage counts gate abstentions as abstentions too.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .evaluate import EVENT_KEY
from .metrics import flip_rate, macro_f1
from .phase5_load import SplitPreds, stat5

MATCH_TOL = 1e-12        # recomputed vs eval JSON macro-F1: same rows, same function, so equal up to float noise


def _f1s(y: np.ndarray, yhat: np.ndarray, keys: Sequence[str]) -> dict[str, float | None]:
    k = len(keys)
    nine = [i for i, key in enumerate(keys) if key != EVENT_KEY]
    if not len(y):
        return {"macro_f1": None, "macro_f1_9": None, "acc": None}
    return {"macro_f1": macro_f1(y, yhat, labels=list(range(k))),
            "macro_f1_9": macro_f1(y, yhat, labels=nine) if len(nine) < k else None,
            "acc": float((y == yhat).mean())}


def answered_at_tau(sp: SplitPreds, keys: Sequence[str], tau: float | None) -> dict[str, Any]:
    """Coverage and metrics on the scored rows with confidence >= tau; coverage over every labelled row."""
    labelled = int((sp.y >= 0).sum())
    if tau is None or not labelled:
        return {"tau": tau, "coverage": None, "n_answered": None, "macro_f1": None, "acc": None}
    keep = sp.scored & (np.nan_to_num(sp.conf, nan=-np.inf) >= tau)
    f = _f1s(sp.y[keep], sp.yhat[keep], keys)
    return {"tau": float(tau), "coverage": int(keep.sum()) / labelled, "n_answered": int(keep.sum()),
            "macro_f1": f["macro_f1"], "acc": f["acc"]}


def run_split(sp: SplitPreds, keys: Sequence[str], tau: float | None = None) -> dict[str, Any] | None:
    """One run, one split: counts, headline F1s, per-class P/R/F1, confusion matrix, metrics at tau; None when
    no row is scored (e.g. stripped_test, where the gate abstains on every row)."""
    s = sp.scored
    if not s.any():
        return None
    y, yhat, k = sp.y[s], sp.yhat[s], len(keys)
    pr, rc, f1, sup = precision_recall_fscore_support(y, yhat, labels=list(range(k)), zero_division=0)
    return {"n_rows": len(sp.ids), "n_scored": int(s.sum()), "n_abstained": int(sp.abstained.sum()),
            "n_unlabelled": int((sp.y < 0).sum()), **_f1s(y, yhat, keys),
            "per_class": {key: {"p": float(pr[i]), "r": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                          for i, key in enumerate(keys)},
            "cm": confusion_matrix(y, yhat, labels=list(range(k))).astype(int),
            "answered": answered_at_tau(sp, keys, tau)}


def lookalike_cells(cms: Sequence[np.ndarray], keys: Sequence[str],
                    pairs: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """Both directions of each look-alike pair: summed confusions, summed true-class support, pooled rate and the
    per-run rates. Pairs whose keys are not options of this scheme (c7 merges some) are skipped."""
    index = {key: i for i, key in enumerate(keys)}
    out = []
    for a, b in pairs:
        if a not in index or b not in index:
            continue
        for t, p in ((a, b), (b, a)):
            i, j = index[t], index[p]
            counts = [int(cm[i, j]) for cm in cms]
            supports = [int(cm[i].sum()) for cm in cms]
            rates = [c / s if s else None for c, s in zip(counts, supports)]
            total = sum(supports)
            out.append({"true": t, "pred": p, "count": sum(counts), "support": total,
                        "rate": sum(counts) / total if total else None, "rate_runs": stat5(rates)})
    return out


def _eval_match(runs: Sequence[dict | None], eval_f1: Sequence[float | None]) -> dict[str, Any]:
    diffs = [abs(r["macro_f1"] - e) for r, e in zip(runs, eval_f1) if r and r["macro_f1"] is not None
             and e is not None]
    missing = sum(1 for r, e in zip(runs, eval_f1) if r and (e is None or r["macro_f1"] is None))
    worst = max(diffs, default=0.0)
    return {"passed": bool(diffs) and not missing and worst <= MATCH_TOL, "max_abs_diff": worst,
            "compared": len(diffs)}


def group_split(per_run: Sequence[dict | None], keys: Sequence[str], pairs: Sequence[Sequence[str]],
                eval_f1: Sequence[float | None]) -> dict[str, Any] | None:
    """Aggregate run_split outputs over a group's runs (run order kept in every stat's values)."""
    runs = [r for r in per_run if r is not None]
    if not runs:
        return None
    first, cms = runs[0], [r["cm"] for r in runs]

    def over(get) -> dict | None:
        return stat5([get(r) if r else None for r in per_run])

    return {"labels": list(keys), **{k: first[k] for k in ("n_rows", "n_scored", "n_abstained", "n_unlabelled")},
            "all": {m: over(lambda r, m=m: r[m]) for m in ("macro_f1", "macro_f1_9", "acc")},
            "eval_match": _eval_match(per_run, eval_f1),
            "answered": {m: over(lambda r, m=m: r["answered"][m])
                         for m in ("tau", "coverage", "n_answered", "macro_f1", "acc")},
            "per_class": {key: {**{m: over(lambda r, m=m, key=key: r["per_class"][key][m]) for m in ("p", "r", "f1")},
                                "support": first["per_class"][key]["support"]} for key in keys},
            "cm": np.sum(cms, axis=0).astype(int).tolist(), "cm_runs": len(cms),
            "lookalike": lookalike_cells(cms, keys, pairs)}


def aligned_argmax(preds: Sequence[SplitPreds]) -> list[np.ndarray]:
    """Each run's argmax on the rows every run answered, aligned by row id (first run's order)."""
    common = set(preds[0].ids)
    for sp in preds[1:]:
        common &= set(sp.ids)
    for sp in preds:
        common -= {i for i, a in zip(sp.ids, sp.abstained) if a}
    order = [i for i in preds[0].ids if i in common]
    out = []
    for sp in preds:
        pos = {rid: n for n, rid in enumerate(sp.ids)}
        out.append(sp.yhat[[pos[i] for i in order]])
    return out


def group_flip(preds: Sequence[SplitPreds]) -> tuple[float | None, int | None]:
    """(flip rate, rows) across the group's seed runs: share of rows whose argmax differs between any two runs."""
    if len(preds) < 2:
        return None, None
    aligned = aligned_argmax(preds)
    n = len(aligned[0])
    return (flip_rate(aligned) if n else None), n

