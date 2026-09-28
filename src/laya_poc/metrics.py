"""Evaluation metrics (design doc §5.11, §7.9). Pure numpy/sklearn, CPU-only."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

EPS = 1e-12


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on (lo, hi]; 0 falls in the first bin."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(P: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean squared distance to the one-hot target."""
    Y = np.eye(P.shape[1])[y]
    return float(((P - Y) ** 2).sum(1).mean())


def nll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0)).mean())


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Accuracy of the most-confident prefix at every coverage level."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    c = np.asarray(correct, dtype=float)[order]
    n = len(c)
    ranks = np.arange(1, n + 1)
    cov, acc = ranks / n, np.cumsum(c) / ranks

    def at(q: float) -> float:
        return float(acc[max(0, int(np.ceil(q * n)) - 1)])

    return cov, acc, {"acc@80": at(0.8), "acc@90": at(0.9)}


def flip_rate(preds_by_seed: Sequence[np.ndarray]) -> float:
    """Share of items whose argmax differs across seeds."""
    P = np.stack(preds_by_seed)
    return float((P != P[0]).any(0).mean())


def order_invariance(pred_fixed: np.ndarray, preds_perm: Sequence[np.ndarray]) -> float:
    """Mean argmax agreement between the fixed field order and each shuffled order."""
    return float(np.mean([(pred_fixed == p).mean() for p in preds_perm]))


def macro_f1(y: np.ndarray, yhat: np.ndarray, labels: Sequence[int] | None = None) -> float:
    return float(f1_score(y, yhat, labels=labels, average="macro", zero_division=0))


def report(P: np.ndarray, y: np.ndarray, labels: Sequence[str], conf: np.ndarray | None = None) -> dict:
    """Headline metric bundle for one split. `conf` defaults to max probability (answer_confidence)."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y)
    if P.ndim != 2 or P.shape[0] != len(y) or P.shape[1] != len(labels):
        raise ValueError(f"shape mismatch: P{P.shape}, y{y.shape}, {len(labels)} labels")
    yhat = P.argmax(1)
    conf = P.max(1) if conf is None else np.asarray(conf, dtype=float)
    correct = (yhat == y).astype(float)
    k = range(len(labels))
    pr, rc, f1, support = precision_recall_fscore_support(y, yhat, labels=list(k), zero_division=0)
    return {
        "n": int(len(y)),
        "macro_f1": macro_f1(y, yhat),
        "acc": float(accuracy_score(y, yhat)),
        "ece": ece(conf, correct),
        "brier": brier(P, y),
        "nll": nll(P, y),
        **risk_coverage(conf, correct)[2],
        "per_class": {lab: {"p": float(pr[i]), "r": float(rc[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, lab in enumerate(labels)},
        "cm": confusion_matrix(y, yhat, labels=list(k)).tolist(),
    }
