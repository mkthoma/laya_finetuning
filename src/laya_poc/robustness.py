"""Selective prediction and field-order robustness (design doc §5.8.2, §5.11, §7.9 steps 5-6).

- `choose_tau`: the abstention threshold, chosen on `val` after temperature: the smallest threshold on
  `answer_confidence` whose answered set (confidence >= tau) reaches `target_acc`; if that answers less
  than `min_coverage` of the rows, the threshold for `fallback_coverage` (the doc's 80%) is used instead,
  and both are reported. Thresholds only ever sit at a distinct confidence value, so a tie group is
  answered or abstained as a whole (a threshold cannot split rows with equal confidence).
- `abstention` / `false_confident_rate`: coverage and answered accuracy at tau; the share of rows with
  `answer_confidence` > 0.8 (the §5.11 "No evidence" metric on `stripped_test`).
- `order_invariance`: argmax agreement between the fixed (alphabetical) field order and shuffled orders
  (§7.9 step 6: 5 orders on 1,000 `test_id` rows, agreement >= 0.99). States stay compact JSON STRINGS.
"""
from __future__ import annotations

import json
import math
import random
import time
from typing import Any, Sequence

import numpy as np

from .fallback_guard import FallbackGuard
from .serialise import dumps, order_fields

TARGET_ACC = 0.95            # design §5.8.2 (config abstain.target_acc)
MIN_COVERAGE = 0.70          # config abstain.min_coverage
FALLBACK_COVERAGE = 0.80     # design §5.8.2 "use the threshold for 80% coverage instead"
FALSE_CONFIDENT_AT = 0.8     # design §5.11: share with answer_confidence > 0.8
PASS_AGREEMENT = 0.99        # design §5.11 / §7.9 step 6


# ---------------------------------------------------------------- selective prediction

def _check(conf: Any, correct: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
    c = np.asarray(conf, dtype=float).reshape(-1)
    if not np.isfinite(c).all():
        raise ValueError("answer confidences must be finite (abstained rows carry none; drop them first)")
    if correct is None:
        return c, None
    k = np.asarray(correct, dtype=float).reshape(-1)
    if len(k) != len(c):
        raise ValueError(f"conf and correct differ in length: {len(c)} vs {len(k)}")
    return c, k


def abstention(conf: Any, correct: Any, tau: float | None) -> dict[str, Any]:
    """Coverage (share with conf >= tau) and accuracy on the answered rows; None where undefined."""
    c, k = _check(conf, correct)
    if tau is None or not len(c):
        return {"tau": tau, "n": len(c), "n_answered": None, "coverage": None, "abstain_rate": None,
                "answered_acc": None}
    answered = c >= tau
    n_ans = int(answered.sum())
    acc = float(k[answered].mean()) if k is not None and n_ans else None
    return {"tau": float(tau), "n": len(c), "n_answered": n_ans, "coverage": n_ans / len(c),
            "abstain_rate": 1 - n_ans / len(c), "answered_acc": acc}


def _cuts(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(confidences sorted descending, True at the last index of each tie group = valid prefix ends)."""
    s = c[np.argsort(-c, kind="stable")]
    return s, np.append(s[:-1] > s[1:], True) if len(s) else np.zeros(0, dtype=bool)


def threshold_for_accuracy(conf: Any, correct: Any, target_acc: float) -> float | None:
    """The smallest threshold whose answered set reaches target_acc (None when no threshold does).
    Prefix accuracy is not monotone, so every cut is checked and the lowest passing one wins."""
    c, k = _check(conf, correct)
    if not len(c):
        return None
    order = np.argsort(-c, kind="stable")
    s, ends = _cuts(c)
    acc = np.cumsum(k[order]) / np.arange(1, len(c) + 1)
    ok = np.flatnonzero(ends & (acc >= target_acc - 1e-12))
    return float(s[ok[-1]]) if len(ok) else None


def threshold_for_coverage(conf: Any, coverage: float) -> float | None:
    """The threshold answering the top ceil(coverage * n) rows (ties can push coverage above it)."""
    c, _ = _check(conf)
    if not len(c):
        return None
    s, _ = _cuts(c)
    k = max(1, math.ceil(round(coverage * len(c), 9)))
    return float(s[min(k, len(c)) - 1])


def choose_tau(conf: Any, correct: Any, target_acc: float = TARGET_ACC, min_coverage: float = MIN_COVERAGE,
               fallback_coverage: float = FALLBACK_COVERAGE) -> dict[str, Any]:
    """Design §5.8.2 on scored val rows (post-T). `rule` says which threshold became tau."""
    for name, v in (("target_acc", target_acc), ("min_coverage", min_coverage),
                    ("fallback_coverage", fallback_coverage)):
        if not 0 < v <= 1:
            raise ValueError(f"{name} must be in (0, 1], got {v}")
    c, k = _check(conf, correct)
    t_acc = threshold_for_accuracy(c, k, target_acc)
    t_cov = threshold_for_coverage(c, fallback_coverage)
    at_acc = abstention(c, k, t_acc) if t_acc is not None else None
    at_cov = abstention(c, k, t_cov) if t_cov is not None else None
    use_acc = at_acc is not None and at_acc["coverage"] >= min_coverage
    rule = "accuracy" if use_acc else ("coverage" if at_cov is not None else None)
    return {"tau": t_acc if use_acc else t_cov, "rule": rule, "n": len(c), "target_acc": target_acc,
            "min_coverage": min_coverage, "fallback_coverage": fallback_coverage,
            "at_target_acc": at_acc, "at_fallback_coverage": at_cov}


def false_confident_rate(conf: Any, threshold: float = FALSE_CONFIDENT_AT) -> float | None:
    """Share of rows with answer_confidence strictly above `threshold` (None for no rows)."""
    c, _ = _check(conf)
    return float((c > threshold).mean()) if len(c) else None


def conf_correct(preds: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(answer_confidence, argmax(p) == y) over the scored records of a preds list (answered, labelled);
    the argmax of the renormalised probabilities, as the accuracy metrics use."""
    scored = [r for r in preds if not r["abstained"] and r["y"] is not None]
    conf = np.array([float(r["answer_confidence"]) for r in scored], dtype=float)
    correct = np.array([float(int(np.argmax(r["p"])) == r["y"]) for r in scored], dtype=float)
    return conf, correct


# ---------------------------------------------------------------- field order

def _record(state: str) -> dict:
    rec = json.loads(state)
    if not isinstance(rec, dict):
        raise ValueError(f"a state must be a JSON object, got {type(rec).__name__}")
    return rec


def canonical_state(state: str) -> str:
    """The compact JSON string with alphabetical keys (the evaluation field order)."""
    return dumps(order_fields(_record(state)))


def permuted_states(states: Sequence[str], perms: int, seed: int) -> list[list[str]]:
    """`perms` lists of the states with shuffled field order: one random.Random(seed + p) per order,
    drawn row by row. Compact JSON strings (never dicts), same content."""
    out = []
    for p in range(perms):
        rng = random.Random(seed + p)
        out.append([dumps(order_fields(_record(s), rng=rng, shuffle=True)) for s in states])
    return out


def agreement(fixed: Any, perm_preds: Sequence[Any]) -> dict[str, Any]:
    """Argmax agreement of each shuffled order with the fixed order; passed_99 = mean >= 0.99."""
    if not perm_preds:
        raise ValueError("need at least one permuted prediction set (perms >= 1)")
    fixed = np.asarray(fixed)
    per = [float((fixed == np.asarray(p)).mean()) for p in perm_preds]
    mean = float(np.mean(per))
    return {"agreement_per_perm": per, "mean_agreement": mean, "passed_99": bool(mean >= PASS_AGREEMENT)}


def _argmax(agent: Any, states: Sequence[str], question: dict, keys: Sequence[str], guard: Any, label: str,
            **budget: int) -> np.ndarray:
    """The model's choice per state: argmax of the raw logits when captured (no 4-dp rounding ties),
    else of the probabilities."""
    from .evaluate import predict_pass

    with guard:
        P, _, Z, _ = predict_pass(agent, states, question, keys, label=label, **budget)
    return (Z if Z is not None else P).argmax(1)


def order_invariance(agent: Any, rows: Sequence[dict], question: dict, *, n: int, perms: int, seed: int,
                     batch_size: int, max_len: int, head_max_len: int,
                     guard: FallbackGuard | None = None) -> dict[str, Any]:
    """First n rows: the fixed (alphabetical) prediction vs `perms` shuffled field orders (design §7.9
    step 6). The caller passes rows the model would answer (the no-evidence gate never reaches it)."""
    if perms < 1 or n < 1:
        raise ValueError(f"order invariance needs perms >= 1 and n >= 1, got perms={perms}, n={n}")
    if not rows:
        raise ValueError("order invariance: no rows")
    guard = guard or FallbackGuard()
    keys = list(question[next(iter(question))]["criteria"])
    budget = {"batch_size": batch_size, "max_len": max_len, "head_max_len": head_max_len}
    t0 = time.perf_counter()
    fixed_states = [canonical_state(r["state"]) for r in rows[:n]]
    fixed = _argmax(agent, fixed_states, question, keys, guard, "order fixed", **budget)
    shuffled = permuted_states(fixed_states, perms, seed)
    preds = [_argmax(agent, s, question, keys, guard, f"order perm {p + 1}/{perms}", **budget)
             for p, s in enumerate(shuffled)]
    unchanged = [float(np.mean([a == b for a, b in zip(fixed_states, s)])) for s in shuffled]
    return {"n": len(fixed_states), "perms": perms, "seed": seed, **agreement(fixed, preds),
            "unchanged_share_per_perm": unchanged, "cpu_fallback": guard.cpu_fallback,
            "seconds": round(time.perf_counter() - t0, 2)}
