"""The shared per-split writer of the Phase 4 baselines (B1, B3, B4, B5): EXACTLY the evaluate multi-split schema.

A baseline scores every row of a split (raw logits or log-probabilities, [N, K] in option-key order); this module
turns the scores into the eval/<split>.json and preds/<split>.jsonl a Phase 3 Laya run writes
(evaluate_splits.py), through the functions that path uses, so a baseline's numbers mean what a Laya run's mean:
- the no-evidence gate (evaluate.evidence_mask, design §5.8.1): rows without evidence abstain;
- POST = softmax(scores / T), PRE = softmax(scores), renormalised in option-key order like the probabilities
  evaluate reads from Laya, with answer_confidence = max p (Laya's); metrics by evaluate's own scorer
  (split_report on the labelled answered rows, NLL from log_softmax(scores / T): nll_source "logits");
- tau chosen on val (robustness.choose_tau on the post-T scored rows, config `abstain`), the abstention block
  (evaluate_splits.selective) on every split;
- stripped_test: the model's own answers with the gate bypassed (no_gate_scores, default: scores) give
  abstain_rate_at_tau, false_confident_rate and the `no_gate` block, as evaluate_splits does.

    payload, preds = split_outputs(rows=rows, keys=keys, scores=Z, T=T, split=split, model="tfidf_lr", ckpt=...,
                                   evidence_fields=cfg["serialise"]["evidence_fields"], abstain_cfg=cfg["abstain"],
                                   tau=val_payload["tau"], meta={"rows": str(path), "device": "cpu"})
    no_gate = no_gate_records(rows=rows, keys=keys, scores=Z, T=T) if split == STRIPPED_SPLIT else None
    write_split(run_dir / EVAL_DIR, run_dir / PREDS_DIR, split, payload, preds, no_gate)

`write_outputs` does those three steps in one call. `meta` carries the info fields only the caller knows (rows
path, device, batch size, timings, e.g. `subset: true` for B5); it cannot override a computed field.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibrate import fit_temperature, is_clamped, softmax
from .env_check import write_json
from .evaluate import _scatter, _score, evidence_mask, label_indices, pred_records
from .evaluate_splits import NO_GATE_DIR, STRIPPED_SPLIT, TAU_SPLIT, order_settings, selective
from .io_utils import write_jsonl
from .labels import SCHEMES, option_keys
from .robustness import (FALLBACK_COVERAGE, FALSE_CONFIDENT_AT, MIN_COVERAGE, TARGET_ACC, abstention, agreement,
                         canonical_state, choose_tau, conf_correct, false_confident_rate, permuted_states)

EVAL_DIR, PREDS_DIR = "eval", "preds"
LOG_FLOOR = float(np.log(1e-12))  # calibrate.fit_temperature's floor (and baselines.LOG_FLOOR)
CALIBRATION_FILE, ORDER_FILE = "calibration.json", "order_invariance.json"
NO_EVIDENCE_BASIS = ("model answers with the no-evidence gate bypassed (with it, every country-only row "
                     "abstains)")  # evaluate_splits.stripped_fields
INFO_DEFAULTS: Mapping[str, Any] = {"rows": None, "cpu_fallback": False, "device": "cpu", "batch_size": None,
                                    "max_len": None, "head_max_len": None}
COMPUTED = frozenset({"model", "ckpt", "split", "labels", "n", "n_unlabelled", "n_abstained_no_evidence",
                      "n_scored", "temperature", "temperature_pre", "post", "pre", "preds", "tau", "abstention",
                      "abstain_rate_at_tau", "false_confident_rate", "false_confident_threshold",
                      "gate_abstain_rate", "no_evidence_basis", "no_gate", "no_gate_preds"})


# ---------------------------------------------------------------- inputs

def abstain_settings(abstain_cfg: Mapping[str, Any] | None) -> dict[str, float]:
    """config `abstain` -> robustness.choose_tau keyword arguments, with evaluate_splits' defaults."""
    ab = dict(abstain_cfg or {})
    return {"target_acc": float(ab.get("target_acc", TARGET_ACC)),
            "min_coverage": float(ab.get("min_coverage", MIN_COVERAGE)),
            "fallback_coverage": float(ab.get("fallback_coverage", FALLBACK_COVERAGE))}


def scheme_of(keys: Sequence[str]) -> str | None:
    """The label scheme whose option keys are `keys` in this order, else None."""
    return next((s for s in sorted(SCHEMES) if option_keys(s) == list(keys)), None)


def _check_T(T: float) -> float:
    t = float(T)
    if not np.isfinite(t) or t <= 0:
        raise ValueError(f"temperature must be finite and > 0, got {T!r}")
    return t


def _check_scores(scores: Any, n: int, k: int, used: np.ndarray, name: str) -> np.ndarray:
    """[n, k] float scores; the rows in `used` must be finite (the others are never read)."""
    Z = np.asarray(scores, dtype=float)
    if Z.shape != (n, k):
        raise ValueError(f"{name} must be [rows, options] = [{n}, {k}], got {list(Z.shape)}")
    if not np.isfinite(Z[used]).all():
        raise ValueError(f"{name} has non-finite values on rows that are scored (floor log-probabilities of "
                         f"impossible options, e.g. at log(1e-12), instead of -inf)")
    return Z


def probabilities(scores: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    """(P, answer_confidence) of softmax(scores / T): P renormalised in option-key order, as evaluate reads
    Laya's answers (probs_from_outputs); answer_confidence = max p, Laya's definition."""
    p = softmax(np.asarray(scores, dtype=float) / T)
    s = p.sum(1, keepdims=True)
    return np.divide(p, s, out=np.full_like(p, np.nan), where=s > 0), p.max(1)


# ---------------------------------------------------------------- one split

def _side(Z: np.ndarray, answered: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Full-length (P, answer_confidence) at temperature T; NaN on gated rows, as evaluate scatters them."""
    P, conf = probabilities(Z[answered], T)
    return _scatter(P, answered, Z.shape[1]), _scatter(conf, answered, None)


def _tau_block(split: str, preds: Sequence[dict], tau: Any, abstain: dict[str, float]) -> dict | None:
    """The val tau dict: chosen here on val when not given; a given dict (choose_tau's) or number is used as is."""
    if tau is None:
        return choose_tau(*conf_correct(preds), **abstain) if split == TAU_SPLIT else None
    if isinstance(tau, Mapping):
        if "tau" not in tau:
            raise ValueError("tau must be the val payload's `tau` dict (robustness.choose_tau) or a number")
        return dict(tau)
    if isinstance(tau, (int, float)) and not isinstance(tau, bool):
        return {"tau": float(tau), "rule": None}
    raise TypeError(f"tau must be a dict, a number or None, got {type(tau).__name__}")


def _no_gate_block(Z: np.ndarray, T: float, tau: float | None) -> dict[str, Any]:
    """evaluate_splits.no_gate's block: the model's answers on every row, gate bypassed, post-T."""
    _, conf = probabilities(Z, T)
    at = abstention(conf, None, tau)
    return {"n": len(Z), "tau": tau, "abstain_rate_at_tau": at["abstain_rate"],
            "false_confident_rate": false_confident_rate(conf), "false_confident_threshold": FALSE_CONFIDENT_AT,
            "mean_answer_confidence": float(conf.mean()) if len(conf) else None, "cpu_fallback": False,
            "seconds": 0.0}


def _stripped(res: dict, Z_ng: np.ndarray, T: float, tau: dict | None) -> dict[str, Any]:
    """evaluate_splits.stripped_fields for gate-bypassed scores (no_gate_preds: filled in by write_split)."""
    block = _no_gate_block(Z_ng, T, None if tau is None else tau["tau"])
    return {"abstain_rate_at_tau": block["abstain_rate_at_tau"], "false_confident_rate": block["false_confident_rate"],
            "false_confident_threshold": FALSE_CONFIDENT_AT,
            "gate_abstain_rate": res["n_abstained_no_evidence"] / res["n"] if res["n"] else None,
            "no_evidence_basis": NO_EVIDENCE_BASIS, "no_gate": block, "no_gate_preds": None,
            "cpu_fallback": bool(res["cpu_fallback"])}


def _with_meta(payload: dict, meta: Mapping[str, Any] | None, seconds: float) -> dict:
    """Caller info over the defaults (new keys are appended); a computed field cannot be overridden."""
    meta = dict(meta or {})
    clash = sorted(COMPUTED & set(meta))
    if clash:
        raise ValueError(f"meta cannot override computed fields: {', '.join(clash)}")
    return {**payload, **meta, "seconds": meta.get("seconds", seconds)}


def split_outputs(*, rows: Sequence[dict], keys: Sequence[str], scores: Any, T: float, split: str, model: str,
                  ckpt: str, evidence_fields: Sequence[str], abstain_cfg: Mapping[str, Any] | None,
                  tau: Any = None, no_gate_scores: Any = None,
                  meta: Mapping[str, Any] | None = None) -> tuple[dict, list[dict]]:
    """(payload in the evaluate multi-split schema, post-T preds records) for one split. `scores` covers ALL
    rows in row order (gated rows are not read); `tau` is the val payload's `tau` (chosen here on val)."""
    t0, rows, keys = time.perf_counter(), list(rows), list(keys)
    if not rows:
        raise ValueError(f"{split}: no rows")
    T = _check_T(T)
    answered, y = evidence_mask(rows, evidence_fields), label_indices(rows, keys)
    Z = _check_scores(scores, len(rows), len(keys), answered, "scores")
    scored = answered & np.array([v is not None for v in y], dtype=bool)
    Z_answered = _scatter(Z[answered], answered, len(keys))
    sides = {"post": _side(Z, answered, T), "pre": _side(Z, answered, 1.0)}
    reports = {side: _score(P, conf, Z_answered, t, y, scored, keys)
               for (side, (P, conf)), t in zip(sides.items(), (T, 1.0))}
    preds = pred_records(rows, y, *sides["post"], ~answered)
    res = {"model": model, "ckpt": ckpt, "split": split, "rows": None, "scheme": scheme_of(keys), "labels": keys,
           "n": len(rows), "n_unlabelled": int(sum(v is None for v in y)),
           "n_abstained_no_evidence": int((~answered).sum()), "n_scored": int(scored.sum()),
           "temperature": T, "temperature_pre": 1.0, **reports, "seconds_post": 0.0, "seconds_pre": 0.0,
           **INFO_DEFAULTS, "preds": None}
    tau_block = _tau_block(split, preds, tau, abstain_settings(abstain_cfg))
    res = {**res, **({"tau": tau_block} if split == TAU_SPLIT else {}), "abstention": selective(preds, tau_block)}
    if split == STRIPPED_SPLIT:
        ng = Z if no_gate_scores is None else no_gate_scores
        res = {**res, **_stripped(res, _check_scores(ng, len(rows), len(keys), np.ones(len(rows), bool),
                                                     "no_gate_scores"), T, tau_block)}
    return _with_meta(res, meta, round(time.perf_counter() - t0, 2)), preds


def no_gate_records(*, rows: Sequence[dict], keys: Sequence[str], scores: Any, T: float) -> list[dict]:
    """preds/no_gate/stripped_test.jsonl records: every row answered (gate bypassed), post-T."""
    rows, keys = list(rows), list(keys)
    Z = _check_scores(scores, len(rows), len(keys), np.ones(len(rows), dtype=bool), "no_gate_scores")
    P, conf = probabilities(Z, _check_T(T))
    return pred_records(rows, label_indices(rows, keys), P, conf, np.zeros(len(rows), dtype=bool))


def write_split(out_dir: str | Path, preds_dir: str | Path, split: str, payload: dict, preds: Sequence[dict],
                no_gate_preds: Sequence[dict] | None = None) -> dict:
    """Write <preds_dir>/<split>.jsonl (+ no_gate/stripped_test.jsonl) and <out_dir>/<split>.json with the
    paths filled in, as evaluate_splits does; returns the written payload."""
    if payload.get("split") != split:
        raise ValueError(f"payload is for split {payload.get('split')!r}, not {split!r}")
    if len(preds) != payload["n"]:
        raise ValueError(f"{split}: {len(preds)} preds records for {payload['n']} rows")
    preds_path, paths = Path(preds_dir) / f"{split}.jsonl", {}
    if split == STRIPPED_SPLIT:
        if no_gate_preds is None or len(no_gate_preds) != payload["n"]:
            raise ValueError(f"{split} needs no_gate_preds for every row (no_gate_records(...))")
        ng_path = Path(preds_dir) / NO_GATE_DIR / f"{STRIPPED_SPLIT}.jsonl"
        write_jsonl(ng_path, no_gate_preds)
        paths["no_gate_preds"] = str(ng_path)
    elif no_gate_preds is not None:
        raise ValueError(f"no_gate_preds are written for {STRIPPED_SPLIT} only, not {split}")
    write_jsonl(preds_path, preds)
    final = {**payload, "preds": str(preds_path), **paths}
    write_json(Path(out_dir) / f"{split}.json", final)
    return final


def write_outputs(out_dir: str | Path, preds_dir: str | Path, **kwargs: Any) -> dict:
    """split_outputs + (stripped_test) no_gate_records + write_split in one call; returns the written payload."""
    payload, preds = split_outputs(**kwargs)
    no_gate = None
    if kwargs["split"] == STRIPPED_SPLIT:
        ng = kwargs.get("no_gate_scores")
        no_gate = no_gate_records(rows=kwargs["rows"], keys=kwargs["keys"],
                                  scores=kwargs["scores"] if ng is None else ng, T=kwargs["T"])
    return write_split(out_dir, preds_dir, kwargs["split"], payload, preds, no_gate)


# ---------------------------------------------------------------- temperature and calibration.json

def fit_T(val_scores: Any, val_y: Any) -> tuple[float, bool]:
    """(T, clamped): calibrate.fit_temperature on labelled val rows (hard labels), clamped per is_clamped.
    fit_temperature floors its input at log(1e-12) before dividing by T, which would distort raw scores that
    reach below it (e.g. summed token log-probabilities); those are divided by s first (T = s * T'), so the
    floor never binds. Log-probabilities floored at log(1e-12) (B1, B3) are fitted exactly as calibrate does."""
    Z, y = np.asarray(val_scores, dtype=float), np.asarray(val_y)
    if Z.ndim != 2 or len(Z) == 0 or len(y) != len(Z):
        raise ValueError(f"need non-empty val scores [N, K] and labels [N]; got {list(Z.shape)} and {list(y.shape)}")
    if not np.isfinite(Z).all():
        raise ValueError("val scores must be finite (no NaN; floor impossible options at log(1e-12), not -inf)")
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= Z.shape[1]:
        raise ValueError(f"val labels must be option indices in [0, {Z.shape[1]})")
    s = max(1.0, float(Z.min()) / LOG_FLOOR)
    T = fit_temperature(Z, y) if s == 1.0 else s * fit_temperature(Z / s, y)
    return T, is_clamped(T)


def calibration_record(*, T: float, clamped: bool, n: int, fitted_on: str | None = TAU_SPLIT,
                       extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """calibration.json (spec §1): {T, clamped, fitted_on, n, extra}; fitted_on None when T is not fitted."""
    return {"T": float(T), "clamped": bool(clamped), "fitted_on": fitted_on, "n": int(n), "extra": dict(extra or {})}


# ---------------------------------------------------------------- order invariance

def order_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """{split, n, perms, seed}: config phase3.order_invariance over evaluate's defaults."""
    return order_settings(dict(cfg), SimpleNamespace(order_split=None, order_n=None, order_perms=None,
                                                     order_seed=None))


def order_invariance_fn(predict_argmax: Callable[[list[str]], np.ndarray], rows: Sequence[dict], *, n: int,
                        perms: int, seed: int) -> dict[str, Any]:
    """robustness.order_invariance for any model: the first n rows' fixed (alphabetical) field order vs `perms`
    shuffled orders; predict_argmax maps compact JSON state STRINGS to option indices. Same output keys."""
    if perms < 1 or n < 1:
        raise ValueError(f"order invariance needs perms >= 1 and n >= 1, got perms={perms}, n={n}")
    if not rows:
        raise ValueError("order invariance: no rows")
    t0 = time.perf_counter()
    fixed_states = [canonical_state(r["state"]) for r in rows[:n]]

    def predict(states: list[str]) -> np.ndarray:
        out = np.asarray(predict_argmax(list(states))).reshape(-1)
        if len(out) != len(states):
            raise ValueError(f"predict_argmax returned {len(out)} predictions for {len(states)} states")
        return out

    fixed = predict(fixed_states)
    shuffled = permuted_states(fixed_states, perms, seed)
    preds = [predict(s) for s in shuffled]
    unchanged = [float(np.mean([a == b for a, b in zip(fixed_states, s)])) for s in shuffled]
    return {"n": len(fixed_states), "perms": perms, "seed": seed, **agreement(fixed, preds),
            "unchanged_share_per_perm": unchanged, "cpu_fallback": False,
            "seconds": round(time.perf_counter() - t0, 2)}


def order_invariance_payload(predict_argmax: Callable[[list[str]], np.ndarray], rows: Sequence[dict], *,
                             model: str, ckpt: str, split: str, rows_path: str | None,
                             evidence_fields: Sequence[str], n: int, perms: int, seed: int) -> dict[str, Any]:
    """order_invariance.json as evaluate_splits.run_order writes it: only rows the model answers (the gate
    skips the rest), plus model/ckpt/split/rows and n_skipped_no_evidence."""
    answered = [r for r, a in zip(rows, evidence_mask(rows, evidence_fields)) if a]
    res = order_invariance_fn(predict_argmax, answered, n=n, perms=perms, seed=seed)
    return {"model": model, "ckpt": ckpt, "split": split, "rows": rows_path, **res,
            "n_skipped_no_evidence": len(rows) - len(answered)}
