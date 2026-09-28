"""Paired bootstrap of macro-F1 differences (design §5.11: 95% CIs from 1,000 resamples for the macro-F1 difference
between Laya and each baseline on each pool; spec P5 §2).

For one comparison on one pool, the rows are the ids every run of both groups scored (a B5 baseline evaluated a
subset: only its ids). A resample draws row ids with replacement; each seed run's macro-F1 is computed on it, the
seeds are averaged per group, and the difference is candidate - baseline. The OOD average combines the pools'
independent resamples index by index: diff_avg[b] = mean over pools of diff_pool[b].

Vectorised: a resample is a count vector over the rows, so the per-class true positives, predicted and true counts
of every resample are one [B, n] @ [n, K] product each, and macro-F1 = mean_k 2TP / (pred + true) over every option
(0 when a class is neither true nor predicted: metrics.macro_f1 with labels=range(K), zero_division=0).
Deterministic: the resamples of a row set depend only on (seed, pool, the row ids).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .phase5_load import SplitPreds


def resample_counts(n: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """[resamples, n] float counts: how often each row is drawn in each resample (n draws with replacement)."""
    if n < 1 or resamples < 1:
        raise ValueError(f"bootstrap needs rows and resamples >= 1, got n={n}, resamples={resamples}")
    idx = rng.integers(0, n, size=(resamples, n))
    flat = (idx + np.arange(resamples)[:, None] * n).ravel()
    return np.bincount(flat, minlength=resamples * n).reshape(resamples, n).astype(float)


def one_hot(v: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(v), k))
    out[np.arange(len(v)), v] = 1.0
    return out


def weighted_macro_f1(W: np.ndarray, y: np.ndarray, yhat: np.ndarray, k: int) -> np.ndarray:
    """Macro-F1 over every option for each row of the count matrix W ([B, n]); W = ones gives the plain value."""
    W = np.atleast_2d(np.asarray(W, dtype=float))
    Y, H = one_hot(y, k), one_hot(yhat, k)
    tp, true, pred = W @ (Y * H), W @ Y, W @ H
    denom = true + pred
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(1)


def row_seed(seed: int, pool: str, ids: Sequence[str]) -> list[int]:
    """Seed sequence of a row set: the same rows in the same pool always get the same resamples."""
    return [int(seed), zlib.crc32(pool.encode("utf-8")), zlib.crc32("\n".join(ids).encode("utf-8"))]


def common_rows(preds: Sequence[SplitPreds]) -> list[str]:
    """Ids scored by every run (first run's order)."""
    keep = set.intersection(*({i for i, s in zip(sp.ids, sp.scored) if s} for sp in preds))
    return [i for i in preds[0].ids if i in keep]


def take(sp: SplitPreds, ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(y, argmax) of a run on the ids, in their order."""
    pos = {rid: n for n, rid in enumerate(sp.ids)}
    idx = np.array([pos[i] for i in ids], dtype=int)
    return sp.y[idx], sp.yhat[idx]


@dataclass
class Bootstrap:
    """Cached resamples per (pool, row set) and per-run resampled macro-F1s; one instance per metrics run."""
    resamples: int
    seed: int
    ci: float = 0.95
    _counts: dict = field(default_factory=dict)
    _f1: dict = field(default_factory=dict)

    def counts(self, pool: str, ids: Sequence[str]) -> np.ndarray:
        key = (pool, tuple(ids))
        if key not in self._counts:
            rng = np.random.default_rng(row_seed(self.seed, pool, ids))
            self._counts[key] = resample_counts(len(ids), self.resamples, rng)
        return self._counts[key]

    def run_f1(self, run: str, sp: SplitPreds, pool: str, ids: Sequence[str], k: int) -> tuple[float, np.ndarray]:
        """(macro-F1 on the rows, [B] macro-F1 on each resample) of one run."""
        key = (run, pool, tuple(ids))
        if key not in self._f1:
            y, yhat = take(sp, ids)
            full = float(weighted_macro_f1(np.ones((1, len(ids))), y, yhat, k)[0])
            self._f1[key] = (full, weighted_macro_f1(self.counts(pool, ids), y, yhat, k))
        return self._f1[key]

    def group_f1(self, runs: Sequence[tuple[str, SplitPreds]], pool: str, ids: Sequence[str],
                 k: int) -> tuple[float, np.ndarray]:
        """Seed mean of run_f1 over a group's runs."""
        vals = [self.run_f1(name, sp, pool, ids, k) for name, sp in runs]
        return float(np.mean([v[0] for v in vals])), np.mean([v[1] for v in vals], axis=0)


def check_labels(runs: Sequence[tuple[str, SplitPreds]], ids: Sequence[str]) -> None:
    """Paired rows must carry the same gold label in every run (same scheme, same split build)."""
    ref = take(runs[0][1], ids)[0]
    for name, sp in runs[1:]:
        if not np.array_equal(take(sp, ids)[0], ref):
            raise ValueError(f"{name}: gold labels differ from {runs[0][0]} on the paired rows (another scheme "
                             "or another data build)")


def pool_diff(bs: Bootstrap, cand: Sequence[tuple[str, SplitPreds]], base: Sequence[tuple[str, SplitPreds]],
              pool: str, k: int) -> dict[str, Any] | None:
    """Observed and resampled seed-mean macro-F1 of both groups on the paired rows of one pool."""
    ids = common_rows([sp for _, sp in (*cand, *base)])
    if not ids:
        return None
    check_labels([*cand, *base], ids)
    c_full, c_boot = bs.group_f1(cand, pool, ids, k)
    b_full, b_boot = bs.group_f1(base, pool, ids, k)
    return {"n": len(ids), "candidate_f1": c_full, "baseline_f1": b_full, "diff": c_full - b_full,
            "boot": c_boot - b_boot, "restricted": len(ids) < min(int(sp.scored.sum()) for _, sp in cand)}


def summarise(boot: np.ndarray, ci: float, lead: float) -> dict[str, float]:
    """Bootstrap mean, percentile CI and the shares of resamples above 0 and at or above the lead (fractions)."""
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": float(boot.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_gt_0": float((boot > 0).mean()), "p_ge_lead": float((boot >= lead - 1e-12).mean())}


def compare(bs: Bootstrap, cand: Sequence[tuple[str, dict[str, SplitPreds]]],
            base: Sequence[tuple[str, dict[str, SplitPreds]]], pools: Sequence[str], avg_pools: Sequence[str],
            k: int, lead: float) -> list[dict[str, Any]]:
    """One entry per pool (both groups have preds there) plus the OOD average over `avg_pools` (when every one of
    them compared). `cand`/`base`: [(run name, {split: SplitPreds})]; `lead` in fractions (0.03)."""
    rows, per_pool = [], {}
    for pool in pools:
        c = [(n, p[pool]) for n, p in cand if pool in p]
        b = [(n, p[pool]) for n, p in base if pool in p]
        if len(c) != len(cand) or len(b) != len(base) or not c or not b:
            continue
        d = pool_diff(bs, c, b, pool, k)
        if d is None:
            continue
        per_pool[pool] = d
        rows.append({"pool": pool, "pools": [pool], "n_rows": {pool: d["n"]}, "restricted": d["restricted"],
                     **{x: d[x] for x in ("candidate_f1", "baseline_f1", "diff")}, **summarise(d["boot"], bs.ci, lead)})
    if avg_pools and all(p in per_pool for p in avg_pools):
        ds = [per_pool[p] for p in avg_pools]
        rows.append({"pool": "ood_average", "pools": list(avg_pools),
                     "n_rows": {p: per_pool[p]["n"] for p in avg_pools},
                     "restricted": any(d["restricted"] for d in ds),
                     **{x: float(np.mean([d[x] for d in ds])) for x in ("candidate_f1", "baseline_f1", "diff")},
                     **summarise(np.mean([d["boot"] for d in ds], axis=0), bs.ci, lead)})
    return rows
