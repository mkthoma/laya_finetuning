"""Temperature scaling fitted on `val` only (design doc §5.10, §7.8).

The doc fits one scalar with torch LBFGS on NLL. NLL is convex in 1/T, so a bounded scalar
search on log T reaches the same optimum without needing torch, and runs in the unit tests.
Writing the temperature into a checkpoint's rl_agent_config.json lives in `ckpt_config.py`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

# Laya clamps loaded temperatures to this range (laya.common.clamp_temperature).
TEMP_MIN, TEMP_MAX = 0.5, 5.0
_SEARCH_LOG_T = (np.log(0.05), np.log(50.0))


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def apply_temperature(logp: np.ndarray, T: float) -> np.ndarray:
    """Re-temper log-probabilities (equivalent to logits up to a per-row constant)."""
    return softmax(np.asarray(logp, dtype=float) / T)


def nll_at(logp: np.ndarray, y: np.ndarray, T: float) -> float:
    lp = log_softmax(np.asarray(logp, dtype=float) / T)
    return float(-lp[np.arange(len(y)), y].mean())


def fit_temperature(logp: np.ndarray, y: np.ndarray) -> float:
    """Single temperature minimising NLL of softmax(logp / T) against integer labels y."""
    logp = np.asarray(logp, dtype=float)
    y = np.asarray(y)
    if logp.ndim != 2 or len(y) != logp.shape[0] or len(y) == 0:
        raise ValueError(f"need non-empty logp [N, K] and y [N]; got {logp.shape} and {y.shape}")
    # Clip -inf from zero probabilities so the objective stays finite.
    logp = np.maximum(logp, np.log(1e-12))
    res = minimize_scalar(lambda lt: nll_at(logp, y, float(np.exp(lt))),
                          bounds=_SEARCH_LOG_T, method="bounded", options={"xatol": 1e-5})
    return float(np.exp(res.x))


def is_clamped(T: float) -> bool:
    """True when Laya would clamp this temperature at load time (signals a mis-specified model)."""
    return not TEMP_MIN <= T <= TEMP_MAX
