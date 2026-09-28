import numpy as np
import pytest

from laya_poc import calibrate as C


def _overconfident_logp(n=4000, k=5, true_T=2.5, seed=0):
    """Sample labels from softmax(z / true_T) but report softmax(z): an over-confident model."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 3, size=(n, k))
    p_true = C.softmax(z / true_T)
    y = np.array([rng.choice(k, p=row) for row in p_true])
    return C.log_softmax(z), y


def test_softmax_rows_sum_to_one_and_are_stable_for_large_logits():
    P = C.softmax(np.array([[1000.0, 1000.0], [0.0, -1000.0]]))
    assert np.allclose(P.sum(1), 1.0)
    assert np.allclose(P[0], [0.5, 0.5])


def test_fit_temperature_recovers_known_temperature():
    logp, y = _overconfident_logp()
    T = C.fit_temperature(logp, y)
    assert T == pytest.approx(2.5, rel=0.1)


def test_fit_temperature_is_near_one_for_calibrated_model():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, size=(4000, 4))
    p = C.softmax(z)
    y = np.array([rng.choice(4, p=row) for row in p])
    assert C.fit_temperature(np.log(p), y) == pytest.approx(1.0, abs=0.1)


def test_temperature_scaling_reduces_nll():
    logp, y = _overconfident_logp(seed=3)
    T = C.fit_temperature(logp, y)
    before = C.nll_at(logp, y, 1.0)
    after = C.nll_at(logp, y, T)
    assert after < before


def test_apply_temperature_accepts_probabilities_via_log():
    P = np.array([[0.7, 0.2, 0.1]])
    out = C.apply_temperature(np.log(P), 1.0)
    assert np.allclose(out, P)


def test_fit_temperature_rejects_bad_input():
    with pytest.raises(ValueError):
        C.fit_temperature(np.zeros((3, 2)), np.array([0, 1]))
    with pytest.raises(ValueError):
        C.fit_temperature(np.zeros((0, 2)), np.array([], dtype=int))


def test_is_clamped_flags_values_outside_runtime_range():
    assert C.is_clamped(0.4) and C.is_clamped(5.5)
    assert not C.is_clamped(1.3)
