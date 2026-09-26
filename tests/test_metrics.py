import numpy as np
import pytest

from laya_poc import metrics as M


def test_ece_zero_for_perfectly_calibrated_bins():
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2, dtype=float)
    assert M.ece(conf, correct, bins=15) == pytest.approx(0.0)


def test_ece_matches_hand_computation():
    conf = np.array([0.95, 0.95, 0.55, 0.55])
    correct = np.array([1.0, 0.0, 1.0, 1.0])
    # bin(0.95): acc .5 vs conf .95 -> .45 * .5 ; bin(0.55): acc 1 vs .55 -> .45 * .5
    assert M.ece(conf, correct, bins=10) == pytest.approx(0.45)


def test_ece_counts_zero_confidence_in_first_bin():
    conf = np.array([0.0, 1.0])
    correct = np.array([0.0, 1.0])
    assert M.ece(conf, correct, bins=15) == pytest.approx(0.0)


def test_brier_and_nll_on_one_hot_predictions():
    P = np.eye(3)
    y = np.array([0, 1, 2])
    assert M.brier(P, y) == pytest.approx(0.0)
    assert M.nll(P, y) == pytest.approx(0.0, abs=1e-9)


def test_brier_and_nll_on_uniform_predictions():
    P = np.full((2, 4), 0.25)
    y = np.array([0, 3])
    assert M.brier(P, y) == pytest.approx(0.75)
    assert M.nll(P, y) == pytest.approx(np.log(4))


def test_nll_is_finite_when_true_class_has_zero_probability():
    assert np.isfinite(M.nll(np.array([[1.0, 0.0]]), np.array([1])))


def test_risk_coverage_orders_by_confidence():
    conf = np.array([0.9, 0.1, 0.8, 0.7, 0.6])
    correct = np.array([1, 0, 1, 1, 0], dtype=float)
    cov, acc, at = M.risk_coverage(conf, correct)
    assert cov[-1] == pytest.approx(1.0)
    assert acc[0] == 1.0  # most confident item is correct
    assert at["acc@80"] == pytest.approx(3 / 4)
    assert at["acc@90"] == pytest.approx(3 / 5)


def test_flip_rate_and_order_invariance():
    a, b, c = np.array([0, 1, 2, 3]), np.array([0, 1, 2, 0]), np.array([0, 1, 2, 3])
    assert M.flip_rate([a, b, c]) == pytest.approx(0.25)
    assert M.order_invariance(a, [b, c]) == pytest.approx((0.75 + 1.0) / 2)


def test_report_contains_headline_metrics_and_confusion_matrix():
    P = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 1, 1, 1])
    r = M.report(P, y, ["a", "b"])
    assert r["acc"] == pytest.approx(0.75)
    assert 0.0 < r["macro_f1"] <= 1.0
    assert r["cm"] == [[1, 0], [1, 2]]
    assert set(r["per_class"]) == {"a", "b"}
    assert {"ece", "brier", "nll", "acc@80", "acc@90", "n"} <= set(r)
    assert r["n"] == 4


def test_report_uses_external_confidence_when_given():
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    y = np.array([0, 1])
    r = M.report(P, y, ["a", "b"], conf=np.array([1.0, 1.0]))
    assert r["ece"] == pytest.approx(0.0)


def test_report_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        M.report(np.eye(2), np.array([0, 1, 1]), ["a", "b"])


def test_macro_f1_can_exclude_a_class():
    y = np.array([0, 0, 1, 1, 2])
    yhat = np.array([0, 0, 1, 1, 0])
    full = M.macro_f1(y, yhat)
    without_2 = M.macro_f1(y, yhat, labels=[0, 1])
    assert without_2 > full
