"""§5.12 decision rule, the pure core (spec P5 §3): each criterion's pass / fail / pending edges and near misses,
the stop and investigate checks, candidate and overall verdicts, and the pending-input outlook. Plain numbers in,
no metrics JSON (test_decision_eval.py covers the adapter and evaluate())."""
from __future__ import annotations

import pytest

from laya_poc import decision as D


@pytest.fixture(scope="module")
def th():
    from laya_poc.config import load_config
    from conftest import ROOT
    return D.thresholds(load_config(ROOT / "config.yaml"))


def test_thresholds_from_config(th):
    assert th["lead"] == pytest.approx(0.03) and th["max_gap"] == pytest.approx(0.05)
    assert th["max_ece"] == 0.05 and th["stop_ece"] == 0.10 and th["stop_p95_ms"] == 1000
    assert th["seed_range"] == pytest.approx(0.03) and th["fraction"] == 0.5
    assert (th["p95_ms"], th["min_rps"], th["vcpus"]) == (500, 8, 4)


# ---------------------------------------------------------------- judge (the shared edge logic)

@pytest.mark.parametrize("value, status, near", [
    (0.03, D.PASS, None), (0.0300000000001, D.PASS, None), (0.02999999999999, D.PASS, None),
    (0.029, D.FAIL, True), (0.0151, D.FAIL, True), (0.015, D.FAIL, False), (-0.01, D.FAIL, False),
    (None, D.PENDING, None)])
def test_judge_higher_is_better(value, status, near):
    assert D.judge(value, 0.03, higher=True, margin=0.03, fraction=0.5) == (status, near)


@pytest.mark.parametrize("value, status, near", [
    (0.05, D.PASS, None), (-0.2, D.PASS, None), (0.051, D.FAIL, True), (0.0749, D.FAIL, True),
    (0.075, D.FAIL, False), (0.2, D.FAIL, False)])
def test_judge_lower_is_better(value, status, near):
    assert D.judge(value, 0.05, higher=False, margin=0.05, fraction=0.5) == (status, near)


def test_judge_without_margin_is_never_a_near_miss():
    assert D.judge(0.299, 0.30, higher=True, margin=None, fraction=0.5) == (D.FAIL, False)


# ---------------------------------------------------------------- criteria

def test_c1_passes_at_exactly_three_points_over_both(th):
    c = D.criterion_1(0.53, 0.40, 0.50, th)
    assert c["status"] == D.PASS and c["value"] == pytest.approx(0.03)
    assert c["leads"] == {"B3": pytest.approx(0.13), "B4": pytest.approx(0.03)}


def test_c1_needs_both_baselines(th):
    c = D.criterion_1(0.52, 0.40, 0.50, th)            # +12 over B3 but +2 over B4
    assert c["status"] == D.FAIL and c["near_miss"] is True and c["value"] == pytest.approx(0.02)
    assert D.criterion_1(0.51, 0.40, 0.50, th)["near_miss"] is False    # +1: more than half the margin short
    assert D.criterion_1(0.45, 0.50, 0.40, th)["status"] == D.FAIL      # below B3


def test_c1_pending_when_an_input_is_missing(th):
    c = D.criterion_1(0.53, None, 0.50, th)
    assert c["status"] == D.PENDING and c["source"] == "metrics" and "B3" in c["note"]


def test_c2_gap_edges(th):
    assert D.criterion_2({"ood_country": 0.05, "ood_brand": -0.02}, th)["status"] == D.PASS
    c = D.criterion_2({"ood_country": 0.06, "ood_brand": 0.01}, th)
    assert c["status"] == D.FAIL and c["near_miss"] is True and c["value"] == pytest.approx(0.06)
    assert c["worst_pool"] == "ood_country"
    assert D.criterion_2({"ood_country": 0.08, "ood_brand": 0.01}, th)["near_miss"] is False
    assert D.criterion_2({"ood_country": None, "ood_brand": 0.01}, th)["status"] == D.PENDING


def test_c3_ece_on_test_id_and_ood_average(th):
    assert D.criterion_3(0.05, 0.049, th)["status"] == D.PASS
    c = D.criterion_3(0.04, 0.06, th)
    assert c["status"] == D.FAIL and c["near_miss"] is True and c["value"] == pytest.approx(0.06)
    assert D.criterion_3(0.08, 0.04, th)["near_miss"] is False
    assert D.criterion_3(None, 0.04, th)["status"] == D.PENDING


def test_c4_pending_until_annotated(th):
    c = D.criterion_4(None, {}, False, th)
    assert c["status"] == D.PENDING and c["source"] == "traps"


def test_c4_at_least_the_best_baseline(th):
    bases = {"B3 tfidf_lr c10": 0.31, "B4 mmbert_small c10": 0.37, "B1 prior c10": None}
    c = D.criterion_4(0.37, bases, True, th)
    assert c["status"] == D.PASS and c["best"] == "B4 mmbert_small c10"
    c = D.criterion_4(0.369, bases, True, th)
    assert c["status"] == D.FAIL and c["near_miss"] is False     # no margin: never a near miss
    assert D.criterion_4(0.4, {}, True, th)["status"] == D.PENDING


def _row(backend, threads, p95, rps, model="laya"):
    return {"model": model, "backend": backend, "threads": threads, "p95_ms": p95, "batch_rps": rps}


def test_c5_pending_without_bench_or_four_thread_row(th):
    assert D.criterion_5(None, "laya", th)["status"] == D.PENDING
    c = D.criterion_5([_row("torch", 2, 100, 50)], "laya", th)
    assert c["status"] == D.PENDING and c["source"] == "bench" and "4-thread" in c["note"]


def test_c5_whichever_backend_meets_the_budget(th):
    rows = [_row("torch", 4, 520, 30), _row("onnx", 4, 300, 20), _row("onnx", 8, 100, 90),
            _row("torch", 4, 50, 500, model="laya_ml")]
    c = D.criterion_5(rows, "laya", th)
    assert c["status"] == D.PASS and c["value"]["backend"] == "onnx" and c["value"]["p95_ms"] == 300


def test_c5_edges_and_near_misses(th):
    assert D.criterion_5([_row("torch", 4, 500, 8)], "laya", th)["status"] == D.PASS
    c = D.criterion_5([_row("torch", 4, 600, 10)], "laya", th)
    assert c["status"] == D.FAIL and c["near_miss"] is True
    assert D.criterion_5([_row("torch", 4, 400, 5)], "laya", th)["near_miss"] is True
    assert D.criterion_5([_row("torch", 4, 800, 10)], "laya", th)["near_miss"] is False
    assert D.criterion_5([_row("torch", 4, 400, 3.9)], "laya", th)["near_miss"] is False


def test_c5_ignores_onnx_rows_when_the_acceptance_check_failed(th):
    rows = [_row("torch", 4, 700, 10), _row("onnx", 4, 300, 20)]
    c = D.criterion_5(rows, "laya", th, onnx_ok=False)
    assert c["status"] == D.FAIL and c["value"]["backend"] == "torch" and "acceptance" in c["note"]


# ---------------------------------------------------------------- stop checks

def test_stop_below_both_trained_baselines_on_id_and_ood():
    s = D.stop_below_baselines((0.50, 0.40), (0.55, 0.45), (0.60, 0.50))
    assert s["holds"] is True
    assert D.stop_below_baselines((0.50, 0.46), (0.55, 0.45), (0.60, 0.50))["holds"] is False   # OOD above B3
    assert D.stop_below_baselines((0.56, 0.40), (0.55, 0.45), (0.60, 0.50))["holds"] is False   # ID above B3
    assert D.stop_below_baselines((0.50, None), (0.55, 0.45), (0.60, 0.50))["holds"] is None


def test_stop_ece(th):
    assert D.stop_ece(0.10, 0.10, th)["holds"] is False
    assert D.stop_ece(0.04, 0.11, th)["holds"] is True
    assert D.stop_ece(0.12, None, th)["holds"] is True
    assert D.stop_ece(0.04, None, th)["holds"] is None


def test_stop_latency_with_onnx_on_eight_threads(th):
    assert D.stop_latency(None, "laya", th)["holds"] is None
    assert D.stop_latency([_row("torch", 8, 1500, 2)], "laya", th)["holds"] is None     # no ONNX row
    assert D.stop_latency([_row("onnx", 8, 1000, 2)], "laya", th)["holds"] is False
    s = D.stop_latency([_row("onnx", 4, 900, 2), _row("onnx", 8, 1200, 2)], "laya", th)
    assert s["holds"] is True and s["threads"] == 8
    # no 8-thread row (threads capped at physical cores): the best measured ONNX setting can refute the stop,
    # never confirm it
    s = D.stop_latency([_row("onnx", 2, 1500, 2), _row("onnx", 4, 1100, 2)], "laya", th)
    assert s["holds"] is None and s["threads"] == 4 and "8 threads" in s["note"] and s["source"] == "bench"
    s = D.stop_latency([_row("onnx", 4, 900, 2)], "laya", th)
    assert s["holds"] is False and "8 threads" in s["note"]


# ---------------------------------------------------------------- investigate checks

def _crit(status, near=None):
    return {"id": 0, "status": status, "near_miss": near}


def test_near_miss_check():
    ok, near, far, pend = _crit(D.PASS), _crit(D.FAIL, True), _crit(D.FAIL, False), _crit(D.PENDING)
    assert D.near_miss_check([near, ok, ok, ok, ok])["holds"] is True
    assert D.near_miss_check([far, ok, ok, ok, ok])["holds"] is False
    assert D.near_miss_check([near, near, ok, ok, ok])["holds"] is False
    assert D.near_miss_check([near, ok, ok, pend, ok])["holds"] is None
    assert D.near_miss_check([far, ok, ok, pend, ok])["holds"] is False
    assert D.near_miss_check([near, far, ok, pend, ok])["holds"] is False


def test_seed_range_check(th):
    s = D.seed_range_check({"test_id": 0.009, "ood_brand": 0.07}, th)
    assert s["holds"] is True and s["flagged"] == ["ood_brand"]
    assert D.seed_range_check({"test_id": 0.03, "ood_brand": 0.01}, th)["holds"] is False
    assert D.seed_range_check({"test_id": None}, th)["holds"] is None


def test_gate_check_is_not_applicable():
    g = D.gate_check()
    assert g["holds"] is False and "not applicable" in g["note"]


# ---------------------------------------------------------------- verdicts

P_, F_, N_ = (D.PASS, None), (D.FAIL, False), (D.FAIL, True)


@pytest.mark.parametrize("crits, stops, seed, verdict", [
    ([P_] * 5, [False] * 3, False, D.PASS),
    ([P_] * 5, [False] * 3, True, D.PASS),                    # a pass stands; seed variance is a note then
    ([F_, P_, P_, P_, P_], [True, False, False], False, D.STOP),
    ([N_, P_, P_, P_, P_], [False] * 3, False, D.INVESTIGATE),
    ([P_, P_, N_, P_, P_], [False] * 3, False, D.INVESTIGATE),
    ([N_, N_, P_, P_, P_], [False] * 3, False, D.NO_PASS),
    ([F_, P_, P_, P_, P_], [False] * 3, True, D.INVESTIGATE),
    ([F_, P_, P_, P_, P_], [False] * 3, False, D.NO_PASS),
    ([N_, P_, P_, P_, P_], [False, True, False], True, D.STOP),
])
def test_candidate_verdict(crits, stops, seed, verdict):
    assert D.candidate_verdict(crits, stops, seed) == verdict


@pytest.mark.parametrize("verdicts, overall", [
    ([D.PASS, D.STOP], D.PASS), ([D.STOP, D.STOP], D.STOP), ([D.STOP, D.INVESTIGATE], D.INVESTIGATE),
    ([D.STOP, D.NO_PASS], D.NO_PASS), ([D.NO_PASS, D.INVESTIGATE], D.INVESTIGATE), ([D.NO_PASS] * 2, D.NO_PASS),
    ([], D.NO_PASS)])
def test_overall_verdict_stop_is_global(verdicts, overall):
    assert D.overall_verdict(verdicts) == overall


def test_no_pass_wording():
    assert D.NO_PASS == "NO PASS (no formal stop condition met — LDS judgement)"


def test_near_miss_check_lists_every_pending_source():
    crits = [_crit(D.PASS), _crit(D.PASS), _crit(D.PASS), {**_crit(D.PENDING), "source": "traps"},
             {**_crit(D.PENDING), "source": "bench"}]
    assert D.near_miss_check(crits)["source"] == "bench, traps"


def test_c2_without_pools_and_config_without_thresholds(th):
    c = D.criterion_2({}, th)
    assert c["status"] == D.PENDING and c["note"] == "no included OOD pool"
    with pytest.raises(ValueError, match="phase5.decision"):
        D.thresholds({"cpu_budget": {}})
