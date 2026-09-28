"""§5.12 verdict outlook (spec P5 §3): with pending inputs (trap annotation, the CPU benchmark) every resolution is
enumerated, so the result says which verdicts are still reachable and which pending inputs can still change it."""
from __future__ import annotations

from laya_poc import decision as D

CRIT = {"P": (D.PASS, None), "F": (D.FAIL, False), "N": (D.FAIL, True), "?": (D.PENDING, None)}
SOURCES = {1: "metrics", 2: "metrics", 3: "metrics", 4: "traps", 5: "bench"}


def _cand(label, crits, stops=(False, False, False), seed=False):
    criteria = [{"id": i, "name": f"c{i}", "status": CRIT[c][0], "near_miss": CRIT[c][1],
                 "source": SOURCES[i] if c == "?" else None} for i, c in enumerate(crits, start=1)]
    stop = [{"id": k, "holds": h, "source": ("bench" if k == "c" else "metrics") if h is None else None}
            for k, h in zip("abc", stops)]
    inv = [D.near_miss_check(criteria), {"id": "seed_range", "holds": seed, "source": None}, D.gate_check()]
    return {"label": label, "criteria": criteria, "stop": stop, "investigate": inv}


def test_everything_in_and_one_candidate_passes():
    out = D.outlook([_cand("E2 laya c10", "FPPPP"), _cand("E3 laya_ml c10", "PPPPP")])
    assert out["verdict"] == D.PASS and out["final"] and out["possible"] == [D.PASS]
    assert out["pending_inputs"] == [] and "no pending input" in out["statement"]


def test_c1_failed_everywhere_pending_traps_and_bench_cannot_pass():
    cands = [_cand("E2 laya c10", "FPP??", (False, False, None), seed=True),
             _cand("E3 laya_ml c10", "FPF??", (False, False, None), seed=True)]
    out = D.outlook(cands)
    assert out["verdict"] == D.INVESTIGATE and not out["final"]
    assert out["possible"] == [D.STOP, D.INVESTIGATE]
    inputs = {p["input"]: p for p in out["pending_inputs"]}
    assert inputs["bench"]["can_change_verdict"] is True and inputs["traps"]["can_change_verdict"] is False
    assert "E2 laya c10 C5" in inputs["bench"]["items"] and "E3 laya_ml c10 stop (c)" in inputs["bench"]["items"]
    assert "PASS is out of reach" in out["statement"] and "E3 laya_ml c10: C1, C3" in out["statement"]


def test_a_pending_trap_result_decides_between_investigate_and_no_pass():
    out = D.outlook([_cand("E2 laya c10", "FFPPP"), _cand("E3 laya_ml c10", "NPP?P")])
    assert out["verdict"] == D.NO_PASS and out["possible"] == [D.INVESTIGATE, D.NO_PASS]
    assert out["per_candidate"][1]["possible"] == [D.INVESTIGATE, D.NO_PASS]
    assert out["pending_inputs"] == [{"input": "traps", "items": ["E3 laya_ml c10 C4"], "can_change_verdict": True}]


def test_pass_still_reachable_while_c4_and_c5_are_pending():
    out = D.outlook([_cand("E2 laya c10", "FPPPP"), _cand("E3 laya_ml c10", "PPP??", (False, False, None))])
    assert D.PASS in out["possible"] and not out["final"]
    assert "PASS is out of reach" not in out["statement"]
    assert out["per_candidate"][1]["possible"][0] == D.PASS


def test_final_despite_pending_inputs():
    out = D.outlook([_cand("E2 laya c10", "FFPP?"), _cand("E3 laya_ml c10", "FFP?P")])
    assert out["final"] and out["verdict"] == D.NO_PASS
    assert all(not p["can_change_verdict"] for p in out["pending_inputs"])
    assert "cannot change it" in out["statement"]


def test_global_stop_needs_every_candidate():
    cands = [_cand("E2 laya c10", "FPPPP", (True, False, False)), _cand("E3 laya_ml c10", "FPPPP", (False, True, False))]
    assert D.outlook(cands)["verdict"] == D.STOP
    cands[1] = _cand("E3 laya_ml c10", "FPPPP")
    out = D.outlook(cands)
    assert out["verdict"] == D.NO_PASS and [c["verdict"] for c in out["per_candidate"]] == [D.STOP, D.NO_PASS]


def test_pending_seed_range_from_missing_metrics():
    out = D.outlook([_cand("E2 laya c10", "FPPPP", seed=None)])
    assert out["possible"] == [D.INVESTIGATE, D.NO_PASS]
    assert out["pending_inputs"][0]["input"] == "metrics"
