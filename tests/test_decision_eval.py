"""§5.12 decision rule on synthetic phase5_metrics ("phase5_metrics/1") and bench JSONs (spec P5 §3): the adapter
(seed means, pool sets, matched B4, bootstrap CIs, trap and bench inputs) and evaluate(). The fixture's numbers
mirror the Phase 3/4 seed means (docs/results/phase34_report_2026-09-28.md); no FSQ rows anywhere.
`metrics_json` / `bench_json` are reused by the phase5_report tests."""
from __future__ import annotations

import copy

import pytest

from laya_poc import decision as D
from laya_poc import decision_eval as E
from laya_poc.labels import option_keys

POOLS = ("test_id", "ood_country", "ood_script", "ood_brand")
PAIRS = [["dining", "arts"], ["sports", "outdoors"], ["services", "community"], ["retail", "services"]]
MATCH = {"laya": "modernbert_base", "laya_ml": "mmbert_small"}

# per-seed macro-F1 (test_id, ood_country, ood_script, ood_brand) and post-T ECE (same order)
F1 = {
    "E2 laya c10": ([.5602, .5688, .5626], [.4349, .4318, .4233], [.3175, .3320, .3237], [.5566, .6261, .5816]),
    "E3 laya_ml c10": ([.5842, .5897, .5915], [.4901, .4918, .5001], [.5033, .5040, .5101], [.5180, .5627, .5726]),
    "B3 tfidf_lr c10": ([.4866], [.3425], [.2786], [.4493]),
    "B4 mmbert_small c10": ([.5858, .6047, .5971], [.4999, .5039, .5028], [.4766, .4891, .4821],
                            [.5652, .5876, .5821]),
    "B4 modernbert_base c10": ([.5596, .5627, .5570], [.4083, .4131, .4140], [.3136, .3322, .3352],
                               [.4982, .5243, .5408]),
    "B1 majority c10": ([.0205], [.0191], [.0197], [.0191]),
    "B2 laya c10": ([.3126], [.2393], [.2331], [.2345]),
    "B5 qwen3_4b c10": ([.5022], [.4486], [.4627], [.4006]),
    "E4 laya c10 head": ([.3177], [.2395], [.2344], [.2369]),
}
ECE = {"E2 laya c10": ([.0552, .0688, .0611], [.07, .08, .075], [.10, .11, .12], [.05, .06, .055]),
       "E3 laya_ml c10": ([.0348, .0489, .0480], [.05, .06, .055], [.04, .045, .05], [.055, .06, .065])}
KIND = {"E": "laya", "B1": "majority", "B2": "laya", "B3": "tfidf_lr", "B4": "small_encoder", "B5": "llm"}
TRAPS = {"E2 laya c10": .38, "E3 laya_ml c10": .37, "B3 tfidf_lr c10": .31, "B4 mmbert_small c10": .37,
         "B4 modernbert_base c10": .36, "B1 majority c10": .10, "B2 laya c10": .19, "B5 qwen3_4b c10": .26,
         "E4 laya c10 head": .20}


def stat(values):
    xs = [float(v) for v in values]
    return {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "range": max(xs) - min(xs), "n": len(xs),
            "values": xs}


def _block(f1s, eces, shift=0.0):
    return {"macro_f1": stat(f1s), "macro_f1_9": stat([v + .02 for v in f1s]), "acc": stat([v + .03 for v in f1s]),
            "ece": stat([e + shift for e in eces]), "brier": stat([.55] * len(f1s)), "nll": stat([1.3] * len(f1s)),
            "acc@80": stat([v + .1 for v in f1s]), "acc@90": stat([v + .06 for v in f1s])}


def _recomputed(scheme="c10"):
    keys = option_keys(scheme)
    cm = [[40 if i == j else 2 for j in range(len(keys))] for i in range(len(keys))]
    per = {k: {"p": stat([.6]), "r": stat([.55]), "f1": stat([.57]), "support": 58} for k in keys}
    look = [{"true": a, "pred": b, "count": 2, "support": 58, "rate": 2 / 58, "rate_runs": stat([2 / 58])}
            for x, y in PAIRS for a, b in ((x, y), (y, x))]
    return {"labels": keys, "n_rows": 580, "n_scored": 580, "n_abstained": 0, "n_unlabelled": 0,
            "all": {"macro_f1": stat([.57])}, "per_class": per, "cm": cm, "cm_runs": 1, "lookalike": look}


def _ood_averages(splits, eces):
    out = {}
    for pools in (("ood_country", "ood_brand"), ("ood_country", "ood_script", "ood_brand")):
        idx = [POOLS.index(p) for p in pools]
        n = len(splits[0])
        per = [sum(splits[i][s] for i in idx) / len(idx) for s in range(n)]
        ece = [sum(eces[i][s] for i in idx) / len(idx) for s in range(n)]
        out["+".join(pools)] = {"pools": list(pools), "macro_f1": stat(per), "ece_post": stat(ece),
                                "ece_pre": stat([e + .01 for e in ece]),
                                "gap": stat([splits[0][s] - per[s] for s in range(n)])}
    return out


def group(gid):
    arm, model, scheme, *rest = gid.split()
    kind = KIND.get(arm, KIND["E"] if arm.startswith("E") else "laya")
    f1s = F1[gid]
    eces = ECE.get(gid, tuple([.03] * len(f1s[0]) for _ in POOLS))
    splits = {p: {"post": _block(f1s[i], eces[i]), "pre": _block(f1s[i], eces[i], .01),
                  "selective": {"coverage": stat([.8] * len(f1s[i]))}, "recomputed": _recomputed(scheme),
                  "flip_rate": .12 if len(f1s[i]) > 1 else None, "flip_n": 3000,
                  **({"gap": stat([t - v for t, v in zip(f1s[0], f1s[i])])} if i else {})}
              for i, p in enumerate(POOLS)}
    role = "candidate" if arm in ("E2", "E3") else ("baseline" if arm.startswith("B") else "other")
    n = len(f1s[0])
    return {"id": gid, "label": gid.replace(" head", " head-only"), "arm": arm, "model": model, "scheme": scheme,
            "subset": None, "head_only": bool(rest), "kind": kind, "zero_shot": arm == "B2",
            "eval_subset": arm == "B5", "role": role, "runs": [f"run-{gid}-{s}" for s in range(n)],
            "seeds": [11, 22, 33][:n] if n > 1 else [11], "n_runs": n,
            "ood_pools": ["ood_country", "ood_brand"] if model == "laya" else ["ood_country", "ood_script", "ood_brand"],
            "matched_small_encoder": f"B4 {MATCH[model]} c10" if role == "candidate" else None,
            "T": stat([1.0] * n), "splits": splits, "ood_average": _ood_averages(f1s, eces),
            "stripped": {"abstain_rate_at_tau": stat([1.0] * n), "false_confident_rate": stat([0.0] * n)},
            "order_invariance": {"mean_agreement": stat([.9] * n), "all_passed_99": False},
            "traps": {"status": "pending", "basis": "unannotated candidates", "acc": stat([TRAPS[gid]] * n),
                      "acc_multi": None, "n": 700, "n_multi": None}}


def _comparisons(groups):
    out = []
    for cid in ("E2 laya c10", "E3 laya_ml c10"):
        cand = groups[cid]
        for bid in ("B1 majority c10", "B3 tfidf_lr c10", "B4 mmbert_small c10", "B4 modernbert_base c10"):
            for pool in (*POOLS, "ood_average"):
                if pool == "ood_average":
                    key = "+".join(cand["ood_pools"])
                    a, b = cand["ood_average"][key]["macro_f1"]["mean"], groups[bid]["ood_average"][key]["macro_f1"]["mean"]
                else:
                    a, b = E.f1(cand, pool), E.f1(groups[bid], pool)
                d = a - b
                out.append({"candidate": cid, "baseline": bid, "baseline_kind": groups[bid]["kind"], "judged": True,
                            "pool": pool, "pools": cand["ood_pools"] if pool == "ood_average" else [pool],
                            "restricted": False, "candidate_f1": a, "baseline_f1": b, "diff": d, "mean": d,
                            "ci_low": d - .02, "ci_high": d + .02, "p_gt_0": 1.0 if d > 0 else 0.0,
                            "p_ge_lead": 1.0 if d >= .03 else 0.0})
    return out


def metrics_json(traps_annotated=False):
    groups = {gid: group(gid) for gid in F1}
    if traps_annotated:
        for g in groups.values():
            g["traps"] = {**g["traps"], "status": "ok", "basis": "annotated keep set", "n": 200, "n_multi": 30,
                          "acc_multi": stat([g["traps"]["acc"]["mean"] - .05] * g["n_runs"])}
    return {"schema": "phase5_metrics/1", "generated_at": "2026-09-28T10:00:00+00:00",
            "settings": {"candidates": ["E2", "E3"], "small_encoder_match": MATCH, "lookalike_pairs": PAIRS,
                         "pools": list(POOLS)},
            "groups": groups, "recomputed_check": {"passed": True, "compared": 27, "max_abs_diff": 0.0},
            "bootstrap": {"resamples": 1000, "ci": .95, "seed": 20260925, "comparisons": _comparisons(groups)},
            "traps": {"status": "ok" if traps_annotated else "pending",
                      "basis": "annotated keep set" if traps_annotated else "unannotated candidates",
                      "n_kept": 200 if traps_annotated else None, "note": ""}}


def bench_json(p95=(150, 120, 400, 300), rps=(40, 60, 30, 45), with_eight=True):
    """Laya torch/onnx x 4 (and 8) threads; p95 / rps per (laya torch, laya onnx, laya_ml torch, laya_ml onnx)."""
    rows = []
    for (model, backend), p, r in zip([("laya", "torch"), ("laya", "onnx"), ("laya_ml", "torch"),
                                       ("laya_ml", "onnx")], p95, rps):
        for threads in (1, 2, 4, 8) if with_eight else (1, 2, 4):
            scale = 4 / threads
            rows.append({"model": model, "backend": backend, "threads": threads, "p95_ms": p * scale,
                         "p50_ms": p * scale * .8, "batch_rps": r / scale, "cold_s": 3.0, "peak_rss_gb": 1.2,
                         "dtype": "float32", "amp": False})
    rows.append({"model": "modernbert_base", "backend": "torch", "threads": 4, "p95_ms": 60, "p50_ms": 50,
                 "batch_rps": 120, "cold_s": 2.0, "peak_rss_gb": .8})
    return {"rows": rows, "machine": {"cpu_model": "Test CPU", "physical_cores": 8, "cpu_count": 16, "ram_gb": 32,
                                      "platform": "test"},
            "onnx_check": {"laya": {"passed": True}, "laya_ml": {"passed": True}}}


@pytest.fixture(scope="module")
def cfg():
    from laya_poc.config import load_config
    from conftest import ROOT
    return load_config(ROOT / "config.yaml")


def _crit(cand, cid):
    return next(c for c in cand["criteria"] if c["id"] == cid)


def _check(cand, part, cid):
    return next(c for c in cand[part] if c["id"] == cid)


def test_phase34_like_inputs_without_bench_or_traps(cfg):
    res = E.evaluate(metrics_json(), None, cfg)
    e2, e3 = res["candidates"]
    assert (e2["id"], e3["id"]) == ("E2 laya c10", "E3 laya_ml c10") and e2["excluded_pools"] == ["ood_script"]
    c1 = _crit(e2, 1)                                   # matched B4 for laya: ModernBERT-base
    assert c1["status"] == D.PASS and c1["b4"] == "B4 modernbert_base c10"
    assert c1["value"] == pytest.approx(0.50905 - 0.46645, abs=1e-4)
    assert c1["ci"]["B4"]["ci_low"] == pytest.approx(c1["value"] - .02, abs=1e-9)
    strict = e2["c1_strict"]                            # sensitivity: best B4 of either language (mmBERT-small)
    assert strict["status"] == D.FAIL and strict["b4"] == "B4 mmbert_small c10" and strict["near_miss"] is False
    assert _crit(e3, 1)["status"] == D.FAIL and _crit(e3, 1)["b4"] == "B4 mmbert_small c10"
    assert _crit(e2, 2)["status"] == D.FAIL and _crit(e2, 2)["worst_pool"] == "ood_country"
    assert _crit(e2, 3)["status"] == D.FAIL and _crit(e2, 3)["near_miss"] is True
    assert _crit(e2, 4)["status"] == D.PENDING and _crit(e2, 5)["status"] == D.PENDING
    assert _check(e2, "stop", "a")["holds"] is False and _check(e2, "stop", "c")["holds"] is None
    assert _check(e2, "investigate", "seed_range")["flagged"] == ["ood_brand", "ood_average"]
    assert res["verdict"] == D.INVESTIGATE and not res["final"] and res["possible"] == [D.STOP, D.INVESTIGATE]
    inputs = {p["input"]: p["can_change_verdict"] for p in res["pending_inputs"]}
    assert inputs == {"bench": True, "traps": False}
    assert "PASS is out of reach" in res["statement"] and res["bench_present"] is False


def test_with_bench_and_annotated_traps_the_verdict_is_final(cfg):
    res = E.evaluate(metrics_json(traps_annotated=True), bench_json(), cfg)
    e2, e3 = res["candidates"]
    c4 = _crit(e2, 4)
    assert c4["status"] == D.PASS and c4["best"] == "B4 mmbert_small c10" and c4["multi"] == pytest.approx(.33)
    assert _crit(e3, 4)["status"] == D.PASS                # 0.37 ties the best baseline: "at least"
    c5 = _crit(e2, 5)                                   # both backends meet it: the faster (lower p95) is shown
    assert c5["status"] == D.PASS and c5["value"]["backend"] == "onnx" and c5["value"]["p95_ms"] == 120
    assert set(c5["backends"]) == {"torch", "onnx"}
    assert _check(e2, "stop", "c")["holds"] is False and _check(e2, "stop", "c")["threads"] == 8
    assert res["final"] and res["verdict"] == D.INVESTIGATE and res["pending_inputs"] == []
    assert res["machine"]["label"] == "Test CPU (8C/16T, 32 GB RAM)"


def test_bench_stop_for_every_candidate(cfg):
    slow = bench_json(p95=(3000, 2600, 3000, 2800), rps=(1, 1, 1, 1))
    res = E.evaluate(metrics_json(traps_annotated=True), slow, cfg)
    assert res["verdict"] == D.STOP and res["final"]
    assert [c["verdict"] for c in res["candidates"]] == [D.STOP, D.STOP]


def _passing_metrics():
    m = metrics_json(traps_annotated=True)
    e3 = m["groups"]["E3 laya_ml c10"]
    for s in e3["ood_average"].values():
        s["macro_f1"] = stat([.60, .60, .60])
        s["ece_post"] = stat([.04, .04, .04])
    for p in ("ood_country", "ood_script", "ood_brand"):
        e3["splits"][p]["gap"] = stat([.02, .02, .02])
    e3["splits"]["ood_brand"]["post"]["macro_f1"] = stat([.60, .61, .60])
    e3["traps"]["acc"] = stat([.40, .40, .40])
    return m


def test_a_candidate_passing_every_criterion(cfg):
    res = E.evaluate(_passing_metrics(), bench_json(), cfg)
    e3 = res["candidates"][1]
    assert [c["status"] for c in e3["criteria"]] == [D.PASS] * 5
    assert res["verdict"] == D.PASS and res["final"] and e3["verdict"] == D.PASS


def test_pass_reachable_while_bench_and_traps_are_pending(cfg):
    m = _passing_metrics()
    m["traps"]["status"] = "pending"
    res = E.evaluate(m, None, cfg)
    assert D.PASS in res["possible"] and not res["final"]
    assert {p["input"] for p in res["pending_inputs"] if p["can_change_verdict"]} == {"bench", "traps"}


def test_onnx_rows_ignored_when_the_acceptance_check_failed(cfg):
    bench = bench_json(p95=(700, 300, 700, 300), rps=(10, 20, 10, 20))
    bench["onnx_check"]["laya"] = {"passed": False}
    res = E.evaluate(metrics_json(), bench, cfg)
    e2, e3 = res["candidates"]
    assert _crit(e2, 5)["status"] == D.FAIL and _crit(e2, 5)["near_miss"] is True
    assert _crit(e3, 5)["status"] == D.PASS and _crit(e3, 5)["value"]["backend"] == "onnx"


def test_bench_rows_phase2_form_and_filters():
    bench = {"model": "laya", "backend": "torch", "rows": "data/test_id.jsonl", "cpu_count": 2,
             "platform": "Linux", "results": [{"threads": 1, "p95_ms": 300.0, "batch_rps": 9.0},
                                              {"threads": 2, "error": "timed out"},
                                              {"threads": 2, "p95_ms": 90.0, "batch_rps": 30.0, "amp": True}]}
    rows = E.bench_rows(bench)
    assert rows == [{"threads": 1, "p95_ms": 300.0, "batch_rps": 9.0, "model": "laya", "backend": "torch"}]
    assert E.bench_rows(None) is None and E.machine(bench)["label"] == "unknown CPU (?C/2T)"


def test_no_candidate_or_no_groups_is_an_error(cfg):
    with pytest.raises(ValueError, match="groups"):
        E.evaluate({"schema": "x"}, None, cfg)
    m = metrics_json()
    m["groups"] = {k: v for k, v in m["groups"].items() if v["role"] != "candidate"}
    with pytest.raises(ValueError, match="no candidate"):
        E.evaluate(m, None, cfg)


def test_missing_baseline_leaves_criterion_1_pending(cfg):
    m = copy.deepcopy(metrics_json())
    del m["groups"]["B4 modernbert_base c10"]
    e2 = E.evaluate(m, None, cfg)["candidates"][0]
    assert _crit(e2, 1)["status"] == D.PENDING and "B4" in _crit(e2, 1)["note"]
    assert _check(e2, "stop", "a")["holds"] is None


def test_no_formal_stop_and_no_investigate_trigger_is_no_pass(cfg):
    m = metrics_json(traps_annotated=True)
    for gid in ("E2 laya c10", "E3 laya_ml c10"):          # seeds agree: no seed-range trigger
        g = m["groups"][gid]
        for split in g["splits"].values():
            split["post"]["macro_f1"] = {**split["post"]["macro_f1"], "range": .01}
        for avg in g["ood_average"].values():
            avg["macro_f1"] = {**avg["macro_f1"], "range": .01}
    res = E.evaluate(m, bench_json(), cfg)
    assert res["verdict"] == D.NO_PASS and res["final"]
    assert "LDS judgement" in res["verdict"] and "every input is in" in res["statement"]


def test_bench_without_eight_threads(cfg):
    res = E.evaluate(metrics_json(), bench_json(with_eight=False), cfg)
    stop_c = _check(res["candidates"][0], "stop", "c")
    assert stop_c["holds"] is False and stop_c["threads"] == 4 and "not measured" in stop_c["note"]
    assert _crit(res["candidates"][0], 5)["status"] == D.PASS


def test_bench_cpu_schema_v2_layout(cfg):
    """bench_cpu schema_version 2: rows under `results` (`rows` is the input path), ONNX acceptance under `onnx`."""
    v1 = bench_json(p95=(700, 300, 700, 300), rps=(10, 20, 10, 20))
    v2 = {"form": "sweep", "schema_version": 2, "rows": "data/test_id.jsonl", "results": v1["rows"],
          "machine": v1["machine"], "hardware": "Test CPU (8C/16T, 32 GB RAM)", "errors": [],
          "onnx": {"laya": {"accepted": False, "acceptance": {"passed": False}},
                   "laya_ml": {"accepted": None, "acceptance": {"passed": True}}}}
    assert E.onnx_acceptance(v2, "laya") is False and E.onnx_acceptance(v2, "laya_ml") is True
    assert E.onnx_acceptance(v2, "modernbert_base") is None
    res = E.evaluate(metrics_json(), v2, cfg)
    e2, e3 = res["candidates"]
    assert _crit(e2, 5)["status"] == D.FAIL and "acceptance" in _crit(e2, 5)["note"]
    assert _crit(e3, 5)["value"]["backend"] == "onnx" and res["machine"]["label"] == v2["hardware"]
