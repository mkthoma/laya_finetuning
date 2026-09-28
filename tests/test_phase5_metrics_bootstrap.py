"""Phase 5 paired bootstrap (spec P5 §2): exactness of the vectorised macro-F1, determinism, a known difference,
B5's restricted rows, the OOD average and speed."""
from __future__ import annotations

import time

import numpy as np
import pytest

from laya_poc import phase5_metrics as M
from laya_poc.metrics import macro_f1
from laya_poc.phase5_bootstrap import (Bootstrap, compare, resample_counts, row_seed, summarise, weighted_macro_f1)
from laya_poc.phase5_load import SplitPreds
from test_phase5_metrics import write_run

K = 10


def preds(y: np.ndarray, yhat: np.ndarray, ids: list[str] | None = None, abstained=None) -> SplitPreds:
    """SplitPreds whose argmax is yhat (one-hot-ish probabilities)."""
    n = len(y)
    P = np.full((n, K), 0.01)
    P[np.arange(n), yhat] = 0.91
    ab = np.zeros(n, bool) if abstained is None else np.asarray(abstained, bool)
    P[ab] = np.nan
    ids = ids or [f"r{i}" for i in range(n)]
    return SplitPreds(tuple(ids), np.asarray(y, int), P, np.where(ab, np.nan, 0.91), ab)


def noisy(y: np.ndarray, acc: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wrong = rng.random(len(y)) > acc
    return np.where(wrong, (y + rng.integers(1, K, len(y))) % K, y)


def test_weighted_macro_f1_equals_sklearn_on_the_rows_and_on_a_resample():
    rng = np.random.default_rng(0)
    y = rng.integers(0, K - 1, 500)                   # class K-1 never true: scores 0, still averaged in
    yhat = noisy(y, 0.6, 1)
    assert weighted_macro_f1(np.ones((1, 500)), y, yhat, K)[0] == pytest.approx(
        macro_f1(y, yhat, labels=list(range(K))), abs=1e-12)
    W = resample_counts(500, 3, np.random.default_rng(5))
    assert (W.sum(1) == 500).all()
    for b in range(3):
        idx = np.repeat(np.arange(500), W[b].astype(int))
        assert weighted_macro_f1(W[b:b + 1], y, yhat, K)[0] == pytest.approx(
            macro_f1(y[idx], yhat[idx], labels=list(range(K))), abs=1e-12)


def _groups(n: int = 400, acc_c: float = 0.8, acc_b: float = 0.5, seeds=(1, 2, 3)) -> tuple:
    y = np.random.default_rng(9).integers(0, K, n)
    cand = [(f"c{s}", {"test_id": preds(y, noisy(y, acc_c, s))}) for s in seeds]
    base = [(f"b{s}", {"test_id": preds(y, noisy(y, acc_b, 10 + s))}) for s in seeds]
    return y, cand, base


def test_known_difference_is_found_with_a_ci_excluding_zero():
    y, cand, base = _groups()
    (row,) = compare(Bootstrap(resamples=500, seed=1), cand, base, ["test_id"], [], K, 0.03)
    exp_c = np.mean([macro_f1(y, sp["test_id"].yhat, labels=list(range(K))) for _, sp in cand])
    exp_b = np.mean([macro_f1(y, sp["test_id"].yhat, labels=list(range(K))) for _, sp in base])
    assert row["candidate_f1"] == pytest.approx(exp_c) and row["baseline_f1"] == pytest.approx(exp_b)
    assert row["diff"] == pytest.approx(exp_c - exp_b) and row["diff"] > 0.2
    assert 0 < row["ci_low"] < row["diff"] < row["ci_high"] and row["p_gt_0"] == 1.0 and row["p_ge_lead"] == 1.0
    assert abs(row["mean"] - row["diff"]) < 0.02 and row["n_rows"] == {"test_id": 400} and not row["restricted"]


def test_identical_groups_differ_by_exactly_zero():
    _, cand, _ = _groups()
    (row,) = compare(Bootstrap(resamples=100, seed=1), cand, cand, ["test_id"], [], K, 0.03)
    assert row["diff"] == 0 and row["ci_low"] == row["ci_high"] == 0 and row["p_gt_0"] == 0


def test_deterministic_and_seeded():
    _, cand, base = _groups()
    a = compare(Bootstrap(resamples=300, seed=4), cand, base, ["test_id"], [], K, 0.03)
    b = compare(Bootstrap(resamples=300, seed=4), cand, base, ["test_id"], [], K, 0.03)
    c = compare(Bootstrap(resamples=300, seed=5), cand, base, ["test_id"], [], K, 0.03)
    assert a == b and a[0]["ci_low"] != c[0]["ci_low"] and a[0]["diff"] == c[0]["diff"]
    assert row_seed(4, "test_id", ["a", "b"]) == row_seed(4, "test_id", ["a", "b"]) != row_seed(4, "ood_brand",
                                                                                                ["a", "b"])


def test_rows_are_paired_by_id_and_restricted_to_a_subset_baseline():
    y, cand, _ = _groups(n=300)
    keep = list(range(0, 300, 3))
    sub = preds(y[keep], y[keep], ids=[f"r{i}" for i in keep])       # a perfect B5 on 100 rows, in order or not
    shuffled = [f"r{i}" for i in keep][::-1]
    order = np.array(keep)[::-1]
    sub_shuffled = preds(y[order], y[order], ids=shuffled)
    for b in (sub, sub_shuffled):
        (row,) = compare(Bootstrap(resamples=50, seed=1), cand, [("b5", {"test_id": b})], ["test_id"], [], K, 0.03)
        assert row["restricted"] and row["n_rows"] == {"test_id": 100} and row["baseline_f1"] == 1.0
        assert row["candidate_f1"] == pytest.approx(np.mean(
            [macro_f1(y[keep], sp["test_id"].yhat[keep], labels=list(range(K))) for _, sp in cand]))


def test_gated_rows_are_left_out_and_label_mismatches_raise():
    y, cand, base = _groups(n=200)
    ab = np.zeros(200, bool)
    ab[:20] = True
    gated = [(n, {"test_id": preds(y, sp["test_id"].yhat, abstained=ab)}) for n, sp in base]
    (row,) = compare(Bootstrap(resamples=20, seed=1), cand, gated, ["test_id"], [], K, 0.03)
    assert row["n_rows"] == {"test_id": 180}
    other = [(n, {"test_id": preds((y + 1) % K, sp["test_id"].yhat)}) for n, sp in base]
    with pytest.raises(ValueError, match="gold labels differ"):
        compare(Bootstrap(resamples=20, seed=1), cand, other, ["test_id"], [], K, 0.03)


def test_ood_average_combines_pool_resamples_index_by_index():
    y = np.random.default_rng(3).integers(0, K, 300)
    pools = ("ood_country", "ood_brand")
    cand = [(f"c{s}", {p: preds(y, noisy(y, 0.8, s + j)) for j, p in enumerate(pools)}) for s in (1, 2)]
    base = [(f"b{s}", {p: preds(y, noisy(y, 0.6, 20 + s + j)) for j, p in enumerate(pools)}) for s in (1, 2)]
    rows = compare(Bootstrap(resamples=200, seed=2), cand, base, ["test_id", *pools], list(pools), K, 0.03)
    by = {r["pool"]: r for r in rows}
    assert set(by) == {"ood_country", "ood_brand", "ood_average"}          # no test_id preds: skipped
    avg = by["ood_average"]
    assert avg["pools"] == list(pools) and avg["diff"] == pytest.approx(
        (by["ood_country"]["diff"] + by["ood_brand"]["diff"]) / 2)
    assert avg["mean"] == pytest.approx((by["ood_country"]["mean"] + by["ood_brand"]["mean"]) / 2)
    assert avg["ci_high"] - avg["ci_low"] < max(r["ci_high"] - r["ci_low"] for r in rows[:2])
    none = compare(Bootstrap(resamples=20, seed=2), cand, base, list(pools), ["ood_country", "ood_script"], K, 0.03)
    assert "ood_average" not in {r["pool"] for r in none}                  # a pool of the average is missing


def test_summarise_shares_and_percentiles():
    boot = np.linspace(-0.01, 0.05, 601)
    s = summarise(boot, 0.95, 0.03)
    assert s["p_gt_0"] == pytest.approx(500 / 601) and s["p_ge_lead"] == pytest.approx(201 / 601)
    assert s["ci_low"] == pytest.approx(np.quantile(boot, 0.025)) and s["ci_high"] == pytest.approx(
        np.quantile(boot, 0.975))


def test_speed_1000_resamples_3000_rows_10_groups():
    y = np.random.default_rng(0).integers(0, K, 3000)
    groups = [[(f"g{g}s{s}", {"test_id": preds(y, noisy(y, 0.6, 100 * g + s))}) for s in range(3)] for g in range(10)]
    bs, t0 = Bootstrap(resamples=1000, seed=1), time.perf_counter()
    for g in groups[1:]:
        compare(bs, groups[0], g, ["test_id"], [], K, 0.03)
    assert time.perf_counter() - t0 < 60          # spec: < 2-3 min for the whole matrix; ~1-2 s here


def test_bootstrap_block_pairs_candidates_with_same_scheme_baselines(tmp_path):
    from laya_poc.config import load_config
    cfg = load_config()
    cfg = {**cfg, "phase5": {**cfg["phase5"], "bootstrap": {"resamples": 50, "ci": 0.9, "seed": 3}}}
    root = tmp_path / "runs"
    write_run(root, cfg, "fsq-c10-E2-laya-s11", arm="E2", model="laya", seed=11, n=60)
    write_run(root, cfg, "fsq-c7-E5-laya-s11", arm="E5", model="laya", seed=11, scheme="c7", n=60)
    write_run(root, cfg, "fsq-c10-B3-tfidf_lr", arm="B3", model="tfidf_lr", seed=None, kind="tfidf_lr", n=60)
    write_run(root, cfg, "fsq-c7-B3-tfidf_lr", arm="B3", model="tfidf_lr", seed=None, kind="tfidf_lr", scheme="c7",
              n=60)
    write_run(root, cfg, "fsq-c10-B1-prior", arm="B1", model="prior", seed=None, kind="prior", n=60)
    write_run(root, cfg, "fsq-c10-B5-qwen3_4b", arm="B5", model="qwen3_4b", seed=None, kind="llm", n=60,
              drop={("test_id", i) for i in range(0, 60, 2)})
    res = M.build(cfg, [root], {"status": "pending"}, {})
    comps = res["bootstrap"]["comparisons"]
    pairs = {(c["candidate"], c["baseline"]) for c in comps}
    assert pairs == {("E2 laya c10", "B3 tfidf_lr c10"), ("E2 laya c10", "B5 qwen3_4b c10"),
                     ("E5 laya c7", "B3 tfidf_lr c7")}                     # B1 prior is not a bootstrap baseline
    e2 = [c for c in comps if c["candidate"] == "E2 laya c10" and c["baseline"] == "B3 tfidf_lr c10"]
    assert [c["pool"] for c in e2] == ["test_id", "ood_country", "ood_script", "ood_brand", "ood_average"]
    assert e2[-1]["pools"] == ["ood_country", "ood_brand"] and all(c["judged"] for c in e2)
    b5 = next(c for c in comps if c["baseline"] == "B5 qwen3_4b c10" and c["pool"] == "test_id")
    assert b5["restricted"] and b5["n_rows"] == {"test_id": 30}
    assert not any(c["judged"] for c in comps if c["candidate"] == "E5 laya c7")
    assert res["bootstrap"]["ci"] == 0.9 and res["bootstrap"]["lead_points"] == 3.0
