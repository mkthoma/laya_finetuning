"""baseline_runs: B1 / B3 as Phase 4 runs in the Phase 3 run layout, matching the Phase 2 baselines CLI."""
import json

import numpy as np
import pytest
import yaml
from test_baselines import _noisy

from laya_poc import baseline_runs as BR
from laya_poc import baselines as B
from laya_poc import labels as L
from laya_poc.config import with_overrides
from laya_poc.io_utils import read_jsonl, write_jsonl
from synth import rows_from_records

IN_DIR = ("val", "test_id", "ood_country", "ood_script", "ood_brand")
ALL_SPLITS = [*IN_DIR, "stripped_test", "trap_candidates"]
TIMING_AND_PATHS = {"seconds", "seconds_post", "seconds_pre", "seconds_fit", "preds", "no_gate_preds", "rows"}


def _records(n, seed, scheme):
    return [(rec, lab if scheme == "c10" else L.C7_MAP[lab]) for rec, lab in _noisy(n, seed)]


def _data(tmp_path, scheme="c10", n_train=300, n_eval=100):
    """A scheme data dir (train, val, test_id, ood_*, stripped_test) and a trap file outside it."""
    d = tmp_path / f"data_{scheme}"
    null_row = {**rows_from_records([({"country": "GB"}, None)], "train", scheme)[0], "id": "train-null"}
    write_jsonl(d / "train.jsonl", rows_from_records(_records(n_train, 1, scheme), "train", scheme) + [null_row])
    for i, split in enumerate(IN_DIR):
        write_jsonl(d / f"{split}.jsonl", rows_from_records(_records(n_eval, 2 + i, scheme), split, scheme))
    test = read_jsonl(d / "test_id.jsonl")
    write_jsonl(d / "stripped_test.jsonl", [{**r, "id": f"stripped_test-{i:06d}", "split": "stripped_test",
                                             "state": json.dumps({"country": "GB"}), "has_evidence": False}
                                            for i, r in enumerate(test[:30])])
    trap = tmp_path / "data_eval" / f"trap_candidates{'' if scheme == 'c10' else '_c7'}.jsonl"
    write_jsonl(trap, rows_from_records(_records(40, 9, scheme), "trap_candidates", scheme))
    return d, trap


@pytest.fixture(scope="module")
def fast_cfg(cfg, tmp_path_factory):
    """The project config with a 2-value C grid: both CLIs read it, so their numbers stay comparable."""
    path = tmp_path_factory.mktemp("cfg") / "config.yaml"
    path.write_text(yaml.safe_dump(with_overrides(cfg, {"baselines.lr.C_grid": [0.5, 2]})), encoding="utf-8")
    return path


def _run(kind, d, run_dir, cfg_path, *extra):
    assert BR.main([kind, "--data-dir", str(d), "--run-dir", str(run_dir), "--n-jobs", "1", "--config",
                    str(cfg_path), *extra]) == 0
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (run_dir / "eval").glob("*.json")}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in TIMING_AND_PATHS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


@pytest.fixture(scope="module")
def phase2(fast_cfg, tmp_path_factory):
    """The Phase 2 CLI (python -m laya_poc.baselines) on the same c10 data: the numbers to reproduce."""
    tmp = tmp_path_factory.mktemp("phase2")
    d, trap = _data(tmp)
    out = tmp / "baselines.json"
    assert B.main(["--data-dir", str(d), "--out", str(out), "--splits", "val", "test_id", "--schemes", "c10",
                   "--n-jobs", "1", "--config", str(fast_cfg)]) == 0
    return d, trap, _json(out)["schemes"]


# ---------------------------------------------------------------- the run layout and the Phase 2 numbers

def test_tfidf_lr_writes_the_run_layout_with_the_phase2_numbers(phase2, fast_cfg, tmp_path, capsys):
    d, trap, p2 = phase2
    run = tmp_path / "runs" / "fsq-c10-B3-tfidf_lr"
    ev = _run("tfidf_lr", d, run, fast_cfg, "--extra", f"trap_candidates={trap}", "--order-invariance")
    assert sorted(ev) == sorted(ALL_SPLITS)                               # config phase3.eval_splits by default
    assert sorted(p.stem for p in (run / "preds").glob("*.jsonl")) == sorted(ALL_SPLITS)
    assert len(read_jsonl(run / "preds" / "no_gate" / "stripped_test.jsonl")) == 30
    cal, b3 = _json(run / "calibration.json"), p2["c10"]["b3_tfidf_lr"]
    assert set(cal) == {"T", "clamped", "fitted_on", "n", "extra"} and cal["fitted_on"] == "val" and cal["n"] == 100
    assert cal["extra"]["C"] == b3["C"] and cal["T"] == b3["T"] and cal["clamped"] == b3["is_clamped"]
    assert [g["C"] for g in cal["extra"]["grid"]] == [g["C"] for g in b3["grid"]]
    assert cal["extra"]["n_train_unlabelled"] == 1 and cal["extra"]["n_train"] == 300
    for split in ("val", "test_id"):                                     # every row answered: same rows as Phase 2
        for side in ("pre", "post"):
            for k in ("macro_f1", "macro_f1_9", "acc", "n"):
                assert ev[split][side][k] == b3[split][side][k], (split, side, k)
            for k in ("ece", "brier", "nll"):
                assert ev[split][side][k] == pytest.approx(b3[split][side][k], rel=1e-9, abs=1e-12)
    assert ev["val"]["temperature"] == b3["T"] and "tau" in ev["val"] and "tau" not in ev["test_id"]
    assert ev["test_id"]["abstention"]["tau"] == ev["val"]["tau"]["tau"]
    s = ev["stripped_test"]
    assert s["post"] is None and s["gate_abstain_rate"] == 1.0 and s["no_gate"]["n"] == 30
    assert s["no_gate_preds"] == str(run / "preds" / "no_gate" / "stripped_test.jsonl")
    assert ev["trap_candidates"]["rows"] == str(trap) and ev["val"]["model"] == "tfidf_lr"
    oi = _json(run / "order_invariance.json")
    assert oi["split"] == "test_id" and oi["n"] == 100 and oi["perms"] == 5 and oi["n_skipped_no_evidence"] == 0
    assert 0.0 <= oi["mean_agreement"] <= 1.0 and isinstance(oi["passed_99"], bool)
    printed = capsys.readouterr().out.strip().splitlines()
    assert any("C " in line and "T " in line for line in printed) and len(printed) <= 16


@pytest.mark.parametrize("kind", ["majority", "prior"])
def test_b1_matches_phase2_with_t_1_and_no_order_invariance(phase2, fast_cfg, tmp_path, capsys, kind):
    d, trap, p2 = phase2
    run = tmp_path / f"fsq-c10-B1-{kind}"
    ev = _run(kind, d, run, fast_cfg, "--extra", f"trap_candidates={trap}", "--order-invariance")
    ref = p2["c10"][f"b1_{kind}"]
    for split in ("val", "test_id"):
        for k in ("macro_f1", "macro_f1_9", "acc"):
            assert ev[split]["post"][k] == ref[split][k]
        assert ev[split]["post"]["ece"] == pytest.approx(ref[split]["ece"], abs=1e-9)
    cal = _json(run / "calibration.json")
    assert cal["T"] == 1.0 and cal["fitted_on"] is None and cal["clamped"] is False
    assert cal["extra"]["majority_class"] == p2["c10"]["majority_class"]
    assert not (run / "order_invariance.json").exists() and "order invariance skipped" in capsys.readouterr().out
    preds = read_jsonl(run / "preds" / "val.jsonl")
    assert len({tuple(p["p"]) for p in preds}) == 1                      # B1 ignores the input
    if kind == "majority":
        assert max(preds[0]["p"]) == pytest.approx(1.0, abs=1e-10)


def test_c7_runs_on_the_c7_data_dir_like_the_phase2_c7_collapse(tmp_path, fast_cfg):
    d10, _ = _data(tmp_path)
    d7, trap7 = _data(tmp_path, "c7")
    out = tmp_path / "baselines.json"
    assert B.main(["--data-dir", str(d10), "--out", str(out), "--schemes", "c7", "--splits", "val", "--n-jobs",
                   "1", "--config", str(fast_cfg)]) == 0
    b3 = _json(out)["schemes"]["c7"]["b3_tfidf_lr"]
    run = tmp_path / "fsq-c7-B3-tfidf_lr"
    ev = _run("tfidf_lr", d7, run, fast_cfg, "--scheme", "c7", "--extra", f"trap_candidates={trap7}")
    cal = _json(run / "calibration.json")
    assert cal["extra"]["C"] == b3["C"] and cal["T"] == b3["T"] and cal["extra"]["scheme"] == "c7"
    assert ev["val"]["post"]["macro_f1"] == b3["val"]["post"]["macro_f1"]
    assert ev["val"]["labels"] == L.option_keys("c7") and ev["val"]["scheme"] == "c7"
    assert ev["val"]["post"]["macro_f1_9"] is None


def test_scheme_mismatch_fails_before_any_fit(tmp_path, capsys):
    d10, trap = _data(tmp_path)
    run = tmp_path / "run"
    assert BR.main(["tfidf_lr", "--scheme", "c7", "--data-dir", str(d10), "--run-dir", str(run),
                    "--splits", "val"]) == 1
    assert "question" in capsys.readouterr().err and not (run / "eval").exists()


def test_a_missing_split_fails_before_any_fit_unless_optional(tmp_path, capsys, fast_cfg):
    d, trap = _data(tmp_path)
    (d / "ood_script.jsonl").unlink()
    run = tmp_path / "run"
    assert BR.main(["prior", "--data-dir", str(d), "--run-dir", str(run), "--extra", f"trap_candidates={trap}"]) == 1
    assert "ood_script" in capsys.readouterr().err and not (run / "eval").exists()
    ev = _run("prior", d, run, fast_cfg, "--extra", f"trap_candidates={trap}", "--optional", "ood_script")
    assert "ood_script" not in ev and len(ev) == len(ALL_SPLITS) - 1


def test_needs_train_jsonl_and_val_for_tfidf_lr(tmp_path, capsys, fast_cfg):
    d, _ = _data(tmp_path)
    (d / "train.jsonl").unlink()
    assert BR.main(["majority", "--data-dir", str(d), "--run-dir", str(tmp_path / "r"), "--splits", "test_id"]) == 1
    assert "train.jsonl" in capsys.readouterr().err
    d2, _ = _data(tmp_path / "two")
    (d2 / "val.jsonl").unlink()
    assert BR.main(["tfidf_lr", "--data-dir", str(d2), "--run-dir", str(tmp_path / "r2"), "--splits", "test_id",
                    "--n-jobs", "1", "--config", str(fast_cfg)]) == 1
    assert "val.jsonl" in capsys.readouterr().err


def test_without_val_in_the_splits_tfidf_lr_still_fits_on_val_but_has_no_tau(tmp_path, capsys, fast_cfg):
    d, _ = _data(tmp_path)
    ev = _run("tfidf_lr", d, tmp_path / "run", fast_cfg, "--splits", "test_id")
    assert list(ev) == ["test_id"] and ev["test_id"]["abstention"]["tau"] is None
    assert _json(tmp_path / "run" / "calibration.json")["n"] == 100 and "no tau" in capsys.readouterr().out


def test_deterministic_across_runs_and_workers(tmp_path, fast_cfg):
    d, trap = _data(tmp_path, n_train=200, n_eval=60)
    runs = [tmp_path / "a", tmp_path / "b"]
    for run, jobs in zip(runs, ("1", "2")):
        assert BR.main(["tfidf_lr", "--data-dir", str(d), "--run-dir", str(run), "--n-jobs", jobs, "--splits", "val",
                        "test_id", "stripped_test", "--config", str(fast_cfg)]) == 0
    for name in ("val", "test_id", "stripped_test"):
        assert _strip(_json(runs[0] / "eval" / f"{name}.json")) == _strip(_json(runs[1] / "eval" / f"{name}.json"))
        assert read_jsonl(runs[0] / "preds" / f"{name}.jsonl") == read_jsonl(runs[1] / "preds" / f"{name}.jsonl")
    cal = [_strip(_json(r / "calibration.json")) for r in runs]
    assert {k: v for k, v in cal[0]["extra"].items() if k not in ("n_jobs", "grid", "tfidf")} == \
        {k: v for k, v in cal[1]["extra"].items() if k not in ("n_jobs", "grid", "tfidf")}


def test_rows_without_evidence_abstain_and_unlabelled_rows_are_not_scored(tmp_path, fast_cfg):
    d, _ = _data(tmp_path)
    val = read_jsonl(d / "val.jsonl")
    gated = {**val[0], "id": "val-gated", "state": json.dumps({"country": "GB"})}
    write_jsonl(d / "val.jsonl", [*val, gated, {**val[1], "id": "val-null", "label": None}])
    ev = _run("tfidf_lr", d, tmp_path / "run", fast_cfg, "--splits", "val")
    assert ev["val"]["n"] == 102 and ev["val"]["n_abstained_no_evidence"] == 1 and ev["val"]["n_unlabelled"] == 1
    assert ev["val"]["n_scored"] == 100
    preds = {p["id"]: p for p in read_jsonl(tmp_path / "run" / "preds" / "val.jsonl")}
    assert preds["val-gated"]["abstained"] and preds["val-gated"]["p"] is None
    assert preds["val-null"]["y"] is None and not preds["val-null"]["abstained"]
    assert _json(tmp_path / "run" / "calibration.json")["n"] == 101      # T: every labelled val row, as Phase 2


def test_fit_c_is_baselines_fit_one_plus_the_fitted_model(cfg):
    train = rows_from_records(_noisy(200, 1), "train")
    val = rows_from_records(_noisy(60, 2), "val")
    y_train = np.array([L.option_keys("c10").index(r["label"]) for r in train])
    y_val = np.array([L.option_keys("c10").index(r["label"]) for r in val])
    vec = B.make_vectorizer(cfg["baselines"]["tfidf"])
    X_train, X_val = vec.fit_transform([r["state"] for r in train]), vec.transform([r["state"] for r in val])
    lr_cfg = cfg["baselines"]["lr"]
    ref = B.fit_one(X_train, y_train, {"val": X_val}, y_val, 2.0, lr_cfg, 10)
    got = BR.fit_c(X_train, y_train, X_val, y_val, 2.0, lr_cfg, 10)
    assert np.array_equal(got["logp"]["val"], ref["logp"]["val"])
    assert {k: got[k] for k in ("C", "val_macro_f1", "n_iter", "converged")} == \
        {k: ref[k] for k in ("C", "val_macro_f1", "n_iter", "converged")}
    model = got["model"]
    assert np.array_equal(B.embed_log_proba(model.predict_log_proba(X_val), model.classes_, 10), ref["logp"]["val"])


def test_fit_b1_scores():
    keys = ["a", "b", "c"]
    y = np.array([2, 1, 2, 0, 2])
    maj = BR.fit_b1("majority", y, keys)
    assert maj.score(["x", "y"]).argmax(1).tolist() == [2, 2] and maj.T == 1.0
    assert maj.score(["x"])[0].tolist() == [B.LOG_FLOOR, B.LOG_FLOOR, 0.0]
    pri = BR.fit_b1("prior", np.array([1, 1, 2]), keys)
    assert np.exp(pri.score(["x"])[0]) == pytest.approx([1e-12, 2 / 3, 1 / 3])
    assert pri.extra["majority_class"] == "b" and pri.fitted_on is None


def test_parser():
    args = BR.parse_args(["tfidf_lr", "--data-dir", "/d", "--run-dir", "/r", "--extra", "a=/x.jsonl", "--extra",
                          "b=/y.jsonl", "--splits", "val", "test_id", "--order-invariance"])
    assert args.kind == "tfidf_lr" and args.extra == ["a=/x.jsonl", "b=/y.jsonl"] and args.scheme is None
    assert args.order_invariance and args.splits == ["val", "test_id"]
    for bad in (["knn", "--data-dir", "/d", "--run-dir", "/r"], ["prior", "--run-dir", "/r"],
                ["prior", "--data-dir", "/d", "--run-dir", "/r", "--n-jobs", "0"],
                ["prior", "--data-dir", "/d", "--run-dir", "/r", "--scheme", "c3"]):
        with pytest.raises(SystemExit):
            BR.parse_args(bad)
