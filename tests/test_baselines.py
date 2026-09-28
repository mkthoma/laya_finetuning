import json

import numpy as np
import pytest

from laya_poc import baselines as B
from laya_poc import labels as L
from laya_poc.io_utils import read_jsonl, write_jsonl
from synth import rows_from_records, synthetic_records

K10 = L.option_keys("c10")
K7 = L.option_keys("c7")


def _noisy(n, seed):
    """Synthetic records with every 4th label moved to another class, so B3 is good but not perfect."""
    return [(rec, K10[(K10.index(lab) + 3) % len(K10)] if i % 4 == 0 else lab)
            for i, (rec, lab) in enumerate(synthetic_records(n, seed))]


def _data_dir(tmp_path, n_train=300, n_eval=100, extra_train=()):
    d = tmp_path / "data"
    train = rows_from_records(_noisy(n_train, 1), "train") + list(extra_train)
    write_jsonl(d / "train.jsonl", train)
    write_jsonl(d / "val.jsonl", rows_from_records(_noisy(n_eval, 2), "val"))
    write_jsonl(d / "test_id.jsonl", rows_from_records(_noisy(n_eval, 3), "test_id"))
    return d


def _run(d, out, *extra):
    rc = B.main(["--data-dir", str(d), "--out", str(out), *extra])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _strip_times(obj):
    if isinstance(obj, dict):
        return {k: _strip_times(v) for k, v in obj.items() if not k.startswith("seconds")}
    if isinstance(obj, list):
        return [_strip_times(v) for v in obj]
    return obj


# ---------------------------------------------------------------- pure helpers

def test_scheme_labels_collapse_to_c7():
    assert B.scheme_labels(["arts", "event", "dining", "services"], "c10") == ["arts", "event", "dining", "services"]
    assert B.scheme_labels(["arts", "event", "dining", "services"], "c7") == ["culture", "culture", "dining",
                                                                            "civic_services"]
    with pytest.raises(ValueError, match="scheme"):
        B.scheme_labels(["arts"], "c3")


def test_majority_breaks_ties_by_option_order_and_prior_sums_to_one():
    y = np.array([3, 1, 3, 1, 0])
    assert B.majority_index(y, 5) == 1
    p = B.class_prior(y, 5)
    assert p.tolist() == pytest.approx([0.2, 0.4, 0.0, 0.4, 0.0]) and p.sum() == pytest.approx(1.0)


def test_b1_reports_one_hot_majority_and_prior():
    y_train = np.array([0, 0, 0, 1, 2])
    rep = B.b1_reports(y_train, {"val": np.array([0, 1, 2, 0])}, ["dining", "retail", "health"])
    maj, pri = rep["b1_majority"]["val"], rep["b1_prior"]["val"]
    assert maj["acc"] == pytest.approx(0.5) and pri["acc"] == pytest.approx(0.5)
    assert maj["ece"] == pytest.approx(0.5)            # confidence 1.0 on a 50% accurate predictor
    assert pri["nll"] < maj["nll"]                      # the prior hedges, the one-hot majority does not
    assert rep["majority_class"] == "dining" and rep["prior"]["dining"] == pytest.approx(0.6)


def test_embed_log_proba_puts_absent_classes_at_the_floor():
    logp = np.log(np.array([[0.7, 0.3], [0.1, 0.9]]))
    full = B.embed_log_proba(logp, np.array([0, 2]), 3)
    assert full.shape == (2, 3) and full[0, 0] == pytest.approx(np.log(0.7)) and full[1, 2] == pytest.approx(
        np.log(0.9))
    assert (full[:, 1] == B.LOG_FLOOR).all()


def test_select_best_c_prefers_the_first_grid_value_on_ties():
    grid = [{"C": 0.5, "val_macro_f1": 0.60}, {"C": 1.0, "val_macro_f1": 0.62}, {"C": 2.0, "val_macro_f1": 0.62}]
    assert B.select_best(grid)["C"] == 1.0


def test_vectorizer_and_model_use_the_config(cfg):
    vec = B.make_vectorizer(cfg["baselines"]["tfidf"])
    assert vec.analyzer == "char_wb" and vec.ngram_range == (2, 5) and vec.min_df == 2 and vec.sublinear_tf
    assert vec.max_features == 500000
    lr = B.make_model(2.0, cfg["baselines"]["lr"])
    assert lr.C == 2.0 and lr.class_weight == "balanced" and lr.max_iter == 3000 and lr.solver == B.DEFAULT_SOLVER
    assert lr.random_state == B.RANDOM_STATE
    assert B.make_model(1.0, {**cfg["baselines"]["lr"], "solver": "lbfgs"}).solver == "lbfgs"  # config override


# ---------------------------------------------------------------- CLI

def test_cli_writes_both_schemes_and_all_baselines(tmp_path, capsys):
    null_row = {**rows_from_records([({"country": "GB"}, None)], "train")[0], "id": "train-null"}
    d = _data_dir(tmp_path, extra_train=[null_row])
    res = _run(d, tmp_path / "baselines.json", "--n-jobs", "1", "--preds-dir", str(tmp_path / "preds"))
    assert res["n_train"] == 300 and res["n_train_unlabelled"] == 1 and set(res["schemes"]) == {"c10", "c7"}
    for scheme, keys in (("c10", K10), ("c7", K7)):
        s = res["schemes"][scheme]
        assert s["labels"] == keys
        for arm in ("b1_majority", "b1_prior"):
            assert set(s[arm]) == {"val", "test_id"} and s[arm]["val"]["n"] == 100
        b3 = s["b3_tfidf_lr"]
        assert b3["C"] in [0.5, 1, 2, 4, 8] and b3["T"] > 0 and isinstance(b3["is_clamped"], bool)
        assert [g["C"] for g in b3["grid"]] == [0.5, 1, 2, 4, 8]
        for split in ("val", "test_id"):
            assert set(b3[split]) == {"pre", "post"}
            assert b3[split]["pre"]["macro_f1"] == b3[split]["post"]["macro_f1"]  # T never moves the argmax
            assert "detail" in b3[split]["post"] and "per_class" not in b3[split]["post"]
        assert b3["val"]["post"]["nll"] <= b3["val"]["pre"]["nll"] + 1e-9    # T is fitted on val NLL
        assert b3["val"]["post"]["macro_f1"] > s["b1_majority"]["val"]["macro_f1"] + 0.2
        assert b3["is_clamped"] == (not 0.5 <= b3["T"] <= 5.0)  # Laya's load-time clamp range, info only
    assert res["schemes"]["c10"]["b3_tfidf_lr"]["val"]["post"]["macro_f1_9"] is not None
    assert res["schemes"]["c7"]["b3_tfidf_lr"]["val"]["post"]["macro_f1_9"] is None
    preds = read_jsonl(tmp_path / "preds" / "c10_b3_tfidf_lr_test_id.jsonl")
    assert len(preds) == 100 and set(preds[0]) == {"id", "y", "p", "answer_confidence"}
    assert abs(sum(preds[0]["p"]) - 1) < 1e-9
    printed = capsys.readouterr().out.strip().splitlines()
    assert 1 <= len(printed) <= 8


def test_cli_is_deterministic_across_runs_and_workers(tmp_path):
    d = _data_dir(tmp_path, n_train=200, n_eval=60)
    a = _run(d, tmp_path / "a.json", "--n-jobs", "1")
    b = _run(d, tmp_path / "b.json", "--n-jobs", "2")
    assert _strip_times(a["schemes"]) == _strip_times(b["schemes"])


def test_cli_subset_of_schemes_and_splits(tmp_path):
    d = _data_dir(tmp_path, n_train=200, n_eval=60)
    res = _run(d, tmp_path / "b.json", "--schemes", "c10", "--splits", "val", "--n-jobs", "1")
    assert set(res["schemes"]) == {"c10"} and set(res["schemes"]["c10"]["b1_majority"]) == {"val"}
    assert "test_id" not in res["schemes"]["c10"]["b3_tfidf_lr"]


def test_cli_needs_train_jsonl(tmp_path, capsys):
    d = _data_dir(tmp_path)
    (d / "train.jsonl").unlink()
    assert B.main(["--data-dir", str(d), "--out", str(tmp_path / "b.json")]) == 1
    assert "train.jsonl" in capsys.readouterr().err


def test_cli_always_selects_c_and_t_on_val(tmp_path):
    d = _data_dir(tmp_path, n_train=200, n_eval=60)
    res = _run(d, tmp_path / "b.json", "--splits", "test_id", "--schemes", "c7", "--n-jobs", "1")
    b3 = res["schemes"]["c7"]["b3_tfidf_lr"]
    assert set(res["schemes"]["c7"]["b1_majority"]) == {"test_id"} and "val" not in b3
    assert b3["grid"][0]["val_macro_f1"] is not None and b3["T"] > 0


def test_cli_needs_val_jsonl(tmp_path, capsys):
    d = _data_dir(tmp_path)
    (d / "val.jsonl").unlink()
    assert B.main(["--data-dir", str(d), "--out", str(tmp_path / "b.json"), "--splits", "test_id"]) == 1
    assert "val.jsonl" in capsys.readouterr().err
