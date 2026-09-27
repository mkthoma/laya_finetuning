"""Multi-split evaluate (Phase 3): one agent load, every split, tau from val, stripped_test, order invariance."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_evaluate import KEYS, FakeAgent, _fake_setup, _logits, _state

from laya_poc import evaluate as E
from laya_poc import labels as L
from laya_poc import robustness as R
from laya_poc.io_utils import read_jsonl, write_jsonl

SPLITS = ["val", "test_id", "ood_country", "stripped_test"]


class CanonLogits(dict):
    """Logits keyed by the alphabetical state, so a shuffled field order answers the same."""

    def __getitem__(self, state):
        return super().__getitem__(R.canonical_state(state))


def _as_split(rows, split):
    return [{**r, "id": f"{split}-{i:06d}", "split": split} for i, r in enumerate(rows)]


def _stripped(rows):
    """Country-only copies with TRUE labels: GB answers confidently (0.858 at T=2), US uniformly (0.1)."""
    return [{**r, "id": f"stripped_test-{i:06d}", "split": "stripped_test", "has_evidence": False,
             "state": _state(country="GB" if i % 2 == 0 else "US")} for i, r in enumerate(rows[:4])]


@pytest.fixture()
def multi(tmp_path, monkeypatch):
    pytest.importorskip("laya")
    from laya_poc import hub
    rows, logits = _fake_setup()
    data, extra = tmp_path / "data", tmp_path / "data_eval"
    write_jsonl(data / "val.jsonl", rows)
    write_jsonl(data / "test_id.jsonl", _as_split(rows, "test_id"))
    write_jsonl(data / "stripped_test.jsonl", _stripped(rows))
    write_jsonl(extra / "trap_candidates.jsonl", _as_split(rows[:6], "trap_candidates"))
    logits = CanonLogits({**logits, _state(country="GB"): _logits("dining", 8.0),
                          _state(country="US"): [0.0] * len(KEYS)})
    made, agents = [], []

    def load(source, device="cpu"):
        made.append((source, device))
        agents.append(FakeAgent(logits, T=2.0))
        return agents[-1]

    monkeypatch.setattr(hub, "load_agent", load)
    out, preds = tmp_path / "eval", tmp_path / "preds"
    argv = ["--ckpt", "hub", "--model", "laya", "--data-dir", str(data), "--out-dir", str(out),
            "--preds-dir", str(preds), "--device", "cpu", "--batch-size", "4"]
    return SimpleNamespace(data=data, trap=extra / "trap_candidates.jsonl", out=out, preds=preds, made=made,
                           agents=agents, argv=argv, tmp=tmp_path, rows=rows)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_multi_split_writes_every_split_with_one_agent_load(multi):
    rc = E.main([*multi.argv, "--splits", *SPLITS, "--optional", "ood_country",
                 "--extra", f"trap_candidates={multi.trap}"])
    assert rc == 0 and len(multi.made) == 1
    done = ["val", "test_id", "stripped_test", "trap_candidates"]
    assert sorted(p.stem for p in multi.out.glob("*.json")) == sorted(done)          # ood_country: optional, absent
    assert sorted(p.stem for p in multi.preds.glob("*.jsonl")) == sorted(done)
    val = _read(multi.out / "val.json")
    for split in done:
        res = _read(multi.out / f"{split}.json")
        for k in ("model", "ckpt", "split", "n", "n_unlabelled", "n_abstained_no_evidence", "temperature", "pre",
                  "post", "cpu_fallback", "seconds", "labels", "abstention"):
            assert k in res
        assert res["split"] == split and res["preds"] == str(multi.preds / f"{split}.jsonl")
        assert res["abstention"]["tau"] == val["tau"]["tau"] and res["abstention"]["tau_split"] == "val"
        assert len(read_jsonl(multi.preds / f"{split}.jsonl")) == res["n"]
    # every split's post pass ran at the checkpoint's own T: the pre pass's neutralisation is undone in between
    assert [c["T"] for c in multi.agents[0].calls] == [2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0]
    assert _read(multi.out / "trap_candidates.json")["rows"] == str(multi.trap)


def test_tau_is_chosen_on_val_after_temperature(multi):
    assert E.main([*multi.argv, "--splits", "val", "test_id"]) == 0
    val = _read(multi.out / "val.json")
    conf, correct = R.conf_correct(read_jsonl(multi.preds / "val.jsonl"))
    assert val["tau"] == R.choose_tau(conf, correct, 0.95, 0.70, 0.80)             # config abstain.*
    assert val["tau"]["rule"] == "accuracy" and val["tau"]["at_target_acc"]["coverage"] == pytest.approx(7 / 8)
    assert val["abstention"]["coverage"] == pytest.approx(7 / 8) and val["abstention"]["answered_acc"] == 1.0
    test = _read(multi.out / "test_id.json")
    assert test["abstention"]["tau"] == val["tau"]["tau"] and "tau" not in test


def test_stripped_test_gets_model_level_no_evidence_metrics(multi):
    assert E.main([*multi.argv, "--splits", "val", "stripped_test"]) == 0
    res = _read(multi.out / "stripped_test.json")
    assert res["n"] == 4 and res["n_abstained_no_evidence"] == 4 and res["post"] is None   # the gate: all abstain
    assert res["gate_abstain_rate"] == 1.0
    assert res["false_confident_rate"] == 0.5 and res["false_confident_threshold"] == 0.8
    assert res["abstain_rate_at_tau"] == 0.5                                        # US rows: 0.1 < tau
    ng = res["no_gate"]
    assert ng["n"] == 4 and ng["false_confident_rate"] == 0.5 and ng["abstain_rate_at_tau"] == 0.5
    recs = read_jsonl(multi.preds / "no_gate" / "stripped_test.jsonl")
    assert len(recs) == 4 and all(r["p"] is not None and not r["abstained"] for r in recs)
    assert recs[0]["answer_confidence"] > 0.8 > recs[1]["answer_confidence"]
    assert all(r["p"] is None for r in read_jsonl(multi.preds / "stripped_test.jsonl"))  # the system output


def test_without_val_there_is_no_tau(multi, capsys):
    assert E.main([*multi.argv, "--splits", "test_id", "stripped_test"]) == 0
    assert _read(multi.out / "test_id.json")["abstention"]["tau"] is None
    s = _read(multi.out / "stripped_test.json")
    assert s["abstain_rate_at_tau"] is None and s["false_confident_rate"] == 0.5
    assert "no tau" in capsys.readouterr().out


def test_a_missing_split_fails_before_loading_unless_optional(multi, capsys):
    assert E.main([*multi.argv, "--splits", "val", "ood_country", "ood_brand"]) == 1
    err = capsys.readouterr().err
    assert "ood_country" in err and "ood_brand" in err and multi.made == []
    assert E.main([*multi.argv, "--splits", "val", "ood_country", "ood_brand",
                   "--optional", "ood_country", "ood_brand"]) == 0


@pytest.mark.parametrize("extra, message", [("trap_candidates=rel/trap.jsonl", "absolute"),
                                            ("trap_candidates", "name=")])
def test_extra_needs_a_name_and_an_absolute_path(multi, capsys, extra, message):
    assert E.main([*multi.argv, "--splits", "val", "--extra", extra]) == 1
    assert message in capsys.readouterr().err and multi.made == []


def test_extra_splits_not_listed_in_splits_are_evaluated_too(multi):
    assert E.main([*multi.argv, "--splits", "val", "--extra", f"trap_candidates={multi.trap}", "--n", "3"]) == 0
    assert _read(multi.out / "trap_candidates.json")["n"] == 3 and _read(multi.out / "val.json")["n"] == 3


def test_order_invariance_json(multi, cfg):
    order = multi.tmp / "run" / "order_invariance.json"
    assert E.main([*multi.argv, "--splits", "val", "--order-invariance-out", str(order), "--order-n", "5",
                   "--order-perms", "3"]) == 0
    res = _read(order)
    oi = cfg["phase3"]["order_invariance"]
    assert res["split"] == oi["split"] == "test_id" and res["seed"] == oi["seed"]   # config defaults
    assert res["n"] == 5 and res["perms"] == 3 and len(res["agreement_per_perm"]) == 3
    assert res["mean_agreement"] == 1.0 and res["passed_99"] is True
    assert res["rows"] == str(multi.data / "test_id.jsonl") and len(multi.made) == 1
    calls = multi.agents[0].calls[-4:]                                              # fixed + 3 orders
    assert all(isinstance(s, str) for c in calls for s in c["states"])
    assert _state(country="GB") not in calls[0]["states"]                          # the gated row is left out


def test_order_invariance_seed_and_split_flags(multi):
    order = multi.tmp / "oi.json"
    assert E.main([*multi.argv, "--splits", "val", "--order-invariance-out", str(order), "--order-split", "val",
                   "--order-n", "4", "--order-perms", "1", "--order-seed", "9"]) == 0
    res = _read(order)
    assert res["split"] == "val" and res["seed"] == 9 and res["n"] == 4


def _c7(rows):
    return [{**r, "questions": json.dumps(L.question("c7"), ensure_ascii=False),
             "label": None if r["label"] is None else L.C7_MAP[r["label"]]} for r in rows]


@pytest.fixture()
def c7(multi, monkeypatch):
    from laya_poc import hub
    d7 = multi.tmp / "data_c7"
    write_jsonl(d7 / "val.jsonl", _c7(multi.rows))
    write_jsonl(d7 / "test_id.jsonl", _c7(_as_split(multi.rows, "test_id")))
    logits7 = {r["state"]: np.zeros(7).tolist() for r in multi.rows}
    made = []

    def load(source, device="cpu"):
        made.append(source)
        return FakeAgent(logits7, T=2.0)

    monkeypatch.setattr(hub, "load_agent", load)
    return SimpleNamespace(dir=d7, made=made, argv=[a if a != str(multi.data) else str(d7) for a in multi.argv])


def test_multi_split_c7_uses_the_flag_or_the_config_scheme(multi, c7, cfg, capsys):
    import yaml
    from laya_poc.config import with_overrides
    assert E.main([*c7.argv, "--splits", "val", "test_id"]) == 1              # config labels.scheme is c10
    assert "--scheme c7" in capsys.readouterr().err and c7.made == []        # refused before the load
    assert E.main([*c7.argv, "--splits", "val", "test_id", "--scheme", "c7"]) == 0
    res = _read(multi.out / "val.json")
    assert res["labels"] == L.option_keys("c7") and res["scheme"] == "c7" and res["post"]["macro_f1_9"] is None
    assert "scheme c7" in capsys.readouterr().out
    run_cfg = multi.tmp / "run_config.yaml"                                  # the matrix's per-run config
    run_cfg.write_text(yaml.safe_dump(with_overrides(cfg, {"labels.scheme": "c7"})), encoding="utf-8")
    assert E.main([*c7.argv, "--splits", "val", "--config", str(run_cfg)]) == 0
    write_jsonl(c7.dir / "test_id.jsonl", _as_split(multi.rows, "test_id"))  # c10 rows in a c7 run
    loads = len(c7.made)
    assert E.main([*c7.argv, "--splits", "val", "test_id", "--scheme", "c7"]) == 1
    assert "test_id rows carry the c10 question" in capsys.readouterr().err and len(c7.made) == loads


def test_single_split_scheme_comes_from_config_or_the_flag(multi, c7, capsys):
    single = multi.tmp / "single.json"
    base = ["--ckpt", "hub", "--rows", str(c7.dir / "val.jsonl"), "--out", str(single), "--device", "cpu"]
    assert E.main(base) == 1 and "--scheme" in capsys.readouterr().err       # unchanged: config labels.scheme
    assert E.main([*base, "--scheme", "c7"]) == 0
    assert _read(single)["labels"] == L.option_keys("c7")


def test_rows_scheme_needs_an_exact_question(tmp_path):
    from laya_poc.evaluate_splits import rows_scheme
    q10 = L.question("c10")
    crit = dict(reversed(list(q10[L.QUESTION_NAME]["criteria"].items())))
    cases = {"c10": json.dumps(q10), "c7": json.dumps(L.question("c7")),
             "reordered": json.dumps({L.QUESTION_NAME: {**q10[L.QUESTION_NAME], "criteria": crit}})}
    got = {}
    for name, q in cases.items():
        write_jsonl(tmp_path / f"{name}.jsonl", [{"id": "a", "state": "{}", "questions": q}])
        got[name] = rows_scheme(tmp_path / f"{name}.jsonl")
    write_jsonl(tmp_path / "bare.jsonl", [{"id": "a", "state": "{}"}])
    assert got == {"c10": "c10", "c7": "c7", "reordered": None} and rows_scheme(tmp_path / "bare.jsonl") is None


MULTI = ["--splits", "val", "--data-dir", "/d", "--out-dir", "/o", "--preds-dir", "/p"]
SINGLE = ["--rows", "x.jsonl", "--out", "x.json"]


def _drop(argv, flag):
    i = argv.index(flag)
    return argv[:i] + argv[i + 2:]


@pytest.mark.parametrize("argv", [
    [*MULTI, "--rows", "x.jsonl"],                             # single-split flags in multi mode
    [*MULTI, "--split", "val"],
    [*MULTI, "--out", "x.json"],
    [*MULTI, "--preds", "x.jsonl"],
    _drop(MULTI, "--out-dir"),                                 # multi mode needs its dirs
    _drop(MULTI, "--preds-dir"),
    _drop(MULTI, "--data-dir"),
    [*MULTI, "--order-n", "0"],
    [*MULTI, "--order-perms", "0"],
    [*SINGLE, "--out-dir", "o"],                               # multi-split flags in single mode
    [*SINGLE, "--preds-dir", "p"],
    [*SINGLE, "--order-invariance-out", "o.json"],
    [*SINGLE, "--extra", "a=/b"],
    [*SINGLE, "--optional", "ood_brand"],
    [*SINGLE, "--order-split", "val"],
    ["--rows", "x.jsonl"],                                     # single mode still needs --out
])
def test_parser_keeps_the_two_modes_apart(argv):
    E.parse_args(["--ckpt", "hub", *MULTI])                    # the complete forms parse
    E.parse_args(["--ckpt", "hub", *SINGLE])
    with pytest.raises(SystemExit):
        E.parse_args(["--ckpt", "hub", *argv])


def test_parser_accepts_repeated_and_multi_value_lists():
    args = E.parse_args(["--ckpt", "hub", "--splits", "val", "test_id", "--data-dir", "/d", "--out-dir", "/o",
                         "--preds-dir", "/p", "--extra", "a=/x.jsonl", "--extra", "b=/y.jsonl", "c=/z.jsonl",
                         "--optional", "a", "--optional", "b"])
    assert args.extra == ["a=/x.jsonl", "b=/y.jsonl", "c=/z.jsonl"] and args.optional == ["a", "b"]
    single = E.parse_args(["--ckpt", "hub", "--rows", "/v.jsonl", "--out", "/e.json"])
    assert single.splits is None and single.scheme is None


# ---------------------------------------------------------------- real tiny checkpoint

@pytest.mark.torch
def test_multi_split_on_the_tiny_checkpoint(tiny_ckpt_dir, synthetic_data_dir, tmp_path):
    val = read_jsonl(synthetic_data_dir / "val.jsonl")
    write_jsonl(synthetic_data_dir / "stripped_test.jsonl", _stripped(val))
    trap = tmp_path / "data_eval" / "trap_candidates.jsonl"
    write_jsonl(trap, _as_split(val[:5], "trap_candidates"))
    out, preds, order = tmp_path / "eval", tmp_path / "preds", tmp_path / "order_invariance.json"
    rc = E.main(["--ckpt", str(tiny_ckpt_dir), "--splits", "val", "test_id", "ood_country", "stripped_test",
                 "--optional", "ood_country", "--data-dir", str(synthetic_data_dir),
                 "--extra", f"trap_candidates={trap}", "--out-dir", str(out), "--preds-dir", str(preds),
                 "--device", "cpu", "--batch-size", "8", "--n", "12",
                 "--order-invariance-out", str(order), "--order-n", "6", "--order-perms", "2"])
    assert rc == 0
    val_res = _read(out / "val.json")
    assert val_res["n"] == 12 and val_res["post"]["nll_source"] == "logits" and val_res["tau"]["n"] == 12
    for split in ("val", "test_id", "trap_candidates"):
        res = _read(out / f"{split}.json")
        assert np.isfinite(res["post"]["nll"]) and res["abstention"]["tau"] == val_res["tau"]["tau"]
    s = _read(out / "stripped_test.json")
    assert s["n"] == 4 and s["no_gate"]["n"] == 4 and 0 <= s["false_confident_rate"] <= 1
    oi = _read(order)
    assert oi["n"] == 6 and len(oi["agreement_per_perm"]) == 2 and isinstance(oi["passed_99"], bool)
