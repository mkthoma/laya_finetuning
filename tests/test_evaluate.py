import json
from types import SimpleNamespace

import numpy as np
import pytest

from laya_poc import evaluate as E
from laya_poc import labels as L
from laya_poc.metrics import macro_f1

KEYS = L.option_keys("c10")
Q = L.question("c10")
QID = L.QUESTION_NAME
EVIDENCE = ["name", "address", "locality", "tel", "website"]
LAYA_OOM = "Warning: GPU memory exceeded during inference. Retrying this request on CPU..."


def _state(**fields):
    return json.dumps(dict(sorted(fields.items())), ensure_ascii=False, separators=(",", ":"))


def _row(i, label, **fields):
    return {"id": f"val-{i:06d}", "split": "val", "state": _state(country="GB", **fields),
            "questions": json.dumps(Q, ensure_ascii=False), "label": label}


class FakeAgent:
    """predict_batch stand-in: fixed logits per state, Laya's temperature lookup, 4-dp rounding, keys shuffled."""

    def __init__(self, logits_by_state, T=2.0, by_options=None, device="cpu", noisy=False):
        self.logits = logits_by_state
        self.temperature = [T, 1.0, 1.0]
        self.temperature_by_options = dict(by_options or {})
        self.lang_temperatures = {}
        self.device = SimpleNamespace(type=device)
        self.calls, self.noisy = [], noisy

    def predict_batch(self, states, questions, batch_size=None, max_len=None, head_max_len=None,
                      sort_by_length=False):
        self.calls.append({"states": list(states), "batch_size": batch_size, "max_len": max_len,
                           "head_max_len": head_max_len, "sort_by_length": sort_by_length,
                           "T": self._T()})
        if self.noisy:
            print(LAYA_OOM)
        qid = next(iter(questions))
        keys = list(questions[qid]["criteria"])
        out = []
        for s in states:
            assert isinstance(s, str)
            z = np.asarray(self.logits[s], dtype=float) / self._T()
            p = np.exp(z - z.max())
            p /= p.sum()
            probs = {k: round(float(v), 4) for k, v in reversed(list(zip(keys, p)))}
            out.append({"answers": {qid: {"choice": keys[int(p.argmax())], "probabilities": probs,
                                          "answer_confidence": round(float(p.max()), 4)}}})
        return out

    def _T(self):
        return self.temperature_by_options.get("choice:6-10", self.temperature[0])


class LayaLikeAgent(FakeAgent):
    """Routes every state through `_decode_answers` the way Laya's predict_batch does: states in a reordered
    (reversed) batch order, the logit row at `offset`, the options in the INTERNAL criteria order (reversed
    here, so a positional mix-up between logits and option keys shows), the temperature applied inside."""

    def predict_batch(self, states, questions, batch_size=None, max_len=None, head_max_len=None,
                      sort_by_length=False):
        self.calls.append({"states": list(states), "batch_size": batch_size, "sort_by_length": sort_by_length,
                           "T": self._T()})
        qid = next(iter(questions))
        crit = dict(reversed(list(questions[qid]["criteria"].items())))
        internal, ids = {qid: {"t": "choice", "crit": crit}}, [qid]
        order = list(reversed(range(len(states))))
        logits = np.array([[self.logits[states[i]][KEYS.index(k)] for k in crit] for i in order], dtype=np.float32)
        items = [{"markers": list(range(len(crit)))}]
        out = [None] * len(states)
        for row, i in enumerate(order):
            out[i] = {"answers": self._decode_answers(logits, None, items, ids, internal, row)}
        return out

    def _decode_answers(self, logits, act, items, ids, internal, offset, lang=None):
        qid, keys = ids[0], list(internal[ids[0]]["crit"])
        z = logits[offset, :len(items[0]["markers"])] / self._T()
        p = np.exp(z - z.max())
        p = p / p.sum()
        probs = {k: round(float(v), 4) for k, v in zip(keys, p)}
        return {qid: {"choice": keys[int(p.argmax())], "probabilities": probs,
                      "answer_confidence": round(float(p.max()), 4)}}


def _logits(label, margin=3.0):
    z = np.zeros(len(KEYS))
    z[KEYS.index(label)] = margin
    return z.tolist()


def _exact_nll(logits_rows, y, T):
    z = np.asarray(logits_rows, dtype=np.float64) / T
    m = z.max(1, keepdims=True)
    logp = z - m - np.log(np.exp(z - m).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


# ---------------------------------------------------------------- pure helpers

def test_evidence_mask_country_alone_is_not_evidence():
    rows = [_row(0, "dining", name="Rosa"), _row(1, "retail"), _row(2, None, tel="0123"),
            {"id": "x", "state": _state(country="GB", region="")}]
    assert E.evidence_mask(rows, EVIDENCE).tolist() == [True, False, True, False]


def test_label_indices_keep_nulls_and_reject_unknown_labels():
    rows = [_row(0, "dining"), _row(1, None), _row(2, "travel")]
    assert E.label_indices(rows, KEYS) == [KEYS.index("dining"), None, KEYS.index("travel")]
    with pytest.raises(ValueError, match="nightlife"):
        E.label_indices([_row(0, "nightlife")], KEYS)


def test_check_question_rejects_a_different_key_order():
    E.check_question([_row(0, "dining"), {"id": "no-q", "state": "{}"}], Q)  # rows without questions are fine
    crit = dict(reversed(list(Q[QID]["criteria"].items())))
    other = {QID: {**Q[QID], "criteria": crit}}
    bad = {**_row(1, "dining"), "questions": json.dumps(other)}
    with pytest.raises(ValueError, match="question"):
        E.check_question([bad], Q)


def test_probs_from_outputs_renormalise_in_option_key_order():
    out = [{"answers": {QID: {"probabilities": {k: (0.5 if k == "health" else 0.0) for k in reversed(KEYS)},
                              "answer_confidence": 0.5}}}]
    P, conf = E.probs_from_outputs(out, QID, KEYS)
    assert P.shape == (1, len(KEYS)) and P[0, KEYS.index("health")] == pytest.approx(1.0)
    assert P.sum() == pytest.approx(1.0) and conf.tolist() == [0.5]


def test_split_report_moves_detail_and_adds_the_9_class_macro_f1():
    rng = np.random.default_rng(0)
    y = np.arange(200) % len(KEYS)
    P = rng.dirichlet(np.ones(len(KEYS)), size=200)
    P[np.arange(150), y[:150]] += 2.0
    P /= P.sum(1, keepdims=True)
    rep = E.split_report(P, y, KEYS)
    assert "per_class" not in rep and "cm" not in rep
    assert set(rep["detail"]) == {"per_class", "cm"} and len(rep["detail"]["cm"]) == len(KEYS)
    nine = [i for i, k in enumerate(KEYS) if k != "event"]
    assert rep["macro_f1_9"] == pytest.approx(macro_f1(y, P.argmax(1), labels=nine))
    assert rep["macro_f1_9"] != rep["macro_f1"]
    for k in ("n", "macro_f1", "acc", "ece", "brier", "nll", "acc@80", "acc@90", "n_p_true_zero"):
        assert k in rep
    assert rep["n"] == 200


def test_split_report_counts_true_probabilities_rounded_to_zero():
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    rep = E.split_report(P, np.array([0, 0]), ["dining", "retail"])
    assert rep["n_p_true_zero"] == 1 and rep["macro_f1_9"] is None  # no event option: no 9-class variant
    assert rep["nll_source"] == "probs" and rep["nll"] == pytest.approx(-np.log(1e-12) / 2)  # the clip, bounded


def test_split_report_macro_f1_averages_over_every_option_even_an_absent_one():
    ev = KEYS.index("event")
    present = [i for i in range(len(KEYS)) if i != ev]
    y = np.array(present * 3)
    yhat = y.copy()
    yhat[:4] = [present[(present.index(v) + 1) % len(present)] for v in y[:4]]  # 4 errors, never onto event
    P = np.full((len(y), len(KEYS)), 0.01)
    P[np.arange(len(y)), yhat] = 0.91
    rep = E.split_report(P, y, KEYS)                          # event absent from truth AND predictions
    assert rep["macro_f1"] == pytest.approx(macro_f1(y, yhat, labels=list(range(len(KEYS)))))
    assert rep["macro_f1_9"] == pytest.approx(macro_f1(y, yhat, labels=present))
    assert rep["macro_f1"] < rep["macro_f1_9"]               # the 10-class headline is not a silent 9-class mean


def test_split_report_takes_nll_from_log_probabilities_when_given():
    P = np.array([[1.0, 0.0], [0.0, 1.0]])                    # 4-dp rounded: row 1's true class shows 0
    logp = np.log(np.array([[1 - 1e-5, 1e-5], [1e-6, 1 - 1e-6]]))
    y = np.array([0, 0])
    rep = E.split_report(P, y, ["dining", "retail"], logp=logp)
    assert rep["nll"] == pytest.approx(-(np.log(1 - 1e-5) + np.log(1e-6)) / 2)  # 6.9, not the clip's 13.8
    assert rep["nll_source"] == "logits" and rep["n_p_true_zero"] == 1
    assert rep["brier"] == E.split_report(P, y, ["dining", "retail"])["brier"]  # only the NLL changes source
    with pytest.raises(ValueError, match="logp"):
        E.split_report(P, y, ["dining", "retail"], logp=logp[:1])


def test_split_report_uses_answer_confidence_for_ece():
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    y = np.array([0, 1])
    assert E.split_report(P, y, ["a", "b"], conf=np.array([0.5, 0.5]))["ece"] == pytest.approx(0.5)


def test_pred_records_mark_abstained_rows_without_probabilities():
    rows = [_row(0, "dining", name="a"), _row(1, None)]
    P = np.array([[0.25, 0.75], [np.nan, np.nan]])
    recs = E.pred_records(rows, [1, None], P, np.array([0.75, np.nan]), np.array([False, True]))
    assert recs[0] == {"id": "val-000000", "y": 1, "p": [0.25, 0.75], "answer_confidence": 0.75, "abstained": False}
    assert recs[1] == {"id": "val-000001", "y": None, "p": None, "answer_confidence": None, "abstained": True}


# ---------------------------------------------------------------- both passes on a fake agent

def _fake_setup():
    rows = [_row(i, lab, name=f"n{i}") for i, lab in enumerate(["dining", "retail", "health", "travel", "arts",
                                                                 "dining", "sports", "retail"])]
    rows.append(_row(8, None, name="unlabelled"))
    rows.append(_row(9, "dining"))                       # country only: abstained, no forward pass
    wrong = {rows[5]["state"]: _logits("retail", 1.0)}   # one confidently-ish wrong answer
    logits = {r["state"]: wrong.get(r["state"], _logits(r["label"] or "outdoors")) for r in rows}
    return rows, logits


def test_evaluate_rows_runs_post_then_neutralised_pre():
    pytest.importorskip("laya")
    rows, logits = _fake_setup()
    agent = FakeAgent(logits, T=2.0, by_options={"choice:6-10": 3.0})
    res, preds = E.evaluate_rows(agent, rows, Q, evidence_fields=EVIDENCE, batch_size=4, max_len=512,
                                 head_max_len=192)
    assert [c["T"] for c in agent.calls] == [3.0, 1.0]         # post (bucket wins over temperature[0]), then pre
    sent = agent.calls[0]["states"]
    assert rows[9]["state"] not in sent and len(sent) == 9     # the no-evidence row never reaches the model
    assert all(c["sort_by_length"] and c["batch_size"] == 4 and c["max_len"] == 512 and c["head_max_len"] == 192
               for c in agent.calls)
    assert res["temperature"] == 3.0 and res["n"] == 10 and res["n_unlabelled"] == 1
    assert res["n_abstained_no_evidence"] == 1 and res["n_scored"] == 8
    assert res["pre"]["n"] == res["post"]["n"] == 8
    assert res["post"]["acc"] == res["pre"]["acc"] == pytest.approx(7 / 8)
    assert res["post"]["ece"] != res["pre"]["ece"]              # the temperature changes calibration only
    assert len(preds) == 10 and preds[9]["abstained"] and preds[8]["y"] is None
    assert preds[0]["p"] is not None and abs(sum(preds[0]["p"]) - 1) < 1e-9
    assert res["post"]["nll_source"] == res["pre"]["nll_source"] == "probs"  # no _decode_answers: no raw logits


def test_evaluate_rows_takes_nll_from_raw_logits_over_T():
    pytest.importorskip("laya")
    rows, logits = _fake_setup()
    logits[rows[3]["state"]] = _logits("retail", 30.0)         # travel row: p_true = e^-10 post, e^-30 pre
    agent = LayaLikeAgent(logits, T=2.0, by_options={"choice:6-10": 3.0})
    res, preds = E.evaluate_rows(agent, rows, Q, evidence_fields=EVIDENCE, batch_size=4, max_len=512,
                                 head_max_len=192)
    y = [KEYS.index(r["label"]) for r in rows[:8]]
    z = [logits[r["state"]] for r in rows[:8]]
    assert res["post"]["nll_source"] == res["pre"]["nll_source"] == "logits"
    assert res["post"]["nll"] == pytest.approx(_exact_nll(z, y, 3.0), rel=1e-6)  # not the 27.63 clip
    assert res["pre"]["nll"] == pytest.approx(_exact_nll(z, y, 1.0), rel=1e-6)   # above the clip: 30 nats
    assert res["post"]["n_p_true_zero"] == res["pre"]["n_p_true_zero"] == 1       # still disclosed
    assert res["post"]["acc"] == pytest.approx(6 / 8)
    assert preds[3]["p"][KEYS.index("retail")] == pytest.approx(1.0, abs=1e-3)   # P still the rounded dict
    assert "_decode_answers" not in vars(agent)                                   # the wrapper is removed


def test_capture_choice_logits_restores_an_instance_decoder():
    pytest.importorskip("laya")
    agent = LayaLikeAgent({})
    own = agent._decode_answers
    agent._decode_answers = own                                    # an instance attribute already there
    with E.capture_choice_logits(agent):
        assert agent._decode_answers is not own
    assert vars(agent)["_decode_answers"] is own
    plain = FakeAgent({})
    with E.capture_choice_logits(plain):                           # nothing to wrap: a no-op
        assert not hasattr(plain, "_decode_answers")


def test_evaluate_rows_with_nothing_to_score_reports_none():
    pytest.importorskip("laya")
    rows = [_row(0, "dining"), _row(1, None, name="x")]
    agent = FakeAgent({rows[1]["state"]: _logits("dining")})
    res, preds = E.evaluate_rows(agent, rows, Q, evidence_fields=EVIDENCE, batch_size=8, max_len=512,
                                 head_max_len=192)
    assert res["pre"] is None and res["post"] is None and res["n_scored"] == 0 and len(preds) == 2


def test_applied_choice_temperature_follows_laya_lookup():
    pytest.importorskip("laya")
    assert E.applied_choice_temperature(FakeAgent({}, T=1.7), len(KEYS)) == 1.7
    agent = FakeAgent({}, T=1.7, by_options={"choice:6-10": 1.0000159, "choice:2": 9.0})
    assert E.applied_choice_temperature(agent, len(KEYS)) == 1.0000159


# ---------------------------------------------------------------- CLI with a patched loader

@pytest.fixture()
def fake_cli(tmp_path, monkeypatch):
    pytest.importorskip("laya")
    from laya_poc import hub
    from laya_poc.io_utils import write_jsonl
    rows, logits = _fake_setup()
    data = tmp_path / "data"
    write_jsonl(data / "val.jsonl", rows)
    made = []

    def load(source, device="cpu"):
        made.append((source, device))
        return FakeAgent(logits, T=2.0, noisy=getattr(load, "noisy", False))

    monkeypatch.setattr(hub, "load_agent", load)
    return SimpleNamespace(data=data, rows=rows, made=made, load=load, tmp=tmp_path)


def test_cli_writes_the_report_and_predictions(fake_cli, capsys):
    out, preds = fake_cli.tmp / "eval.json", fake_cli.tmp / "preds.jsonl"
    rc = E.main(["--ckpt", "hub", "--model", "laya", "--rows", str(fake_cli.data / "val.jsonl"), "--out", str(out),
                 "--preds", str(preds), "--device", "cpu", "--batch-size", "4"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    for k in ("model", "ckpt", "split", "n", "n_unlabelled", "n_abstained_no_evidence", "temperature", "pre", "post",
              "cpu_fallback", "seconds"):
        assert k in res
    assert res["model"] == "laya" and res["ckpt"] == "hub" and res["split"] == "val" and res["n"] == 10
    assert res["cpu_fallback"] is False and res["temperature"] == 2.0 and res["labels"] == KEYS
    assert set(res["post"]["detail"]) == {"per_class", "cm"}
    from laya_poc.io_utils import read_jsonl
    lines = read_jsonl(preds)
    assert len(lines) == 10 and set(lines[0]) == {"id", "y", "p", "answer_confidence", "abstained"}
    assert fake_cli.made[0][1] == "cpu"
    printed = capsys.readouterr().out.strip().splitlines()
    assert 1 <= len(printed) <= 12


def test_cli_split_and_data_dir_form_and_n(fake_cli):
    out = fake_cli.tmp / "eval.json"
    rc = E.main(["--ckpt", "hub", "--split", "val", "--data-dir", str(fake_cli.data), "--out", str(out),
                 "--device", "cpu", "--n", "4"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["split"] == "val" and res["n"] == 4 and res["post"]["n"] == 4


def test_cli_flags_a_cpu_fallback(fake_cli, capsys):
    fake_cli.load.noisy = True
    out = fake_cli.tmp / "eval.json"
    assert E.main(["--ckpt", "hub", "--rows", str(fake_cli.data / "val.jsonl"), "--out", str(out),
                   "--device", "cpu"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["cpu_fallback"] is True
    assert "CPU fallback" in capsys.readouterr().out


def test_cli_refuses_an_agent_that_is_not_on_cuda(fake_cli, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    rc = E.main(["--ckpt", "hub", "--rows", str(fake_cli.data / "val.jsonl"), "--out", str(fake_cli.tmp / "e.json"),
                 "--device", "cuda"])
    assert rc == 1 and "not cuda" in capsys.readouterr().err


def test_cli_needs_rows_or_split(tmp_path, capsys):
    with pytest.raises(SystemExit):
        E.main(["--ckpt", "hub", "--out", str(tmp_path / "e.json")])


def test_cli_rejects_relative_ckpt(fake_cli, capsys):
    rc = E.main(["--ckpt", "rel/ckpt", "--rows", str(fake_cli.data / "val.jsonl"), "--out",
                 str(fake_cli.tmp / "e.json"), "--device", "cpu"])
    assert rc == 1 and "absolute" in capsys.readouterr().err


# ---------------------------------------------------------------- real tiny checkpoint

@pytest.mark.torch
def test_cli_on_the_tiny_checkpoint(tiny_ckpt_dir, synthetic_data_dir, tmp_path):
    import shutil
    from laya_poc.io_utils import read_jsonl, write_jsonl
    ckpt = shutil.copytree(tiny_ckpt_dir, tmp_path / "ckpt")
    cfg_path = ckpt / "rl_agent_config.json"
    c = json.loads(cfg_path.read_text(encoding="utf-8"))
    c = {**{k: v for k, v in c.items() if k != "temperature_by_options"}, "temperature": [2.5, 1.0, 1.0]}
    cfg_path.write_text(json.dumps(c), encoding="utf-8")
    rows = read_jsonl(synthetic_data_dir / "val.jsonl")[:12]
    rows.append({**rows[0], "id": "val-stripped", "state": _state(country="GB")})
    write_jsonl(tmp_path / "val.jsonl", rows)
    out, preds = tmp_path / "eval.json", tmp_path / "preds.jsonl"
    rc = E.main(["--ckpt", str(ckpt), "--rows", str(tmp_path / "val.jsonl"), "--out", str(out), "--preds", str(preds),
                 "--device", "cpu", "--batch-size", "4"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n"] == 13 and res["n_abstained_no_evidence"] == 1 and res["n_scored"] == 12
    assert res["temperature"] == pytest.approx(2.5) and res["cpu_fallback"] is False
    for side in ("pre", "post"):
        assert res[side]["n"] == 12 and np.isfinite(res[side]["nll"]) and 0 <= res[side]["macro_f1"] <= 1
        assert res[side]["nll_source"] == "logits"
    p = read_jsonl(preds)
    assert p[-1]["abstained"] is True and p[-1]["p"] is None and len(p[0]["p"]) == len(KEYS)


@pytest.mark.torch
def test_captured_logits_match_laya_probabilities_on_the_tiny_checkpoint(tiny_ckpt_dir, synthetic_data_dir):
    from laya_poc.hub import load_agent
    from laya_poc.io_utils import read_jsonl
    agent = load_agent(tiny_ckpt_dir, device="cpu")
    agent.temperature_by_options = {"choice:6-10": 1.7}
    states = [r["state"] for r in read_jsonl(synthetic_data_dir / "val.jsonl")[:10]]
    P, conf, Z, _ = E.predict_pass(agent, states, Q, KEYS, batch_size=4, max_len=512, head_max_len=192, label="t")
    assert Z.shape == P.shape == (10, len(KEYS)) and np.isfinite(Z).all()
    zT = Z / 1.7
    soft = np.exp(zT - zT.max(1, keepdims=True))
    soft /= soft.sum(1, keepdims=True)
    assert np.abs(soft - P).max() < 5e-4                       # option-keyed, input-ordered, pre-temperature
    assert "_decode_answers" not in vars(agent)
