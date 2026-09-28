"""baseline_eval: the Phase 4 baselines' per-split outputs in EXACTLY the evaluate multi-split schema (spec §1-§2)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_evaluate import KEYS, FakeAgent, Q, _state

from laya_poc import baseline_eval as BE
from laya_poc import evaluate as E
from laya_poc import robustness as R
from laya_poc.calibrate import fit_temperature, is_clamped, softmax
from laya_poc.io_utils import read_jsonl, write_jsonl

EVIDENCE = ["name", "address", "locality", "tel", "website"]
ABSTAIN = {"target_acc": 0.95, "min_coverage": 0.70, "fallback_coverage": 0.80}
TIMING_AND_PATHS = {"seconds", "seconds_post", "seconds_pre", "preds", "no_gate_preds"}


def _rows(split, n=40, seed=0, stripped=False):
    """Rows + raw logits: every 7th row country-only (gated), every 9th unlabelled, most true classes boosted."""
    rng = np.random.default_rng(seed)
    rows, Z = [], []
    for i in range(n):
        label = None if (i % 9 == 4 and not stripped) else KEYS[i % len(KEYS)]
        gated = stripped or i % 7 == 3
        fields = {"country": f"{split[:4]}{i}"} if gated else {"country": "GB", "name": f"Place {split} {i}"}
        rows.append({"id": f"{split}-{i:06d}", "split": split, "state": _state(**fields),
                     "questions": json.dumps(Q, ensure_ascii=False), "label": label})
        z = rng.normal(0.0, 1.0, len(KEYS))
        if label is not None and i % 5:
            z[KEYS.index(label)] += 3.0
        Z.append(z)
    return rows, np.array(Z)


def _outputs(split, rows, Z, T=1.7, **kw):
    return BE.split_outputs(rows=rows, keys=KEYS, scores=Z, T=T, split=split, model="m", ckpt="c",
                            evidence_fields=EVIDENCE, abstain_cfg=ABSTAIN, **kw)


def _strip(obj):
    """Drop timings and file paths (they differ between two writers of the same numbers)."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in TIMING_AND_PATHS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


# ---------------------------------------------------------------- split_outputs

def test_split_outputs_gates_counts_and_scores_like_evaluate():
    rows, Z = _rows("test_id")
    payload, preds = _outputs("test_id", rows, Z)
    answered = E.evidence_mask(rows, EVIDENCE)
    y = E.label_indices(rows, KEYS)
    scored = answered & np.array([v is not None for v in y])
    assert payload["n"] == 40 and payload["n_abstained_no_evidence"] == int((~answered).sum()) == 6
    assert payload["n_unlabelled"] == 4 and payload["n_scored"] == int(scored.sum())
    assert payload["temperature"] == 1.7 and payload["temperature_pre"] == 1.0
    assert payload["scheme"] == "c10" and payload["labels"] == KEYS and "tau" not in payload
    yy = np.array([v for v, s in zip(y, scored) if s])
    for side, t in (("post", 1.7), ("pre", 1.0)):
        want = E.split_report(softmax(Z[scored] / t), yy, KEYS, logp=E.log_softmax(Z[scored], t))
        assert payload[side]["nll_source"] == "logits"
        for k in ("macro_f1", "macro_f1_9", "acc", "ece", "brier", "nll", "acc@80", "n_p_true_zero"):
            assert payload[side][k] == pytest.approx(want[k], abs=1e-12)
    assert payload["post"]["macro_f1"] == payload["pre"]["macro_f1"]          # T never moves the argmax
    assert payload["abstention"]["tau"] is None and payload["abstention"]["tau_split"] == "val"
    assert [p["abstained"] for p in preds] == (~answered).tolist() and len(preds) == 40
    for p, z, a, yi in zip(preds, Z, answered, y):
        assert p["y"] == yi
        if a:
            assert p["p"] == pytest.approx(softmax(z[None] / 1.7)[0].tolist(), abs=1e-15)
            assert p["answer_confidence"] == pytest.approx(max(p["p"]), abs=1e-15)
        else:
            assert p["p"] is None and p["answer_confidence"] is None


def test_val_chooses_tau_and_other_splits_reuse_it():
    rows, Z = _rows("val", n=60, seed=1)
    val, preds = _outputs("val", rows, Z)
    assert val["tau"] == R.choose_tau(*R.conf_correct(preds), **ABSTAIN)
    assert val["abstention"]["tau"] == val["tau"]["tau"] and val["abstention"]["tau_rule"] == val["tau"]["rule"]
    trows, TZ = _rows("test_id", seed=2)
    test, _ = _outputs("test_id", trows, TZ, tau=val["tau"])
    assert "tau" not in test and test["abstention"]["tau"] == val["tau"]["tau"]
    assert test["abstention"]["tau_rule"] == val["tau"]["rule"]
    as_number, _ = _outputs("test_id", trows, TZ, tau=val["tau"]["tau"])
    assert as_number["abstention"]["coverage"] == test["abstention"]["coverage"]
    assert as_number["abstention"]["tau_rule"] is None
    given, _ = _outputs("val", rows, Z, tau={"tau": 0.5, "rule": "coverage"})   # a given tau is used as is
    assert given["tau"] == {"tau": 0.5, "rule": "coverage"} and given["abstention"]["tau"] == 0.5


def test_stripped_test_metrics_come_from_the_gate_bypassed_scores():
    rows, Z = _rows("stripped_test", n=20, seed=3, stripped=True)
    Z[::2, 0] += 8.0                                                    # half the rows answer confidently
    tau = {"tau": 0.6, "rule": "accuracy"}
    res, preds = _outputs("stripped_test", rows, Z, T=1.0, tau=tau)
    assert res["n_abstained_no_evidence"] == 20 and res["post"] is None and res["pre"] is None
    assert res["gate_abstain_rate"] == 1.0 and all(p["abstained"] for p in preds)
    conf = softmax(Z).max(1)
    assert res["false_confident_rate"] == pytest.approx(float((conf > 0.8).mean())) and (conf > 0.8).mean() >= 0.5
    assert res["abstain_rate_at_tau"] == pytest.approx(float((conf < 0.6).mean()))
    assert res["no_gate"]["n"] == 20 and res["no_gate"]["tau"] == 0.6 and res["no_gate_preds"] is None
    assert res["no_gate"]["mean_answer_confidence"] == pytest.approx(conf.mean())
    assert res["false_confident_threshold"] == 0.8
    other, _ = _outputs("stripped_test", rows, Z, T=1.0, tau=tau, no_gate_scores=np.zeros_like(Z))
    assert other["false_confident_rate"] == 0.0 and other["no_gate"]["mean_answer_confidence"] == pytest.approx(0.1)
    recs = BE.no_gate_records(rows=rows, keys=KEYS, scores=Z, T=1.0)
    assert [r["answer_confidence"] for r in recs] == pytest.approx(conf.tolist())
    assert all(not r["abstained"] and r["y"] == KEYS.index(row["label"]) for r, row in zip(recs, rows))


def test_meta_adds_and_overrides_info_fields_but_never_computed_ones():
    rows, Z = _rows("test_id", n=10)
    res, _ = _outputs("test_id", rows, Z, meta={"rows": "/d/test_id.jsonl", "device": "cuda", "batch_size": 64,
                                                 "subset": True, "seconds_post": 3.5})
    assert res["rows"] == "/d/test_id.jsonl" and res["device"] == "cuda" and res["batch_size"] == 64
    assert res["subset"] is True and res["seconds_post"] == 3.5 and list(res)[-1] == "seconds"
    plain, _ = _outputs("test_id", rows, Z)
    assert plain["device"] == "cpu" and plain["rows"] is None and plain["cpu_fallback"] is False
    for key in ("n", "post", "abstention", "temperature", "preds"):
        with pytest.raises(ValueError, match=key):
            _outputs("test_id", rows, Z, meta={key: 1})


def test_split_outputs_validates_its_inputs():
    rows, Z = _rows("test_id", n=10)
    with pytest.raises(ValueError, match=r"\[10, 10\]"):
        _outputs("test_id", rows, Z[:, :9])
    bad = Z.copy()
    bad[3] = np.nan                                                     # row 3 is gated: never read
    _outputs("test_id", rows, bad)
    bad[0, 0] = -np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _outputs("test_id", rows, bad)
    for T in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="temperature"):
            _outputs("test_id", rows, Z, T=T)
    with pytest.raises(ValueError, match="no rows"):
        _outputs("test_id", [], np.zeros((0, len(KEYS))))
    with pytest.raises(ValueError, match="nightlife"):
        _outputs("test_id", [{**rows[0], "label": "nightlife"}], Z[:1])
    with pytest.raises(TypeError, match="tau"):
        _outputs("test_id", rows, Z, tau="0.5")
    with pytest.raises(ValueError, match="tau"):
        _outputs("test_id", rows, Z, tau={"rule": "accuracy"})


# ---------------------------------------------------------------- write_split

def test_write_split_writes_the_run_layout(tmp_path):
    out, pdir = tmp_path / "eval", tmp_path / "preds"
    rows, Z = _rows("stripped_test", n=8, stripped=True)
    res, preds = _outputs("stripped_test", rows, Z)
    ng = BE.no_gate_records(rows=rows, keys=KEYS, scores=Z, T=1.7)
    final = BE.write_split(out, pdir, "stripped_test", res, preds, ng)
    assert json.loads((out / "stripped_test.json").read_text(encoding="utf-8")) == final
    assert final["preds"] == str(pdir / "stripped_test.jsonl") and read_jsonl(pdir / "stripped_test.jsonl") == preds
    assert final["no_gate_preds"] == str(pdir / "no_gate" / "stripped_test.jsonl")
    assert read_jsonl(pdir / "no_gate" / "stripped_test.jsonl") == ng and res["no_gate_preds"] is None  # not mutated
    with pytest.raises(ValueError, match="no_gate_records"):
        BE.write_split(out, pdir, "stripped_test", res, preds)
    vrows, VZ = _rows("val", n=12)
    val, vpreds = _outputs("val", vrows, VZ)
    with pytest.raises(ValueError, match="stripped_test only"):
        BE.write_split(out, pdir, "val", val, vpreds, ng)
    with pytest.raises(ValueError, match="not 'test_id'"):
        BE.write_split(out, pdir, "test_id", val, vpreds)
    with pytest.raises(ValueError, match="11 preds records for 12 rows"):
        BE.write_split(out, pdir, "val", val, vpreds[:-1])


def test_write_outputs_is_the_three_steps_in_one_call(tmp_path):
    rows, Z = _rows("stripped_test", n=8, stripped=True)
    kw = dict(rows=rows, keys=KEYS, scores=Z, T=1.3, split="stripped_test", model="m", ckpt="c",
              evidence_fields=EVIDENCE, abstain_cfg=ABSTAIN, tau={"tau": 0.3, "rule": "coverage"})
    one = BE.write_outputs(tmp_path / "a" / "eval", tmp_path / "a" / "preds", **kw)
    res, preds = BE.split_outputs(**kw)
    two = BE.write_split(tmp_path / "b" / "eval", tmp_path / "b" / "preds", "stripped_test", res, preds,
                         BE.no_gate_records(rows=rows, keys=KEYS, scores=Z, T=1.3))
    assert _strip(one) == _strip(two)
    for name in ("stripped_test.jsonl", "no_gate/stripped_test.jsonl"):
        assert read_jsonl(tmp_path / "a" / "preds" / name) == read_jsonl(tmp_path / "b" / "preds" / name)


# ---------------------------------------------------------------- temperature

def test_fit_T_matches_calibrate_and_survives_scores_far_below_zero():
    rng = np.random.default_rng(5)
    Z = rng.normal(0.0, 3.0, size=(4000, 5))
    y = np.array([rng.choice(5, p=p) for p in softmax(Z / 2.0)])      # labels drawn at a true T of 2
    T, clamped = BE.fit_T(Z, y)
    assert T == fit_temperature(Z, y) and T == pytest.approx(2.0, rel=0.1)   # the floor never binds: as is
    assert clamped is is_clamped(T) is False
    logp = np.maximum(np.log(softmax(Z)), BE.LOG_FLOOR)                  # B3-style floored log-probabilities
    assert BE.fit_T(logp, y)[0] == fit_temperature(logp, y)
    low = Z - 200.0                                                     # e.g. summed token log-probabilities
    assert BE.fit_T(low, y)[0] == pytest.approx(T, rel=1e-4)
    assert abs(fit_temperature(low, y) - T) > 0.5                       # the log(1e-12) floor flattens them
    assert BE.fit_T(Z * 20.0, y) == (pytest.approx(T * 20.0, rel=1e-4), True)
    with pytest.raises(ValueError, match="labels"):
        BE.fit_T(Z, y + 5)
    for bad in (np.nan, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            BE.fit_T(np.where(np.eye(5)[y[:5]] > 0, bad, Z[:5]), y[:5])
    with pytest.raises(ValueError, match="non-empty"):
        BE.fit_T(Z[:0], y[:0])


def test_calibration_record_abstain_settings_and_order_config(cfg):
    rec = BE.calibration_record(T=np.float64(0.97), clamped=np.bool_(False), n=3000, extra={"C": 8})
    assert rec == {"T": 0.97, "clamped": False, "fitted_on": "val", "n": 3000, "extra": {"C": 8}}
    assert type(rec["T"]) is float and type(rec["clamped"]) is bool
    assert BE.calibration_record(T=1.0, clamped=False, n=0, fitted_on=None)["fitted_on"] is None
    assert BE.abstain_settings(cfg["abstain"]) == BE.abstain_settings(None) == ABSTAIN
    assert BE.order_config(cfg) == {"split": "test_id", "n": 1000, "perms": 5, "seed": 20260925}


# ---------------------------------------------------------------- order invariance

def _first_field_class(state):
    """An order-SENSITIVE model: the class follows the first JSON key of the state."""
    return len(next(iter(json.loads(state)))) % len(KEYS)


class FirstField(dict):
    def __getitem__(self, state):
        return [3.0 if i == _first_field_class(state) else 0.0 for i in range(len(KEYS))]


def _order_rows(n=30):
    return [{"id": f"test_id-{i:06d}", "state": _state(country="GB", name=f"P{i}", locality="Leeds",
                                                         tel=f"0{i}", website=f"w{i}.com")} for i in range(n)]


def test_order_invariance_fn_equals_robustness_order_invariance():
    rows = _order_rows()
    agent = FakeAgent(FirstField(), T=1.0)
    ref = R.order_invariance(agent, rows, Q, n=20, perms=3, seed=7, batch_size=8, max_len=512, head_max_len=192)
    got = BE.order_invariance_fn(lambda s: np.array([_first_field_class(x) for x in s]), rows, n=20, perms=3,
                                 seed=7)
    assert set(got) == set(ref) and _strip(got) == _strip(ref)
    assert got["mean_agreement"] < 1.0 and got["passed_99"] is False
    same = BE.order_invariance_fn(lambda s: np.zeros(len(s), dtype=int), rows, n=5, perms=2, seed=7)
    assert same["mean_agreement"] == 1.0 and same["passed_99"] is True and same["n"] == 5
    with pytest.raises(ValueError, match="1 predictions for 5 states"):
        BE.order_invariance_fn(lambda s: np.zeros(1, dtype=int), rows, n=5, perms=1, seed=7)
    with pytest.raises(ValueError, match="perms >= 1"):
        BE.order_invariance_fn(lambda s: np.zeros(len(s)), rows, n=5, perms=0, seed=7)


def test_order_invariance_payload_skips_rows_without_evidence():
    rows = [{"id": "g", "state": _state(country="GB")}, *_order_rows(6)]
    seen = []

    def predict(states):
        seen.extend(states)
        return np.zeros(len(states), dtype=int)

    res = BE.order_invariance_payload(predict, rows, model="tfidf_lr", ckpt="c", split="test_id",
                                      rows_path="/d/test_id.jsonl", evidence_fields=EVIDENCE, n=10, perms=2, seed=1)
    assert res["n"] == 6 and res["n_skipped_no_evidence"] == 1 and _state(country="GB") not in seen
    assert [res[k] for k in ("model", "ckpt", "split", "rows")] == ["tfidf_lr", "c", "test_id", "/d/test_id.jsonl"]
    assert all(isinstance(s, str) for s in seen)


# ---------------------------------------------------------------- identity with evaluate_splits

class ExactAgent:
    """A Laya-shaped agent whose probabilities are EXACTLY softmax(logits / T) (no 4-dp rounding), decoded through
    `_decode_answers` so evaluate captures the raw logits, as with a real Laya agent."""

    def __init__(self, logits_by_state, T):
        self.logits, self.temperature = logits_by_state, [T, 1.0, 1.0]
        self.temperature_by_options, self.lang_temperatures = {}, {}
        self.device = SimpleNamespace(type="cpu")

    def predict_batch(self, states, questions, batch_size=None, max_len=None, head_max_len=None,
                      sort_by_length=False):
        qid = next(iter(questions))
        crit = questions[qid]["criteria"]
        Z = np.array([self.logits[s] for s in states], dtype=float)
        items, internal = [{"markers": list(range(len(crit)))}], {qid: {"t": "choice", "crit": crit}}
        return [{"answers": self._decode_answers(Z, None, items, [qid], internal, i)} for i in range(len(states))]

    def _decode_answers(self, logits, act, items, ids, internal, offset, lang=None):
        qid, k = ids[0], len(items[0]["markers"])
        keys = list(internal[qid]["crit"])
        p = softmax(logits[offset:offset + 1, :k] / self.temperature[0])[0]
        return {qid: {"choice": keys[int(p.argmax())], "probabilities": dict(zip(keys, p.tolist())),
                      "answer_confidence": float(p.max())}}


@pytest.fixture()
def exact_run(tmp_path, monkeypatch):
    pytest.importorskip("laya")
    from laya_poc import hub
    data, trap = tmp_path / "data", tmp_path / "data_eval" / "trap_candidates.jsonl"
    splits = {"val": _rows("val", n=60, seed=11), "test_id": _rows("test_id", seed=12),
              "stripped_test": _rows("stripped_test", n=20, seed=13, stripped=True),
              "trap_candidates": _rows("trap_candidates", n=15, seed=14)}
    splits["stripped_test"][1][::2, 0] += 8.0
    logits = {}
    for name, (rows, Z) in splits.items():
        write_jsonl(trap if name == "trap_candidates" else data / f"{name}.jsonl", rows)
        logits.update({r["state"]: z.tolist() for r, z in zip(rows, Z) if r["state"] not in logits})
    monkeypatch.setattr(hub, "load_agent", lambda source, device="cpu": ExactAgent(logits, T=2.0))
    rc = E.main(["--ckpt", "hub", "--model", "laya", "--splits", "val", "test_id", "stripped_test", "--data-dir",
                 str(data), "--extra", f"trap_candidates={trap}", "--out-dir", str(tmp_path / "eval"),
                 "--preds-dir", str(tmp_path / "preds"), "--device", "cpu", "--batch-size", "4"])
    assert rc == 0
    paths = {name: trap if name == "trap_candidates" else data / f"{name}.jsonl" for name in splits}
    # the scores a baseline hands over: the logits of EVERY row (gated rows included), row order
    scores = {name: np.array([logits[r["state"]] for r in rows]) for name, (rows, _) in splits.items()}
    return SimpleNamespace(tmp=tmp_path, splits=splits, paths=paths, scores=scores)


def test_identical_to_evaluate_splits_when_fed_the_same_probabilities(exact_run, cfg):
    run, mc, tau, mine = exact_run, cfg["model"]["laya"], None, {}
    for name, (rows, _) in run.splits.items():                          # val first: its tau for the others
        mine[name] = BE.write_outputs(
            run.tmp / "b_eval", run.tmp / "b_preds", rows=rows, keys=KEYS, scores=run.scores[name], T=2.0,
            split=name, model="laya", ckpt="hub", evidence_fields=cfg["serialise"]["evidence_fields"],
            abstain_cfg=cfg["abstain"], tau=tau, meta={"rows": str(run.paths[name]), "batch_size": 4,
                                                       "max_len": mc["max_len"], "head_max_len": mc["head_max_len"]})
        tau = mine[name]["tau"] if name == "val" else tau
        ref = json.loads((run.tmp / "eval" / f"{name}.json").read_text(encoding="utf-8"))
        assert list(mine[name]) == list(ref)                            # same keys, same order
        assert _strip(mine[name]) == _strip(ref), name                  # identical numbers
    for name in [*run.splits, "no_gate/stripped_test"]:
        assert read_jsonl(run.tmp / "b_preds" / f"{name}.jsonl") == read_jsonl(run.tmp / "preds" / f"{name}.jsonl")
    assert mine["val"]["tau"]["rule"] is not None and mine["stripped_test"]["false_confident_rate"] > 0


# ---------------------------------------------------------------- key structure on the tiny Laya checkpoint

DATA_DEPENDENT = {"at_target_acc", "at_fallback_coverage"}   # None when no threshold qualifies: depends on values


def _structure_diff(a, b, path=""):
    """Paths where the key structure of two payloads differs (dict keys, recursively; leaves are equal)."""
    if isinstance(a, dict) and isinstance(b, dict):
        diff = [f"{path}/{k}: only in one" for k in sorted(set(a) ^ set(b))]
        return diff + [d for k in sorted(set(a) & set(b)) for d in _structure_diff(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, dict) != isinstance(b, dict):
        lenient = path.rsplit("/", 1)[-1] in DATA_DEPENDENT and (a is None or b is None)
        return [] if lenient else [f"{path}: dict vs {type(b if isinstance(a, dict) else a).__name__}"]
    return []


@pytest.mark.torch
def test_same_key_structure_as_evaluate_splits_on_the_tiny_checkpoint(tiny_ckpt_dir, synthetic_data_dir, tmp_path,
                                                                      monkeypatch, cfg):
    from laya_poc import evaluate_splits as ES
    val, trap = read_jsonl(synthetic_data_dir / "val.jsonl"), tmp_path / "data_eval" / "trap_candidates.jsonl"
    write_jsonl(synthetic_data_dir / "stripped_test.jsonl", [{**r, "id": f"stripped_test-{i:06d}", "state": _state(
        country="GB")} for i, r in enumerate(val[:6])])
    write_jsonl(trap, [{**r, "id": f"trap_candidates-{i:06d}"} for i, r in enumerate(val[:5])])
    seen, real = {}, E.predict_pass

    def spy(agent, states, question, keys, **kw):
        out = real(agent, states, question, keys, **kw)
        seen[kw["label"]] = (list(states), out[2])
        return out

    monkeypatch.setattr(E, "predict_pass", spy)
    monkeypatch.setattr(ES, "predict_pass", spy)
    out, preds = tmp_path / "eval", tmp_path / "preds"
    assert E.main(["--ckpt", str(tiny_ckpt_dir), "--splits", "val", "test_id", "stripped_test", "--data-dir",
                   str(synthetic_data_dir), "--extra", f"trap_candidates={trap}", "--out-dir", str(out),
                   "--preds-dir", str(preds), "--device", "cpu", "--batch-size", "8", "--n", "12"]) == 0
    tau = None
    for name in ("val", "test_id", "stripped_test", "trap_candidates"):
        ref = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        rows = E.load_rows(trap if name == "trap_candidates" else synthetic_data_dir / f"{name}.jsonl", 12)
        Z = np.zeros((len(rows), len(KEYS)))
        if name == "stripped_test":
            Z = np.asarray(seen["evaluate stripped_test no-gate"][1])
        else:
            states, captured = seen[f"evaluate {name} post-T"]
            Z[E.evidence_mask(rows, cfg["serialise"]["evidence_fields"])] = captured
            assert states == [r["state"] for r in rows]
        mine = BE.write_outputs(tmp_path / "b_eval", tmp_path / "b_preds", rows=rows, keys=KEYS, scores=Z,
                                T=ref["temperature"], split=name, model="laya", ckpt=str(tiny_ckpt_dir),
                                evidence_fields=cfg["serialise"]["evidence_fields"], abstain_cfg=cfg["abstain"],
                                tau=tau)
        tau = mine["tau"] if name == "val" else tau
        assert _structure_diff(mine, ref) == [], name
        for k in ("n", "n_unlabelled", "n_abstained_no_evidence", "n_scored", "temperature", "temperature_pre"):
            assert mine[k] == ref[k], (name, k)
        for side in ("post", "pre"):
            assert (mine[side] is None) == (ref[side] is None)
            if ref[side] is not None:                                   # both NLLs: log_softmax(raw logits / T)
                assert mine[side]["nll"] == pytest.approx(ref[side]["nll"], rel=1e-9)
    assert (tmp_path / "b_preds" / "no_gate" / "stripped_test.jsonl").is_file()
