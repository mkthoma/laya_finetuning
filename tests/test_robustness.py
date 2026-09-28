import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from laya_poc import labels as L
from laya_poc import robustness as R

KEYS = L.option_keys("c10")
Q = L.question("c10")
QID = L.QUESTION_NAME


def _state(**fields):
    return json.dumps(dict(sorted(fields.items())), ensure_ascii=False, separators=(",", ":"))


def _canon(state):
    return json.dumps(dict(sorted(json.loads(state).items())), ensure_ascii=False, separators=(",", ":"))


class KeyAgent:
    """predict_batch stand-in whose logits depend on the record content only (order-invariant), or on the
    FIRST field of the string when `order_sensitive` (so a permutation can flip the answer)."""

    def __init__(self, order_sensitive=False):
        self.order_sensitive, self.calls = order_sensitive, []
        self.device = SimpleNamespace(type="cpu")

    def _label(self, state):
        rec = json.loads(state)
        if self.order_sensitive:
            return KEYS[len(next(iter(rec))) % len(KEYS)]
        return KEYS[sum(map(ord, rec.get("name", ""))) % len(KEYS)]

    def predict_batch(self, states, questions, batch_size=None, max_len=None, head_max_len=None,
                      sort_by_length=False):
        self.calls.append(list(states))
        qid = next(iter(questions))
        out = []
        for s in states:
            assert isinstance(s, str), "states must be strings"
            lab = self._label(s)
            probs = {k: (0.91 if k == lab else 0.01) for k in reversed(KEYS)}
            out.append({"answers": {qid: {"choice": lab, "probabilities": probs, "answer_confidence": 0.91}}})
        return out


def _rows(n):
    rng = random.Random(3)
    rows = []
    for i in range(n):
        fields = {"country": "GB", "name": f"Café {i}", "locality": rng.choice(["Leeds", "Åre", "Kyoto"]),
                  "tel": f"0{i:04d}", "website": f"ex{i}.com"}
        rows.append({"id": f"test_id-{i:06d}", "state": _state(**fields), "label": KEYS[i % len(KEYS)]})
    return rows


# ---------------------------------------------------------------- tau (design §5.8.2)

def _desc_conf(n):
    return np.array([1 - i / n for i in range(n)])


def test_choose_tau_takes_the_smallest_threshold_reaching_the_target_accuracy():
    conf = _desc_conf(100)
    correct = np.array([i < 80 and i not in (10, 50) for i in range(100)], dtype=float)
    res = R.choose_tau(conf, correct, 0.95, 0.70)
    # prefix accuracy dips below 0.95 at k=11 (10/11) but the LOWEST threshold that reaches it is k=82 (78/82)
    assert res["rule"] == "accuracy" and res["tau"] == pytest.approx(conf[81])
    acc = res["at_target_acc"]
    assert acc["tau"] == pytest.approx(conf[81]) and acc["coverage"] == pytest.approx(0.82)
    assert acc["answered_acc"] == pytest.approx(78 / 82) and acc["n_answered"] == 82
    assert res["at_fallback_coverage"]["coverage"] == pytest.approx(0.80)   # both are reported
    assert res["n"] == 100 and res["target_acc"] == 0.95 and res["min_coverage"] == 0.70


def test_choose_tau_falls_back_to_the_coverage_threshold_when_coverage_is_too_low():
    conf = _desc_conf(100)
    correct = np.array([i < 50 or i % 2 == 0 for i in range(100)], dtype=float)
    res = R.choose_tau(conf, correct, 0.95, 0.70)
    assert res["at_target_acc"]["coverage"] == pytest.approx(0.55)          # 53/55 >= 0.95; 53/56 is not
    assert res["rule"] == "coverage" and res["tau"] == pytest.approx(conf[79])
    fb = res["at_fallback_coverage"]
    assert fb["coverage"] == pytest.approx(0.80) and fb["answered_acc"] == pytest.approx(65 / 80)
    assert res["fallback_coverage"] == 0.80


def test_choose_tau_with_no_threshold_reaching_the_target():
    conf = _desc_conf(10)
    res = R.choose_tau(conf, np.zeros(10), 0.95, 0.70)
    assert res["at_target_acc"] is None and res["rule"] == "coverage"
    assert res["tau"] == pytest.approx(conf[7])                            # 80% coverage: 8 of 10


def test_choose_tau_only_cuts_between_distinct_confidences():
    conf, correct = np.array([0.9, 0.9, 0.5, 0.5]), np.array([1.0, 0.0, 1.0, 1.0])
    res = R.choose_tau(conf, correct, 0.95, 0.70)
    assert res["at_target_acc"] is None          # tau=0.9 answers BOTH 0.9 rows (acc 0.5), never just the first
    assert R.threshold_for_coverage(np.array([0.9, 0.9, 0.9, 0.1]), 0.5) == 0.9
    assert R.abstention(np.array([0.9, 0.9, 0.9, 0.1]), None, 0.9)["coverage"] == 0.75


def test_choose_tau_on_nothing_scored():
    res = R.choose_tau(np.zeros(0), np.zeros(0), 0.95, 0.70)
    assert res["tau"] is None and res["rule"] is None and res["n"] == 0


def test_choose_tau_rejects_bad_inputs():
    with pytest.raises(ValueError, match="length"):
        R.choose_tau(np.ones(3), np.ones(2), 0.95, 0.7)
    with pytest.raises(ValueError, match="finite"):
        R.choose_tau(np.array([0.5, np.nan]), np.ones(2), 0.95, 0.7)
    with pytest.raises(ValueError, match="target_acc"):
        R.choose_tau(np.ones(2), np.ones(2), 1.5, 0.7)


# ---------------------------------------------------------------- abstention and false-confident rate

def test_abstention_answers_at_or_above_tau():
    conf, correct = np.array([0.9, 0.8, 0.3, 0.5]), np.array([1, 1, 0, 0], dtype=float)
    res = R.abstention(conf, correct, 0.5)
    assert res == {"tau": 0.5, "n": 4, "n_answered": 3, "coverage": 0.75, "abstain_rate": 0.25,
                   "answered_acc": pytest.approx(2 / 3)}
    assert R.abstention(conf, None, 0.5)["answered_acc"] is None                 # coverage needs no labels
    assert R.abstention(conf, correct, 0.95)["answered_acc"] is None             # nothing answered
    none = R.abstention(conf, correct, None)
    assert none["coverage"] is None and none["abstain_rate"] is None and none["n"] == 4
    empty = R.abstention(np.zeros(0), np.zeros(0), 0.5)
    assert empty["n"] == 0 and empty["coverage"] is None


def test_false_confident_rate_is_strictly_above_the_threshold():
    assert R.false_confident_rate(np.array([0.81, 0.8, 0.95, 0.2])) == 0.5
    assert R.false_confident_rate(np.array([0.5]), 0.4) == 1.0
    assert R.false_confident_rate(np.zeros(0)) is None


def test_conf_correct_uses_scored_prediction_records_only():
    preds = [{"y": 1, "p": [0.2, 0.8], "answer_confidence": 0.8, "abstained": False},
             {"y": 0, "p": [0.3, 0.7], "answer_confidence": 0.7, "abstained": False},
             {"y": None, "p": [0.9, 0.1], "answer_confidence": 0.9, "abstained": False},
             {"y": 0, "p": None, "answer_confidence": None, "abstained": True}]
    conf, correct = R.conf_correct(preds)
    assert conf.tolist() == [0.8, 0.7] and correct.tolist() == [1.0, 0.0]
    conf, correct = R.conf_correct([])
    assert conf.shape == correct.shape == (0,)


# ---------------------------------------------------------------- field orders

def test_canonical_and_shuffled_states_stay_compact_strings():
    s = _state(name="Café Ω", country="GB", locality="Åre", tel="01")
    shuffled = '{"tel":"01","name":"Café Ω","locality":"Åre","country":"GB"}'
    assert R.canonical_state(shuffled) == s
    perms = R.permuted_states([s, shuffled], perms=3, seed=7)
    assert len(perms) == 3 and all(len(p) == 2 for p in perms)
    for perm in perms:
        for orig, new in zip([s, shuffled], perm):
            assert isinstance(new, str) and json.loads(new) == json.loads(orig)
            assert ", " not in new and '": ' not in new and "\\u" not in new           # compact, not ASCII-escaped
            assert new == json.dumps(json.loads(new), ensure_ascii=False, separators=(",", ":"))
    assert R.permuted_states([s], perms=3, seed=7) == [p[:1] for p in perms]           # one rng per perm, seed + p
    assert R.permuted_states([s, shuffled], perms=3, seed=7) == perms                  # deterministic
    assert len({p[0] for p in R.permuted_states([s], perms=8, seed=1)}) > 1           # orders really change
    with pytest.raises(ValueError, match="object"):
        R.canonical_state('["a"]')


def test_agreement_and_the_99_percent_bar():
    fixed = np.arange(100) % 7
    flip = fixed.copy()
    flip[0] += 1
    two = fixed.copy()
    two[:2] += 1
    res = R.agreement(fixed, [fixed, flip])
    assert res["agreement_per_perm"] == [1.0, 0.99] and res["mean_agreement"] == pytest.approx(0.995)
    assert res["passed_99"] is True
    assert R.agreement(fixed, [two])["passed_99"] is False
    with pytest.raises(ValueError, match="perm"):
        R.agreement(fixed, [])


# ---------------------------------------------------------------- order invariance on an agent

def test_order_invariance_on_an_order_invariant_agent():
    pytest.importorskip("laya")
    rows = _rows(12)
    agent = KeyAgent()
    res = R.order_invariance(agent, rows, Q, n=10, perms=3, seed=5, batch_size=4, max_len=512, head_max_len=192)
    assert res["n"] == 10 and res["perms"] == 3 and res["seed"] == 5
    assert res["agreement_per_perm"] == [1.0, 1.0, 1.0] and res["mean_agreement"] == 1.0 and res["passed_99"]
    assert len(agent.calls) == 4                                          # fixed + 3 permutations
    assert agent.calls[0] == [R.canonical_state(r["state"]) for r in rows[:10]]
    assert all(s == R.canonical_state(s) for s in agent.calls[0])        # fixed = alphabetical
    assert all(agent.calls[p + 1] == R.permuted_states(agent.calls[0], 3, 5)[p] for p in range(3))
    assert 0 <= min(res["unchanged_share_per_perm"]) and max(res["unchanged_share_per_perm"]) < 1
    assert res["cpu_fallback"] is False


def test_order_invariance_detects_an_order_sensitive_agent():
    pytest.importorskip("laya")
    res = R.order_invariance(KeyAgent(order_sensitive=True), _rows(40), Q, n=1000, perms=2, seed=1,
                             batch_size=8, max_len=512, head_max_len=192)
    assert res["n"] == 40 and res["mean_agreement"] < 0.99 and res["passed_99"] is False
    assert all(0 <= a <= 1 for a in res["agreement_per_perm"])


def test_order_invariance_rejects_bad_arguments():
    with pytest.raises(ValueError, match="perms"):
        R.order_invariance(KeyAgent(), _rows(2), Q, n=2, perms=0, seed=1, batch_size=2, max_len=512,
                           head_max_len=192)
    with pytest.raises(ValueError, match="no rows"):
        R.order_invariance(KeyAgent(), [], Q, n=2, perms=1, seed=1, batch_size=2, max_len=512, head_max_len=192)


@pytest.mark.torch
def test_order_invariance_on_the_tiny_checkpoint(tiny_ckpt_dir, synthetic_data_dir):
    from laya_poc.hub import load_agent
    from laya_poc.io_utils import read_jsonl
    agent = load_agent(tiny_ckpt_dir, device="cpu")
    rows = read_jsonl(synthetic_data_dir / "test_id.jsonl")
    res = R.order_invariance(agent, rows, Q, n=8, perms=2, seed=3, batch_size=4, max_len=512, head_max_len=192)
    assert res["n"] == 8 and len(res["agreement_per_perm"]) == 2
    assert 0 <= res["mean_agreement"] <= 1 and isinstance(res["passed_99"], bool)
    assert np.isfinite(res["seconds"])
