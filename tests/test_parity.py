import json
import os
import re

import numpy as np
import pytest

from laya_poc import labels as L
from laya_poc import parity as P


# ---- pure comparison --------------------------------------------------------------------------

def test_compare_identical_matrices():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    res = P.compare(p, p.copy())
    assert res == {"max_dp": 0.0, "argmax_agree": 1.0, "nan_ref": 0, "nan_test": 0}


def test_compare_reports_max_dp_and_disagreement():
    ref = np.array([[0.6, 0.4], [0.45, 0.55], [0.9, 0.1]])
    test = np.array([[0.6, 0.4], [0.55, 0.45], [0.89, 0.11]])
    res = P.compare(ref, test)
    assert res["max_dp"] == pytest.approx(0.1)
    assert res["argmax_agree"] == pytest.approx(2 / 3)


def test_compare_counts_nan_rows_as_disagreement():
    ref = np.array([[0.6, 0.4], [0.3, 0.7]])
    test = np.array([[0.6, 0.4], [np.nan, np.nan]])
    res = P.compare(ref, test)
    assert res["nan_test"] == 1 and res["nan_ref"] == 0
    assert res["argmax_agree"] == 0.5
    assert res["max_dp"] == 0.0  # over the finite rows only; NaN is flagged separately
    json.dumps(res)


def test_compare_all_nan_gives_none_max_dp():
    res = P.compare(np.full((2, 2), np.nan), np.full((2, 2), np.nan))
    assert res["max_dp"] is None and res["argmax_agree"] == 0.0


def test_compare_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        P.compare(np.zeros((2, 3)), np.zeros((3, 3)))


def test_renormalise_rows_handles_rounding():
    p = P.renormalise(np.array([[0.3333, 0.3333, 0.3333], [0.0, 0.0, 0.0]]))
    assert p[0].sum() == pytest.approx(1.0)
    assert np.isnan(p[1]).all()  # a zero row cannot be a distribution: surface it as NaN


def test_mixed_length_indices_span_lengths():
    states = ["x" * n for n in (5, 50, 10, 40, 20, 30, 15, 25)]
    idx = P.mixed_length_indices(states, 4)
    lengths = sorted(len(states[i]) for i in idx)
    assert len(idx) == 4 and lengths[0] == 5 and lengths[-1] == 50
    assert P.mixed_length_indices(states, 20) == list(range(8))


def test_evaluate_parity_thresholds():
    thr = {"parity_max_dp": 0.02, "parity_min_agree": 0.995}
    ok = {"max_dp": 0.01, "argmax_agree": 1.0, "nan_ref": 0, "nan_test": 0}
    assert P.passed(ok, padded_max_dp=0.001, padded_nan=False, thresholds=thr)
    assert not P.passed({**ok, "max_dp": 0.03}, padded_max_dp=0.001, padded_nan=False, thresholds=thr)
    assert not P.passed({**ok, "argmax_agree": 0.99}, padded_max_dp=0.001, padded_nan=False, thresholds=thr)
    assert not P.passed({**ok, "nan_test": 1}, padded_max_dp=0.001, padded_nan=False, thresholds=thr)
    assert not P.passed(ok, padded_max_dp=0.05, padded_nan=False, thresholds=thr)
    assert not P.passed(ok, padded_max_dp=None, padded_nan=True, thresholds=thr)
    assert not P.passed({**ok, "max_dp": None}, padded_max_dp=0.0, padded_nan=False, thresholds=thr)


# ---- reference speed-ups and the progress line (fake agent) ------------------------------------

LAYA_OOM = "Warning: GPU memory exceeded during inference. Retrying this request on CPU..."
PROGRESS_RE = re.compile(r"^parity laya: reference (\d+)/(\d+) rows, \d+s$")


class _FakeAgent:
    """predict_batch batch by batch through self._forward, as Laya does; uniform answers."""

    def __init__(self, keys):
        self.keys, self.calls = list(keys), []

    def _forward(self, b):
        return None

    def predict_batch(self, states, question, *, batch_size, max_len, head_max_len, sort_by_length=False):
        self.calls.append({"n": len(states), "batch_size": batch_size, "sort_by_length": sort_by_length})
        qid = next(iter(question))
        for s in range(0, len(states), batch_size):
            self._forward({"input_ids": np.zeros((len(states[s:s + batch_size]), 3))})
        probs = {k: round(1 / len(self.keys), 4) for k in self.keys}
        return [{"answers": {qid: {"probabilities": probs}}} for _ in states]


def test_probs_matrix_sorts_by_length_only_when_asked():
    keys = L.option_keys("c10")
    agent = _FakeAgent(keys)
    P.probs_matrix(agent, ["{}"] * 3, L.question("c10"), keys, 2, 512, 192)
    P.probs_matrix(agent, ["{}"] * 3, L.question("c10"), keys, 2, 512, 192, sort_by_length=True)
    assert [c["sort_by_length"] for c in agent.calls] == [False, True]


def test_batch_progress_prints_about_every_quarter_and_unhooks(capsys):
    keys = L.option_keys("c10")
    agent = _FakeAgent(keys)
    with P.batch_progress(agent, 200, "parity laya: reference"):
        agent.predict_batch(["{}"] * 200, L.question("c10"), batch_size=16, max_len=512, head_max_len=192)
    lines = capsys.readouterr().out.strip().splitlines()
    assert all(PROGRESS_RE.match(line) for line in lines), lines
    assert [int(PROGRESS_RE.match(line).group(1)) for line in lines] == [64, 112, 160, 200]
    assert "_forward" not in vars(agent)  # the class method is back once the block ends


def test_all_cpu_threads_is_scoped():
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    with P.all_cpu_threads() as n:
        assert n == (os.cpu_count() or 1) and torch.get_num_threads() == n
    assert torch.get_num_threads() == before


# ---- with the tiny Laya checkpoint --------------------------------------------------------------

@pytest.fixture()
def tiny_agent(tiny_ckpt_dir):
    pytest.importorskip("laya")
    from laya_poc import hub
    return hub.load_agent(tiny_ckpt_dir, device="cpu")


def _states(synthetic_data_dir, n):
    from laya_poc.io_utils import read_jsonl
    return [r["state"] for r in read_jsonl(synthetic_data_dir / "val.jsonl")[:n]]


@pytest.mark.torch
def test_probs_matrix_rows_are_distributions(tiny_agent, synthetic_data_dir):
    keys = L.option_keys("c10")
    p = P.probs_matrix(tiny_agent, _states(synthetic_data_dir, 6), L.question("c10"), keys,
                       batch_size=4, max_len=512, head_max_len=192)
    assert p.shape == (6, 10)
    assert np.allclose(p.sum(1), 1.0)


@pytest.mark.torch
def test_probs_matrix_rejects_dict_states(tiny_agent):
    with pytest.raises(TypeError, match="string"):
        P.probs_matrix(tiny_agent, [{"name": "x"}], L.question("c10"), L.option_keys("c10"),
                       batch_size=1, max_len=512, head_max_len=192)


@pytest.mark.torch
def test_cli_two_cpu_agents_pass(tiny_ckpt_dir, synthetic_data_dir, tmp_path, capsys):
    out = tmp_path / "parity_laya.json"
    rc = P.main(["--model", "laya", "--init", str(tiny_ckpt_dir), "--rows", str(synthetic_data_dir / "val.jsonl"),
                 "--n", "20", "--out", str(out), "--allow-cpu"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    for key in ("model", "n", "max_dp", "argmax_agree", "padded_max_dp", "nan", "passed", "thresholds",
                "device_ref", "device_test", "dtype_test", "seconds", "seconds_ref", "seconds_test", "cpu_fallback"):
        assert key in res, key
    assert res["model"] == "laya" and res["n"] == 20
    assert res["max_dp"] <= 1e-6 and res["argmax_agree"] == 1.0
    assert res["padded_max_dp"] <= 1e-3 and res["padded_n"] == 16
    assert res["nan"] is False and res["passed"] is True and res["cpu_fallback"] is False
    assert res["device_ref"] == "cpu" and res["device_test"] == "cpu"
    assert res["seconds_ref"] >= 0 and res["seconds_test"] >= 0
    out = capsys.readouterr().out
    assert "PASS" in out and any(PROGRESS_RE.match(line) for line in out.splitlines())


def _spy_predict(loader, role, calls):
    """Wrap a loader so its agent's predict_batch records (role, sort_by_length, torch threads)."""
    import torch

    def load(*args, **kwargs):
        agent = loader(*args, **kwargs)
        predict = agent.predict_batch

        def spy(*a, **kw):
            calls.append((role, kw.get("sort_by_length", False), torch.get_num_threads()))
            return predict(*a, **kw)

        agent.predict_batch = spy
        return agent
    return load


@pytest.mark.torch
def test_cli_sorts_and_threads_only_the_reference(tiny_ckpt_dir, synthetic_data_dir, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    calls, threads_before = [], torch.get_num_threads()
    monkeypatch.setattr(P, "load_reference", _spy_predict(P.load_reference, "ref", calls))
    monkeypatch.setattr(P, "load_test", _spy_predict(P.load_test, "test", calls))
    rc = P.main(["--model", "laya", "--init", str(tiny_ckpt_dir), "--rows", str(synthetic_data_dir / "val.jsonl"),
                 "--n", "20", "--out", str(tmp_path / "p.json"), "--allow-cpu"])
    assert rc == 0
    ref = [c for c in calls if c[0] == "ref"]
    test = [c for c in calls if c[0] == "test"]
    assert ref == [("ref", True, os.cpu_count() or 1)]
    assert len(test) == 3 and all(not s and t == threads_before for _, s, t in test)  # padding check still bites
    assert torch.get_num_threads() == threads_before


@pytest.mark.torch
def test_cli_fails_when_the_test_agent_fell_back_to_cpu(tiny_ckpt_dir, synthetic_data_dir, tmp_path, monkeypatch,
                                                        capsys):
    real_load_test = P.load_test

    def noisy_load_test(*args, **kwargs):
        agent = real_load_test(*args, **kwargs)
        forward = agent._forward

        def oom_then_forward(b):
            print(LAYA_OOM)  # what Laya prints before answering the batch on CPU
            return forward(b)

        agent._forward = oom_then_forward
        return agent

    monkeypatch.setattr(P, "load_test", noisy_load_test)
    out = tmp_path / "parity.json"
    rc = P.main(["--model", "laya", "--init", str(tiny_ckpt_dir), "--rows", str(synthetic_data_dir / "val.jsonl"),
                 "--n", "8", "--out", str(out), "--allow-cpu"])
    assert rc == 0  # the smoke report decides
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["cpu_fallback"] is True and res["passed"] is False
    assert res["max_dp"] <= 1e-6  # the numbers alone look perfect: only the guard catches it
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.torch
def test_cli_refuses_cpu_test_agent_without_allow_cpu(tiny_ckpt_dir, synthetic_data_dir, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("needs a machine without CUDA")
    out = tmp_path / "parity.json"
    rc = P.main(["--model", "laya", "--init", str(tiny_ckpt_dir), "--rows", str(synthetic_data_dir / "val.jsonl"),
                 "--n", "4", "--out", str(out)])
    assert rc == 1
    assert "cuda" in capsys.readouterr().err.lower()


def test_cli_rejects_relative_init(synthetic_data_dir, tmp_path, capsys):
    rc = P.main(["--model", "laya", "--init", "relative/ckpt", "--rows", str(synthetic_data_dir / "val.jsonl"),
                 "--out", str(tmp_path / "p.json"), "--allow-cpu"])
    assert rc == 1
    assert "absolute" in capsys.readouterr().err
