import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from laya_poc import export_check as X
from laya_poc import labels as L
from laya_poc.calibrate import fit_temperature, log_softmax, softmax

pytestmark = pytest.mark.torch
pytest.importorskip("laya")

# Offset-logit checkpoint: spread-out logits sitting ~40 below zero, so raw values fall under
# calibrate's log(1e-12) floor and 4-dp rounding of the probabilities is visible.
SHARPEN, OFFSET = 3e4, -40.0


def _spec_write_choice_temperature(ckpt_dir, T):
    """Spec §3 semantics of export.write_choice_temperature, used when TRAIN's module is absent."""
    p = Path(ckpt_dir) / "rl_agent_config.json"
    c = json.loads(p.read_text(encoding="utf-8"))
    t = list(c.get("temperature", [1.0, 1.0, 1.0]))
    t[0] = T
    c["temperature"] = t
    c.pop("temperature_by_options", None)
    p.write_text(json.dumps(c, indent=2), encoding="utf-8")
    return c


@pytest.fixture()
def writer(monkeypatch):
    """Use laya_poc.export.write_choice_temperature when it exists, else a spec-faithful stub."""
    try:
        from laya_poc.export import write_choice_temperature  # noqa: F401
    except ImportError:
        monkeypatch.setattr(X, "write_temperature", _spec_write_choice_temperature)


@pytest.fixture()
def ckpt_copy(tiny_ckpt_dir, tmp_path):
    # export_check rewrites rl_agent_config.json: never touch the session-wide fixture.
    return Path(shutil.copytree(tiny_ckpt_dir, tmp_path / "final"))


@pytest.fixture()
def tiny_agent(tiny_ckpt_dir):
    from laya_poc import hub
    return hub.load_agent(tiny_ckpt_dir, device="cpu")


@pytest.fixture()
def offset_ckpt(ckpt_copy):
    import torch
    from safetensors.torch import load_file, save_file
    path = ckpt_copy / "model.safetensors"
    sd = load_file(str(path))
    sd["scorer.3.weight"] = (sd["scorer.3.weight"].float() * SHARPEN).half()
    sd["scorer.3.bias"] = torch.full_like(sd["scorer.3.bias"], OFFSET)
    save_file(sd, str(path))
    return ckpt_copy


def _val_rows(data_dir, n=None):
    from laya_poc.io_utils import read_jsonl
    rows = read_jsonl(data_dir / "val.jsonl")
    return rows if n is None else rows[:n]


def _labelled(data_dir, n=None):
    rows = [r for r in _val_rows(data_dir) if r.get("label") is not None]
    return rows if n is None else rows[:n]


def _run(ckpt, data_dir, out, *extra):
    rc = X.main(["--ckpt", str(ckpt), "--rows", str(data_dir / "val.jsonl"), "--out", str(out), "--device", "cpu",
                 *extra])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_raw_logits_match_predict_batch_at_unit_temperature(tiny_agent, synthetic_data_dir):
    states = [r["state"] for r in _val_rows(synthetic_data_dir, 7)]
    q = L.question("c10")
    z = X.raw_logits(tiny_agent, states, q, batch_size=3)
    assert z.shape == (7, 10) and z.dtype == np.float64 and np.isfinite(z).all()
    tiny_agent.temperature, tiny_agent.temperature_by_options, tiny_agent.lang_temperatures = [1.0, 1.0, 1.0], {}, {}
    out = tiny_agent.predict_batch(states, q, batch_size=7)
    p = np.array([[o["answers"][L.QUESTION_NAME]["probabilities"][k] for k in L.option_keys("c10")] for o in out])
    from laya_poc.calibrate import softmax
    assert np.abs(softmax(z) - p).max() <= 1e-4  # 4-dp rounding only


def test_raw_logits_reject_dict_states(tiny_agent):
    with pytest.raises(TypeError, match="string"):
        X.raw_logits(tiny_agent, [{"name": "x"}], L.question("c10"))


def test_label_indices():
    keys = L.option_keys("c10")
    rows = [{"id": "a", "label": "dining"}, {"id": "b", "label": "arts"}]
    assert X.label_indices(rows, keys).tolist() == [keys.index("dining"), keys.index("arts")]
    with pytest.raises(ValueError, match="culture"):
        X.label_indices([{"id": "c", "label": "culture"}], keys)


def test_cli_round_trip_passes_and_rewrites_config(ckpt_copy, synthetic_data_dir, tmp_path, writer, capsys):
    out = tmp_path / "export_check.json"
    rc = X.main(["--ckpt", str(ckpt_copy), "--rows", str(synthetic_data_dir / "val.jsonl"), "--out", str(out),
                 "--device", "cpu", "--n", "24"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    for key in ("T", "T_applied", "clamped", "n", "pre", "post", "roundtrip_max_dp", "passed"):
        assert key in res, key
    assert res["n"] == 24 and res["roundtrip_n"] == 20
    assert res["passed"] is True, res
    assert res["roundtrip_max_dp"] <= 1e-3
    assert "per_class" not in res["pre"] and "cm" not in res["post"]
    assert res["post"]["nll"] <= res["pre"]["nll"] + 1e-6
    cfg = json.loads((ckpt_copy / "rl_agent_config.json").read_text(encoding="utf-8"))
    assert cfg["temperature"][0] == res["T"] and "temperature_by_options" not in cfg
    assert res["clamped"] == (not 0.5 <= res["T"] <= 5.0)
    assert res["calibration_ok"] is (not res["clamped"])
    # no summary.json beside the checkpoint and a non-PoC model_name: cross-checks are skipped, not failed
    assert res["train_eval"]["ok"] is None and "summary.json" in res["train_eval"]["note"]
    assert res["budgets"]["ok"] is None and res["budgets"]["saved"] == {"max_len": 512, "head_max_len": 192}
    assert "weights_max_dlogit" not in res
    assert res["cpu_fallback"] is False
    assert "PASS" in capsys.readouterr().out


LAYA_OOM = "Warning: GPU memory exceeded during inference. Retrying this request on CPU..."


def test_cli_fails_when_an_inference_call_fell_back_to_cpu(ckpt_copy, synthetic_data_dir, tmp_path, writer,
                                                           monkeypatch, capsys):
    from laya_poc import hub
    real_load = hub.load_agent
    loads = []

    def noisy_on_reload(*args, **kwargs):
        agent = real_load(*args, **kwargs)
        loads.append(agent)
        if len(loads) == 2:  # only the reload after the write-back: the guard must span both loads
            forward = agent._forward

            def oom_then_forward(b):
                print(LAYA_OOM)  # what Laya prints before answering the batch on CPU
                return forward(b)

            agent._forward = oom_then_forward
        return agent

    monkeypatch.setattr(hub, "load_agent", noisy_on_reload)
    res = _run(ckpt_copy, synthetic_data_dir, tmp_path / "export_check.json", "--n", "12")
    assert len(loads) == 2
    assert res["cpu_fallback"] is True and res["passed"] is False
    assert res["config_ok"] is True and res["roundtrip_max_dp"] <= 1e-3  # everything else still passes
    assert "CPU fallback" in capsys.readouterr().out


def test_temperature_is_fitted_on_log_softmax_of_raw_logits(offset_ckpt, synthetic_data_dir, tmp_path, writer,
                                                            monkeypatch):
    seen = []

    def spy(logp, y):
        seen.append((np.array(logp, dtype=float), np.array(y)))
        return fit_temperature(logp, y)

    monkeypatch.setattr(X, "fit_temperature", spy)
    res = _run(offset_ckpt, synthetic_data_dir, tmp_path / "export_check.json", "--n", "24")
    from laya_poc import hub
    rows = _labelled(synthetic_data_dir, 24)
    z = X.raw_logits(hub.load_agent(offset_ckpt, device="cpu"), [r["state"] for r in rows], L.question("c10"))
    y = X.label_indices(rows, L.option_keys("c10"))
    assert z.max() < np.log(1e-12) and np.ptp(z, axis=1).mean() > 1.0  # the fixture does what it says
    (got, got_y), = seen
    assert np.array_equal(got_y, y)
    assert np.allclose(got, log_softmax(z), rtol=0, atol=1e-6)
    rounded = np.log(np.maximum(np.round(softmax(z), 4), 1e-12))  # the doc's predict_batch recipe
    assert np.abs(got - rounded).max() > 1e-4
    assert res["T"] == pytest.approx(fit_temperature(log_softmax(z), y), rel=1e-9)


def test_logit_fit_ignores_a_constant_offset():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(400, 10)) * 3.0
    y = np.array([rng.choice(10, p=p) for p in softmax(z / 1.5)])
    T = X.fit_logit_temperature(z, y)
    assert 1.2 < T < 1.9
    for shift in (-35.0, -25.0, 12.0):
        assert X.fit_logit_temperature(z + shift, y) == pytest.approx(T, rel=1e-6)


def test_cli_cross_checks_training_summary_and_budgets(ckpt_copy, synthetic_data_dir, tmp_path, writer,
                                                       tiny_agent):
    rows = _labelled(synthetic_data_dir)
    z = X.raw_logits(tiny_agent, [r["state"] for r in rows], L.question("c10"))
    acc = float((z.argmax(1) == X.label_indices(rows, L.option_keys("c10"))).mean())
    summary = tmp_path / "summary.json"  # train_single writes <run>/summary.json beside <run>/final
    summary.write_text(json.dumps({"model": "laya", "final_eval": {"val_acc": acc, "n": len(rows)}}), "utf-8")
    res = _run(ckpt_copy, synthetic_data_dir, tmp_path / "ok.json")
    assert res["train_eval"]["ok"] is True and res["budgets"]["ok"] is True and res["passed"] is True, res

    wrong = 1.0 if acc < 0.5 else 0.0
    summary.write_text(json.dumps({"model": "laya_ml", "final_eval": {"val_acc": wrong, "n": len(rows)}}), "utf-8")
    res = _run(ckpt_copy, synthetic_data_dir, tmp_path / "bad.json")
    assert res["train_eval"]["ok"] is False and res["budgets"]["ok"] is False and res["passed"] is False
    assert "256" in res["budgets"]["note"]  # laya_ml builds at head_max_len 256; the checkpoint says 192


def test_check_train_eval():
    pre = {"n": 200, "acc": 0.605}
    assert X.check_train_eval(pre, {"final_eval": {"val_acc": 0.61, "n": 200}}, subset=False)["ok"] is True
    bad = X.check_train_eval(pre, {"final_eval": {"val_acc": 0.70, "n": 200}}, subset=False)
    assert bad["ok"] is False and "0.7000" in bad["note"]
    assert X.check_train_eval(pre, {"final_eval": {"val_acc": 0.605, "n": 180}}, subset=False)["ok"] is False
    assert X.check_train_eval(pre, {"final_eval": {"val_acc": 0.605, "n": 180}}, subset=True)["ok"] is None
    assert X.check_train_eval(pre, None, subset=False)["ok"] is None
    assert X.check_train_eval(pre, {"final_eval": None}, subset=False)["ok"] is None
    assert X.check_train_eval(pre, {"final_eval": {"val_acc": "0.6", "n": 200}}, subset=False)["ok"] is False
    assert X.check_train_eval(pre, [1, 2], subset=False)["ok"] is False


def test_check_budgets(cfg):
    saved = {"max_len": 512, "head_max_len": 192}
    assert X.check_budgets(saved, cfg, "laya")["ok"] is True
    upstream = X.check_budgets({"max_len": 1024, "head_max_len": 256}, cfg, "laya")
    assert upstream["ok"] is False and "1024" in upstream["note"]
    assert X.check_budgets(saved, cfg, None)["ok"] is None
    assert X.check_budgets(saved, cfg, "bert")["ok"] is False


def test_resolve_model():
    assert X.resolve_model("laya_ml", {"model": "laya"}, "laya-poc-laya-s11") == "laya_ml"
    assert X.resolve_model(None, {"model": "laya_ml"}, "rl-agent") == "laya_ml"
    assert X.resolve_model(None, None, "laya-poc-laya_ml-s22") == "laya_ml"
    assert X.resolve_model(None, [1], "rl-agent") is None


def test_cli_detects_a_broken_write_back(ckpt_copy, synthetic_data_dir, tmp_path, monkeypatch):
    def wrong(ckpt_dir, T):
        return _spec_write_choice_temperature(ckpt_dir, T * 1.5 + 0.1)

    monkeypatch.setattr(X, "write_temperature", wrong)
    out = tmp_path / "export_check.json"
    rc = X.main(["--ckpt", str(ckpt_copy), "--rows", str(synthetic_data_dir / "val.jsonl"), "--out", str(out),
                 "--device", "cpu", "--n", "12"])
    assert rc == 0  # the smoke report decides
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["config_ok"] is False and res["passed"] is False


def test_cli_rejects_relative_ckpt(synthetic_data_dir, tmp_path, capsys):
    rc = X.main(["--ckpt", "runs/final", "--rows", str(synthetic_data_dir / "val.jsonl"),
                 "--out", str(tmp_path / "x.json"), "--device", "cpu"])
    assert rc == 1
    assert "absolute" in capsys.readouterr().err
