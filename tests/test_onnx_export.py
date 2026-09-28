"""onnx_export (upstream exporter replica + §7.11 acceptance) and bench_onnx (thread-pinned sessions, batched ONNX
inference through the PyTorch agent's collate path) on a tiny Laya checkpoint with DISTINCT option logits."""
import json
from pathlib import Path

import pytest

from laya_poc import onnx_export as OX
from laya_poc.io_utils import write_jsonl

UPSTREAM_DYNAMIC_AXES = {  # laya scripts/export_onnx.py at the pinned commit, verbatim
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
    "marker_pos": {0: "batch_size", 1: "num_markers"},
    "marker_mask": {0: "batch_size", 1: "num_markers"},
    "qtype": {0: "batch_size"},
    "logits": {0: "batch_size", 1: "num_markers"},
    "act_logits": {0: "batch_size"},
}


def _states(n=30):
    """Mixed lengths; every sequence (question + options + state) is longer than ModernBERT's 128-token window."""
    return [json.dumps({"address": "12 High Street " * (i % 4), "country": "GB", "locality": "Leeds " * (i % 5),
                        "name": f"Rosa Pizza {i} " + "x " * (i * 13 % 60)}, separators=(",", ":")) for i in range(n)]


# ---------------------------------------------------------------- pure

def test_export_call_replicates_upstream_names_axes_and_dummy():
    import torch

    assert OX.INPUT_NAMES == ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    assert OX.OUTPUT_NAMES == ["logits", "act_logits"] and OX.DYNAMIC_AXES == UPSTREAM_DYNAMIC_AXES
    ids, mask, pos, mmask, qtype = OX.dummy_inputs()
    assert ids.shape == (1, 16) and ids.dtype == torch.long and 0 <= int(ids.min()) and int(ids.max()) < 100
    assert mask.tolist() == [[1] * 16] and pos.tolist() == [[1, 5]] and mmask.tolist() == [[True, True]]
    assert mmask.dtype == torch.bool and qtype.tolist() == [0]
    two = OX.dummy_inputs(2)
    assert two[0].shape == (2, 16) and two[2].tolist() == [[1, 5], [1, 5]] and two[4].tolist() == [0, 0]
    assert OX.EXPORTERS["auto"] == ((None, 1), (False, 1), (None, 2))  # upstream verbatim first


def test_export_model_falls_back_in_order_and_records_every_attempt(monkeypatch, tmp_path):
    seen = []

    def fake_export(model, out_path, opset, dynamo, batch):
        seen.append((dynamo, batch))
        Path(out_path).write_bytes(b"x")
        return ["TracerWarning: something (f.py:1)"]

    def fake_probe(path):
        if seen[-1] != (None, 2):
            raise ValueError("dynamic-shape probe failed: logits (1, 4)")
        return {"shape": [3, 21, 4]}

    monkeypatch.setattr(OX, "_export_once", fake_export)
    monkeypatch.setattr(OX, "probe", fake_probe)
    rec = OX.export_model(object(), tmp_path / "m.onnx", 18, "auto")
    assert seen == [(None, 1), (False, 1), (None, 2)]
    assert rec["attempt"] == 3 and rec["dummy_batch"] == 2 and rec["upstream_verbatim"] is False
    assert [a["attempt"] for a in rec["failed_attempts"]] == [1, 2] and rec["opset"] == 18
    assert "probe failed" in rec["failed_attempts"][0]["error"] and rec["warnings"]


def test_export_model_raises_and_cleans_up_when_every_attempt_fails(monkeypatch, tmp_path):
    def fake_export(model, out_path, opset, dynamo, batch):
        Path(out_path).write_bytes(b"x")
        Path(f"{out_path}.data").write_bytes(b"y")
        raise RuntimeError("unsupported op")

    monkeypatch.setattr(OX, "_export_once", fake_export)
    with pytest.raises(RuntimeError, match="#1 .*unsupported op"):
        OX.export_model(object(), tmp_path / "m.onnx", 18, "torchscript")
    assert not (tmp_path / "m.onnx").exists() and not (tmp_path / "m.onnx.data").exists()


def test_check_states_reads_paths_rows_and_strings_but_never_dicts(tmp_path):
    rows = [{"id": str(i), "state": s} for i, s in enumerate(_states(5))]
    write_jsonl(tmp_path / "r.jsonl", rows)
    assert OX.check_states(tmp_path / "r.jsonl", 3) == ([r["state"] for r in rows[:3]], str(tmp_path / "r.jsonl"))
    assert OX.check_states(rows, 2)[0] == [rows[0]["state"], rows[1]["state"]]
    assert OX.check_states(_states(2), 5)[1] is None
    with pytest.raises(TypeError, match="strings"):
        OX.check_states([{"state": {"name": "x"}}], 1)
    with pytest.raises(ValueError, match="no rows"):
        OX.check_states([], 1)


# ---------------------------------------------------------------- real tiny checkpoint

@pytest.fixture(scope="module")
def distinct_ckpt(en_snapshot, tmp_path_factory):
    from conftest import build_tiny_checkpoint

    return build_tiny_checkpoint(en_snapshot, tmp_path_factory.mktemp("onnx_ckpt") / "ckpt", init_scale=0.5)


@pytest.fixture(scope="module")
def reference(distinct_ckpt):
    from laya_poc.parity import load_reference

    return load_reference(distinct_ckpt)


@pytest.fixture(scope="module")
def exported(reference, tmp_path_factory):
    pytest.importorskip("onnxruntime")
    path = tmp_path_factory.mktemp("onnx_out") / "tiny.onnx"
    return path, OX.export(None, path, 18, agent=reference)


@pytest.mark.torch
def test_export_yields_a_graph_with_symbolic_batch_sequence_and_markers(exported):
    import onnx

    path, rec = exported
    assert path.is_file() and rec["bytes"] > 0 and rec["opset"] == 18 and rec["probe"]["logits_shape"] == [3, 4]
    assert rec["exporter"] in ("dynamo", "torchscript") and rec["attempt"] in (1, 2, 3)
    model = onnx.load(str(path), load_external_data=False)
    dims = {i.name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input}
    assert [i.name for i in model.graph.input] == OX.INPUT_NAMES
    assert all(isinstance(d, str) and d for d in dims["input_ids"] + dims["marker_pos"])  # no baked sizes
    assert [o.name for o in model.graph.output] == OX.OUTPUT_NAMES
    assert max(op.version for op in model.opset_import if op.domain in ("", "ai.onnx")) == 18


@pytest.mark.torch
def test_onnx_agrees_with_pytorch_fp32_on_identical_batches(exported, reference):
    from laya_poc import labels as L

    path, _ = exported
    acc = OX.check(None, path, _states(30), 30, question=L.question("c10"), max_len=512, head_max_len=192,
                   batch_size=8, agent=reference)
    assert acc["passed"] and acc["n"] == 30 and acc["argmax_agree"] == 1.0 and acc["n_disagree"] == 0
    assert acc["max_dp"] <= 1e-3 and acc["max_dlogit"] < 1e-2 and acc["nan_ref"] == acc["nan_test"] == 0
    assert acc["options"] == 10 and acc["seq_len_range"][0] > 128  # past the sliding-attention window
    strict = OX.check(None, path, _states(8), 8, question=L.question("c10"), max_len=512, head_max_len=192,
                      max_dp=0.0, agent=reference)
    assert strict["passed"] is False  # the bar is enforced, not decorative


@pytest.mark.torch
def test_batched_onnx_matches_the_pytorch_agent_on_a_dynamic_shape_batch(exported, reference, distinct_ckpt):
    from laya_poc import labels as L
    from laya_poc.bench_onnx import load_onnx_agent, onnx_predict_batch

    path, _ = exported
    q, states, budget = L.question("c10"), _states(11), {"max_len": 512, "head_max_len": 192}
    agent = load_onnx_agent(distinct_ckpt, path, threads=2)
    got = onnx_predict_batch(agent, states, q, batch_size=4, sort_by_length=True, **budget)  # batches 4, 4, 3
    want = reference.predict_batch(states, q, batch_size=4, sort_by_length=True, **budget)
    assert len(got) == len(want) == 11
    for g, w in zip(got, want):
        ga, wa = g["answers"]["place_category"], w["answers"]["place_category"]
        assert ga["choice"] == wa["choice"] and g["usage"] == w["usage"]
        assert max(abs(ga["probabilities"][k] - wa["probabilities"][k]) for k in wa["probabilities"]) <= 1e-3
    one = onnx_predict_batch(agent, states[:1], q, batch_size=1, **budget)[0]
    assert one["answers"] == agent.predict(states[0], q, **budget)["answers"]  # batch-1 == upstream ONNXAgent
    assert onnx_predict_batch(agent, [], q) == []
    with pytest.raises(TypeError):
        onnx_predict_batch(agent, states[0], q)


@pytest.mark.torch
def test_onnx_sessions_are_pinned_to_the_thread_count(exported, distinct_ckpt):
    import onnxruntime as ort
    from laya_poc.bench_onnx import load_onnx_agent, make_session

    before = ort.SessionOptions
    agent = load_onnx_agent(distinct_ckpt, exported[0], threads=3)
    assert agent.session.get_session_options().intra_op_num_threads == 3
    assert agent.session.get_providers()[0] == "CPUExecutionProvider"
    assert ort.SessionOptions is before  # the construction-time patch is undone
    assert make_session(exported[0], 1).get_session_options().intra_op_num_threads == 1
    with pytest.raises(ValueError, match="absolute"):
        load_onnx_agent(Path("rel"), exported[0], 1)


@pytest.mark.torch
def test_cli_writes_the_record_and_reuses_a_matching_export(distinct_ckpt, tmp_path, capsys):
    rows = tmp_path / "test_id.jsonl"
    write_jsonl(rows, [{"id": str(i), "state": s} for i, s in enumerate(_states(12))])
    out = tmp_path / "onnx" / "laya.onnx"
    argv = ["--model", "laya", "--ckpt", str(distinct_ckpt), "--out", str(out), "--check-rows", str(rows),
            "--check-n", "12", "--batch-size", "4", "--reuse"]
    assert OX.main(argv) == 0
    meta = json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))
    assert meta["passed"] and meta["acceptance"]["n"] == 12 and meta["export"]["opset"] == 18
    assert meta["identity"]["check"]["n"] == 12 and meta["ckpt_dir"] == str(distinct_ckpt)
    stamp = out.stat().st_mtime_ns
    assert OX.main(argv) == 0 and "reusing" in capsys.readouterr().out and out.stat().st_mtime_ns == stamp
    assert OX.main([*argv[:-1], "--check-n", "6"]) == 0 and out.stat().st_mtime_ns != stamp  # new check: redone


def test_cli_rejects_non_laya_models_and_relative_outputs(tmp_path, capsys):
    assert OX.main(["--model", "laya", "--out", "rel.onnx"]) == 1
    assert "absolute" in capsys.readouterr().err
    enc = tmp_path / "enc"
    enc.mkdir()
    (enc / "config.json").write_text("{}", encoding="utf-8")
    assert OX.main(["--model", "mmbert_small", "--ckpt", str(enc), "--out", str(tmp_path / "x.onnx")]) == 1
    assert "not a Laya model" in capsys.readouterr().err
