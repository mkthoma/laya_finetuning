import json

import pytest

from laya_poc import zeroshot as Z


def _row(i, label):
    return {"id": f"val-{i:06d}", "state": json.dumps({"name": f"n{i}"}), "label": label, "questions": "{}"}


def test_select_rows_skips_null_labels_and_keeps_order():
    rows = [_row(0, None), _row(1, "dining"), _row(2, None), _row(3, "retail"), _row(4, "arts")]
    picked = Z.select_rows(rows, 2)
    assert [r["id"] for r in picked] == ["val-000001", "val-000003"]


def test_select_rows_raises_when_nothing_labelled():
    with pytest.raises(ValueError, match="labelled"):
        Z.select_rows([_row(0, None)], 5)


def test_summarise_counts_correct_and_confidence():
    rows = [_row(1, "dining"), _row(2, "retail")]
    answers = [{"choice": "dining", "answer_confidence": 0.8}, {"choice": "arts", "answer_confidence": 0.4}]
    s = Z.summarise(rows, answers)
    assert s["correct"] == 1 and s["n"] == 2 and s["acc"] == 0.5
    assert s["mean_answer_confidence"] == pytest.approx(0.6)
    assert s["per_row"][1] == {"id": "val-000002", "label": "retail", "choice": "arts", "answer_confidence": 0.4}


def test_shared_question_rejects_mixed_questions():
    a = {"questions": json.dumps({"q": {"type": "choice"}})}
    b = {"questions": json.dumps({"q": {"type": "score"}})}
    assert Z.shared_question([a, a]) == {"q": {"type": "choice"}}
    with pytest.raises(ValueError, match="same question"):
        Z.shared_question([a, b])


@pytest.mark.torch
def test_cli_both_models_on_cpu(tiny_ckpt_dir, synthetic_data_dir, tmp_path, capsys):
    out = tmp_path / "zeroshot.json"
    rc = Z.main(["--models", "laya", "laya_ml", "--init", str(tiny_ckpt_dir), "--rows",
                 str(synthetic_data_dir / "val.jsonl"), "--n", "5", "--out", str(out), "--device", "cpu"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert set(res) == {"laya", "laya_ml"}
    for m in res.values():
        assert m["n"] == 5 and 0 <= m["correct"] <= 5 and m["acc"] == m["correct"] / 5
        assert 0.0 < m["mean_answer_confidence"] <= 1.0
        assert len(m["per_row"]) == 5 and m["device"] == "cpu" and m["cpu_fallback"] is False
        assert set(m["per_row"][0]) == {"id", "label", "choice", "answer_confidence"}
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) <= 4 and any("laya_ml" in line for line in printed)


LAYA_OOM = "Warning: GPU memory exceeded during inference. Retrying this request on CPU..."


@pytest.mark.torch
def test_cli_records_a_per_call_cpu_fallback(tiny_ckpt_dir, synthetic_data_dir, tmp_path, monkeypatch, capsys):
    from laya_poc import hub
    real_load = hub.load_agent

    def noisy_load(*args, **kwargs):
        agent = real_load(*args, **kwargs)
        forward = agent._forward

        def oom_then_forward(b):
            print(LAYA_OOM)  # what Laya prints before answering the batch on CPU
            return forward(b)

        agent._forward = oom_then_forward
        return agent

    monkeypatch.setattr(hub, "load_agent", noisy_load)
    out = tmp_path / "zeroshot.json"
    rc = Z.main(["--models", "laya", "--init", str(tiny_ckpt_dir), "--rows", str(synthetic_data_dir / "val.jsonl"),
                 "--n", "3", "--out", str(out), "--device", "cpu"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert set(res) == {"laya"} and res["laya"]["cpu_fallback"] is True
    assert "CPU fallback" in capsys.readouterr().out


@pytest.mark.torch
def test_cli_requires_cuda_when_asked(tiny_ckpt_dir, synthetic_data_dir, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("needs a machine without CUDA")
    rc = Z.main(["--models", "laya", "--init", str(tiny_ckpt_dir), "--rows", str(synthetic_data_dir / "val.jsonl"),
                 "--n", "2", "--out", str(tmp_path / "z.json")])
    assert rc == 1
    assert "cuda" in capsys.readouterr().err.lower()


def test_cli_rejects_relative_init(synthetic_data_dir, tmp_path, capsys):
    rc = Z.main(["--models", "laya", "--init", "rel/ckpt", "--rows", str(synthetic_data_dir / "val.jsonl"),
                 "--out", str(tmp_path / "z.json"), "--device", "cpu"])
    assert rc == 1
    assert "absolute" in capsys.readouterr().err
