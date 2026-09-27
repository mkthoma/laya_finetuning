import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("laya")
from laya_poc import train_single as T  # noqa: E402

pytestmark = pytest.mark.torch

ROOT = Path(__file__).resolve().parents[1]
MAX_MICRO = 12


def _args(run_dir, data_dir, init, *extra):
    return ["--run-dir", str(run_dir), "--data-dir", str(data_dir), "--config", str(ROOT / "config.yaml"),
            "--init", str(init), "--device", "cpu", "--micro-batch", "2", "--effective-batch", "4",
            "--max-micro-steps", str(MAX_MICRO), "--ckpt-every-micro-steps", "4", "--ckpt-every-min", "0",
            "--eval-every-opt-steps", "0", "--keep-last", "2", "--print-every", "4", *extra]


def _events(run_dir, kind=None):
    rows = [json.loads(line) for line in (Path(run_dir) / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if kind is None or r["event"] == kind]


def _after_last(events, kind):
    idx = max(i for i, e in enumerate(events) if e["event"] == kind)
    return events[idx + 1:]


def _ce_by_micro(events):
    return {e["micro_step"]: e["loss_ce"] for e in events if e["event"] == "micro"}


def _same(a, b, where="ckpt"):
    """Recursive bit-equality (tensors via torch.equal). The tiny model's loss_ce sits at ~ln(10)
    whatever its weights, so loss curves cannot tell a correct run from a broken one; CPU training
    is deterministic, so the checkpoints themselves must match exactly."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), where
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), where
        for k in a:
            _same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    else:
        assert a == b, (where, a, b)


def _assert_same_checkpoint(path_a, path_b):
    """Everything a resume restores. `state.initial_eval` is left out: it records the opt-0 eval of a run with
    --initial-eval (the control), which does not change training."""
    a, b = (torch.load(str(p), map_location="cpu", weights_only=False) for p in (path_a, path_b))
    for key in ("model", "optimizer", "scheduler", "scaler"):
        _same(a[key], b[key], key)
    _same(*({k: v for k, v in ck["state"].items() if k != "initial_eval"} for ck in (a, b)), "state")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    from synth import write_synthetic_data_dir
    return write_synthetic_data_dir(tmp_path_factory.mktemp("train_data") / "data")


@pytest.fixture(scope="module")
def control(tiny_ckpt_dir, data_dir, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("control") / "run"
    rc = T.main(_args(run_dir, data_dir, tiny_ckpt_dir, "--initial-eval", "--final-eval", "--save-final"))
    assert rc == 0
    return run_dir


def test_run_completes_with_consistent_log_and_summary(control):
    events = _events(control)
    start = events[0]
    assert start["event"] == "start" and start["device"] == "cpu" and start["card"] == "CPU"
    assert (start["micro_batch"], start["grad_accum"], start["n_items"]) == (2, 2, 64)
    assert start["total_opt_steps"] == 6 and 1.0 <= start["padding_ratio"] < 2.0
    assert start["truncated_items"] == 0
    micro = _events(control, "micro")
    assert [e["micro_step"] for e in micro] == list(range(1, MAX_MICRO + 1))
    assert all(e["finite"] and e["loss_ce"] > 0 and e["sigma"] == pytest.approx(0.4) for e in micro)
    opt = _events(control, "opt")
    assert [(e["opt_step"], e["micro_step"]) for e in opt] == [(n, 2 * n) for n in range(1, 7)]
    assert all(e["lr_enc"] > 0 and e["lr_head"] > e["lr_enc"] and e["scale"] == 1.0 for e in opt)
    assert [(e["opt_step"], e["micro_step"]) for e in _events(control, "ckpt")] == [(2, 4), (4, 8), (6, 12)]
    evals = _events(control, "eval")
    assert [(e["opt_step"], e.get("initial")) for e in evals] == [(0, True), (6, None)]
    assert all(e["n"] == 32 and 0.0 <= e["val_acc"] <= 1.0 and e["seconds"] >= 0 for e in evals)
    kinds = [e["event"] for e in events]
    assert kinds.index("start") < kinds.index("eval") < kinds.index("micro")  # before the first micro-step
    assert events[-1] == {**events[-1], "event": "done", "micro_steps": MAX_MICRO, "opt_steps": 6}

    summary = json.loads((control / "summary.json").read_text(encoding="utf-8"))
    assert summary["micro_steps"] == MAX_MICRO and summary["opt_steps"] == 6
    assert summary["resumed_from"] is None and summary["nonfinite"] == 0 and summary["min_scale"] == 1.0
    assert summary["final_eval"]["n"] == 32 and summary["n_train_items"] == 64
    metrics = ("val_ce", "val_acc", "val_macro_f1", "n")
    assert {k: summary["initial_eval"][k] for k in metrics} == {k: evals[0][k] for k in metrics}
    assert (summary["device"], summary["card"], summary["micro_batch"], summary["grad_accum"]) == ("cpu", "CPU", 2, 2)
    assert (summary["model"], summary["seed"]) == ("laya", 11)
    assert summary["peak_vram_reserved_gb"] is None and summary["sec_per_micro_median"] > 0
    assert (control / "config.yaml").exists()


def test_full_fine_tuning_is_the_default_and_logged_as_such(control, tiny_ckpt_dir):
    from safetensors.torch import load_file
    (start,) = _events(control, "start")
    params = sum(v.numel() for k, v in load_file(str(tiny_ckpt_dir / "model.safetensors")).items()
                 if k != "temperature")                                  # the only buffer in the state dict
    assert start["head_only"] is False and start["trainable_params"] == params
    assert json.loads((control / "summary.json").read_text(encoding="utf-8"))["head_only"] is False
    fingerprint = torch.load(str(control / "ckpt" / "step0000006.pt"), map_location="cpu",
                             weights_only=False)["state"]["fingerprint"]
    assert fingerprint["head_only"] is False


def test_optimizer_updates_encoder_and_head(control, tiny_ckpt_dir):
    from safetensors.torch import load_file
    before = load_file(str(tiny_ckpt_dir / "model.safetensors"))
    after = load_file(str(control / "final" / "model.safetensors"))
    assert set(before) == set(after)
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert any(k.startswith("encoder.") for k in changed) and any(k.startswith("scorer.") for k in changed)


def test_checkpoints_pruned_to_keep_last_without_tmp_files(control):
    names = sorted(p.name for p in (control / "ckpt").iterdir())
    assert names == ["step0000004.pt", "step0000006.pt"]


def test_final_checkpoint_loads_and_predicts(control):
    import laya

    from laya_poc import labels as L
    from safetensors.torch import load_file

    trained = torch.load(str(control / "ckpt" / "step0000006.pt"), map_location="cpu", weights_only=False)["model"]
    exported = load_file(str(control / "final" / "model.safetensors"))
    assert set(exported) == set(trained)
    for k, v in trained.items():               # final/ holds the trained weights, cast to fp16
        assert torch.equal(exported[k], v.half() if v.is_floating_point() else v), k
    cfg = json.loads((control / "final" / "rl_agent_config.json").read_text(encoding="utf-8"))
    assert cfg["model_name"] == "laya-poc-laya-s11" and (cfg["max_len"], cfg["head_max_len"]) == (512, 192)
    agent = laya.load(str(control / "final"), device="cpu")
    state = json.dumps({"country": "GB", "name": "Rosa Pizza 1"}, separators=(",", ":"))
    ans = agent.predict_batch([state], L.question("c10"), batch_size=1)[0]["answers"][L.QUESTION_NAME]
    assert set(ans["probabilities"]) == set(L.option_keys("c10"))


def test_final_checkpoint_notice_records_its_provenance(control, tiny_ckpt_dir):
    from laya_poc.config import load_config
    cfg = load_config(ROOT / "config.yaml")
    notice = (control / "final" / "NOTICE.md").read_text(encoding="utf-8")
    assert f"Base checkpoint: {tiny_ckpt_dir} (" in notice  # a local --init has no Hub revision
    assert f"/blob/{cfg['laya']['commit']}/LICENSE" in notice and "commit was not recorded" not in notice
    fsq = (control / "final" / "NOTICE_FSQ.txt").read_text(encoding="utf-8")
    assert f"release {cfg['data']['fsq_release']};" in fsq


def test_resume_after_hard_kill_replays_the_control_run(control, tiny_ckpt_dir, data_dir, tmp_path, capsys):
    run_dir = tmp_path / "resumed"
    # Bit-equality needs the same CPU reduction order: the crash subprocess must use this process's torch
    # thread count (an earlier in-process test may have changed it with torch.set_num_threads).
    threads = str(torch.get_num_threads())
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
           "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
    crash = subprocess.run([sys.executable, "-m", "laya_poc.train_single",
                            *_args(run_dir, data_dir, tiny_ckpt_dir, "--crash-at-micro-step", "6")],
                           env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert crash.returncode != 0, crash.stdout + crash.stderr
    crashed = _events(run_dir)
    assert crashed[-1] == {**crashed[-1], "event": "crash_injected", "micro_step": 6}
    assert [p.name for p in (run_dir / "ckpt").iterdir()] == ["step0000002.pt"]

    assert T.main(_args(run_dir, data_dir, tiny_ckpt_dir, "--initial-eval")) == 0
    assert "resumed from" in capsys.readouterr().out
    events = _events(run_dir)
    (resumed,) = [e for e in events if e["event"] == "resumed"]
    assert (resumed["from_micro_step"], resumed["from_opt_step"], resumed["epoch"]) == (4, 2, 0)
    after = _ce_by_micro(_after_last(events, "resumed"))
    assert sorted(after) == list(range(5, MAX_MICRO + 1))
    reference = _ce_by_micro(_events(control))
    before = _ce_by_micro(events[:events.index(resumed)])
    for k in range(1, 7):
        assert before[k] == pytest.approx(reference[k], abs=1e-5), k
    for k in range(5, MAX_MICRO + 1):
        assert after[k] == pytest.approx(reference[k], abs=1e-5), k
    # The loss curve alone misses a resume that skips the model/optimizer/scheduler restore. The control
    # ran --initial-eval and this run did not: the equality also shows that eval leaves training untouched.
    _assert_same_checkpoint(control / "ckpt" / "step0000006.pt", run_dir / "ckpt" / "step0000006.pt")
    assert not _events(run_dir, "eval")      # --initial-eval is skipped on a resume: not the initial weights
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["resumed_from"] == 4 and summary["micro_steps"] == MAX_MICRO and summary["opt_steps"] == 6
    assert summary["initial_eval"] is None
    assert not list((run_dir / "ckpt").glob("*.tmp"))
    _assert_report_reads_the_logs(_events(control), events)


def _assert_report_reads_the_logs(control_log, resumed_log):
    """smoke_report's resume and loss-drop criteria on the real log schema (not only on hand-made events)."""
    from laya_poc import smoke_report as S
    from laya_poc.config import load_config
    exit_cfg = load_config(ROOT / "config.yaml")["smoke"]["exit"]
    crit = {r["name"]: r for r in S.evaluate_exit({"control_log": control_log, "resumed_log": resumed_log},
                                                   exit_cfg, kill_at=6, ckpt_every=4, compare_steps=8)}
    assert crit[S.RESUME_POINT]["passed"], crit[S.RESUME_POINT]
    resume = crit[S.RESUME_LOSS]
    assert resume["passed"] and "lr_enc, lr_head, scale match at 4 opt steps" in resume["note"], resume
    assert crit[S.LOSS_DROP]["value"] is not None, crit[S.LOSS_DROP]   # the initial and final eval were found


def test_gradient_checkpointing_gives_the_same_trajectory(control, tiny_ckpt_dir, data_dir, tmp_path):
    run_dir = tmp_path / "gc"
    assert T.main(_args(run_dir, data_dir, tiny_ckpt_dir, "--grad-ckpt", "on")) == 0
    assert _events(run_dir, "start")[0]["grad_ckpt"] is True
    ours, reference = _ce_by_micro(_events(run_dir)), _ce_by_micro(_events(control))
    assert sorted(ours) == sorted(reference)
    for k in ours:
        assert ours[k] == pytest.approx(reference[k], abs=1e-5), k
    _assert_same_checkpoint(control / "ckpt" / "step0000006.pt", run_dir / "ckpt" / "step0000006.pt")


def test_resume_refuses_a_checkpoint_from_different_settings(control, tiny_ckpt_dir, data_dir, tmp_path, capsys):
    import shutil
    run_dir = tmp_path / "other"
    shutil.copytree(control / "ckpt", run_dir / "ckpt")
    args = _args(run_dir, data_dir, tiny_ckpt_dir)
    args[args.index("--max-micro-steps") + 1] = "10"
    assert T.main(args) == 1
    ours = [x for x in capsys.readouterr().err.splitlines() if x.startswith("train_single:")]  # not laya warnings
    assert len(ours) == 1 and "different settings" in ours[0]
    assert (run_dir / "error.log").exists()


@pytest.mark.parametrize("flag", ["--final-eval", "--initial-eval"])
def test_missing_val_file_fails_before_training(flag, tiny_ckpt_dir, data_dir, tmp_path, capsys):
    import shutil
    no_val = tmp_path / "data"
    shutil.copytree(data_dir, no_val)
    (no_val / "val.jsonl").unlink()
    run_dir = tmp_path / "noval"
    assert T.main(_args(run_dir, no_val, tiny_ckpt_dir, flag)) == 1
    assert "val.jsonl" in capsys.readouterr().err
    assert not (run_dir / "log.jsonl").exists()      # no training compute spent


def test_final_export_survives_a_failing_final_eval(tiny_ckpt_dir, data_dir, tmp_path, monkeypatch):
    def oom(self):
        raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr(T.Trainer, "evaluate", oom)
    run_dir = tmp_path / "evalfail"
    assert T.main(_args(run_dir, data_dir, tiny_ckpt_dir, "--final-eval", "--save-final")) == 1
    assert (run_dir / "final" / "model.safetensors").exists()   # the trained weights are not lost


def test_nonfinite_losses_abort_the_run(tiny_ckpt_dir, data_dir, tmp_path, monkeypatch, capsys):
    from laya_poc.loss import LossParts

    real = T.upstream_loss

    def nan_loss(*args, **kwargs):
        p = real(*args, **kwargs)
        return LossParts(total=p.total * float("nan"), ce=p.ce, rl=p.rl, reward=p.reward)

    monkeypatch.setattr(T, "upstream_loss", nan_loss)
    run_dir = tmp_path / "nan"
    assert T.main(_args(run_dir, data_dir, tiny_ckpt_dir, "--max-nonfinite", "2")) == 1
    micro = _events(run_dir, "micro")
    assert len(micro) == 3 and not any(e["finite"] for e in micro)
    assert "non-finite" in capsys.readouterr().err
    assert not (run_dir / "summary.json").exists()


def test_relative_paths_are_rejected_with_one_line_error(tiny_ckpt_dir, data_dir, capsys):
    assert T.main(_args("relative/run", data_dir, tiny_ckpt_dir)) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "absolute" in err


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA error path")
def test_cuda_requested_without_cuda_fails_fast(tiny_ckpt_dir, data_dir, tmp_path, capsys):
    args = _args(tmp_path / "gpu", data_dir, tiny_ckpt_dir)
    args[args.index("--device") + 1] = "cuda"
    assert T.main(args) == 1
    assert "CUDA" in capsys.readouterr().err


# ---- pure helpers ----

def test_vram_is_reported_in_decimal_gigabytes(monkeypatch):
    # the exit cap (config smoke.exit.max_vram_gb) is in GB = 10^9 bytes, as the design doc means it
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 7_250_000_000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 14_500_000_000)
    assert T.vram_gb("cuda") == {"vram_alloc_gb": 7.25, "vram_reserved_gb": 14.5}
    assert T.vram_gb("cpu") == {"vram_alloc_gb": None, "vram_reserved_gb": None}
