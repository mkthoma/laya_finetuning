"""E1 behaviour of train_single (Phase 2 spec §3): --save-best, the gate's GradScaler accounting and
exact multi-epoch resume, on CPU with tiny checkpoints.

Bookkeeping tests script the val metrics by optimiser step (Trainer._val_metrics), so which eval is
"best" is fixed; the replay tests use a learnable tiny model and compare with an uninterrupted run.
"""
import json
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("laya")
from laya_poc import ckpt as C  # noqa: E402
from laya_poc import train_single as T  # noqa: E402

pytestmark = pytest.mark.torch

EPOCHS, MICRO_PER_EPOCH = 3, 6          # 12 rows / MB 2; ACC 2 -> 3 opt steps per epoch, 9 in total


class Killed(BaseException):
    """Stands in for SIGKILL in-process: escapes main()'s `except Exception`, nothing is cleaned up."""


@pytest.fixture(scope="module")
def learnable_ckpt(en_snapshot, tmp_path_factory):
    from conftest import build_tiny_checkpoint
    return build_tiny_checkpoint(en_snapshot, tmp_path_factory.mktemp("learnable") / "ckpt", init_scale=0.5)


@pytest.fixture(scope="module")
def data3(tmp_path_factory):
    from synth import write_synthetic_data_dir
    return write_synthetic_data_dir(tmp_path_factory.mktemp("e1") / "data", n_train=12, n_val=32, epochs=EPOCHS)


@pytest.fixture(scope="module")
def fast_cfg(cfg, tmp_path_factory):
    """Raised learning rates so the tiny model's val metrics move between evals (doc rates are for 421M)."""
    from laya_poc.config import with_overrides
    path = tmp_path_factory.mktemp("cfg") / "config.yaml"
    fast = with_overrides(cfg, {"train.lr_encoder": 1e-3, "train.lr_head": 3e-3, "train.warmup_frac": 0.0})
    path.write_text(yaml.safe_dump(fast), encoding="utf-8")
    return path


def _args(run_dir, data, init, config, *extra, ckpt_every=2, eval_every=2):
    return ["--run-dir", str(run_dir), "--data-dir", str(data), "--config", str(config), "--init", str(init),
            "--device", "cpu", "--micro-batch", "2", "--effective-batch", "4", "--epochs", str(EPOCHS),
            "--ckpt-every-micro-steps", str(ckpt_every), "--ckpt-every-min", "0", "--keep-last", "50",
            "--eval-every-opt-steps", str(eval_every), "--print-every", "0", "--save-best", *extra]


def _events(run_dir, kind=None):
    rows = [json.loads(x) for x in (Path(run_dir) / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if kind is None or r["event"] == kind]


def _summary(run_dir):
    return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))


def _tensors(path):
    from safetensors.torch import load_file
    return load_file(str(path))


def _same_tensors(a, b):
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])][:3]


def _ckpt_weights_fp16(run_dir, opt_step):
    sd = torch.load(str(run_dir / "ckpt" / C.ckpt_name(opt_step)), map_location="cpu", weights_only=False)["model"]
    return {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}


def _crash_at(monkeypatch, run_argv):
    monkeypatch.setattr(C, "hard_kill", lambda: (_ for _ in ()).throw(Killed()))
    with pytest.raises(Killed):
        T.main(run_argv)
    monkeypatch.undo()


def _script(monkeypatch, f1_by_opt, calls=None):
    """Val metrics as a function of the optimiser step; `calls` collects the opt steps evaluated."""
    def metrics(self, items):
        opt = self.state["opt_step"]
        if calls is not None:
            calls.append(opt)
        f1 = f1_by_opt(opt) if callable(f1_by_opt) else f1_by_opt[opt]
        return {"val_ce": 2.0 - f1, "val_acc": f1, "val_macro_f1": f1, "n": len(items)}
    monkeypatch.setattr(T.Trainer, "_val_metrics", metrics)


# ---- --save-best bookkeeping (scripted metrics) ----

def test_save_best_exports_on_each_improvement_and_the_final_eval_counts(tiny_ckpt_dir, data3, cfg, tmp_path,
                                                                          monkeypatch):
    run = tmp_path / "run"
    # min_delta 0.002: opt 4 (+0.001) is not an improvement; the final eval (opt 9) beats every periodic one
    _script(monkeypatch, {2: 0.30, 4: 0.301, 6: 0.40, 8: 0.35, 9: 0.45})
    assert T.main(_args(run, data3, tiny_ckpt_dir, _config(cfg, tmp_path, {}), "--final-eval")) == 0
    best = _events(run, "best")
    assert [(e["opt_step"], e["val_macro_f1"]) for e in best] == [(2, 0.30), (6, 0.40), (9, 0.45)]
    assert all(e["seconds"] >= 0 for e in best)
    s = _summary(run)
    assert (s["best_opt_step"], s["best_eval"]["val_macro_f1"], s["evals"], s["stop_reason"]) == (9, 0.45, 5, "epochs")
    assert s["final_eval"]["val_macro_f1"] == 0.45
    marker = json.loads((run / "best" / C.BEST_INFO).read_text(encoding="utf-8"))
    assert (marker["opt_step"], marker["micro_step"], marker["model"], marker["seed"]) == (9, 18, "laya", 11)
    assert marker["final_eval"]["val_macro_f1"] == 0.45
    _same_tensors(_tensors(run / "best" / "model.safetensors"), _ckpt_weights_fp16(run, 9))
    assert not (run / C.BEST_PREV).exists() and not (run / "best.partial").exists()


def test_best_holds_the_weights_of_the_best_eval(tiny_ckpt_dir, data3, cfg, tmp_path, monkeypatch):
    import laya

    run = tmp_path / "run"
    _script(monkeypatch, {2: 0.2, 4: 0.5, 6: 0.3, 8: 0.3})
    assert T.main(_args(run, data3, tiny_ckpt_dir, _config(cfg, tmp_path, {"train.patience": 10}))) == 0
    assert _summary(run)["best_opt_step"] == 4
    _same_tensors(_tensors(run / "best" / "model.safetensors"), _ckpt_weights_fp16(run, 4))
    from laya_poc import labels as L
    rl = json.loads((run / "best" / "rl_agent_config.json").read_text(encoding="utf-8"))
    assert rl["model_name"] == "laya-poc-laya-s11" and "temperature_by_options" not in rl
    agent = laya.load(str(run / "best"), device="cpu")          # the marker file does not bother the loader
    state = json.dumps({"country": "GB", "name": "Rosa Pizza 1"}, separators=(",", ":"))
    ans = agent.predict_batch([state], L.question("c10"), batch_size=1)[0]["answers"][L.QUESTION_NAME]
    assert set(ans["probabilities"]) == set(L.option_keys("c10"))


def test_early_stop_reuses_the_last_eval_as_final_eval(tiny_ckpt_dir, data3, cfg, tmp_path, monkeypatch):
    run, calls = tmp_path / "run", []
    _script(monkeypatch, lambda opt: 0.5, calls)                  # never improves after the first eval
    assert T.main(_args(run, data3, tiny_ckpt_dir, _config(cfg, tmp_path, {}), "--final-eval")) == 0
    s = _summary(run)
    assert (s["stop_reason"], s["opt_steps"], s["micro_steps"]) == ("early_stop", 8, 16)   # patience 3: 4, 6, 8
    assert calls == [2, 4, 6, 8] and s["evals"] == 4                 # no second eval of the opt-8 weights
    assert s["final_eval"]["val_macro_f1"] == 0.5 and s["best_opt_step"] == 2


def test_early_stop_survives_a_kill_after_the_stopping_checkpoint(tiny_ckpt_dir, data3, cfg, tmp_path,
                                                                  monkeypatch):
    run, config = tmp_path / "run", _config(cfg, tmp_path, {})
    monkeypatch.setattr(T.Trainer, "finish", lambda self: (_ for _ in ()).throw(Killed()))
    _script(monkeypatch, lambda opt: 0.5)
    with pytest.raises(Killed):
        T.main(_args(run, data3, tiny_ckpt_dir, config))
    monkeypatch.undo()
    assert C.latest_ckpt(run / "ckpt").name == C.ckpt_name(8)       # taken at the stopping boundary
    _script(monkeypatch, lambda opt: 0.5)
    assert T.main(_args(run, data3, tiny_ckpt_dir, config)) == 0
    after = _events(run)[max(i for i, e in enumerate(_events(run)) if e["event"] == "resumed"):]
    assert not [e for e in after if e["event"] in ("micro", "opt", "eval")]   # does not train past the stop
    s = _summary(run)
    assert (s["stop_reason"], s["opt_steps"], s["best_opt_step"], s["evals"]) == ("early_stop", 8, 2, 4)


def test_best_exported_after_the_last_checkpoint_is_rolled_back_on_resume(tiny_ckpt_dir, data3, cfg, tmp_path,
                                                                          monkeypatch):
    """GPU replays are not bit-exact: the lost trajectory's best@6 must not survive when the replay's eval at
    opt 6 no longer improves. The state restored from ckpt@4 says best@4, so best/ must hold the opt-4 weights."""
    run, config = tmp_path / "run", _config(cfg, tmp_path, {"train.patience": 10})
    argv = _args(run, data3, tiny_ckpt_dir, config, ckpt_every=8)            # checkpoints at opt 4 and 8 only
    _script(monkeypatch, {2: 0.3, 4: 0.4, 6: 0.5})
    _crash_at(monkeypatch, argv + ["--crash-at-micro-step", "13"])          # after best@6, before ckpt@8
    assert C.best_step(run / "best") == 6 and C.best_step(run / C.BEST_PREV) == 4
    _script(monkeypatch, {2: 0.3, 4: 0.4, 6: 0.401, 8: 0.39})                # the replay's opt-6 eval differs
    assert T.main(argv) == 0
    (rec,) = _events(run, "best_reconciled")
    assert (rec["outcome"], rec["best_opt_step"]) == ("restored", 4)
    assert _summary(run)["best_opt_step"] == 4 and C.best_step(run / "best") == 4
    _same_tensors(_tensors(run / "best" / "model.safetensors"), _ckpt_weights_fp16(run, 4))
    assert not (run / C.BEST_PREV).exists()


def test_without_any_eval_save_best_exports_the_final_weights_with_a_warning(tiny_ckpt_dir, data3, cfg, tmp_path):
    run = tmp_path / "run"
    assert T.main(_args(run, data3, tiny_ckpt_dir, _config(cfg, tmp_path, {}), eval_every=0)) == 0
    (warn,) = _events(run, "warning")
    assert "no val eval" in warn["message"]
    assert [e["opt_step"] for e in _events(run, "best")] == [9]
    s = _summary(run)
    assert (s["best_opt_step"], s["best_eval"], s["evals"]) == (9, None, 0)
    _same_tensors(_tensors(run / "best" / "model.safetensors"), _ckpt_weights_fp16(run, 9))


def test_eval_uses_the_configured_batch_size(tiny_ckpt_dir, data3, cfg, tmp_path, monkeypatch):
    seen = []
    real = T.evaluate_items

    def spy(*a, **kw):
        seen.append(kw["batch_size"])
        return real(*a, **kw)

    monkeypatch.setattr(T, "evaluate_items", spy)
    run = tmp_path / "run"
    assert T.main(_args(run, data3, tiny_ckpt_dir, _config(cfg, tmp_path, {"eval.batch_size": 5}),
                        "--max-micro-steps", "4")) == 0
    assert seen == [5] and _events(run, "eval")[0]["n"] == 32


# ---- gate accounting: GradScaler skips vs non-finite gradients applied ----

def test_scaler_skips_are_counted_and_not_reported_as_applied_nonfinite(tiny_ckpt_dir, data3, cfg, tmp_path,
                                                                        monkeypatch):
    from laya_poc.loss import LossParts
    monkeypatch.setattr(T, "make_scaler", lambda enabled: torch.amp.GradScaler("cpu", init_scale=16.0,
                                                                                growth_interval=3))
    real, n = T.upstream_loss, {"micro": 0}

    def inf_on_micro_3(*a, **kw):
        n["micro"] += 1
        p = real(*a, **kw)
        return LossParts(total=p.total * float("inf"), ce=p.ce, rl=p.rl, reward=p.reward) if n["micro"] == 3 else p

    monkeypatch.setattr(T, "upstream_loss", inf_on_micro_3)
    run = tmp_path / "run"
    assert T.main(_args(run, data3, tiny_ckpt_dir, _config(cfg, tmp_path, {}), eval_every=0)) == 0
    opt = _events(run, "opt")
    assert [e["skipped"] for e in opt] == [False, True] + [False] * 7
    assert [e["scale"] for e in opt] == [16.0, 8.0, 8.0, 8.0, 16.0, 16.0, 16.0, 32.0, 32.0]
    assert opt[1]["grad_norm"] != opt[1]["grad_norm"] or opt[1]["grad_norm"] == float("inf")
    s = _summary(run)
    assert (s["opt_steps_skipped"], s["nonfinite_grad_applied"], s["min_scale"], s["nonfinite"]) == (1, 0, 8.0, 1)


def test_nonfinite_grad_norm_on_an_applied_step_is_counted(tiny_ckpt_dir, data3, cfg, tmp_path, monkeypatch):
    real, n = torch.nn.utils.clip_grad_norm_, {"opt": 0}

    def clip(*a, **kw):
        n["opt"] += 1
        norm = real(*a, **kw)
        return torch.tensor(float("nan")) if n["opt"] == 5 else norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    run = tmp_path / "run"
    assert T.main(_args(run, data3, tiny_ckpt_dir, _config(cfg, tmp_path, {}), eval_every=0)) == 0
    s = _summary(run)
    assert (s["opt_steps_skipped"], s["nonfinite_grad_applied"], s["min_scale"]) == (0, 1, 1.0)
    assert [e["skipped"] for e in _events(run, "opt")] == [False] * 9


# ---- multi-epoch resume replays the uninterrupted run exactly (learnable model, real evals) ----

@pytest.fixture(scope="module")
def control(learnable_ckpt, data3, fast_cfg, tmp_path_factory):
    run = tmp_path_factory.mktemp("control") / "run"
    assert T.main(_args(run, data3, learnable_ckpt, fast_cfg, "--final-eval", "--save-final")) == 0
    return run


def _micro(events):
    return {e["micro_step"]: (e["epoch"], e["loss_ce"], e["sigma"]) for e in events if e["event"] == "micro"}


def _opt(events):
    return {e["opt_step"]: (e["lr_enc"], e["lr_head"], e["scale"]) for e in events if e["event"] == "opt"}


@pytest.mark.parametrize("ckpt_every,crash_at,resume_micro,resume_epoch,reconciled", [
    (2, 7, 6, 1, []),       # checkpoint exactly at the end of epoch 0: resumes at epoch 1, batch 0
    (2, 9, 8, 1, []),       # mid epoch 1
    (2, 15, 14, 2, []),     # epoch 2
    (8, 13, 8, 1, ["restored"]),  # the opt-6 eval exported best/ after the last checkpoint: rolled back, redone
])
def test_resume_at_any_micro_step_replays_the_control_run(control, learnable_ckpt, data3, fast_cfg, tmp_path,
                                                          monkeypatch, ckpt_every, crash_at, resume_micro,
                                                          resume_epoch, reconciled):
    run = tmp_path / "run"
    argv = _args(run, data3, learnable_ckpt, fast_cfg, "--final-eval", "--save-final", ckpt_every=ckpt_every)
    _crash_at(monkeypatch, argv + ["--crash-at-micro-step", str(crash_at)])
    assert T.main(argv) == 0
    events = _events(run)
    (resumed,) = [e for e in events if e["event"] == "resumed"]
    assert (resumed["from_micro_step"], resumed["epoch"]) == (resume_micro, resume_epoch)
    assert [e["outcome"] for e in events if e["event"] == "best_reconciled"] == reconciled
    ref, after = _events(control), events[events.index(resumed):]
    total = EPOCHS * MICRO_PER_EPOCH
    assert sorted(_micro(after)) == list(range(resume_micro + 1, total + 1))
    for k, (epoch, ce, sigma) in _micro(after).items():
        assert (epoch, sigma) == _micro(ref)[k][::2] and ce == pytest.approx(_micro(ref)[k][1], abs=1e-6), k
    assert {k: v for k, v in _opt(ref).items() if k > resume_micro // 2} == _opt(after)
    ref_evals = {e["opt_step"]: e["val_macro_f1"] for e in ref if e["event"] == "eval"}
    for e in (e for e in after if e["event"] == "eval"):
        assert e["val_macro_f1"] == pytest.approx(ref_evals[e["opt_step"]], abs=1e-9)
    ours, theirs = _summary(run), _summary(control)
    for key in ("best_opt_step", "evals", "stop_reason", "opt_steps", "micro_steps", "opt_steps_skipped",
                "nonfinite_grad_applied", "min_scale", "truncated_items"):
        assert ours[key] == theirs[key], key
    assert ours["best_eval"]["val_macro_f1"] == pytest.approx(theirs["best_eval"]["val_macro_f1"], abs=1e-9)
    _same_tensors(_tensors(run / "best" / "model.safetensors"), _tensors(control / "best" / "model.safetensors"))
    _same_tensors(_tensors(run / "final" / "model.safetensors"), _tensors(control / "final" / "model.safetensors"))


def test_control_run_moves_its_best_and_logs_the_gate_fields(control):
    s = _summary(control)
    for key in ("best_eval", "best_opt_step", "opt_steps_skipped", "nonfinite_grad_applied", "min_scale", "evals",
                "stop_reason"):
        assert key in s, key
    assert s["evals"] >= 4 and s["opt_steps_skipped"] == 0 and s["nonfinite_grad_applied"] == 0
    assert C.best_step(control / "best") == s["best_opt_step"]
    best = [e["opt_step"] for e in _events(control, "best")]
    assert best[-1] == s["best_opt_step"] and len(best) >= 2       # the best moves: the replay test means something


def test_resume_from_a_checkpoint_without_the_phase2_state_keys(control, learnable_ckpt, data3, fast_cfg, tmp_path,
                                                                monkeypatch):
    """A Phase 1 checkpoint lacks skipped/evals/best_* keys: they default instead of failing the resume."""
    run = tmp_path / "run"
    argv = _args(run, data3, learnable_ckpt, fast_cfg, "--final-eval", "--save-final", eval_every=0)
    _crash_at(monkeypatch, argv + ["--crash-at-micro-step", "5"])
    path = C.latest_ckpt(run / "ckpt")
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    old_keys = ("epoch", "batch_idx", "micro_step", "opt_step", "best", "bad", "nonfinite", "min_scale", "fingerprint")
    ck["state"] = {k: ck["state"][k] for k in old_keys}
    torch.save(ck, str(path))
    assert T.main(argv) == 0
    s = _summary(run)
    assert s["resumed_from"] == 4 and s["opt_steps_skipped"] == 0 and s["micro_steps"] == EPOCHS * MICRO_PER_EPOCH


def test_save_best_flag_is_parsed():
    ns = T.build_parser().parse_args(["--run-dir", "/r", "--data-dir", "/d", "--save-best"])
    assert ns.save_best is True
    assert T.build_parser().parse_args(["--run-dir", "/r", "--data-dir", "/d"]).save_best is False



# ---- best/ must match the run's best: mirrored for a resume from --mirror-dir, checked at the end ----

def test_resume_from_the_mirror_after_the_run_dir_is_lost_restores_best(tiny_ckpt_dir, data3, cfg, tmp_path,
                                                                        monkeypatch):
    import shutil
    run, mirror, config = tmp_path / "run", tmp_path / "drive", _config(cfg, tmp_path, {"train.patience": 10})
    argv = _args(run, data3, tiny_ckpt_dir, config, "--mirror-dir", str(mirror), ckpt_every=8)
    f1 = {2: 0.5, 4: 0.3, 6: 0.3, 8: 0.3}
    _script(monkeypatch, f1)
    _crash_at(monkeypatch, argv + ["--crash-at-micro-step", "13"])            # ckpt@4 mirrored with best@2
    weights = (run / "best" / "model.safetensors").read_bytes()
    assert sorted(p.name for p in mirror.iterdir()) == ["best-0000002", C.ckpt_name(4)]
    shutil.rmtree(run)                                                       # VM recycled: only Drive survives
    _script(monkeypatch, f1)
    assert T.main(argv) == 0
    (rec,) = _events(run, "best_reconciled")
    assert (rec["outcome"], rec["best_opt_step"]) == ("from_mirror", 2)
    assert _summary(run)["best_opt_step"] == 2 and C.best_step(run / "best") == 2
    assert (run / "best" / "model.safetensors").read_bytes() == weights
    assert sorted(p.name for p in mirror.iterdir() if p.name.startswith("best")) == ["best-0000002"]


def test_mirror_follows_the_best_and_survives_a_kill_before_the_mirrored_checkpoint(tiny_ckpt_dir, data3, cfg,
                                                                                   tmp_path, monkeypatch):
    """best@6 is mirrored before ckpt@8, which is killed before it reaches the mirror: the mirror's newest
    checkpoint (opt 4) still finds its best@2 there, and the replay moves the mirror on to best@6."""
    import shutil
    run, mirror, config = tmp_path / "run", tmp_path / "drive", _config(cfg, tmp_path, {"train.patience": 10})
    argv = _args(run, data3, tiny_ckpt_dir, config, "--mirror-dir", str(mirror), ckpt_every=8)
    f1 = {2: 0.3, 4: 0.3, 6: 0.5, 8: 0.3}
    real, calls = C.mirror_ckpt, []

    def killed_at_opt_8(path, *a, **kw):
        calls.append(Path(path).name)
        if Path(path).name == C.ckpt_name(8):
            raise Killed()
        return real(path, *a, **kw)

    monkeypatch.setattr(C, "mirror_ckpt", killed_at_opt_8)
    _script(monkeypatch, f1)
    with pytest.raises(Killed):
        T.main(argv)
    monkeypatch.undo()
    assert sorted(p.name for p in mirror.iterdir()) == ["best-0000002", "best-0000006", C.ckpt_name(4)]
    shutil.rmtree(run)
    _script(monkeypatch, f1)
    assert T.main(argv) == 0
    assert [e["outcome"] for e in _events(run, "best_reconciled")] == ["from_mirror"]
    assert [e["opt_step"] for e in _events(run, "best")] == [6]              # re-exported by the replay
    assert _summary(run)["best_opt_step"] == 6 and C.best_step(run / "best") == 6
    names = sorted(p.name for p in mirror.iterdir())
    assert names == ["best-0000006", C.ckpt_name(4), C.ckpt_name(8)]


def test_resume_fails_fast_when_the_checkpointed_best_is_gone(tiny_ckpt_dir, data3, cfg, tmp_path, monkeypatch,
                                                                capsys):
    import shutil
    run, config = tmp_path / "run", _config(cfg, tmp_path, {"train.patience": 10})
    argv = _args(run, data3, tiny_ckpt_dir, config, ckpt_every=8)
    _script(monkeypatch, {2: 0.5, 4: 0.3, 6: 0.3})
    _crash_at(monkeypatch, argv + ["--crash-at-micro-step", "13"])
    shutil.rmtree(run / "best")
    n_events = len(_events(run))
    _script(monkeypatch, {2: 0.5, 4: 0.3, 6: 0.3, 8: 0.3})
    assert T.main(argv) == 1
    assert "opt 2" in capsys.readouterr().err
    assert not [e for e in _events(run)[n_events:] if e["event"] in ("micro", "opt", "eval")]   # did not train
    assert not (run / "summary.json").exists() and "best" in (run / "error.log").read_text(encoding="utf-8")


def test_run_does_not_report_success_without_a_matching_best(tiny_ckpt_dir, data3, cfg, tmp_path, monkeypatch):
    run = tmp_path / "run"
    _script(monkeypatch, {2: 0.5, 4: 0.3, 6: 0.3, 8: 0.3})
    monkeypatch.setattr(T.Trainer, "_export_best", lambda self, m: None)     # best/ never written
    assert T.main(_args(run, data3, tiny_ckpt_dir, _config(cfg, tmp_path, {"train.patience": 10}))) == 1
    assert not (run / "summary.json").exists()
    assert "best" in (run / "error.log").read_text(encoding="utf-8")


def test_initial_eval_survives_the_forced_resume(tiny_ckpt_dir, data3, cfg, tmp_path, monkeypatch):
    """E1 runs --initial-eval in both the crash and the resume command; only the crash process evaluates opt 0."""
    run, config = tmp_path / "run", _config(cfg, tmp_path, {"train.patience": 10})
    argv = _args(run, data3, tiny_ckpt_dir, config, "--initial-eval")
    _script(monkeypatch, lambda opt: 0.1 + opt / 100)
    _crash_at(monkeypatch, argv + ["--crash-at-micro-step", "13"])
    _script(monkeypatch, lambda opt: 0.1 + opt / 100)
    assert T.main(argv) == 0
    (initial,) = [e for e in _events(run, "eval") if e.get("initial")]      # evaluated once, before the crash
    s = _summary(run)
    assert s["resumed_from"] == 12 and s["initial_eval"]["val_macro_f1"] == pytest.approx(0.1)
    assert s["initial_eval"] == {k: initial[k] for k in s["initial_eval"]}
    assert s["evals"] == 4                                                  # the opt-0 eval is not a periodic one

def _config(cfg, tmp_path, overrides):
    from laya_poc.config import with_overrides
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(with_overrides(cfg, overrides)), encoding="utf-8")
    return path
