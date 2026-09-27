"""Head-only training (E4, design §7.10 Option B): train_single --freeze-encoder, on CPU with tiny checkpoints.

The encoder's parameters are frozen (requires_grad False) and the optimiser gets the head alone; loss, schedule,
eval, --save-best and resume are the full fine-tuning code paths. A resume refuses a checkpoint written with the
other freeze setting (the settings fingerprint carries head_only).
"""
import json
import shutil
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("laya")
from laya_poc import ckpt as C  # noqa: E402
from laya_poc import head_only as H  # noqa: E402
from laya_poc import schedule as S  # noqa: E402
from laya_poc import train_single as T  # noqa: E402

pytestmark = pytest.mark.torch

EPOCHS, MICRO_PER_EPOCH = 2, 6            # 12 rows / MB 2; ACC 2 -> 3 opt steps per epoch, 6 in total
TOTAL_MICRO = EPOCHS * MICRO_PER_EPOCH


class Killed(BaseException):
    """Stands in for SIGKILL in-process: escapes main()'s `except Exception`, nothing is cleaned up."""


@pytest.fixture(scope="module")
def learnable_ckpt(en_snapshot, tmp_path_factory):
    from conftest import build_tiny_checkpoint
    return build_tiny_checkpoint(en_snapshot, tmp_path_factory.mktemp("learnable") / "ckpt", init_scale=0.5)


@pytest.fixture(scope="module")
def data2(tmp_path_factory):
    from synth import write_synthetic_data_dir
    return write_synthetic_data_dir(tmp_path_factory.mktemp("head") / "data", n_train=12, n_val=16, epochs=EPOCHS)


@pytest.fixture(scope="module")
def fast_cfg(cfg, tmp_path_factory):
    """A raised head rate so the tiny head moves visibly; no early stop inside the 6 optimiser steps."""
    from laya_poc.config import with_overrides
    path = tmp_path_factory.mktemp("cfg") / "config.yaml"
    fast = with_overrides(cfg, {"train.lr_head": 3e-3, "train.warmup_frac": 0.0, "train.patience": 10})
    path.write_text(yaml.safe_dump(fast), encoding="utf-8")
    return path


def _args(run_dir, data, init, config, *extra, head_only=True, save_best=True, ckpt_every=2):
    return ["--run-dir", str(run_dir), "--data-dir", str(data), "--config", str(config), "--init", str(init),
            "--device", "cpu", "--micro-batch", "2", "--effective-batch", "4", "--epochs", str(EPOCHS),
            "--ckpt-every-micro-steps", str(ckpt_every), "--ckpt-every-min", "0", "--keep-last", "50",
            "--eval-every-opt-steps", "2", "--print-every", "0", *(["--save-best"] if save_best else []),
            *(["--freeze-encoder"] if head_only else []), *extra]


def _events(run_dir, kind=None):
    path = Path(run_dir) / "log.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [r for r in rows if kind is None or r["event"] == kind]


def _summary(run_dir):
    return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))


def _tensors(path):
    from safetensors.torch import load_file
    return load_file(str(path))


def _ckpt(run_dir, opt_step):
    return torch.load(str(Path(run_dir) / "ckpt" / C.ckpt_name(opt_step)), map_location="cpu", weights_only=False)


def _same_tensors(a, b):
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])][:3]


def _encoder(sd):
    return {k: v for k, v in sd.items() if k.startswith("encoder.")}


def _crash_at(monkeypatch, argv):
    monkeypatch.setattr(C, "hard_kill", lambda: (_ for _ in ()).throw(Killed()))
    with pytest.raises(Killed):
        T.main(argv)
    monkeypatch.undo()


def _ce_by_micro(events):
    return {e["micro_step"]: e["loss_ce"] for e in events if e["event"] == "micro"}


@pytest.fixture(scope="module")
def head_run(learnable_ckpt, data2, fast_cfg, tmp_path_factory):
    run = tmp_path_factory.mktemp("head_run") / "run"
    assert T.main(_args(run, data2, learnable_ckpt, fast_cfg, "--final-eval", "--save-final")) == 0
    return run


# ---- the freeze itself ----

class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Embedding(5, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 4))
        self.scorer = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 1))
        self.type_emb = torch.nn.Embedding(3, 4)


def test_freeze_encoder_freezes_the_encoder_and_leaves_the_optimiser_the_head():
    m = _Toy()
    n_encoder = sum(p.numel() for p in m.encoder.parameters())
    n_head = sum(p.numel() for name, p in m.named_parameters() if not name.startswith("encoder."))
    assert H.freeze_encoder(m) == n_encoder
    assert not any(p.requires_grad for p in m.encoder.parameters())
    assert all(p.requires_grad for name, p in m.named_parameters() if not name.startswith("encoder."))
    assert H.trainable_parameters(m) == n_head
    groups = S.param_groups(m, 2e-5, 1e-4, 0.01)
    assert sorted(g["name"] for g in groups) == ["head", "head_no_decay"]
    assert all(g["lr"] == 1e-4 for g in groups)                          # lr_head, decay rules as today
    assert {g["name"]: g["weight_decay"] for g in groups} == {"head": 0.01, "head_no_decay": 0.0}


def test_freeze_encoder_switches_off_encoder_checkpointing_only(tiny_ckpt_dir, cfg):
    from laya_poc import export
    s = S.Settings(model="laya", seed=11, device="cpu", card="T4", micro_batch=2, grad_accum=2, grad_ckpt=True,
                   fp16=False, epochs=1, max_len=512, head_max_len=192, max_micro_steps=None,
                   ckpt_every_micro_steps=None, ckpt_every_min=0.0, eval_every_opt_steps=None, keep_last=2,
                   head_only=True)
    model, _, _ = export.load_for_training(str(tiny_ckpt_dir), cfg, s)
    assert model.encoder.is_gradient_checkpointing and model.head_checkpointing is True
    H.freeze_encoder(model)
    assert not model.encoder.is_gradient_checkpointing        # nothing backpropagates into the encoder
    assert model.head_checkpointing is True                   # the head's checkpointing follows the card
    assert model.training                                     # everything else as loaded


# ---- a head-only run ----

def test_head_only_run_logs_head_only_and_trains_with_the_head_rates(head_run, learnable_ckpt):
    init = _tensors(learnable_ckpt / "model.safetensors")
    (start,) = _events(head_run, "start")
    n_head = sum(v.numel() for k, v in init.items() if not k.startswith("encoder.") and k != "temperature")
    assert start["head_only"] is True and start["trainable_params"] == n_head
    assert _summary(head_run)["head_only"] is True
    micro = _events(head_run, "micro")
    assert [e["micro_step"] for e in micro] == list(range(1, TOTAL_MICRO + 1))
    assert all(e["finite"] and e["loss_ce"] > 0 for e in micro)
    opt = _events(head_run, "opt")
    assert len(opt) == 6 and all(e["lr_enc"] is None and e["lr_head"] > 0 for e in opt)
    s = _summary(head_run)
    assert (s["nonfinite"], s["stop_reason"], s["opt_steps"], s["micro_steps"]) == (0, "epochs", 6, TOTAL_MICRO)


def test_frozen_encoder_stays_bit_identical_while_the_head_moves(head_run, learnable_ckpt):
    init = _tensors(learnable_ckpt / "model.safetensors")                 # fp16 on disk, trained in fp32
    trained = _ckpt(head_run, 6)["model"]
    _same_tensors(_encoder(trained), {k: v.float() for k, v in _encoder(init).items()})
    for name in ("final", "best"):                                        # the exports carry it unchanged too
        _same_tensors(_encoder(_tensors(head_run / name / "model.safetensors")), _encoder(init))
    moved = {k for k, v in trained.items() if not k.startswith("encoder.") and not torch.equal(v, init[k].float())}
    assert any(k.startswith("scorer.") for k in moved) and any(k.startswith("head.") for k in moved)
    assert any(k.startswith("type_emb.") for k in moved)


def test_the_optimiser_holds_only_head_parameters(head_run, learnable_ckpt):
    opt = _ckpt(head_run, 6)["optimizer"]
    assert sorted(g["name"] for g in opt["param_groups"]) == ["head", "head_no_decay"]
    init = _tensors(learnable_ckpt / "model.safetensors")
    n_head_tensors = sum(1 for k in init if not k.startswith("encoder.") and k != "temperature")
    assert sum(len(g["params"]) for g in opt["param_groups"]) == n_head_tensors


def test_save_best_exports_the_best_head_only_weights(head_run):
    import laya

    from laya_poc import labels as L
    s = _summary(head_run)
    assert s["evals"] >= 3 and C.best_step(head_run / "best") == s["best_opt_step"]
    best = _tensors(head_run / "best" / "model.safetensors")
    trained = _ckpt(head_run, s["best_opt_step"])["model"]
    _same_tensors(best, {k: v.half() if v.is_floating_point() else v for k, v in trained.items()})
    agent = laya.load(str(head_run / "best"), device="cpu")
    state = json.dumps({"country": "GB", "name": "Rosa Pizza 1"}, separators=(",", ":"))
    ans = agent.predict_batch([state], L.question("c10"), batch_size=1)[0]["answers"][L.QUESTION_NAME]
    assert set(ans["probabilities"]) == set(L.option_keys("c10"))


def test_head_only_resume_replays_the_uninterrupted_run(head_run, learnable_ckpt, data2, fast_cfg, tmp_path,
                                                        monkeypatch):
    run = tmp_path / "run"
    argv = _args(run, data2, learnable_ckpt, fast_cfg, "--final-eval", "--save-final", ckpt_every=4)
    _crash_at(monkeypatch, argv + ["--crash-at-micro-step", "9"])          # epoch 1; checkpoint at micro 8
    assert T.main(argv) == 0
    events = _events(run)
    (resumed,) = [e for e in events if e["event"] == "resumed"]
    assert (resumed["from_micro_step"], resumed["epoch"]) == (8, 1)
    after, ref = _ce_by_micro(events[events.index(resumed):]), _ce_by_micro(_events(head_run))
    assert sorted(after) == list(range(9, TOTAL_MICRO + 1))
    for k, ce in after.items():
        assert ce == pytest.approx(ref[k], abs=1e-6), k
    ours, theirs = _summary(run), _summary(head_run)
    for key in ("best_opt_step", "evals", "stop_reason", "opt_steps", "head_only"):
        assert ours[key] == theirs[key], key
    assert ours["resumed_from"] == 8
    for name in ("best", "final"):
        _same_tensors(_tensors(run / name / "model.safetensors"), _tensors(head_run / name / "model.safetensors"))


def test_head_checkpointing_does_not_change_the_head_only_trajectory(head_run, learnable_ckpt, data2, fast_cfg,
                                                                      tmp_path):
    run = tmp_path / "gc"
    assert T.main(_args(run, data2, learnable_ckpt, fast_cfg, "--grad-ckpt", "on")) == 0
    (start,) = _events(run, "start")
    assert start["grad_ckpt"] is True and start["head_only"] is True
    ours, ref = _ce_by_micro(_events(run)), _ce_by_micro(_events(head_run))
    assert sorted(ours) == sorted(ref)
    for k in ours:
        assert ours[k] == pytest.approx(ref[k], abs=1e-5), k
    _same_tensors(_ckpt(run, 6)["model"], _ckpt(head_run, 6)["model"])


# ---- resume refuses a checkpoint written with the other freeze setting ----

@pytest.fixture(scope="module")
def full_run(learnable_ckpt, data2, fast_cfg, tmp_path_factory):
    """A short full fine-tuning run (no --freeze-encoder) whose checkpoints the resume tests borrow."""
    run = tmp_path_factory.mktemp("full_run") / "run"
    assert T.main(_args(run, data2, learnable_ckpt, fast_cfg, head_only=False)) == 0
    return run


@pytest.mark.parametrize("source,resume_head_only", [("head", False), ("full", True)])
def test_resume_refuses_a_checkpoint_with_the_other_freeze_setting(head_run, full_run, learnable_ckpt, data2,
                                                                   fast_cfg, tmp_path, capsys, source,
                                                                   resume_head_only):
    run = tmp_path / "other"
    src = head_run if source == "head" else full_run
    shutil.copytree(src / "ckpt", run / "ckpt")
    assert T.main(_args(run, data2, learnable_ckpt, fast_cfg, head_only=resume_head_only)) == 1
    ours = [x for x in capsys.readouterr().err.splitlines() if x.startswith("train_single:")]
    assert len(ours) == 1 and "different settings" in ours[0] and "head_only" in ours[0], ours
    assert not _events(run, "micro") and not (run / "summary.json").exists()   # no training compute spent
    assert "head_only" in (run / "error.log").read_text(encoding="utf-8")


def test_a_checkpoint_from_before_the_flag_resumes_as_full_fine_tuning(full_run, learnable_ckpt, data2, fast_cfg,
                                                                       tmp_path, capsys):
    """Phase 2 checkpoints carry no head_only in their fingerprint: they were full fine-tuning runs."""
    run = tmp_path / "legacy"
    (run / "ckpt").mkdir(parents=True)
    ck = _ckpt(full_run, 4)
    legacy = {k: v for k, v in ck["state"]["fingerprint"].items() if k != "head_only"}
    torch.save({**ck, "state": {**ck["state"], "fingerprint": legacy}}, str(run / "ckpt" / C.ckpt_name(4)))
    assert T.main(_args(run, data2, learnable_ckpt, fast_cfg, head_only=True, save_best=False)) == 1
    assert "head_only" in capsys.readouterr().err
    assert T.main(_args(run, data2, learnable_ckpt, fast_cfg, head_only=False, save_best=False)) == 0
    (resumed,) = _events(run, "resumed")
    assert resumed["from_micro_step"] == 8 and _summary(run)["head_only"] is False
    assert _ckpt(run, 6)["state"]["fingerprint"] == {**legacy, "head_only": False}   # upgraded going forward
