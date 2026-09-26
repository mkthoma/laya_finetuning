import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from laya_poc import ckpt as C  # noqa: E402

pytestmark = pytest.mark.torch


def _setup(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (s + 1))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, opt, sched, scaler


def _train_step(model, opt, sched):
    loss = model(torch.randn(5, 4)).pow(2).sum()
    loss.backward()
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)


def _save(tmp_path, parts, opt_step, **kw):
    model, opt, sched, scaler = parts
    state = {"epoch": 0, "batch_idx": opt_step * 2, "micro_step": opt_step * 2, "opt_step": opt_step}
    return C.save_train_ckpt(tmp_path / "ckpt", model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                             state=state, **kw)


def test_save_names_by_opt_step_and_leaves_no_tmp(tmp_path):
    path = _save(tmp_path, _setup(), 30)
    assert path == tmp_path / "ckpt" / "step0000030.pt"
    assert path.exists() and not list((tmp_path / "ckpt").glob("*.tmp"))


def test_prunes_to_keep_last_and_removes_stale_tmp(tmp_path):
    parts = _setup()
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "step0000001.pt.tmp").write_bytes(b"partial write from a killed process")
    for step in (2, 4, 6, 8):
        _save(tmp_path, parts, step, keep_last=2)
    names = sorted(p.name for p in (tmp_path / "ckpt").iterdir())
    assert names == ["step0000006.pt", "step0000008.pt"]


def test_mirror_copy_is_pruned_too(tmp_path):
    parts = _setup()
    mirror = tmp_path / "drive" / "run"
    for step in (1, 2, 3):
        _save(tmp_path, parts, step, keep_last=2, mirror_dir=mirror)
    assert sorted(p.name for p in mirror.iterdir()) == ["step0000002.pt", "step0000003.pt"]
    assert (mirror / "step0000003.pt").read_bytes() == (tmp_path / "ckpt" / "step0000003.pt").read_bytes()


def test_keep_last_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="keep_last"):
        _save(tmp_path, _setup(), 1, keep_last=0)


def test_latest_prefers_local_ignores_tmp_and_falls_back_to_mirror(tmp_path):
    local, mirror = tmp_path / "ckpt", tmp_path / "mirror"
    assert C.latest_ckpt(local, mirror) is None
    mirror.mkdir()
    for name in ("step0000009.pt", "step0000010.pt"):
        (mirror / name).write_bytes(b"x")
    assert C.latest_ckpt(local, mirror) == mirror / "step0000010.pt"
    local.mkdir()
    (local / "step0000012.pt.tmp").write_bytes(b"x")
    (local / "notes.txt").write_bytes(b"x")
    assert C.latest_ckpt(local, mirror) == mirror / "step0000010.pt"
    for name in ("step0000002.pt", "step0000008.pt"):
        (local / name).write_bytes(b"x")
    assert C.latest_ckpt(local, mirror) == local / "step0000008.pt"   # local wins even if older
    assert C.latest_ckpt(local) == local / "step0000008.pt"


def test_round_trip_restores_model_optimizer_scheduler_and_state(tmp_path):
    parts = _setup(seed=0)
    model, opt, sched, _ = parts
    for _ in range(3):
        _train_step(model, opt, sched)
    path = _save(tmp_path, parts, 3)
    fresh = _setup(seed=1)
    state = C.load_train_ckpt(path, model=fresh[0], optimizer=fresh[1], scheduler=fresh[2], scaler=fresh[3])
    assert state == {"epoch": 0, "batch_idx": 6, "micro_step": 6, "opt_step": 3}
    for a, b in zip(model.state_dict().values(), fresh[0].state_dict().values()):
        assert torch.equal(a, b)
    assert fresh[2].last_epoch == sched.last_epoch == 3
    assert fresh[1].param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    # Continuing from the restored copy gives exactly the same next step.
    torch.manual_seed(5)
    _train_step(model, opt, sched)
    torch.manual_seed(5)
    _train_step(fresh[0], fresh[1], fresh[2])
    for a, b in zip(model.state_dict().values(), fresh[0].state_dict().values()):
        assert torch.equal(a, b)


def test_load_maps_to_cpu_and_never_to_cuda(tmp_path, monkeypatch):
    parts = _setup()
    path = _save(tmp_path, parts, 1)
    seen = {}
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    fresh = _setup(seed=1)
    C.load_train_ckpt(path, model=fresh[0], optimizer=fresh[1], scheduler=fresh[2], scaler=fresh[3])
    assert seen["map_location"] == "cpu" and seen["weights_only"] is False


def test_load_rejects_a_file_that_is_not_a_training_checkpoint(tmp_path):
    bad = tmp_path / "step0000001.pt"
    torch.save({"weights": 1}, bad)
    fresh = _setup()
    with pytest.raises(ValueError, match="missing"):
        C.load_train_ckpt(bad, model=fresh[0], optimizer=fresh[1], scheduler=fresh[2], scaler=fresh[3])


def test_state_is_copied_not_aliased(tmp_path):
    parts = _setup()
    state = {"epoch": 0, "batch_idx": 2, "micro_step": 2, "opt_step": 1}
    C.save_train_ckpt(tmp_path, model=parts[0], optimizer=parts[1], scheduler=parts[2], scaler=parts[3],
                      state=state)
    assert state == {"epoch": 0, "batch_idx": 2, "micro_step": 2, "opt_step": 1}


def test_ckpt_due_fires_once_per_crossed_multiple():
    assert C.ckpt_due(0, 4, 4) and not C.ckpt_due(4, 6, 4) and C.ckpt_due(6, 8, 4)
    assert C.ckpt_due(118, 120, 40) and not C.ckpt_due(120, 124, 40)
    assert C.ckpt_due(266, 271, 40) is False and C.ckpt_due(271, 275, 40) is False
    assert C.ckpt_due(278, 282, 40)                      # 280 crossed between unaligned boundaries
    assert not C.ckpt_due(0, 4, None) and not C.ckpt_due(0, 4, 0)


def test_hard_kill_ends_the_process_with_a_nonzero_code():
    code = "from laya_poc.ckpt import hard_kill; print('before', flush=True); hard_kill(); print('after')"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode in (-9, 137) and out.stdout.strip() == "before"
