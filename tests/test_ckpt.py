import json
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


# ---- loop state ----

def test_initial_state_has_every_key_the_trainer_and_the_gate_read():
    fp = {"model": "laya", "seed": 11}
    st = C.initial_state(fp)
    assert st == {"epoch": 0, "batch_idx": 0, "micro_step": 0, "opt_step": 0, "best": -1.0, "bad": 0,
                  "nonfinite": 0, "min_scale": None, "skipped": 0, "nonfinite_applied": 0, "evals": 0,
                  "best_opt_step": None, "best_eval": None, "last_eval": None, "initial_eval": None,
                  "truncated": {}, "fingerprint": fp}
    assert C.initial_state(fp)["truncated"] is not st["truncated"]       # no shared mutable default


# ---- GradScaler step outcome (gate: skipped fp16 steps are reported, non-finite grads applied must be 0) ----

@pytest.mark.parametrize("enabled,before,after,norm,expected", [
    (True, 65536.0, 32768.0, float("inf"), (True, False)),    # found inf: step skipped, scale backed off
    (True, 65536.0, 32768.0, float("nan"), (True, False)),
    (True, 65536.0, 65536.0, 3.2, (False, False)),            # normal step
    (True, 65536.0, 131072.0, 3.2, (False, False)),           # growth after growth_interval clean steps
    (True, 8.0, 8.0, float("inf"), (False, True)),            # finite grads whose norm overflowed: applied
    (False, 1.0, 1.0, float("nan"), (False, True)),           # no scaler (CPU): nothing is ever skipped
    (False, 1.0, 1.0, 0.7, (False, False)),
])
def test_opt_step_outcome(enabled, before, after, norm, expected):
    assert C.opt_step_outcome(enabled, before, after, norm) == expected


def test_opt_step_outcome_matches_a_real_grad_scaler():
    """The scale drop is exactly GradScaler's found-inf decision (CPU GradScaler, same update kernel)."""
    scaler = torch.amp.GradScaler("cpu", init_scale=16.0, growth_interval=2)
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.SGD([p], lr=0.1)
    seen = []
    for bad in (False, True, False, False, False, True):
        before, w = scaler.get_scale(), p.detach().clone()
        scaler.scale((p * (float("inf") if bad else 1.0)).sum()).backward()
        scaler.unscale_(opt)
        norm = float(torch.nn.utils.clip_grad_norm_([p], 1.0))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        skipped, applied = C.opt_step_outcome(True, before, scaler.get_scale(), norm)
        assert skipped is bad and applied is False
        assert torch.equal(p.detach(), w) is bad                         # a skipped step leaves the weights
        seen.append(scaler.get_scale())
    assert seen == [16.0, 8.0, 8.0, 16.0, 16.0, 8.0]


# ---- best/ directory across crashes (save-best) ----

def _best(run_dir, name, opt_step):
    d = run_dir / name
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(f"weights@{opt_step}".encode())
    (d / C.BEST_INFO).write_text(json.dumps(C.best_marker(opt_step=opt_step, micro_step=4 * opt_step,
                                                          model="laya", seed=11, eval_result=None)))
    return d


def _weights(d):
    return (d / "model.safetensors").read_bytes().decode()


def test_best_marker_is_readable_by_export_check_as_a_train_summary():
    ev = {"val_ce": 1.2, "val_acc": 0.5, "val_macro_f1": 0.4, "n": 32, "seconds": 1.0}
    m = C.best_marker(opt_step=250, micro_step=1000, model="laya", seed=11, eval_result=ev)
    assert (m["opt_step"], m["micro_step"], m["val_macro_f1"], m["model"], m["seed"]) == (250, 1000, 0.4, "laya", 11)
    assert m["final_eval"] == ev          # export_check --train-summary <best>/train_eval.json compares with this
    assert C.best_marker(opt_step=9, micro_step=36, model="laya", seed=11, eval_result=None)["val_macro_f1"] is None


def test_best_step_reads_the_marker(tmp_path):
    assert C.best_step(tmp_path / "best") is None
    _best(tmp_path, "best", 7)
    assert C.best_step(tmp_path / "best") == 7
    (tmp_path / "best" / C.BEST_INFO).write_text("{not json")
    assert C.best_step(tmp_path / "best") is None


def test_stage_best_parks_only_a_committed_best(tmp_path):
    _best(tmp_path, "best", 3)
    C.stage_best(tmp_path, committed_opt_step=2)          # best@3 came after the last checkpoint: overwritten
    assert C.best_step(tmp_path / "best") == 3 and not (tmp_path / C.BEST_PREV).exists()
    C.stage_best(tmp_path, committed_opt_step=None)       # no checkpoint yet: nothing is committed
    assert not (tmp_path / C.BEST_PREV).exists()
    C.stage_best(tmp_path, committed_opt_step=3)          # committed by the checkpoint at opt 3: keep it
    assert not (tmp_path / "best").exists() and C.best_step(tmp_path / C.BEST_PREV) == 3
    C.stage_best(tmp_path, committed_opt_step=3)          # nothing to park (best/ is being rewritten)
    assert C.best_step(tmp_path / C.BEST_PREV) == 3


def test_commit_best_drops_the_parked_copy(tmp_path):
    _best(tmp_path, "best", 5)
    _best(tmp_path, C.BEST_PREV, 3)
    C.commit_best(tmp_path)
    assert C.best_step(tmp_path / "best") == 5 and not (tmp_path / C.BEST_PREV).exists()
    C.commit_best(tmp_path)                               # idempotent


def test_reconcile_restores_the_best_the_checkpoint_knows(tmp_path):
    # ckpt at opt 4 had best@4; eval at opt 6 exported best@6 (best@4 parked); killed before the next ckpt
    _best(tmp_path, C.BEST_PREV, 4)
    _best(tmp_path, "best", 6)
    (tmp_path / "best.partial").mkdir()
    assert C.reconcile_best(tmp_path, best_opt_step=4) == "restored"
    assert C.best_step(tmp_path / "best") == 4 and _weights(tmp_path / "best") == "weights@4"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best"]


def test_reconcile_keeps_a_matching_best_and_drops_a_stale_parked_copy(tmp_path):
    _best(tmp_path, "best", 4)
    _best(tmp_path, C.BEST_PREV, 2)
    assert C.reconcile_best(tmp_path, best_opt_step=4) == "kept"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best"]


def test_reconcile_removes_a_best_the_checkpoint_never_saw(tmp_path):
    _best(tmp_path, "best", 3)                            # exported before the first checkpoint
    assert C.reconcile_best(tmp_path, best_opt_step=None) == "removed"
    assert not (tmp_path / "best").exists()


def test_reconcile_restores_after_a_kill_between_park_and_export(tmp_path):
    _best(tmp_path, C.BEST_PREV, 4)                       # best/ parked, the new export never landed
    assert C.reconcile_best(tmp_path, best_opt_step=4) == "restored"
    assert C.best_step(tmp_path / "best") == 4


def test_reconcile_reports_a_lost_best_and_leaves_foreign_dirs_alone(tmp_path):
    assert C.reconcile_best(tmp_path, best_opt_step=None) == "kept"          # fresh run, nothing on disk
    assert C.reconcile_best(tmp_path, best_opt_step=4) == "missing"          # deleted by hand
    (tmp_path / "best").mkdir()                                              # no marker: not ours
    assert C.reconcile_best(tmp_path, best_opt_step=None) == "kept" and (tmp_path / "best").exists()


# ---- best/ in the --mirror-dir: a resume from the mirror after the run dir is lost ----

def test_mirror_best_copies_the_named_best_and_skips_an_identical_copy(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "drive"
    _best(run, "best", 4)
    C.mirror_best(run, mirror, 4)
    copy = C.mirror_best_dir(mirror, 4)
    assert copy.name == "best-0000004" and C.best_step(copy) == 4 and _weights(copy) == "weights@4"
    (copy / "model.safetensors").write_bytes(b"sentinel")          # same export (same marker): not copied again
    C.mirror_best(run, mirror, 4)
    assert _weights(copy) == "sentinel"
    C.mirror_best(run, mirror, None)                               # no best yet: nothing to mirror
    assert sorted(p.name for p in mirror.iterdir()) == ["best-0000004"]


def test_mirror_best_replaces_a_copy_of_another_export_at_the_same_step(tmp_path):
    """A lost trajectory's orphan copy (mirrored, then killed before its checkpoint) must not stand in for
    the replay's export at that step: GPU replays are not bit-exact, so the markers (eval) differ."""
    run, mirror = tmp_path / "run", tmp_path / "drive"
    orphan = _best(mirror, "best-0000004", 4)
    (orphan / C.BEST_INFO).write_text(json.dumps(C.best_marker(opt_step=4, micro_step=16, model="laya", seed=11,
                                                               eval_result={"val_macro_f1": 0.3})))
    (mirror / "best-0000004.partial").mkdir()                      # a copy killed half-way
    _best(run, "best", 4)
    C.mirror_best(run, mirror, 4)
    assert _weights(C.mirror_best_dir(mirror, 4)) == "weights@4"
    assert sorted(p.name for p in mirror.iterdir()) == ["best-0000004"]


def test_mirror_best_refuses_a_best_dir_that_is_not_the_named_best(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "drive"
    _best(run, "best", 6)
    with pytest.raises(RuntimeError, match="opt 4"):
        C.mirror_best(run, mirror, 4)
    assert not C.mirror_best_dir(mirror, 4).exists()


def test_prune_mirror_best_keeps_only_the_copy_the_mirrored_checkpoint_names(tmp_path):
    mirror = tmp_path / "drive"
    for step in (2, 6, 9):
        _best(mirror, f"best-{step:07d}", step)
    (mirror / "best-0000009.partial").mkdir()
    (mirror / "step0000008.pt").write_bytes(b"x")
    (mirror / "best").mkdir()                                      # not a copy of ours: left alone
    C.prune_mirror_best(mirror, 6)
    assert sorted(p.name for p in mirror.iterdir()) == ["best", "best-0000006", "step0000008.pt"]
    C.prune_mirror_best(mirror, None)
    assert sorted(p.name for p in mirror.iterdir()) == ["best", "step0000008.pt"]
    C.prune_mirror_best(tmp_path / "absent", 3)                    # no mirror yet: nothing to do


def test_mirror_ckpt_copies_atomically_and_prunes(tmp_path):
    parts, mirror = _setup(), tmp_path / "drive"
    paths = [_save(tmp_path, parts, step, keep_last=5) for step in (1, 2, 3)]
    for p in paths:
        C.mirror_ckpt(p, mirror, keep_last=2)
    assert sorted(p.name for p in mirror.iterdir()) == ["step0000002.pt", "step0000003.pt"]
    assert (mirror / "step0000003.pt").read_bytes() == paths[-1].read_bytes()


def test_reconcile_restores_the_best_from_the_mirror_when_the_run_dir_lost_it(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "drive"
    run.mkdir()
    _best(mirror, "best-0000002", 2)
    _best(mirror, "best-0000006", 6)       # mirrored, then killed before the checkpoint that names it
    assert C.reconcile_best(run, best_opt_step=2, mirror_dir=mirror) == "from_mirror"
    assert C.best_step(run / "best") == 2 and _weights(run / "best") == "weights@2"
    assert sorted(p.name for p in run.iterdir()) == ["best"]
    assert C.best_step(C.mirror_best_dir(mirror, 2)) == 2          # the mirror keeps its copy


def test_reconcile_prefers_the_local_copies_over_the_mirror(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "drive"
    _best(run, C.BEST_PREV, 4)
    _best(run, "best", 6)
    m = _best(mirror, "best-0000004", 4)
    (m / "model.safetensors").write_bytes(b"mirror copy")
    assert C.reconcile_best(run, best_opt_step=4, mirror_dir=mirror) == "restored"
    assert _weights(run / "best") == "weights@4"


def test_reconcile_reports_missing_when_neither_the_run_dir_nor_the_mirror_has_it(tmp_path):
    run, mirror = tmp_path / "run", tmp_path / "drive"
    _best(run, "best", 6)
    _best(mirror, "best-0000002", 2)
    assert C.reconcile_best(run, best_opt_step=4, mirror_dir=mirror) == "missing"
    assert C.reconcile_best(run, best_opt_step=4, mirror_dir=tmp_path / "no-mirror") == "missing"
    assert C.best_step(run / "best") == 6                          # left for the user to inspect
