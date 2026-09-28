"""Resumable training checkpoints (design doc §7.6.5, fixed per critique §A).

Checkpoints are written only at optimiser-step boundaries (right after zero_grad), so no
gradient state is lost. No RNG state is saved: the trainer reseeds the global RNG every
micro-step from (seed, epoch, micro index), which makes a resumed run replay exactly.
Loading always maps to CPU; `map_location="cuda"` breaks RNG/optimizer restores (critique §A).

Also here: the loop state every checkpoint carries (incl. the gate's GradScaler accounting), and the
`best/` export (--save-best) kept consistent with that state across a kill (see reconcile_best), also in
the --mirror-dir copy a resume falls back to when the run dir is lost (see mirror_best).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import signal
import sys
from pathlib import Path
from typing import Any

import torch

_NAME = re.compile(r"^step(\d+)\.pt$")
_TMP = re.compile(r"^step\d+\.pt\.tmp$")
_KEYS = ("model", "optimizer", "scheduler", "scaler", "state")
FORMAT_VERSION = 1


def ckpt_name(opt_step: int) -> str:
    return f"step{opt_step:07d}.pt"


def _checkpoints(directory: Path) -> list[Path]:
    """Complete checkpoints in a directory, oldest first (by step number, never by mtime)."""
    if not directory.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in directory.iterdir() if (m := _NAME.match(p.name))]
    return [p for _, p in sorted(found)]


def _prune(directory: Path, keep_last: int) -> None:
    for stale in (p for p in directory.iterdir() if _TMP.match(p.name)):
        stale.unlink(missing_ok=True)
    for old in _checkpoints(directory)[:-keep_last]:
        old.unlink(missing_ok=True)


def _atomic_copy(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir / (src.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst_dir / src.name)
    return dst_dir / src.name


def save_train_ckpt(ckpt_dir: str | Path, *, model: Any, optimizer: Any, scheduler: Any, scaler: Any,
                    state: dict, keep_last: int = 2, mirror_dir: str | Path | None = None) -> Path:
    """Write `step{opt_step:07d}.pt` atomically, prune to `keep_last`, optionally mirror a copy.

    The local write completes before the mirror copy starts, so a slow or failing mirror (e.g. a
    Drive FUSE mount) can never corrupt the local checkpoint the next resume prefers.
    """
    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / ckpt_name(int(state["opt_step"]))
    tmp = path.with_name(path.name + ".tmp")
    payload = {"format": FORMAT_VERSION, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "state": dict(state)}
    torch.save(payload, tmp)
    os.replace(tmp, path)
    _prune(ckpt_dir, keep_last)
    if mirror_dir is not None:
        mirror_ckpt(path, mirror_dir, keep_last)
    return path


def mirror_ckpt(path: str | Path, mirror_dir: str | Path, keep_last: int) -> Path:
    """Copy a complete local checkpoint to the mirror atomically and prune the mirror to `keep_last`."""
    out = _atomic_copy(Path(path), Path(mirror_dir))
    _prune(Path(mirror_dir), keep_last)
    return out


def latest_ckpt(ckpt_dir: str | Path, mirror_dir: str | Path | None = None) -> Path | None:
    """Newest complete local checkpoint; the mirror is used only when there is none locally."""
    local = _checkpoints(Path(ckpt_dir))
    if local:
        return local[-1]
    mirrored = _checkpoints(Path(mirror_dir)) if mirror_dir is not None else []
    return mirrored[-1] if mirrored else None


def load_train_ckpt(path: str | Path, *, model: Any, optimizer: Any, scheduler: Any, scaler: Any) -> dict:
    """Restore everything in place (the model must already be on its training device) and
    return a copy of the saved loop state. Optimizer state follows the parameters' device."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    missing = [k for k in _KEYS if not isinstance(ck, dict) or k not in ck]
    if missing:
        raise ValueError(f"{path} is not a training checkpoint: missing {missing}")
    model.load_state_dict(ck["model"], strict=True)
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    scaler.load_state_dict(ck["scaler"])
    return dict(ck["state"])


def ckpt_due(prev_micro: int, micro: int, every: int | None) -> bool:
    """True when a multiple of `every` micro-steps was reached since the previous optimiser
    boundary (checkpoints are only ever taken at boundaries)."""
    return bool(every) and micro // every > prev_micro // every


def hard_kill() -> None:
    """Kill this process the way a Colab kernel kill does, to drill resume (SIGKILL: no cleanup,
    no atexit). Windows has no SIGKILL, so exit with 137 (128 + 9) instead."""
    sys.stdout.flush()
    sys.stderr.flush()
    if hasattr(signal, "SIGKILL"):
        os.kill(os.getpid(), signal.SIGKILL)
    os._exit(137)


# ---- loop state: what every checkpoint carries besides the tensors ----

def initial_state(fingerprint: dict) -> dict:
    """Loop state of a fresh run; a resume overlays the saved state on it, so keys added after
    Phase 1 (skipped ... truncated) get their defaults from an older checkpoint too."""
    return {"epoch": 0, "batch_idx": 0, "micro_step": 0, "opt_step": 0, "best": -1.0, "bad": 0,
            "nonfinite": 0, "min_scale": None, "skipped": 0, "nonfinite_applied": 0, "evals": 0,
            "best_opt_step": None, "best_eval": None, "last_eval": None, "initial_eval": None, "truncated": {},
            "fingerprint": fingerprint}


def opt_step_outcome(scaler_enabled: bool, scale_before: float, scale_after: float,
                     grad_norm: float) -> tuple[bool, bool]:
    """(skipped, nonfinite_applied) of one optimiser step.

    GradScaler.step skips optimizer.step exactly when unscale_ found an inf/NaN gradient, and update()
    then multiplies the scale by backoff_factor (< 1, asserted by GradScaler); a step that ran keeps or
    grows it. So a drop of get_scale() across step+update IS the skip decision. A non-finite grad norm
    on a step that ran means non-finite gradients reached AdamW (the gate requires none); without a
    scaler (CPU) nothing is ever skipped.
    """
    skipped = bool(scaler_enabled) and scale_after < scale_before
    return skipped, not skipped and not math.isfinite(grad_norm)


# ---- best/ (--save-best): the best-val weights, consistent with the checkpointed state ----
#
# A new best is exported right after its eval, but the state that records it is only saved at the next
# checkpoint. If the run is killed in between, the resumed state knows the previous best, which the
# export overwrote. So the best a checkpoint committed is parked in best.prev/ until the next checkpoint
# (commit_best), and a resume puts back the best its state refers to (reconcile_best).
# With --mirror-dir, the best each mirrored checkpoint names is copied to <mirror>/best-<opt step>/ before that
# checkpoint is (mirror_best), so a resume from the mirror after the run dir is lost finds it too.

BEST_DIR, BEST_PREV, BEST_INFO = "best", "best.prev", "train_eval.json"
_MIRROR_BEST = re.compile(r"^best-\d+(\.partial)?$")


def best_marker(*, opt_step: int, micro_step: int, model: str, seed: int, eval_result: dict | None) -> dict:
    """Content of <best>/train_eval.json. `final_eval` repeats the eval under the key export_check reads
    from a train summary (`--train-summary <best>/train_eval.json` compares best/ with its own eval)."""
    f1 = None if eval_result is None else eval_result.get("val_macro_f1")
    return {"opt_step": int(opt_step), "micro_step": int(micro_step), "val_macro_f1": f1, "model": model,
            "seed": seed, "final_eval": eval_result,
            "note": "final_eval = the training-time val eval of these weights (not of the final weights)"}


def best_step(directory: str | Path) -> int | None:
    """opt_step in a best dir's marker; None when the dir or a readable marker is absent."""
    try:
        return int(json.loads((Path(directory) / BEST_INFO).read_text(encoding="utf-8"))["opt_step"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _replace_dir(src: Path, dst: Path) -> None:
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(src, dst)


def stage_best(run_dir: str | Path, committed_opt_step: int | None) -> None:
    """Before exporting a new best: park best/ as best.prev/ if the newest checkpoint committed it."""
    best = Path(run_dir) / BEST_DIR
    step = best_step(best)
    if step is not None and committed_opt_step is not None and step <= committed_opt_step:
        _replace_dir(best, Path(run_dir) / BEST_PREV)


def commit_best(run_dir: str | Path) -> None:
    """After a checkpoint: its state records the current best/, so the parked copy is not needed."""
    shutil.rmtree(Path(run_dir) / BEST_PREV, ignore_errors=True)


def _copy_dir(src: Path, dst: Path) -> None:
    """Replace dst with a copy of src; dst only ever appears complete (copied to dst.partial, then renamed)."""
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    _replace_dir(tmp, dst)


def mirror_best_dir(mirror_dir: str | Path, opt_step: int) -> Path:
    return Path(mirror_dir) / f"best-{opt_step:07d}"


def mirror_best(run_dir: str | Path, mirror_dir: str | Path, best_opt_step: int | None) -> None:
    """Before a checkpoint is mirrored: copy best/ (the export its state names) to <mirror>/best-<step>/.

    Skipped when that copy holds the same export (same marker bytes: its eval, incl. seconds). An orphan
    copy of the same step from a lost trajectory (mirrored, then killed before its checkpoint) is replaced.
    """
    if best_opt_step is None:
        return
    src, dst = Path(run_dir) / BEST_DIR, mirror_best_dir(mirror_dir, best_opt_step)
    if best_step(src) != best_opt_step:
        raise RuntimeError(f"{src} does not hold the best at opt {best_opt_step} the checkpoint records "
                           f"(it holds opt {best_step(src)})")
    with contextlib.suppress(OSError):
        if (dst / BEST_INFO).read_bytes() == (src / BEST_INFO).read_bytes():
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    _copy_dir(src, dst)


def prune_mirror_best(mirror_dir: str | Path, keep_opt_step: int | None) -> None:
    """After a checkpoint is mirrored: drop every best copy but the one it names (and half-done copies).
    Like the local best/, only the newest checkpoint's best is kept: a resume always takes the newest."""
    mirror_dir = Path(mirror_dir)
    if not mirror_dir.is_dir():
        return
    keep = None if keep_opt_step is None else mirror_best_dir(mirror_dir, keep_opt_step).name
    for p in mirror_dir.iterdir():
        if _MIRROR_BEST.match(p.name) and p.name != keep:
            shutil.rmtree(p, ignore_errors=True)


def reconcile_best(run_dir: str | Path, best_opt_step: int | None, mirror_dir: str | Path | None = None) -> str:
    """On (re)start: make best/ the export the restored state calls best (`best_opt_step`).

    kept: already so (or nothing on disk, or a best/ without our marker, which is left alone);
    restored: the parked best.prev/ put back; from_mirror: copied back from the --mirror-dir (the run dir
    was lost, e.g. a recycled VM); removed: a best exported after the checkpoint (the replay re-exports it
    if it recurs); missing: no copy of the state's best anywhere (the run dir was lost without a mirrored
    best, best/ was deleted or replaced by hand, or the checkpoint came from a run without --save-best).
    The caller must not train on after "missing": the run would end without its best.
    """
    run_dir = Path(run_dir)
    best, prev = run_dir / BEST_DIR, run_dir / BEST_PREV
    shutil.rmtree(run_dir / (BEST_DIR + ".partial"), ignore_errors=True)
    on_disk = best_step(best)
    mirrored = None if mirror_dir is None or best_opt_step is None else mirror_best_dir(mirror_dir, best_opt_step)
    if on_disk == best_opt_step:
        outcome = "kept"
    elif best_opt_step is not None and best_step(prev) == best_opt_step:
        _replace_dir(prev, best)
        outcome = "restored"
    elif mirrored is not None and best_step(mirrored) == best_opt_step:
        _copy_dir(mirrored, best)
        outcome = "from_mirror"
    elif best_opt_step is None:
        shutil.rmtree(best)
        outcome = "removed"
    else:
        outcome = "missing"
    shutil.rmtree(prev, ignore_errors=True)
    return outcome
