"""Resumable training checkpoints (design doc §7.6.5, fixed per critique §A).

Checkpoints are written only at optimiser-step boundaries (right after zero_grad), so no
gradient state is lost. No RNG state is saved: the trainer reseeds the global RNG every
micro-step from (seed, epoch, micro index), which makes a resumed run replay exactly.
Loading always maps to CPU; `map_location="cuda"` breaks RNG/optimizer restores (critique §A).
"""
from __future__ import annotations

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
        _atomic_copy(path, Path(mirror_dir))
        _prune(Path(mirror_dir), keep_last)
    return path


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
