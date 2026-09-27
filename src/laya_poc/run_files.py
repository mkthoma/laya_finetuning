"""The files a train_single run writes besides log.jsonl and its checkpoints, and the figures they report.

config.yaml (the config, CLI args and resolved settings the run started with), summary.json (written
atomically at the end: the gate and the Phase 3 matrix read it) and error.log (the traceback of a failed
run; the notebook shows only a one-line error). Peak VRAM is in decimal GB, the unit of the design doc's
"<= 14 GB" cap (smoke.exit.max_vram_gb).
"""
from __future__ import annotations

import contextlib
import json
import os
import statistics
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

SKIP_TIMING = 5  # first micro-steps of each process (warm-up) are excluded from the s/micro stats
GB = 10 ** 9     # decimal GB, not GiB


def vram_gb(device: str) -> dict:
    """Peak allocated/reserved CUDA memory so far (None on CPU)."""
    if device != "cuda":
        return {"vram_alloc_gb": None, "vram_reserved_gb": None}
    import torch

    return {"vram_alloc_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            "vram_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3)}


def micro_timing(timings: list[float]) -> dict:
    """Median and mean seconds per micro-step of this process, warm-up excluded (None when too short)."""
    timed = timings[SKIP_TIMING:]
    return {"sec_per_micro_median": statistics.median(timed) if timed else None,
            "sec_per_micro_mean": statistics.fmean(timed) if timed else None}


def write_snapshot(run_dir: Path, cfg: dict, args: Any, settings: Any) -> None:
    """<run_dir>/config.yaml: what this (re)start ran with (rewritten by every start, incl. resumes)."""
    snapshot = {"config": cfg, "args": vars(args), "settings": asdict(settings)}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")


def write_summary(run_dir: Path, summary: dict) -> Path:
    """<run_dir>/summary.json, replaced only once complete (readers never see a half-written file)."""
    out, tmp = run_dir / "summary.json", run_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return out


def record_error(run_dir: str, exc: BaseException) -> None:
    """Append the traceback to <run_dir>/error.log when the run dir exists (never raises)."""
    path = Path(run_dir)
    if path.is_absolute() and path.is_dir():
        with contextlib.suppress(OSError), (path / "error.log").open("a", encoding="utf-8") as fh:
            fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
