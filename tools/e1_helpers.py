"""Helper functions tools/build_e1_notebook.py embeds verbatim in the E1 notebook's Step 1, after the Phase 1
helpers (tools/notebook_helpers.py).

Same rules as there: each is copied with inspect.getsource, so it may use only json, os, shutil, subprocess,
time, Path and the other helpers (Step 1 imports and defines those); module constants here do not exist in the
notebook.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from notebook_helpers import proc_cmdline  # a Phase 1 helper: Step 1 defines it before these


def read_events(path) -> list:
    """The JSON events of a run log; unparsable lines (a SIGKILL can leave half a line) are skipped."""
    path, events = Path(path), []
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def crash_problem(exit_file, log_file, kill_at=None):
    """Why E1's forced crash has not happened as planned, or None when it has. Unlike the smoke drill, any
    number of resumes may follow it (after a Colab disconnect Step 9 simply resumes again), so the log only
    has to contain the injected crash, at micro-step kill_at when given."""
    exit_file = Path(exit_file)
    if not exit_file.exists():
        return "no crash_exit.json: the crash run (Step 8) has not finished"
    rc = json.loads(exit_file.read_text(encoding="utf-8")).get("returncode")
    if rc not in (-9, 137):
        return f"the crash run exited with {rc}, not -9/137 (hard kill)"
    crashes = [e for e in read_events(log_file) if e.get("event") == "crash_injected"]
    if not crashes:
        return f"{Path(log_file).name} has no crash_injected event"
    step = crashes[0].get("micro_step")
    if kill_at is not None and step != kill_at:
        return f"the crash was injected at micro-step {step}, not at the planned {kill_at}"
    return None


def next_log(logs, stem) -> Path:
    """logs/<stem>.log, or <stem>_2.log, <stem>_3.log, ... when earlier attempts exist: a resume after a
    disconnect keeps the logs of the attempts before it."""
    logs, n = Path(logs), 1
    while (logs / (f"{stem}.log" if n == 1 else f"{stem}_{n}.log")).exists():
        n += 1
    return logs / (f"{stem}.log" if n == 1 else f"{stem}_{n}.log")


def trainer_pids(module="laya_poc.train_single") -> list:
    """PIDs of running trainers other than this kernel (Linux /proc; [] elsewhere). A trainer started before a
    kernel restart can outlive the kernel for a while: a second one must never start on the same run dir."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    pids = [int(p.name) for p in proc.iterdir() if p.name.isdigit() and int(p.name) != os.getpid()]
    return sorted(pid for pid in pids if module in proc_cmdline(pid))


def fnum(value, nd=4) -> str:
    """A metric for a one-line summary: rounded when numeric, else as is (None when absent)."""
    return f"{value:.{nd}f}" if isinstance(value, float) else str(value)


def dig(obj, *keys):
    """obj[k1][k2]... or None when any level is missing (to print optional report fields)."""
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


E1_HELPERS = (read_events, crash_problem, next_log, trainer_pids, fnum, dig)
