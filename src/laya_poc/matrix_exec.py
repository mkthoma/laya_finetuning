"""How one Phase 3 run is executed: the CLI commands of its steps, a subprocess runner that tees their output to
<run>/logs/<step>.log, cleanup of the training directory and the per-step timings.

Every step is `sys.executable -m laya_poc.<cli> ...` in its own process, as in the E1 notebook, so a CUDA error
or an OOM in one run cannot leave the next one with a poisoned GPU context. The console only gets sparse
progress (the notebook cell truncates long outputs): every line except the progress ticks (train_single's
per-micro-step lines, the `<label> k/N rows, Ns` batch progress of evaluate/export_check), of which one a minute
per kind is shown; the log file has everything. The HF token reaches the children through the
inherited environment only and is never part of a command line or a log.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .matrix_plan import DataPaths, RunSpec

PACKAGE = "laya_poc"
PROGRESS_TICKS = {"micro": re.compile(r"^micro \d"),              # train_single: micro k/total | loss ...
                  "rows": re.compile(r" \d+/\d+ rows, \d+s\s*$")}  # parity.batch_progress in the CLIs
TICK_PRINT_SECONDS = 60.0
TIMING_NAME = "timing.json"

Runner = Callable[[Sequence[str], Path], int]


@dataclass(frozen=True)
class RunContext:
    """What the commands of one run need besides its spec and data paths."""
    cfg: dict
    run_dir: Path
    config_path: Path             # <run>/run_config.yaml: the config with labels.scheme = the run's scheme
    init: str                     # 'hub' or an absolute Laya checkpoint dir (training init; B2's checkpoint)
    device: str
    card: str
    train_extra: tuple[str, ...]

    @property
    def best(self) -> Path:
        return self.run_dir / "train" / "best"


def module_cmd(module: str, args: Sequence[Any]) -> list[str]:
    return [sys.executable, "-m", f"{PACKAGE}.{module}", *(str(a) for a in args)]


# ---------------------------------------------------------------- step arguments

def train_args(ctx: RunContext, spec: RunSpec, paths: DataPaths, eval_every: int | None) -> list[str]:
    """train_single: resumes by itself from <run>/train/ckpt; --train-extra comes last so its values win."""
    args = ["--run-dir", ctx.run_dir / "train", "--data-dir", paths.train, "--config", ctx.config_path,
            "--model", spec.model, "--seed", spec.seed, "--init", ctx.init, "--device", ctx.device,
            "--card", ctx.card, "--epochs", ctx.cfg["train"]["epochs"], "--save-best", "--final-eval"]
    args += ["--freeze-encoder"] if spec.head_only else []
    args += ["--eval-every-opt-steps", eval_every] if eval_every else []
    return [str(a) for a in (*args, *ctx.train_extra)]


def export_args(ctx: RunContext, spec: RunSpec, paths: DataPaths) -> list[str]:
    """export_check fits T on val and writes it into best/; it cross-checks best/ against the eval that
    selected it (best/train_eval.json), not the final weights' eval in summary.json (as in the E1 notebook)."""
    return [str(a) for a in ("--ckpt", ctx.best, "--rows", paths.train / "val.jsonl", "--model", spec.model,
                             "--out", ctx.run_dir / "export_check.json", "--device", ctx.device,
                             "--config", ctx.config_path, "--train-summary", ctx.best / "train_eval.json")]


def evaluate_args(ctx: RunContext, spec: RunSpec, paths: DataPaths, splits: Sequence[str]) -> list[str]:
    """evaluate multi-split on every eval split (extra splits by path) plus order invariance."""
    oi = ctx.cfg["phase3"]["order_invariance"]
    ckpt = ctx.init if spec.zero_shot else ctx.best
    args: list[Any] = ["--ckpt", ckpt, "--model", spec.model, "--splits", *splits, "--data-dir", paths.eval]
    for name, path in paths.extras:
        args += ["--extra", f"{name}={path}"]
    args += ["--out-dir", ctx.run_dir / "eval", "--preds-dir", ctx.run_dir / "preds", "--device", ctx.device,
             "--config", ctx.config_path, "--order-invariance-out", ctx.run_dir / "order_invariance.json",
             "--order-split", oi["split"], "--order-n", oi["n"], "--order-perms", oi["perms"],
             "--order-seed", oi["seed"]]
    return [str(a) for a in args]


# ---------------------------------------------------------------- subprocess runner

class _SparsePrinter:
    """Echo child output to the console, thinning each kind of progress tick to one a minute."""

    def __init__(self, prefix: str = "    "):
        self.prefix = prefix
        self._last: dict[str, float] = {}  # tick kind -> when one was last shown

    def _thinned(self, line: str) -> bool:
        kind = next((k for k, pat in PROGRESS_TICKS.items() if pat.search(line)), None)
        if kind is None:
            return False
        now = time.monotonic()
        if now - self._last.get(kind, float("-inf")) < TICK_PRINT_SECONDS:
            return True
        self._last = {**self._last, kind: now}
        return False

    def __call__(self, line: str) -> None:
        if self._thinned(line):
            return
        text = self.prefix + line + ("" if line.endswith("\n") else "\n")
        try:
            sys.stdout.write(text)
        except UnicodeEncodeError:  # a non-UTF-8 console (Windows pipe): the log file keeps the exact text
            sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))
        sys.stdout.flush()


def tee_run(cmd: Sequence[str], log_path: Path) -> int:
    """Run cmd with stdin closed; every output line goes to log_path and, sparsely, to the console.
    An interrupted cell kills the child, so no orphan keeps the GPU. Returns the exit code."""
    cmd, log_path = [str(c) for c in cmd], Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    echo = _SparsePrinter()
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write("$ " + " ".join(cmd) + "\n")
        with subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1,
                              env=os.environ.copy()) as proc:
            try:
                for line in proc.stdout:
                    log.write(line)
                    echo(line)
            except BaseException:
                proc.kill()
                raise
    return proc.returncode


def next_log(logs_dir: Path, stem: str) -> Path:
    """logs/<stem>.log, or <stem>_2.log, ... when earlier attempts exist (a resume keeps their logs)."""
    n = 1
    while (logs_dir / (f"{stem}.log" if n == 1 else f"{stem}_{n}.log")).exists():
        n += 1
    return logs_dir / (f"{stem}.log" if n == 1 else f"{stem}_{n}.log")


def running_trainers(train_dir: Path) -> list[int]:
    """PIDs of train_single processes already training into train_dir (Linux /proc; [] elsewhere). After a
    kernel restart an old trainer can outlive the kernel: a second one must never share its run dir."""
    proc, me = Path("/proc"), os.getpid()
    if not proc.is_dir():
        return []
    found = []
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            line = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if f"{PACKAGE}.train_single" in line and str(train_dir) in line:
            found.append(int(p.name))
    return sorted(found)


# ---------------------------------------------------------------- files

def write_json(path: Path, obj: Any) -> Path:
    """Atomic JSON write (a kill mid-write never leaves half a file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> Any:
    """Parsed JSON, or None when the file is absent or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def add_timing(run_dir: Path, step: str, seconds: float) -> dict:
    """Accumulate a step's wall-clock seconds in <run>/timing.json (attempts after a failure add up)."""
    old = read_json(run_dir / TIMING_NAME)
    old = old if isinstance(old, dict) else {}
    new = {**old, step: round(float(old.get(step, 0.0)) + seconds, 3)}
    write_json(run_dir / TIMING_NAME, new)
    return new


def cleanup_train(train_dir: Path, keep: Sequence[str]) -> list[str]:
    """Delete every directory under train/ not in `keep` (ckpt/, final/, best.prev/ ...); files (log.jsonl,
    summary.json, config.yaml, error.log) are small and stay. Returns the removed names."""
    if not train_dir.is_dir():
        return []
    removed = sorted(p.name for p in train_dir.iterdir() if p.is_dir() and p.name not in keep)
    for name in removed:
        shutil.rmtree(train_dir / name)
    return removed
