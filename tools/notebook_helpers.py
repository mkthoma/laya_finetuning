"""Helper functions that tools/build_notebook.py embeds verbatim in the notebook's Step 1.

They are copied with inspect.getsource, so the tests exercise exactly the code the kernel runs. Each one
must therefore be self-contained: it may use only json, os, shutil, subprocess, Path and the other helpers
(Step 1 imports those), and module-level constants here do not exist in the notebook.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def write_json(path, obj) -> None:
    """Write JSON atomically, so a kill mid-write never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def run_logged(cmd, log_path, expect_returncode=0, hint=None) -> int:
    """Run cmd with stdin closed (nothing can prompt); stream its output here and into log_path.

    VS Code truncates long cell output, so the log file is the full record. expect_returncode is an
    int, a tuple of ints, or None (accept any). Raises on an unexpected code; returns the code.
    """
    cmd, log_path = [str(c) for c in cmd], Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write("$ " + " ".join(cmd) + "\n")
        with subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1, env=os.environ.copy()) as proc:
            try:
                for line in proc.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
            except BaseException:  # interrupted cell: never leave an orphan holding the GPU
                proc.kill()
                raise
    rc = proc.returncode
    expected = (expect_returncode,) if isinstance(expect_returncode, int) else expect_returncode
    if expected is not None and rc not in expected:
        name = cmd[cmd.index("-m") + 1] if "-m" in cmd else Path(cmd[0]).name
        msg = f"{name} exited with {rc} (expected {' or '.join(map(str, expected))}); full log: {log_path}"
        raise RuntimeError(msg + (f"\nhint: {hint}" if hint else ""))
    return rc


def sh(cmd, check=True) -> str:
    """Run a short command with stdin closed; print and return its combined output."""
    res = subprocess.run(cmd, shell=isinstance(cmd, str), stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, errors="replace")
    out = (res.stdout + res.stderr).strip()
    if out:
        print(out)
    if check and res.returncode:
        raise RuntimeError(f"command failed with exit code {res.returncode}: {cmd}")
    return out


def already_done(path) -> bool:
    """True (and says so) when a step's output exists and REDO is False, so re-runs skip finished steps."""
    if Path(path).exists() and not globals().get("REDO", False):
        print(f"skip: {path} exists (set REDO = True in Step 1 to recompute)")
        return True
    return False


def fresh_dir(path) -> Path:
    """Empty a run directory so a re-run starts from scratch instead of resuming a stale checkpoint."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def show_json(path, keys=None, width=150):
    """Print one line per top-level key (only `keys` when any of them exist), without bulky per-row or
    per-class detail; return the whole object."""
    def brief(v):
        if isinstance(v, dict):
            return {k: brief(x) for k, x in v.items() if k not in ("per_row", "per_class", "cm", "confusion")}
        return v
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    shown = brief(obj) if isinstance(obj, dict) else {"value": obj}
    if keys and any(k in shown for k in keys):
        shown = {k: shown[k] for k in keys if k in shown}
    for key, val in shown.items():
        text = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        print(f"  {key}: {text if len(text) <= width else text[:width - 3] + '...'}")
    return obj


def show_markdown(path) -> None:
    """Render a markdown file in the notebook (plain text outside IPython, e.g. in the local dry run)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print(text)
        return
    display(Markdown(text))


def restore_hf_token() -> bool:
    """Put a previously saved HF token into os.environ for the CLIs. Never prints it."""
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from huggingface_hub import constants  # importing constants does not query the Colab vault
        token_file = Path(constants.HF_TOKEN_PATH)
    except ImportError:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
    token = token_file.read_text(encoding="utf-8").strip() if token_file.exists() else ""
    if token:
        os.environ["HF_TOKEN"] = token
    return bool(token)


def resume_blocker(runs, kill_at=None):
    """Why Step 10 must not run now, or None. A resume is only valid straight after the crash run: the
    resumed log must end at the injected crash (at micro-step kill_at, when given). A -9 without it is some
    other kill, e.g. the host OOM killer. A second resume (interrupted Step 10, or REDO) would restart from
    a later checkpoint and the report, which grades the last 'resumed' event, would silently FAIL."""
    runs = Path(runs)
    exit_file, log_file = runs / "crash_exit.json", runs / "resumed" / "log.jsonl"
    if not exit_file.exists():
        return "no crash_exit.json: run Step 9 (crash run) first"
    rc = json.loads(exit_file.read_text(encoding="utf-8")).get("returncode")
    if rc not in (-9, 137):
        return f"the crash run exited with {rc}, not -9/137 (hard kill)"
    events = []
    for line in log_file.read_text(encoding="utf-8").splitlines() if log_file.exists() else []:
        try:
            event = json.loads(line)
        except ValueError:  # a SIGKILL can leave half a line
            continue
        if isinstance(event, dict):
            events.append(event)
    names = [e.get("event") for e in events]
    if "crash_injected" not in names:
        return "resumed/log.jsonl has no crash_injected event"
    if names[-1] != "crash_injected":
        return "a resume already ran (or was interrupted) after the crash"
    step = events[-1].get("micro_step")
    if kill_at is not None and step != kill_at:
        return f"the crash was injected at micro-step {step}, not at the planned {kill_at}"
    return None


def log_tail(path, n=8) -> str:
    """The last n lines of a log file, for error messages ('' when it does not exist)."""
    path = Path(path)
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def gpu_pids() -> list[int]:
    """PIDs of processes holding GPU memory according to nvidia-smi; [] without an NVIDIA driver."""
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
                         capture_output=True, stdin=subprocess.DEVNULL).stdout
    return [int(x) for x in out.split() if x.isdigit()]


def pid_alive(pid) -> bool:
    """True while a process with this PID exists. Linux /proc only (the drill runs on Colab); False elsewhere,
    so nothing ever signals a PID just to probe it."""
    return Path(f"/proc/{int(pid)}").exists()


def proc_cmdline(pid) -> str:
    """The command line of a process in this PID namespace ('' when it is gone or not visible)."""
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def drill_orphans(gpu_pids_left, child_pid, marker="LAYA_POC_KILL_DRILL"):
    """(ours, others): ours are the kill drill's own GPU holders, recognised by the marker in their command
    line, and are safe to kill; never this kernel. nvidia-smi may report host-namespace PIDs, so any other
    PID is only reported: the same number can be an unrelated process in this container."""
    me = os.getpid()
    candidates = {int(p) for p in gpu_pids_left} | {int(child_pid)}
    ours = sorted(p for p in candidates if p != me and marker in proc_cmdline(p))
    others = sorted(p for p in {int(p) for p in gpu_pids_left} if p != me and p not in ours)
    return ours, others


HELPERS = (write_json, run_logged, sh, already_done, fresh_dir, show_json, show_markdown, restore_hf_token,
           resume_blocker, log_tail, gpu_pids, pid_alive, proc_cmdline, drill_orphans)
