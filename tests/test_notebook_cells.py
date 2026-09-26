"""Behaviour of the notebook's step cells, executed with the CLIs stubbed (tools/notebook_cells.py).

Each test renders a cell exactly as tools/build_notebook.py does and runs it in a namespace that holds the
Step 1 helpers (tools/notebook_helpers.py), as the Colab kernel would.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_notebook as bn  # noqa: E402
import notebook_helpers as nh  # noqa: E402

FAKE_TOKEN = "hf_" + "Zq7" * 10
NO_BUNDLE = bn.Bundle("", "0" * 64, [])


def _cells(cfg: dict, target: bn.Target | None = None) -> dict[str, bn.Cell]:
    return {c.key: c for c in bn.build_cells(cfg, target or bn.Target(), NO_BUNDLE)}


def _kernel_ns(runs: Path) -> dict:
    """A namespace holding the helpers exactly as Step 1 defines them in the kernel."""
    import getpass
    import signal
    ns = {"__name__": "__main__", "json": json, "os": os, "shutil": shutil, "subprocess": subprocess, "sys": sys,
          "time": time, "signal": signal, "getpass": getpass, "Path": Path}
    exec("\n\n".join(inspect.getsource(f) for f in nh.HELPERS), ns)
    work = runs.parent.parent
    ns.update(WORK=work, DATA=work / "data", RUNS=runs, LOGS=runs / "logs", SENTINEL=work / "sentinel.json",
              DRILL_MARK=work.parent / "drill_mark.json")
    return ns


# ---------------------------------------------------------------- Step 2: install

def _run_install(cfg: dict, monkeypatch, changed: dict) -> list:
    """Step 2 with pip stubbed; `changed` holds the versions pip leaves behind. Returns the pip commands."""
    import importlib.metadata as md
    before = {"torch": "2.8.0", "transformers": "5.1.0", "protobuf": "5.29.6", "numpy": "2.0.2", "duckdb": "1.3.2"}
    state: dict = {"after": False, "calls": []}

    def version(name):
        v = {**before, **(changed if state["after"] else {})}.get(name)
        if v is None:
            raise md.PackageNotFoundError(name)
        return v

    class Dist:
        def __init__(self, name):
            self.name = name

        def read_text(self, _):
            if self.name == "laya" and state["after"]:
                return json.dumps({"vcs_info": {"commit_id": cfg["laya"]["commit"]}})
            return json.dumps({"dir_info": {"editable": True}}) if self.name == "laya-poc" else None

    def run_logged(cmd, log_path, **_):
        state["calls"].append(cmd)
        state["after"] = True
        return 0

    monkeypatch.setattr(md, "version", version)
    monkeypatch.setattr(md, "distribution", Dist)
    exec(_cells(cfg)["install"].source, {"json": json, "sys": sys, "run_logged": run_logged, "LOGS": Path("logs"),
                                         "WORK": Path("work")})
    return state["calls"]


def test_install_cell_installs_plain_laya_without_the_onnx_extra(cfg, monkeypatch):
    calls = _run_install(cfg, monkeypatch, {})
    assert [a for c in calls for a in c if "git+" in a] == [
        f"laya @ git+{cfg['laya']['repo']}.git@{cfg['laya']['commit']}"]


@pytest.mark.parametrize("pkg", ["torch", "transformers", "protobuf", "numpy"])
def test_install_cell_refuses_a_pip_run_that_changed_a_colab_package(cfg, monkeypatch, pkg):
    with pytest.raises(RuntimeError, match=f"pip changed.*{pkg}"):
        _run_install(cfg, monkeypatch, {pkg: "99.0"})


# ---------------------------------------------------------------- Step 4: environment errors

def _run_env(cfg: dict, tmp_path: Path, rc: int, env: dict | None = None, stale: dict | None = None) -> str:
    """Step 4 with env_check stubbed: it exits `rc` and writes `env` (None: writes nothing). Returns stdout."""
    runs = tmp_path / "work" / "runs" / "smoke"
    ns = _kernel_ns(runs)
    if stale is not None:
        nh.write_json(runs / "env.json", stale)

    def run_logged(cmd, log_path, expect_returncode=0, hint=None):
        if env is not None:
            nh.write_json(runs / "env.json", env)
        if expect_returncode is not None and rc not in ((expect_returncode,) if isinstance(expect_returncode, int)
                                                        else expect_returncode):
            raise RuntimeError(f"env_check exited with {rc}" + (f"\nhint: {hint}" if hint else ""))
        return rc

    ns.update(run_logged=run_logged, sh=lambda *a, **k: "")
    exec(_cells(cfg)["env"].source, ns)
    return ""


def test_env_step_failure_shows_the_errors_from_env_json_not_a_fixed_hint(cfg, tmp_path):
    err = "torch has no kernels for this GPU (sm_75; arch list ['sm_80']); install a torch build that includes it"
    with pytest.raises(RuntimeError) as exc:
        _run_env(cfg, tmp_path, 1, env={"errors": [err], "warnings": [], "passed": False})
    msg = str(exc.value)
    assert err in msg and "Remove Server" not in msg and "runbook" in msg


def test_env_step_failure_without_env_json_points_to_the_log_not_a_stale_file(cfg, tmp_path):
    stale = {"errors": ["laya commit abc != pinned def"], "warnings": [], "passed": False}
    with pytest.raises(RuntimeError) as exc:
        _run_env(cfg, tmp_path, 1, env=None, stale=stale)
    msg = str(exc.value)
    assert "04_env.log" in msg and "abc" not in msg and "Remove Server" not in msg


def test_env_step_passes_through_on_success(cfg, tmp_path, capsys):
    _run_env(cfg, tmp_path, 0, env={"errors": [], "warnings": [], "passed": True, "gpu_name": "Tesla T4"})
    assert "Tesla T4" in capsys.readouterr().out


# ---------------------------------------------------------------- Step 3: token errors

class _HubHTTPError(Exception):
    def __init__(self, status: int):
        super().__init__(f"{status} Client Error for url https://huggingface.co/api/whoami-v2")
        self.response = types.SimpleNamespace(status_code=status)


@pytest.mark.parametrize("error, want", [
    (_HubHTTPError(401), "rejected the token"),
    (_HubHTTPError(403), "rejected the token"),
    (ConnectionError(f"Failed to resolve huggingface.co (token {FAKE_TOKEN})"), "could not reach the Hub"),
])
def test_token_cell_errors_say_how_to_enter_a_new_token(cfg, monkeypatch, error, want):
    def login(token):
        raise error

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.login = login
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    ns = {"os": os, "getpass": None, "restore_hf_token": lambda: True}
    with pytest.raises(RuntimeError) as exc:
        exec(_cells(cfg)["token"].source, ns)
    msg = str(exc.value)
    assert want in msg and "ASK_AGAIN = True" in msg
    assert FAKE_TOKEN not in msg and "HF_TOKEN" not in os.environ


def _token_cell(cfg: dict, monkeypatch, token: str, error: Exception | None = None) -> tuple[str, list]:
    calls: list = []

    def login(token):
        calls.append(token)
        if error is not None:
            raise error

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.login = login
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HF_TOKEN", token)
    with pytest.raises(RuntimeError) as exc:
        exec(_cells(cfg)["token"].source, {"os": os, "getpass": None, "restore_hf_token": lambda: True})
    return str(exc.value), calls


@pytest.mark.parametrize("bad", [
    "hf_FAKEsecretA\nSECRETpartB" + "x" * 12,   # a line break: httpx would echo the rest in its header error
    "hf_" + "SECRETa" * 3 + "\r" + "SECRETb" * 3,
    "hf_" + "SECRETa" * 4 + "\u200b",            # zero-width space from a copy/paste
    "hf_SECRETshort",                            # truncated paste
    "api_" + "SECRETa" * 5,
])
def test_token_cell_rejects_a_malformed_token_without_echoing_it(cfg, monkeypatch, bad):
    msg, calls = _token_cell(cfg, monkeypatch, bad)
    assert "does not look like a Hugging Face token" in msg and "ASK_AGAIN = True" in msg
    assert "SECRET" not in msg and calls == [] and "HF_TOKEN" not in os.environ


def test_token_cell_detail_carries_the_http_status_and_never_the_token(cfg, monkeypatch):
    error = _HubHTTPError(500)
    error.args = (f"500 Server Error; Authorization: Bearer {FAKE_TOKEN}",)
    msg, calls = _token_cell(cfg, monkeypatch, FAKE_TOKEN, error)
    assert calls == [FAKE_TOKEN] and "HTTP 500" in msg and "could not reach the Hub" in msg
    assert FAKE_TOKEN not in msg and FAKE_TOKEN[3:12] not in msg


def test_data_step_hint_points_to_ask_again(cfg):
    assert "ASK_AGAIN = True" in _cells(cfg)["data"].source


# ---------------------------------------------------------------- Step 10: one resume per crash

KILL_AT = 122  # config.yaml smoke.kill_at_micro_step


def _write_events(runs: Path, events: tuple[str, ...], kill_at: int = KILL_AT) -> None:
    log = runs / "resumed" / "log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = ['{"event": "mic'] + [json.dumps({"event": e, **({"micro_step": kill_at} if e == "crash_injected" else {})})
                                  for e in events]  # a half line (SIGKILL mid-write) is tolerated
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _drill_runs(tmp_path: Path, rc: int = -9, events: tuple[str, ...] = ("start", "micro", "crash_injected"),
                kill_at: int = KILL_AT) -> Path:
    runs = tmp_path / "work" / "runs" / "smoke"
    nh.write_json(runs / "crash_exit.json", {"returncode": rc})
    _write_events(runs, events, kill_at)
    return runs


def test_resume_blocker_accepts_a_log_that_ends_at_the_injected_crash(tmp_path):
    assert nh.resume_blocker(_drill_runs(tmp_path)) is None
    assert nh.resume_blocker(_drill_runs(tmp_path, rc=137)) is None


@pytest.mark.parametrize("rc, events, want", [
    (0, ("start", "micro", "end"), "exited with 0"),
    (-9, ("start", "micro"), "crash_injected"),
    (-9, ("start", "crash_injected", "start", "resumed", "micro"), "already"),
    (-9, ("start", "crash_injected", "start", "micro"), "already"),
])
def test_resume_blocker_refuses_an_incomplete_or_used_drill(tmp_path, rc, events, want):
    assert want in nh.resume_blocker(_drill_runs(tmp_path, rc=rc, events=events))


def test_resume_blocker_checks_the_planned_crash_step(tmp_path):
    assert nh.resume_blocker(_drill_runs(tmp_path), KILL_AT) is None
    assert "micro-step 60" in nh.resume_blocker(_drill_runs(tmp_path, kill_at=60), KILL_AT)


def _run_crash_cell(cfg: dict, tmp_path: Path, rc: int, events: tuple[str, ...], kill_at: int = KILL_AT) -> Path:
    """Step 9 with the training CLI stubbed: it exits `rc` after logging `events` to resumed/log.jsonl."""
    runs = tmp_path / "work" / "runs" / "smoke"
    ns = _kernel_ns(runs)

    def run_logged(cmd, log_path, expect_returncode=0, hint=None):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("micro 100 | loss_ce 1.9\nmicro 110 | loss_ce 1.8\n", encoding="utf-8")
        _write_events(runs, events, kill_at)
        return rc

    ns["run_logged"] = run_logged
    exec(_cells(cfg)["crash"].source, ns)
    return runs


def test_crash_cell_accepts_the_injected_crash(cfg, tmp_path, capsys):
    runs = _run_crash_cell(cfg, tmp_path, -9, ("start", "micro", "crash_injected"))
    assert json.loads((runs / "crash_exit.json").read_text(encoding="utf-8")) == {"returncode": -9}
    assert "as planned" in capsys.readouterr().out


@pytest.mark.parametrize("rc, events, kill_at", [
    (-9, ("start", "micro"), KILL_AT),                     # the host OOM killer, not the injected crash
    (137, ("start", "micro", "crash_injected"), 60),       # killed at the wrong micro-step
])
def test_crash_cell_refuses_a_kill_that_is_not_the_injected_crash(cfg, tmp_path, rc, events, kill_at):
    with pytest.raises(RuntimeError) as exc:
        _run_crash_cell(cfg, tmp_path, rc, events, kill_at)
    msg = str(exc.value)
    assert "dmesg" in msg and "loss_ce 1.8" in msg  # the OOM hint and the log tail


def test_crash_cell_refuses_a_run_that_was_not_killed(cfg, tmp_path):
    with pytest.raises(RuntimeError, match="exited with 0"):
        _run_crash_cell(cfg, tmp_path, 0, ("start", "micro", "done"))


def _fake_run_logged(calls: list, summary: Path):
    def run_logged(cmd, log_path, expect_returncode=0, hint=None):
        calls.append(cmd)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("resumed from ckpt/step0000004.pt (micro-step 8, opt-step 4)\n", encoding="utf-8")
        nh.write_json(summary, {"micro_steps": 24})
        return 0
    return run_logged


def test_resume_cell_runs_once_after_the_crash(cfg, tmp_path):
    runs, calls = _drill_runs(tmp_path), []
    ns = _kernel_ns(runs)
    ns["run_logged"] = _fake_run_logged(calls, runs / "resumed" / "summary.json")
    exec(_cells(cfg)["resume"].source, ns)
    assert len(calls) == 1


def test_resume_cell_refuses_a_second_resume(cfg, tmp_path):
    runs, calls = _drill_runs(tmp_path, events=("start", "crash_injected", "start", "resumed", "micro")), []
    ns = _kernel_ns(runs)
    ns["run_logged"] = _fake_run_logged(calls, runs / "resumed" / "summary.json")
    with pytest.raises(RuntimeError, match="re-run Step 9"):
        exec(_cells(cfg)["resume"].source, ns)
    assert calls == []


# ---------------------------------------------------------------- Step 13a: archive

def test_archive_carries_the_fsq_notice_but_no_fsq_rows_or_checkpoints(cfg, tmp_path):
    runs = tmp_path / "work" / "runs" / "smoke"
    ns = {**_kernel_ns(runs), "zipfile": zipfile}
    ns["DATA"].mkdir(parents=True)
    for name in ("NOTICE_FSQ.txt", "data_report.json", "SHA256SUMS", "val.jsonl", "split_val.parquet"):
        (ns["DATA"] / name).write_text("x", encoding="utf-8")
    nh.write_json(runs / "control" / "summary.json", {})
    nh.write_json(runs / "control" / "ckpt" / "meta.json", {})
    exec(_cells(cfg)["archive"].source, ns)
    with zipfile.ZipFile(ns["WORK"] / "smoke_artifacts.zip") as zf:
        names = set(zf.namelist())
    assert {"data/NOTICE_FSQ.txt", "data/data_report.json", "runs/smoke/control/summary.json"} <= names
    assert not any(n.startswith("data/") and n.endswith((".jsonl", ".parquet")) for n in names)
    assert not any("/ckpt/" in n for n in names)


# ---------------------------------------------------------------- Step 13c/13d: kernel-kill drill

class _FakeChild:
    pid = 4242

    def poll(self):
        return 1


def _run_kill_cell(cfg: dict, ns: dict) -> list:
    """Step 13c with KILL_DRILL on and os.kill/Popen stubbed; returns the kill calls."""
    kills: list = []
    ns["os"] = types.SimpleNamespace(**{**vars(os), "getpid": lambda: 1111,
                                        "kill": lambda pid, sig: kills.append((pid, sig))})
    ns["subprocess"] = types.SimpleNamespace(Popen=lambda *a, **k: _FakeChild(), DEVNULL=None)
    ns["signal"] = types.SimpleNamespace(SIGKILL=9)
    ns["gpu_pids"] = lambda: []
    exec(_cells(cfg)["kill"].source.replace("KILL_DRILL = False", "KILL_DRILL = True"), ns)
    ns.update(os=os, subprocess=subprocess)
    return kills


def test_kill_drill_records_the_sentinel_inside_and_outside_work(cfg, tmp_path):
    ns = _kernel_ns(tmp_path / "work" / "runs" / "smoke")
    nh.write_json(ns["SENTINEL"], {"created": "2026-09-25 10:00:00"})
    assert _run_kill_cell(cfg, ns) == [(1111, 9)]
    for path in (ns["RUNS"] / "kill_drill.json", ns["DRILL_MARK"]):
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["sentinel_created"] == "2026-09-25 10:00:00" and rec["child_pid"] == 4242


def _verify(cfg: dict, ns: dict, capsys) -> str:
    ns.update(gpu_pids=lambda: [], pid_alive=lambda pid: False)
    exec(_cells(cfg)["verify_kill"].source, ns)
    return capsys.readouterr().out


def test_verify_kill_passes_when_the_sentinel_survived(cfg, tmp_path, capsys):
    ns = _kernel_ns(tmp_path / "work" / "runs" / "smoke")
    nh.write_json(ns["SENTINEL"], {"created": "A"})
    _run_kill_cell(cfg, ns)
    assert "PASS /content survived" in _verify(cfg, ns, capsys)


def test_verify_kill_fails_when_the_sentinel_was_recreated(cfg, tmp_path, capsys):
    ns = _kernel_ns(tmp_path / "work" / "runs" / "smoke")
    nh.write_json(ns["SENTINEL"], {"created": "A"})
    _run_kill_cell(cfg, ns)
    nh.write_json(ns["SENTINEL"], {"created": "B"})  # Step 1 after a wipe writes a new one
    assert "FAIL /content survived" in _verify(cfg, ns, capsys)


def test_verify_kill_fails_when_the_drill_record_under_work_is_gone(cfg, tmp_path, capsys):
    ns = _kernel_ns(tmp_path / "work" / "runs" / "smoke")
    nh.write_json(ns["SENTINEL"], {"created": "A"})
    _run_kill_cell(cfg, ns)
    (ns["RUNS"] / "kill_drill.json").unlink()
    assert "FAIL /content survived" in _verify(cfg, ns, capsys)


def test_drill_orphans_kills_only_the_drill_holder_never_the_kernel(monkeypatch):
    me = os.getpid()
    cmdlines = {200: "python -c import time, torch; LAYA_POC_KILL_DRILL", 300: "python -c LAYA_POC_KILL_DRILL",
                100: "/usr/bin/some-other-gpu-user", me: "python -c LAYA_POC_KILL_DRILL"}
    monkeypatch.setattr(nh, "proc_cmdline", lambda pid: cmdlines.get(pid, ""))
    ours, others = nh.drill_orphans([100, 200, me], child_pid=300)
    assert ours == [200, 300] and others == [100]
