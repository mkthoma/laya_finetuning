"""Cell sources of the smoke notebook: setup/install/token templates, the step cells and the markdown.

`code_sources` renders every code cell from config.yaml, a `Target` and the code bundle; `steps` and
`intro_md` hold the markdown. tools/build_notebook.py assembles them into the notebook.
"""
from __future__ import annotations

import inspect
import json
from string import Template
from typing import NamedTuple

from notebook_commands import Target, commands, guarded, render_cmd
from notebook_helpers import HELPERS

from laya_poc import bundle as B
from laya_poc.smoke_report import expected_resume_step


class Bundle(NamedTuple):
    b64: str
    sha256: str
    members: list[str]


_SETUP = Template('''\
import getpass, json, os, shutil, signal, subprocess, sys, time, zipfile
from pathlib import Path

os.environ.update({
    "COLAB_DISABLE_STDIN_FOR_SHELL_MAGICS": "1",  # a command that prompts gets EOF instead of hanging
    "GIT_TERMINAL_PROMPT": "0", "PIP_NO_INPUT": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
    "LAYA_CUDA_AMP": "fp16",  # fp16 autocast on every card, also if the extension assigns an L4/A100
})
WORK = Path($work)
DATA, RUNS = WORK / "data", WORK / "runs" / "smoke"
LOGS = RUNS / "logs"
for _d in (DATA, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
os.environ["LAYA_POC_ROOT"] = str(WORK)  # the CLIs read WORK/config.yaml
REDO = False  # True: recompute steps whose outputs already exist
DRILL_MARK = Path.home() / ".laya_poc_kill_drill.json"  # outside WORK: Step 13d can tell a wiped /content


$helpers


SENTINEL = WORK / "sentinel.json"
if SENTINEL.exists():
    print("sentinel from", json.loads(SENTINEL.read_text(encoding="utf-8"))["created"], "found: /content survived")
else:
    write_json(SENTINEL, {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid()})
    print("sentinel created: first Step 1 on this server (if Step 1 ran here before, /content was wiped)")
print("HF token: restored from the token file" if restore_hf_token() else "HF token: not set yet (Step 3)")

$unpack
BUNDLE_SHA256 = "$sha"
BUNDLE_B64 = "$b64"
print(f"unpacked {len(_unpack(BUNDLE_B64, BUNDLE_SHA256, WORK))} files into {WORK}; bundle sha256 {BUNDLE_SHA256[:16]}")
''')

_INSTALL = Template('''\
import importlib, importlib.metadata as md

LAYA_COMMIT = "$commit"
PIP = [sys.executable, "-m", "pip", "install", "-q", "--no-input", "--progress-bar", "off"]


def dist_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def direct_url(name):  # PEP 610 metadata: what pip actually installed, and from where
    try:
        return json.loads(md.distribution(name).read_text("direct_url.json") or "{}")
    except md.PackageNotFoundError:
        return {}


before = {p: dist_version(p) for p in ("torch", "transformers", "protobuf", "numpy")}  # Colab's own builds
if direct_url("laya").get("vcs_info", {}).get("commit_id") != LAYA_COMMIT:  # no [onnx] extra: it pulls protobuf 6+
    run_logged(PIP + [f"laya @ git+$repo.git@{LAYA_COMMIT}"], LOGS / "02_install_laya.log")
duck = dist_version("duckdb")
if not duck or not (1, 1) <= tuple(int(x) for x in duck.split(".")[:2]) < (2, 0):
    run_logged(PIP + ["duckdb>=1.1,<2"], LOGS / "02_install_duckdb.log")
if not direct_url("laya-poc").get("dir_info", {}).get("editable"):
    run_logged(PIP + ["--no-deps", "-e", str(WORK)], LOGS / "02_install_project.log")
importlib.invalidate_caches()
changed = {p: f"{before[p]} -> {dist_version(p)}" for p in before if dist_version(p) != before[p]}
if changed:
    raise RuntimeError(f"pip changed {changed}; run Colab: Remove Server and start again")
got = direct_url("laya").get("vcs_info", {}).get("commit_id")
if got != LAYA_COMMIT:
    raise RuntimeError(f"installed laya commit {got} != pinned {LAYA_COMMIT}")
print({p: dist_version(p) for p in ("torch", "transformers", "huggingface_hub", "duckdb", "laya", "laya-poc")},
      "laya commit", got[:12])
''')

_TOKEN = '''\
import re
from huggingface_hub import login

PRESET_TOKEN = ""  # filled only in the gitignored smoke_test.local.ipynb (build_notebook.py --with-token)
ASK_AGAIN = False  # True: ignore the saved token (e.g. one without access to the gated dataset) and ask for a new one
NEW_TOKEN = "set ASK_AGAIN = True at the top of this cell and re-run it to enter a new token"
restore_hf_token()
token = os.environ.pop("HF_TOKEN", "").strip()
if PRESET_TOKEN and not ASK_AGAIN:  # unattended run: no prompt
    token = PRESET_TOKEN.strip()
del PRESET_TOKEN
if ASK_AGAIN or not token:  # VS Code shows a masked input box at the top of the window
    token = getpass.getpass("Hugging Face token (read access to foursquare/fsq-os-places; hidden): ").strip()
if not token:
    raise RuntimeError("no token entered (Esc/cancel sends an empty string); re-run this cell and paste it")
if not re.fullmatch(r"hf_[A-Za-z0-9]{20,}", token):  # never echo it: a line break would leak the rest in errors
    del token
    raise RuntimeError("that does not look like a Hugging Face token (hidden characters or line breaks?): "
                       + NEW_TOKEN)
try:
    login(token=token)  # validates with whoami and saves the token file, so a restart does not re-prompt
except Exception as exc:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        raise RuntimeError(f"the Hub rejected the token (HTTP {status}): {NEW_TOKEN}") from None
    text = str(exc).replace(token, "***")[:200]  # the exact token first, then anything token-like
    detail = re.sub(r"hf_\\w+", "hf_***", type(exc).__name__ + (f", HTTP {status}" if status else "") + f": {text}")
    raise RuntimeError(f"could not reach the Hub to check the token ({detail}); check the network and re-run this "
                       f"cell, or {NEW_TOKEN}") from None
os.environ["HF_TOKEN"] = token  # inherited by the CLIs (DuckDB reads the gated dataset with it)
del token
print("HF token OK (hidden)")
'''

_SHOW_KEYS = {  # what each step echoes from its JSON output (show_json falls back to all keys if none exist)
    "env": ("gpu_name", "capability", "has_sm75", "driver", "vram_total_gb", "disk_free_gb", "torch", "transformers",
            "laya_commit", "card", "warnings"),
    "data": ("split_sizes", "budgets", "warnings"),
    "parity": ("max_dp", "argmax_agree", "padded_max_dp", "nan", "passed", "device_test", "dtype_test"),
    "train": ("micro_steps", "opt_steps", "resumed_from", "peak_vram_reserved_gb", "sec_per_micro_median", "nonfinite",
              "min_scale", "initial_eval", "final_eval"),
    "export": ("T", "clamped", "roundtrip_max_dp", "passed"),
}


def _show(var: str, path: str, keys: str | None = None) -> str:
    return f"{var} = show_json({path}" + (f", {json.dumps(_SHOW_KEYS[keys])})\n" if keys else ")\n")


_ENV = Template('''\
if shutil.which("nvidia-smi"):
    sh(["nvidia-smi"], check=False)
env_json = RUNS / "env.json"
env_json.unlink(missing_ok=True)  # a stale file must not explain this run's failure
cmd = $cmd
if run_logged(cmd, LOGS / "04_env.log", expect_returncode=None):  # each error in env.json names its fix
    errors = json.loads(env_json.read_text(encoding="utf-8")).get("errors") if env_json.exists() else None
    detail = "; ".join(errors) if errors else f"no env.json written; see {LOGS / '04_env.log'}"
    raise RuntimeError(f"environment check failed: {detail} (runbook: When something fails)")
''')


def _eval_code(c: dict) -> dict[str, str]:
    gated =("401/403: accept the terms of the gated dataset foursquare/fsq-os-places on the Hub; for a different "
             "token set ASK_AGAIN = True in Step 3 and re-run it; then re-run this step")
    parity = [guarded(c, f"parity_{m}", f'RUNS / "parity_{m}.json"', f"07_parity_{m}.log")
              for m in ("laya", "laya_ml")]
    return {
        "env": _ENV.substitute(cmd=render_cmd(*c["env"], indent=6)) + _show("env", "env_json", "env"),
        "data": guarded(c, "data", 'DATA / "SHA256SUMS"', "05_data.log", hint=gated) + "\n"
        + _show("data_report", 'DATA / "data_report.json"', "data"),
        "zeroshot": guarded(c, "zeroshot", 'RUNS / "zeroshot.json"', "06_zeroshot.log") + "\n"
        + _show("zeroshot", 'RUNS / "zeroshot.json"'),
        "parity": "\n".join(parity) + '\nfor m in ("laya", "laya_ml"):\n    print("parity", m)\n    '
        + _show("parity", 'RUNS / f"parity_{m}.json"', "parity"),
    }


_CRASH_POST = Template('''\
write_json(RUNS / "crash_exit.json", {"returncode": rc})
if rc not in (-9, 137):
    raise RuntimeError(f"crash run exited with {rc}, expected -9 or 137 (hard kill); see {LOGS / '09_crash.log'}")
problem = resume_blocker(RUNS, $kill)  # the kill must be the injected crash, not e.g. the OOM killer
if problem:
    raise RuntimeError(f"the crash run was killed ({rc}) but not as planned: {problem}. A kill without "
                       "crash_injected is usually the host OOM killer: check RAM (free -h) and the kernel log "
                       f"(dmesg | tail), then re-run this step. Last lines of {LOGS / '09_crash.log'}:\\n"
                       + log_tail(LOGS / "09_crash.log"))
print("crash run was hard-killed as planned at micro-step $kill; exit code", rc)''')


def _train_code(c: dict, kill_at: int) -> dict[str, str]:
    done = 'RUNS / "resumed" / "summary.json"'
    oom = ("CUDA OOM: add '--micro-batch', '4', '--effective-batch', '32' to cmd here and in Steps 9-10 "
           "(runbook: When something fails)")
    crash_post = _CRASH_POST.substitute(kill=kill_at).splitlines()
    resume_pre = (f"problem = resume_blocker(RUNS, {kill_at})  # one resume per crash, else the report grades another",
                  "if problem:",
                  '    raise RuntimeError(f"the crash/resume drill is incomplete ({problem}): re-run Step 9 (with "',
                  '                       "REDO = True in Step 1 if Step 10 had finished), then Step 10")')
    resume_post = ('if "resumed from" not in (LOGS / "10_resume.log").read_text(encoding="utf-8"):',
                   '    raise RuntimeError("no \'resumed from ...\' line: the run restarted from scratch")')
    return {
        "control": guarded(c, "control", 'RUNS / "control" / "summary.json"', "08_control.log", hint=oom,
                           pre=('fresh_dir(RUNS / "control")',)) + "\n"
        + _show("control", 'RUNS / "control" / "summary.json"', "train"),
        "crash": guarded(c, "crash", done, "09_crash.log", lhs="rc = ", extra=", expect_returncode=None",
                         pre=('fresh_dir(RUNS / "resumed")', '(RUNS / "crash_exit.json").unlink(missing_ok=True)'),
                         post=crash_post) + "\n",
        "resume": guarded(c, "resume", done, "10_resume.log", pre=resume_pre, post=resume_post) + "\n"
        + _show("resumed", done, "train"),
        "export": guarded(c, "export", 'RUNS / "export_check.json"', "11_export_check.log") + "\n"
        + _show("export", 'RUNS / "export_check.json"', "export"),
        "report": f'cmd = {render_cmd(*c["report"], indent=6)}\nrun_logged(cmd, LOGS / "12_report.log")\n'
                  'show_markdown(RUNS / "smoke_report.md")\n',
    }


_OPTIONAL = {
    "archive": '''\
MAKE_ARCHIVE = True    # logs, JSON, the report and the FSQ NOTICE only: no checkpoints, no FSQ rows
COPY_TO_DRIVE = False  # interactive Google sign-in (about 2 min); fails for some Workspace accounts
ARCHIVE = WORK / "smoke_artifacts.zip"
if MAKE_ARCHIVE:
    small = {".json", ".jsonl", ".md", ".log", ".yaml"}
    keep = [p for p in sorted(RUNS.rglob("*"))
            if p.is_file() and p.suffix in small and not {"ckpt", "final"} & set(p.relative_to(RUNS).parts)]
    keep += [p for p in (DATA / "data_report.json", DATA / "SHA256SUMS", DATA / "NOTICE_FSQ.txt") if p.exists()]
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in keep:
            zf.write(p, p.relative_to(WORK).as_posix())
    print(f"{ARCHIVE}: {len(keep)} files; Colab view > Contents > right-click > Download...")
if COPY_TO_DRIVE and ARCHIVE.exists():
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    dest = Path("/content/drive/MyDrive/laya_poc")
    dest.mkdir(parents=True, exist_ok=True)
    print("copied to", shutil.copy2(ARCHIVE, dest / ARCHIVE.name))
''',
    "cleanup": '''\
CLEANUP_CKPTS = False  # True: delete the resumable checkpoints (about 5 GB each); final/ is kept
if CLEANUP_CKPTS:
    for ckpt_dir in sorted(RUNS.glob("*/ckpt")):
        size = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
        shutil.rmtree(ckpt_dir)
        print(f"removed {ckpt_dir} ({size / 1e9:.1f} GB)")
else:
    print("checkpoint cleanup skipped (CLEANUP_CKPTS = False)")
''',
    "kill": '''\
KILL_DRILL = False  # True: hard-kill this kernel (VS Code reports a crash - expected), then re-run Step 1 and 13d
if KILL_DRILL:
    # keeps a CUDA context; the marker lets Step 13d recognise (and only ever kill) this process
    holder = "import time, torch; LAYA_POC_KILL_DRILL = 1; torch.ones(1, device='cuda'); time.sleep(900)"
    child = subprocess.Popen([sys.executable, "-c", holder],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):  # wait until the child holds a CUDA context (the kernel itself never touches the GPU)
        if gpu_pids() or child.poll() is not None:
            break
        time.sleep(1)
    created = json.loads(SENTINEL.read_text(encoding="utf-8"))["created"] if SENTINEL.exists() else None
    record = {"kernel_pid": os.getpid(), "child_pid": child.pid, "gpu_pids": gpu_pids(), "sentinel_created": created,
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for path in (RUNS / "kill_drill.json", DRILL_MARK):
        write_json(path, record)
    os.kill(os.getpid(), signal.SIGKILL)
else:
    print("kernel-kill drill skipped (KILL_DRILL = False)")
''',
    "verify_kill": '''\
import importlib.util

drill = RUNS / "kill_drill.json"
record = drill if drill.exists() else DRILL_MARK if DRILL_MARK.exists() else None
if record is None:
    print("no kill drill recorded (run Step 13c with KILL_DRILL = True first)")
else:
    d, left = json.loads(record.read_text(encoding="utf-8")), gpu_pids()
    sentinel = json.loads(SENTINEL.read_text(encoding="utf-8")) if SENTINEL.exists() else {}
    ours, others = drill_orphans(left, d["child_pid"])
    checks = {"kernel restarted": d["kernel_pid"] != os.getpid(),
              "/content survived (drill record and the original sentinel)":
                  drill.exists() and sentinel.get("created") == d.get("sentinel_created"),
              "site-packages survived (laya importable)": importlib.util.find_spec("laya") is not None,
              "GPU child died with the kernel": not pid_alive(d["child_pid"]),
              "no process left on the GPU": not left}
    write_json(RUNS / "kill_drill_result.json", {**checks, "gpu_pids_after": left, "drill": d})
    for name, ok in checks.items():
        print("PASS" if ok else "FAIL", name)
    for pid in ours:
        try:
            os.kill(pid, signal.SIGKILL)
            print("killed the drill's orphan", pid)
        except OSError as exc:
            print("could not kill", pid, exc)
    if others:
        print("other GPU processes, not started by the drill (left alone):", others)
''',
}


def code_sources(cfg: dict, target: Target, bundle: Bundle) -> dict[str, str]:
    """Every code cell's source, keyed by step."""
    cmds = commands(cfg, target)
    setup = _SETUP.substitute(work=json.dumps(target.work), sha=bundle.sha256, b64=bundle.b64,
                              helpers="\n\n".join(inspect.getsource(f).rstrip() for f in HELPERS),
                              unpack=B.UNPACK_SOURCE.rstrip())
    return {"setup": setup, "install": _INSTALL.substitute(commit=cfg["laya"]["commit"], repo=cfg["laya"]["repo"]),
            "token": _TOKEN, **_eval_code(cmds), **_train_code(cmds, cfg["smoke"]["kill_at_micro_step"]), **_OPTIONAL}


def resume_step(cfg: dict) -> int:
    """The micro-step the resumed run restarts from; the report grades against the same function."""
    return expected_resume_step(cfg["smoke"]["kill_at_micro_step"], cfg["smoke"]["ckpt_every_micro_steps"])
