"""Code cells of the E1 notebook (design doc §6.2 Phase 2): the Phase 1 setup/install/token/env cells, reused,
plus the E1 steps (data, baselines, zero-shot, crash, resume, temperature fit, evaluation, CPU bench, gate).

The Phase 1 templates are adapted by exact substitution (`_derive`): if one of them changes, the build fails
here instead of silently producing a notebook with the smoke-test paths or the unpinned DuckDB.
"""
from __future__ import annotations

import inspect
import json
import textwrap
from string import Template

from e1_commands import e1_commands
from e1_helpers import E1_HELPERS
from notebook_cells import _ENV, _INSTALL, _SETUP, _SHOW_KEYS, _TOKEN, Bundle
from notebook_commands import Target, guarded, render_cmd
from notebook_helpers import HELPERS

from laya_poc import bundle as B
from laya_poc.smoke_report import expected_resume_step

OOM = ("CUDA OOM: add '--micro-batch', '4', '--effective-batch', '32' to cmd in Steps 8 and 9 alike (the resume "
       "refuses other settings), delete runs/e1/crash_exit.json so Step 8 starts E1 afresh, then run Steps 8 and 9 "
       "(runbook: When something fails)")
GATED = ("401/403: accept the terms of the gated dataset foursquare/fsq-os-places on the Hub; for a different "
         "token set ASK_AGAIN = True in Step 3 and re-run it; then re-run this step")
FROZEN_HINT = ("if the fingerprint differs from the frozen manifest, this build is not the frozen E1 data: compare "
               "the versions in data_report.json with the manifest and do not train on it (runbook)")
SHOW = {
    "data": ("split_sizes", "headline_rule", "traps", "fingerprint_sha256", "frozen_check", "warnings"),
    "eval": ("model", "n", "n_unlabelled", "n_abstained_no_evidence", "temperature", "pre", "post", "cpu_fallback",
             "seconds"),
    "train": ("micro_steps", "opt_steps", "stop_reason", "resumed_from", "evals", "best_opt_step", "best_eval",
              "peak_vram_reserved_gb", "sec_per_micro_mean", "nonfinite", "opt_steps_skipped",
              "nonfinite_grad_applied", "min_scale"),
    "export": ("T", "clamped", "roundtrip_max_dp", "passed", "cpu_fallback", "train_eval"),
}


def _derive(src: str, old: str, new: str) -> str:
    """src with its single occurrence of old replaced by new."""
    if src.count(old) != 1:
        raise ValueError(f"a Phase 1 cell template changed: expected exactly one {old[:60]!r}")
    return src.replace(old, new)


def _show_keys(var: str, path: str, keys: tuple[str, ...]) -> str:
    return f"{var} = show_json({path}, {json.dumps(keys)})\n"


def _show(var: str, path: str, keys: str) -> str:
    return _show_keys(var, path, SHOW[keys])


def setup_source(target: Target, bundle: Bundle) -> str:
    """Phase 1 Step 1 with RUNS = runs/e1 and the E1 helpers defined after the Phase 1 ones."""
    helpers = "\n\n".join(inspect.getsource(f).rstrip() for f in (*HELPERS, *E1_HELPERS))
    src = _SETUP.substitute(work=json.dumps(target.work), sha=bundle.sha256, b64=bundle.b64, helpers=helpers,
                            unpack=B.UNPACK_SOURCE.rstrip())
    return _derive(src, 'WORK / "runs" / "smoke"', 'WORK / "runs" / "e1"')


_DUCK_OLD = '''\
duck = dist_version("duckdb")
if not duck or not (1, 1) <= tuple(int(x) for x in duck.split(".")[:2]) < (2, 0):
    run_logged(PIP + ["duckdb>=1.1,<2"], LOGS / "02_install_duckdb.log")
'''
_DUCK_NEW = Template('''\
DUCKDB = "$version"  # config data.duckdb_version: hash()-based sampling must match the frozen build exactly
if dist_version("duckdb") != DUCKDB:
    run_logged(PIP + [f"duckdb=={DUCKDB}"], LOGS / "02_install_duckdb.log")
''')
_LAYA_CHECK = '    raise RuntimeError(f"installed laya commit {got} != pinned {LAYA_COMMIT}")\n'
_DUCK_CHECK = ('if dist_version("duckdb") != DUCKDB:\n    raise RuntimeError(f"duckdb {dist_version(\'duckdb\')} != '
               'pinned {DUCKDB}; see {LOGS / \'02_install_duckdb.log\'}")\n')


def install_source(cfg: dict) -> str:
    """Phase 1 Step 2 with DuckDB pinned to data.duckdb_version (still never touching torch & co)."""
    src = _INSTALL.substitute(commit=cfg["laya"]["commit"], repo=cfg["laya"]["repo"])
    src = _derive(src, _DUCK_OLD, _DUCK_NEW.substitute(version=cfg["data"]["duckdb_version"]))
    return _derive(src, _LAYA_CHECK, _LAYA_CHECK + _DUCK_CHECK)


def token_source() -> str:
    return _derive(_TOKEN, "smoke_test.local.ipynb (build_notebook.py --with-token)",
                   "e1_first_run.local.ipynb (build_e1_notebook.py --with-token)")


def _data(c: dict, cfg: dict) -> str:
    frozen = cfg["data"].get("frozen_manifest")
    head = (f"FROZEN = {frozen!r}  # config data.frozen_manifest: --verify-frozen compares the build with it\n"
            + ("" if frozen else 'print("WARNING: data.frozen_manifest is null: this build is not checked against a '
                                 'frozen fingerprint")\n'))
    marker = 'DATA / "build_ok.json"'  # written only after a successful (verified) build
    done = f'write_json({marker}, {{"frozen_manifest": FROZEN, "time": time.strftime("%Y-%m-%d %H:%M")}})'
    body = guarded(c, "data", marker, "05_data.log", hint=f"{GATED}; {FROZEN_HINT}",
                   pre=(f"({marker}).unlink(missing_ok=True)",), post=(done,))
    return head + body + "\n" + _show("data_report", 'DATA / "data_report.json"', "data")


def _baselines(c: dict) -> str:
    return guarded(c, "baselines", 'RUNS / "baselines.json"', "06_baselines.log") + '''
baselines = json.loads((RUNS / "baselines.json").read_text(encoding="utf-8"))
for scheme in ("c10", "c7"):
    arms = dig(baselines, "schemes", scheme) or {}
    print(f"{scheme} val macro-F1: B1 majority {fnum(dig(arms, 'b1_majority', 'val', 'macro_f1'))}, B3 TF-IDF+LR "
          f"{fnum(dig(arms, 'b3_tfidf_lr', 'val', 'post', 'macro_f1'))} (C {dig(arms, 'b3_tfidf_lr', 'C')}, "
          f"T {fnum(dig(arms, 'b3_tfidf_lr', 'T'), 3)})")
'''


def _zeroshot(c: dict) -> str:
    ml = textwrap.indent(guarded(c, "zeroshot_ml", 'RUNS / "eval_zeroshot_ml_val.json"', "07_zeroshot_ml.log"), "    ")
    return ("ZERO_SHOT_ML = False  # True: also zero-shot laya-multilingual (B2; ~1.2 GB download); the gate shows it\n"
            + guarded(c, "zeroshot", 'RUNS / "eval_zeroshot_val.json"', "07_zeroshot.log") + "\nif ZERO_SHOT_ML:\n"
            + ml + '\nfor name in ("eval_zeroshot_val.json", "eval_zeroshot_ml_val.json"):\n'
            '    if (RUNS / name).exists():\n        print(name)\n        ' + _show("zs", "RUNS / name", "eval"))


_CRASH = Template('''\
KILL_AT = $kill  # config e1.crash_at_micro_step: the resume restarts from the micro-step-$resume checkpoint
TRAIN, OOM_HINT = RUNS / "train", $oom
if already_done(TRAIN / "summary.json"):
    pass
elif not REDO and crash_problem(RUNS / "crash_exit.json", TRAIN / "log.jsonl", KILL_AT) is None:
    print(f"skip: E1 was already hard-killed at micro-step {KILL_AT} as planned; run Step 9 (it resumes E1)")
elif trainer_pids():
    raise RuntimeError(f"a trainer is still running (pid {trainer_pids()}, from before a kernel restart?): wait "
                       "for it to stop, then re-run this step")
else:
    fresh_dir(TRAIN)  # the crash run always starts E1 from scratch
    (RUNS / "crash_exit.json").unlink(missing_ok=True)
    cmd = $cmd
    rc = run_logged(cmd, LOGS / "08_e1_crash.log", expect_returncode=None)
    write_json(RUNS / "crash_exit.json", {"returncode": rc})
    tail = "\\n" + log_tail(LOGS / "08_e1_crash.log")
    if rc not in (-9, 137):
        raise RuntimeError(f"the E1 crash run exited with {rc}, not -9/137 (hard kill); if CUDA OOM: {OOM_HINT}" + tail)
    problem = crash_problem(RUNS / "crash_exit.json", TRAIN / "log.jsonl", KILL_AT)
    if problem:  # e.g. the host OOM killer, not the injected crash
        raise RuntimeError(f"E1 was killed ({rc}) but not as planned: {problem}. A kill without crash_injected is "
                           "usually the host OOM killer: check RAM (free -h) and the kernel log (dmesg | tail), "
                           "then re-run this step." + tail)
    print("E1 was hard-killed as planned at micro-step", KILL_AT, "- exit code", rc)
''')

_RESUME = Template('''\
KILL_AT, TRAIN = $kill, RUNS / "train"
if not already_done(TRAIN / "summary.json"):
    problem = crash_problem(RUNS / "crash_exit.json", TRAIN / "log.jsonl", KILL_AT)
    if problem:
        raise RuntimeError(f"E1 cannot resume yet ({problem}): run Step 8 first")
    if trainer_pids():  # e.g. the run from before a kernel restart: two trainers must never share a run dir
        raise RuntimeError(f"an E1 trainer is still running (pid {trainer_pids()}): wait for it to stop, or stop "
                           "it with os.kill(pid, 9) (checkpoints are written atomically), then re-run this step")
    log = next_log(LOGS, "09_e1_resume")  # one log per attempt: a resume after a disconnect keeps the earlier ones
    cmd = $cmd
    run_logged(cmd, log, hint=$oom)
    if "resumed from" not in log.read_text(encoding="utf-8"):
        raise RuntimeError("no 'resumed from ...' line: E1 restarted from scratch instead of resuming")
''')


def _train(c: dict, cfg: dict) -> dict[str, str]:
    e1 = cfg["e1"]
    kill, resume = e1["crash_at_micro_step"], expected_resume_step(e1["crash_at_micro_step"],
                                                                    e1["ckpt_every_micro_steps"])
    return {"crash": _CRASH.substitute(kill=kill, resume=resume, cmd=render_cmd(*c["crash"], indent=10),
                                       oom=json.dumps(OOM)),
            "resume": _RESUME.substitute(kill=kill, cmd=render_cmd(*c["resume"], indent=10), oom=json.dumps(OOM))
            + _show("e1", 'TRAIN / "summary.json"', "train")}


def _bench(c: dict) -> str:
    # bench_cpu writes bench_cpu.json and then exits 1 when every thread setting failed: a separate marker,
    # written only after a successful run, keeps that failed JSON from making a re-run skip this step.
    marker = 'RUNS / "bench_ok.json"'
    done = f'write_json({marker}, {{"time": time.strftime("%Y-%m-%d %H:%M")}})'
    return (guarded(c, "bench", marker, "12_bench_cpu.log", pre=(f"({marker}).unlink(missing_ok=True)",),
                    post=(done,))
            + '\nbench = show_json(RUNS / "bench_cpu.json")\n')


def _after(c: dict) -> dict[str, str]:
    export_warn = ('if not export.get("passed"):\n    print("WARNING: export_check did not pass (see train_eval, '
                   'budgets, roundtrip_max_dp above); the gate will FAIL end-to-end")\n')
    return {
        "export": guarded(c, "export", 'RUNS / "export_check.json"', "10_export_check.log") + "\n"
        + _show("export", 'RUNS / "export_check.json"', "export") + export_warn,
        "evaluate": guarded(c, "evaluate", 'RUNS / "eval_e1_val.json"', "11_evaluate.log") + "\n"
        + _show("e1_val", 'RUNS / "eval_e1_val.json"', "eval"),
        "bench": _bench(c),
        "gate": f'cmd = {render_cmd(*c["gate"], indent=6)}\nrun_logged(cmd, LOGS / "13_gate.log")\n'
                'show_markdown(RUNS / "gate_report.md")\n',
    }


_OPTIONAL = {
    "archive": '''\
MAKE_ARCHIVE = True         # logs, JSON, markdown, the gate report and the FSQ NOTICE: no checkpoints, no FSQ rows
COPY_TO_DRIVE = False       # interactive Google sign-in (about 2 min); fails for some Workspace accounts
COPY_BEST_TO_DRIVE = False  # also copy the E1 best/ checkpoint (about 0.9 GB fp16, with its NOTICE files)
ARCHIVE = WORK / "e1_artifacts.zip"
if MAKE_ARCHIVE:
    small = {".json", ".jsonl", ".md", ".log", ".yaml"}
    keep = [p for p in sorted(RUNS.rglob("*")) if p.is_file() and p.suffix in small
            and not {"ckpt", "best", "best.prev", "final"} & set(p.relative_to(RUNS).parts)]
    keep += [p for p in (DATA / "data_report.json", DATA / "SHA256SUMS", DATA / "NOTICE_FSQ.txt") if p.exists()]
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in keep:
            zf.write(p, p.relative_to(WORK).as_posix())
    print(f"{ARCHIVE}: {len(keep)} files; Colab view > Contents > right-click > Download...")
if COPY_TO_DRIVE or COPY_BEST_TO_DRIVE:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    dest = Path("/content/drive/MyDrive/laya_poc")
    dest.mkdir(parents=True, exist_ok=True)
    if COPY_TO_DRIVE and ARCHIVE.exists():
        print("copied to", shutil.copy2(ARCHIVE, dest / ARCHIVE.name))
    if COPY_BEST_TO_DRIVE:
        print("copied to", shutil.copytree(RUNS / "train" / "best", dest / "e1_best", dirs_exist_ok=True))
''',
    "cleanup": '''\
CLEANUP_CKPTS = False  # True: delete the resumable checkpoints (about 5 GB each); best/ and final/ are kept
if CLEANUP_CKPTS and not (RUNS / "train" / "summary.json").exists():
    print("E1 has not finished: keeping its checkpoints (Step 9 resumes from them)")
elif CLEANUP_CKPTS:
    for ckpt_dir in sorted(RUNS.glob("*/ckpt")):
        size = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
        shutil.rmtree(ckpt_dir)
        print(f"removed {ckpt_dir} ({size / 1e9:.1f} GB)")
else:
    print("checkpoint cleanup skipped (CLEANUP_CKPTS = False)")
''',
}


def code_sources(cfg: dict, target: Target, bundle: Bundle) -> dict[str, str]:
    """Every code cell's source, keyed by step (notebook order)."""
    c = e1_commands(cfg, target)
    env = _ENV.substitute(cmd=render_cmd(*c["env"], indent=6)) + _show_keys("env", "env_json", _SHOW_KEYS["env"])
    return {"setup": setup_source(target, bundle), "install": install_source(cfg), "token": token_source(),
            "env": env, "data": _data(c, cfg), "baselines": _baselines(c), "zeroshot": _zeroshot(c),
            **_train(c, cfg), **_after(c), **_OPTIONAL}
