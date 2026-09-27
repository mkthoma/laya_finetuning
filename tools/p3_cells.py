"""Code cells of the Phase 3 notebook (spec §0, §6): the Phase 1/E1 setup, install, token, env and data cells,
reused, plus the Phase 3 steps (variants, parity on this card, the matrix plan, B2, one cell per arm, the report,
the archive and the optional Drive copy).

Earlier templates are adapted by exact substitution (e1_cells._derive): if one of them changes, the build fails
here instead of silently producing a notebook with the E1 paths.
"""
from __future__ import annotations

import inspect
import json
from string import Template

from e1_cells import _data, _derive, _show_keys, install_source
from e1_helpers import E1_HELPERS
from notebook_cells import _ENV, _SETUP, _SHOW_KEYS, _TOKEN, Bundle
from notebook_commands import Target, guarded, render_cmd
from notebook_helpers import HELPERS
from p3_commands import B2, arm_ids, p3_commands, variant_dirs
from p3_helpers import P3_HELPERS

from laya_poc import bundle as B

_REDO = "REDO = False  # True: recompute steps whose outputs already exist\n"
_CARD = Template('''\
# The card profile every run trains with (config phase3.card; profiles in config train.micro_batch). On another GPU
# set its profile BEFORE the first training step: a run resumes only with the settings it started with.
CARD = $card
RESULTS = WORK / "results"  # results/runs.csv, phase3_report.md (+ .json), written by the matrix
''')
RUN_HINT = ("Fix the cause, then re-run this step: finished runs are skipped and an interrupted run resumes from its "
            "newest checkpoint (runbook: When something fails)")
PARITY_KEYS = ("max_dp", "argmax_agree", "padded_max_dp", "nan", "passed", "device_test", "dtype_test")


def setup_source(target: Target, bundle: Bundle) -> str:
    """Phase 1 Step 1 with RUNS = runs/p3, CARD and RESULTS, and the E1 + Phase 3 helpers after the Phase 1 ones."""
    helpers = "\n\n".join(inspect.getsource(f).rstrip() for f in (*HELPERS, *E1_HELPERS, *P3_HELPERS))
    src = _SETUP.substitute(work=json.dumps(target.work), sha=bundle.sha256, b64=bundle.b64, helpers=helpers,
                            unpack=B.UNPACK_SOURCE.rstrip())
    src = _derive(src, 'WORK / "runs" / "smoke"', 'WORK / "runs" / "p3"')
    return _derive(src, _REDO, _REDO + _CARD.substitute(card=json.dumps(target.card)))


def token_source() -> str:
    return _derive(_TOKEN, "smoke_test.local.ipynb (build_notebook.py --with-token)",
                   "phase3_matrix.local.ipynb (build_p3_notebook.py --with-token)")


def env_source(c: dict) -> str:
    """Phase 1 Step 4 expecting CARD (a warning, not a failure, on another GPU)."""
    return (_ENV.substitute(cmd=render_cmd(*c["env"], indent=6)) + _show_keys("env", "env_json", _SHOW_KEYS["env"])
            + "note = card_note(env_json, CARD)\nif note:\n    print(note)\n")


def variants_source(c: dict, cfg: dict) -> str:
    """One guarded build per variant directory. Its build_ok.json marker lives inside the directory, so a partial
    build (interrupted cell) has none and is deleted and rebuilt; the variants CLI never sees a half-written dir."""
    blocks = []
    for name in variant_dirs(cfg):
        marker = f'WORK / "{name}" / "build_ok.json"'
        done = f'write_json({marker}, {{"time": time.strftime("%Y-%m-%d %H:%M")}})'
        blocks.append(guarded(c, f"variant_{name}", marker, f"06_variant_{name}.log",
                              pre=(f'shutil.rmtree(WORK / "{name}", ignore_errors=True)  # a partial build, if any',),
                              post=(done,)))
    listing = ('for d in sorted(p for p in WORK.glob("data_*") if (p / "build_ok.json").exists()):\n'
               '    print(f"{d.name}: {len(list(d.glob(\'*.jsonl\')))} JSONL files")\n')
    return "\n".join(blocks) + "\n" + listing


_PARITY_CHECK = Template('''\
parity = show_json(RUNS / "parity_$model.json", $keys)
if not parity.get("passed"):
    msg = "parity FAILED for $model on this card (design doc §7.7: CPU fp32 vs GPU fp16, padded vs unpadded)"
    if PARITY_MUST_PASS:
        raise RuntimeError(msg + "; do not train on it: Remove Server and ask for another card (runbook), or set "
                           "PARITY_MUST_PASS = False in this cell to continue anyway")
    print("WARNING: " + msg)
''')


def parity_source(c: dict, model: str) -> str:
    head = "PARITY_MUST_PASS = True  # False: continue even when this card fails the parity check\n"
    body = guarded(c, f"parity_{model}", f'RUNS / "parity_{model}.json"', f"07_parity_{model}.log")
    return head + body + "\n" + _PARITY_CHECK.substitute(model=model, keys=json.dumps(PARITY_KEYS))


_ARM = Template('''\
ARM = "$arm"
busy = busy_pids()
if busy:  # e.g. the matrix from before a kernel restart: two must never train into the same run directories
    raise RuntimeError(f"a matrix or trainer process is still running (pid {busy}): wait for it to stop, or stop it "
                       "with os.kill(pid, 9) (checkpoints are written atomically), then re-run this step")
log = next_log(LOGS, "$log")  # one log per attempt: a re-run keeps the earlier ones
cmd = $cmd
rc = run_logged(cmd, log, expect_returncode=None)  # a failed run does not stop the arm's other runs
print(f"{ARM}: finished runs in results/runs.csv")
rows = show_runs(RESULTS / "runs.csv", ARM)
if rc:
    raise RuntimeError(f"{ARM}: matrix run exited with {rc}; the 'matrix:' lines in {log} name each failed run and "
                       "its stage log. " + $hint)
''')


def arm_source(c: dict, arm: str, step: str) -> str:
    return _ARM.substitute(arm=arm, cmd=render_cmd(*c[arm], indent=6), log=f"{step.zfill(2)}_{arm}",
                           hint=json.dumps(RUN_HINT))


def matrix_arms(cfg: dict) -> list[str]:
    """The matrix steps in notebook order: B2 (when phase3.zero_shot is set), then every trained arm."""
    return [*([B2] if cfg["phase3"].get("zero_shot") else []), *arm_ids(cfg)]


def step_numbers(cfg: dict) -> dict[str, str]:
    """Step label per cell key; the markdown titles and the log file names both use it."""
    arms = matrix_arms(cfg)
    fixed = {"setup": "1", "install": "2", "token": "3", "env": "4", "data": "5", "variants": "6",
             "parity_laya": "7a", "parity_laya_ml": "7b", "plan": "8"}
    report = 9 + len(arms)
    return {**fixed, **{arm: str(9 + i) for i, arm in enumerate(arms)}, "report": str(report),
            "archive": f"{report + 1}a", "drive": f"{report + 1}b"}


_REPORT = Template('''\
cmd = $cmd
rc = run_logged(cmd, LOGS / "$log", expect_returncode=None)
if (RESULTS / "phase3_report.md").exists():
    show_markdown(RESULTS / "phase3_report.md")
rows = show_runs(RESULTS / "runs.csv")
if rc:  # shown first: the report explains a failed exit check better than the exit code
    raise RuntimeError(f"matrix report exited with {rc}; full log: {LOGS / '$log'}")
''')


def _plan_report(c: dict, steps: dict[str, str]) -> dict[str, str]:
    return {"plan": f'cmd = {render_cmd(*c["plan"], indent=6)}\nrun_logged(cmd, LOGS / "08_plan.log")\n',
            "report": _REPORT.substitute(cmd=render_cmd(*c["report"], indent=6),
                                         log=f"{steps['report'].zfill(2)}_report.log")}


_OPTIONAL = {
    "archive": '''\
MAKE_ARCHIVE = True  # logs, JSON, per-row predictions (ids, probabilities), results/: no checkpoints, no FSQ rows
ARCHIVE = WORK / "phase3_artifacts.zip"
if MAKE_ARCHIVE:
    small, heavy = {".json", ".jsonl", ".md", ".log", ".yaml", ".csv"}, {"ckpt", "best", "best.partial", "best.prev",
                                                                         "final"}
    keep = [p for top in (RUNS, RESULTS) if top.exists() for p in sorted(top.rglob("*"))
            if p.is_file() and p.suffix in small and not heavy & set(p.relative_to(top).parts)]
    keep += [p for p in (DATA / "data_report.json", DATA / "SHA256SUMS", DATA / "NOTICE_FSQ.txt") if p.exists()]
    keep += sorted(WORK.glob("data_*/variant.json"))  # hashes and counts only
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in keep:
            zf.write(p, p.relative_to(WORK).as_posix())
    print(f"{ARCHIVE}: {len(keep)} files, {ARCHIVE.stat().st_size / 1e6:.1f} MB; Colab view > Contents > "
          "right-click > Download...")
''',
    "drive": '''\
COPY_TO_DRIVE = False       # the archive (previous step); interactive Google sign-in (about 2 min)
COPY_BEST_TO_DRIVE = False  # every run's best/ (about 0.9 GB each for laya, fp16, with its NOTICE files)
ARCHIVE = WORK / "phase3_artifacts.zip"
if COPY_TO_DRIVE or COPY_BEST_TO_DRIVE:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    dest = Path("/content/drive/MyDrive/laya_poc/phase3")
    dest.mkdir(parents=True, exist_ok=True)
    if COPY_TO_DRIVE and ARCHIVE.exists():
        print("copied to", shutil.copy2(ARCHIVE, dest / ARCHIVE.name))
    for best in sorted(RUNS.glob("*/train/best")) if COPY_BEST_TO_DRIVE else []:
        print("copied to", shutil.copytree(best, dest / "best" / best.parent.parent.name, dirs_exist_ok=True))
else:
    print("Drive copy skipped (COPY_TO_DRIVE and COPY_BEST_TO_DRIVE are False)")
''',
}


def code_sources(cfg: dict, target: Target, bundle: Bundle) -> dict[str, str]:
    """Every code cell's source, keyed by step (notebook order); arms are keyed by their id (B2, E2, ...)."""
    c, steps = p3_commands(cfg, target), step_numbers(cfg)
    out = {"setup": setup_source(target, bundle), "install": install_source(cfg), "token": token_source(),
           "env": env_source(c), "data": _data(c, cfg), "variants": variants_source(c, cfg),
           "parity_laya": parity_source(c, "laya"), "parity_laya_ml": parity_source(c, "laya_ml")}
    plan_report = _plan_report(c, steps)
    out["plan"] = plan_report["plan"]
    out.update({arm: arm_source(c, arm, steps[arm]) for arm in matrix_arms(cfg)})
    return {**out, "report": plan_report["report"], **_OPTIONAL}
