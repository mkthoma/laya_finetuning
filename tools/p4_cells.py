"""Code cells of the Phase 4 notebook (spec §0, §6): the Phase 1/E1/Phase 3 setup, install, token, env, data and
variants cells, reused, plus one cell per baseline group (B1, B3, B4, B5 behind RUN_B5), the report and the archive.

Earlier templates are adapted by exact substitution (e1_cells._derive): if one of them changes, the build fails
here instead of silently producing a notebook with the Phase 3 paths.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from string import Template

from e1_cells import _data, _derive, _show_keys, install_source
from e1_helpers import E1_HELPERS
from notebook_cells import _ENV, _SETUP, _SHOW_KEYS, _TOKEN, Bundle
from notebook_commands import Target, guarded, render_cmd
from notebook_helpers import HELPERS
from p3_cells import _REDO
from p3_helpers import P3_HELPERS
from p4_commands import REPORT_MD, SKIP_OPTIONAL, arm_ids, optional_arm, p4_commands, variant_dirs
from p4_helpers import P4_HELPERS

from laya_poc import bundle as B

_CARD = Template('''\
# The card this session runs on (config phase4.card), recorded with every run in results/runs.csv. The estimates
# assume a G4; B1 and B3 run on the CPU whatever the card.
CARD = $card
RESULTS = WORK / "results"  # results/runs.csv, $report (+ .json), written by the matrix
''')
RUN_HINT = ("Fix the cause, then re-run this step: finished runs are skipped and an unfinished run starts again "
            "(runbook: When something fails)")


def run_flag(arm: str) -> str:
    return f"RUN_{arm}"


def setup_source(target: Target, bundle: Bundle) -> str:
    """Phase 1 Step 1 with RUNS = runs/p4, CARD and RESULTS, and the E1 + Phase 3 + Phase 4 helpers."""
    helpers = "\n\n".join(inspect.getsource(f).rstrip() for f in (*HELPERS, *E1_HELPERS, *P3_HELPERS, *P4_HELPERS))
    src = _SETUP.substitute(work=json.dumps(target.work), sha=bundle.sha256, b64=bundle.b64, helpers=helpers,
                            unpack=B.UNPACK_SOURCE.rstrip())
    src = _derive(src, 'WORK / "runs" / "smoke"', 'WORK / "runs" / "p4"')
    return _derive(src, _REDO, _REDO + _CARD.substitute(card=json.dumps(target.card), report=REPORT_MD))


def token_source() -> str:
    return _derive(_TOKEN, "smoke_test.local.ipynb (build_notebook.py --with-token)",
                   "phase4_baselines.local.ipynb (build_p4_notebook.py --with-token)")


def env_source(c: dict) -> str:
    """Phase 1 Step 4 expecting CARD (a warning, not a failure, on another GPU)."""
    return (_ENV.substitute(cmd=render_cmd(*c["env"], indent=6)) + _show_keys("env", "env_json", _SHOW_KEYS["env"])
            + "note = p4_card_note(env_json, CARD)\nif note:\n    print(note)\n")


def variants_source(c: dict, dirs: list[str]) -> str:
    """The Phase 3 variants step for `dirs`: build_ok.json marks a finished directory, so a partial build (an
    interrupted cell) is deleted and rebuilt; the variants CLI never sees a half-written directory."""
    blocks = []
    for name in dirs:
        marker = f'WORK / "{name}" / "build_ok.json"'
        done = f'write_json({marker}, {{"time": time.strftime("%Y-%m-%d %H:%M")}})'
        blocks.append(guarded(c, f"variant_{name}", marker, f"06_variant_{name}.log",
                              pre=(f'shutil.rmtree(WORK / "{name}", ignore_errors=True)  # a partial build, if any',),
                              post=(done,)))
    listing = ('for d in sorted(p for p in WORK.glob("data_*") if (p / "build_ok.json").exists()):\n'
               '    print(f"{d.name}: {len(list(d.glob(\'*.jsonl\')))} JSONL files")\n')
    return "\n".join(blocks) + "\n" + listing


_OPTIONAL_FLAG = Template('''\
$flag = True  # False: skip this optional reference; the matrix records it as "optional, disabled"
''')
_SKIP = Template('''\
if not $flag:
    cmd.append("$skip")
''')
_ARM = Template('''\
ARM = "$arm"
busy = baseline_pids()
if busy:  # e.g. the matrix from before a kernel restart: two must never write into the same run directories
    raise RuntimeError(f"a matrix or baseline process is still running (pid {busy}): wait for it to stop, or stop "
                       "it with os.kill(pid, 9), then re-run this step")
log = next_log(LOGS, "$log")  # one log per attempt: a re-run keeps the earlier ones
cmd = $cmd
${skip}rc = run_logged(cmd, log, expect_returncode=None)  # a failed run does not stop the arm's other runs
print(f"{ARM}: finished runs in results/runs.csv")
rows = show_runs(RESULTS / "runs.csv", ARM)
if rc:
    raise RuntimeError(f"{ARM}: matrix run exited with {rc}; the 'matrix:' lines in {log} name each failed run and "
                       "its stage log. " + $hint)
''')


def arm_source(c: dict, arm: str, step: str, optional: bool) -> str:
    """`matrix run --only <arm>` with the busy guard and one log per attempt; an optional arm gets RUN_<arm>."""
    flag = run_flag(arm)
    body = _ARM.substitute(arm=arm, cmd=render_cmd(*c[arm], indent=6), log=f"{step.zfill(2)}_{arm}",
                           hint=json.dumps(RUN_HINT),
                           skip=_SKIP.substitute(flag=flag, skip=SKIP_OPTIONAL) if optional else "")
    return (_OPTIONAL_FLAG.substitute(flag=flag) if optional else "") + body


def step_numbers(cfg: dict) -> dict[str, str]:
    """Step label per cell key; the markdown titles and the log file names both use it."""
    arms = arm_ids(cfg)
    fixed = {"setup": "1", "install": "2", "token": "3", "env": "4", "data": "5", "variants": "6"}
    report = 7 + len(arms)
    return {**fixed, **{arm: str(7 + i) for i, arm in enumerate(arms)}, "report": str(report),
            "archive": str(report + 1)}


_REPORT = Template('''\
cmd = $cmd
rc = run_logged(cmd, LOGS / "$log", expect_returncode=None)
if (RESULTS / "$report").exists():
    show_markdown(RESULTS / "$report")
rows = show_runs(RESULTS / "runs.csv")
print(phase4_verdict(RESULTS / "$report_json"))
if rc:  # shown first: the report explains a failed exit check better than the exit code
    raise RuntimeError(f"matrix report exited with {rc}; full log: {LOGS / '$log'}")
''')

ARCHIVE = '''\
MAKE_ARCHIVE = True  # logs, JSON, per-row predictions (ids, probabilities), results/: no weights, no FSQ rows
ARCHIVE = WORK / "phase4_artifacts.zip"
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
'''


def report_source(c: dict, step: str) -> str:
    return _REPORT.substitute(cmd=render_cmd(*c["report"], indent=6), log=f"{step.zfill(2)}_report.log",
                              report=REPORT_MD, report_json=Path(REPORT_MD).with_suffix(".json").name)


def code_sources(cfg: dict, target: Target, bundle: Bundle) -> dict[str, str]:
    """Every code cell's source, keyed by step (notebook order); arm groups are keyed by their id (B1, ...)."""
    c, steps = p4_commands(cfg, target), step_numbers(cfg)
    out = {"setup": setup_source(target, bundle), "install": install_source(cfg), "token": token_source(),
           "env": env_source(c), "data": _data(c, cfg), "variants": variants_source(c, variant_dirs(cfg))}
    out.update({arm: arm_source(c, arm, steps[arm], optional_arm(cfg, arm)) for arm in arm_ids(cfg)})
    return {**out, "report": report_source(c, steps["report"]), "archive": ARCHIVE}
