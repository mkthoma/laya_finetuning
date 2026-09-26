"""The generated E1 notebook (tools/build_e1_notebook.py), its cells run with stubbed CLIs, and the CPU dry run."""
from __future__ import annotations

import base64
import importlib
import importlib.util
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

from laya_poc import bundle as B  # noqa: E402
from laya_poc.config import load_config, validate_config, with_overrides  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_e1_notebook as be  # noqa: E402
import build_notebook as bn  # noqa: E402
import dry_run_e1_local as dr  # noqa: E402
import e1_helpers as eh  # noqa: E402
import notebook_helpers as nh  # noqa: E402
from e1_commands import e1_commands  # noqa: E402

FAKE_TOKEN = "hf_" + "Zq7" * 10
NO_BUNDLE = bn.Bundle("", "0" * 64, [])
SPEC_ORDER = ["env_check", "build_data", "baselines", "evaluate", "evaluate", "train_single", "train_single",
              "export_check", "evaluate", "bench_cpu", "gate"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("nb") / "e1_first_run.ipynb"
    path, sha = be.build(ROOT, out)
    return path, sha, nbformat.read(str(path), as_version=4)


def _code(nb) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _markdown(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")


def _cells(cfg: dict, target=None) -> dict[str, bn.Cell]:
    return {c.key: c for c in be.build_cells(cfg, target or be.colab_target(cfg), NO_BUNDLE)}


def _flat(cmds: dict) -> dict[str, list[str]]:
    return {k: [a.src if isinstance(a, bn.Expr) else a for a in args] for k, (_, args) in cmds.items()}


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


# ---------------------------------------------------------------- notebook structure

def test_notebook_is_valid_compiles_and_has_no_outputs_or_magics(built):
    _, _, nb = built
    nbformat.validate(nb)
    assert nb.metadata.kernelspec.name == "python3" and nb.metadata.colab.gpuType == "T4"
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, f"<cell {i}>", "exec")
            assert not any(line.lstrip().startswith(("!", "%")) for line in cell.source.splitlines())


def test_embedded_bundle_matches_the_repo_bundle_with_the_frozen_manifest(built, cfg):
    _, sha, nb = built
    b64, want, members = B.bundle(ROOT, be.bundle_patterns(cfg, ROOT))
    assert sha == want and b64 in "\n".join(_code(nb)) and want in _markdown(nb)
    manifest = cfg["data"].get("frozen_manifest")
    if manifest:
        assert manifest in members  # build_data --verify-frozen reads it under WORK on Colab


def test_bundle_patterns_add_the_frozen_manifest_and_refuse_a_missing_one(cfg, tmp_path):
    assert be.bundle_patterns(with_overrides(cfg, {"data.frozen_manifest": None}), tmp_path) == B.DEFAULT_INCLUDE
    manifest = tmp_path / "docs" / "results" / "fp.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    got = be.bundle_patterns(with_overrides(cfg, {"data.frozen_manifest": "docs/results/fp.json"}), tmp_path)
    assert got[-1] == "docs/results/fp.json"
    with pytest.raises(FileNotFoundError):
        be.bundle_patterns(with_overrides(cfg, {"data.frozen_manifest": "docs/results/nope.json"}), tmp_path)


def test_no_forbidden_strings_or_token_in_the_notebook_or_its_bundle(built):
    path, _, nb = built
    text = path.read_text(encoding="utf-8")
    for bad in ("userdata", "files.upload", "eval_js", " -U"):
        assert bad not in text, bad
    assert not bn.TOKEN_RE.search(text)
    b64 = re.search(r'BUNDLE_B64 = "([A-Za-z0-9+/=]+)"', "\n".join(_code(nb))).group(1)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as zf:
        assert not any(bn.TOKEN_RE.search(zf.read(n).decode("utf-8", "replace")) for n in zf.namelist())


def test_cli_sequence_follows_the_phase_2_flow(built):
    _, _, nb = built
    assert re.findall(r'"-m", "laya_poc\.(\w+)"', "\n".join(_code(nb))) == SPEC_ORDER


def test_connection_disconnect_and_runtime_notes_agree_across_notebook_runbook_and_readme(built):
    _, _, nb = built
    md = _markdown(nb)
    for phrase in ("Select Kernel", "New Colab Server", "T4", "Auto Connect", "Step 1", "Step 9", "disconnect"):
        assert phrase in md, phrase
    docs = {"notebook": md, "runbook": (ROOT / "docs" / "phase2_e1_runbook.md").read_text(encoding="utf-8"),
            "README": (ROOT / "README.md").read_text(encoding="utf-8")}
    for name, text in docs.items():
        assert be.RUNTIME in text, name


def test_build_is_deterministic_and_the_token_variant_stays_local(tmp_path):
    a, _ = be.build(ROOT, tmp_path / "a.ipynb")
    b, _ = be.build(ROOT, tmp_path / "b.ipynb")
    assert a.read_bytes() == b.read_bytes()
    local, _ = be.build(ROOT, tmp_path / "e1.local.ipynb", token=FAKE_TOKEN)
    holders = [c for c in json.loads(local.read_text(encoding="utf-8"))["cells"] if FAKE_TOKEN in "".join(c["source"])]
    assert len(holders) == 1 and "".join(holders[0]["source"]).startswith("# Step 3")
    with pytest.raises(ValueError, match="gitignored"):
        be.build(ROOT, be.DEFAULT_OUT, token=FAKE_TOKEN)
    assert subprocess.run(["git", "check-ignore", "-q", "notebooks/e1_first_run.local.ipynb"], cwd=ROOT).returncode == 0


@pytest.mark.parametrize("rel", ["notebooks/e1_first_run.ipynb", "notebooks/smoke_test.ipynb", "notebooks/e1.ipynb",
                                 "e1_first_run.local.ipynb", "notebooks/sub/e1.local.ipynb", "docs/e1.local.ipynb"])
def test_a_token_is_refused_for_any_path_git_could_track(tmp_path, rel):
    # A fake root: without the guard, build() would fail on the missing config.yaml, never write into the repo.
    with pytest.raises(ValueError, match="gitignored"):
        be.build(tmp_path, tmp_path / rel, token=FAKE_TOKEN)
    with pytest.raises(ValueError, match="gitignored"):
        be.check_token_out(ROOT, ROOT / rel)


@pytest.mark.parametrize("out", [be.LOCAL_OUT, ROOT / "notebooks" / "other.local.ipynb", None])
def test_a_token_may_go_to_a_gitignored_local_notebook_or_outside_the_repo(tmp_path, out):
    out = out or tmp_path / "e1.local.ipynb"
    be.check_token_out(ROOT, out)  # no error
    if out.is_relative_to(ROOT):
        rel = out.relative_to(ROOT).as_posix()
        assert subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode == 0, rel


def test_committed_notebook_embeds_the_current_code(cfg):
    committed = be.DEFAULT_OUT
    if not committed.exists():
        pytest.skip("notebook not generated yet")
    _, sha, _ = B.bundle(ROOT, be.bundle_patterns(cfg, ROOT))
    code = "\n".join(_code(nbformat.read(str(committed), as_version=4)))
    assert f'BUNDLE_SHA256 = "{sha}"' in code, "notebooks/e1_first_run.ipynb is stale: run tools/build_e1_notebook.py"


def test_phase_1_templates_are_adapted_not_copied(cfg):
    cells = _cells(cfg)
    assert 'WORK / "runs" / "e1"' in be.code_sources(cfg, be.colab_target(cfg), NO_BUNDLE)["setup"]
    assert '"runs" / "smoke"' not in cells["setup"].source
    for f in eh.E1_HELPERS:
        assert f"def {f.__name__}(" in cells["setup"].source
    assert "e1_first_run.local.ipynb" in cells["token"].source and 'PRESET_TOKEN = ""' in cells["token"].source
    with pytest.raises(ValueError, match="template changed"):
        import e1_cells
        e1_cells._derive("abc", "x", "y")


# ---------------------------------------------------------------- commands

def test_crash_and_resume_share_every_setting_but_the_crash(cfg):
    flat, e1 = _flat(e1_commands(cfg, be.colab_target(cfg))), cfg["e1"]
    crash, resume = flat["crash"], flat["resume"]
    assert crash[:-2] == resume and crash[-2:] == ["--crash-at-micro-step", str(e1["crash_at_micro_step"])]
    assert _arg(resume, "--ckpt-every-micro-steps") == str(e1["ckpt_every_micro_steps"])
    assert _arg(resume, "--ckpt-every-min") == "0" and _arg(resume, "--card") == e1["card"]
    assert _arg(resume, "--seed") == str(e1["seed"]) and _arg(resume, "--model") == e1["model"]
    assert _arg(resume, "--epochs") == str(cfg["train"]["epochs"]) and "--init" not in resume
    assert {"--save-best", "--initial-eval", "--final-eval", "--save-final"} <= set(resume)


def test_downstream_steps_use_best_and_the_configured_sizes(cfg):
    flat, bench = _flat(e1_commands(cfg, be.colab_target(cfg))), cfg["bench"]
    for key in ("export", "evaluate", "bench"):
        assert _arg(flat[key], "--ckpt").endswith('"train" / "best")'), key
    assert _arg(flat["export"], "--train-summary").endswith('"best" / "train_eval.json")')
    assert _arg(flat["zeroshot"], "--ckpt") == "hub" and "--smoke" not in flat["data"]
    assert "--verify-frozen" in flat["data"] and "--require-gpu" in flat["env"]
    b = flat["bench"]
    assert _arg(b, "--n") == str(bench["n_records"]) and b[b.index("--threads") + 1:b.index("--warmup")] == \
        [str(t) for t in bench["threads"]]


@pytest.mark.torch
def test_every_rendered_command_is_accepted_by_its_cli_parser(cfg, tmp_path):
    """Catches a renamed or missing flag here instead of on Colab (modules not written yet are skipped)."""
    pytest.importorskip("torch")
    ns = {"WORK": tmp_path, "DATA": tmp_path / "data", "RUNS": tmp_path / "runs" / "e1"}
    missing = []
    for target in (be.colab_target(cfg), dr.local_target(tmp_path, tmp_path / "ckpt", tmp_path / "pool.parquet")):
        for key, (module, args) in e1_commands(cfg, target).items():
            if importlib.util.find_spec(module) is None:
                missing.append(module)
                continue
            mod = importlib.import_module(module)
            parse = mod.build_parser().parse_args if hasattr(mod, "build_parser") else mod.parse_args
            argv = [str(eval(a.src, ns)) if isinstance(a, bn.Expr) else a for a in args]
            try:
                parse(argv)
            except SystemExit:
                pytest.fail(f"{module} rejects the '{key}' command ({target.device}): {argv}")
    if missing:
        pytest.skip(f"not implemented yet: {sorted(set(missing))}")


# ---------------------------------------------------------------- Step 1 / Step 2

def test_setup_cell_creates_runs_e1_and_never_prints_the_saved_token(cfg, tmp_path):
    target = dr.local_target(tmp_path / "work", tmp_path / "ckpt", tmp_path / "pool.parquet")
    script = tmp_path / "setup.py"
    script.write_text(be.build_cells(cfg, target, be.make_bundle(ROOT, cfg))[2].source, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HF_TOKEN_PATH", "LAYA_POC_ROOT")}
    env["HF_HOME"] = str(tmp_path / "hf_home")
    (tmp_path / "hf_home").mkdir()
    (tmp_path / "hf_home" / "token").write_text(FAKE_TOKEN + "\n", encoding="utf-8")
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "work" / "runs" / "e1" / "logs").is_dir() and (tmp_path / "work" / "src" / "laya_poc").is_dir()
    assert FAKE_TOKEN not in res.stdout + res.stderr


def _run_install(cfg: dict, monkeypatch, installed_duckdb: str, changed: dict | None = None) -> list:
    import importlib.metadata as md
    before = {"torch": "2.8.0", "transformers": "5.1.0", "protobuf": "5.29.6", "numpy": "2.0.2",
              "duckdb": installed_duckdb}
    state: dict = {"after": False, "calls": []}

    def version(name):
        after = {"duckdb": cfg["data"]["duckdb_version"], **(changed or {})} if state["after"] else {}
        v = {**before, **after}.get(name)
        if v is None:
            raise md.PackageNotFoundError(name)
        return v

    class Dist:
        def __init__(self, name):
            self.name = name

        def read_text(self, _):
            if self.name == "laya":
                return json.dumps({"vcs_info": {"commit_id": cfg["laya"]["commit"]}})
            return json.dumps({"dir_info": {"editable": True}})

    def run_logged(cmd, log_path, **_):
        state["calls"].append(cmd)
        state["after"] = True
        return 0

    monkeypatch.setattr(md, "version", version)
    monkeypatch.setattr(md, "distribution", Dist)
    exec(_cells(cfg)["install"].source, {"json": json, "sys": sys, "run_logged": run_logged, "LOGS": Path("logs"),
                                         "WORK": Path("work")})
    return state["calls"]


def test_install_pins_duckdb_to_the_frozen_build_version(cfg, monkeypatch):
    pinned = cfg["data"]["duckdb_version"]
    calls = _run_install(cfg, monkeypatch, "1.3.2")
    assert [a for c in calls for a in c if a.startswith("duckdb")] == [f"duckdb=={pinned}"]
    assert _run_install(cfg, monkeypatch, pinned) == []
    with pytest.raises(RuntimeError, match="pip changed.*torch"):
        _run_install(cfg, monkeypatch, "1.3.2", {"torch": "99.0"})


# ---------------------------------------------------------------- E1 helpers

def _crash_files(runs: Path, rc: int | None, events: tuple[str, ...], kill_at: int = 2002) -> Path:
    if rc is not None:
        nh.write_json(runs / "crash_exit.json", {"returncode": rc})
    log = runs / "train" / "log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"event": e, **({"micro_step": kill_at} if e == "crash_injected" else {})}) for e in events]
    log.write_text("\n".join(['{"event": "mic', *lines]) + "\n", encoding="utf-8")  # a half line is tolerated
    return runs


def test_crash_problem_allows_any_number_of_resumes_after_the_injected_crash(tmp_path):
    runs = _crash_files(tmp_path, -9, ("start", "crash_injected", "start", "resumed", "micro", "start", "resumed"))
    assert eh.crash_problem(runs / "crash_exit.json", runs / "train" / "log.jsonl", 2002) is None


@pytest.mark.parametrize("rc, events, kill_at, want", [
    (None, ("start",), 2002, "no crash_exit.json"),
    (0, ("start", "done"), 2002, "exited with 0"),
    (-9, ("start", "micro"), 2002, "no crash_injected"),
    (137, ("start", "crash_injected"), 60, "micro-step 60"),
])
def test_crash_problem_names_what_is_missing(tmp_path, rc, events, kill_at, want):
    runs = _crash_files(tmp_path, rc, events, kill_at)
    assert want in eh.crash_problem(runs / "crash_exit.json", runs / "train" / "log.jsonl", 2002)


def test_next_log_numbers_attempts_and_dig_tolerates_missing_levels(tmp_path):
    assert eh.next_log(tmp_path, "09_e1_resume").name == "09_e1_resume.log"
    (tmp_path / "09_e1_resume.log").write_text("x")
    assert eh.next_log(tmp_path, "09_e1_resume").name == "09_e1_resume_2.log"
    assert eh.dig({"a": {"b": 1}}, "a", "b") == 1 and eh.dig({"a": 1}, "a", "b") is None
    assert (eh.fnum(0.123456), eh.fnum(1.23456, 1), eh.fnum(None), eh.fnum(4)) == ("0.1235", "1.2", "None", "4")


def test_trainer_pids_finds_other_trainers_in_proc(tmp_path, monkeypatch):
    cmdlines = {7: "python -m laya_poc.train_single --run-dir x", 8: "python -m laya_poc.gate",
                os.getpid(): "python -m laya_poc.train_single"}
    for pid in (*cmdlines, "self"):
        (tmp_path / str(pid)).mkdir()
    monkeypatch.setattr(eh, "Path", lambda _: tmp_path)
    monkeypatch.setattr(eh, "proc_cmdline", lambda pid: cmdlines.get(int(pid), ""))
    assert eh.trainer_pids() == [7]
    monkeypatch.setattr(eh, "Path", lambda _: tmp_path / "no_proc")  # Windows / macOS: no /proc
    assert eh.trainer_pids() == []


# ---------------------------------------------------------------- Steps 8/9 with the trainer stubbed

def _kernel_ns(work: Path) -> dict:
    import getpass
    import signal
    ns = {"__name__": "__main__", "json": json, "os": os, "shutil": shutil, "subprocess": subprocess, "sys": sys,
          "time": time, "signal": signal, "getpass": getpass, "Path": Path, "zipfile": zipfile, "REDO": False}
    exec("\n\n".join(inspect.getsource(f) for f in (*nh.HELPERS, *eh.E1_HELPERS)), ns)
    runs = work / "runs" / "e1"
    ns.update(WORK=work, DATA=work / "data", RUNS=runs, LOGS=runs / "logs")
    return ns


def _trainer(ns: dict, calls: list, rc: int, events: tuple[str, ...], kill_at: int = 2002):
    def run_logged(cmd, log_path, expect_returncode=0, hint=None):
        calls.append(cmd)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("resumed from ckpt/step0000500.pt\nmicro 2010 | loss_ce 1.8\n", encoding="utf-8")
        _crash_files(ns["RUNS"], None, events, kill_at)
        if "--crash-at-micro-step" not in cmd and "done" in events:
            nh.write_json(ns["RUNS"] / "train" / "summary.json", {"micro_steps": 13376})
        return rc
    ns["run_logged"] = run_logged


def test_crash_step_records_the_injected_crash(cfg, tmp_path, capsys):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    _trainer(ns, calls, -9, ("start", "micro", "crash_injected"))
    exec(_cells(cfg)["crash"].source, ns)
    assert json.loads((ns["RUNS"] / "crash_exit.json").read_text()) == {"returncode": -9} and len(calls) == 1
    assert "as planned" in capsys.readouterr().out


@pytest.mark.parametrize("rc, events, kill_at, want", [
    (0, ("start", "done"), 2002, "exited with 0"),
    (1, ("start", "micro"), 2002, "CUDA OOM"),
    (-9, ("start", "micro"), 2002, "dmesg"),
    (137, ("start", "crash_injected"), 60, "micro-step 60"),
])
def test_crash_step_refuses_anything_but_the_planned_kill(cfg, tmp_path, rc, events, kill_at, want):
    ns = _kernel_ns(tmp_path / "w")
    _trainer(ns, [], rc, events, kill_at)
    with pytest.raises(RuntimeError, match=want) as exc:
        exec(_cells(cfg)["crash"].source, ns)
    assert "loss_ce 1.8" in str(exc.value)  # the log tail


def test_run_all_after_a_disconnect_skips_the_crash_step_and_keeps_training_state(cfg, tmp_path, capsys):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    _crash_files(ns["RUNS"], -9, ("start", "crash_injected", "start", "resumed", "micro"))
    (ns["RUNS"] / "train" / "ckpt").mkdir()
    _trainer(ns, calls, 0, ("start", "crash_injected", "start", "resumed", "micro"))
    exec(_cells(cfg)["crash"].source, ns)
    assert calls == [] and (ns["RUNS"] / "train" / "ckpt").is_dir() and "skip" in capsys.readouterr().out


@pytest.mark.parametrize("step, events", [("crash", ("start", "micro")),
                                           ("resume", ("start", "crash_injected"))])
def test_train_steps_never_start_a_second_trainer(cfg, tmp_path, step, events):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    _crash_files(ns["RUNS"], -9 if step == "resume" else None, events)
    _trainer(ns, calls, 0, events)
    ns["trainer_pids"] = lambda: [4242]  # e.g. the trainer from before a kernel restart
    with pytest.raises(RuntimeError, match="still running.*4242"):
        exec(_cells(cfg)[step].source, ns)
    assert calls == [] and (ns["RUNS"] / "train" / "log.jsonl").exists()  # nothing was wiped


def test_resume_step_refuses_before_the_crash(cfg, tmp_path):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    _trainer(ns, calls, 0, ("start", "done"))
    with pytest.raises(RuntimeError, match="run Step 8 first"):
        exec(_cells(cfg)["resume"].source, ns)
    assert calls == []


def test_resume_step_resumes_again_after_a_disconnect_with_a_new_log(cfg, tmp_path):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    events = ("start", "crash_injected", "start", "resumed", "micro")
    _crash_files(ns["RUNS"], -9, events)
    (ns["LOGS"]).mkdir(parents=True)
    (ns["LOGS"] / "09_e1_resume.log").write_text("first attempt, cut by a disconnect\n")
    _trainer(ns, calls, 0, (*events, "start", "resumed", "done"))
    exec(_cells(cfg)["resume"].source, ns)
    assert len(calls) == 1 and "--crash-at-micro-step" not in calls[0]
    assert (ns["LOGS"] / "09_e1_resume.log").read_text().startswith("first attempt")
    assert (ns["LOGS"] / "09_e1_resume_2.log").exists() and ns["e1"] == {"micro_steps": 13376}


# ---------------------------------------------------------------- Step 12

def test_bench_step_reruns_after_a_benchmark_with_no_thread_setting(cfg, tmp_path, capsys):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    outcomes = iter(([], [{"threads": 1, "p50_ms": 40.0}]))

    def run_logged(cmd, log_path, expect_returncode=0, hint=None):
        calls.append(cmd)
        results = next(outcomes)
        # bench_cpu writes its JSON first, then exits 1 when every thread setting failed
        nh.write_json(ns["RUNS"] / "bench_cpu.json", {"results": results, "errors": [] if results else ["boom"]})
        if not results:
            raise RuntimeError(f"{cmd[2]} exited with 1")
        return 0

    ns["run_logged"] = run_logged
    src = _cells(cfg)["bench"].source
    with pytest.raises(RuntimeError, match="exited with 1"):
        exec(src, ns)
    exec(src, ns)  # after fixing the cause: retried, not skipped because the failed run left bench_cpu.json
    assert len(calls) == 2 and ns["bench"]["results"]
    capsys.readouterr()
    exec(src, ns)  # finished: skipped
    assert len(calls) == 2 and "skip" in capsys.readouterr().out


# ---------------------------------------------------------------- Step 14

def test_archive_has_no_checkpoints_and_no_fsq_rows(cfg, tmp_path):
    ns = _kernel_ns(tmp_path / "w")
    data, runs = ns["DATA"], ns["RUNS"]
    data.mkdir(parents=True)
    for name in ("NOTICE_FSQ.txt", "data_report.json", "SHA256SUMS", "val.jsonl", "split_val.parquet",
                 "trap_candidates.csv"):
        (data / name).write_text("x", encoding="utf-8")
    for rel in ("train/summary.json", "train/ckpt/meta.json", "train/best/train_eval.json", "train/final/x.json",
                "gate_report.json", "preds/e1_val.jsonl"):
        nh.write_json(runs / rel, {})
    exec(_cells(cfg)["archive"].source, ns)
    with zipfile.ZipFile(ns["WORK"] / "e1_artifacts.zip") as zf:
        names = set(zf.namelist())
    assert {"data/NOTICE_FSQ.txt", "runs/e1/train/summary.json", "runs/e1/gate_report.json"} <= names
    assert not any(n.startswith("data/") and n.endswith((".jsonl", ".parquet", ".csv")) for n in names)
    assert not any(part in n for n in names for part in ("/ckpt/", "/best/", "/final/"))


def test_cleanup_keeps_checkpoints_until_e1_has_finished(cfg, tmp_path, capsys):
    ns = _kernel_ns(tmp_path / "w")
    nh.write_json(ns["RUNS"] / "train" / "ckpt" / "step.json", {})
    src = _cells(cfg)["cleanup"].source.replace("CLEANUP_CKPTS = False", "CLEANUP_CKPTS = True")
    exec(src, ns)
    assert (ns["RUNS"] / "train" / "ckpt").is_dir() and "has not finished" in capsys.readouterr().out
    nh.write_json(ns["RUNS"] / "train" / "summary.json", {})
    exec(src, ns)
    assert not (ns["RUNS"] / "train" / "ckpt").exists()


# ---------------------------------------------------------------- dry run

def test_dry_config_is_valid_tiny_and_does_not_touch_the_project_config(cfg):
    dcfg = dr.dry_config(cfg)
    validate_config(dcfg)
    assert dcfg["data"]["train_size"] <= 400 and dcfg["train"]["epochs"] == 2
    assert dcfg["train"]["eval_every_opt_steps"] == 2 and dcfg["data"]["frozen_manifest"] is None
    assert dcfg["e1"]["crash_at_micro_step"] > dcfg["e1"]["ckpt_every_micro_steps"]
    # Early stopping can never fire: the run crosses the epoch boundary and ends with stop_reason 'epochs'.
    # (evals <= opt steps <= micro-steps <= items; an epoch has at most 2 x train_size items with stripped copies)
    max_evals = 2 + dcfg["train"]["epochs"] * 2 * dcfg["data"]["train_size"]
    assert dcfg["train"]["patience"] > max_evals
    assert cfg["train"]["epochs"] == load_config(ROOT / "config.yaml")["train"]["epochs"]
    assert cfg["train"]["patience"] == load_config(ROOT / "config.yaml")["train"]["patience"]


def _train_run(runs: Path, total_micro: int, **summary) -> Path:
    log = [{"event": "start", "total_micro_steps": total_micro}, {"event": "crash_injected", "micro_step": 10},
           {"event": "start", "total_micro_steps": total_micro}, {"event": "resumed"}]
    (runs / "train").mkdir(parents=True, exist_ok=True)
    (runs / "train" / "log.jsonl").write_text("\n".join(json.dumps(e) for e in log) + "\n", encoding="utf-8")
    nh.write_json(runs / "train" / "summary.json", {"stop_reason": "epochs", "epochs": 2, "micro_steps": total_micro,
                                                    **summary})
    return runs


@pytest.mark.parametrize("summary, want", [
    ({}, None),
    ({"stop_reason": "early_stop", "micro_steps": 20}, "early_stop"),
    ({"micro_steps": 100}, "100 of the 128"),
    ({"epochs": 1}, "epoch boundary"),
])
def test_training_failures_require_every_planned_micro_step_of_both_epochs(tmp_path, summary, want):
    got = dr.training_failures(_train_run(tmp_path, 128, **summary))
    assert (got == []) if want is None else (len(got) == 1 and want in got[0]), got


def test_training_failures_name_a_missing_summary(tmp_path):
    assert "summary.json" in " ".join(dr.training_failures(tmp_path))


def test_dry_run_fails_when_training_stopped_early_even_if_the_gate_plumbing_passed(tmp_path, monkeypatch, capsys):
    from laya_poc import gate as G

    runs = _train_run(tmp_path / "runs", 128, stop_reason="early_stop", micro_steps=20)
    nh.write_json(runs / "gate_report.json", {"verdict": "FAIL", "criteria": [
        {"name": G.END_TO_END, "passed": True, "note": ""}, {"name": G.NUMERICS, "passed": True, "note": ""}]})
    monkeypatch.setattr(dr, "dry_run", lambda work, tiny=None: runs)
    assert dr.main(["--work", str(tmp_path / "w")]) == 1
    assert "early_stop" in capsys.readouterr().err
    _train_run(runs, 128)
    assert dr.main(["--work", str(tmp_path / "w")]) == 0


def test_synthetic_full_pool_covers_every_country_with_trap_material(cfg, tmp_path):
    import pandas as pd

    pool = pd.read_parquet(dr.write_synthetic_full_pool(tmp_path / "pool.parquet", dr.dry_config(cfg)))
    d = cfg["data"]
    want = {*d["id_countries"], *d["ood_country"], d["ood_script"], "KR"}
    assert want <= set(pool["country"]) and (pool["n_l1"] > 1).any()
    assert pool["fsq_place_id"].is_unique
    top = pool.loc[pool["n_l1"] == 1, "name"].value_counts()
    assert top.iloc[0] >= dr.dry_config(cfg)["data"]["ood_size"] // 3  # brands for ood_brand


@pytest.mark.torch
@pytest.mark.skipif(not os.environ.get("LAYA_DRY_RUN_TEST"),
                    reason="slow (a few minutes on CPU, many subprocesses): set LAYA_DRY_RUN_TEST=1 to run")
def test_dry_run_end_to_end_passes_the_plumbing_criteria(en_snapshot, tmp_path):
    missing = [m for m in ("evaluate", "baselines", "bench_cpu", "traps")
               if importlib.util.find_spec(f"laya_poc.{m}") is None]
    if missing:
        pytest.skip(f"Phase 2 modules not implemented yet: {missing}")
    from conftest import build_tiny_checkpoint

    tiny = build_tiny_checkpoint(en_snapshot, tmp_path / "tiny", init_scale=0.5)
    env = {k: v for k, v in os.environ.items() if k != "LAYA_POC_ROOT"}
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "dry_run_e1_local.py"), "--work", str(tmp_path / "w"),
                          "--tiny-ckpt", str(tiny)], capture_output=True, text=True, env=env, timeout=1800)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    runs = tmp_path / "w" / "laya_poc" / "runs" / "e1"
    report = json.loads((runs / "gate_report.json").read_text(encoding="utf-8"))
    crit = {c["name"]: c for c in report["criteria"]}
    from laya_poc import gate as G
    assert crit[G.END_TO_END]["passed"], crit[G.END_TO_END]
    assert crit[G.NUMERICS]["passed"], crit[G.NUMERICS]
    # both epochs trained to the end: the epoch boundary crossed, the natural-end final eval and exports
    summary = json.loads((runs / "train" / "summary.json").read_text(encoding="utf-8"))
    planned = [e["total_micro_steps"] for e in eh.read_events(runs / "train" / "log.jsonl") if e["event"] == "start"]
    assert summary["stop_reason"] == "epochs" and summary["epochs"] == 2, summary
    assert summary["micro_steps"] == planned[-1] and summary["opt_steps"] == summary["total_opt_steps"], summary
    assert (runs / "train" / "final").is_dir() and summary["final_eval"]
    assert json.loads((runs / "crash_exit.json").read_text())["returncode"] in (-9, 137)
    assert (runs / "train" / "best").is_dir() and (tmp_path / "w" / "laya_poc" / "e1_artifacts.zip").exists()
