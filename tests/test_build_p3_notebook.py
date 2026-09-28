"""The generated Phase 3 notebook (tools/build_p3_notebook.py), its cells run with stubbed CLIs, and the CPU dry run."""
from __future__ import annotations

import base64
import contextlib
import importlib.util  # also binds importlib
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
from laya_poc.config import load_config, validate_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_e1_notebook as be  # noqa: E402
import build_notebook as bn  # noqa: E402
import build_p3_notebook as bp  # noqa: E402
import dry_run_p3_local as dr  # noqa: E402
import e1_cells  # noqa: E402
import e1_helpers as eh  # noqa: E402
import notebook_helpers as nh  # noqa: E402
import p3_helpers as ph  # noqa: E402
from e1_commands import e1_commands  # noqa: E402
from p3_commands import B2, all_run_names, p3_commands, run_names, subset_sizes, variant_dirs  # noqa: E402

FAKE_TOKEN = "hf_" + "Xy9" * 10
NO_BUNDLE = bn.Bundle("", "0" * 64, [])
CLI_RE = re.compile(r'"-m", "laya_poc\.(\w+)"(?:, "(\w+)")?')


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path, sha = bp.build(ROOT, tmp_path_factory.mktemp("nb") / "phase3_matrix.ipynb")
    return path, sha, nbformat.read(str(path), as_version=4)


def _src(nb, kind: str = "code") -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == kind)


def _cells(cfg: dict, target=None) -> dict[str, bn.Cell]:
    return {c.key: c for c in bp.build_cells(cfg, target or bp.colab_target(cfg), NO_BUNDLE)}


def _flat(cmds: dict) -> dict[str, list[str]]:
    return {k: [a.src if isinstance(a, bn.Expr) else a for a in args] for k, (_, args) in cmds.items()}


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _local(tmp_path: Path):
    return dr.local_target(tmp_path / "work", tmp_path / "ckpt", tmp_path / "pool.parquet")


def test_notebook_is_valid_compiles_and_embeds_the_repo_bundle_without_a_token(built, cfg):
    path, sha, nb = built
    nbformat.validate(nb)
    assert nb.metadata.kernelspec.name == "python3" and nb.metadata.colab.gpuType == cfg["phase3"]["card"]
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, f"<cell {i}>", "exec")
            assert not any(line.lstrip().startswith(("!", "%")) for line in cell.source.splitlines())
    b64, want, members = B.bundle(ROOT, be.bundle_patterns(cfg, ROOT))
    assert sha == want and b64 in _src(nb) and want in _src(nb, "markdown")
    assert cfg["data"]["frozen_manifest"] in members  # build_data --verify-frozen reads it on Colab
    text = path.read_text(encoding="utf-8")
    assert not [bad for bad in ("userdata", "files.upload", "eval_js", " -U") if bad in text]
    assert not bn.TOKEN_RE.search(text)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as zf:
        assert not any(bn.TOKEN_RE.search(zf.read(n).decode("utf-8", "replace")) for n in zf.namelist())


def test_cli_sequence_follows_the_phase_3_session(built, cfg):
    _, _, nb = built
    assert subset_sizes(cfg) == [1000, 3000, 10000]  # E6
    want = [("env_check", ""), ("build_data", ""), ("variants", "c7"), *[("variants", "subset")] * 3,
            ("variants", "traps"), ("parity", ""), ("parity", ""), ("matrix", "plan"), *[("matrix", "run")] * 6,
            ("matrix", "report")]
    assert CLI_RE.findall(_src(nb)) == want
    assert re.findall(r'"--only", "(\w+)"', _src(nb)) == ["B2", "E2", "E3", "E4", "E5", "E6"]


def test_connection_disconnect_and_runtime_notes_agree_across_notebook_runbook_and_readme(built):
    _, _, nb = built
    md = _src(nb, "markdown")
    for phrase in ("Select Kernel", "New Colab Server", "G4", "Auto Connect", "Step 1", "disconnect", "CARD",
                   "Remove Server", "exit check"):
        assert phrase in md, phrase
    docs = {"notebook": md, "runbook": (ROOT / "docs" / "phase3_matrix_runbook.md").read_text(encoding="utf-8"),
            "README": (ROOT / "README.md").read_text(encoding="utf-8")}
    for name, text in docs.items():
        assert bp.RUNTIME in text and "phase3_matrix" in text, name
    titles = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("###")]
    assert titles[:3] == ["### Step 1 - Setup", "### Step 2 - Install", "### Step 3 - Hugging Face token"]
    assert any(t.startswith("### Step 10 - E2") for t in titles) and "### Step 15 - Report" in "\n".join(titles)
    for arm in (B2, "E2", "E3", "E4", "E5", "E6"):  # the intro's matrix table estimates every step
        assert re.search(rf"\| {arm} \|.*\| ~\d+-\d+ min \|", nb.cells[0].source), arm


def test_build_is_deterministic_and_the_token_variant_stays_local(tmp_path):
    a, _ = bp.build(ROOT, tmp_path / "a.ipynb")
    b, _ = bp.build(ROOT, tmp_path / "b.ipynb")
    assert a.read_bytes() == b.read_bytes()
    local, _ = bp.build(ROOT, tmp_path / "p3.local.ipynb", token=FAKE_TOKEN)
    holders = [c for c in json.loads(local.read_text(encoding="utf-8"))["cells"] if FAKE_TOKEN in "".join(c["source"])]
    assert len(holders) == 1 and "".join(holders[0]["source"]).startswith("# Step 3")
    with pytest.raises(ValueError, match="gitignored"):
        bp.build(ROOT, bp.DEFAULT_OUT, token=FAKE_TOKEN)
    assert subprocess.run(["git", "check-ignore", "-q", bp.LOCAL_OUT.relative_to(ROOT).as_posix()],
                          cwd=ROOT).returncode == 0


@pytest.mark.parametrize("rel", ["notebooks/phase3_matrix.ipynb", "phase3_matrix.local.ipynb",
                                 "notebooks/sub/p3.local.ipynb", "docs/p3.local.ipynb"])
def test_a_token_is_refused_for_any_path_git_could_track(tmp_path, rel):
    with pytest.raises(ValueError, match="phase3_matrix.local.ipynb"):
        bp.build(tmp_path, tmp_path / rel, token=FAKE_TOKEN)  # a fake root: refused before reading any config
    with pytest.raises(ValueError, match="gitignored"):
        bp.check_token_out(ROOT, ROOT / rel)
    bp.check_token_out(ROOT, bp.LOCAL_OUT)  # the default local variant is fine


def test_committed_notebook_embeds_the_current_code(cfg):
    if not bp.DEFAULT_OUT.exists():
        pytest.skip("notebook not generated yet")
    _, sha, _ = B.bundle(ROOT, be.bundle_patterns(cfg, ROOT))
    code = _src(nbformat.read(str(bp.DEFAULT_OUT), as_version=4))
    assert f'BUNDLE_SHA256 = "{sha}"' in code, "notebooks/phase3_matrix.ipynb is stale: run tools/build_p3_notebook.py"


def test_earlier_templates_are_adapted_and_the_data_step_is_e1s(cfg):
    cells, e1 = _cells(cfg), e1_cells.code_sources(cfg, be.colab_target(cfg), NO_BUNDLE)
    setup = cells["setup"].source
    assert 'WORK / "runs" / "p3"' in setup and f'CARD = "{cfg["phase3"]["card"]}"' in setup
    assert 'RESULTS = WORK / "results"' in setup and '"runs" / "smoke"' not in setup
    for f in (*nh.HELPERS, *eh.E1_HELPERS, *ph.P3_HELPERS):
        assert f"def {f.__name__}(" in setup
    assert "phase3_matrix.local.ipynb" in cells["token"].source and 'PRESET_TOKEN = ""' in cells["token"].source
    assert e1["install"] in cells["install"].source and e1["data"] in cells["data"].source


def test_colab_and_local_commands_follow_the_spec(cfg, tmp_path):
    flat = _flat(p3_commands(cfg, bp.colab_target(cfg)))
    assert flat["data"] == _flat(e1_commands(cfg, be.colab_target(cfg)))["data"]
    assert "--require-gpu" in flat["env"] and _arg(flat["env"], "--expect-card") == "CARD"
    assert [_arg(flat[f"variant_{d}"], "--out") for d in variant_dirs(cfg)] == \
        [f'str(WORK / "{d}")' for d in ("data_c7", "data_lc1000", "data_lc3000", "data_lc10000", "data_eval")]
    assert [_arg(flat[f"variant_data_lc{n}"], "--n") for n in (1000, 3000, 10000)] == ["1000", "3000", "10000"]
    for arm in (B2, "E2", "E3", "E4", "E5", "E6"):
        args = flat[arm]
        assert args[0] == "run" and _arg(args, "--only") == arm and _arg(args, "--card") == "CARD"
        assert _arg(args, "--device") == "cuda" and "--init" not in args and _arg(args, "--work") == "str(WORK)"
    assert _arg(flat["report"], "--out") == 'str(WORK / "results" / "phase3_report.md")'
    local = _flat(p3_commands(cfg, _local(tmp_path)))
    tiny = str(tmp_path / "ckpt")
    assert _arg(local["E3"], "--init") == tiny and _arg(local["E3"], "--device") == "cpu"
    assert _arg(local["variant_data_c7"], "--init-tokenizer") == tiny and "--allow-cpu" in local["parity_laya_ml"]
    assert local["variant_data_c7"][-2:] == ["laya", "laya_ml"] and "--require-gpu" not in local["env"]


def test_run_names_follow_the_spec_contract_and_the_matrix_plan(cfg):
    assert run_names(cfg, "E2") == ["fsq-c10-E2-laya-s11", "fsq-c10-E2-laya-s22", "fsq-c10-E2-laya-s33"]
    assert run_names(cfg, "E4") == ["fsq-c10-E4-laya-s11-head"]
    assert run_names(cfg, "E5") == ["fsq-c7-E5-laya-s11", "fsq-c7-E5-laya_ml-s11"]
    assert run_names(cfg, "E6")[1] == "fsq-c10-E6-laya-s11-n3000"
    assert run_names(cfg, B2) == ["fsq-c10-B2-laya-zs", "fsq-c10-B2-laya_ml-zs"]
    assert len(all_run_names(cfg)) == 3 + 3 + 1 + 2 + 3 + 2 == len(set(all_run_names(cfg)))
    if importlib.util.find_spec("laya_poc.matrix_plan"):  # the notebook, the dry run and the matrix agree
        from laya_poc.matrix_plan import expand_specs
        for c in (cfg, dr.dry_config(cfg)):
            assert [s.name for s in expand_specs(c)] == all_run_names(c)


@pytest.mark.torch
def test_every_rendered_command_is_accepted_by_its_cli_parser(cfg, tmp_path):
    """A renamed or missing flag fails here instead of on Colab (a module not written yet is skipped, named)."""
    pytest.importorskip("torch")
    ns = {"WORK": tmp_path, "DATA": tmp_path / "data", "RUNS": tmp_path / "runs" / "p3", "CARD": "G4"}
    missing = set()
    for target in (bp.colab_target(cfg), _local(tmp_path)):
        for key, (module, args) in p3_commands(cfg, target).items():
            if importlib.util.find_spec(module) is None:
                missing.add(module)
                continue
            mod = importlib.import_module(module)
            parser = getattr(mod, "build_parser", None)
            parse = parser().parse_args if parser else getattr(mod, "parse_args", None)
            assert parse, f"{module} has neither build_parser() nor parse_args(): update this test"
            # Expr sources are this project's own rendered path expressions (str(WORK / "x"), CARD), as in the E1 test
            argv = [str(eval(a.src, ns)) if isinstance(a, bn.Expr) else a for a in args]
            try:
                parse(argv)
            except SystemExit:
                pytest.fail(f"{module} rejects the '{key}' command ({target.device}): {argv}")
    if missing:
        pytest.skip(f"Phase 3 CLIs not implemented yet (their commands were not parsed): {sorted(missing)}")


def test_setup_cell_creates_runs_p3_and_never_prints_the_saved_token(cfg, tmp_path):
    cells = {c.key: c for c in bp.build_cells(cfg, _local(tmp_path), be.make_bundle(ROOT, cfg))}
    script = tmp_path / "setup.py"
    script.write_text(cells["setup"].source, encoding="utf-8")
    env = {**{k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HF_TOKEN_PATH", "LAYA_POC_ROOT")},
           "HF_HOME": str(tmp_path / "hf_home")}
    (tmp_path / "hf_home").mkdir()
    (tmp_path / "hf_home" / "token").write_text(FAKE_TOKEN + "\n", encoding="utf-8")
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "work" / "runs" / "p3" / "logs").is_dir() and (tmp_path / "work" / "src" / "laya_poc").is_dir()
    assert FAKE_TOKEN not in res.stdout + res.stderr


def _kernel_ns(work: Path, card: str = "G4") -> dict:  # what Step 1 leaves in the kernel for the later steps
    ns = {"__name__": "__main__", "json": json, "os": os, "shutil": shutil, "subprocess": subprocess, "sys": sys,
          "time": time, "Path": Path, "zipfile": zipfile, "REDO": False}
    exec("\n\n".join(inspect.getsource(f) for f in (*nh.HELPERS, *eh.E1_HELPERS, *ph.P3_HELPERS)), ns)
    runs = work / "runs" / "p3"
    ns.update(WORK=work, DATA=work / "data", RUNS=runs, LOGS=runs / "logs", RESULTS=work / "results", CARD=card)
    return ns


def _recorder(ns: dict, calls: list, effect=None, rc: int = 0):
    def run_logged(cmd, log_path, expect_returncode=0, hint=None):
        calls.append((cmd, Path(log_path)))
        nh.write_json(log_path, "log")  # next_log numbers the attempts by the logs that exist
        if effect:
            effect(cmd)
        return rc
    ns["run_logged"] = run_logged


def test_variants_step_rebuilds_a_partial_directory_and_skips_finished_ones(cfg, tmp_path, capsys):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    partial = ns["WORK"] / "data_c7" / "train_e0.jsonl"
    nh.write_json(partial, "half")

    def build(cmd):
        out = Path(cmd[cmd.index("--out") + 1])
        assert not out.exists(), "the variants CLI must never see a half-written directory"
        out.mkdir(parents=True)
        (out / "variant.json").write_text("{}", encoding="utf-8")

    _recorder(ns, calls, build)
    exec(_cells(cfg)["variants"].source, ns)
    assert len(calls) == len(variant_dirs(cfg)) and not partial.exists()
    assert all((ns["WORK"] / d / "build_ok.json").exists() for d in variant_dirs(cfg))
    capsys.readouterr()
    exec(_cells(cfg)["variants"].source, ns)
    assert len(calls) == len(variant_dirs(cfg)) and capsys.readouterr().out.count("skip:") == len(variant_dirs(cfg))


@pytest.mark.parametrize("passed, allow, raises", [(True, False, False), (False, False, True), (False, True, False)])
def test_parity_step_stops_the_notebook_on_fail_unless_allowed(cfg, tmp_path, capsys, passed, allow, raises):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    _recorder(ns, calls, lambda cmd: nh.write_json(ns["RUNS"] / "parity_laya_ml.json", {"passed": passed}))
    src = _cells(cfg)["parity_laya_ml"].source.replace("PARITY_MUST_PASS = True", f"PARITY_MUST_PASS = {not allow}")
    with pytest.raises(RuntimeError, match="parity FAILED for laya_ml") if raises else contextlib.nullcontext():
        exec(src, ns)
    assert raises or ("WARNING" in capsys.readouterr().out) == (not passed)
    assert len(calls) == 1


def test_arm_step_refuses_while_a_matrix_runs_and_keeps_one_log_per_attempt(cfg, tmp_path, capsys):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    src = _cells(cfg)["E2"].source
    _recorder(ns, calls)
    ns["busy_pids"] = lambda: [4242]  # e.g. the matrix from before a kernel restart
    with pytest.raises(RuntimeError, match="still running.*4242"):
        exec(src, ns)
    assert calls == []
    ns["busy_pids"] = lambda: []
    nh.write_json(ns["RESULTS"] / "x.json", {})
    (ns["RESULTS"] / "runs.csv").write_text("run_name,arm,val_macro_f1\nfsq-c10-E2-laya-s11,E2,0.5761234\n"
                                            "fsq-c10-E3-laya_ml-s11,E3,0.5\n", encoding="utf-8")
    exec(src, ns)
    assert "fsq-c10-E2-laya-s11" in (out := capsys.readouterr().out) and "0.5761" in out and "Step 10 - E2:" in out
    _recorder(ns, calls, rc=1)  # e.g. one run failed: its rows are shown first, then the step fails
    with pytest.raises(RuntimeError, match=r"E2: matrix run exited with 1; the 'matrix:' lines in .*10_E2_2\.log"):
        exec(src, ns)  # a re-run (after a fix or a disconnect) keeps the earlier log
    assert [log.name for _, log in calls] == ["10_E2.log", "10_E2_2.log"] and _arg(calls[0][0], "--card") == "G4"
    assert "fsq-c10-E2-laya-s11" in (out := capsys.readouterr().out) and "E3-laya_ml" not in out


def test_report_step_shows_the_report_before_failing_on_its_exit_code(cfg, tmp_path):
    ns, calls, shown = _kernel_ns(tmp_path / "w"), [], []
    _recorder(ns, calls, lambda cmd: nh.write_json(ns["RESULTS"] / "phase3_report.md", "FAIL"), rc=1)
    ns["show_markdown"] = shown.append
    with pytest.raises(RuntimeError, match="matrix report exited with 1"):
        exec(_cells(cfg)["report"].source, ns)
    assert shown == [ns["RESULTS"] / "phase3_report.md"]


def test_archive_has_no_checkpoints_and_no_fsq_rows_and_drive_is_off(cfg, tmp_path, capsys):
    ns = _kernel_ns(tmp_path / "w")
    work, run = ns["WORK"], "fsq-c10-E2-laya-s11"
    for rel in ("data/NOTICE_FSQ.txt", "data/data_report.json", "data/val.jsonl", "data/split_val.parquet",
                "data/trap_candidates.csv", "data_eval/trap_candidates.jsonl", "data_c7/val.jsonl",
                "data_c7/variant.json", "results/phase3_report.json", "results/runs.csv",
                *(f"runs/p3/{run}/{r}" for r in ("done.json", "eval/val.json", "preds/val.jsonl", "train/summary.json",
                                                 "train/best/train_eval.json", "train/ckpt/meta.json",
                                                 "train/final/x.json", "logs/train.log"))):
        nh.write_json(work / rel, {})
    exec(_cells(cfg)["archive"].source, ns)
    exec(_cells(cfg)["drive"].source, ns)
    assert "Drive copy skipped" in capsys.readouterr().out
    with zipfile.ZipFile(work / "phase3_artifacts.zip") as zf:
        names = set(zf.namelist())
    assert {f"runs/p3/{run}/preds/val.jsonl", f"runs/p3/{run}/done.json", "results/runs.csv",
            "data/NOTICE_FSQ.txt", "data_c7/variant.json"} <= names
    assert not [n for n in names if n.startswith("data") and n.endswith((".jsonl", ".parquet", ".csv"))]  # FSQ rows
    assert not any(part in n for n in names for part in ("/ckpt/", "/best/", "/final/"))


def test_card_note_show_runs_busy_pids_and_elapsed(tmp_path, monkeypatch, capsys):
    env = tmp_path / "env.json"
    assert "no GPU" in ph.card_note(env, "G4")  # missing env.json
    nh.write_json(env, {"gpu_name": "NVIDIA L4", "card": "L4"})
    assert "CARD" in ph.card_note(env, "G4") and ph.card_note(env, "L4") is None and ph.card_note(env, "CPU") is None
    assert ph.show_runs(tmp_path / "runs.csv") == [] and "no run has finished" in capsys.readouterr().out
    monkeypatch.setattr(ph, "trainer_pids", {"laya_poc.matrix": [9, 3], "laya_poc.train_single": [3]}.get)
    assert ph.busy_pids() == [3, 9]
    monkeypatch.setattr(ph.time, "time", lambda: 1000.0)
    assert (ph.elapsed(990.0), ph.elapsed(700.0)) == ("10 s", "5.0 min")


def test_dry_config_is_valid_tiny_and_covers_every_arm_feature(cfg):
    dcfg = dr.dry_config(cfg)
    validate_config(dcfg)
    arms, train = dcfg["phase3"]["arms"], dcfg["data"]["train_size"]
    assert train <= 400 and dcfg["train"]["epochs"] == 1 and dcfg["data"]["frozen_manifest"] is None
    assert max(n for a in arms for n in a.get("train_subset", [])) < train
    assert [a["model"] for a in arms if a["id"] == "E5"] == ["laya", "laya_ml"] and len(run_names(dcfg, B2)) == 2
    assert any(a.get("freeze_encoder") for a in arms) and any(len(a["seeds"]) > 1 for a in arms)
    assert cfg["phase3"] == load_config(ROOT / "config.yaml")["phase3"]  # the project config is untouched


def _fake_finished_work(dcfg: dict, work: Path) -> Path:
    splits = dcfg["phase3"]["eval_splits"]
    for name in all_run_names(dcfg):
        trained = name not in run_names(dcfg, B2)
        for rel in ("done.json", "order_invariance.json", *(f"eval/{s}.json" for s in splits),
                    *(f"preds/{s}.jsonl" for s in splits), *(["export_check.json", "train/summary.json",
                                                              "train/best/x.json"] if trained else [])):
            nh.write_json(work / "runs" / "p3" / name / rel, {})
    nh.write_json(work / "results" / "phase3_report.json", {"exit_check": {"passed": True}})
    rows = "".join(f"{n}{',' * (len(dr.RUNS_CSV_COLUMNS) - 1)}\n" for n in all_run_names(dcfg))
    (work / "results" / "runs.csv").write_text(",".join(dr.RUNS_CSV_COLUMNS) + "\n" + rows, encoding="utf-8")
    return work


def test_plumbing_failures_name_whatever_the_run_layout_lacks(cfg, tmp_path, monkeypatch, capsys):
    dcfg = dr.dry_config(cfg)
    work = _fake_finished_work(dcfg, tmp_path)
    assert dr.plumbing_failures(dcfg, work) == []
    monkeypatch.setattr(dr, "dry_run", lambda root, tiny=None: work)
    assert dr.main(["--work", str(tmp_path / "x")]) == 0
    first = work / "runs" / "p3" / all_run_names(dcfg)[-1]
    (first / "eval" / "ood_brand.json").unlink()
    nh.write_json(first / "train" / "ckpt" / "step.json", {})
    nh.write_json(work / "results" / "phase3_report.json", {"exit_check": {"passed": False, "runs": [
        {"run_name": "r1", "passed": False, "missing": ["best/", "T"]}, {"run_name": "r2", "passed": True}]}})
    got = "\n".join(dr.plumbing_failures(dcfg, work))
    assert "no eval/ood_brand.json" in got and "train/ckpt left" in got and "did not pass: r1: best/, T" in got
    assert not dr.exit_check({})[0] and not dr.exit_check({"exit_check": {"verdict": "PASS"}})[0]  # passed: True only
    monkeypatch.setattr(dr, "dry_run", lambda root, tiny=None: work)
    assert dr.main(["--work", str(tmp_path / "x")]) == 1 and "ood_brand" in capsys.readouterr().err


@pytest.mark.torch
@pytest.mark.skipif(not os.environ.get("LAYA_DRY_RUN_TEST"),
                    reason="slow (tens of minutes on CPU, many subprocesses): set LAYA_DRY_RUN_TEST=1 to run")
def test_dry_run_end_to_end_passes_the_phase_3_exit_check(en_snapshot, tmp_path):
    for module in ("laya_poc.matrix", "laya_poc.variants", "laya_poc.robustness"):
        pytest.importorskip(module, reason="Phase 3 module not implemented yet")
    from conftest import build_tiny_checkpoint

    tiny = build_tiny_checkpoint(en_snapshot, tmp_path / "tiny", init_scale=0.5)
    env = {k: v for k, v in os.environ.items() if k != "LAYA_POC_ROOT"}
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "dry_run_p3_local.py"), "--work", str(tmp_path / "w"),
                          "--tiny-ckpt", str(tiny)], capture_output=True, text=True, env=env, timeout=5400)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    assert (tmp_path / "w" / "laya_poc" / "phase3_artifacts.zip").exists()
