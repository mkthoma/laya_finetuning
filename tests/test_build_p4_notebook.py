"""The generated Phase 4 notebook (tools/build_p4_notebook.py), its cells run with stubbed CLIs, and the CPU dry run."""
from __future__ import annotations

import base64
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
import build_p4_notebook as bp  # noqa: E402
import dry_run_p4_local as dr  # noqa: E402
import e1_cells  # noqa: E402
import e1_helpers as eh  # noqa: E402
import notebook_helpers as nh  # noqa: E402
import p3_commands  # noqa: E402
import p3_helpers as ph  # noqa: E402
import p4_helpers as p4h  # noqa: E402
from p4_commands import (SKIP_OPTIONAL, all_run_names, optional_run_names, p4_commands, row_run_names,  # noqa: E402
                         run_names, variant_dirs)

FAKE_TOKEN = "hf_" + "Xy9" * 10
NO_BUNDLE = bn.Bundle("", "0" * 64, [])
CLI_RE = re.compile(r'"-m", "laya_poc\.(\w+)"(?:, "(\w+)")?')
ARMS = ["B1", "B3", "B4", "B5"]
MERGE = "--runs-root runs/p3_colab/runs/p3 --runs-root runs/p4_colab/runs/p4"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path, sha = bp.build(ROOT, tmp_path_factory.mktemp("nb") / "phase4_baselines.ipynb")
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
    return dr.local_target(tmp_path / "work", tmp_path / "ckpt", tmp_path / "pool.parquet", tmp_path / "enc",
                           tmp_path / "llm")


def test_notebook_is_valid_compiles_and_embeds_the_repo_bundle_without_a_token(built, cfg):
    path, sha, nb = built
    nbformat.validate(nb)
    assert nb.metadata.kernelspec.name == "python3" and nb.metadata.colab.gpuType == cfg["phase4"]["card"]
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


def test_cli_sequence_follows_the_phase_4_session(built):
    _, _, nb = built
    want = [("env_check", ""), ("build_data", ""), ("variants", "c7"), ("variants", "traps"),
            *[("matrix", "run")] * 4, ("matrix", "report")]
    assert CLI_RE.findall(_src(nb)) == want  # no parity, no learning-curve subsets, no Laya training
    assert re.findall(r'"--only", "(\w+)"', _src(nb)) == ARMS
    code = {c.id.split("-", 1)[1]: c.source for c in nb.cells if c.cell_type == "code"}
    assert "RUN_B5 = True" in code["B5"] and f'cmd.append("{SKIP_OPTIONAL}")' in code["B5"]
    assert not [a for a in ARMS[:3] if "RUN_" in code[a]]


def test_connection_disconnect_and_runtime_notes_agree_across_notebook_runbook_and_readme(built):
    _, _, nb = built
    md = _src(nb, "markdown")
    for phrase in ("Select Kernel", "New Colab Server", "G4", "Auto Connect", "Step 1", "disconnect", "RUN_B5",
                   "Remove Server", "exit check", "decision criterion 1", MERGE):
        assert phrase in md, phrase
    runbook = (ROOT / "docs" / "phase4_baselines_runbook.md").read_text(encoding="utf-8")
    docs = {"notebook": md, "runbook": runbook, "README": (ROOT / "README.md").read_text(encoding="utf-8")}
    for name, text in docs.items():
        assert bp.RUNTIME in text and "phase4_baselines" in text, name
    assert MERGE in runbook and "Phase 5" in runbook
    titles = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("###")]
    assert titles[:3] == ["### Step 1 - Setup", "### Step 2 - Install", "### Step 3 - Hugging Face token"]
    assert any(t.startswith("### Step 7 - B1") for t in titles) and "### Step 11 - Report" in "\n".join(titles)
    for arm in ARMS:  # the intro's table estimates every arm group
        assert re.search(rf"\| {arm} \|.*\| ~\d+-\d+ min \|", nb.cells[0].source), arm


def test_build_is_deterministic_and_the_token_variant_stays_local(tmp_path):
    a, _ = bp.build(ROOT, tmp_path / "a.ipynb")
    b, _ = bp.build(ROOT, tmp_path / "b.ipynb")
    assert a.read_bytes() == b.read_bytes()
    local, _ = bp.build(ROOT, tmp_path / "p4.local.ipynb", token=FAKE_TOKEN)
    holders = [c for c in json.loads(local.read_text(encoding="utf-8"))["cells"] if FAKE_TOKEN in "".join(c["source"])]
    assert len(holders) == 1 and "".join(holders[0]["source"]).startswith("# Step 3")
    assert subprocess.run(["git", "check-ignore", "-q", bp.LOCAL_OUT.relative_to(ROOT).as_posix()],
                          cwd=ROOT).returncode == 0
    bp.check_token_out(ROOT, bp.LOCAL_OUT)  # the default local variant is fine; any path git could track is not
    for rel in ("notebooks/phase4_baselines.ipynb", "phase4_baselines.local.ipynb", "notebooks/sub/p4.local.ipynb",
                "docs/p4.local.ipynb"):
        with pytest.raises(ValueError, match="gitignored.*phase4_baselines.local.ipynb"):
            bp.build(tmp_path, tmp_path / rel, token=FAKE_TOKEN)  # a fake root: refused before reading any config
        with pytest.raises(ValueError, match="gitignored"):
            bp.check_token_out(ROOT, ROOT / rel)


def test_committed_notebook_embeds_the_current_code(cfg):
    if not bp.DEFAULT_OUT.exists():
        pytest.skip("notebook not generated yet")
    _, sha, _ = B.bundle(ROOT, be.bundle_patterns(cfg, ROOT))
    code = _src(nbformat.read(str(bp.DEFAULT_OUT), as_version=4))
    assert f'BUNDLE_SHA256 = "{sha}"' in code, "phase4_baselines.ipynb is stale: run tools/build_p4_notebook.py"


def test_earlier_templates_are_adapted_and_the_data_step_is_e1s(cfg):
    cells, e1 = _cells(cfg), e1_cells.code_sources(cfg, be.colab_target(cfg), NO_BUNDLE)
    setup = cells["setup"].source
    assert 'WORK / "runs" / "p4"' in setup and f'CARD = "{cfg["phase4"]["card"]}"' in setup
    assert 'RESULTS = WORK / "results"' in setup and '"runs" / "smoke"' not in setup and '"p3"' not in setup
    for f in (*nh.HELPERS, *eh.E1_HELPERS, *ph.P3_HELPERS, *p4h.P4_HELPERS):
        assert f"def {f.__name__}(" in setup
    assert "phase4_baselines.local.ipynb" in cells["token"].source and 'PRESET_TOKEN = ""' in cells["token"].source
    assert e1["install"] in cells["install"].source and e1["data"] in cells["data"].source


def test_colab_and_local_commands_follow_the_spec(cfg, tmp_path):
    flat = _flat(p4_commands(cfg, bp.colab_target(cfg)))
    p3 = _flat(p3_commands.p3_commands(cfg, bp.colab_target(cfg)))
    assert flat["env"] == p3["env"] and flat["data"] == p3["data"] and flat["variant_data_c7"] == p3["variant_data_c7"]
    assert "--require-gpu" in flat["env"] and _arg(flat["env"], "--expect-card") == "CARD"
    assert variant_dirs(cfg) == ["data_c7", "data_eval"]  # c7 for B1/B3, traps; no learning-curve subsets
    for arm in ARMS:
        args = flat[arm]
        assert args[0] == "run" and _arg(args, "--only") == arm and _arg(args, "--card") == "CARD"
        assert _arg(args, "--device") == "cuda" and "--init" not in args and _arg(args, "--work") == "str(WORK)"
        assert SKIP_OPTIONAL not in args  # added by the B5 cell only when RUN_B5 = False
    assert flat["report"][0] == "report" and _arg(flat["report"], "--runs-root") == "str(RUNS)"
    assert _arg(flat["report"], "--out") == 'str(WORK / "results" / "phase4_report.md")'
    local = _flat(p4_commands(cfg, _local(tmp_path)))
    assert _arg(local["B4"], "--init") == str(tmp_path / "enc") and _arg(local["B5"], "--init") == str(tmp_path / "llm")
    assert "--init" not in local["B1"] and "--init" not in local["B3"] and _arg(local["B4"], "--device") == "cpu"
    assert _arg(local["variant_data_c7"], "--init-tokenizer") == str(tmp_path / "ckpt")  # tiny Laya: tokenizers
    assert local["variant_data_c7"][-2:] == ["laya", "laya_ml"] and "--require-gpu" not in local["env"]


def test_run_names_follow_the_spec_contract_and_the_matrix_plan(cfg):
    assert run_names(cfg, "B1") == [f"fsq-{s}-B1-{k}" for s in ("c10", "c7") for k in ("majority", "prior")]
    assert run_names(cfg, "B3") == ["fsq-c10-B3-tfidf_lr", "fsq-c7-B3-tfidf_lr"]
    assert run_names(cfg, "B4") == [f"fsq-c10-B4-{m}-s{s}" for m in ("modernbert_base", "mmbert_small")
                                    for s in (11, 22, 33)]
    assert run_names(cfg, "B5") == optional_run_names(cfg) == ["fsq-c10-B5-qwen3_4b"]
    assert len(all_run_names(cfg)) == 4 + 2 + 6 + 1 == len(set(all_run_names(cfg)))
    if importlib.util.find_spec("laya_poc.matrix_baselines"):  # the notebook, the dry run and the matrix agree
        from laya_poc.matrix_baselines import expand_phase4
        for c in (cfg, dr.dry_config(cfg)):
            assert [s.name for s in expand_phase4(c)] == all_run_names(c)
            assert [s.name for s in expand_phase4(c) if s.optional] == optional_run_names(c)


@pytest.mark.torch
def test_every_rendered_command_is_accepted_by_its_cli_parser(cfg, tmp_path):
    """A renamed or missing flag fails here instead of on Colab (a CLI not written yet is skipped, named)."""
    pytest.importorskip("torch")
    ns = {"WORK": tmp_path, "DATA": tmp_path / "data", "RUNS": tmp_path / "runs" / "p4", "CARD": "G4"}
    missing = set()
    for target in (bp.colab_target(cfg), _local(tmp_path)):
        cmds = p4_commands(cfg, target)
        cmds.update({f"{a}+skip": (m, [*args, SKIP_OPTIONAL]) for a, (m, args) in cmds.items() if a == "B5"})
        for key, (module, args) in cmds.items():
            if importlib.util.find_spec(module) is None:
                missing.add(module)
                continue
            mod = importlib.import_module(module)
            parser = getattr(mod, "build_parser", None)
            parse = parser().parse_args if parser else getattr(mod, "parse_args", None)
            assert parse, f"{module} has neither build_parser() nor parse_args(): update this test"
            # Expr sources are this project's own rendered path expressions (str(WORK / "x"), CARD), as in the P3 test
            argv = [str(eval(a.src, ns)) if isinstance(a, bn.Expr) else a for a in args]
            try:
                parse(argv)
            except SystemExit:
                pytest.fail(f"{module} rejects the '{key}' command ({target.device}): {argv}")
    if missing:
        pytest.skip(f"CLIs not implemented yet (their commands were not parsed): {sorted(missing)}")


def test_setup_cell_creates_runs_p4_and_never_prints_the_saved_token(cfg, tmp_path):
    cells = {c.key: c for c in bp.build_cells(cfg, _local(tmp_path), be.make_bundle(ROOT, cfg))}
    script = tmp_path / "setup.py"
    script.write_text(cells["setup"].source, encoding="utf-8")
    env = {**{k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HF_TOKEN_PATH", "LAYA_POC_ROOT")},
           "HF_HOME": str(tmp_path / "hf_home")}
    (tmp_path / "hf_home").mkdir()
    (tmp_path / "hf_home" / "token").write_text(FAKE_TOKEN + "\n", encoding="utf-8")
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "work" / "runs" / "p4" / "logs").is_dir() and (tmp_path / "work" / "src" / "laya_poc").is_dir()
    assert FAKE_TOKEN not in res.stdout + res.stderr


def _kernel_ns(work: Path, card: str = "G4") -> dict:  # what Step 1 leaves in the kernel for the later steps
    ns = {"__name__": "__main__", "json": json, "os": os, "shutil": shutil, "subprocess": subprocess, "sys": sys,
          "time": time, "Path": Path, "zipfile": zipfile, "REDO": False}
    helpers = (*nh.HELPERS, *eh.E1_HELPERS, *ph.P3_HELPERS, *p4h.P4_HELPERS)
    exec("\n\n".join(inspect.getsource(f) for f in helpers), ns)
    runs = work / "runs" / "p4"
    ns.update(WORK=work, DATA=work / "data", RUNS=runs, LOGS=runs / "logs", RESULTS=work / "results", CARD=card)
    return ns


def _recorder(ns: dict, calls: list, effect=None, rc: int = 0):
    def run_logged(cmd, log_path, expect_returncode=0, hint=None):
        calls.append((list(cmd), Path(log_path)))
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

    _recorder(ns, calls, build)
    exec(_cells(cfg)["variants"].source, ns)
    assert [c[2:4] for c, _ in calls] == [["laya_poc.variants", "c7"], ["laya_poc.variants", "traps"]]
    assert not partial.exists() and all((ns["WORK"] / d / "build_ok.json").exists() for d in variant_dirs(cfg))
    capsys.readouterr()
    exec(_cells(cfg)["variants"].source, ns)
    assert len(calls) == 2 and capsys.readouterr().out.count("skip:") == 2


def test_arm_step_refuses_while_a_matrix_runs_and_keeps_one_log_per_attempt(cfg, tmp_path, capsys):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    src = _cells(cfg)["B4"].source
    _recorder(ns, calls)
    ns["baseline_pids"] = lambda: [4242]  # e.g. the matrix from before a kernel restart
    with pytest.raises(RuntimeError, match="still running.*4242"):
        exec(src, ns)
    assert calls == []
    ns["baseline_pids"] = lambda: []
    (ns["RESULTS"]).mkdir(parents=True)
    (ns["RESULTS"] / "runs.csv").write_text("run_name,arm,val_macro_f1\nfsq-c10-B4-mmbert_small-s11,B4,0.5461234\n"
                                            "fsq-c10-B3-tfidf_lr,B3,0.49\n", encoding="utf-8")
    exec(src, ns)
    assert "fsq-c10-B4-mmbert_small-s11" in (out := capsys.readouterr().out) and "0.5461" in out
    assert "Step 9 - B4:" in out and "tfidf_lr" not in out
    _recorder(ns, calls, rc=1)  # e.g. one run failed: its rows are shown first, then the step fails
    with pytest.raises(RuntimeError, match=r"B4: matrix run exited with 1; the 'matrix:' lines in .*09_B4_2\.log"):
        exec(src, ns)  # a re-run (after a fix or a disconnect) keeps the earlier log
    assert [log.name for _, log in calls] == ["09_B4.log", "09_B4_2.log"] and _arg(calls[0][0], "--card") == "G4"
    _recorder(ns, calls)
    for run_b5 in (True, False):  # the optional B5 passes --skip-optional only when switched off
        exec(_cells(cfg)["B5"].source.replace("RUN_B5 = True", f"RUN_B5 = {run_b5}"), ns)
        assert _arg(calls[-1][0], "--only") == "B5" and (SKIP_OPTIONAL in calls[-1][0]) is not run_b5


def test_report_step_shows_the_report_and_the_phase_4_verdict_before_failing_on_its_exit_code(cfg, tmp_path, capsys):
    ns, calls, shown = _kernel_ns(tmp_path / "w"), [], []
    p4 = {"verdict": "PASS", "passed": True, "complete": 12, "total": 13, "skipped_optional": ["fsq-c10-B5-qwen3_4b"]}

    def report(cmd):
        nh.write_json(ns["RESULTS"] / "phase4_report.md", "report")
        nh.write_json(ns["RESULTS"] / "phase4_report.json", {"exit_check": {"passed": False}, "phase4_exit_check": p4})

    _recorder(ns, calls, report, rc=1)
    ns["show_markdown"] = shown.append
    with pytest.raises(RuntimeError, match="matrix report exited with 1"):
        exec(_cells(cfg)["report"].source, ns)
    assert shown == [ns["RESULTS"] / "phase4_report.md"] and calls[0][1].name == "11_report.log"
    assert ("Phase 4 exit check: PASS (12/13 baseline runs complete; optional, not run: fsq-c10-B5-qwen3_4b)"
            in capsys.readouterr().out)
    assert "not found" in p4h.phase4_verdict(tmp_path / "missing.json")


def test_archive_has_no_weights_and_no_fsq_rows(cfg, tmp_path):
    ns = _kernel_ns(tmp_path / "w")
    work, b4, b3 = ns["WORK"], "runs/p4/fsq-c10-B4-mmbert_small-s11", "runs/p4/fsq-c10-B3-tfidf_lr"
    for rel in ("data/NOTICE_FSQ.txt", "data/data_report.json", "data/val.jsonl", "data/train.jsonl",
                "data/split_val.parquet", "data/trap_candidates.csv", "data_eval/trap_candidates.jsonl",
                "data_c7/val.jsonl", "data_c7/variant.json", "results/phase4_report.json", "results/runs.csv",
                f"{b3}/calibration.json", f"{b3}/preds/no_gate/stripped_test.jsonl", "runs/p4/logs/09_B4.log",
                *(f"{b4}/{r}" for r in ("done.json", "eval/val.json", "preds/val.jsonl", "train/summary.json",
                                        "train/best/config.json", "train/best/tokenizer.json", "logs/x.log"))):
        nh.write_json(work / rel, {})
    (work / b4 / "train" / "best" / "model.safetensors").write_bytes(b"weights")
    exec(_cells(cfg)["archive"].source, ns)
    with zipfile.ZipFile(work / "phase4_artifacts.zip") as zf:
        names = set(zf.namelist())
    assert {f"{b4}/preds/val.jsonl", f"{b4}/train/summary.json", f"{b3}/preds/no_gate/stripped_test.jsonl",
            "results/runs.csv", "data/NOTICE_FSQ.txt", "data_c7/variant.json", "runs/p4/logs/09_B4.log"} <= names
    assert not [n for n in names if n.startswith("data") and n.endswith((".jsonl", ".parquet", ".csv"))]  # FSQ rows
    assert not any("/best/" in n or n.endswith(".safetensors") for n in names)


def test_baseline_pids_and_the_card_note(tmp_path, monkeypatch):
    table = {"laya_poc.matrix": [9], "laya_poc.small_encoder": [3], "laya_poc.llm_baseline": [3, 7]}
    monkeypatch.setattr(p4h, "trainer_pids", lambda module: table.get(module, []))
    assert p4h.baseline_pids() == [3, 7, 9]
    env = tmp_path / "env.json"
    assert "no GPU" in p4h.p4_card_note(env, "G4")  # missing env.json
    nh.write_json(env, {"gpu_name": "NVIDIA L4", "card": "L4"})
    assert "B5" in p4h.p4_card_note(env, "G4") and p4h.p4_card_note(env, "L4") is p4h.p4_card_note(env, "CPU") is None


def test_dry_config_is_valid_tiny_and_covers_every_baseline_kind(cfg):
    validate_config(dcfg := dr.dry_config(cfg))
    p4 = dcfg["phase4"]
    assert dcfg["data"]["train_size"] <= 400 and dcfg["data"]["frozen_manifest"] is None
    assert p4["small_encoder_train"]["epochs"] <= dcfg["train"]["epochs"]  # B4 trains on the built train_e* files
    assert {a["kind"] for a in p4["arms"]} == {a["kind"] for a in cfg["phase4"]["arms"]}
    assert {a["model"] for a in p4["arms"] if a["kind"] == "small_encoder"} == set(p4["small_encoders"])
    assert optional_run_names(dcfg) == ["fsq-c10-B5-qwen3_4b"] and len(run_names(dcfg, "B4")) == 3
    assert cfg["phase4"] == load_config(ROOT / "config.yaml")["phase4"]  # the project config is untouched


def _fake_finished_work(dcfg: dict, work: Path) -> Path:
    for arm in dcfg["phase4"]["arms"]:
        for name, rel in ((n, r) for n in row_run_names(arm) for r in dr._needs(arm, dcfg["phase3"]["eval_splits"])):
            nh.write_json(work / "runs" / "p4" / name / rel, {})
    nh.write_json(work / "results" / "phase4_report.json", {"phase4_exit_check": {"passed": True}})
    rows = "".join(f"{n}{',' * (len(dr.RUNS_CSV_COLUMNS) - 1)}\n" for n in all_run_names(dcfg))
    (work / "results" / "runs.csv").write_text(",".join(dr.RUNS_CSV_COLUMNS) + "\n" + rows, encoding="utf-8")
    return work


def test_plumbing_failures_name_whatever_the_run_layout_lacks(cfg, tmp_path, monkeypatch, capsys):
    dcfg = dr.dry_config(cfg)
    work = _fake_finished_work(dcfg, tmp_path)
    assert dr.plumbing_failures(dcfg, work) == []
    monkeypatch.setattr(dr, "dry_run", lambda root, *a: work)
    assert dr.main(["--work", str(tmp_path / "x")]) == 0
    for rel in ("preds/ood_brand.jsonl", "order_invariance.json"):
        (work / "runs" / "p4" / "fsq-c10-B4-modernbert_base-s22" / rel).unlink()
    nh.write_json(work / "results" / "phase4_report.json", {"phase4_exit_check": {"passed": False, "runs": [
        {"run_name": "r1", "passed": False, "missing": ["preds/val.jsonl"]}, {"run_name": "r2", "passed": True}]}})
    got = "\n".join(dr.plumbing_failures(dcfg, work))
    assert "no preds/ood_brand.jsonl" in got and "no order_invariance.json" in got
    assert "did not pass: r1: preds/val.jsonl" in got and "fsq-c10-B1" not in got
    assert not dr.phase4_exit({})[0] and not dr.phase4_exit({"phase4_exit_check": {"verdict": "PASS"}})[0]
    assert dr.main(["--work", str(tmp_path / "x")]) == 1 and "ood_brand" in capsys.readouterr().err


@pytest.mark.torch
@pytest.mark.skipif(not os.environ.get("LAYA_DRY_RUN_TEST"),
                    reason="slow (several minutes on CPU, many subprocesses): set LAYA_DRY_RUN_TEST=1 to run")
def test_dry_run_end_to_end_passes_the_phase_4_exit_check(en_snapshot, tmp_path):
    for module in ("baseline_runs", "small_encoder", "llm_baseline", "matrix_baselines"):
        pytest.importorskip(f"laya_poc.{module}", reason="Phase 4 module not implemented yet")
    from conftest import build_tiny_checkpoint

    tiny = build_tiny_checkpoint(en_snapshot, tmp_path / "tiny", init_scale=0.5)
    env = {k: v for k, v in os.environ.items() if k != "LAYA_POC_ROOT"}
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "dry_run_p4_local.py"), "--work", str(tmp_path / "w"),
                          "--tiny-ckpt", str(tiny)], capture_output=True, text=True, env=env, timeout=3600)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    assert (tmp_path / "w" / "laya_poc" / "phase4_artifacts.zip").exists()
