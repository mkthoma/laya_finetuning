"""The generated smoke-test notebook and the local CPU dry run that executes the same cells."""
from __future__ import annotations

import base64
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

from laya_poc import bundle as B  # noqa: E402
from laya_poc import smoke_report as SR  # noqa: E402
from laya_poc.config import load_config, validate_config, with_overrides  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_notebook as bn  # noqa: E402
import dry_run_local as dr  # noqa: E402

TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
FAKE_TOKEN = "hf_" + "Zq7" * 10
SPEC_ORDER = ["env_check", "build_data", "zeroshot", "parity", "parity", "train_single", "train_single",
              "train_single", "export_check", "smoke_report"]
NO_BUNDLE = bn.Bundle("", "0" * 64, [])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("nb") / "smoke_test.ipynb"
    path, sha = bn.build(ROOT, out)
    return path, sha, nbformat.read(str(path), as_version=4)


def _code(nb) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _markdown(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")


def _local_target(tmp_path: Path) -> bn.Target:
    return bn.Target(work=str(tmp_path / "work"), device="cpu", require_gpu=False, init=str(tmp_path / "ckpt"),
                     pool=str(tmp_path / "pool.parquet"), data_models=("laya",), micro_batch=2, effective_batch=4,
                     card="CPU")


def _flat(cmds: dict) -> dict[str, list[str]]:
    return {k: [a.src if isinstance(a, bn.Expr) else a for a in args] for k, (_, args) in cmds.items()}


def _cells(cfg: dict, target: bn.Target | None = None) -> dict[str, bn.Cell]:
    return {c.key: c for c in bn.build_cells(cfg, target or bn.Target(), NO_BUNDLE)}


# ---------------------------------------------------------------- notebook structure

def test_notebook_is_valid_nbformat_with_python3_kernel_and_no_outputs(built):
    _, _, nb = built
    nbformat.validate(nb)
    assert nb.metadata.kernelspec.name == "python3"
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None


def test_every_code_cell_compiles(built):
    _, _, nb = built
    for i, src in enumerate(_code(nb)):
        compile(src, f"<cell {i}>", "exec")


def test_no_shell_or_line_magics(built):
    _, _, nb = built
    for src in _code(nb):
        for line in src.splitlines():
            assert not line.lstrip().startswith(("!", "%")), line


def test_embedded_bundle_matches_repo_bundle(built):
    _, sha, nb = built
    b64, want_sha, members = B.bundle(ROOT)
    text = "\n".join(_code(nb))
    assert sha == want_sha
    assert want_sha in text and b64 in text
    assert want_sha in _markdown(nb)
    assert "src/laya_poc/bundle.py" in members


def test_no_forbidden_strings_in_notebook(built):
    path, _, _ = built
    text = path.read_text(encoding="utf-8")
    for bad in ("userdata", "files.upload", "eval_js", " -U"):
        assert bad not in text, bad
    assert not TOKEN_RE.search(text)


def test_decoded_bundle_contains_no_token(built):
    _, _, nb = built
    b64 = re.search(r'BUNDLE_B64 = "([A-Za-z0-9+/=]+)"', "\n".join(_code(nb))).group(1)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as zf:
        for name in zf.namelist():
            assert not TOKEN_RE.search(zf.read(name).decode("utf-8", "replace")), name


def test_token_is_read_with_getpass_and_never_printed(built):
    _, _, nb = built
    token_cell = next(s for s in _code(nb) if "getpass.getpass(" in s)
    assert "login(token=token)" in token_cell
    assert "print(token" not in token_cell and "{token}" not in token_cell


def test_t4_connection_instructions_present(built):
    _, _, nb = built
    md = _markdown(nb)
    for phrase in ("Select Kernel", "New Colab Server", "GPU", "T4", "Auto Connect", "Step 1"):
        assert phrase in md, phrase


def test_cli_sequence_matches_spec(built):
    _, _, nb = built
    mods = re.findall(r'"-m", "laya_poc\.(\w+)"', "\n".join(_code(nb)))
    assert mods == SPEC_ORDER


def test_committed_notebook_embeds_the_current_code():
    committed = ROOT / "notebooks" / "smoke_test.ipynb"
    if not committed.exists():
        pytest.skip("notebook not generated yet")
    _, sha, _ = B.bundle(ROOT)
    code = "\n".join(_code(nbformat.read(str(committed), as_version=4)))
    assert f'BUNDLE_SHA256 = "{sha}"' in code, "notebooks/smoke_test.ipynb is stale: run `python tools/build_notebook.py`"


def test_build_is_deterministic(tmp_path):
    a, _ = bn.build(ROOT, tmp_path / "a.ipynb")
    b, _ = bn.build(ROOT, tmp_path / "b.ipynb")
    assert a.read_bytes() == b.read_bytes()


def test_build_refuses_a_bundle_that_contains_a_token(tmp_path):
    (tmp_path / "src" / "laya_poc").mkdir(parents=True)
    (tmp_path / "src" / "laya_poc" / "x.py").write_text("T = 'hf_" + "a" * 30 + "'\n")
    (tmp_path / "config.yaml").write_text((ROOT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    with pytest.raises(ValueError, match="token"):
        bn.build(tmp_path, tmp_path / "nb.ipynb")


def test_tool_modules_stay_under_400_lines():
    for path in sorted((ROOT / "tools").glob("*.py")):
        n = len(path.read_text(encoding="utf-8").splitlines())
        assert n < 400, f"{path.name}: {n} lines (spec §0: every module < 400 lines)"


def test_resume_step_shown_in_the_notebook_matches_the_report(cfg):
    """kill 120 / ckpt 40: the kill fires before step 120's own checkpoint, so the resume is from 80."""
    c = with_overrides(cfg, {"smoke.kill_at_micro_step": 120})
    want = SR.expected_resume_step(120, c["smoke"]["ckpt_every_micro_steps"])
    md = "\n".join(cell.source for cell in _cells(c).values() if cell.kind == "markdown")
    assert want == 80
    assert f"micro-step-{want} checkpoint" in md and f"(micro-step {want})" in md
    assert "micro-step-120 checkpoint" not in md


# ---------------------------------------------------------------- commands

def test_colab_commands_follow_config_and_spec(cfg):
    sm = cfg["smoke"]
    flat = _flat(bn.commands(cfg, bn.Target()))
    assert "--require-gpu" in flat["env"] and "T4" in flat["env"]
    assert "--allow-cpu" not in flat["parity_laya"] and "--init" not in flat["control"]
    ctl = flat["control"]
    assert ctl[ctl.index("--max-micro-steps") + 1] == str(sm["micro_steps"])
    assert ctl[ctl.index("--epochs") + 1] == str(sm["epochs"])
    assert {"--initial-eval", "--final-eval", "--save-final"} <= set(ctl)
    crash = flat["crash"]
    assert crash[crash.index("--crash-at-micro-step") + 1] == str(sm["kill_at_micro_step"])
    assert flat["zeroshot"][flat["zeroshot"].index("--n") + 1] == str(sm["zero_shot_n"])
    assert flat["parity_laya_ml"][flat["parity_laya_ml"].index("--n") + 1] == str(sm["parity_n"])


def _without(args: list[str], *flags: str) -> list[str]:
    """args minus each flag and its value (the flags given all take exactly one value)."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in flags:
            skip = True
        else:
            out.append(a)
    return out


def _ckpt_flags(args: list[str]) -> tuple[str | None, str | None]:
    return tuple(args[args.index(f) + 1] if f in args else None
                 for f in ("--ckpt-every-micro-steps", "--ckpt-every-min"))


@pytest.mark.parametrize("local", [False, True])
def test_only_the_crash_run_writes_resumable_checkpoints(cfg, tmp_path, local):
    """Nothing reads control/ckpt or the resumed run's later checkpoints (about 5 GB each on the T4); the crash
    run's own checkpoints every ckpt_every_micro_steps are what Step 10 resumes from."""
    flat = _flat(bn.commands(cfg, _local_target(tmp_path) if local else bn.Target()))
    assert _ckpt_flags(flat["control"]) == ("0", "0") and _ckpt_flags(flat["resume"]) == ("0", "0")
    assert _ckpt_flags(flat["crash"])[0] == str(cfg["smoke"]["ckpt_every_micro_steps"])
    assert "--crash-at-micro-step" not in flat["resume"]
    assert "--initial-eval" in flat["control"]  # val_ce at opt 0, graded against the final eval
    ckpt = ("--ckpt-every-micro-steps", "--ckpt-every-min", "--run-dir")
    shared = _without(flat["resume"], *ckpt)
    assert _without(flat["crash"], *ckpt, "--crash-at-micro-step") == shared  # an exact resume needs equal settings
    assert _without(flat["control"], *ckpt)[:len(shared)] == shared


def test_training_runs_pin_the_t4_profile_on_any_colab_gpu(cfg):
    """With --card auto an L4/A10G gets MB 32 x ACC 1 (the run ends near micro-step 67, before the crash) and an
    A100/H100 has no profile at all; the smoke plan (ckpt every 40, kill at 122) assumes the T4's MB 8 x ACC 4."""
    flat = _flat(bn.commands(cfg, bn.Target()))
    for key in ("control", "crash", "resume"):
        assert flat[key][flat[key].index("--card") + 1] == "T4", key


def test_local_target_adds_cpu_flags(cfg, tmp_path):
    target = _local_target(tmp_path)
    cmds = _flat(bn.commands(cfg, target))
    assert "--require-gpu" not in cmds["env"]
    assert "--allow-cpu" in cmds["parity_laya"] and target.init in cmds["parity_laya"]
    assert cmds["data"][cmds["data"].index("--pool") + 1] == target.pool
    assert cmds["data"][cmds["data"].index("--init-tokenizer") + 1] == target.init
    assert cmds["control"][cmds["control"].index("--device") + 1] == "cpu"
    assert cmds["control"][cmds["control"].index("--card") + 1] == "CPU"
    assert cmds["control"][cmds["control"].index("--micro-batch") + 1] == "2"


def test_render_cmd_is_valid_python_that_builds_the_argv(tmp_path):
    src = bn.render_cmd("laya_poc.x", ["--out", bn.Expr('str(RUNS / "a.json")'), "--n", "20", "--flag"])
    ns = {"sys": sys, "RUNS": tmp_path}
    assert eval(src, ns) == [sys.executable, "-m", "laya_poc.x", "--out", str(tmp_path / "a.json"), "--n", "20",
                             "--flag"]


@pytest.mark.torch
def test_every_rendered_command_is_accepted_by_its_cli_parser(cfg, tmp_path):
    """Catches a renamed or removed flag here instead of on Colab (the dry run is opt-in)."""
    pytest.importorskip("torch")
    pytest.importorskip("laya")
    ns = {"WORK": tmp_path, "DATA": tmp_path / "data", "RUNS": tmp_path / "runs" / "smoke"}
    for target in (bn.Target(), _local_target(tmp_path)):
        for key, (module, args) in bn.commands(cfg, target).items():
            argv = [str(eval(a.src, ns)) if isinstance(a, bn.Expr) else a for a in args]
            mod = importlib.import_module(module)
            parse = mod.build_parser().parse_args if hasattr(mod, "build_parser") else mod.parse_args
            try:
                parse(argv)
            except SystemExit:
                pytest.fail(f"{module} rejects the '{key}' command ({target.device}): {argv}")


def test_setup_cell_unpacks_bundle_writes_sentinel_and_never_prints_the_saved_token(cfg, tmp_path):
    target = _local_target(tmp_path)
    cells = {c.key: c for c in bn.build_cells(cfg, target, bn.make_bundle(ROOT))}
    script = tmp_path / "setup.py"
    script.write_text(cells["setup"].source, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HF_TOKEN_PATH", "LAYA_POC_ROOT")}
    env["HF_HOME"] = str(tmp_path / "hf_home")
    (tmp_path / "hf_home").mkdir()
    (tmp_path / "hf_home" / "token").write_text(FAKE_TOKEN + "\n", encoding="utf-8")
    runs = [subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
            for _ in range(2)]
    assert [r.returncode for r in runs] == [0, 0], runs[0].stderr + runs[1].stderr
    work = Path(target.work)
    assert (work / "src" / "laya_poc" / "bundle.py").exists() and (work / "config.yaml").exists()
    assert (work / "sentinel.json").exists()
    assert "sentinel created" in runs[0].stdout and "survived" in runs[1].stdout
    assert (work / "runs" / "smoke" / "logs").is_dir()
    assert "restored from the token file" in runs[0].stdout
    for r in runs:
        assert FAKE_TOKEN not in r.stdout + r.stderr


# ---------------------------------------------------------------- estimates and step notes

def test_runtime_estimates_agree_across_notebook_runbook_and_readme(built):
    _, _, nb = built
    docs = {"notebook": _markdown(nb), "runbook": (ROOT / "docs" / "smoke_test_runbook.md").read_text(encoding="utf-8"),
            "README": (ROOT / "README.md").read_text(encoding="utf-8")}
    for name, text in docs.items():
        assert "60-90 min" in text and "45-75" not in text, name
        assert "25-35 min" in text, name  # Step 7: the CPU fp32 parity reference
    step7 = next(c.source for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("### Step 7"))
    assert "25-35 min" in step7 and "10 min" not in step7


def test_step_notes_describe_the_checkpoint_plan(cfg):
    md = {k: c.source for k, c in _cells(cfg).items() if c.kind == "markdown"}
    assert "same command without the crash flag" not in md["md_resume"] and "no checkpoints" in md["md_resume"]
    assert "initial eval" in md["md_control"] and "no resumable checkpoints" in md["md_control"]
    assert f"every {cfg['smoke']['ckpt_every_micro_steps']}" in md["md_crash"]


# ---------------------------------------------------------------- dry run pieces

def test_dry_config_is_valid_and_tiny(cfg):
    dcfg = dr.dry_config(cfg)
    validate_config(dcfg)
    sm = dcfg["smoke"]
    assert sm["micro_steps"] <= 32 and sm["kill_at_micro_step"] < sm["micro_steps"]
    assert cfg["smoke"]["micro_steps"] == load_config(ROOT / "config.yaml")["smoke"]["micro_steps"]  # not mutated


def test_synthetic_pool_matches_extract_schema_and_splits(cfg, tmp_path):
    import pandas as pd
    from laya_poc import labels as L
    from laya_poc.config import split_spec
    from laya_poc.splits import make_splits

    path = dr.write_synthetic_pool(tmp_path / "pool.parquet")
    pool = pd.read_parquet(path)
    assert set(dr.POOL_COLUMNS) <= set(pool.columns)
    assert set(pool["label"]) <= set(L.L1_TO_KEY) and (pool["n_l1"] == 1).all()
    fb = pool["facebook_id"].dropna()
    assert len(fb) and all(isinstance(v, str) for v in fb)
    dcfg = dr.dry_config(cfg)
    spec = split_spec(dcfg, smoke=True)
    splits, _ = make_splits(pool, spec)
    assert len(splits["val"]) == spec.val_size and len(splits["test_id"]) == spec.test_size
    assert len(splits["train"]) == spec.train_size


def test_read_verdict_handles_json_and_markdown(tmp_path):
    assert dr.read_verdict(tmp_path) == ("UNKNOWN", [])
    (tmp_path / "smoke_report.md").write_text("## Overall verdict: **FAIL**\n")
    assert dr.read_verdict(tmp_path)[0] == "FAIL"
    (tmp_path / "smoke_report.json").write_text(json.dumps(
        {"verdict": "fail", "criteria": [{"name": "parity", "passed": True}, {"name": "vram", "passed": False}]}))
    assert dr.read_verdict(tmp_path) == ("FAIL", ["vram"])


_PIPELINE = ["env_check", "build_data", "zeroshot", "parity", "train_single", "export_check", "smoke_report"]
_PLUMBING = (SR.CRASH, SR.RESUME_POINT, SR.RESUME_LOSS, SR.NUMERICS)  # must PASS even on the tiny CPU model


def _events(path: Path, name: str) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == name:
            out.append(event)
    return out


@pytest.mark.torch
@pytest.mark.skipif(not os.environ.get("LAYA_DRY_RUN_TEST"),
                    reason="slow (~4-5 min on CPU, 11 subprocesses): set LAYA_DRY_RUN_TEST=1 to run")
def test_dry_run_end_to_end(tiny_ckpt_dir, tmp_path):
    missing = [m for m in _PIPELINE if importlib.util.find_spec(f"laya_poc.{m}") is None]
    if missing:
        pytest.skip(f"pipeline modules not implemented yet: {missing}")
    env = {k: v for k, v in os.environ.items() if k != "LAYA_POC_ROOT"}
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "dry_run_local.py"), "--work", str(tmp_path / "w"),
                          "--tiny-ckpt", str(tiny_ckpt_dir)], capture_output=True, text=True, env=env, timeout=900)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    runs = tmp_path / "w" / "laya_poc" / "runs" / "smoke"
    assert (runs / "smoke_report.md").exists() and "verdict:" in res.stdout
    assert json.loads((runs / "crash_exit.json").read_text())["returncode"] in (-9, 137)
    report = json.loads((runs / "smoke_report.json").read_text(encoding="utf-8"))
    passed = {c["name"]: c for c in report["criteria"]}
    for name in _PLUMBING:
        assert passed[name]["passed"] is True, passed[name]
    sm = dr.dry_config(load_config(ROOT / "config.yaml"))["smoke"]
    want = SR.expected_resume_step(sm["kill_at_micro_step"], sm["ckpt_every_micro_steps"])
    assert [e["from_micro_step"] for e in _events(runs / "resumed" / "log.jsonl", "resumed")] == [want]
    assert [e["opt_step"] for e in _events(runs / "control" / "log.jsonl", "eval") if e.get("initial")] == [0]
    assert not _events(runs / "control" / "log.jsonl", "ckpt") and not list(runs.glob("control/ckpt/*"))
    # only the crash run checkpoints: the resumed run (after the "resumed" event) writes none
    log = [json.loads(x) for x in (runs / "resumed" / "log.jsonl").read_text(encoding="utf-8").splitlines()
           if x.strip().startswith("{") and x.strip().endswith("}")]
    first_resume = next(i for i, e in enumerate(log) if e.get("event") == "resumed")
    assert not [e for e in log[first_resume:] if e.get("event") == "ckpt"]
    with zipfile.ZipFile(runs.parent.parent / "smoke_artifacts.zip") as zf:
        assert "data/NOTICE_FSQ.txt" in zf.namelist()
    assert (runs / "control" / "final" / "NOTICE.md").exists()
