"""The Phase 7 GPU benchmark notebook (tools/build_p7_gpu_notebook.py): a valid Colab GPU notebook that embeds the
current code bundle and no token, its CLI sequence, every notebook and runbook command against the real parsers, the
env / bench / archive cells run with stubbed CLIs, the --with-token guard, and the runbook."""
from __future__ import annotations

import base64
import importlib
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

from laya_poc import bench_gpu  # noqa: E402
from laya_poc import bundle as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_e1_notebook as be  # noqa: E402
import build_notebook as bn  # noqa: E402
import build_p7_gpu_notebook as bp  # noqa: E402
import e1_cells  # noqa: E402
import e1_commands  # noqa: E402
import e1_helpers as eh  # noqa: E402
import notebook_helpers as nh  # noqa: E402
import p3_helpers as ph  # noqa: E402
import p7_helpers as p7h  # noqa: E402
import p7_md  # noqa: E402
from p7_cells import ARCHIVE_NAME  # noqa: E402
from p7_commands import BENCH_OUT, P7_COLAB, bench_models, p7_commands, runbook_commands, shell  # noqa: E402

FAKE_TOKEN = "hf_" + "Qz7" * 10
NO_BUNDLE = bn.Bundle("", "0" * 64, [])
CLI_RE = re.compile(r'"-m", "laya_poc\.(\w+)"')


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path, sha = bp.build(ROOT, tmp_path_factory.mktemp("nb") / "phase7_gpu_bench.ipynb")
    return path, sha, nbformat.read(str(path), as_version=4)


def _src(nb, kind: str = "code") -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == kind)


def _cells(cfg: dict) -> dict[str, bn.Cell]:
    return {c.key: c for c in bp.build_cells(cfg, bp.colab_target(cfg), NO_BUNDLE)}


def _flat(args: list) -> list[str]:
    return [a.src if isinstance(a, bn.Expr) else a for a in args]


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_notebook_is_a_valid_gpu_notebook_that_embeds_the_bundle_without_a_token(built, cfg):
    path, sha, nb = built
    nbformat.validate(nb)
    assert nb.metadata.accelerator == "GPU" and nb.metadata.colab.gpuType == "T4"  # a GPU runtime, T4 by default
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, f"<cell {i}>", "exec")
            assert not any(line.lstrip().startswith(("!", "%")) for line in cell.source.splitlines())
    b64, want, members = B.bundle(ROOT, be.bundle_patterns(cfg, ROOT))
    assert sha == want and b64 in _src(nb) and want in _src(nb, "markdown")
    assert cfg["data"]["frozen_manifest"] in members and "src/laya_poc/bench_gpu.py" in members
    text = path.read_text(encoding="utf-8")
    assert not [bad for bad in ("userdata", "files.upload", "eval_js", " -U") if bad in text]
    assert not bn.TOKEN_RE.search(text)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as zf:
        assert not any(bn.TOKEN_RE.search(zf.read(n).decode("utf-8", "replace")) for n in zf.namelist())


def test_cli_sequence_and_commands_run_the_gpu_benchmark_on_the_frozen_test_id(built, cfg):
    _, _, nb = built
    assert CLI_RE.findall(_src(nb)) == ["env_check", "build_data", "bench_gpu"]  # no training, no ONNX
    cmds = {k: _flat(args) for k, (_, args) in p7_commands(cfg, bp.colab_target(cfg)).items()}
    assert cmds["env"] == ["--out", 'str(RUNS / "env.json")', "--require-gpu"]  # no card expected: T4 or G4
    assert cmds["data"] == _flat(e1_commands.e1_commands(cfg, bp.colab_target(cfg))["data"][1])
    bench = cmds["bench"]
    assert bench[:1 + len(bench_models(cfg))] == ["--models", *cfg["phase5"]["bench"]["models"]]
    assert _arg(bench, "--rows") == 'str(DATA / "test_id.jsonl")' and _arg(bench, "--out") == "str(OUT)"
    assert _arg(bench, "--latency-n") == str(bench_gpu.LATENCY_N) and _arg(bench, "--batch-n") == str(bench_gpu.BATCH_N)
    assert bench[bench.index("--batch-sizes") + 1:bench.index("--warmup")] == [str(b) for b in bench_gpu.BATCH_SIZES]
    assert _arg(bench, "--warmup") == str(bench_gpu.WARMUP) and _arg(bench, "--device") == "cuda"
    assert "--ckpt" not in bench  # the pinned Hub checkpoints (the CLI default): latency needs no fine-tuned weights
    titles = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("###")]
    assert titles == ["### Step 1 - Setup", "### Step 2 - Install", "### Step 3 - Hugging Face token",
                      "### Step 4 - Environment check", "### Step 5 - Full data (frozen)", "### Step 6 - GPU benchmark",
                      "### Step 7 - Archive", "### Finish"]
    cells, e1 = _cells(cfg), e1_cells.code_sources(cfg, be.colab_target(cfg), NO_BUNDLE)
    setup = cells["setup"].source
    assert 'WORK / "runs" / "p7"' in setup and '"runs" / "smoke"' not in setup and '"LAYA_CUDA_AMP": "fp16"' in setup
    for f in (*nh.HELPERS, *eh.E1_HELPERS, *ph.P3_HELPERS, *p7h.P7_HELPERS):
        assert f"def {f.__name__}(" in setup
    assert "phase7_gpu_bench.local.ipynb" in cells["token"].source and 'PRESET_TOKEN = ""' in cells["token"].source
    assert e1["data"] in cells["data"].source and e1["install"].rstrip() in cells["install"].source
    assert "ONNX = " not in cells["install"].source  # the E1 install as is: no ONNX packages on the GPU runtime


def test_build_is_deterministic_and_the_token_variant_stays_local(tmp_path):
    a, b = (bp.build(ROOT, tmp_path / name)[0] for name in ("a.ipynb", "b.ipynb"))
    assert a.read_bytes() == b.read_bytes()
    local, _ = bp.build(ROOT, tmp_path / "p7.local.ipynb", token=FAKE_TOKEN)
    holders = [c for c in json.loads(local.read_text(encoding="utf-8"))["cells"] if FAKE_TOKEN in "".join(c["source"])]
    assert len(holders) == 1 and "".join(holders[0]["source"]).startswith("# Step 3")
    assert subprocess.run(["git", "check-ignore", "-q", bp.LOCAL_OUT.relative_to(ROOT).as_posix()],
                          cwd=ROOT).returncode == 0
    bp.check_token_out(ROOT, bp.LOCAL_OUT)
    for rel in ("notebooks/phase7_gpu_bench.ipynb", "phase7_gpu_bench.local.ipynb", "notebooks/x/p7.local.ipynb",
                "docs/p7.local.ipynb"):
        with pytest.raises(ValueError, match="gitignored.*phase7_gpu_bench.local.ipynb"):
            bp.build(tmp_path, tmp_path / rel, token=FAKE_TOKEN)
        with pytest.raises(ValueError, match="gitignored"):
            bp.check_token_out(ROOT, ROOT / rel)


def test_committed_notebook_embeds_the_current_code(cfg):
    if not bp.DEFAULT_OUT.exists():
        pytest.skip("notebook not generated yet")
    _, sha, _ = B.bundle(ROOT, be.bundle_patterns(cfg, ROOT))
    code = _src(nbformat.read(str(bp.DEFAULT_OUT), as_version=4))
    assert f'BUNDLE_SHA256 = "{sha}"' in code, "phase7_gpu_bench.ipynb is stale: run tools/build_p7_gpu_notebook.py"


def test_every_notebook_and_runbook_command_is_accepted_by_its_cli_parser(cfg, tmp_path):
    runs = tmp_path / "runs" / "p7"
    ns = {"WORK": tmp_path, "DATA": tmp_path / "data", "RUNS": runs, "OUT": runs / BENCH_OUT}
    # Expr sources are this project's own rendered path expressions (str(DATA / "x")), as in the P3-P5 tests
    commands = [(f"colab {k}", m, [str(eval(a.src, ns)) if isinstance(a, bn.Expr) else a for a in args])
                for k, (m, args) in p7_commands(cfg, bp.colab_target(cfg)).items()]
    commands += [(k, m, list(a)) for k, (m, a) in runbook_commands(cfg).items()]
    for key, module, argv in commands:
        mod = importlib.import_module(module)
        parse = mod.build_parser().parse_args if hasattr(mod, "build_parser") else getattr(mod, "parse_args", None)
        assert parse, f"{module} has neither build_parser() nor parse_args(): update this test"
        try:
            args = parse(argv)
        except SystemExit:
            pytest.fail(f"{module} rejects the '{key}' command: {argv}")
        if module == "laya_poc.bench_gpu":
            bench_gpu.make_plan(args, cfg)  # models known, sizes valid


def test_runbook_and_notebook_markdown_agree(built, cfg):
    md = _src(built[2], "markdown")
    for phrase in ("Select Kernel", "New Colab Server", "GPU", "T4", "G4", "Auto Connect", "Step 1", "disconnect",
                   "Remove Server", "fp16", "fresh process", bp.RUNTIME, ARCHIVE_NAME, "runs/p7_colab"):
        assert phrase in md, phrase
    runbook = (ROOT / "docs" / "phase7_gpu_runbook.md").read_text(encoding="utf-8")
    for key, (module, args) in runbook_commands(cfg).items():
        assert shell(module, args) in runbook, f"runbook lacks the {key} command: {shell(module, args)}"
    for word in ("build_p7_gpu_notebook.py", "--with-token", "phase7_gpu_bench.local.ipynb", bp.RUNTIME, "T4", "G4",
                 ARCHIVE_NAME, f"{P7_COLAB}/{BENCH_OUT}", "--batch-sizes", "OOM", "Change runtime type", "p50",
                 "p95", "rec/s", "peak VRAM"):
        assert word in runbook, word
    minutes = {m: p7_md.model_minutes(m) for m in bench_models(cfg)}
    assert minutes["laya"] > minutes["laya_ml"] > minutes["modernbert_base"] > 0  # laya is the slowest
    for m in minutes:
        assert re.search(rf"\| `{m}` \|.*\| ~\d+(\.\d)?-\d+(\.\d)? min \|", built[2].cells[0].source), m
    (lo, hi), (s_lo, s_hi) = (map(int, re.fullmatch(r"(\d+)-(\d+) min", s).groups())
                              for s in (bp.RUNTIME, p7_md.SETUP_MIN + " min"))
    total = sum(minutes.values()) + p7_md.DOWNLOAD_MIN
    assert lo <= total + s_lo and total + s_hi <= hi


def _kernel_ns(work: Path) -> dict:  # what Step 1 leaves in the kernel for the later steps
    ns = {"__name__": "__main__", "json": json, "os": os, "shutil": shutil, "subprocess": subprocess, "sys": sys,
          "time": time, "Path": Path, "zipfile": zipfile, "REDO": False}
    exec("\n\n".join(inspect.getsource(f) for f in (*nh.HELPERS, *eh.E1_HELPERS, *ph.P3_HELPERS, *p7h.P7_HELPERS)),
         ns)
    runs = work / "runs" / "p7"
    ns.update(WORK=work, DATA=work / "data", RUNS=runs, LOGS=runs / "logs")
    return ns


def _recorder(ns: dict, calls: list, effect=None, rc: int = 0):
    def run_logged(cmd, log_path, expect_returncode=0, hint=None):
        calls.append((list(cmd), Path(log_path)))
        nh.write_json(log_path, "log")  # next_log numbers the attempts by the logs that exist
        if effect:
            effect(cmd)
        if rc and expect_returncode is not None:
            raise RuntimeError(f"exited with {rc}")
        return rc
    ns["run_logged"] = run_logged


def test_env_step_requires_a_gpu_and_names_the_one_that_runs(cfg, tmp_path, capsys):
    ns, calls = _kernel_ns(tmp_path / "w"), []
    _recorder(ns, calls, lambda cmd: nh.write_json(ns["RUNS"] / "env.json", {
        "cuda_available": True, "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "card": "G4",
        "capability": [12, 0], "vram_total_gb": 94.97}))
    exec(_cells(cfg)["env"].source, ns)
    out = capsys.readouterr().out
    assert calls[0][0][2:] == ["laya_poc.env_check", "--out", str(ns["RUNS"] / "env.json"), "--require-gpu"]
    assert "benchmark GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (card G4" in out
    assert "Step 4 - Environment check:" in out


BENCH = {"hardware": "Tesla T4 (cc 7.5, 14.56 GiB), CUDA 12.8; host Test CPU (2 vCPUs)",
         "results": [{"model": "laya", "backend": "torch", "batch_size": 32, "cold_s": 12.34, "first_predict_ms": 900,
                      "p50_ms": 25.04, "p95_ms": 31.46, "batch_rps": 81.234, "peak_vram_gb": 2.346}],
         "errors": [{"model": "laya", "backend": "torch", "batch_size": 64, "error": "OutOfMemoryError: CUDA oom"},
                    {"model": "mmbert_small", "backend": "hf", "batch_size": None, "error": "timed out"}]}


def test_bench_step_guards_a_running_benchmark_skips_when_done_and_archives_small_files(cfg, tmp_path, capsys):
    ns, calls, src = _kernel_ns(tmp_path / "w"), [], _cells(cfg)["bench"].source
    _recorder(ns, calls, lambda cmd: nh.write_json(Path(cmd[cmd.index("--out") + 1]), BENCH))
    ns["gpu_bench_pids"] = lambda: [777]  # e.g. the benchmark from before a kernel restart
    with pytest.raises(RuntimeError, match="still running.*777"):
        exec(src, ns)
    ns["gpu_bench_pids"] = lambda: []
    exec(src, ns)
    (cmd, log), = calls
    assert cmd[2] == "laya_poc.bench_gpu" and _arg(cmd, "--out") == str(ns["RUNS"] / BENCH_OUT)
    assert log.name == "06_bench_gpu.log" and _arg(cmd, "--device") == "cuda"
    out = capsys.readouterr().out
    assert "laya torch bs 32: cold 12.3 s, first 900 ms, p50 25.0 ms, p95 31.5 ms, batch 81.2 rec/s, peak VRAM " \
           "2.35 GiB" in out
    assert "laya torch bs 64: FAILED: OutOfMemoryError" in out and "mmbert_small hf: FAILED: timed out" in out
    assert "machine: Tesla T4" in out
    exec(src, ns)
    assert len(calls) == 1 and "skip:" in capsys.readouterr().out
    work, small = ns["WORK"], {"data/NOTICE_FSQ.txt", "data/data_report.json"}
    for rel in (*small, "data/test_id.jsonl", "data/split_val.parquet"):
        nh.write_json(work / rel, {})
    (work / "runs/p7/model.safetensors").write_bytes(b"weights")
    exec(_cells(cfg)["archive"].source, ns)  # small artefacts only: no weights, no FSQ rows
    runs = {f"runs/p7/{n}" for n in (BENCH_OUT, "bench_gpu_ok.json", "logs/06_bench_gpu.log")}
    assert set(zipfile.ZipFile(work / ARCHIVE_NAME).namelist()) == small | runs
    (ns["RUNS"] / "bench_gpu_ok.json").unlink()
    _recorder(ns, calls, rc=1)
    with pytest.raises(RuntimeError, match="exited with 1"):
        exec(src, ns)
    assert not (ns["RUNS"] / "bench_gpu_ok.json").exists()  # a failed run never marks the step done
    assert calls[-1][1].name == "06_bench_gpu_2.log"  # the earlier attempt's log is kept
    assert p7h.gpu_bench_summary(tmp_path / "none.json") == {} and "no benchmark results" in capsys.readouterr().out
    nh.write_json(tmp_path / "b.json", {"results": [{"model": "mmbert_small", "backend": "hf", "batch_size": 64}]})
    p7h.gpu_bench_summary(tmp_path / "b.json")  # a partial row: printed, never an exception
    assert "mmbert_small hf bs 64: cold n/a s" in capsys.readouterr().out
