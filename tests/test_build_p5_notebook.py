"""The Phase 5 CPU benchmark notebook (tools/build_p5_bench_notebook.py), its cells run with stubbed CLIs, every
notebook and runbook command against the real parsers, the trap-annotation check and the CPU dry run's plumbing."""
from __future__ import annotations

import base64
import importlib.metadata
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

import pandas as pd
import pytest

nbformat = pytest.importorskip("nbformat")

from laya_poc import bundle as B  # noqa: E402
from laya_poc import traps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_e1_notebook as be  # noqa: E402
import build_notebook as bn  # noqa: E402
import build_p5_bench_notebook as bp  # noqa: E402
import e1_cells  # noqa: E402
import e1_helpers as eh  # noqa: E402
import notebook_helpers as nh  # noqa: E402
import p3_commands  # noqa: E402
import p3_helpers as ph  # noqa: E402
import p5_dry_run as dr5  # noqa: E402
import p5_helpers as p5h  # noqa: E402
import p5_md  # noqa: E402
import p5_trap_check as tc  # noqa: E402
from p5_commands import (A1_CSV, A2_CSV, BENCH_OUT, ONNX_PINS, QUICK_OUT, bench_models, p5_commands,  # noqa: E402
                         pip_install_line, runbook_commands, shell)

FAKE_TOKEN = "hf_" + "Qz7" * 10
NO_BUNDLE = bn.Bundle("", "0" * 64, [])
CLI_RE = re.compile(r'"-m", "laya_poc\.(\w+)"')
TRAP_CHECK = f".venv/Scripts/python tools/p5_trap_check.py {A1_CSV} {A2_CSV}"
ARCHIVE = "phase5_bench_artifacts.zip"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path, sha = bp.build(ROOT, tmp_path_factory.mktemp("nb") / "phase5_cpu_bench.ipynb")
    return path, sha, nbformat.read(str(path), as_version=4)


def _src(nb, kind: str = "code") -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == kind)


def _cells(cfg: dict) -> dict[str, bn.Cell]:
    return {c.key: c for c in bp.build_cells(cfg, bp.colab_target(cfg), NO_BUNDLE)}


def _flat(args: list) -> list[str]:
    return [a.src if isinstance(a, bn.Expr) else a for a in args]


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_notebook_is_a_valid_cpu_notebook_that_embeds_the_bundle_without_a_token(built, cfg):
    path, sha, nb = built
    nbformat.validate(nb)
    assert "accelerator" not in nb.metadata and "gpuType" not in nb.metadata.colab  # a CPU runtime ...
    assert nb.metadata.colab.machine_shape == "hm"                                  # ... with High-RAM
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, f"<cell {i}>", "exec")
            assert not any(line.lstrip().startswith(("!", "%")) for line in cell.source.splitlines())
    b64, want, members = B.bundle(ROOT, be.bundle_patterns(cfg, ROOT))
    assert sha == want and b64 in _src(nb) and want in _src(nb, "markdown")
    assert cfg["data"]["frozen_manifest"] in members
    text = path.read_text(encoding="utf-8")
    assert not [bad for bad in ("userdata", "files.upload", "eval_js", " -U") if bad in text]
    assert not bn.TOKEN_RE.search(text)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as zf:
        assert not any(bn.TOKEN_RE.search(zf.read(n).decode("utf-8", "replace")) for n in zf.namelist())


def test_cli_sequence_commands_and_adapted_templates_follow_the_phase_5_spec(built, cfg):
    _, _, nb = built
    assert CLI_RE.findall(_src(nb)) == ["env_check", "build_data", "bench_cpu"]  # no training, no matrix, no GPU
    cmds = {k: _flat(args) for k, (_, args) in p5_commands(cfg, bp.colab_target(cfg)).items()}
    assert cmds["env"] == ["--out", 'str(RUNS / "env.json")']  # no --require-gpu, no --expect-card
    assert cmds["data"] == _flat(p3_commands.p3_commands(cfg, bp.colab_target(cfg))["data"][1])
    bench, b = cmds["bench"], cfg["phase5"]["bench"]
    assert bench[:1 + len(b["models"])] == ["--models", *b["models"]] and _arg(bench, "--out") == "str(OUT)"
    assert _arg(bench, "--rows") == 'str(DATA / "test_id.jsonl")' and _arg(bench, "--onnx-dir") == 'str(RUNS / "onnx")'
    assert bench[bench.index("--threads") + 1:bench.index("--onnx-dir")] == [str(t) for t in b["threads"]]
    assert "--ckpt" not in bench and "--quick" not in bench  # hub checkpoints (the CLI default); QUICK adds --quick
    titles = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("###")]
    assert titles == ["### Step 1 - Setup", "### Step 2 - Install", "### Step 3 - Hugging Face token",
                      "### Step 4 - Environment check", "### Step 5 - Full data (frozen)", "### Step 6 - CPU benchmark",
                      "### Step 7 - Archive", "### Finish"]
    cells, e1 = _cells(cfg), e1_cells.code_sources(cfg, be.colab_target(cfg), NO_BUNDLE)
    setup = cells["setup"].source
    assert 'WORK / "runs" / "p5"' in setup and '"runs" / "smoke"' not in setup and "\nCARD = " not in setup
    for f in (*nh.HELPERS, *eh.E1_HELPERS, *ph.P3_HELPERS, *p5h.P5_HELPERS):
        assert f"def {f.__name__}(" in setup
    assert "phase5_cpu_bench.local.ipynb" in cells["token"].source and 'PRESET_TOKEN = ""' in cells["token"].source
    assert e1["data"] in cells["data"].source


def test_install_step_pins_the_onnx_packages_under_the_runtimes_own_stack(cfg, tmp_path, monkeypatch, capsys):
    for name, pinned in ONNX_PINS.items():  # the pins are the versions the benchmark was tested with (this venv)
        if importlib.util.find_spec(name.replace("-", "_")):
            assert importlib.metadata.version(name) == pinned, f"update p5_commands.ONNX_PINS[{name!r}]"
    stack = {"torch": "2.11.0+cpu", "transformers": "5.17.0", "protobuf": "6.33.6", "numpy": "2.1.3"}
    versions = {**stack, "duckdb": cfg["data"]["duckdb_version"], "laya": "0.3.20", "laya-poc": "0.1"}
    urls = {"laya": {"vcs_info": {"commit_id": cfg["laya"]["commit"]}}, "laya-poc": {"dir_info": {"editable": True}}}

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib.metadata, "distribution", type("Dist", (), {
        "__init__": lambda self, name: setattr(self, "name", version(name) and name),
        "read_text": lambda self, _: json.dumps(urls.get(self.name, {}))}))
    calls, bump = [], {}

    def pip(cmd, log, hint=None):
        calls.append((list(cmd), Path(log).name))
        versions.update({**(ONNX_PINS if "-c" in cmd else {}), **bump})

    src = _cells(cfg)["install"].source
    ns = {"sys": sys, "json": json, "WORK": tmp_path, "LOGS": tmp_path / "logs", "run_logged": pip}
    exec(src, dict(ns))
    (cmd, log), = calls
    pins = [f"{p}=={v}" for p, v in ONNX_PINS.items()]
    assert log == "02_install_onnx.log" and cmd[-2 - len(pins):] == [*pins, "-c", cmd[-1]]
    assert Path(cmd[-1]).read_text(encoding="utf-8").split() == [f"{p}=={v}" for p, v in stack.items()]
    assert "'onnx': '1.23.0'" in capsys.readouterr().out.replace('"', "'")
    exec(src, dict(ns))
    assert len(calls) == 1  # the pinned versions are present: pip is not called again
    versions = {k: v for k, v in versions.items() if k not in ONNX_PINS}  # the closures see the new dict
    bump["protobuf"] = "7.0.0"  # pip moved protobuf despite the constraints: the step must fail
    with pytest.raises(RuntimeError, match="pip changed .*protobuf"):
        exec(src, dict(ns))


def test_build_is_deterministic_and_the_token_variant_stays_local(tmp_path):
    a, b = (bp.build(ROOT, tmp_path / name)[0] for name in ("a.ipynb", "b.ipynb"))
    assert a.read_bytes() == b.read_bytes()
    local, _ = bp.build(ROOT, tmp_path / "p5.local.ipynb", token=FAKE_TOKEN)
    holders = [c for c in json.loads(local.read_text(encoding="utf-8"))["cells"] if FAKE_TOKEN in "".join(c["source"])]
    assert len(holders) == 1 and "".join(holders[0]["source"]).startswith("# Step 3")
    assert subprocess.run(["git", "check-ignore", "-q", bp.LOCAL_OUT.relative_to(ROOT).as_posix()],
                          cwd=ROOT).returncode == 0
    bp.check_token_out(ROOT, bp.LOCAL_OUT)
    for rel in ("notebooks/phase5_cpu_bench.ipynb", "phase5_cpu_bench.local.ipynb", "notebooks/x/p5.local.ipynb",
                "docs/p5.local.ipynb"):
        with pytest.raises(ValueError, match="gitignored.*phase5_cpu_bench.local.ipynb"):
            bp.build(tmp_path, tmp_path / rel, token=FAKE_TOKEN)
        with pytest.raises(ValueError, match="gitignored"):
            bp.check_token_out(ROOT, ROOT / rel)


def test_committed_notebook_embeds_the_current_code(cfg):
    if not bp.DEFAULT_OUT.exists():
        pytest.skip("notebook not generated yet")
    _, sha, _ = B.bundle(ROOT, be.bundle_patterns(cfg, ROOT))
    code = _src(nbformat.read(str(bp.DEFAULT_OUT), as_version=4))
    assert f'BUNDLE_SHA256 = "{sha}"' in code, "phase5_cpu_bench.ipynb is stale: run tools/build_p5_bench_notebook.py"


def _commands(cfg: dict, tmp_path: Path) -> list[tuple[str, str, list[str]]]:
    """(name, module, argv) of every notebook command (Colab, the local dry run) and every runbook command."""
    runs, out = tmp_path / "runs" / "p5", []
    ns = {"WORK": tmp_path, "DATA": tmp_path / "data", "RUNS": runs, "OUT": runs / BENCH_OUT}
    local = dr5.local_target(tmp_path / "work", tmp_path / "ckpt", tmp_path / "pool.parquet", tmp_path / "enc", cfg)
    for where, target in (("colab", bp.colab_target(cfg)), ("local", local)):
        for key, (module, args) in p5_commands(cfg, target).items():
            # Expr sources are this project's own rendered path expressions (str(WORK / "x")), as in P3/P4's tests
            argv = [str(eval(a.src, ns)) if isinstance(a, bn.Expr) else a for a in args]
            out += [(f"{where} {key}", module, argv)] + ([(f"{where} quick", module, [*argv, "--quick"])]
                                                         if key == "bench" else [])
    return out + [(k, m, list(a)) for k, (m, a) in runbook_commands(cfg).items()]


def test_every_notebook_and_runbook_command_is_accepted_by_its_cli_parser(cfg, tmp_path):
    """A renamed or missing flag fails here instead of on Colab; a CLI still being written is skipped, named."""
    pending = set()
    for key, module, argv in _commands(cfg, tmp_path):
        if importlib.util.find_spec(module) is None:
            pending.add(f"{module} (not written yet)")
            continue
        mod = importlib.import_module(module)
        if module.endswith("bench_cpu") and "--models" not in inspect.getsource(mod):
            pending.add(f"{module} (multi-model --models CLI of spec §4 not in place yet)")
            continue
        parse = mod.build_parser().parse_args if hasattr(mod, "build_parser") else getattr(mod, "parse_args", None)
        assert parse, f"{module} has neither build_parser() nor parse_args(): update this test"
        try:
            parse(argv)
        except SystemExit:
            pytest.fail(f"{module} rejects the '{key}' command: {argv}")
    assert tc.parse_args(TRAP_CHECK.split()[2:]).a2_csv == Path(A2_CSV)
    if pending:
        pytest.skip(f"not parsed yet: {sorted(pending)}")


def test_runbook_guide_readme_notebook_and_estimates_agree(built, cfg):
    md = _src(built[2], "markdown")
    for phrase in ("Select Kernel", "New Colab Server", "CPU", "High-RAM", "Auto Connect", "Step 1", "disconnect",
                   "QUICK", "Remove Server", "criterion 5", "physical cores", bp.RUNTIME):
        assert phrase in md, phrase
    runbook = (ROOT / "docs" / "phase5_runbook.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "trap_annotation_guide.md").read_text(encoding="utf-8")
    for key, (module, args) in runbook_commands(cfg).items():
        assert shell(module, args) in runbook, f"runbook lacks the {key} command: {shell(module, args)}"
    assert shell(*runbook_commands(cfg)["merge"]) in guide and TRAP_CHECK in guide
    for word in ("build_p5_bench_notebook.py", bp.RUNTIME, "trap_annotation_guide", pip_install_line(), TRAP_CHECK):
        assert word in runbook, word  # pip_install_line: the venv gets the same ONNX pins as Colab
    for word in (*traps.ANNOTATION_COLUMNS, "UTF-8", "BOM", "multi", "l1s", "label_key", "fsq_place_id",
                 *map(str, cfg["data"]["trap_target"]), "data/trap_candidates.csv"):
        assert word in guide, word
    phase5 = (ROOT / "README.md").read_text(encoding="utf-8").split("## Phase 5", 1)[1]
    assert all(w in phase5 for w in ("build_p5_bench_notebook.py", "phase5_cpu_bench", bp.RUNTIME,
                                     "docs/phase5_runbook.md", "docs/trap_annotation_guide.md"))
    minutes = {m: p5_md.model_minutes(cfg, m) for m in bench_models(cfg)}  # the intro's estimates
    assert minutes["laya"] > minutes["laya_ml"] > minutes["modernbert_base"] > 0  # laya is the slowest
    for m in minutes:
        assert re.search(rf"\| `{m}` \|.*\| ~\d+-\d+ min \|", built[2].cells[0].source), m
    (lo, hi), setup = map(float, re.fullmatch(r"(\d+)-(\d+) h", bp.RUNTIME).groups()), p5_md.SETUP_MIN.split("-")
    total = sum(minutes.values()) / 60
    assert lo <= total + int(setup[0]) / 60 and total + int(setup[1]) / 60 <= hi
    assert p5_md.model_minutes(cfg, "laya", cores=8) > minutes["laya"]  # 8 physical cores: the 8-thread row too


def _kernel_ns(work: Path) -> dict:  # what Step 1 leaves in the kernel for the later steps
    ns = {"__name__": "__main__", "json": json, "os": os, "shutil": shutil, "subprocess": subprocess, "sys": sys,
          "time": time, "Path": Path, "zipfile": zipfile, "REDO": False}
    exec("\n\n".join(inspect.getsource(f) for f in (*nh.HELPERS, *eh.E1_HELPERS, *ph.P3_HELPERS, *p5h.P5_HELPERS)),
         ns)
    runs = work / "runs" / "p5"
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


CPUINFO = "".join(f"processor\t: {i}\nmodel name\t: Test CPU @ 2.20GHz\nphysical id\t: {i // 4}\n"
                  f"core id\t\t: {(i // 2) % 2}\nflags\t\t: fpu sse avx2 avx512f avx512_vnni\n\n" for i in range(8))


def test_env_step_records_the_cpu_and_warns_about_a_gpu_few_cores_skipped_threads_and_a_quota(cfg, tmp_path, capsys):
    assert p5h.cpuinfo_facts(CPUINFO) == {"cpu_model": "Test CPU @ 2.20GHz", "physical_cores": 4,
                                          "isa": ["avx2", "avx512f", "avx512_vnni"]}
    assert p5h.cpuinfo_facts("") == {"cpu_model": None, "physical_cores": None, "isa": []}
    (tmp_path / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    assert p5h.cgroup_cpus(tmp_path) == 4.0 and p5h.cgroup_cpus(tmp_path / "missing") is None
    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    assert p5h.cgroup_cpus(tmp_path) is None
    facts = p5h.cpu_facts()  # this machine: best effort, never an exception
    assert facts["logical_cpus"] == os.cpu_count() and set(facts) >= {"cpu_model", "physical_cores", "cgroup_cpus"}
    assert p5h.cpu_notes({"physical_cores": 4, "cgroup_cpus": None}, {}, [1, 2, 4]) == []
    notes = p5h.cpu_notes({"physical_cores": 2, "cgroup_cpus": 1.5}, {"cuda_available": True, "gpu_name": "T4"},
                          [1, 2, 4, 8])
    assert len(notes) == 4 and "GPU is attached (T4)" in notes[0] and "criterion 5" in notes[1]
    assert "[4, 8] exceed the 2 physical cores" in notes[2] and "quota is 1.5" in notes[3]
    assert "unknown" in p5h.cpu_notes({}, {}, [1])[0]
    ns, calls = _kernel_ns(tmp_path / "w"), []  # Step 4 itself, with a stubbed env_check and CPU
    _recorder(ns, calls, lambda cmd: nh.write_json(ns["RUNS"] / "env.json", {"cuda_available": True,
                                                                             "gpu_name": "NVIDIA L4"}))
    ns["cpu_facts"] = lambda: {"cpu_model": "Test CPU", "physical_cores": 4, "logical_cpus": 8, "cgroup_cpus": None}
    exec(_cells(cfg)["env"].source, ns)
    out = capsys.readouterr().out
    assert calls[0][0][2:] == ["laya_poc.env_check", "--out", str(ns["RUNS"] / "env.json")]
    assert json.loads((ns["RUNS"] / "cpu.json").read_text(encoding="utf-8"))["physical_cores"] == 4
    assert "WARNING: a GPU is attached (NVIDIA L4)" in out and "[8] exceed the 4 physical cores" in out
    assert '"cpu_model": "Test CPU"' in out and "Step 4 - Environment check:" in out


BENCH = {"hardware": "Test CPU, 4 physical cores", "threads_skipped": [{"threads": 8, "reason": "above 4 cores"}],
         "results": [{"model": "laya", "backend": "onnx", "threads": 4, "cold_s": 3.21, "p50_ms": 301.4,
                      "p95_ms": 402.9, "batch_rps": 3.456, "peak_rss_gb": 2.5}],
         "errors": [{"model": "laya_ml", "backend": "torch", "threads": 1, "error": "worker timed out"}],
         "onnx": {"laya": {"accepted": True, "export": {"exporter": "torchscript"},  # bench_cpu schema 2
                           "acceptance": {"n": 1000, "argmax_agree": 1.0, "max_dp": 2e-5, "passed": True}},
                  "laya_ml": {"accepted": None, "error": "export failed"}}}


def test_bench_step_guards_the_cores_keeps_quick_apart_skips_when_done_and_archives_no_graphs(cfg, tmp_path, capsys):
    ns, calls, src = _kernel_ns(tmp_path / "w"), [], _cells(cfg)["bench"].source
    _recorder(ns, calls, lambda cmd: nh.write_json(Path(cmd[cmd.index("--out") + 1]), BENCH))
    ns["bench_pids"] = lambda: [777]  # e.g. the benchmark from before a kernel restart
    with pytest.raises(RuntimeError, match="still running.*777"):
        exec(src, ns)
    ns["bench_pids"] = lambda: []
    exec(src.replace("QUICK = False", "QUICK = True"), ns)  # the plumbing check: own output, own marker, --quick
    (cmd, log), = calls
    assert cmd[-1] == "--quick" and _arg(cmd, "--out").endswith(QUICK_OUT) and log.name == "06_bench_quick.log"
    exec(src, ns)
    assert "--quick" not in calls[1][0] and _arg(calls[1][0], "--out").endswith(BENCH_OUT)
    out = capsys.readouterr().out
    assert "laya onnx 4t: cold 3.2 s, p50 301 ms, p95 403 ms, batch 3.46 rec/s, peak RSS 2.50 GiB" in out
    assert "laya_ml torch 1t: FAILED: worker timed out" in out and "machine: Test CPU, 4 physical cores" in out
    assert 'ONNX laya: {"accepted": true, "exporter": "torchscript", "n": 1000, "argmax_agree": 1.0' in out
    assert '"error": "export failed"' in out and 'skipped: {"threads": 8' in out
    exec(src, ns)
    assert len(calls) == 2 and "skip:" in capsys.readouterr().out
    work, small = ns["WORK"], {"data/NOTICE_FSQ.txt", "data/data_report.json", "runs/p5/onnx/laya.onnx.json"}
    for rel in (*small, "data/test_id.jsonl", "data/split_val.parquet", "data/trap_candidates.csv"):
        nh.write_json(work / rel, {})
    for rel in ("runs/p5/onnx/laya.onnx", "runs/p5/onnx/laya.onnx.data", "runs/p5/onnx/model.safetensors"):
        (work / rel).write_bytes(b"graph")
    exec(_cells(cfg)["archive"].source, ns)  # small artefacts only: no ONNX graph, no weights, no FSQ rows
    runs = {f"runs/p5/{n}" for s in ("", "_quick") for n in (f"bench_cpu{s}.json", f"bench_cpu{s}_ok.json",
                                                              f"logs/06_bench{s}.log")}
    assert set(zipfile.ZipFile(work / ARCHIVE).namelist()) == small | runs
    (ns["RUNS"] / "bench_cpu_ok.json").unlink()
    _recorder(ns, calls, rc=1)
    with pytest.raises(RuntimeError, match="exited with 1"):
        exec(src, ns)
    assert not (ns["RUNS"] / "bench_cpu_ok.json").exists()  # a failed run never marks the step done
    assert p5h.bench_summary(tmp_path / "none.json") == {} and "no benchmark results" in capsys.readouterr().out
    nh.write_json(tmp_path / "b.json", {"results": [{"model": "mmbert_small", "backend": "hf", "threads": 2}]})
    p5h.bench_summary(tmp_path / "b.json")  # a partial row: printed, never an exception
    assert "mmbert_small hf 2t: cold n/a s, p50 n/a ms" in capsys.readouterr().out


def _annotated(tmp_path: Path, name: str, col: str, marks: list[str], lead=None, order=None) -> Path:
    cols = traps.candidate_columns()
    rows = [{**{c: "" for c in cols}, "pattern": "bank_word", "fsq_place_id": f"id{i}", "multi": str(i % 2),
             "label": "Retail", "label_key": "retail", col: m} for i, m in enumerate(marks)]
    for i, value in (lead or {}).items():
        rows[i]["keep_lead"] = value
    path = tmp_path / name
    traps.write_candidates(pd.DataFrame([rows[i] for i in (order or range(len(rows)))], columns=cols), path)
    return path


def test_trap_check_counts_agreement_lists_the_leads_rows_and_rejects_bad_files(tmp_path, capsys):
    a1 = _annotated(tmp_path, "a1.csv", "keep_a1", ["1", "1", "0", "0", "1", ""])
    a2 = _annotated(tmp_path, "a2.csv", "keep_a2", ["1", "0", "0", "1.0", "1", "1"], order=[5, 4, 3, 2, 1, 0])
    records, reordered = tc.align(a1, a2)
    s = tc.summarise(records)
    assert reordered and s["both_marked"] == 5 and s["unmarked"] == 1 and s["kept"] == 2 and s["kept_multi"] == 0
    assert [r["row"] for r in s["need_lead"]] == [3, 5] and s["agreement"] == pytest.approx(0.6)
    assert s["kappa"] == pytest.approx((0.6 - 0.52) / 0.48)  # p_e = .6 x .6 + .4 x .4
    assert tc.main([str(a1), str(a2)]) == 1 and "lead: row 3 (bank_word, id1): a1 1, a2 0" in capsys.readouterr().out
    done = _annotated(tmp_path, "a1_done.csv", "keep_a1", ["1", "1", "0", "0", "1", "1"], lead={1: "1", 3: "0"})
    assert tc.main([str(done), str(a2)]) == 0 and "ready for traps merge" in capsys.readouterr().out
    assert tc.kappa([1, 1], [1, 1]) == 1.0 and tc.kappa([], []) is None
    bad = _annotated(tmp_path, "bad.csv", "keep_a1", ["1", "yes", "0", "0", "1", "1"])
    assert tc.main([str(bad), str(a2)]) == 2 and "keep_a1: only 1, 0 or empty" in capsys.readouterr().err
    short = _annotated(tmp_path, "short.csv", "keep_a2", ["1"])
    assert tc.main([str(a1), str(short)]) == 2 and "same candidates" in capsys.readouterr().err


def test_dry_run_plumbing_check_names_what_the_quick_benchmark_lacks(cfg, tmp_path):
    dcfg = dr5.dry_config(cfg)
    assert dcfg["phase5"]["bench"]["threads"] == [1, 2] and dcfg["data"]["frozen_manifest"] is None
    backends = {m: (["torch", "onnx"] if m in cfg["model"] else ["hf"]) for m in bench_models(cfg)}
    rows = [{"model": m, "backend": b, "threads": 1} for m, bs in backends.items() for b in bs]
    good = {"backends": backends, "results": rows, "onnx": {m: {"accepted": True} for m in cfg["model"]}}
    assert dr5.bench_failures(dcfg, good) == [] and dr5.plumbing_failures(dcfg, tmp_path)[0].startswith("no ")
    bad = {**good, "results": rows[1:], "errors": [{"model": "laya", "backend": "onnx", "threads": 2, "error": "x"}],
           "onnx": {"laya": {"error": "export failed"}}}
    assert dr5.bench_failures(dcfg, bad) == ["no 1-thread row for laya torch", "setting failed: laya onnx 2t: x",
                                             "ONNX export of laya: export failed", "ONNX export of laya_ml: missing"]


@pytest.mark.torch
@pytest.mark.skipif(not os.environ.get("LAYA_DRY_RUN_TEST"),
                    reason="slow (several minutes on CPU, the real benchmark CLI): set LAYA_DRY_RUN_TEST=1 to run")
def test_dry_run_end_to_end_runs_every_cell_with_tiny_models(tmp_path):
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "p5_dry_run.py"), "--work", str(tmp_path / "w")],
                         capture_output=True, text=True, timeout=3600,
                         env={k: v for k, v in os.environ.items() if k != "LAYA_POC_ROOT"})
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    assert (tmp_path / "w" / "laya_poc" / ARCHIVE).exists()
