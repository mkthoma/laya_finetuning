"""Code cells of the Phase 5 CPU benchmark notebook (Phase 5 spec §5): the Phase 1/E1/Phase 3 setup, install,
token and data cells, reused, plus the CPU environment check, the benchmark and the archive.

Earlier templates are adapted by exact substitution (e1_cells._derive): if one of them changes, the build fails
here instead of silently producing a notebook with another phase's paths.
"""
from __future__ import annotations

import inspect
import json
from string import Template

from e1_cells import _data, _derive, _show_keys, install_source
from e1_helpers import E1_HELPERS
from notebook_cells import _ENV, _SETUP, _TOKEN, Bundle
from notebook_commands import Target, render_cmd
from notebook_helpers import HELPERS
from p3_helpers import P3_HELPERS
from p5_commands import BENCH_OUT, ONNX_DIR, ONNX_PINS, QUICK_OUT, bench_threads, p5_commands
from p5_helpers import P5_HELPERS

from laya_poc import bundle as B

ARCHIVE_NAME = "phase5_bench_artifacts.zip"
ENV_KEYS = ("cuda_available", "gpu_name", "driver", "torch", "transformers", "laya_commit", "ram_gb", "disk_free_gb",
            "on_colab", "warnings")
CPU_KEYS = ("cpu_model", "logical_cpus", "physical_cores", "affinity_cpus", "cgroup_cpus", "isa", "ram_gb")


def setup_source(target: Target, bundle: Bundle) -> str:
    """Phase 1 Step 1 with RUNS = runs/p5 and the E1 + Phase 3 + Phase 5 helpers."""
    helpers = "\n\n".join(inspect.getsource(f).rstrip() for f in (*HELPERS, *E1_HELPERS, *P3_HELPERS, *P5_HELPERS))
    src = _SETUP.substitute(work=json.dumps(target.work), sha=bundle.sha256, b64=bundle.b64, helpers=helpers,
                            unpack=B.UNPACK_SOURCE.rstrip())
    return _derive(src, 'WORK / "runs" / "smoke"', 'WORK / "runs" / "p5"')


_ONNX = Template('''\
ONNX = $pins  # the versions the ONNX export and benchmark were tested with (tools/p5_commands.py)
if any(dist_version(p) != v for p, v in ONNX.items()):
    # the constraints hold Colab's torch, transformers, protobuf and numpy where they are: an onnx that needs another
    # protobuf makes pip fail here, before anything changes, instead of breaking the stack
    pins = WORK / "constraints_p5.txt"
    pins.write_text("".join(f"{p}=={v}\\n" for p, v in before.items() if v), encoding="utf-8")
    run_logged(PIP + [f"{p}=={v}" for p, v in ONNX.items()] + ["-c", str(pins)], LOGS / "02_install_onnx.log",
               hint="pip could not install the pinned ONNX packages next to this runtime's protobuf and numpy "
                    "(read the log): use the Latest runtime (runbook: When something fails)")
''')
_ONNX_CHECK = '''\
if any(dist_version(p) != v for p, v in ONNX.items()):
    raise RuntimeError(f"ONNX packages {[dist_version(p) for p in ONNX]} != pinned {list(ONNX.values())}; see "
                       f"{LOGS / '02_install_onnx.log'}")
'''
_LAYA_GOT = 'got = direct_url("laya").get("vcs_info", {}).get("commit_id")\n'
_VERSIONS = '("torch", "transformers", "huggingface_hub", "duckdb", "laya", "laya-poc")'


def install_p5_source(cfg: dict) -> str:
    """The E1 install (pinned laya, pinned DuckDB, this project; torch/transformers/protobuf/numpy unchanged) plus
    the ONNX packages at ONNX_PINS, constrained to the runtime's own torch, transformers, protobuf and numpy."""
    src = install_source(cfg)
    src = _derive(src, "importlib.invalidate_caches()\n",
                  _ONNX.substitute(pins=json.dumps(ONNX_PINS)) + "importlib.invalidate_caches()\n")
    src = _derive(src, _LAYA_GOT, _ONNX_CHECK + _LAYA_GOT)
    return _derive(src, _VERSIONS, _VERSIONS[:-1] + "".join(f", {json.dumps(p)}" for p in ONNX_PINS) + ")")


def token_source() -> str:
    return _derive(_TOKEN, "smoke_test.local.ipynb (build_notebook.py --with-token)",
                   "phase5_cpu_bench.local.ipynb (build_p5_bench_notebook.py --with-token)")


_CPU = Template('''\
cpu = cpu_facts()
write_json(RUNS / "cpu.json", cpu)
print("cpu:", json.dumps({k: cpu.get(k) for k in $keys}, ensure_ascii=False))
for note in cpu_notes(cpu, json.loads(env_json.read_text(encoding="utf-8")), $threads):
    print("WARNING:", note)
''')


def env_source(c: dict, cfg: dict) -> str:
    """Phase 1 Step 4 without a GPU requirement, then the CPU facts (runs/p5/cpu.json) and the CPU warnings."""
    env = _ENV.substitute(cmd=render_cmd(*c["env"], indent=6))
    return (env + _show_keys("env", "env_json", ENV_KEYS)
            + _CPU.substitute(keys=json.dumps(CPU_KEYS), threads=json.dumps([int(t) for t in bench_threads(cfg)])))


_BENCH = Template('''\
QUICK = False  # True: a few rows per setting, a plumbing check in minutes; its numbers are NOT the benchmark
OUT = RUNS / ("$quick_out" if QUICK else "$out")
MARKER = OUT.with_name(OUT.stem + "_ok.json")  # written only after a successful run (a failed JSON never skips it)
if not already_done(MARKER):
    busy = bench_pids()
    if busy:  # e.g. the benchmark from before a kernel restart: two would share the cores and distort every timing
        raise RuntimeError(f"a benchmark process is still running (pid {busy}): wait for it to stop, or stop it "
                           "with os.kill(pid, 9), then re-run this step")
    MARKER.unlink(missing_ok=True)
    cmd = $cmd
    if QUICK:
        cmd.append("--quick")
    run_logged(cmd, next_log(LOGS, "06_bench" + ("_quick" if QUICK else "")), hint=$hint)
    write_json(MARKER, {"time": time.strftime("%Y-%m-%d %H:%M"), "quick": QUICK})
bench = bench_summary(OUT)
''')
BENCH_HINT = ("read the log: every setting runs in a fresh worker process and the JSON lists the failed ones; fix "
              "the cause and re-run this step (runbook: When something fails)")


def bench_source(c: dict) -> str:
    return _BENCH.substitute(quick_out=QUICK_OUT, out=BENCH_OUT, cmd=render_cmd(*c["bench"], indent=10),
                             hint=json.dumps(BENCH_HINT))


ARCHIVE = Template('''\
MAKE_ARCHIVE = True  # env and CPU facts, the benchmark JSON, logs: no ONNX graphs, no weights, no FSQ rows
ARCHIVE = WORK / "$name"
if MAKE_ARCHIVE:
    small = {".json", ".md", ".log", ".yaml", ".txt"}  # $onnx/: only the export records (*.onnx.json), not the graphs
    keep = [p for p in sorted(RUNS.rglob("*")) if p.is_file() and p.suffix in small]
    keep += [p for p in (DATA / "data_report.json", DATA / "SHA256SUMS", DATA / "NOTICE_FSQ.txt") if p.exists()]
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in keep:
            zf.write(p, p.relative_to(WORK).as_posix())
    print(f"{ARCHIVE}: {len(keep)} files, {ARCHIVE.stat().st_size / 1e6:.2f} MB; Colab view > Contents > "
          "right-click > Download...")
''').substitute(name=ARCHIVE_NAME, onnx=ONNX_DIR)


def code_sources(cfg: dict, target: Target, bundle: Bundle) -> dict[str, str]:
    """Every code cell's source, keyed by step (notebook order)."""
    c = p5_commands(cfg, target)
    return {"setup": setup_source(target, bundle), "install": install_p5_source(cfg), "token": token_source(),
            "env": env_source(c, cfg), "data": _data(c, cfg), "bench": bench_source(c), "archive": ARCHIVE}
