"""Code cells of the Phase 7 GPU benchmark notebook: the Phase 1/E1/Phase 3 setup, install, token and data cells,
reused, plus the GPU environment check, the benchmark and the archive.

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
from p7_commands import BENCH_OUT, p7_commands
from p7_helpers import P7_HELPERS

from laya_poc import bundle as B

ARCHIVE_NAME = "phase7_gpu_artifacts.zip"
ENV_KEYS = ("gpu_name", "capability", "driver", "vram_total_gb", "torch", "torch_cuda", "transformers",
            "laya_commit", "card", "disk_free_gb", "warnings")


def setup_source(target: Target, bundle: Bundle) -> str:
    """Phase 1 Step 1 (LAYA_CUDA_AMP=fp16, as every evaluation ran) with RUNS = runs/p7 and the E1 + Phase 3 +
    Phase 7 helpers."""
    helpers = "\n\n".join(inspect.getsource(f).rstrip() for f in (*HELPERS, *E1_HELPERS, *P3_HELPERS, *P7_HELPERS))
    src = _SETUP.substitute(work=json.dumps(target.work), sha=bundle.sha256, b64=bundle.b64, helpers=helpers,
                            unpack=B.UNPACK_SOURCE.rstrip())
    return _derive(src, 'WORK / "runs" / "smoke"', 'WORK / "runs" / "p7"')


def token_source() -> str:
    return _derive(_TOKEN, "smoke_test.local.ipynb (build_notebook.py --with-token)",
                   "phase7_gpu_bench.local.ipynb (build_p7_gpu_notebook.py --with-token)")


_GPU = '''\
print(f"benchmark GPU: {env.get('gpu_name')} (card {env.get('card') or 'no profile'}, "
      f"{env.get('vram_total_gb')} GiB); the benchmark JSON records it on every row")
'''


def env_source(c: dict) -> str:
    """Phase 1 Step 4 with a GPU required (any card: T4, G4, ...), then the GPU that will run the benchmark."""
    return _ENV.substitute(cmd=render_cmd(*c["env"], indent=6)) + _show_keys("env", "env_json", ENV_KEYS) + _GPU


_BENCH = Template('''\
OUT = RUNS / "$out"
MARKER = OUT.with_name(OUT.stem + "_ok.json")  # written only after a successful run (a failed JSON never skips it)
if not already_done(MARKER):
    busy = gpu_bench_pids()
    if busy:  # e.g. the benchmark from before a kernel restart: two would share the GPU and distort every timing
        raise RuntimeError(f"a GPU benchmark process is still running (pid {busy}): wait for it to stop, or stop it "
                           "with os.kill(pid, 9), then re-run this step")
    MARKER.unlink(missing_ok=True)
    cmd = $cmd
    run_logged(cmd, next_log(LOGS, "06_bench_gpu"), hint=$hint)
    write_json(MARKER, {"time": time.strftime("%Y-%m-%d %H:%M")})
bench = gpu_bench_summary(OUT)
''')
BENCH_HINT = ("read the log: each model runs in a fresh worker process and the JSON lists the failed ones; on a CUDA "
              "out-of-memory error lower --batch-sizes in cmd above and re-run this step with REDO = True in Step 1 "
              "(runbook: When something fails)")


def bench_source(c: dict) -> str:
    return _BENCH.substitute(out=BENCH_OUT, cmd=render_cmd(*c["bench"], indent=10), hint=json.dumps(BENCH_HINT))


ARCHIVE = Template('''\
MAKE_ARCHIVE = True  # env facts, the benchmark JSON, logs, the data report and NOTICE: no weights, no FSQ rows
ARCHIVE = WORK / "$name"
if MAKE_ARCHIVE:
    small = {".json", ".md", ".log", ".txt"}
    keep = [p for p in sorted(RUNS.rglob("*")) if p.is_file() and p.suffix in small]
    keep += [p for p in (DATA / "data_report.json", DATA / "SHA256SUMS", DATA / "NOTICE_FSQ.txt") if p.exists()]
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in keep:
            zf.write(p, p.relative_to(WORK).as_posix())
    print(f"{ARCHIVE}: {len(keep)} files, {ARCHIVE.stat().st_size / 1e6:.2f} MB; Colab view > Contents > "
          "right-click > Download...")
''').substitute(name=ARCHIVE_NAME)


def code_sources(cfg: dict, target: Target, bundle: Bundle) -> dict[str, str]:
    """Every code cell's source, keyed by step (notebook order)."""
    c = p7_commands(cfg, target)
    return {"setup": setup_source(target, bundle), "install": install_source(cfg), "token": token_source(),
            "env": env_source(c), "data": _data(c, cfg), "bench": bench_source(c), "archive": ARCHIVE}
