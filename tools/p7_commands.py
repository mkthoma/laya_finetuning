"""The CLI commands of the Phase 7 GPU inference benchmark: the Colab GPU notebook's and the runbook's.

The notebook runs on a Colab GPU runtime (T4 by default; G4 works too, and the result records which GPU ran): the
env check with a GPU required, the frozen full data build of E1 (unchanged: `test_id.jsonl` is the benchmark's
input) and one `python -m laya_poc.bench_gpu` over config `phase5.bench.models` (the CPU benchmark's four models),
the pinned Hub checkpoints and the CLI's row counts and batch sizes spelled out. The runbook's optional local CPU
plumbing check is rendered here too, so the tests check both against the CLI's real parser and the runbook's text.
"""
from __future__ import annotations

from dataclasses import dataclass

from e1_commands import e1_commands
from notebook_commands import Expr, Target, _p

from laya_poc.bench_gpu import BATCH_N, BATCH_SIZES, LATENCY_N, WARMUP

BENCH = "laya_poc.bench_gpu"
BENCH_OUT = "bench_gpu.json"            # under RUNS = WORK/runs/p7
OUT = Expr("str(OUT)")                  # the bench cell sets OUT = RUNS / BENCH_OUT
P7_COLAB = "runs/p7_colab/runs/p7"      # where the unzipped archive puts it locally (gitignored)
LOCAL_CHECK_OUT = "runs/p7_local/bench_gpu_cpu_check.json"
PY = ".venv/Scripts/python"             # the runbook's interpreter (README: Windows venv)


@dataclass(frozen=True)
class P7Target(Target):
    """The Colab GPU runtime (the Target defaults: cuda, GPU required). Nothing trains, so no card profile applies;
    `card` is only the notebook's accelerator hint."""
    card: str = "T4"


def bench_models(cfg: dict) -> list[str]:
    return [str(m) for m in cfg["phase5"]["bench"]["models"]]


def bench_args(cfg: dict, rows, out, device: str = "cuda") -> list:
    """Every benchmarked model on the pinned Hub checkpoints (the CLI default), with the CLI defaults explicit so the
    notebook shows them (edit --batch-sizes there after a CUDA OOM: runbook)."""
    return ["--models", *bench_models(cfg), "--rows", rows, "--out", out, "--latency-n", str(LATENCY_N),
            "--batch-n", str(BATCH_N), "--batch-sizes", *map(str, BATCH_SIZES), "--warmup", str(WARMUP),
            "--device", device]


def p7_commands(cfg: dict, t: Target) -> dict[str, tuple[str, list]]:
    """Every CLI of the Phase 7 GPU notebook, keyed by step: (module, args)."""
    env = ("laya_poc.env_check", ["--out", _p("RUNS", "env.json"), *(["--require-gpu"] if t.require_gpu else [])])
    data = e1_commands(cfg, t)["data"]  # = E1 Step 5: the FULL build with --verify-frozen
    bench = (BENCH, bench_args(cfg, _p("DATA", "test_id.jsonl"), OUT, t.device))
    return {"env": env, "data": data, "bench": bench}


def runbook_commands(cfg: dict) -> dict[str, tuple[str, list[str]]]:
    """The local commands of docs/phase7_gpu_runbook.md (repo-relative paths, run from the repo root)."""
    check = ["--models", "mmbert_small", "--rows", "data/test_id.jsonl", "--out", LOCAL_CHECK_OUT, "--device", "cpu",
             "--latency-n", "10", "--batch-n", "64", "--batch-sizes", "16", "32", "--warmup", "2"]
    return {"local_check": (BENCH, check)}


def shell(module: str, args: list[str], py: str = PY) -> str:
    """One runbook command line (every argument is a plain word: no quoting needed)."""
    return " ".join([py, "-m", module, *args])
