"""The CLI commands of Phase 5 (Phase 5 spec §4-§5): the Colab CPU benchmark notebook's and the runbook's.

The notebook runs on a Colab CPU runtime: the env check without a GPU, the frozen full data build of E1/Phase 3
(unchanged: `test_id.jsonl` is the benchmark's input) and one `python -m laya_poc.bench_cpu --models ...` over
every config `phase5.bench.models` (spec §4), with the ONNX export and acceptance check inside it. The runbook's
local commands (metrics, report, laptop benchmark, trap merge) are rendered here too, so the tests check each one
against its CLI's real parser and against the text of docs/phase5_runbook.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from notebook_commands import Expr, Target, _p
from p3_commands import p3_commands

BENCH = "laya_poc.bench_cpu"
BENCH_OUT = "bench_cpu.json"         # the benchmark JSON phase5_report --bench reads (spec §3)
QUICK_OUT = "bench_cpu_quick.json"   # QUICK = True: a plumbing check, never the benchmark
ONNX_DIR = "onnx"                    # exported .onnx files under RUNS: weights, never archived
OUT = Expr("str(OUT)")               # the bench cell sets OUT = RUNS / (BENCH_OUT or QUICK_OUT)
PY = ".venv/Scripts/python"          # the runbook's interpreter (README: Windows venv)
# The versions the Phase 5 ONNX export and benchmark were developed and tested with (the project venv). onnxscript
# (+ onnx-ir) lets torch's default dynamo exporter run, as it does locally (without it the export falls back to the
# TorchScript exporter). Colab's latest image has protobuf 6.33 and numpy 2.1, which these accept (onnx 1.23 needs
# protobuf >= 6.31.1).
ONNX_PINS = {"onnx": "1.23.0", "onnxruntime": "1.30.0", "onnxscript": "0.7.2", "onnx-ir": "1.0.0"}

# Local paths of the runbook (repo-relative; unzipped archives are gitignored under runs/).
P3_ROOT, P4_ROOT = "runs/p3_colab/runs/p3", "runs/p4_colab/runs/p4"
P5_COLAB = "runs/p5_colab/runs/p5"
LAPTOP = "runs/p5_laptop"
METRICS_JSON, REPORT_MD = "results/phase5_metrics.json", "results/phase5_report.md"
TRAPS = "data/trap.jsonl"
A1_CSV, A2_CSV = "data/trap_candidates_a1.csv", "data/trap_candidates_a2.csv"


@dataclass(frozen=True)
class P5Target(Target):
    """A Target for the CPU benchmark: no GPU, no card profile (nothing trains). `bench_ckpt` holds the local dry
    run's `bench_cpu --ckpt MODEL=/abs/dir` stand-ins (empty on Colab: the pinned Hub checkpoints)."""
    device: str = "cpu"
    require_gpu: bool = False
    card: str = "CPU"
    bench_ckpt: tuple[str, ...] = ()


def bench_models(cfg: dict) -> list[str]:
    return [str(m) for m in cfg["phase5"]["bench"]["models"]]


def bench_threads(cfg: dict) -> list[str]:
    return [str(t) for t in cfg["phase5"]["bench"]["threads"]]


def bench_args(cfg: dict, rows, out, onnx_dir) -> list:
    """Spec §4: every model, the pinned hub checkpoints (the CLI default: latency does not depend on fine-tuned
    weights), the config thread sweep; backends come from config phase5.bench.backends."""
    return ["--models", *bench_models(cfg), "--rows", rows, "--out", out, "--threads", *bench_threads(cfg),
            "--onnx-dir", onnx_dir]


def p5_commands(cfg: dict, t: Target) -> dict[str, tuple[str, list]]:
    """Every CLI of the Phase 5 CPU notebook, keyed by step: (module, args)."""
    env = ("laya_poc.env_check", ["--out", _p("RUNS", "env.json"), *(["--require-gpu"] if t.require_gpu else [])])
    data = p3_commands(cfg, t)["data"]  # = E1 Step 5: the FULL build with --verify-frozen
    stand_ins = ["--ckpt", *getattr(t, "bench_ckpt", ())] if getattr(t, "bench_ckpt", ()) else []
    bench = (BENCH, [*bench_args(cfg, _p("DATA", "test_id.jsonl"), OUT, _p("RUNS", ONNX_DIR)), *stand_ins])
    return {"env": env, "data": data, "bench": bench}


def runbook_commands(cfg: dict) -> dict[str, tuple[str, list[str]]]:
    """The local commands of docs/phase5_runbook.md, in order (repo-relative paths, run from the repo root)."""
    roots = ["--runs-root", P3_ROOT, "--runs-root", P4_ROOT, "--data-dir", "data"]
    colab = ["--metrics", METRICS_JSON, "--bench", f"{P5_COLAB}/{BENCH_OUT}"]  # the FIRST --bench is judged (C5)

    def laptop(out: str) -> list[str]:
        return bench_args(cfg, "data/test_id.jsonl", f"{LAPTOP}/{out}", f"{LAPTOP}/{ONNX_DIR}")

    return {
        "metrics": ("laya_poc.phase5_metrics", [*roots, "--out", METRICS_JSON]),
        "report": ("laya_poc.phase5_report", ["--metrics", METRICS_JSON, "--out", REPORT_MD]),
        "laptop_quick": (BENCH, [*laptop(QUICK_OUT), "--quick"]),
        "laptop": (BENCH, laptop(BENCH_OUT)),
        "report_colab": ("laya_poc.phase5_report", [*colab, "--out", REPORT_MD]),
        "merge": ("laya_poc.traps", ["merge", "--candidates", A1_CSV, A2_CSV, "--out", TRAPS, "--data-dir", "data"]),
        "metrics_traps": ("laya_poc.phase5_metrics", [*roots, "--traps", TRAPS, "--out", METRICS_JSON]),
        "report_final": ("laya_poc.phase5_report", [*colab, "--bench", f"{LAPTOP}/{BENCH_OUT}", "--out", REPORT_MD]),
    }


def pip_install_line(py: str = PY) -> str:
    """The runbook's venv install of the pinned ONNX packages."""
    return " ".join([py, "-m", "pip", "install", *(f"{p}=={v}" for p, v in ONNX_PINS.items())])


def shell(module: str, args: list[str], py: str = PY) -> str:
    """One runbook command line (every argument is a plain word: no quoting needed)."""
    return " ".join([py, "-m", module, *args])
