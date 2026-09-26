"""The CLI commands the E1 notebook runs (design doc §6.2 Phase 2, spec §1), rendered like the smoke ones.

Same `Target` as the smoke notebook: the Colab GPU runtime, or the local CPU dry run (tiny checkpoint,
synthetic pool). Paths under the notebook's DATA/RUNS variables stay Python expressions (`Expr`).
"""
from __future__ import annotations

from notebook_commands import Expr, Target, _p

TRAINER = "laya_poc.train_single"


def train_args(cfg: dict, t: Target) -> list[str | Expr]:
    """Arguments of the E1 crash run and its resume: they must match exactly (train_single refuses a resume
    whose settings differ). Checkpoints every e1.ckpt_every_micro_steps micro-steps, not on a timer, so the
    forced crash always has a checkpoint to resume from; --save-best keeps the best eval's weights in best/."""
    e1 = cfg["e1"]
    batch = [a for flag, v in (("--micro-batch", t.micro_batch), ("--effective-batch", t.effective_batch)) if v
             for a in (flag, str(v))]
    return ["--run-dir", _p("RUNS", "train"), "--data-dir", _p("DATA"), "--model", e1["model"],
            "--seed", str(e1["seed"]), "--epochs", str(cfg["train"]["epochs"]), *(["--init", t.init] if t.init else []),
            "--device", t.device, "--card", t.card, *batch,
            "--ckpt-every-micro-steps", str(e1["ckpt_every_micro_steps"]), "--ckpt-every-min", "0",
            "--initial-eval", "--final-eval", "--save-final", "--save-best"]


def _zero_shot(model: str, t: Target, out: str) -> tuple[str, list]:
    return ("laya_poc.evaluate", ["--ckpt", t.init or "hub", "--model", model, "--rows", _p("DATA", "val.jsonl"),
                                  "--out", _p("RUNS", out), "--device", t.device])


def e1_commands(cfg: dict, t: Target) -> dict[str, tuple[str, list]]:
    """Every CLI of the E1 notebook, keyed by step: (module, args)."""
    e1, bench = cfg["e1"], cfg["bench"]
    dev, val, best = ["--device", t.device], ["--rows", _p("DATA", "val.jsonl")], _p("RUNS", "train", "best")
    data = ["--out", _p("DATA"), "--verify-frozen", *(["--pool", t.pool] if t.pool else []),
            *(["--init-tokenizer", t.init] if t.init else []), *(["--models", *t.data_models] if t.data_models else [])]
    return {
        "env": ("laya_poc.env_check", ["--out", _p("RUNS", "env.json"), *(["--require-gpu"] if t.require_gpu else []),
                                       "--expect-card", e1["card"]]),
        "data": ("laya_poc.build_data", data),
        "baselines": ("laya_poc.baselines", ["--data-dir", _p("DATA"), "--out", _p("RUNS", "baselines.json")]),
        "zeroshot": _zero_shot(e1["model"], t, "eval_zeroshot_val.json"),
        "zeroshot_ml": _zero_shot("laya_ml", t, "eval_zeroshot_ml_val.json"),
        "crash": (TRAINER, [*train_args(cfg, t), "--crash-at-micro-step", str(e1["crash_at_micro_step"])]),
        "resume": (TRAINER, train_args(cfg, t)),
        # best/ holds the best eval's weights: export_check cross-checks them against that eval, which
        # train_single writes to best/train_eval.json (the default, summary.json, holds the final weights' eval)
        "export": ("laya_poc.export_check", ["--ckpt", best, *val, "--out", _p("RUNS", "export_check.json"), *dev,
                                             "--train-summary", _p("RUNS", "train", "best", "train_eval.json")]),
        "evaluate": ("laya_poc.evaluate", ["--ckpt", best, "--model", e1["model"], *val,
                                           "--out", _p("RUNS", "eval_e1_val.json"),
                                           "--preds", _p("RUNS", "preds", "e1_val.jsonl"), *dev]),
        "bench": ("laya_poc.bench_cpu", ["--ckpt", best, "--model", e1["model"], "--rows", _p("DATA", "test_id.jsonl"),
                                         "--n", str(bench["n_records"]), "--threads", *map(str, bench["threads"]),
                                         "--warmup", str(bench["warmup"]), "--batch-size", str(bench["batch_size"]),
                                         "--out", _p("RUNS", "bench_cpu.json")]),
        "gate": ("laya_poc.gate", ["--run-root", _p("RUNS"), "--data-dir", _p("DATA"),
                                   "--out", _p("RUNS", "gate_report.md")]),
    }
