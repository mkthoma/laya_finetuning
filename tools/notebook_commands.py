"""The CLI commands the smoke notebook runs, and how they are rendered into cell source.

A `Target` says where the cells run: the Colab GPU runtime (the defaults) or the local CPU dry run. Paths
under the notebook's WORK/DATA/RUNS variables stay Python expressions (`Expr`), so one rendering serves both.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Target:
    """Where the cells run: the Colab GPU runtime (the defaults) or the local CPU dry run."""
    work: str = "/content/laya_poc"
    device: str = "cuda"
    require_gpu: bool = True
    init: str | None = None            # absolute checkpoint dir used instead of the Hub checkpoints
    pool: str | None = None            # absolute pool parquet used instead of the DuckDB extraction
    data_models: tuple[str, ...] = ()  # build_data --models (empty: the CLI default)
    micro_batch: int | None = None
    effective_batch: int | None = None
    # train_single --card. The smoke plan (ckpt every 40, kill at 122 -> resume from 120, 250 micro-steps)
    # assumes the T4 profile (MB 8 x ACC 4). With "auto", a fallback L4/A10G gets MB 32 x ACC 1 and ends near
    # micro-step 67, before the crash, and an A100/H100 has no profile at all. Pinning T4 keeps the plan intact
    # on whatever GPU the extension assigns (only VRAM and speed then stop describing a T4).
    card: str = "T4"


@dataclass(frozen=True)
class Expr:
    """Python source placed verbatim in a rendered command (e.g. a path under WORK)."""
    src: str


def _p(var: str, *parts: str) -> Expr:
    """str() of a path under one of the notebook's directory variables (WORK, DATA, RUNS)."""
    return Expr(f"str({var}{''.join(f' / {json.dumps(p)}' for p in parts)})")


def _train_args(sm: dict, t: Target) -> list:
    """Arguments shared by the control, crash and resume runs (they must match for an exact resume).
    The checkpoint cadence is not part of the resume fingerprint, so it is set per run."""
    batch = [a for flag, v in (("--micro-batch", t.micro_batch), ("--effective-batch", t.effective_batch)) if v
             for a in (flag, str(v))]
    return ["--data-dir", _p("DATA"), "--model", sm["ckpt"], "--seed", str(sm["seed"]), "--epochs", str(sm["epochs"]),
            *(["--init", t.init] if t.init else []), "--device", t.device, "--card", t.card, *batch,
            "--max-micro-steps", str(sm["micro_steps"])]


# Only the crash run's checkpoints are read (Step 10 resumes from them). The control and resumed runs would
# otherwise write about 5 GB per checkpoint (model + AdamW state) that nothing uses.
NO_CKPT = ["--ckpt-every-micro-steps", "0", "--ckpt-every-min", "0"]


def commands(cfg: dict, t: Target) -> dict[str, tuple[str, list]]:
    """Every CLI the notebook runs, keyed by step: (module, args). Sizes come from cfg['smoke']."""
    sm = cfg["smoke"]
    init, dev = (["--init", t.init] if t.init else []), ["--device", t.device]
    rows = ["--rows", _p("DATA", "val.jsonl")]
    train = _train_args(sm, t)
    data = ["--smoke", "--out", _p("DATA"), *(["--pool", t.pool] if t.pool else []),
            *(["--init-tokenizer", t.init] if t.init else []), *(["--models", *t.data_models] if t.data_models else [])]
    out = {
        "env": ("laya_poc.env_check", ["--out", _p("RUNS", "env.json"), *(["--require-gpu"] if t.require_gpu else []),
                                       "--expect-card", "T4"]),
        "data": ("laya_poc.build_data", data),
        "zeroshot": ("laya_poc.zeroshot", ["--models", "laya", "laya_ml", *init, *rows, "--n", str(sm["zero_shot_n"]),
                                           "--out", _p("RUNS", "zeroshot.json"), *dev]),
    }
    for m in ("laya", "laya_ml"):
        out[f"parity_{m}"] = ("laya_poc.parity", ["--model", m, *init, *rows, "--n", str(sm["parity_n"]), "--out",
                                                  _p("RUNS", f"parity_{m}.json"),
                                                  *([] if t.require_gpu else ["--allow-cpu"])])
    trainer = "laya_poc.train_single"
    # --initial-eval: val CE at opt 0, which the report's loss-drop criterion compares with the final eval
    out["control"] = (trainer, ["--run-dir", _p("RUNS", "control"), *train, *NO_CKPT, "--initial-eval",
                                "--final-eval", "--save-final"])
    out["crash"] = (trainer, ["--run-dir", _p("RUNS", "resumed"), *train,
                              "--ckpt-every-micro-steps", str(sm["ckpt_every_micro_steps"]),
                              "--crash-at-micro-step", str(sm["kill_at_micro_step"])])
    out["resume"] = (trainer, ["--run-dir", _p("RUNS", "resumed"), *train, *NO_CKPT])
    out["export"] = ("laya_poc.export_check", ["--ckpt", _p("RUNS", "control", "final"), *rows,
                                               "--out", _p("RUNS", "export_check.json"), *dev])
    out["report"] = ("laya_poc.smoke_report", ["--run-root", _p("RUNS"), "--data-dir", _p("DATA"),
                                               "--out", _p("RUNS", "smoke_report.md")])
    return out


def render_cmd(module: str, args: Sequence[str | Expr], indent: int = 0, width: int = 110) -> str:
    """Python list literal [sys.executable, "-m", module, *args], wrapped between flag groups.

    `indent` is the column of the opening bracket, so continuation lines line up under it.
    """
    groups: list[list[str]] = [["sys.executable", '"-m"', json.dumps(module)]]
    for a in args:
        if isinstance(a, str) and a.startswith("--"):
            groups.append([])
        groups[-1].append(a.src if isinstance(a, Expr) else json.dumps(a))
    lines, line = [], ""
    for piece in (", ".join(g) for g in groups):
        if line and indent + len(line) + len(piece) + 3 > width:
            lines.append(line + ",")
            line = piece
        else:
            line = f"{line}, {piece}" if line else piece
    return "[" + ("\n" + " " * (indent + 1)).join([*lines, line]) + "]"


def guarded(cmds: dict, key: str, marker: str, log: str, *, hint: str | None = None, lhs: str = "",
            extra: str = "", pre: Sequence[str] = (), post: Sequence[str] = ()) -> str:
    """`if not already_done(marker):` then setup lines, the rendered command, run_logged and follow-ups."""
    call = f"{lhs}run_logged(cmd, LOGS / {json.dumps(log)}{extra}"
    if hint:
        call += ",\n" + " " * (4 + len(lhs) + 11) + f"hint={json.dumps(hint)}"
    body = [*pre, "cmd = " + render_cmd(*cmds[key], indent=10), call + ")", *post]
    return f"if not already_done({marker}):\n" + "\n".join("    " + line for line in body)
