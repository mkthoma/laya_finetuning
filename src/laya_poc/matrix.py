"""Phase 3 seed matrix on one GPU session (design doc §6.2 Phase 3, §5.14): plan, run and report every run.

    python -m laya_poc.matrix plan   [--config yaml] [--work <abs>]
    python -m laya_poc.matrix run    --only <run name | arm id (E2) | arm-model (B2-laya) | all> [...]
                                     [--init hub|<abs dir>] [--device cuda|cpu] [--card auto|T4|L4|A10|G4|CPU]
                                     [--data-root <abs>] [--train-extra="<train_single flags>"] [--fail-fast]
                                     [--config yaml] [--work <abs>]
    python -m laya_poc.matrix report [--out results/phase3_report.md] [--config yaml] [--work <abs>]

A run lives in <work>/runs/p3/<run name>/ (train/, export_check.json, eval/<split>.json, preds/<split>.jsonl,
order_invariance.json, logs/<step>.log, timing.json, run_config.yaml, done.json). `run` skips a run with a
done.json and, inside a run, every step whose outputs exist; an interrupted training is simply started again
and train_single resumes from its ckpt/. A failing step ends its run with one line naming the step's log; the
other selected runs still run (--fail-fast stops instead) and the exit code is 1. --work defaults to the
project root ($LAYA_POC_ROOT), --data-root to --work; relative report paths are under --work.
"""
from __future__ import annotations

import argparse
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import yaml

from . import matrix_exec as X
from . import matrix_plan as P
from .config import load_config, project_root, with_overrides
from .matrix_results import BEST_WEIGHTS, num, refresh_runs_csv

RUNS_REL, RESULTS_REL = Path("runs") / "p3", Path("results")
RUNS_CSV, RUN_CONFIG, DEFAULT_REPORT = "runs.csv", "run_config.yaml", "results/phase3_report.md"
CARDS = ("auto", "T4", "L4", "A10", "G4", "CPU")


class StepFailed(RuntimeError):
    """A step of one run failed; the message is the one line the notebook shows."""


@dataclass(frozen=True)
class Matrix:
    cfg: dict
    work: Path
    data_root: Path
    specs: tuple[P.RunSpec, ...]

    @property
    def runs_dir(self) -> Path:
        return self.work / RUNS_REL

    @property
    def runs_csv(self) -> Path:
        return self.work / RESULTS_REL / RUNS_CSV

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self.cfg["phase3"]["eval_splits"])


@dataclass(frozen=True)
class RunOptions:
    init: str
    device: str
    card: str
    train_extra: tuple[str, ...]
    fail_fast: bool

    @property
    def overrides(self) -> P.TrainOverrides:
        return P.parse_train_extra(self.train_extra)


@dataclass(frozen=True)
class Step:
    name: str
    module: str
    args: list[str]
    finished: Callable[[], bool]          # outputs of an earlier attempt exist: skip
    problem: Callable[[], str | None]     # what is missing after the step (None = complete)


# ---------------------------------------------------------------- options

def abs_dir(value: str | None, flag: str, default: Path) -> Path:
    path = default if value is None else Path(value)
    if not path.is_absolute():
        raise ValueError(f"{flag} must be an absolute path, got {value!r}")
    return path


def load_matrix(args: argparse.Namespace) -> Matrix:
    work = abs_dir(args.work, "--work", project_root())
    cfg = load_config(args.config)
    data_root = abs_dir(getattr(args, "data_root", None), "--data-root", work)
    return Matrix(cfg, work, data_root, P.expand_specs(cfg))


def resolve_card(card: str | None, device: str, cfg: dict) -> str:
    """--card, else phase3.card; `auto` maps the GPU name (CPU on --device cpu)."""
    card = card or cfg["phase3"].get("card") or "auto"
    if card == "auto":
        if device == "cpu":
            return "CPU"
        import torch
        from .config import card_from_gpu_name
        if not torch.cuda.is_available():
            raise RuntimeError("--card auto --device cuda but CUDA is not available")
        card = card_from_gpu_name(torch.cuda.get_device_name(0))
    if card != "CPU" and card not in cfg["train"]["micro_batch"]:
        raise ValueError(f"card {card!r} has no train.micro_batch profile {sorted(cfg['train']['micro_batch'])}")
    return card


def resolve_init(init: str) -> str:
    if init == "hub":
        return init
    path = Path(init)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"--init must be 'hub' or an existing absolute Laya checkpoint dir, got {init!r}")
    return str(path)


def run_options(args: argparse.Namespace, cfg: dict) -> RunOptions:
    extra = tuple(tok for chunk in args.train_extra for tok in shlex.split(chunk))
    return RunOptions(resolve_init(args.init), args.device, resolve_card(args.card, args.device, cfg), extra,
                      args.fail_fast)


# ---------------------------------------------------------------- one run

def write_run_config(cfg: dict, spec: P.RunSpec, run_dir: Path) -> Path:
    """The config every step of the run reads: labels.scheme is the run's scheme (evaluate asks that question)."""
    path = run_dir / RUN_CONFIG
    text = yaml.safe_dump(with_overrides(cfg, {"labels.scheme": spec.scheme}), sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    return path


def _train_problem(run_dir: Path) -> str | None:
    train = run_dir / "train"
    missing = [str(p) for p in (train / "summary.json", train / "best" / BEST_WEIGHTS) if not p.is_file()]
    return f"missing {', '.join(missing)}" if missing else None


def _export_problem(run_dir: Path) -> str | None:
    ec = X.read_json(run_dir / "export_check.json")
    if not isinstance(ec, dict) or num(ec.get("T")) is None:
        return f"{run_dir / 'export_check.json'} has no fitted temperature T"
    return None


def _eval_problem(run_dir: Path, splits: Sequence[str]) -> str | None:
    need = [run_dir / "eval" / f"{s}.json" for s in splits] + [run_dir / "preds" / f"{s}.jsonl" for s in splits]
    missing = [p.relative_to(run_dir).as_posix() for p in (*need, run_dir / "order_invariance.json")
               if not p.is_file()]
    return f"missing {', '.join(missing)} in {run_dir}" if missing else None


def build_steps(m: Matrix, ctx: X.RunContext, spec: P.RunSpec, paths: P.DataPaths,
                eval_every: int | None) -> list[Step]:
    rd, splits = ctx.run_dir, m.splits
    evaluate = Step("evaluate", "evaluate", X.evaluate_args(ctx, spec, paths, splits),
                    lambda: P.eval_complete(rd, splits), lambda: _eval_problem(rd, splits))
    if spec.zero_shot:
        return [evaluate]
    return [Step("train", "train_single", X.train_args(ctx, spec, paths, eval_every),
                 lambda: P.train_complete(rd), lambda: _train_problem(rd)),
            Step("export_check", "export_check", X.export_args(ctx, spec, paths),
                 lambda: _export_problem(rd) is None, lambda: _export_problem(rd)),  # re-fits T otherwise
            evaluate]


def execute_step(step: Step, spec: P.RunSpec, run_dir: Path, runner: X.Runner) -> None:
    if step.finished():
        problem = step.problem()
        if problem:
            raise StepFailed(f"{spec.name}: {step.name} finished earlier but {problem}; delete {run_dir / 'train'} "
                             f"to retrain (or the whole run dir)")
        print(f"  {step.name}: skip (finished earlier)", flush=True)
        return
    if step.module == "train_single" and (pids := X.running_trainers(run_dir / "train")):
        raise StepFailed(f"{spec.name}: train_single is already running on {run_dir / 'train'} (pids {pids}); "
                         f"wait for it or kill it, then re-run")
    log = X.next_log(run_dir / "logs", step.name)
    print(f"  {step.name}: python -m laya_poc.{step.module} (log {log})", flush=True)
    t0 = time.monotonic()
    rc = runner(X.module_cmd(step.module, step.args), log)
    seconds = time.monotonic() - t0
    X.add_timing(run_dir, step.name, seconds)
    if rc != 0:
        raise StepFailed(f"{spec.name}: {step.name} failed (exit {rc}); log: {log}")
    if problem := step.problem():
        raise StepFailed(f"{spec.name}: {step.name} exited 0 but {problem}; log: {log}")
    print(f"  {step.name}: ok ({seconds:.0f} s)", flush=True)


def _warn_export(run_dir: Path) -> None:
    ec = X.read_json(run_dir / "export_check.json")
    if isinstance(ec, dict) and ec.get("passed") is False:
        print(f"  export_check: WARNING: a check FAILED (T {ec.get('T')}); see {run_dir / 'export_check.json'} "
              f"and logs/export_check.log. The run continues; the report lists it.", flush=True)


def finish_run(m: Matrix, opts: RunOptions, spec: P.RunSpec, run_dir: Path) -> dict:
    """Cleanup (train/ keeps best/, phase3.keep_after_run and its small files), then done.json."""
    keep = ("best", *(m.cfg["phase3"].get("keep_after_run") or ()))  # best/ is the run's result: never deleted
    removed = X.cleanup_train(run_dir / "train", keep)
    if removed:
        print(f"  cleanup: removed train/{{{','.join(removed)}}}", flush=True)
    timing = X.read_json(run_dir / X.TIMING_NAME) or {}
    done = {"run_name": spec.name, "arm": spec.arm, "model": spec.model, "scheme": spec.scheme, "seed": spec.seed,
            "subset": spec.subset, "head_only": spec.head_only, "card": opts.card,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(sum(float(v) for v in timing.values()), 1)}
    X.write_json(run_dir / "done.json", done)
    return done


def run_one(m: Matrix, opts: RunOptions, spec: P.RunSpec, runner: X.Runner) -> bool:
    """Run (or finish) one run. False when it was already done."""
    run_dir = m.runs_dir / spec.name
    if (run_dir / "done.json").is_file():
        stale = "" if P.eval_complete(run_dir, m.splits) else (
            f" (WARNING: {_eval_problem(run_dir, m.splits)} for the current phase3 eval splits; delete "
            f"{run_dir / 'done.json'} and re-run to evaluate again, training is not redone)")
        print(f"  skip: done.json exists{stale}", flush=True)
        return False
    paths = P.data_paths(spec, m.data_root, m.splits)
    P.check_data(spec, paths, m.data_root, m.splits, epochs=P.train_epochs(m.cfg, opts.overrides))
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = X.RunContext(m.cfg, run_dir, write_run_config(m.cfg, spec, run_dir), opts.init, opts.device, opts.card,
                       opts.train_extra)
    every = None if spec.zero_shot else P.eval_every(m.cfg, spec, paths.train, opts.card, opts.overrides)
    for step in build_steps(m, ctx, spec, paths, every):
        execute_step(step, spec, run_dir, runner)
        if step.name == "export_check":
            _warn_export(run_dir)
    done = finish_run(m, opts, spec, run_dir)
    print(f"  done in {done['seconds']:.0f} s -> {run_dir / 'done.json'}", flush=True)
    return True


# ---------------------------------------------------------------- commands

def cmd_plan(m: Matrix) -> int:
    header = f"{'run':<34} {'arm':<4} {'model':<8} {'scheme':<6} {'seed':<5} {'subset':<6} {'head':<5} status   stage"
    print(header)
    counts: dict[str, int] = {"done": 0, "partial": 0, "todo": 0}
    for s in m.specs:
        rd = m.runs_dir / s.name
        st, stg = P.status(rd, s, m.splits), P.stage(rd, s, m.splits)
        counts[st] += 1
        print(f"{s.name:<34} {s.arm:<4} {s.model:<8} {s.scheme:<6} {s.seed if s.seed is not None else '-':<5} "
              f"{s.subset or '-':<6} {'yes' if s.head_only else '-':<5} {st:<8} {stg}")
    print(f"{len(m.specs)} runs: {counts['done']} done, {counts['partial']} partial, {counts['todo']} todo "
          f"(run dirs under {m.runs_dir})")
    return 0


def cmd_run(m: Matrix, args: argparse.Namespace, runner: X.Runner) -> int:
    opts = run_options(args, m.cfg)
    selected = P.select(m.specs, args.only)
    print(f"matrix run: {len(selected)} run(s) on {opts.device}/{opts.card}, init {opts.init}, data {m.data_root}",
          flush=True)
    failed: list[str] = []
    for i, spec in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {spec.name}", flush=True)
        try:
            if run_one(m, opts, spec, runner):
                refresh_runs_csv(m.runs_dir, m.runs_csv)
        except Exception as exc:  # noqa: BLE001 - one line per failed run; the step log has the detail
            failed.append(spec.name)
            print(f"matrix: {exc}" if isinstance(exc, StepFailed) else f"matrix: {spec.name}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if opts.fail_fast:
                break
    refresh_runs_csv(m.runs_dir, m.runs_csv)
    if failed:
        print(f"matrix: {len(failed)} failed: {', '.join(failed)} (re-run the cell after fixing; finished runs "
              f"and steps are skipped)", file=sys.stderr, flush=True)
        return 1
    print(f"matrix run: all {len(selected)} run(s) done; {m.runs_csv}", flush=True)
    return 0


def cmd_report(m: Matrix, args: argparse.Namespace) -> int:
    from .matrix_report import build_report, write_report

    out = Path(args.out) if Path(args.out).is_absolute() else m.work / args.out
    refresh_runs_csv(m.runs_dir, m.runs_csv)
    result = build_report(m.cfg, m.specs, m.runs_dir, m.runs_csv)
    md, js = write_report(result, out)
    ex = result["exit_check"]
    print(f"matrix report: Phase 3 exit {ex['verdict']} ({ex['complete']}/{ex['total']} runs complete); {md}; {js}; "
          f"{m.runs_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="config.yaml (default: <project root>/config.yaml)")
    common.add_argument("--work", default=None, help="absolute work dir (default: $LAYA_POC_ROOT / project root)")
    p = argparse.ArgumentParser(prog="python -m laya_poc.matrix", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", parents=[common], help="list every configured run with its status")
    r = sub.add_parser("run", parents=[common], help="train -> T fit -> evaluate -> cleanup, per selected run")
    r.add_argument("--only", nargs="+", required=True, help="run names, arm ids (E2), arm-model (B2-laya) or all")
    r.add_argument("--init", default="hub", help="'hub' or an absolute Laya checkpoint dir (B2 evaluates it)")
    r.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    r.add_argument("--card", choices=CARDS, default=None, help="default: phase3.card")
    r.add_argument("--data-root", default=None, help="absolute dir holding data/, data_c7/, data_lc<N>/, data_eval/")
    r.add_argument("--train-extra", action="append", default=[],
                   help='extra train_single flags, e.g. --train-extra="--max-micro-steps 20" (last value wins)')
    r.add_argument("--fail-fast", action="store_true", help="stop at the first failed run")
    rep = sub.add_parser("report", parents=[common], help="results/phase3_report.md/.json and runs.csv")
    rep.add_argument("--out", default=DEFAULT_REPORT, help="report markdown (relative: under --work)")
    return p


def main(argv: list[str] | None = None, runner: X.Runner | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        m = load_matrix(args)
        if args.command == "plan":
            return cmd_plan(m)
        if args.command == "run":
            return cmd_run(m, args, runner or X.tee_run)
        return cmd_report(m, args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: one line for the notebook
        msg = (str(exc).strip().splitlines() or [""])[0]
        print(f"matrix: error: {type(exc).__name__}: {msg}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
