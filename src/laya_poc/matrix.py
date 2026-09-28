"""Phase 3 seed matrix and Phase 4 baselines on one GPU session (design doc §6.2, §5.13, §5.14): plan, run, report.

    python -m laya_poc.matrix plan   [--runs-root <dir>] [--config yaml] [--work <abs>]
    python -m laya_poc.matrix run    --only <run name | arm id (E2, B4) | arm-model (B2-laya, B1-prior) | all> [...]
                                     [--init hub|<abs dir>] [--device cuda|cpu] [--card auto|T4|L4|A10|G4|CPU]
                                     [--data-root <abs>] [--train-extra="<train_single flags>"]
                                     [--baseline-extra="<baseline CLI flags>"] [--skip-optional] [--fail-fast]
                                     [--runs-root <dir>] [--config yaml] [--work <abs>]
    python -m laya_poc.matrix report [--out results/phase3_report.md] [--runs-root <dir> ...] [--config yaml]
                                     [--work <abs>]

A Phase 3 run (Laya: the trained arms and B2 zero-shot) lives in <work>/runs/p3/<run name>/ (train/,
export_check.json, eval/<split>.json, preds/<split>.jsonl, order_invariance.json, logs/<step>.log, timing.json,
run_config.yaml, done.json); a Phase 4 baseline (phase4.arms: B1 majority/prior, B3 TF-IDF+LR, B4 small encoders,
B5 the optional reference LLM) in <work>/runs/p4/<run name>/, same layout with calibration.json (B4 also train/).
`run` skips a run with a done.json and, inside a run, every step whose outputs exist; an interrupted training is
simply started again and train_single resumes from its ckpt/ (a baseline is one step and simply re-runs). A
failing step ends its run with one line naming the step's log; the other selected runs still run (--fail-fast
stops instead) and the exit code is 1. --skip-optional skips optional runs (B5) with a status line, not a
failure. --work defaults to the project root ($LAYA_POC_ROOT), --data-root to --work; relative report and
--runs-root paths are under --work. --runs-root replaces the per-phase roots for plan/run (one value); report
merges every --runs-root given (default: runs/p3 and runs/p4), e.g. a Phase 3 and a Phase 4 archive.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml

from . import matrix_baselines as B
from . import matrix_exec as X
from . import matrix_plan as P
from .config import load_config, project_root, with_overrides
from .matrix_exec import Step, StepFailed  # noqa: F401 - re-exported: the matrix's step type and its error
from .matrix_options import (RunOptions, abs_dir, check_init_family, resolve_card, resolve_init,  # noqa: F401
                             run_options, runs_roots)
from .matrix_results import BEST_WEIGHTS, num, refresh_runs_csv

RUNS_REL, RESULTS_REL = Path("runs") / "p3", Path("results")
RUNS_REL_BY_PHASE = {3: RUNS_REL, 4: Path("runs") / "p4"}
RUNS_CSV, RUN_CONFIG, DEFAULT_REPORT = "runs.csv", "run_config.yaml", "results/phase3_report.md"
CARDS = ("auto", "T4", "L4", "A10", "G4", "CPU")


@dataclass(frozen=True)
class Matrix:
    cfg: dict
    work: Path
    data_root: Path
    specs: tuple[P.RunSpec, ...]
    roots: tuple[Path, ...] = ()          # --runs-root (absolute); empty: runs/p3 and runs/p4 by phase

    def root(self, phase: int) -> Path:
        return self.roots[0] if self.roots else self.work / RUNS_REL_BY_PHASE[phase]

    def run_dir(self, spec: P.RunSpec) -> Path:
        return self.root(spec.phase) / spec.name

    @property
    def runs_roots(self) -> tuple[Path, ...]:
        return self.roots or tuple(self.work / rel for rel in RUNS_REL_BY_PHASE.values())

    @property
    def runs_csv(self) -> Path:
        return self.work / RESULTS_REL / RUNS_CSV

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self.cfg["phase3"]["eval_splits"])


# ---------------------------------------------------------------- options

def load_matrix(args: argparse.Namespace) -> Matrix:
    work = abs_dir(args.work, "--work", project_root())
    cfg = load_config(args.config)
    data_root = abs_dir(getattr(args, "data_root", None), "--data-root", work)
    return Matrix(cfg, work, data_root, B.expand_all(cfg), runs_roots(args, work))


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
    """A Laya run: train -> export_check (T on val) -> evaluate; zero-shot: evaluate only."""
    rd, splits = ctx.run_dir, m.splits
    redo = f"delete {rd / 'train'} to retrain (or the whole run dir)"
    evaluate = Step("evaluate", "evaluate", X.evaluate_args(ctx, spec, paths, splits),
                    lambda: P.eval_complete(rd, splits), lambda: _eval_problem(rd, splits), redo)
    if spec.zero_shot:
        return [evaluate]
    return [Step("train", "train_single", X.train_args(ctx, spec, paths, eval_every),
                 lambda: P.train_complete(rd), lambda: _train_problem(rd), redo),
            Step("export_check", "export_check", X.export_args(ctx, spec, paths),
                 lambda: _export_problem(rd) is None, lambda: _export_problem(rd), redo),  # re-fits T otherwise
            evaluate]


def baseline_steps(m: Matrix, opts: RunOptions, ctx: X.RunContext, spec: P.RunSpec, paths: P.DataPaths) -> list[Step]:
    """A Phase 4 baseline: one CLI writes the whole run layout; it re-runs from scratch until it is complete."""
    rd, splits = ctx.run_dir, m.splits
    return [Step(B.step_name(spec), B.module(spec), B.baseline_args(ctx, spec, paths, splits, opts.baseline_extra),
                 lambda: B.outputs_problem(rd, spec, splits) is None, lambda: B.outputs_problem(rd, spec, splits),
                 f"delete {rd} to redo the run")]


def _warn_export(run_dir: Path) -> None:
    ec = X.read_json(run_dir / "export_check.json")
    if isinstance(ec, dict) and ec.get("passed") is False:
        print(f"  export_check: WARNING: a check FAILED (T {ec.get('T')}); see {run_dir / 'export_check.json'} "
              f"and logs/export_check.log. The run continues; the report lists it.", flush=True)


def finish_run(m: Matrix, opts: RunOptions, spec: P.RunSpec, run_dir: Path) -> dict:
    """Cleanup (train/ keeps best/, phase3.keep_after_run and its small files; Laya and B4), then done.json
    (baselines also record their kind)."""
    keep = ("best", *(m.cfg["phase3"].get("keep_after_run") or ()))  # best/ is the run's result: never deleted
    removed = X.cleanup_train(run_dir / "train", keep)
    if removed:
        print(f"  cleanup: removed train/{{{','.join(removed)}}}", flush=True)
    timing = X.read_json(run_dir / X.TIMING_NAME) or {}
    done = {"run_name": spec.name, "arm": spec.arm, "model": spec.model, "scheme": spec.scheme, "seed": spec.seed,
            "subset": spec.subset, "head_only": spec.head_only, "card": opts.card,
            **({} if spec.kind == P.LAYA else {"kind": spec.kind}),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(sum(float(v) for v in timing.values()), 1)}
    X.write_json(run_dir / "done.json", done)
    return done


def _done_note(m: Matrix, spec: P.RunSpec, run_dir: Path) -> str:
    """Empty, or a warning when a finished run lacks outputs for the currently configured eval splits."""
    if P.eval_complete(run_dir, m.splits, P.computes_order_invariance(spec)):
        return ""
    if spec.kind == P.LAYA:
        problem, redo = _eval_problem(run_dir, m.splits), "training is not redone"
    else:
        problem, redo = B.outputs_problem(run_dir, spec, m.splits), "the baseline runs again"
    return (f" (WARNING: {problem} for the current phase3 eval splits; delete {run_dir / 'done.json'} and re-run to "
            f"evaluate again, {redo})")


def _steps(m: Matrix, opts: RunOptions, ctx: X.RunContext, spec: P.RunSpec, paths: P.DataPaths) -> list[Step]:
    if spec.kind != P.LAYA:
        return baseline_steps(m, opts, ctx, spec, paths)
    every = None if spec.zero_shot else P.eval_every(m.cfg, spec, paths.train, opts.card, opts.overrides)
    return build_steps(m, ctx, spec, paths, every)


def run_one(m: Matrix, opts: RunOptions, spec: P.RunSpec, runner: X.Runner) -> bool:
    """Run (or finish) one run. False when it was already done."""
    run_dir = m.run_dir(spec)
    if (run_dir / "done.json").is_file():
        print(f"  skip: done.json exists{_done_note(m, spec, run_dir)}", flush=True)
        return False
    paths = P.data_paths(spec, m.data_root, m.splits)
    P.check_data(spec, paths, m.data_root, m.splits,
                 epochs=B.epochs(m.cfg, spec, opts.overrides, opts.baseline_overrides))
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = X.RunContext(m.cfg, run_dir, write_run_config(m.cfg, spec, run_dir), opts.init, opts.device, opts.card,
                       opts.train_extra)
    for step in _steps(m, opts, ctx, spec, paths):
        X.execute_step(step, spec.name, run_dir, runner)
        if step.name == "export_check":
            _warn_export(run_dir)
    done = finish_run(m, opts, spec, run_dir)
    print(f"  done in {done['seconds']:.0f} s -> {run_dir / 'done.json'}", flush=True)
    return True


# ---------------------------------------------------------------- commands

def _plan_rows(m: Matrix, specs: Sequence[P.RunSpec], row) -> dict[str, int]:
    counts: dict[str, int] = {"done": 0, "partial": 0, "todo": 0}
    for s in specs:
        rd = m.run_dir(s)
        st, stg = P.status(rd, s, m.splits), P.stage(rd, s, m.splits)
        counts[st] += 1
        print(row(s, st, stg))
    return counts


def _row3(s: P.RunSpec, st: str, stg: str) -> str:
    return (f"{s.name:<34} {s.arm:<4} {s.model:<8} {s.scheme:<6} {s.seed if s.seed is not None else '-':<5} "
            f"{s.subset or '-':<6} {'yes' if s.head_only else '-':<5} {st:<8} {stg}")


def _row4(s: P.RunSpec, st: str, stg: str) -> str:
    return (f"{s.name:<34} {s.arm:<4} {s.kind:<13} {s.model:<15} {s.scheme:<6} "
            f"{s.seed if s.seed is not None else '-':<5} {st:<8} {stg}{' (optional)' if s.optional else ''}")


def cmd_plan(m: Matrix) -> int:
    p3, p4 = [s for s in m.specs if s.phase == 3], [s for s in m.specs if s.phase == 4]
    print(f"{'run':<34} {'arm':<4} {'model':<8} {'scheme':<6} {'seed':<5} {'subset':<6} {'head':<5} status   stage")
    c = _plan_rows(m, p3, _row3)
    print(f"{len(p3)} runs: {c['done']} done, {c['partial']} partial, {c['todo']} todo (run dirs under {m.root(3)})")
    if p4:
        print(f"\nPhase 4 baselines\n{'run':<34} {'arm':<4} {'kind':<13} {'model':<15} {'scheme':<6} {'seed':<5} "
              f"status   stage")
        c = _plan_rows(m, p4, _row4)
        print(f"{len(p4)} baseline runs: {c['done']} done, {c['partial']} partial, {c['todo']} todo (run dirs "
              f"under {m.root(4)})")
    return 0


def _run_selected(m: Matrix, opts: RunOptions, selected: Sequence[P.RunSpec], runner: X.Runner) -> list[str]:
    failed: list[str] = []
    for i, spec in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {spec.name}", flush=True)
        if spec.optional and opts.skip_optional:
            print("  skip: optional, disabled (--skip-optional)", flush=True)
            continue
        try:
            if run_one(m, opts, spec, runner):
                refresh_runs_csv(m.runs_roots, m.runs_csv)
        except Exception as exc:  # noqa: BLE001 - one line per failed run; the step log has the detail
            failed.append(spec.name)
            print(f"matrix: {exc}" if isinstance(exc, StepFailed) else f"matrix: {spec.name}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if opts.fail_fast:
                break
    return failed


def cmd_run(m: Matrix, args: argparse.Namespace, runner: X.Runner) -> int:
    opts = run_options(args, m.cfg)
    selected = P.select(m.specs, args.only)
    check_init_family(opts.init, [s for s in selected if not (s.optional and opts.skip_optional)])
    skipped = sum(s.optional and opts.skip_optional for s in selected)
    note = f" ({skipped} optional skipped)" if skipped else ""
    print(f"matrix run: {len(selected)} run(s){note} on {opts.device}/{opts.card}, init {opts.init}, data "
          f"{m.data_root}", flush=True)
    failed = _run_selected(m, opts, selected, runner)
    refresh_runs_csv(m.runs_roots, m.runs_csv)
    if failed:
        print(f"matrix: {len(failed)} failed: {', '.join(failed)} (re-run the cell after fixing; finished runs "
              f"and steps are skipped)", file=sys.stderr, flush=True)
        return 1
    print(f"matrix run: all {len(selected)} run(s) done; {m.runs_csv}", flush=True)
    return 0


def cmd_report(m: Matrix, args: argparse.Namespace) -> int:
    from .matrix_report import build_report, write_report

    out = Path(args.out) if Path(args.out).is_absolute() else m.work / args.out
    refresh_runs_csv(m.runs_roots, m.runs_csv)
    result = build_report(m.cfg, m.specs, m.runs_roots, m.runs_csv, archived=args.archived)
    md, js = write_report(result, out)
    ex, p4 = result["exit_check"], result.get("phase4_exit_check")
    skipped = f", {len(p4['skipped_optional'])} optional not run" if p4 and p4["skipped_optional"] else ""
    tail = (f"; Phase 4 exit {p4['verdict']} ({p4['complete']}/{p4['total']} baseline runs complete{skipped})"
            if p4 else "")
    print(f"matrix report: Phase 3 exit {ex['verdict']} ({ex['complete']}/{ex['total']} runs complete){tail}; {md}; "
          f"{js}; {m.runs_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="config.yaml (default: <project root>/config.yaml)")
    common.add_argument("--work", default=None, help="absolute work dir (default: $LAYA_POC_ROOT / project root)")
    common.add_argument("--runs-root", action="append", default=None,
                        help="runs root (relative: under --work); default runs/p3 (Laya) and runs/p4 (baselines); "
                             "report merges every one given")
    p = argparse.ArgumentParser(prog="python -m laya_poc.matrix", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", parents=[common], help="list every configured run (both phases) with its status")
    r = sub.add_parser("run", parents=[common], help="train -> T fit -> evaluate -> cleanup, per selected run")
    r.add_argument("--only", nargs="+", required=True,
                   help="run names, arm ids (E2, B4), arm-model (B2-laya, B4-mmbert_small, B1-prior) or all")
    r.add_argument("--init", default="hub", help="'hub' or an absolute local model dir: a Laya checkpoint (B2 "
                                                 "evaluates it), or the B4 encoder / B5 causal LM (one kind per call)")
    r.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    r.add_argument("--card", choices=CARDS, default=None, help="default: phase3.card")
    r.add_argument("--data-root", default=None, help="absolute dir holding data/, data_c7/, data_lc<N>/, data_eval/")
    r.add_argument("--train-extra", action="append", default=[],
                   help='extra train_single flags, e.g. --train-extra="--max-micro-steps 20" (last value wins)')
    r.add_argument("--baseline-extra", action="append", default=[],
                   help='extra flags for the baseline CLIs (baseline_runs, small_encoder, llm_baseline), e.g. '
                        '--baseline-extra="--max-steps 4" (appended last; select one kind per call)')
    r.add_argument("--skip-optional", action="store_true", help="skip optional runs (B5) with a status line")
    r.add_argument("--fail-fast", action="store_true", help="stop at the first failed run")
    rep = sub.add_parser("report", parents=[common], help="results/phase3_report.md/.json and runs.csv")
    rep.add_argument("--out", default=DEFAULT_REPORT, help="report markdown (relative: under --work)")
    rep.add_argument("--archived", action="store_true",
                     help="the roots are downloaded archives (no weights): a recorded best step counts as best/")
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
