"""The Phase 3 seed matrix as run specs (design doc §5.14 / §6.2 Phase 3; run names after §7.3).

`phase3.arms` and `phase3.zero_shot` expand into one RunSpec per run, named
`fsq-{scheme}-{arm}-{model}-s{seed}[-n{subset}][-head]` (zero-shot: `fsq-{scheme}-B2-{model}-zs`). The date
suffix of §7.3 is dropped on purpose: a run name must stay the same across Colab sessions so a re-run finds,
resumes or skips its directory. Also here: which data directory each run trains and evaluates on (a missing
variant names the command that builds it), the eval cadence of the small learning-curve subsets, and the
status of a run directory. Pure apart from reading the files it is pointed at.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import labels
from .schedule import batching, count_items, plan_steps

ZERO_SHOT_ARM = "B2"
BASE_SCHEME = "c10"               # the scheme build_data writes to <data-root>/data
EXTRA_DIR = "data_eval"           # `variants traps` output: splits that are not in the scheme's data dir
EXTRA_SPLITS = ("trap_candidates",)
MIN_EVAL_EVERY = 10               # phase3.evals_per_run comment: eval every max(10, total_opt // n), cap 250


@dataclass(frozen=True)
class RunSpec:
    arm: str
    model: str
    scheme: str
    seed: int | None = None
    subset: int | None = None
    head_only: bool = False
    zero_shot: bool = False

    @property
    def name(self) -> str:
        base = f"fsq-{self.scheme}-{self.arm}-{self.model}"
        if self.zero_shot:
            return f"{base}-zs"
        return (f"{base}-s{self.seed}" + (f"-n{self.subset}" if self.subset else "")
                + ("-head" if self.head_only else ""))


@dataclass(frozen=True)
class DataPaths:
    train: Path | None                       # None for zero-shot runs
    eval: Path                               # <split>.jsonl for every eval split not in `extras`
    extras: tuple[tuple[str, Path], ...]     # (split, abs JSONL) passed to evaluate as --extra name=path


@dataclass(frozen=True)
class TrainOverrides:
    """train_single options in --train-extra that change how many optimiser steps a run takes."""
    epochs: int | None = None
    micro_batch: int | None = None
    effective_batch: int | None = None
    max_micro_steps: int | None = None


# ---------------------------------------------------------------- expansion

def _phase3(cfg: dict) -> dict:
    p3 = cfg.get("phase3")
    if not isinstance(p3, dict):
        raise ValueError("config has no phase3 section (arms, zero_shot, eval_splits, ...)")
    return p3


def _check_common(where: str, model: Any, scheme: Any, cfg: dict) -> None:
    if model not in cfg["model"]:
        raise ValueError(f"{where}: model {model!r} is not in config.model {sorted(cfg['model'])}")
    if scheme not in labels.SCHEMES:
        raise ValueError(f"{where}: scheme {scheme!r} is not one of {sorted(labels.SCHEMES)}")


def _positive_ints(where: str, key: str, values: Any, minimum: int) -> list[int]:
    ok = isinstance(values, list) and values and all(isinstance(v, int) and not isinstance(v, bool)
                                                      and v >= minimum for v in values)
    if not ok:
        raise ValueError(f"{where}: {key} must be a non-empty list of integers >= {minimum}, got {values!r}")
    return values


def _arm_specs(i: int, arm: Any, cfg: dict) -> list[RunSpec]:
    where = f"phase3.arms[{i}]"
    if not isinstance(arm, dict) or not isinstance(arm.get("id"), str) or not arm["id"]:
        raise ValueError(f"{where}: needs an id (e.g. E2), got {arm!r}")
    if arm["id"].upper() == ZERO_SHOT_ARM:
        raise ValueError(f"{where}: {ZERO_SHOT_ARM} is the zero-shot arm; list its models in phase3.zero_shot")
    _check_common(where, arm.get("model"), arm.get("scheme"), cfg)
    seeds = _positive_ints(where, "seeds", arm.get("seeds"), 0)
    subsets = _positive_ints(where, "train_subset", arm["train_subset"], 1) if "train_subset" in arm else [None]
    head = bool(arm.get("freeze_encoder", False))
    return [RunSpec(arm["id"], arm["model"], arm["scheme"], seed, n, head) for seed in seeds for n in subsets]


def _zero_shot_spec(i: int, entry: Any, cfg: dict) -> RunSpec:
    """A model name (scored on c10), or {model, scheme} for another scheme."""
    model, scheme = (entry, BASE_SCHEME) if isinstance(entry, str) else \
        ((entry.get("model"), entry.get("scheme", BASE_SCHEME)) if isinstance(entry, dict) else (entry, None))
    _check_common(f"phase3.zero_shot[{i}]", model, scheme, cfg)
    return RunSpec(ZERO_SHOT_ARM, model, scheme, zero_shot=True)


def expand_specs(cfg: dict) -> tuple[RunSpec, ...]:
    """Every configured run: the zero-shot evaluations first (cheap), then the arms in config order."""
    p3 = _phase3(cfg)
    specs = [_zero_shot_spec(i, e, cfg) for i, e in enumerate(p3.get("zero_shot") or [])]
    specs += [s for i, arm in enumerate(p3.get("arms") or []) for s in _arm_specs(i, arm, cfg)]
    names = [s.name for s in specs]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"phase3: duplicate runs {dupes} (same arm, model, scheme, seed and subset)")
    return tuple(specs)


def arm_ids(specs: Iterable[RunSpec]) -> list[str]:
    return sorted({s.arm for s in specs})


def _matches(token: str, spec: RunSpec) -> bool:
    t = token.lower()
    return token == spec.name or t == spec.arm.lower() or t == f"{spec.arm}-{spec.model}".lower()


def select(specs: Sequence[RunSpec], only: Sequence[str]) -> tuple[RunSpec, ...]:
    """Runs named by --only (run names, arm ids like E2, `<arm>-<model>` like B2-laya, or all), plan order."""
    if "all" in only:
        return tuple(specs)
    for token in only:
        if not any(_matches(token, s) for s in specs):
            raise ValueError(f"--only {token!r} matches no run; arms: {', '.join(arm_ids(specs))} (or all, "
                             f"<arm>-<model>, or a run name from `python -m laya_poc.matrix plan`)")
    return tuple(s for s in specs if any(_matches(t, s) for t in only))


# ---------------------------------------------------------------- data directories

def scheme_dir(data_root: Path, scheme: str) -> Path:
    return data_root / ("data" if scheme == BASE_SCHEME else f"data_{scheme}")


def subset_dir(data_root: Path, scheme: str, n: int) -> Path:
    return data_root / (f"data_lc{n}" if scheme == BASE_SCHEME else f"data_{scheme}_lc{n}")


def extra_file(data_root: Path, split: str, scheme: str) -> Path:
    return data_root / EXTRA_DIR / (f"{split}.jsonl" if scheme == BASE_SCHEME else f"{split}_{scheme}.jsonl")


def data_paths(spec: RunSpec, data_root: Path, eval_splits: Sequence[str]) -> DataPaths:
    """c10 -> <root>/data, c7 -> <root>/data_c7; subsets train on <root>/data_lc<N> but are evaluated on the
    full splits of their scheme; trap candidates come from <root>/data_eval."""
    evald = scheme_dir(data_root, spec.scheme)
    train = None if spec.zero_shot else (subset_dir(data_root, spec.scheme, spec.subset) if spec.subset else evald)
    extras = tuple((s, extra_file(data_root, s, spec.scheme)) for s in eval_splits if s in EXTRA_SPLITS)
    return DataPaths(train, evald, extras)


def build_hint(data_root: Path, target: Path, spec: RunSpec) -> str:
    """The command that creates `target` (a data dir or an extra split file)."""
    base = scheme_dir(data_root, BASE_SCHEME)
    if target.parent == data_root / EXTRA_DIR:
        return f"python -m laya_poc.variants traps --data-dir {base} --out {data_root / EXTRA_DIR}"
    if spec.subset and target == subset_dir(data_root, spec.scheme, spec.subset):
        return (f"python -m laya_poc.variants subset --data-dir {scheme_dir(data_root, spec.scheme)} "
                f"--n {spec.subset} --out {target}")
    if spec.scheme != BASE_SCHEME and target == scheme_dir(data_root, spec.scheme):
        return f"python -m laya_poc.variants {spec.scheme} --data-dir {base} --out {target}"
    return f"python -m laya_poc.build_data --out {base} --verify-frozen"


def _required(spec: RunSpec, paths: DataPaths, eval_splits: Sequence[str], epochs: int) -> list[tuple[Path, Path]]:
    """(file, the directory whose build creates it) for every input the run reads."""
    extras = dict(paths.extras)
    need = [(paths.eval / f"{s}.jsonl", paths.eval) for s in eval_splits if s not in extras]
    need += [(p, p) for p in extras.values()]
    if paths.train is not None:
        need += [(paths.train / f"train_e{e}.jsonl", paths.train) for e in range(epochs)]
        need += [(paths.train / "val.jsonl", paths.train)]
    return need


def check_data(spec: RunSpec, paths: DataPaths, data_root: Path, eval_splits: Sequence[str], *,
               epochs: int) -> None:
    """Fail before any GPU time is spent: one line naming the missing input and the command that builds it."""
    for path, source in _required(spec, paths, eval_splits, epochs):
        if path.is_file():
            continue
        missing = source if not source.exists() else path
        what = "directory" if missing.suffix != ".jsonl" else "file"
        raise FileNotFoundError(f"{spec.name}: missing {what} {missing}; build it with: "
                                f"{build_hint(data_root, source, spec)}")


# ---------------------------------------------------------------- eval cadence

def parse_train_extra(extra: Sequence[str]) -> TrainOverrides:
    p = argparse.ArgumentParser(add_help=False)
    for flag in ("--epochs", "--micro-batch", "--effective-batch", "--max-micro-steps"):
        p.add_argument(flag, type=int, default=None)
    known, _ = p.parse_known_args(list(extra))
    return TrainOverrides(known.epochs, known.micro_batch, known.effective_batch, known.max_micro_steps)


def eval_every_from_total(total_opt: int, *, evals_per_run: int, cap: int) -> int:
    return min(cap, max(MIN_EVAL_EVERY, total_opt // max(1, evals_per_run)))


def eval_every(cfg: dict, spec: RunSpec, train_dir: Path, card: str, over: TrainOverrides) -> int | None:
    """--eval-every-opt-steps for a subset run (about phase3.evals_per_run evals), None (train default) else.
    total_opt is what train_single will plan: the epoch files' item counts with this card's batching."""
    if not spec.subset:
        return None
    epochs = over.epochs or int(cfg["train"]["epochs"])
    n_items = [count_items(train_dir / f"train_e{e}.jsonl") for e in range(epochs)]
    mb, acc = batching(cfg, card, over.micro_batch, over.effective_batch)
    total = plan_steps(n_items, mb, acc, over.max_micro_steps)["total_opt"]
    return eval_every_from_total(total, evals_per_run=int(_phase3(cfg)["evals_per_run"]),
                                 cap=int(cfg["train"]["eval_every_opt_steps"]))


def train_epochs(cfg: dict, over: TrainOverrides) -> int:
    return over.epochs or int(cfg["train"]["epochs"])


# ---------------------------------------------------------------- status

def eval_complete(run_dir: Path, splits: Sequence[str]) -> bool:
    return (all((run_dir / "eval" / f"{s}.json").is_file() and (run_dir / "preds" / f"{s}.jsonl").is_file()
                for s in splits) and (run_dir / "order_invariance.json").is_file())


def train_complete(run_dir: Path) -> bool:
    return (run_dir / "train" / "summary.json").is_file()


def stage(run_dir: Path, spec: RunSpec, splits: Sequence[str]) -> str:
    """The last finished step: todo, started, training, trained, calibrated, evaluated or done."""
    if (run_dir / "done.json").is_file():
        return "done"
    if not run_dir.exists():
        return "todo"
    if eval_complete(run_dir, splits):
        return "evaluated"
    if (run_dir / "export_check.json").is_file():
        return "calibrated"
    if not spec.zero_shot and train_complete(run_dir):
        return "trained"
    return "training" if (run_dir / "train").exists() else "started"


def status(run_dir: Path, spec: RunSpec, splits: Sequence[str]) -> str:
    st = stage(run_dir, spec, splits)
    return st if st in ("done", "todo") else "partial"

