"""The options of `python -m laya_poc.matrix`: work/data/runs roots, the card, the --init model and the extra flags
passed through to the step CLIs. Validated up front so a bad flag fails before any GPU time is spent."""
from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import matrix_baselines as B
from .matrix_plan import RunSpec, TrainOverrides, parse_train_extra


@dataclass(frozen=True)
class RunOptions:
    init: str
    device: str
    card: str
    train_extra: tuple[str, ...]
    fail_fast: bool
    skip_optional: bool = False
    baseline_extra: tuple[str, ...] = ()

    @property
    def overrides(self) -> TrainOverrides:
        return parse_train_extra(self.train_extra)

    @property
    def baseline_overrides(self) -> TrainOverrides:
        return parse_train_extra(self.baseline_extra)


def abs_dir(value: str | None, flag: str, default: Path) -> Path:
    path = default if value is None else Path(value)
    if not path.is_absolute():
        raise ValueError(f"{flag} must be an absolute path, got {value!r}")
    return path


def runs_roots(args: argparse.Namespace, work: Path) -> tuple[Path, ...]:
    """--runs-root values (relative: under --work); plan and run take at most one."""
    given = [Path(v) if Path(v).is_absolute() else work / v for v in (getattr(args, "runs_root", None) or [])]
    if len(given) > 1 and args.command != "report":
        raise ValueError(f"--runs-root: `{args.command}` takes one runs root (report merges several)")
    return tuple(given)


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
        raise ValueError(f"--init must be 'hub' or an existing absolute local model dir (a Laya checkpoint; for B4/B5 "
                         f"the encoder / causal LM), got {init!r}")
    return str(path)


def check_init_family(init: str, specs: Sequence[RunSpec]) -> None:
    """A local --init is ONE model: the selected runs must all use the same kind of model (B1/B3 use none)."""
    families = sorted({f for s in specs if (f := B.init_family(s))})
    if init != "hub" and len(families) > 1:
        raise ValueError(f"--init {init} is one local model but the selected runs need {', '.join(families)} models; "
                         f"select one kind per call (e.g. --only B4)")


def split_flags(chunks: Sequence[str]) -> tuple[str, ...]:
    return tuple(tok for chunk in chunks for tok in shlex.split(chunk))


def run_options(args: argparse.Namespace, cfg: dict) -> RunOptions:
    return RunOptions(resolve_init(args.init), args.device, resolve_card(args.card, args.device, cfg),
                      split_flags(args.train_extra), args.fail_fast, args.skip_optional,
                      split_flags(args.baseline_extra))
