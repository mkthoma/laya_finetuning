"""The resolved settings of one bench_cpu invocation: models, backends, sources, thread sweep, row counts.

Defaults come from config `phase5.bench` (the Phase 5 sweep form, `--models`) or config `bench` (the Phase 2
single-model form, `--model`); `--quick` shrinks the row counts to QUICK for a smoke run; explicit flags win over both.
Thread settings above the machine's physical cores are skipped with a note (design §7.11 "capped at physical cores";
`--thread-cap logical|none` relaxes it; the Phase 2 form never capped and still does not).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bench_sources import Source, backends_for, model_kind, parse_ckpt, resolve_source
from .config import project_root

QUICK = {"latency_n": 10, "batch_n": 32, "warmup": 2, "check_n": 32}
SWEEP, SINGLE = "sweep", "single"
CAP_BASES = ("physical", "logical", "none")
DEFAULT_ONNX_SUBDIR = ("runs", "onnx")


@dataclass(frozen=True)
class Plan:
    form: str                           # SWEEP | SINGLE
    models: tuple[str, ...]
    backends: dict[str, list[str]]      # model -> backends to run (in order)
    ckpt: dict[str, str]                # model -> 'hub' | absolute dir
    threads: tuple[int, ...]            # requested (before the cap)
    thread_cap: str                     # physical | logical | none
    latency_n: int
    batch_n: int
    warmup: int
    batch_size: int
    check_n: int                        # ONNX acceptance rows (0 = skip the check)
    opset: int
    quick: bool
    onnx_dir: Path
    skipped_models: tuple[dict[str, str], ...] = ()

    def needs_onnx(self) -> list[str]:
        return [m for m in self.models if "onnx" in self.backends[m]]


def _defaults(cfg: Mapping[str, Any], sweep: bool) -> dict[str, Any]:
    p5 = (cfg.get("phase5") or {}).get("bench") or {}
    onnx = p5.get("onnx") or {}
    if sweep:
        base = {k: p5.get(k) for k in ("latency_n", "batch_n", "warmup", "batch_size", "threads")}
    else:
        b = cfg.get("bench") or {}
        base = {"latency_n": b.get("n_records"), "batch_n": b.get("n_records"), "warmup": b.get("warmup"),
                "batch_size": b.get("batch_size"), "threads": b.get("threads")}
    return {**base, "check_n": onnx.get("check_n", 1000), "opset": onnx.get("opset", 18)}


def _pick(args: argparse.Namespace, name: str, defaults: dict[str, Any]) -> int:
    explicit = getattr(args, name)
    if explicit is not None:
        value = explicit
    elif name in ("latency_n", "batch_n") and args.n is not None:
        value = args.n
    elif args.quick and name in QUICK:
        value = min(QUICK[name], int(defaults[name]))
    else:
        value = defaults[name]
    if value is None:
        raise ValueError(f"no value for {name}: pass --{name.replace('_', '-')} or set it in the config")
    return int(value)


def _positive(values: dict[str, int]) -> None:
    bad = [f"{k}={v}" for k, v in values.items() if v < (0 if k in ("warmup", "check_n") else 1)]
    if bad:
        raise ValueError(f"invalid settings: {', '.join(bad)} (row counts and batch size >= 1, warmup >= 0)")


def _backends(cfg: Mapping[str, Any], models: Sequence[str], requested: Sequence[str] | None
              ) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    out, skipped = {}, []
    for m in models:
        model_kind(cfg, m)  # unknown models fail here, before anything runs
        chosen = backends_for(cfg, m, requested)
        if chosen:
            out[m] = chosen
        else:
            skipped.append({"model": m, "reason": f"none of the backends {list(requested or [])} applies"})
    if not out:
        raise ValueError(f"no requested backend applies to any of {list(models)}")
    return out, skipped


def make_plan(args: argparse.Namespace, cfg: Mapping[str, Any]) -> Plan:
    sweep = args.models is not None
    models = list(dict.fromkeys(args.models)) if sweep else [args.model or "laya"]
    backends, skipped = _backends(cfg, models, args.backends if sweep else [args.backend])
    defaults = _defaults(cfg, sweep)
    nums = {k: _pick(args, k, defaults) for k in ("latency_n", "batch_n", "warmup", "batch_size", "check_n", "opset")}
    _positive({k: v for k, v in nums.items() if k != "opset"})
    threads = tuple(dict.fromkeys(args.threads or defaults["threads"] or []))
    if not threads or min(threads) < 1:
        raise ValueError(f"--threads must list positive thread counts, got {list(threads)}")
    kept = [m for m in models if m in backends]
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else project_root().joinpath(*DEFAULT_ONNX_SUBDIR)
    return Plan(form=SWEEP if sweep else SINGLE, models=tuple(kept), backends=backends,
                ckpt=parse_ckpt(args.ckpt, kept), threads=threads,
                thread_cap=args.thread_cap or ("physical" if sweep else "none"), quick=bool(args.quick),
                onnx_dir=onnx_dir.resolve(), skipped_models=tuple(skipped), **nums)


def cap_threads(requested: Sequence[int], basis: str, machine: Mapping[str, Any]
                ) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    """(thread settings to run, the skipped ones with the reason, the cap record)."""
    if basis not in CAP_BASES:
        raise ValueError(f"thread cap must be one of {CAP_BASES}, got {basis!r}")
    physical, logical = machine.get("physical_cores"), machine.get("cpu_count")
    limit = {"physical": physical or logical, "logical": logical, "none": None}[basis]
    run = [t for t in requested if limit is None or t <= limit]
    unit = "physical cores" if basis == "physical" and physical else "logical CPUs"
    skipped = [{"threads": t, "reason": f"above the {limit} {unit} of this machine (design §7.11 cap)"}
               for t in requested if t not in run]
    note = None if basis != "physical" or physical else "physical core count unknown: capped at logical CPUs"
    return run, skipped, {"basis": basis, "limit": limit, "note": note}


def resolve_sources(cfg: Mapping[str, Any], plan: Plan) -> dict[str, Source]:
    """Every model's weights as an absolute local directory (downloads happen here, before any timing)."""
    return {m: resolve_source(cfg, m, plan.ckpt[m]) for m in plan.models}
