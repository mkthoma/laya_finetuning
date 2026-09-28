"""Phase 4 baselines as matrix runs (design doc §6.2 Phase 4, §5.13, §7.10; spec P4 §1/§5).

`phase4.arms` expands into RunSpecs next to the Phase 3 ones (matrix_plan.expand_specs), one per run:

    kind majority      -> fsq-{scheme}-B1-majority and fsq-{scheme}-B1-prior (B1 is "majority + prior")
    kind prior         -> fsq-{scheme}-B1-prior
    kind tfidf_lr      -> fsq-{scheme}-B3-tfidf_lr
    kind small_encoder -> fsq-{scheme}-B4-{model}-s{seed}   (model: a key of phase4.small_encoders)
    kind llm           -> fsq-{scheme}-B5-{model}           (model: a key of phase4.llms; `optional: true`)

A run is ONE subprocess that writes the Phase 3 run layout (eval/, preds/, calibration.json, order_invariance.json
for B3/B4, train/ for B4) straight into the run directory: `python -m laya_poc.baseline_runs {majority,prior,
tfidf_lr}`, `python -m laya_poc.small_encoder` or `python -m laya_poc.llm_baseline`. Their data dirs follow the
Laya runs of the same scheme (c10 -> data, c7 -> data_c7, trap candidates from data_eval). Pure apart from the
files it is pointed at.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import labels
from .matrix_exec import RunContext, read_json
from .matrix_plan import (FIT_KINDS, LAYA, DataPaths, RunSpec, TrainOverrides, computes_order_invariance,
                          expand_specs, train_epochs)
from .matrix_results import num

SMALL_ENCODER, LLM = "small_encoder", "llm"
KINDS = (*FIT_KINDS, SMALL_ENCODER, LLM)
EXPANDS = {"majority": ("majority", "prior")}     # B1 = the train majority class AND the train prior
MODEL_TABLES = {SMALL_ENCODER: "small_encoders", LLM: "llms"}
MODULES = {**{k: "baseline_runs" for k in FIT_KINDS}, SMALL_ENCODER: "small_encoder", LLM: "llm_baseline"}
STEPS = {**{k: "baseline" for k in FIT_KINDS}, SMALL_ENCODER: "small_encoder", LLM: "llm_baseline"}
CALIBRATION = "calibration.json"


# ---------------------------------------------------------------- expansion

def _phase4(cfg: dict) -> dict:
    p4 = cfg.get("phase4")
    if p4 is None:
        return {}
    if not isinstance(p4, dict):
        raise ValueError("config phase4 must be a mapping (card, arms, small_encoders, llms, ...)")
    return p4


def _schemes(where: str, arm: dict) -> list[str]:
    raw = arm["schemes"] if "schemes" in arm else [arm.get("scheme")]
    if not isinstance(raw, list) or not raw or any(s not in labels.SCHEMES for s in raw):
        raise ValueError(f"{where}: scheme(s) {raw!r} must be from {sorted(labels.SCHEMES)}")
    return raw


def _model(where: str, arm: dict, kind: str, p4: dict) -> str:
    table = p4.get(MODEL_TABLES[kind]) or {}
    if arm.get("model") not in table:
        raise ValueError(f"{where}: model {arm.get('model')!r} is not in phase4.{MODEL_TABLES[kind]} {sorted(table)}")
    return arm["model"]


def _seeds(where: str, arm: dict) -> list[int]:
    seeds = arm.get("seeds")
    ok = isinstance(seeds, list) and seeds and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                                                   for v in seeds)
    if not ok:
        raise ValueError(f"{where}: seeds must be a non-empty list of integers >= 0, got {seeds!r}")
    return seeds


def _arm_specs(i: int, arm: Any, p4: dict) -> list[RunSpec]:
    where = f"phase4.arms[{i}]"
    if not isinstance(arm, dict) or not isinstance(arm.get("id"), str) or not arm["id"]:
        raise ValueError(f"{where}: needs an id (e.g. B3), got {arm!r}")
    kind, arm_id, optional = arm.get("kind"), arm["id"], bool(arm.get("optional", False))
    if kind not in KINDS:
        raise ValueError(f"{where}: kind {kind!r} is not one of {list(KINDS)}")
    schemes = _schemes(where, arm)
    if kind in FIT_KINDS:
        return [RunSpec(arm_id, k, sc, kind=k, optional=optional) for sc in schemes for k in EXPANDS.get(kind, (kind,))]
    model = _model(where, arm, kind, p4)
    if kind == LLM:
        return [RunSpec(arm_id, model, sc, kind=kind, optional=optional) for sc in schemes]
    return [RunSpec(arm_id, model, sc, seed, kind=kind, optional=optional) for sc in schemes
            for seed in _seeds(where, arm)]


def expand_phase4(cfg: dict) -> tuple[RunSpec, ...]:
    """Every configured baseline run, in config order (none when the config has no phase4 section)."""
    p4 = _phase4(cfg)
    return tuple(s for i, arm in enumerate(p4.get("arms") or []) for s in _arm_specs(i, arm, p4))


def expand_all(cfg: dict) -> tuple[RunSpec, ...]:
    """The Phase 3 matrix (zero-shot first, then the arms) followed by the Phase 4 baselines."""
    specs = (*expand_specs(cfg), *expand_phase4(cfg))
    names = [s.name for s in specs]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"phase3/phase4: duplicate runs {dupes}")
    return specs


# ---------------------------------------------------------------- per kind

def module(spec: RunSpec) -> str:
    return MODULES[spec.kind]


def step_name(spec: RunSpec) -> str:
    return STEPS[spec.kind]


def init_family(spec: RunSpec) -> str | None:
    """What a local --init dir must be for this run: a Laya checkpoint, an HF encoder, a causal LM, or None
    (B1/B3 have no pretrained model)."""
    return None if spec.kind in FIT_KINDS else spec.kind


def epochs(cfg: dict, spec: RunSpec, laya: TrainOverrides, baseline: TrainOverrides) -> int:
    """Epoch files a run trains on: train.epochs for Laya, phase4.small_encoder_train.epochs for B4 (either
    overridden by an --epochs in the extra flags), none for the others."""
    if spec.kind == LAYA:
        return train_epochs(cfg, laya)
    if spec.kind == SMALL_ENCODER:
        return baseline.epochs or int(_phase4(cfg).get("small_encoder_train", {}).get("epochs", 3))
    return 0


def baseline_args(ctx: RunContext, spec: RunSpec, paths: DataPaths, splits: Sequence[str],
                  extra: Sequence[str] = ()) -> list[str]:
    """The CLI arguments of a baseline run (spec P4 §2-§4); `extra` (--baseline-extra) comes last so it wins."""
    common: list[Any] = ["--scheme", spec.scheme, "--data-dir", paths.eval, "--run-dir", ctx.run_dir,
                         "--splits", *splits]
    for name, path in paths.extras:
        common += ["--extra", f"{name}={path}"]
    common += ["--config", ctx.config_path]
    local_init = [] if ctx.init == "hub" else ["--init", ctx.init]
    if spec.kind in FIT_KINDS:
        args = [spec.kind, *common, *(["--order-invariance"] if computes_order_invariance(spec) else [])]
    elif spec.kind == SMALL_ENCODER:
        args = ["--model", spec.model, "--seed", spec.seed, *common, "--device", ctx.device, *local_init]
    else:
        args = ["--model", spec.model, *common, "--device", ctx.device, *local_init]
    return [str(a) for a in (*args, *extra)]


# ---------------------------------------------------------------- outputs

def _calibration_problem(run_dir: Path) -> str | None:
    cal = read_json(run_dir / CALIBRATION)
    if not isinstance(cal, dict) or num(cal.get("T")) is None:
        return f"{CALIBRATION} with a temperature T"
    return None


def outputs_problem(run_dir: Path, spec: RunSpec, splits: Sequence[str]) -> str | None:
    """What a finished baseline run lacks (None = complete): eval/ + preds/ for every split, calibration.json,
    order_invariance.json (B3/B4) and train/summary.json (B4)."""
    need = [*(f"eval/{s}.json" for s in splits), *(f"preds/{s}.jsonl" for s in splits)]
    need += ["order_invariance.json"] if computes_order_invariance(spec) else []
    need += ["train/summary.json"] if spec.kind == SMALL_ENCODER else []
    missing = [rel for rel in need if not (run_dir / rel).is_file()]
    missing += [p for p in [_calibration_problem(run_dir)] if p]
    return f"missing {', '.join(missing)} in {run_dir}" if missing else None


def required_splits(spec: RunSpec, splits: Sequence[str]) -> tuple[str, ...]:
    """The splits a baseline must have preds for (Phase 4 exit): every eval split; B5 scores each on its
    configured subset (phase4.llm_eval), which is still one preds file per split."""
    return tuple(splits)
