"""Multi-split evaluate (Phase 3, design §7.9 steps 1-6 for one run): ONE agent load, every split.

    python -m laya_poc.evaluate --ckpt hub|<abs Laya dir> [--model laya|laya_ml] [--scheme c10|c7]
        --splits val test_id ood_country ... --data-dir <dir> [--extra <name>=<abs jsonl> ...] [--optional <split> ...]
        --out-dir <dir> --preds-dir <dir> [--order-invariance-out <json> [--order-split test_id] [--order-n 1000]
        [--order-perms 5] [--order-seed S]] [--device cuda|cpu] [--batch-size N] [--n N] [--config yaml]

Split files are <data-dir>/<split>.jsonl, or the --extra path for that name (an --extra not in --splits is
evaluated too). A missing file is an error, found before the agent loads, unless the split is --optional.
The question is labels.question(--scheme, else config labels.scheme) as in single-split mode; a split
whose rows carry another scheme's question fails before the agent loads, naming the --scheme to pass.
`val` is evaluated first: tau (robustness.choose_tau, design §5.8.2, config `abstain`) is chosen on its
post-T scored rows and applied to every split. Outputs:
  <out-dir>/<split>.json    the single-split report + `abstention` (at tau); val also `tau`; stripped_test also
                            abstain_rate_at_tau, false_confident_rate (> 0.8) and `no_gate`
  <preds-dir>/<split>.jsonl {id, y, p, answer_confidence, abstained} post-T, as in single-split mode
  <preds-dir>/no_gate/stripped_test.jsonl  the model's answers on the stripped rows (gate bypassed)
  --order-invariance-out    {split, n, perms, seed, agreement_per_perm, mean_agreement, passed_99, ...}
Every stripped_test row is country-only, so the deterministic gate abstains on all of them; its no-evidence
metrics (§5.11) are therefore measured on the model's own answers with the gate bypassed.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .config import load_config
from .env_check import run_cli, write_json
from .evaluate import (_line, check_question, eval_settings, evaluate_rows, evidence_mask, label_indices, load_checked,
                       load_rows, pred_records, predict_pass)
from .fallback_guard import FallbackGuard
from .io_utils import write_jsonl
from .labels import SCHEMES, question
from .robustness import (FALLBACK_COVERAGE, FALSE_CONFIDENT_AT, abstention, choose_tau, conf_correct,
                         false_confident_rate, order_invariance)

TAU_SPLIT = "val"
STRIPPED_SPLIT = "stripped_test"
NO_GATE_DIR = "no_gate"
TEMPERATURE_ATTRS = ("temperature", "temperature_by_options", "lang_temperatures")  # read by Laya's decode
ORDER_DEFAULTS = {"split": "test_id", "n": 1000, "perms": 5, "seed": 20260925}     # config phase3.order_invariance
MULTI_ONLY = ("extra", "optional", "out_dir", "preds_dir", "order_invariance_out", "order_split", "order_n",
              "order_perms", "order_seed")
SINGLE_ONLY = ("rows", "split", "out", "preds")


# ---------------------------------------------------------------- arguments

def add_multi_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("multi-split mode (Phase 3): one agent load, every split")
    g.add_argument("--splits", nargs="+", default=None, help="split names: <data-dir>/<split>.jsonl")
    g.add_argument("--extra", nargs="+", action="extend", default=None, metavar="NAME=ABS_PATH",
                   help="split files outside --data-dir (e.g. trap_candidates=/abs/trap_candidates.jsonl)")
    g.add_argument("--optional", nargs="+", action="extend", default=None, help="splits allowed to be missing")
    g.add_argument("--out-dir", default=None, help="writes <out-dir>/<split>.json")
    g.add_argument("--preds-dir", default=None, help="writes <preds-dir>/<split>.jsonl")
    g.add_argument("--order-invariance-out", default=None, help="field-order invariance JSON (design §7.9 step 6)")
    g.add_argument("--order-split", default=None, help="default: config phase3.order_invariance.split (test_id)")
    g.add_argument("--order-n", type=int, default=None, help="rows (default: config, 1000)")
    g.add_argument("--order-perms", type=int, default=None, help="shuffled field orders (default: config, 5)")
    g.add_argument("--order-seed", type=int, default=None, help="default: config phase3.order_invariance.seed")


def _flags(args: argparse.Namespace, names: Sequence[str]) -> list[str]:
    return [f"--{k.replace('_', '-')}" for k in names if getattr(args, k) is not None]


def check_mode(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Single-split (--rows/--split + --out) and multi-split (--splits + dirs) flags never mix."""
    if args.splits is None:
        if _flags(args, MULTI_ONLY):
            p.error(f"{', '.join(_flags(args, MULTI_ONLY))}: multi-split flags need --splits")
        if not args.out:
            p.error("--out is required (single-split mode; use --splits ... for multi-split mode)")
        if not args.rows and not args.split:
            p.error("one of --rows or --split is required")
        return
    if _flags(args, SINGLE_ONLY):
        p.error(f"{', '.join(_flags(args, SINGLE_ONLY))}: single-split flags; --splits writes "
                f"<out-dir>/<split>.json and <preds-dir>/<split>.jsonl")
    missing = [f for f in ("data_dir", "out_dir", "preds_dir") if not getattr(args, f)]
    if missing:
        p.error(f"--splits needs {', '.join('--' + f.replace('_', '-') for f in missing)}")
    for k in ("order_n", "order_perms"):
        if getattr(args, k) is not None and getattr(args, k) < 1:
            p.error(f"--{k.replace('_', '-')} must be >= 1")


def parse_extras(items: Sequence[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items or []:
        name, sep, path = item.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"--extra {item!r}: expected name=<absolute path>")
        if not Path(path).is_absolute():
            raise ValueError(f"--extra {name}: the path must be absolute, got {path!r}")
        if name in out:
            raise ValueError(f"--extra {name} is given twice")
        out[name] = Path(path)
    return out


def split_paths(splits: Sequence[str], data_dir: str | Path, extras: dict[str, Path],
                optional: Sequence[str]) -> tuple[dict[str, Path], list[str]]:
    """({split: path} with the tau split first, skipped optional splits). Missing required files: one error."""
    names = list(dict.fromkeys([*splits, *extras]))
    paths = {s: extras.get(s, Path(data_dir) / f"{s}.jsonl") for s in names}
    absent = [s for s in names if not paths[s].exists()]
    required = [f"{s} ({paths[s]})" for s in absent if s not in set(optional)]
    if required:
        raise FileNotFoundError(f"missing split file(s): {', '.join(required)}; build them (build_data / "
                                f"variants) or list the split in --optional")
    kept = sorted((s for s in names if s not in absent), key=lambda s: s != TAU_SPLIT)
    if not kept:
        raise FileNotFoundError(f"none of the splits exist ({', '.join(names)}); nothing to evaluate")
    return {s: paths[s] for s in kept}, absent


# ---------------------------------------------------------------- per split

@dataclass(frozen=True)
class Ctx:
    model: str
    ckpt: str
    device: str
    settings: dict
    n: int | None
    abstain: dict

    @property
    def budget(self) -> dict[str, int]:
        return {k: self.settings[k] for k in ("batch_size", "max_len", "head_max_len")}


@contextmanager
def preserved_temperatures(agent: Any) -> Iterator[None]:
    """evaluate_rows neutralises the temperatures for its PRE pass; put them back so the next split's POST
    pass (and the no-gate / order passes) run at the checkpoint's own T on the same loaded agent."""
    saved = {a: copy.deepcopy(getattr(agent, a)) for a in TEMPERATURE_ATTRS if hasattr(agent, a)}
    try:
        yield
    finally:
        for a, v in saved.items():
            setattr(agent, a, v)


def evaluate_split(agent: Any, name: str, path: Path, ctx: Ctx, preds_path: Path) -> tuple[dict, list[dict], list]:
    """(report without `seconds`, post-T prediction records, rows) for one split, temperatures restored."""
    s, guard = ctx.settings, FallbackGuard()
    rows = load_rows(path, ctx.n)
    check_question(rows, s["question"])
    with preserved_temperatures(agent):
        res, preds = evaluate_rows(agent, rows, s["question"], evidence_fields=s["evidence_fields"], guard=guard,
                                   label=f"evaluate {name}", **ctx.budget)
    return ({"model": ctx.model, "ckpt": ctx.ckpt, "split": name, "rows": str(path), "scheme": s["scheme"],
             "labels": s["keys"], **res, "cpu_fallback": guard.cpu_fallback, "device": ctx.device, **ctx.budget,
             "preds": str(preds_path)}, preds, rows)


def selective(preds: Sequence[dict], tau: dict | None) -> dict[str, Any]:
    """Coverage and answered accuracy at the val tau on this split's scored rows (post-T)."""
    conf, correct = conf_correct(preds)
    return {**abstention(conf, correct, None if tau is None else tau["tau"]), "tau_split": TAU_SPLIT,
            "tau_rule": None if tau is None else tau["rule"]}


def no_gate(agent: Any, rows: Sequence[dict], ctx: Ctx, tau: float | None) -> tuple[dict, list[dict]]:
    """The model's own answers on the (country-only) stripped rows, gate bypassed, post-T."""
    s, guard = ctx.settings, FallbackGuard()
    with guard:
        P, conf, _, secs = predict_pass(agent, [r["state"] for r in rows], s["question"], s["keys"],
                                        label=f"evaluate {STRIPPED_SPLIT} no-gate", **ctx.budget)
    at = abstention(conf, None, tau)
    block = {"n": len(rows), "tau": tau, "abstain_rate_at_tau": at["abstain_rate"],
             "false_confident_rate": false_confident_rate(conf), "false_confident_threshold": FALSE_CONFIDENT_AT,
             "mean_answer_confidence": float(conf.mean()) if len(conf) else None,
             "cpu_fallback": guard.cpu_fallback, "seconds": secs}
    return block, pred_records(rows, label_indices(rows, s["keys"]), P, conf, np.zeros(len(rows), dtype=bool))


def stripped_fields(agent: Any, res: dict, rows: Sequence[dict], ctx: Ctx, tau: float | None,
                    preds_dir: Path) -> dict[str, Any]:
    """stripped_test extras (§5.11 "No evidence"): model-level abstain rate at tau and false-confident rate."""
    block, recs = no_gate(agent, rows, ctx, tau)
    path = preds_dir / NO_GATE_DIR / f"{STRIPPED_SPLIT}.jsonl"
    write_jsonl(path, recs)
    return {"abstain_rate_at_tau": block["abstain_rate_at_tau"], "false_confident_rate": block["false_confident_rate"],
            "false_confident_threshold": FALSE_CONFIDENT_AT,
            "gate_abstain_rate": res["n_abstained_no_evidence"] / res["n"] if res["n"] else None,
            "no_evidence_basis": "model answers with the no-evidence gate bypassed (with it, every country-only "
                                 "row abstains)", "no_gate": block, "no_gate_preds": str(path),
            "cpu_fallback": res["cpu_fallback"] or block["cpu_fallback"]}


def evaluate_all(agent: Any, paths: dict[str, Path], ctx: Ctx, out_dir: Path, preds_dir: Path) -> dict[str, dict]:
    """Every split in order (tau split first): report + preds written as each split finishes."""
    tau, results = None, {}
    if TAU_SPLIT not in paths:
        print(f"evaluate: WARNING: no tau ({TAU_SPLIT} not evaluated): abstention blocks carry tau null", flush=True)
    for name, path in paths.items():
        t0, preds_path = time.perf_counter(), preds_dir / f"{name}.jsonl"
        res, preds, rows = evaluate_split(agent, name, path, ctx, preds_path)
        if name == TAU_SPLIT:
            tau = choose_tau(*conf_correct(preds), **ctx.abstain)
            res = {**res, "tau": tau}
            print(f"evaluate: tau {tau['tau']} on {TAU_SPLIT} (rule {tau['rule']}, n {tau['n']})", flush=True)
        res = {**res, "abstention": selective(preds, tau)}
        if name == STRIPPED_SPLIT:
            res = {**res, **stripped_fields(agent, res, rows, ctx, None if tau is None else tau["tau"], preds_dir)}
        write_jsonl(preds_path, preds)
        res = {**res, "seconds": round(time.perf_counter() - t0, 2)}
        write_json(out_dir / f"{name}.json", res)
        print(_line(res), flush=True)
        results[name] = res
    return results


# ---------------------------------------------------------------- order invariance

def order_settings(cfg: dict, args: argparse.Namespace) -> dict[str, Any]:
    oi = {**ORDER_DEFAULTS, "perms": int(cfg.get("eval", {}).get("order_perms", ORDER_DEFAULTS["perms"])),
          **((cfg.get("phase3") or {}).get("order_invariance") or {})}
    flags = {"split": args.order_split, "n": args.order_n, "perms": args.order_perms, "seed": args.order_seed}
    return {k: (v if flags[k] is None else flags[k]) for k, v in oi.items() if k in flags}


def run_order(agent: Any, order: dict, path: Path, ctx: Ctx, out_path: str | Path) -> dict[str, Any]:
    """Order invariance on the first n rows of the split that the model answers (the gate skips the rest)."""
    s = ctx.settings
    rows = load_rows(path)
    check_question(rows, s["question"])
    answered = [r for r, a in zip(rows, evidence_mask(rows, s["evidence_fields"])) if a]
    with preserved_temperatures(agent):
        res = order_invariance(agent, answered, s["question"], n=int(order["n"]), perms=int(order["perms"]),
                               seed=int(order["seed"]), **ctx.budget)
    out = {"model": ctx.model, "ckpt": ctx.ckpt, "split": order["split"], "rows": str(path), **res,
           "n_skipped_no_evidence": len(rows) - len(answered)}
    write_json(out_path, out)
    print(f"evaluate: order invariance on {order['split']}: mean agreement {res['mean_agreement']:.4f} over "
          f"{res['perms']} orders x {res['n']} rows ({'PASS' if res['passed_99'] else 'FAIL'} >= 0.99)", flush=True)
    return out


# ---------------------------------------------------------------- CLI

def rows_scheme(path: Path) -> str | None:
    """The label scheme whose question the first row carries EXACTLY (option order included), else None."""
    row = load_rows(path, 1)[0]
    if "questions" not in row:
        return None
    got = json.loads(row["questions"])

    def order(q: dict) -> list[list[str]]:
        return [list(v["criteria"]) for v in q.values()]

    return next((s for s in sorted(SCHEMES) if got == question(s) and order(got) == order(question(s))), None)


def precheck_questions(paths: dict[str, Path], scheme: str) -> None:
    """Fail before the agent loads when a split's first row carries another question than the run's scheme
    (--scheme, else config labels.scheme); every row is checked again when its split is evaluated."""
    for name, path in paths.items():
        got = rows_scheme(path)
        if got is not None and got != scheme:
            raise ValueError(f"{name} rows carry the {got} question but the scheme is {scheme}: pass --scheme "
                             f"{got} (or labels.scheme in --config), or point at the {scheme} data")


def _ctx(cfg: dict, args: argparse.Namespace, device: str) -> Ctx:
    ab = cfg.get("abstain") or {}
    abstain = {"target_acc": float(ab.get("target_acc", 0.95)), "min_coverage": float(ab.get("min_coverage", 0.70)),
               "fallback_coverage": float(ab.get("fallback_coverage", FALLBACK_COVERAGE))}
    return Ctx(model=args.model, ckpt=args.ckpt, device=device, settings=eval_settings(cfg, args), n=args.n,
               abstain=abstain)


def _order_file(split: str, paths: dict[str, Path], data_dir: str | Path, extras: dict[str, Path]) -> Path:
    path = paths.get(split) or extras.get(split) or Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"order-invariance split {split}: {path} not found (see --order-split)")
    return path


def run_multi(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from .parity import free_memory, resolve_source

    cfg = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but no CUDA device (use --device cpu for local runs)")
    source = resolve_source(cfg, args.model, args.ckpt)  # validates the path before any load
    extras = parse_extras(args.extra)
    paths, skipped = split_paths(args.splits, args.data_dir, extras, args.optional or [])
    order = order_settings(cfg, args) if args.order_invariance_out else None
    order_file = None if order is None else _order_file(order["split"], paths, args.data_dir, extras)
    ctx = _ctx(cfg, args, args.device)
    precheck_questions({**paths, **({order["split"]: order_file} if order else {})}, ctx.settings["scheme"])
    absent = f" (optional, absent: {', '.join(skipped)})" if skipped else ""
    print(f"evaluate {args.model} ({args.ckpt}) on {', '.join(paths)}{absent}, scheme {ctx.settings['scheme']}, "
          f"{args.device}", flush=True)
    t0, agent = time.perf_counter(), load_checked(source, args.device)
    try:
        ctx = replace(ctx, device=agent.device.type)
        results = evaluate_all(agent, paths, ctx, Path(args.out_dir), Path(args.preds_dir))
        oi = None if order is None else run_order(agent, order, order_file, ctx, args.order_invariance_out)
    finally:
        del agent
        free_memory()
    return {"splits": list(results), "skipped": skipped, "order_invariance": oi,
            "seconds": round(time.perf_counter() - t0, 2)}


def main_multi(args: argparse.Namespace) -> int:
    def body() -> int:
        res = run_multi(args)
        print(f"evaluate: wrote {len(res['splits'])} split reports to {args.out_dir} and predictions to "
              f"{args.preds_dir} ({res['seconds']:.0f}s)")
        return 0

    return run_cli("evaluate", body, Path(args.out_dir) / "evaluate")
