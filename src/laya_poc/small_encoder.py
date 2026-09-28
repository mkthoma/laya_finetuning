"""B4 CLI: a fine-tuned small encoder baseline, trained, calibrated and evaluated like a Laya run (design §5.13 item 4).

    python -m laya_poc.small_encoder --model modernbert_base|mmbert_small --seed S --scheme c10 --data-dir D
        --run-dir R [--extra name=abs_path ...] [--splits ...] [--device cuda|cpu] [--init <abs local HF model dir>]
        [--max-steps N] [--epochs N] [--config yaml]

1. fine-tune (small_encoder_train.py) on D/train_e{0..epochs-1}.jsonl, the epoch picked on val macro-F1;
2. T on the best epoch's raw val logits (baseline_eval.fit_T, hard labels, labelled rows with evidence);
3. every split through the shared baseline writer (baseline_eval), so the numbers mean what a Laya run's mean:
   each split is scored on ALL its rows and the writer applies the no-evidence gate (§5.8.1) and the val tau; for
   stripped_test the model's own answers on the country-only rows become the no_gate block and preds;
4. field-order invariance (config phase3.order_invariance: test_id, 1000 rows, 5 orders) on rows with evidence.
Splits: --splits (default config phase3.eval_splits) read from D/<split>.jsonl, or from an --extra name=path (an
--extra not in --splits is evaluated too); every file must exist, and val must be among them. Each split's rows
must carry labels.question(--scheme), checked before training starts.
Run layout under R (spec P4 §1): train/{log.jsonl, summary.json, best/}, calibration.json, eval/<split>.json,
preds/<split>.jsonl (+ preds/no_gate/stripped_test.jsonl), order_invariance.json. A re-run first deletes these
(and a stale done.json: the matrix writes done.json only after this CLI succeeds); anything else in R is kept.
The HF token (public models need none) is read from HF_TOKEN only and never printed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import baseline_eval as BE
from .config import load_config
from .env_check import run_cli, write_json
from .evaluate import check_question, evidence_mask, label_indices, load_rows
from .evaluate_splits import TAU_SPLIT, parse_extras, split_paths
from .labels import SCHEMES, option_keys, question
from .schedule import abs_path
from .small_encoder_data import score_states, val_set
from .small_encoder_train import BEST_DIR, TrainConfig, encoder_source, load_encoder, train_config, train_encoder

KIND = "small_encoder"
PROG = "small_encoder"
TRAIN_DIR = "train"
TRACEBACK = f"{PROG}.traceback.txt"
OWNED = (TRAIN_DIR, BE.EVAL_DIR, BE.PREDS_DIR, BE.CALIBRATION_FILE, BE.ORDER_FILE, "done.json", TRACEBACK)


@dataclass(frozen=True)
class Inputs:
    epoch_files: list[Path]
    paths: dict[str, Path]           # split -> file, val first
    rows: dict[str, list[dict]]
    order: dict[str, Any]            # {split, n, perms, seed}
    order_path: Path
    order_rows: list[dict]


# ---------------------------------------------------------------- inputs and run dir

def _checked_rows(path: Path, scheme: str) -> list[dict]:
    rows = load_rows(path)
    check_question(rows, question(scheme))
    return rows


def resolve_inputs(args: argparse.Namespace, cfg: dict, data_dir: Path, tc: TrainConfig) -> Inputs:
    """Every file the run reads, loaded and checked BEFORE training (a bad split must not cost a training run)."""
    epoch_files = [data_dir / f"train_e{e}.jsonl" for e in range(tc.epochs)]
    missing = [f.name for f in epoch_files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"{data_dir} lacks {', '.join(missing)}: B4 trains {tc.epochs} epoch(s) on the "
                                f"augmented train_e*.jsonl files build_data writes (see --epochs)")
    splits = list(args.splits or (cfg.get("phase3") or {}).get("eval_splits") or [TAU_SPLIT])
    extras = parse_extras(args.extra)
    if TAU_SPLIT not in splits and TAU_SPLIT not in extras:
        raise ValueError(f"--splits must include {TAU_SPLIT}: it picks the epoch, fits T and sets tau")
    paths, _ = split_paths(splits, data_dir, extras, ())
    rows = {s: _checked_rows(p, args.scheme) for s, p in paths.items()}
    order = BE.order_config(cfg)
    order_path = paths.get(order["split"]) or extras.get(order["split"]) or data_dir / f"{order['split']}.jsonl"
    if not order_path.is_file():
        raise FileNotFoundError(f"order-invariance split {order['split']}: {order_path} not found "
                                f"(config phase3.order_invariance.split)")
    order_rows = rows.get(order["split"]) or _checked_rows(order_path, args.scheme)
    return Inputs(epoch_files=epoch_files, paths=paths, rows=rows, order=order, order_path=order_path,
                  order_rows=order_rows)


def reset_outputs(run_dir: Path) -> list[str]:
    """Delete this CLI's earlier outputs (and a stale done.json) so a re-run starts clean; other files stay."""
    removed = []
    for name in OWNED:
        path = run_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            continue
        removed.append(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return removed


def _check_device(device: str) -> None:
    import torch
    from transformers.utils import logging as hf_logging

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but no CUDA device (use --device cpu for local runs)")
    hf_logging.disable_progress_bar()  # sparse notebook output: no load/save bars (the load report stays)


# ---------------------------------------------------------------- calibration and evaluation

def fit_calibration(Z: np.ndarray, rows: list[dict], keys: list[str], evidence_fields: list[str]
                    ) -> tuple[float, bool, int]:
    """(T, clamped, n) on the val rows the calibrated model answers: labelled, with evidence."""
    y = label_indices(rows, keys)
    use = evidence_mask(rows, evidence_fields) & np.array([v is not None for v in y], dtype=bool)
    if not use.any():
        raise ValueError(f"{TAU_SPLIT} has no labelled rows with evidence: cannot fit T")
    T, clamped = BE.fit_T(Z[use], np.array([v for v, u in zip(y, use) if u], dtype=int))
    return T, clamped, int(use.sum())


def _f(node: dict | None, key: str) -> str:
    v = (node or {}).get(key)
    return "n/a" if v is None else f"{v:.4f}"


def _line(p: dict) -> str:
    return (f"{PROG} {p['model']} {p['split']}: macro-F1 {_f(p.get('post'), 'macro_f1')} (9-class "
            f"{_f(p.get('post'), 'macro_f1_9')}), acc {_f(p.get('post'), 'acc')}, ECE {_f(p.get('pre'), 'ece')}->"
            f"{_f(p.get('post'), 'ece')} at T={p['temperature']:.4f}; {p['n_scored']} scored, "
            f"{p['n_abstained_no_evidence']} abstained (no evidence)")


def evaluate_run(model: Any, tok: Any, tc: TrainConfig, cfg: dict, inputs: Inputs, run_dir: Path,
                 ident: dict[str, Any]) -> dict[str, Any]:
    """calibration.json, then eval/ + preds/ for every split (val first: its tau goes to the others)."""
    keys, fields = option_keys(ident["scheme"]), list(cfg["serialise"]["evidence_fields"])
    scores, secs = {}, {}
    for split, rows in inputs.rows.items():
        t0 = time.perf_counter()
        scores[split] = _score(model, tok, tc, [r["state"] for r in rows])
        secs[split] = round(time.perf_counter() - t0, 2)
    T, clamped, n = fit_calibration(scores[TAU_SPLIT], inputs.rows[TAU_SPLIT], keys, fields)
    extra = {"labels": "hard", "scores": "raw logits of the best epoch", "best_epoch": ident["best_epoch"],
             "model": ident["model"], "seed": tc.seed}
    write_json(run_dir / BE.CALIBRATION_FILE, BE.calibration_record(T=T, clamped=clamped, n=n, extra=extra))
    print(f"{PROG}: T {T:.4f} on {TAU_SPLIT} (n {n}{', CLAMPED' if clamped else ''})", flush=True)
    tau, payloads = None, {}
    for split, rows in inputs.rows.items():
        meta = {"rows": str(inputs.paths[split]), "device": tc.device, "batch_size": tc.eval_batch_size,
                "max_len": tc.max_length, "kind": KIND, "seconds": secs[split]}
        payloads[split] = BE.write_outputs(run_dir / BE.EVAL_DIR, run_dir / BE.PREDS_DIR, rows=rows, keys=keys,
                                           scores=scores[split], T=T, split=split, model=ident["model"],
                                           ckpt=ident["ckpt"], evidence_fields=fields,
                                           abstain_cfg=cfg.get("abstain"), tau=tau, meta=meta)
        tau = payloads[split]["tau"] if split == TAU_SPLIT else tau
        print(_line(payloads[split]), flush=True)
    return {"T": T, "clamped": clamped, "splits": list(payloads)}


def _score(model: Any, tok: Any, tc: TrainConfig, states: list[str]) -> np.ndarray:
    return score_states(model, tok, states, max_length=tc.max_length, batch_size=tc.eval_batch_size,
                        device=tc.device, fp16=tc.fp16)


def order_run(model: Any, tok: Any, tc: TrainConfig, cfg: dict, inputs: Inputs, run_dir: Path,
              ident: dict[str, Any]) -> dict[str, Any]:
    o = inputs.order
    res = BE.order_invariance_payload(lambda states: _score(model, tok, tc, states).argmax(1), inputs.order_rows,
                                      model=ident["model"], ckpt=ident["ckpt"], split=o["split"],
                                      rows_path=str(inputs.order_path),
                                      evidence_fields=cfg["serialise"]["evidence_fields"], n=int(o["n"]),
                                      perms=int(o["perms"]), seed=int(o["seed"]))
    write_json(run_dir / BE.ORDER_FILE, res)
    print(f"{PROG}: order invariance on {o['split']}: mean agreement {res['mean_agreement']:.4f} over "
          f"{res['perms']} orders x {res['n']} rows ({'PASS' if res['passed_99'] else 'FAIL'} >= 0.99)", flush=True)
    return res


# ---------------------------------------------------------------- CLI

def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    run_dir, data_dir = abs_path(args.run_dir, "--run-dir"), abs_path(args.data_dir, "--data-dir")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir {data_dir} does not exist")
    cfg = load_config(args.config)
    _check_device(args.device)
    src = encoder_source(cfg, args.model, args.init)
    tc = train_config(cfg, seed=args.seed, device=args.device, epochs=args.epochs, max_steps=args.max_steps)
    inputs = resolve_inputs(args, cfg, data_dir, tc)
    removed = reset_outputs(run_dir)
    if removed:
        print(f"{PROG}: replaced earlier outputs in {run_dir}: {', '.join(removed)}", flush=True)
    keys = option_keys(args.scheme)
    print(f"{PROG}: loading {src.label} ({len(keys)} labels, {args.scheme})", flush=True)
    model, tok = load_encoder(src, len(keys), tc.seed)
    meta = {"model": args.model, "kind": KIND, "hub_id": src.repo_id, "revision": src.revision,
            "init": None if src.init is None else str(src.init), "scheme": args.scheme}
    summary = train_encoder(model, tok, tc, inputs.epoch_files, val_set(inputs.rows[TAU_SPLIT], keys),
                            run_dir / TRAIN_DIR, meta=meta)
    ident = {"model": args.model, "scheme": args.scheme, "ckpt": str(run_dir / TRAIN_DIR / BEST_DIR),
             "best_epoch": summary["best_epoch"]}
    ev = evaluate_run(model, tok, tc, cfg, inputs, run_dir, ident)
    oi = order_run(model, tok, tc, cfg, inputs, run_dir, ident)
    return {"summary": summary, **ev, "order_invariance": oi["mean_agreement"],
            "seconds": round(time.perf_counter() - t0, 2)}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=f"python -m laya_poc.{PROG}", description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="config phase4.small_encoders key: modernbert_base | mmbert_small")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--scheme", choices=sorted(SCHEMES), default="c10")
    p.add_argument("--data-dir", required=True, help="absolute data dir: train_e{k}.jsonl and the split files")
    p.add_argument("--run-dir", required=True, help="absolute run dir (spec P4 §1 layout)")
    p.add_argument("--extra", nargs="+", action="extend", default=None, metavar="NAME=ABS_PATH",
                   help="split files outside --data-dir (e.g. trap_candidates=/abs/trap_candidates.jsonl)")
    p.add_argument("--splits", nargs="+", default=None, help="default: config phase3.eval_splits (val required)")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--init", default=None, help="absolute local HF model dir instead of the pinned Hub revision")
    p.add_argument("--max-steps", type=int, default=None, help="stop after N optimiser steps (dry runs)")
    p.add_argument("--epochs", type=int, default=None, help="default: config phase4.small_encoder_train.epochs")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        res = run(args)
        s = res["summary"]
        print(f"{PROG}: done: best epoch {s['best_epoch']} (val macro-F1 {s['best_val_macro_f1']:.4f}), T "
              f"{res['T']:.4f}, {len(res['splits'])} splits, order invariance {res['order_invariance']:.4f} -> "
              f"{args.run_dir} ({res['seconds']:.0f}s)")
        return 0

    detail = Path(args.run_dir) / PROG if Path(args.run_dir).is_absolute() else None
    return run_cli(PROG, body, detail)


if __name__ == "__main__":
    sys.exit(main())
