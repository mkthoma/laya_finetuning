"""Zero-shot accuracy of the shipped checkpoints on a few prepared records (design doc §6.2 task 4).

Uses the shipped temperatures (this is what a user of the stock model sees) and the compact JSON
*string* states, exactly as the fine-tuned model will be queried. One model is loaded at a time
and freed before the next, to fit Colab's RAM/VRAM. Each model records "cpu_fallback": whether Laya
answered any batch on CPU after a CUDA OOM (it reports cuda again afterwards; see fallback_guard).

    python -m laya_poc.zeroshot --models laya laya_ml [--init hub|<abs dir>] --rows <val.jsonl> --n 20
                                --out <json> [--device cuda|cpu] [--config yaml]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from typing import Any, Iterable

from .config import load_config
from .env_check import run_cli, write_json
from .fallback_guard import FallbackGuard
from .io_utils import iter_jsonl
from .parity import free_memory, resolve_source


def select_rows(rows: Iterable[dict], n: int) -> list[dict]:
    """The first n rows that have a gold label (no-evidence rows have label null)."""
    picked = list(itertools.islice((r for r in rows if r.get("label") is not None), n))
    if not picked:
        raise ValueError("no labelled rows (label is null everywhere)")
    return picked


def shared_question(rows: list[dict]) -> dict:
    """The question object shared by all rows; predict_batch poses one question set to every state."""
    parsed = [json.loads(r["questions"]) for r in rows]
    if any(q != parsed[0] for q in parsed[1:]):
        raise ValueError("rows do not all carry the same question")
    return parsed[0]


def summarise(rows: list[dict], answers: list[dict]) -> dict[str, Any]:
    per_row = [{"id": r["id"], "label": r["label"], "choice": a["choice"], "answer_confidence": a["answer_confidence"]}
               for r, a in zip(rows, answers, strict=True)]
    correct = sum(p["choice"] == p["label"] for p in per_row)
    n = len(per_row)
    return {"correct": int(correct), "n": n, "acc": correct / n,
            "mean_answer_confidence": sum(p["answer_confidence"] for p in per_row) / n, "per_row": per_row}


def evaluate_model(source: Any, rows: list[dict], question: dict, *, device: str, max_len: int,
                   head_max_len: int, batch_size: int) -> dict[str, Any]:
    """Load one agent, answer the rows, free it."""
    from .hub import load_agent

    agent = load_agent(source, device=device)
    if device == "cuda" and agent.device.type != "cuda":
        raise RuntimeError(f"agent landed on {agent.device.type}, not cuda (OOM at load?)")
    qid, guard = next(iter(question)), FallbackGuard()
    t0 = time.perf_counter()
    with guard:
        out = agent.predict_batch([r["state"] for r in rows], question, batch_size=batch_size, max_len=max_len,
                                  head_max_len=head_max_len)
    res = {**summarise(rows, [o["answers"][qid] for o in out]), "device": agent.device.type,
           "cpu_fallback": guard.cpu_fallback, "seconds": round(time.perf_counter() - t0, 2)}
    del agent
    free_memory()
    return res


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    cfg = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but no CUDA device (use --device cpu for local runs)")
    sources = {m: resolve_source(cfg, m, args.init) for m in args.models}  # validate before any load
    rows = select_rows(iter_jsonl(args.rows), args.n)
    question = shared_question(rows)
    results = {}
    for m in args.models:
        mc = cfg["model"][m]
        results[m] = evaluate_model(sources[m], rows, question, device=args.device, max_len=int(mc["max_len"]),
                                    head_max_len=int(mc["head_max_len"]), batch_size=args.batch_size)
        r = results[m]
        print(f"zeroshot {m}: {r['correct']}/{r['n']} correct, mean answer_confidence "
              f"{r['mean_answer_confidence']:.3f} ({r['device']}{', CPU fallback' if r['cpu_fallback'] else ''}, "
              f"{r['seconds']:.1f}s)", flush=True)
    return results


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.zeroshot", description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True, choices=("laya", "laya_ml"))
    p.add_argument("--init", default="hub", help="'hub' (pinned revision) or an absolute checkpoint dir")
    p.add_argument("--rows", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        write_json(args.out, run(args))
        print(f"zeroshot: wrote {args.out}")
        return 0

    return run_cli("zeroshot", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
