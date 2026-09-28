"""Phase 5 report (design §7.13 templates, §5.12 decision; spec P5 §3): <out>.md and <out>.json.

    python -m laya_poc.phase5_report --metrics <phase5_metrics json> [--bench <bench json> [--bench <json> ...]]
                                     --out <md> [--config yaml]

Tables: main results per pool (test_id, ood_country, ood_script, ood_brand) with every arm in the template's order
(E6 train subsets are left to the Phase 3/4 report), traps / abstention / stability, CPU (every --bench file; the
FIRST is the judged machine for C5 and stop (c), later ones, e.g. the laptop, are reported), the decision
checklist per candidate with the verdict and the pending-input outlook (decision_eval), the bootstrap CIs, and
per-class / look-alike confusion tables for the candidates and the best baseline. Values here, strings in
phase5_report_md. Metrics only: no FSQ row is ever read or written (the inputs are aggregate JSONs).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_config
from .decision_eval import bench_rows, dig, evaluate, f1, machine, mean, onnx_acceptance, srange
from .env_check import run_cli
from .matrix_exec import write_json
from .phase5_report_md import render_markdown

SCHEMA = "phase5_report/1"
POOLS = ("test_id", "ood_country", "ood_script", "ood_brand")
# design §7.13 main-results order: (arm, model) -> row name (model None: any)
TEMPLATE = ((("B1", "majority"), "Majority (B1)"), (("B1", "prior"), "Prior (B1)"),
            (("B2", "laya"), "Zero-shot `laya` (B2)"), (("B2", "laya_ml"), "Zero-shot `laya-ml` (B2)"),
            (("B3", "tfidf_lr"), "TF-IDF+LR (B3)"), (("B4", "modernbert_base"), "ModernBERT-base (B4)"),
            (("B4", "mmbert_small"), "mmBERT-small (B4)"), (("E4", "laya"), "Head-only `laya` (E4)"),
            (("E2", "laya"), "FT `laya` (E2)"), (("E3", "laya_ml"), "FT `laya-multilingual` (E3)"),
            (("B5", None), "Reference LLM {model} (B5, eval subsets)"),
            (("E5", "laya"), "FT `laya` (E5, reported only)"),
            (("E5", "laya_ml"), "FT `laya-multilingual` (E5, reported only)"))


# ---------------------------------------------------------------- groups and names

def _template_index(g: Mapping) -> int:
    for i, ((arm, model), _) in enumerate(TEMPLATE):
        if g.get("arm") == arm and model in (None, g.get("model")):
            return i
    return len(TEMPLATE)


def arm_name(g: Mapping, headline: str = "c10") -> str:
    """The template's row name (other arms: the group label), plus the seed count of a multi-seed group."""
    i = _template_index(g)
    if i == len(TEMPLATE):
        return str(g.get("label") or g["id"])
    seeds = f", {g['n_runs']} seeds" if (g.get("n_runs") or 1) > 1 else ""
    scheme = "" if g.get("scheme") == headline else f" [{g.get('scheme')}]"
    return TEMPLATE[i][1].format(model=g.get("model")) + seeds + scheme


def report_groups(metrics: Mapping, headline: str = "c10") -> list[dict]:
    """Every group but the train subsets: headline scheme first, template order, then the rest by id."""
    gs = [g for g in (metrics.get("groups") or {}).values() if not g.get("subset")]
    return sorted(gs, key=lambda g: (g.get("scheme") != headline, _template_index(g), str(g["id"])))


def best_baseline(metrics: Mapping, headline: str = "c10") -> str | None:
    """The baseline with the highest test_id macro-F1 (headline scheme; eval-subset baselines excluded)."""
    bases = [g for g in (metrics.get("groups") or {}).values() if g.get("role") == "baseline"
             and g.get("scheme") == headline and not g.get("eval_subset") and f1(g, "test_id") is not None]
    return max(bases, key=lambda g: f1(g, "test_id"))["id"] if bases else None


# ---------------------------------------------------------------- table data (values, not strings)

def main_row(g: Mapping, split: str, headline: str) -> dict[str, Any]:
    post, pre = dig(g, "splits", split, "post") or {}, dig(g, "splits", split, "pre") or {}
    return {"group": g["id"], "name": arm_name(g, headline), "macro_f1": post.get("macro_f1"),
            "acc": mean(post.get("acc")), "ece_pre": mean(pre.get("ece")), "ece_post": mean(post.get("ece")),
            "brier": mean(post.get("brier")), "nll": mean(post.get("nll")), "acc80": mean(post.get("acc@80")),
            "acc90": mean(post.get("acc@90")), "gap": mean(dig(g, "splits", split, "gap")),
            "excluded": split.startswith("ood") and split not in (g.get("ood_pools") or POOLS),
            "eval_subset": bool(g.get("eval_subset"))}


def robustness_row(g: Mapping, headline: str) -> dict[str, Any]:
    traps = g.get("traps") or {}
    return {"group": g["id"], "name": arm_name(g, headline), "trap_status": traps.get("status"),
            "trap_basis": traps.get("basis"), "trap_acc": mean(traps.get("acc")),
            "trap_multi": mean(traps.get("acc_multi")),
            "abstain": mean(dig(g, "stripped", "abstain_rate_at_tau")),
            "false_conf": mean(dig(g, "stripped", "false_confident_rate")),
            "flip_rate": dig(g, "splits", "test_id", "flip_rate"),
            "order_inv": mean(dig(g, "order_invariance", "mean_agreement")),
            "order_ok": dig(g, "order_invariance", "all_passed_99"),
            "range_f1": srange(dig(g, "splits", "test_id", "post", "macro_f1")),
            "range_ece": srange(dig(g, "splits", "test_id", "post", "ece")), "n_runs": g.get("n_runs")}


CPU_KEYS = ("model", "backend", "threads", "cold_s", "p50_ms", "p95_ms", "batch_rps", "peak_rss_gb")


def cpu_rows(benches: Sequence[Mapping]) -> list[dict[str, Any]]:
    out = []
    for i, b in enumerate(benches):
        label = (machine(b) or {}).get("label")
        out += [{**{k: r.get(k) for k in CPU_KEYS}, "hardware": label, "judged": i == 0}
                for r in bench_rows(b) or []]
    return out


def bench_notes(benches: Sequence[Mapping]) -> list[str]:
    """Thread settings the benchmark skipped (above the physical cores) and failed settings, per machine."""
    out = []
    for b in benches:
        label = (machine(b) or {}).get("label")
        out += [f"{label}: {s.get('threads')} threads skipped ({s.get('reason')})" for s in b.get("threads_skipped")
                or [] if isinstance(s, Mapping)]
        out += [f"{label}: {e.get('model')}/{e.get('backend')} {e.get('threads')} threads failed ({e.get('error')})"
                for e in b.get("errors") or [] if isinstance(e, Mapping)]
    return out


def relative_throughput(bench: Mapping | None, cfg: Mapping) -> list[dict[str, Any]]:
    """Each Laya model's batched rps (faster backend) over its matched small encoder's, at cpu_budget vcpus."""
    rows, threads = bench_rows(bench), int(cfg["cpu_budget"].get("vcpus", 4))
    out = []
    for model, enc in (dig(cfg, "phase5", "small_encoder_match") or {}).items():
        laya = [r["batch_rps"] for r in rows or [] if r["model"] == model and r["threads"] == threads]
        base = [r["batch_rps"] for r in rows or [] if r["model"] == enc and r["threads"] == threads]
        if laya and base:
            out.append({"model": model, "encoder": enc, "threads": threads, "laya_rps": max(laya),
                        "encoder_rps": max(base), "ratio": max(laya) / max(base)})
    return out


def bootstrap_rows(metrics: Mapping) -> list[dict[str, Any]]:
    comps = [c for c in dig(metrics, "bootstrap", "comparisons") or [] if c.get("judged", True)]
    order = {p: i for i, p in enumerate((*POOLS, "ood_average"))}
    return sorted(comps, key=lambda c: (str(c.get("candidate")), str(c.get("baseline")),
                                        order.get(c.get("pool"), 99)))


def confusion_tables(metrics: Mapping, gids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per pool: {labels, per_class {gid: {key: {p, r, f1, support}}}, lookalike {gid: [cells]}}."""
    groups = metrics.get("groups") or {}
    out = {}
    for pool in POOLS:
        rec = {gid: dig(groups.get(gid), "splits", pool, "recomputed") or {} for gid in gids}
        labels = next((r["labels"] for r in rec.values() if r.get("labels")), [])
        out[pool] = {"labels": labels, "per_class": {g: r.get("per_class") or {} for g, r in rec.items()},
                     "lookalike": {g: r.get("lookalike") or [] for g, r in rec.items()}}
    return out


# ---------------------------------------------------------------- the report

def build_report(metrics: Mapping, benches: Sequence[Mapping], cfg: Mapping, *,
                 inputs: Mapping | None = None) -> dict[str, Any]:
    headline = str(dig(cfg, "labels", "scheme") or "c10")
    judged = benches[0] if benches else None
    decision = evaluate(metrics, judged, cfg)
    groups = report_groups(metrics, headline)
    best = best_baseline(metrics, headline)
    focus = [c["id"] for c in decision["candidates"]] + ([best] if best else [])
    tables = {"main": {p: [main_row(g, p, headline) for g in groups] for p in POOLS},
              "robustness": [robustness_row(g, headline) for g in groups], "cpu": cpu_rows(benches),
              "relative_throughput": relative_throughput(judged, cfg), "bootstrap": bootstrap_rows(metrics),
              "bench_notes": bench_notes(benches),
              "onnx_acceptance": {c["model"]: onnx_acceptance(judged, c["model"])
                                  for c in decision["candidates"]} if judged else {},
              "best_baseline": best, "confusion": confusion_tables(metrics, focus)}
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "inputs": dict(inputs or {}), "metrics_generated_at": metrics.get("generated_at"),
            "recomputed_check": metrics.get("recomputed_check"), "traps": metrics.get("traps"),
            "headline_scheme": headline, "decision": decision, "tables": tables}


def write_report(result: Mapping, out_md: Path) -> tuple[Path, Path]:
    """<out>.md and <out>.json, each written atomically."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_md.with_name(out_md.name + ".tmp")
    tmp.write_text(render_markdown(result), encoding="utf-8")
    tmp.replace(out_md)
    return out_md, write_json(out_md.with_suffix(".json"), result)


def read_json_file(path: str | Path, what: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{what} JSON not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{what} JSON is not an object: {p}")
    return data


def say(line: str) -> None:
    """Print one line that any console encoding can show (the NO PASS verdict has an em dash)."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(line.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.phase5_report", description=__doc__.splitlines()[0])
    p.add_argument("--metrics", required=True, help="phase5_metrics output JSON")
    p.add_argument("--bench", action="append", default=[],
                   help="bench_cpu JSON; repeatable: the first is the judged machine (C5, stop c)")
    p.add_argument("--out", required=True, help="report markdown path (<out>.json is written next to it)")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        cfg = load_config(args.config)
        metrics = read_json_file(args.metrics, "metrics")
        benches = [read_json_file(b, "bench") for b in args.bench]
        inputs = {"metrics": str(args.metrics), "bench": [str(b) for b in args.bench], "config": args.config}
        result = build_report(metrics, benches, cfg, inputs=inputs)
        md, js = write_report(result, Path(args.out))
        d = result["decision"]
        state = "final" if d["final"] else "provisional; reachable: " + ", ".join(d["possible"])
        say(f"phase5_report: verdict {d['verdict']} ({state})")
        say(f"phase5_report: wrote {md} and {js}")
        return 0

    return run_cli("phase5_report", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
