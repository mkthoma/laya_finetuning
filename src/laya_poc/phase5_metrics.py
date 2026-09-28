"""Phase 5 full metric suite (design §5.11, §7.9; spec P5 §2) from the saved eval JSONs and per-row predictions.

    python -m laya_poc.phase5_metrics --runs-root A [--runs-root B ...] --data-dir data [--traps data/trap.jsonl]
        [--trap-candidates <jsonl>] [--data-eval <dir>] [--trap-annotations CSV [CSV]] --out <json> [--config yaml]

Recomputed from preds (per run, then per group): per-class P/R/F1, the confusion matrix (option-key order), the
look-alike cells, the metrics at the val tau, the seed flip rate, the paired bootstrap and trap accuracy on the
annotated keep set. Read from the eval JSONs (the same code computed them in-session) and aggregated: ECE, Brier,
NLL pre/post, acc@80/90, tau and abstention, the stripped-test no-evidence metrics, order invariance, OOD gaps.
Metrics only: no FSQ row is ever printed or written.

OUTPUT JSON (schema "phase5_metrics/1"; stable, read by decision.py and phase5_report.py). A `stat` is
{mean, min, max, range, n, values} over a group's runs (values in the group's run order, None for a gap) or null.
Probabilities/F1 are fractions (0.03 = 3 points).
{
  "schema": "phase5_metrics/1", "generated_at": iso8601, "seconds": float,
  "inputs": {"runs_roots": [str], "data_dir": str|null, "traps": str|null, "config": str|null},
  "settings": {"candidates": [arm], "report_also": [arm], "small_encoder_match": {laya_model: b4_model},
               "lookalike_pairs": [[a, b]], "pools": [test_id, ood_country, ood_script, ood_brand],
               "ood_pools": [...], "no_thai_models": ["laya"],
               "bootstrap": {"resamples", "ci", "seed"}, "lead_points": float},
  "groups": {"<group id>": {             # id: "<arm> <model> <scheme>[ n<subset>][ head]", e.g. "E3 laya_ml c10"
      "id", "label" (matrix-report label), "arm", "model", "scheme", "subset", "head_only", "kind", "zero_shot",
      "eval_subset" (B5: subsets), "role": "candidate"|"report_also"|"baseline"|"other",
      "runs": [run_name], "seeds": [int|null], "n_runs": int,
      "ood_pools": [pool]                # this model's criterion-1 pools (laya: no ood_script)
      "matched_small_encoder": "<B4 group id>"|null,   # candidate/report_also: config small_encoder_match, same scheme
      "T": stat,
      "splits": {"<split>": {            # every split with an eval JSON (val, test_id, ood_*, stripped_test, ...)
          "n": stat (rows), "n_scored": stat,
          "post": {"macro_f1", "macro_f1_9", "acc", "ece", "brier", "nll", "acc@80", "acc@90": stat},
          "pre":  {same keys: stat},     # before temperature
          "selective": {"tau", "coverage", "abstain_rate", "answered_acc": stat},   # eval JSON abstention block
          "gap": stat,                   # OOD pools only: macro-F1(test_id) - macro-F1(pool), per run, post-T
          "recomputed": null | {         # from preds (null when no row is scored, e.g. stripped_test)
              "labels": [key], "n_rows", "n_scored", "n_abstained" (gate), "n_unlabelled": int (first run),
              "all": {"macro_f1", "macro_f1_9", "acc": stat},            # every scored row, argmax (= evaluate)
              "eval_match": {"passed", "max_abs_diff", "compared"},  # all.macro_f1 vs eval post.macro_f1
              "answered": {"tau", "coverage", "n_answered", "macro_f1", "acc": stat},  # conf >= val tau;
                                         # coverage over labelled rows (gate abstentions count as abstentions)
              "per_class": {"<key>": {"p", "r", "f1": stat, "support": int}},
              "cm": [[int]],             # rows true, columns predicted, summed over the group's runs
              "cm_runs": int,
              "lookalike": [{"true": a, "pred": b, "count": int, "support": int, "rate": float|null,
                             "rate_runs": stat}]},   # both directions of each config pair; rate = count / support(true)
          "flip_rate": float|null, "flip_n": int|null}},   # rows whose argmax differs across seeds (n_runs > 1)
      "ood_average": {"<pool>+<pool>[+<pool>]": {"pools": [...], "macro_f1", "ece_post", "ece_pre", "gap": stat}},
                                         # per run mean over the pools, then stat; keys for both pool sets
      "stripped": {"abstain_rate_at_tau", "false_confident_rate", "gate_abstain_rate": stat},
      "order_invariance": {"mean_agreement": stat, "all_passed_99": bool|null},
      "traps": {"status": "ok"|"pending", "basis": "annotated keep set"|"unannotated candidates",
                "acc": stat, "acc_multi": stat|null, "acc_single": stat|null,
                "n": int|null, "n_multi": int|null, "n_answered": stat|null}}},
  "recomputed_check": {"passed": bool, "compared": int, "max_abs_diff": float, "mismatches": [str]},
  "bootstrap": {"resamples", "ci", "seed", "lead_points", "comparisons": [{
      "candidate": gid, "baseline": gid, "baseline_kind": str, "judged": bool (role candidate),
      "pool": "test_id"|"ood_country"|"ood_script"|"ood_brand"|"ood_average", "pools": [pool],
      "n_rows": {pool: int}, "restricted": bool (rows limited to the baseline's evaluated ids, B5),
      "candidate_f1", "baseline_f1", "diff": float (seed means on the same rows; diff = candidate - baseline),
      "mean", "ci_low", "ci_high": float (bootstrap), "p_gt_0", "p_ge_lead": float (share of resamples)}]},
  "traps": {"status": "ok"|"pending", "basis": str, "source": str|null, "n_kept": int|null, "n_multi": int|null,
            "join": str|null, "note": str}
}
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_config
from .io_utils import iter_jsonl
from .matrix_exec import write_json
from .phase5_bootstrap import Bootstrap, compare
from .phase5_classes import group_flip, group_split, run_split
from .phase5_load import (BOOTSTRAP_KINDS, NO_THAI_MODELS, OOD_POOLS, POOLS, Group, RunData, load_runs, make_groups,
                          ood_pools, stat5)
from .phase5_traps import (PENDING_NOTE, TRAP_SPLIT, group_traps, kept_places, pending_traps, resolve_id_map,
                           run_trap_acc)

SCHEMA = "phase5_metrics/1"
METRIC_KEYS = ("macro_f1", "macro_f1_9", "acc", "ece", "brier", "nll", "acc@80", "acc@90")
SELECTIVE_KEYS = ("tau", "coverage", "abstain_rate", "answered_acc")
STRIPPED_KEYS = ("abstain_rate_at_tau", "false_confident_rate", "gate_abstain_rate")
POOL_SETS = (tuple(OOD_POOLS), tuple(p for p in OOD_POOLS if p != "ood_script"))
DEFAULTS: dict[str, Any] = {"candidates": ["E2", "E3"], "report_also": [], "small_encoder_match": {},
                            "lookalike_pairs": [], "bootstrap": {"resamples": 1000, "ci": 0.95, "seed": 20260925},
                            "decision": {"lead_points": 3.0}}


def settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    p5 = {**DEFAULTS, **(cfg.get("phase5") or {})}
    boot = {**DEFAULTS["bootstrap"], **(p5.get("bootstrap") or {})}
    return {"candidates": list(p5["candidates"]), "report_also": list(p5.get("report_also") or []),
            "small_encoder_match": dict(p5.get("small_encoder_match") or {}),
            "lookalike_pairs": [list(x) for x in p5.get("lookalike_pairs") or []], "pools": list(POOLS),
            "ood_pools": list(OOD_POOLS), "no_thai_models": list(NO_THAI_MODELS),
            "bootstrap": {"resamples": int(boot["resamples"]), "ci": float(boot["ci"]), "seed": int(boot["seed"])},
            "lead_points": float((p5.get("decision") or {}).get("lead_points", 3.0))}


# ---------------------------------------------------------------- from the eval JSONs

def _ev(run: RunData, split: str) -> dict:
    return run.evals.get(split) or {}


def _post(run: RunData, split: str, key: str = "macro_f1") -> Any:
    return (_ev(run, split).get("post") or {}).get(key)


def _gap(run: RunData, pool: str) -> float | None:
    a, b = _post(run, "test_id"), _post(run, pool)
    return None if a is None or b is None else a - b


def eval_block(runs: Sequence[RunData], split: str) -> dict[str, Any]:
    """Headline numbers of one split read from the eval JSONs, as stats over the runs."""
    evs = [_ev(r, split) for r in runs]
    out = {"n": stat5([e.get("n") for e in evs]), "n_scored": stat5([e.get("n_scored") for e in evs]),
           **{side: {k: stat5([(e.get(side) or {}).get(k) for e in evs]) for k in METRIC_KEYS}
              for side in ("post", "pre")},
           "selective": {k: stat5([(e.get("abstention") or {}).get(k) for e in evs]) for k in SELECTIVE_KEYS}}
    if split in OOD_POOLS:
        out["gap"] = stat5([_gap(r, split) for r in runs])
    return out


def _mean(xs: Sequence[Any]) -> float | None:
    return None if not xs or any(x is None for x in xs) else sum(xs) / len(xs)


def ood_average(runs: Sequence[RunData]) -> dict[str, Any]:
    """Per run: mean over the pools (every pool needed), then stats; for both criterion-1 pool sets."""
    per = {"macro_f1": lambda r, p: _post(r, p), "ece_post": lambda r, p: _post(r, p, "ece"),
           "ece_pre": lambda r, p: (_ev(r, p).get("pre") or {}).get("ece"), "gap": _gap}

    def one(pools: Sequence[str]) -> dict[str, Any]:
        return {"pools": list(pools), **{k: stat5([_mean([f(r, p) for p in pools]) for r in runs])
                                         for k, f in per.items()}}

    return {"+".join(ps): one(ps) for ps in POOL_SETS}


def run_level(runs: Sequence[RunData]) -> dict[str, Any]:
    stripped = [_ev(r, "stripped_test") for r in runs]
    oi = [r.record.get("order_invariance_passed") for r in runs]
    return {"T": stat5([r.record.get("T") for r in runs]),
            "stripped": {k: stat5([e.get(k) for e in stripped]) for k in STRIPPED_KEYS},
            "order_invariance": {"mean_agreement": stat5([r.record.get("order_invariance") for r in runs]),
                                 "all_passed_99": None if any(v is None for v in oi) else all(oi)}}


# ---------------------------------------------------------------- from the predictions

def recomputed(runs: Sequence[RunData], split: str, pairs: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Per-class/confusion/look-alike/tau metrics and the flip rate of one group on one split."""
    keys = next((r.labels(split) for r in runs if r.labels(split)), None)
    have = [r for r in runs if split in r.preds]
    if keys is None or not have:
        return {"recomputed": None, "flip_rate": None, "flip_n": None}
    per_run = [run_split(r.preds[split], keys, (_ev(r, split).get("abstention") or {}).get("tau"))
               if split in r.preds else None for r in runs]
    flip, n = group_flip([r.preds[split] for r in have]) if len(have) == len(runs) else (None, None)
    return {"recomputed": group_split(per_run, keys, pairs, [_post(r, split) for r in runs]),
            "flip_rate": flip, "flip_n": n}


def matched_encoder(g: Group, groups: Mapping[str, Group], match: Mapping[str, str]) -> str | None:
    """The language-matched B4 group of a candidate (config phase5.small_encoder_match), same scheme."""
    if g.role not in ("candidate", "report_also"):
        return None
    want = match.get(str(g.key[1]))
    return next((o.id for o in groups.values() if o.key[0] == "B4" and o.key[1] == want and o.key[2] == g.key[2]),
                None)


def group_trap(g: Group, traps: Mapping[str, Any]) -> dict[str, Any]:
    if traps.get("status") != "ok":
        return pending_traps([r.evals.get(TRAP_SPLIT) for r in g.runs])
    per = [run_trap_acc(r.preds[TRAP_SPLIT], traps["id_map"], traps["kept"]) if TRAP_SPLIT in r.preds else None
           for r in g.runs]
    return group_traps(per)


def group_block(g: Group, groups: Mapping[str, Group], st: Mapping[str, Any], traps: Mapping[str, Any]) -> dict:
    splits = sorted({s for r in g.runs for s in r.evals}, key=lambda s: (s not in ("val", *POOLS), s))
    blocks = {s: {**eval_block(g.runs, s), **recomputed(g.runs, s, st["lookalike_pairs"])} for s in splits}
    return {**g.identity(), "matched_small_encoder": matched_encoder(g, groups, st["small_encoder_match"]),
            **run_level(g.runs), "splits": blocks, "ood_average": ood_average(g.runs), "traps": group_trap(g, traps)}


def recomputed_check(groups: Mapping[str, dict]) -> dict[str, Any]:
    """Every recomputed all-rows macro-F1 equals its eval JSON's post macro-F1 (same rows, same function)."""
    compared, worst, bad = 0, 0.0, []
    for gid, g in groups.items():
        for split, b in g["splits"].items():
            m = (b.get("recomputed") or {}).get("eval_match")
            if m is None:
                continue
            compared += m["compared"]
            worst = max(worst, m["max_abs_diff"])
            if not m["passed"]:
                bad.append(f"{gid} {split}")
    return {"passed": not bad and compared > 0, "compared": compared, "max_abs_diff": worst, "mismatches": bad}


# ---------------------------------------------------------------- bootstrap

def bootstrap(groups: Mapping[str, Group], st: Mapping[str, Any]) -> dict[str, Any]:
    """Every candidate / report_also group vs every same-scheme B1 majority, B3, B4 and B5 group, per pool and on
    the candidate's OOD average."""
    b = st["bootstrap"]
    bs = Bootstrap(resamples=b["resamples"], seed=b["seed"], ci=b["ci"])
    lead, rows = st["lead_points"] / 100.0, []
    for c in (g for g in groups.values() if g.role in ("candidate", "report_also")):
        keys = next((r.labels(p) for r in c.runs for p in POOLS if r.labels(p)), None)
        if keys is None:
            continue
        for base in (x for x in groups.values() if x.rec.get("kind") in BOOTSTRAP_KINDS and x.key[2] == c.key[2]):
            res = compare(bs, [(r.name, r.preds) for r in c.runs], [(r.name, r.preds) for r in base.runs],
                          POOLS, ood_pools(c.key[1]), len(keys), lead)
            rows += [{"candidate": c.id, "baseline": base.id, "baseline_kind": base.rec.get("kind"),
                      "judged": c.role == "candidate", **x} for x in res]
    return {**b, "lead_points": st["lead_points"], "comparisons": rows}


# ---------------------------------------------------------------- traps

def load_traps(args: argparse.Namespace, cfg: Mapping[str, Any], roots: Sequence[Path]) -> dict[str, Any]:
    """{status, basis, source, n_kept, n_multi, join, note} (+ id_map / kept for the groups; never written)."""
    if not args.traps:
        return {"status": "pending", "basis": "unannotated candidates", "source": None, "n_kept": None,
                "n_multi": None, "join": None, "note": PENDING_NOTE}
    data_dir = Path(args.data_dir) if args.data_dir else None
    rows = list(iter_jsonl(args.traps))
    csvs = [Path(p) for p in (args.trap_annotations or ([data_dir / "trap_candidates.csv"] if data_dir else []))]
    kept, join = kept_places(rows, csvs, data_dir, cfg)
    id_map, how = resolve_id_map(data_dir, roots, Path(args.trap_candidates) if args.trap_candidates else None,
                                 Path(args.data_eval) if args.data_eval else None)
    missing = set(kept) - {v["fsq_place_id"] for v in id_map.values()}
    if missing:
        raise ValueError(f"{len(missing)} kept trap places were not among the scored candidates (another build?)")
    return {"status": "ok", "basis": "annotated keep set", "source": str(args.traps), "n_kept": len(kept),
            "n_multi": sum(kept.values()), "join": f"{join}; {how}",
            "note": "accuracy on the annotated keep set, multi-category items split out", "id_map": id_map,
            "kept": kept}


# ---------------------------------------------------------------- build and CLI

def build(cfg: Mapping[str, Any], roots: Sequence[Path], traps: Mapping[str, Any],
          inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The whole output JSON (module docstring) from the runs roots, config and loaded traps."""
    t0, st = time.perf_counter(), settings(cfg)
    runs = load_runs(roots)
    if not runs:
        raise FileNotFoundError(f"no finished run (done.json) under {', '.join(map(str, roots))}")
    groups = make_groups(runs, st["candidates"], st["report_also"])
    print(f"phase5_metrics: {len(runs)} runs in {len(groups)} groups", flush=True)
    blocks = {gid: group_block(g, groups, st, traps) for gid, g in groups.items()}
    boot = bootstrap(groups, st)
    print(f"phase5_metrics: {len(boot['comparisons'])} bootstrap comparisons "
          f"({st['bootstrap']['resamples']} resamples)", flush=True)
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "inputs": dict(inputs), "settings": st, "groups": blocks, "recomputed_check": recomputed_check(blocks),
            "bootstrap": boot, "traps": {k: v for k, v in traps.items() if k not in ("id_map", "kept")},
            "seconds": round(time.perf_counter() - t0, 2)}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m laya_poc.phase5_metrics", description=__doc__.splitlines()[0])
    p.add_argument("--runs-root", action="append", required=True, help="runs root (repeat: Phase 3 + Phase 4)")
    p.add_argument("--data-dir", default=None, help="the frozen build (trap_candidates.csv, pool.parquet)")
    p.add_argument("--traps", default=None, help="annotated trap JSONL from `traps merge` (else: pending)")
    p.add_argument("--trap-candidates", default=None, help="the scored candidate rows JSONL (variants traps)")
    p.add_argument("--data-eval", default=None, help="variants traps out dir (default: <root>/../../data_eval)")
    p.add_argument("--trap-annotations", nargs="+", default=None,
                   help="annotated CSV(s) for trap rows without fsq_place_id (default: <data-dir>/trap_candidates.csv)")
    p.add_argument("--out", required=True, help="output JSON")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(Path(args.config) if args.config else None)
    roots = [Path(r) for r in args.runs_root]
    traps = load_traps(args, cfg, roots)
    inputs = {"runs_roots": [str(r) for r in roots], "data_dir": args.data_dir, "traps": args.traps,
              "config": args.config}
    return build(cfg, roots, traps, inputs)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        res = run(args)
        write_json(Path(args.out), res)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: one line, non-zero exit
        print(" ".join(f"phase5_metrics: error: {type(exc).__name__}: {exc}".split()), file=sys.stderr, flush=True)
        return 1
    chk = res["recomputed_check"]
    print(f"phase5_metrics: recomputed macro-F1 vs eval JSONs {'PASS' if chk['passed'] else 'FAIL'} "
          f"({chk['compared']} run-splits, max |diff| {chk['max_abs_diff']:.1e}); traps {res['traps']['status']}; "
          f"wrote {args.out} ({res['seconds']:.0f}s)", flush=True)
    if not chk["passed"]:
        print(f"phase5_metrics: error: recomputed macro-F1 differs from the eval JSON: {chk['mismatches'][:5]}",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
