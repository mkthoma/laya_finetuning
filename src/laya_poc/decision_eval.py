"""The §5.12 decision rule on real inputs (spec P5 §3): phase5_metrics JSON + bench_cpu JSON (optional) + config.

This is the only place that knows the two JSON layouts; decision.py judges plain numbers. Metrics schema
"phase5_metrics/1" (see phase5_metrics' docstring): groups keyed by id ("E3 laya_ml c10"), each with a role
(candidate / report_also / baseline / other), its criterion-1 ood_pools, matched_small_encoder, per-split post/pre
stats ({mean, min, max, range, n, values}), per-pool gap, ood_average per pool set, traps; bootstrap comparisons
(candidate, baseline, pool incl. "ood_average"). Seed means are the stats' means.

Bench JSON (bench_cpu, schema_version 2): rows of {model, backend, threads, p95_ms, batch_rps, ...} under "results"
(or a "rows" list; the Phase 2 single-model form puts model/backend at the top level); error / skipped rows and
non-fp32 rows (amp on, or a dtype other than float32) are ignored. The machine is "machine" (bench_machine
.machine_info) or the top-level cpu_count/platform fields; "hardware" is its label. ONNX acceptance: "onnx"
{model: {accepted, acceptance {passed}}} (else "onnx_check" / "onnx_acceptance"); a model whose check FAILED has its
ONNX rows left out of C5.

Choices (design §5.12 read literally, the rest noted): C1's B4 is the language-matched small encoder (config
phase5.small_encoder_match, §5.13 item 4); the strict variant (best B4 of either language) is reported as a
sensitivity check and never enters the verdict. C4's "best baseline" is the best trap accuracy over every §5.13
baseline of the candidate's scheme with a number: B1-B5 (role baseline) and head-only Laya. The seed-range check
covers the headline macro-F1s the rule judges: test_id, each included OOD pool and their OOD average.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import decision as D
from .matrix_results import num

SCHEMA = "phase5_decision/1"
FP32 = ("float32", "fp32", "torch.float32")


# ---------------------------------------------------------------- metrics accessors

def dig(obj: Any, *keys: str) -> Any:
    for k in keys:
        obj = obj.get(k) if isinstance(obj, Mapping) else None
    return obj


def mean(s: Any) -> float | None:
    return num(s.get("mean")) if isinstance(s, Mapping) else None


def srange(s: Any) -> float | None:
    """A stat's seed range (max - min)."""
    if not isinstance(s, Mapping):
        return None
    r = num(s.get("range"))
    lo, hi = num(s.get("min")), num(s.get("max"))
    return r if r is not None else (None if lo is None or hi is None else hi - lo)


def f1(g: Mapping | None, split: str) -> float | None:
    return mean(dig(g, "splits", split, "post", "macro_f1"))


def pool_key(pools: Sequence[str]) -> str:
    return "+".join(pools)


def ood_avg(g: Mapping | None, pools: Sequence[str], key: str = "macro_f1") -> float | None:
    """Seed-mean OOD average over `pools` (the per-run average's mean; the mean of the pool means as fallback)."""
    direct = mean(dig(g, "ood_average", pool_key(pools), key))
    if direct is not None or g is None:
        return direct
    split_key = ("post", "macro_f1") if key == "macro_f1" else ("post", "ece") if key == "ece_post" else None
    vals = [mean(dig(g, "splits", p, *split_key)) for p in pools] if split_key else [None]
    return None if not vals or any(v is None for v in vals) else sum(vals) / len(vals)


def gap(g: Mapping, pool: str) -> float | None:
    direct = mean(dig(g, "splits", pool, "gap"))
    a, b = f1(g, "test_id"), f1(g, pool)
    return direct if direct is not None else (None if a is None or b is None else a - b)


def trap_acc(g: Mapping | None) -> float | None:
    return mean(dig(g, "traps", "acc")) if dig(g, "traps", "status") == "ok" else None


def groups_where(metrics: Mapping, **match: Any) -> list[dict]:
    return [g for g in (metrics.get("groups") or {}).values() if all(g.get(k) == v for k, v in match.items())]


def bootstrap(metrics: Mapping, cand: str, base: str | None, pool: str) -> dict | None:
    """The paired bootstrap comparison (candidate - baseline) for one pool or "ood_average"."""
    for c in dig(metrics, "bootstrap", "comparisons") or []:
        if (c.get("candidate"), c.get("baseline"), c.get("pool")) == (cand, base, pool):
            return {k: c.get(k) for k in ("diff", "mean", "ci_low", "ci_high", "p_gt_0", "p_ge_lead", "restricted")}
    return None


# ---------------------------------------------------------------- bench accessors

def _fp32(r: Mapping) -> bool:
    return r.get("amp") is not True and (r.get("dtype") is None or str(r.get("dtype")) in FP32)


def bench_rows(bench: Mapping | None) -> list[dict] | None:
    """Every usable measurement row, or None when there is no bench JSON."""
    if not isinstance(bench, Mapping):
        return None
    rows = bench.get("rows") if isinstance(bench.get("rows"), list) else bench.get("results") or []
    out = []
    for r in rows:
        if not isinstance(r, Mapping) or r.get("error") or r.get("skipped") or not _fp32(r):
            continue
        p95, rps, threads = num(r.get("p95_ms")), num(r.get("batch_rps")), r.get("threads")
        if p95 is not None and rps is not None and isinstance(threads, int):
            out.append({**r, "model": r.get("model", bench.get("model")), "p95_ms": p95, "batch_rps": rps,
                        "backend": r.get("backend", bench.get("backend", "torch")), "threads": threads})
    return out


def onnx_acceptance(bench: Mapping | None, model: str) -> bool | None:
    """A model's ONNX acceptance result (§7.11): bench_cpu v2 `onnx {model: {accepted, acceptance {passed}}}`, or
    `onnx_check` / `onnx_acceptance` as {model: {passed}} or [{model, passed}]; None when not recorded."""
    rec = dig(bench, "onnx", model)
    for value in (dig(rec, "accepted"), dig(rec, "acceptance", "passed")):
        if isinstance(value, bool):
            return value
    checks = (dig(bench, "onnx_check") or dig(bench, "onnx_acceptance")) if bench else None
    if isinstance(checks, list):
        checks = {c.get("model"): c for c in checks if isinstance(c, Mapping)}
    passed = dig(checks, model, "passed")
    return passed if isinstance(passed, bool) else None


def machine(bench: Mapping | None) -> dict | None:
    if not isinstance(bench, Mapping):
        return None
    m = bench.get("machine") if isinstance(bench.get("machine"), Mapping) else bench
    keys = ("cpu_model", "physical_cores", "cpu_count", "ram_gb", "platform")
    info = {k: m.get(k) for k in keys}
    label = bench.get("hardware") or m.get("hardware") or (
        f"{info['cpu_model'] or 'unknown CPU'} ({info['physical_cores'] or '?'}C/{info['cpu_count'] or '?'}T"
        + (f", {info['ram_gb']:g} GB RAM" if num(info["ram_gb"]) else "") + ")")
    return {**info, "label": label}


# ---------------------------------------------------------------- one candidate

def _baselines(metrics: Mapping, g: Mapping, cfg: Mapping) -> dict[str, dict | None]:
    scheme, pools = g.get("scheme"), g["ood_pools"]
    b3 = next(iter(groups_where(metrics, arm="B3", kind="tfidf_lr", scheme=scheme)), None)
    match_id = g.get("matched_small_encoder") or (
        f"B4 {dig(cfg, 'phase5', 'small_encoder_match', str(g.get('model')))} {scheme}")
    b4 = (metrics.get("groups") or {}).get(match_id)
    b4s = [x for x in groups_where(metrics, kind="small_encoder", scheme=scheme) if ood_avg(x, pools) is not None]
    best = max(b4s, key=lambda x: ood_avg(x, pools), default=None)
    return {"B3": b3, "B4": b4, "B4_best": best}


def _c1(metrics: Mapping, g: Mapping, bases: Mapping, th: Mapping, b4_key: str) -> dict:
    pools, b3, b4 = g["ood_pools"], bases["B3"], bases[b4_key]
    c = D.criterion_1(ood_avg(g, pools), ood_avg(b3, pools), ood_avg(b4, pools), th,
                      b4_label=b4["id"] if b4 else "B4 (none)")
    name = ("OOD lead over B3 and the language-matched B4" if b4_key == "B4"
            else "OOD lead over B3 and the best B4 of either language")
    cis = {"B3": bootstrap(metrics, g["id"], b3 and b3["id"], "ood_average"),
           "B4": bootstrap(metrics, g["id"], b4 and b4["id"], "ood_average")}
    return {**c, "name": name, "b3": b3["id"] if b3 else None, "candidate_ood_avg": ood_avg(g, pools),
            "baseline_ood_avg": {"B3": ood_avg(b3, pools), "B4": ood_avg(b4, pools)}, "ci": cis}


def _c4(metrics: Mapping, g: Mapping, th: Mapping) -> dict:
    annotated = dig(metrics, "traps", "status") == "ok"
    bases = {x["id"]: trap_acc(x) for x in (metrics.get("groups") or {}).values()
             if x.get("scheme") == g.get("scheme") and (x.get("role") == "baseline" or x.get("head_only"))}
    c = D.criterion_4(trap_acc(g), bases, annotated, th)
    return {**c, "multi": mean(dig(g, "traps", "acc_multi")), "n": dig(g, "traps", "n")}


def seed_ranges(g: Mapping) -> dict[str, float | None]:
    pools = g["ood_pools"]
    out = {s: srange(dig(g, "splits", s, "post", "macro_f1")) for s in ("test_id", *pools)}
    return {**out, "ood_average": srange(dig(g, "ood_average", pool_key(pools), "macro_f1"))}


def candidate(metrics: Mapping, g: Mapping, rows: list | None, bench: Mapping | None, cfg: Mapping,
              th: Mapping) -> dict[str, Any]:
    """Criteria 1-5 (+ the strict C1 variant), stop and investigate checks of one candidate group."""
    pools, model = g["ood_pools"], str(g.get("model"))
    bases = _baselines(metrics, g, cfg)
    ece_id, ece_ood = mean(dig(g, "splits", "test_id", "post", "ece")), ood_avg(g, pools, "ece_post")
    crits = [_c1(metrics, g, bases, th, "B4"), D.criterion_2({p: gap(g, p) for p in pools}, th),
             D.criterion_3(ece_id, ece_ood, th), _c4(metrics, g, th),
             D.criterion_5(rows, model, th, onnx_ok=onnx_acceptance(bench, model))]
    b3, b4 = bases["B3"], bases["B4"]
    stops = [D.stop_below_baselines((f1(g, "test_id"), ood_avg(g, pools)), (f1(b3, "test_id"), ood_avg(b3, pools)),
                                    (f1(b4, "test_id"), ood_avg(b4, pools))),
             D.stop_ece(ece_id, ece_ood, th), D.stop_latency(rows, model, th)]
    return {"id": g["id"], "label": g.get("label") or g["id"], "arm": g.get("arm"), "model": model,
            "scheme": g.get("scheme"), "seeds": g.get("seeds"), "pools": list(pools),
            "excluded_pools": [p for p in ("ood_country", "ood_script", "ood_brand") if p not in pools],
            "criteria": crits, "c1_strict": _c1(metrics, g, bases, th, "B4_best"), "stop": stops,
            "investigate": [D.near_miss_check(crits), D.seed_range_check(seed_ranges(g), th), D.gate_check()]}


# ---------------------------------------------------------------- the whole rule

def candidate_groups(metrics: Mapping, cfg: Mapping) -> list[dict]:
    """The judged candidates (role "candidate"), in config phase5.candidates order."""
    order = list(dig(cfg, "phase5", "candidates") or [])
    found = groups_where(metrics, role="candidate")
    return sorted(found, key=lambda g: (order.index(g["arm"]) if g.get("arm") in order else len(order), g["id"]))


def evaluate(metrics: Mapping, bench: Mapping | None, cfg: Mapping) -> dict[str, Any]:
    """The decision rule per candidate and overall, with the outlook over pending inputs."""
    if not isinstance(metrics.get("groups"), Mapping):
        raise ValueError("metrics JSON has no 'groups' object: pass the phase5_metrics output (--metrics)")
    th, rows = D.thresholds(cfg), bench_rows(bench)
    cands = [candidate(metrics, g, rows, bench, cfg, th) for g in candidate_groups(metrics, cfg)]
    if not cands:
        raise ValueError(f"no candidate group (config phase5.candidates {dig(cfg, 'phase5', 'candidates')}) in "
                         "the metrics JSON")
    out = D.outlook(cands)
    judged = [{**c, "verdict": p["verdict"], "possible": p["possible"]} for c, p in zip(cands, out["per_candidate"])]
    return {"schema": SCHEMA, "metrics_schema": metrics.get("schema"), "thresholds": th,
            "bench_present": rows is not None, "machine": machine(bench),
            "traps_status": dig(metrics, "traps", "status") or "pending", "candidates": judged,
            **{k: out[k] for k in ("verdict", "final", "possible", "pending_inputs", "statement")}}
