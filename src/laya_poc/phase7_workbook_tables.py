"""Workbook sheets from the Phase 5 report JSON (phase5_report / decision) and the CPU benchmark JSON (bench_cpu).

Pure functions: report / bench dicts in, phase7_xlsx.Sheet out (values, not strings). The report's tables are already
aggregates; nothing here reads a data row. Torch-free (decision_eval pulls torch in through matrix_results, so `dig`
and the machine label are small local copies of its helpers).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .phase7_xlsx import F0, F1, F2, F3, F4, PCT, SCI, Sheet, Table, cols, flatten_records, kv_table, message_table

POOLS = ("test_id", "ood_country", "ood_script", "ood_brand")
CHECK_KEYS = frozenset({"id", "name", "status", "value", "threshold", "near_miss", "note", "source", "holds"})


def _num(x: Any) -> float | int | None:
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def dig(obj: Any, *keys: str) -> Any:
    """obj[k1][k2]... or None at the first missing level (decision_eval.dig)."""
    for k in keys:
        obj = obj.get(k) if isinstance(obj, Mapping) else None
    return obj


def machine_label(bench: Mapping) -> str | None:
    """The benchmark's one-line hardware label (decision_eval.machine's `label`)."""
    m = bench.get("machine") if isinstance(bench.get("machine"), Mapping) else bench
    if bench.get("hardware") or m.get("hardware"):
        return bench.get("hardware") or m.get("hardware")
    if not m.get("cpu_model"):
        return None
    ram = f", {m['ram_gb']:g} GB RAM" if _num(m.get("ram_gb")) else ""
    return f"{m['cpu_model']} ({m.get('physical_cores') or '?'}C/{m.get('cpu_count') or '?'}T{ram})"


def stat_parts(s: Any) -> tuple[Any, Any, Any, Any]:
    """(mean, min, max, n) of a phase5 stat dict, or (value, None, None, None) for a plain number."""
    if isinstance(s, Mapping):
        return s.get("mean"), s.get("min"), s.get("max"), s.get("n")
    return _num(s), None, None, None


def leaves(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Dotted-path leaves of nested dicts (lists of scalars stay one leaf)."""
    if not isinstance(obj, Mapping):
        return [(prefix, obj)]
    return [kv for k, v in obj.items() for kv in leaves(v, f"{prefix}.{k}" if prefix else str(k))]


def detail(v: Any) -> str | None:
    return "; ".join(f"{k} {x}" for k, x in leaves(v)) if isinstance(v, Mapping) else None


# ---------------------------------------------------------------- Decision

def _criterion(cand: str, c: Mapping, check: str, cid: str) -> dict[str, Any]:
    return {"cand": cand, "check": check, "id": cid, "name": c.get("name"), "result": c.get("status"),
            "value": _num(c.get("value")), "threshold": _num(c.get("threshold")), "near": c.get("near_miss"),
            "value_detail": detail(c.get("value")), "threshold_detail": detail(c.get("threshold")),
            "note": c.get("note") or None}


def _flag(cand: str, c: Mapping, check: str) -> dict[str, Any]:
    holds = c.get("holds")
    result = "pending" if holds is None else "holds" if holds else "does not hold"
    return {"cand": cand, "check": check, "id": str(c.get("id")), "name": c.get("name"), "result": result,
            "note": c.get("note") or None}


def candidate_rows(c: Mapping) -> list[dict[str, Any]]:
    cid = str(c.get("id"))
    rows = [_criterion(cid, k, "criterion", f"C{k.get('id')}") for k in c.get("criteria") or []]
    if isinstance(c.get("c1_strict"), Mapping):
        rows.append(_criterion(cid, c["c1_strict"], "criterion (sensitivity)", "C1 strict"))
    rows += [_flag(cid, s, "stop") for s in c.get("stop") or []]
    rows += [_flag(cid, s, "investigate") for s in c.get("investigate") or []]
    rows.append({"cand": cid, "check": "verdict", "id": None, "name": "candidate verdict", "result": c.get("verdict"),
                 "note": "reachable: " + ", ".join(c.get("possible") or [])})
    return rows


def detail_rows(c: Mapping) -> list[dict[str, Any]]:
    """Every extra leaf of each check (leads, CIs, gaps, backends, ranges, value / threshold dicts)."""
    checks = [(f"C{k.get('id')}", k) for k in c.get("criteria") or []]
    checks += [("C1 strict", c["c1_strict"])] if isinstance(c.get("c1_strict"), Mapping) else []
    checks += [(str(s.get("id")), s) for s in [*(c.get("stop") or []), *(c.get("investigate") or [])]]
    out = []
    for cid, k in checks:
        extra = {key: v for key, v in k.items() if key not in CHECK_KEYS or isinstance(v, Mapping)}
        extra = {key: v for key, v in extra.items() if key not in ("id", "name", "note", "status")}
        out += [{"cand": c.get("id"), "check": cid, "field": f, "value": v} for f, v in leaves(extra)]
    return out


def decision_sheet(report: Mapping) -> Sheet:
    d = report.get("decision") or {}
    cands = d.get("candidates") or []
    rows = [r for c in cands for r in candidate_rows(c)]
    state = "final" if d.get("final") else "provisional"
    rows.append({"cand": "overall", "check": "verdict", "name": "§5.12 decision", "result": d.get("verdict"),
                 "note": f"{state}; reachable: {', '.join(d.get('possible') or [])}. {d.get('statement') or ''}"})
    check = Table(None, cols(("cand", "Candidate"), ("check", "Check"), ("id", "ID"), ("name", "Name"),
                             ("result", "Result"), ("value", "Value", F4), ("threshold", "Threshold", F4),
                             ("near", "Near miss"), ("value_detail", "Value detail"),
                             ("threshold_detail", "Threshold detail"), ("note", "Note")), rows)
    rc = report.get("recomputed_check") or {}
    overall = kv_table("Overall", [
        ("Verdict", d.get("verdict")), ("Final", d.get("final")), ("Reachable", d.get("possible")),
        ("Statement", d.get("statement")), ("Headline scheme", report.get("headline_scheme")),
        ("Traps status", d.get("traps_status")), ("CPU bench present", d.get("bench_present")),
        ("Judged CPU machine", dig(d, "machine", "label")),
        ("Recomputed macro-F1 check", f"passed {rc.get('passed')} on {rc.get('compared')} run-splits, "
                                      f"max |diff| {rc.get('max_abs_diff')}" if rc else None)])
    pending = Table("Pending inputs", cols(("input", "Input"), ("items", "Items"), ("change", "Can change verdict")),
                    [{"input": p.get("input"), "items": p.get("items"), "change": p.get("can_change_verdict")}
                     for p in d.get("pending_inputs") or []])
    thresholds = kv_table("Thresholds (decision_eval)", sorted((d.get("thresholds") or {}).items()),
                          ("Threshold", "Value"))
    details = Table("Details", cols(("cand", "Candidate"), ("check", "Check ID"), ("field", "Field"),
                                    ("value", "Value")), [r for c in cands for r in detail_rows(c)])
    return Sheet("Decision", [check, overall, pending, thresholds, details])


# ---------------------------------------------------------------- results tables

def main_sheet(report: Mapping) -> Sheet:
    rows = []
    for pool in POOLS:
        for r in dig(report, "tables", "main", pool) or []:
            mean, lo, hi, n = stat_parts(r.get("macro_f1"))
            rows.append({**r, "pool": pool, "f1": mean, "f1_min": lo, "f1_max": hi, "seeds": n})
    return Sheet("Main results", [Table(None, cols(
        ("pool", "Pool"), ("group", "Group"), ("name", "Arm"), ("f1", "Macro-F1 mean", F4),
        ("f1_min", "Macro-F1 min", F4), ("f1_max", "Macro-F1 max", F4), ("seeds", "Seeds", F0), ("acc", "Acc", F4),
        ("ece_pre", "ECE pre", F4),
        ("ece_post", "ECE post", F4), ("brier", "Brier", F4), ("nll", "NLL", F3), ("acc80", "Acc@80", F4),
        ("acc90", "Acc@90", F4), ("gap", "Gap vs ID", F4), ("excluded", "Pool excluded (C1)"),
        ("eval_subset", "Eval subset")), rows)])


def robustness_sheet(report: Mapping) -> Sheet:
    return Sheet("Robustness", [Table(None, cols(
        ("group", "Group"), ("name", "Arm"), ("trap_status", "Trap status"), ("trap_basis", "Trap basis"),
        ("trap_acc", "Trap acc", F4), ("trap_multi", "Trap acc (multi)", F4),
        ("abstain", "Abstain at tau (stripped)", PCT), ("false_conf", "False-confident (stripped)", PCT),
        ("flip_rate", "Flip rate (test_id)", PCT), ("order_inv", "Order agreement", F4),
        ("order_ok", "Order >= 0.99 (all)"), ("range_f1", "Seed range macro-F1", F4),
        ("range_ece", "Seed range ECE", F4), ("n_runs", "Runs", F0)), dig(report, "tables", "robustness") or [])])


def bootstrap_sheet(report: Mapping) -> Sheet:
    rows = [{**c, "n": sum(v for v in (c.get("n_rows") or {}).values() if _num(v) is not None) or None}
            for c in dig(report, "tables", "bootstrap") or []]
    return Sheet("Bootstrap", [Table(None, cols(
        ("candidate", "Candidate"), ("baseline", "Baseline"), ("baseline_kind", "Baseline kind"),
        ("judged", "Judged"), ("pool", "Pool"), ("pools", "Pools"), ("n", "Rows", F0), ("restricted", "Restricted"),
        ("candidate_f1", "Candidate F1", F4), ("baseline_f1", "Baseline F1", F4), ("diff", "Diff", F4),
        ("mean", "Bootstrap mean", F4), ("ci_low", "CI low", F4), ("ci_high", "CI high", F4),
        ("p_gt_0", "P(diff > 0)", F3), ("p_ge_lead", "P(diff >= lead)", F3)), rows)])


def _per_class_rows(pool: str, gid: str, per_class: Mapping) -> list[dict[str, Any]]:
    out = []
    for key, m in per_class.items():
        row = {"pool": pool, "group": gid, "class": key, "support": m.get("support")}
        for part in ("p", "r", "f1"):
            row.update(zip((f"{part}", f"{part}_min", f"{part}_max"), stat_parts(m.get(part))[:3]))
        out.append(row)
    return out


def per_class_sheet(report: Mapping) -> Sheet:
    conf = dig(report, "tables", "confusion") or {}
    rows = [r for pool in POOLS for gid, pc in (dig(conf, pool, "per_class") or {}).items()
            for r in _per_class_rows(pool, gid, pc or {})]
    return Sheet("Per-class", [Table(None, cols(
        ("pool", "Pool"), ("group", "Group"), ("class", "Class"), ("support", "Support", F0),
        ("p", "P mean", F4), ("p_min", "P min", F4), ("p_max", "P max", F4), ("r", "R mean", F4),
        ("r_min", "R min", F4), ("r_max", "R max", F4), ("f1", "F1 mean", F4), ("f1_min", "F1 min", F4),
        ("f1_max", "F1 max", F4)), rows)])


def lookalike_sheet(report: Mapping) -> Sheet:
    conf = dig(report, "tables", "confusion") or {}
    rows = []
    for pool in POOLS:
        for gid, cells in (dig(conf, pool, "lookalike") or {}).items():
            for c in cells or []:
                _, lo, hi, n = stat_parts(c.get("rate_runs"))
                rows.append({**c, "pool": pool, "group": gid, "rate_min": lo, "rate_max": hi, "runs": n})
    return Sheet("Look-alikes", [Table(None, cols(
        ("pool", "Pool"), ("group", "Group"), ("true", "True class"), ("pred", "Predicted"), ("count", "Count", F0),
        ("support", "Support", F0), ("rate", "Rate", PCT), ("rate_min", "Rate min (runs)", PCT),
        ("rate_max", "Rate max (runs)", PCT), ("runs", "Runs", F0)), rows)])


# ---------------------------------------------------------------- CPU

CPU_COLS = cols(("model", "Model"), ("backend", "Backend"), ("threads", "Threads", F0), ("batch_size", "Batch", F0),
                ("dtype", "dtype"), ("cold_s", "Cold s", F2), ("first_predict_ms", "First predict ms", F1),
                ("p50_ms", "p50 ms", F1), ("p95_ms", "p95 ms", F1), ("mean_ms", "Mean ms", F1),
                ("max_ms", "Max ms", F1), ("batch_rps", "Rec/s (batched)", F2), ("peak_rss_gb", "Peak RAM GB", F2),
                ("p95_ok", "p95 ok"), ("rps_ok", "rec/s ok"), ("n_latency", "Latency n", F0),
                ("n_batch", "Batch n", F0), ("source", "Source"), ("hardware", "Hardware"), ("error", "Error"))
ONNX_COLS = cols(("model", "Model"), ("accepted", "Accepted"), ("passed", "Passed"), ("n", "Rows", F0),
                 ("batch_size", "Batch", F0), ("max_dp", "Max |dp|", SCI),
                 ("max_dp_threshold", "Max |dp| threshold", SCI),
                 ("argmax_agree", "Argmax agree", F4), ("min_argmax_agree", "Min argmax agree", F4),
                 ("n_disagree", "Disagreements", F0), ("max_dlogit", "Max |dlogit|", SCI),
                 ("seq_min", "Seq len min", F0), ("seq_max", "Seq len max", F0), ("seconds", "Check s", F1),
                 ("exporter", "Exporter"), ("opset", "Opset", F0), ("export_s", "Export s", F1),
                 ("size_gb", "Size GB", F2))


def bench_result_rows(bench: Mapping) -> list[dict[str, Any]]:
    rows = next((v for v in (bench.get("results"), bench.get("rows")) if isinstance(v, list)), [])
    label = machine_label(bench)
    return [{**r, "hardware": r.get("hardware") or label} for r in rows or [] if isinstance(r, Mapping)]


def onnx_rows(bench: Mapping) -> list[dict[str, Any]]:
    out = []
    for model, rec in (bench.get("onnx") or {}).items():
        acc, exp = rec.get("acceptance") or {}, rec.get("export") or {}
        seq = acc.get("seq_len_range") if isinstance(acc.get("seq_len_range"), list) else [None, None]
        out.append({**{k: v for k, v in acc.items() if k != "rows"}, "model": model, "accepted": rec.get("accepted"),
                    "seq_min": seq[0], "seq_max": seq[-1], "exporter": exp.get("exporter"), "opset": exp.get("opset"),
                    "export_s": exp.get("seconds"), "size_gb": _num(exp.get("bytes")) and exp["bytes"] / 1e9})
    checks = bench.get("onnx_check") or bench.get("onnx_acceptance") or {}
    checks = {c.get("model"): c for c in checks if isinstance(c, Mapping)} if isinstance(checks, list) else checks
    return out or [{**c, "model": m} for m, c in checks.items() if isinstance(c, Mapping)]


PLAN_KEYS = ("form", "schema_version", "quick", "latency_n", "batch_n", "warmup", "batch_size", "check_n",
             "threads_requested", "threads_run", "seconds")


def cpu_sheet(bench: Mapping, report: Mapping) -> Sheet:
    results = bench_result_rows(bench)
    first = Table(None, CPU_COLS, results) if results else message_table("no CPU benchmark rows in the given JSON")
    mach = bench.get("machine") if isinstance(bench.get("machine"), Mapping) else {}
    plan = [(k, bench.get(k)) for k in PLAN_KEYS if bench.get(k) is not None]
    plan += [(f"thread_cap.{k}", v) for k, v in (bench.get("thread_cap") or {}).items()]
    machine_t = kv_table("Machine", [("hardware", machine_label(bench)), *leaves(mach), *plan])
    rel = Table("Laya vs matched small encoder (batched rec/s, cpu_budget threads)", cols(
        ("model", "Laya model"), ("encoder", "Encoder"), ("threads", "Threads", F0), ("laya_rps", "Laya rec/s", F2),
        ("encoder_rps", "Encoder rec/s", F2), ("ratio", "Ratio", F3)),
        dig(report, "tables", "relative_throughput") or [])
    notes = kv_table("Benchmark notes", [("note", n) for n in dig(report, "tables", "bench_notes") or []] or
                     [("note", "none")])
    return Sheet("CPU", [first, machine_t, Table("ONNX acceptance", ONNX_COLS, onnx_rows(bench)), rel, notes])


def generic_rows_sheet(name: str, payload: Mapping, rows_keys: Sequence[str] = ("rows", "results")) -> Sheet:
    """A JSON of unknown layout: its row list (scalar leaves, dotted) first, then everything else flattened."""
    rows = next((payload[k] for k in rows_keys if isinstance(payload.get(k), list)), [])
    flat = [dict(leaves({k: v for k, v in r.items() if not isinstance(v, list)})) for r in rows
            if isinstance(r, Mapping)]
    safe, _ = flatten_records([{k: v for k, v in r.items()} for r in flat])
    body = [{k: v for k, v in r.items() if k != "Path"} for r in safe]
    rest, _ = flatten_records({k: v for k, v in payload.items() if k not in rows_keys})
    first = Table(None, cols(*((k, k) for k in dict.fromkeys(k for r in body for k in r))), body) if body \
        else message_table("no rows in the given JSON")
    return Sheet(name, [first, Table("Context", cols(*((k, k) for k in dict.fromkeys(k for r in rest for k in r))),
                                     rest)])
