"""Markdown rendering of the Phase 5 report (phase5_report.build_report's result). Pure string building.

Design §7.13 templates, in order: verdict line, main results per pool, traps / abstention / stability, CPU, the
decision checklist (criteria 1-5 per candidate with their numbers, stop and investigate checks, verdicts, pending
inputs), bootstrap CIs, per-class and look-alike tables. Rates are 4-dp fractions; leads, gaps and differences
are macro-F1 points. Metrics only: no FSQ row reaches the report.
"""
from __future__ import annotations

from typing import Any, Mapping

from .decision import FAIL, PENDING
from .matrix_report_md import fmt, fmt_stat, table

POOLS = ("test_id", "ood_country", "ood_script", "ood_brand")
ARROW = "→"


def pts(x: Any, sign: bool = True) -> str:
    return "n/a" if x is None else (f"{100 * x:+.1f}" if sign else f"{100 * x:.1f}")


def ci(c: Mapping | None) -> str:
    return "" if not c or c.get("ci_low") is None else f" [{pts(c['ci_low'])}, {pts(c['ci_high'])}]"


def status_cell(c: Mapping) -> str:
    if c["status"] == PENDING:
        return f"PENDING ({c.get('source') or 'metrics'})"
    return "FAIL (near miss)" if c["status"] == FAIL and c.get("near_miss") else c["status"]


def holds_cell(chk: Mapping) -> str:
    return {True: "YES", False: "no"}.get(chk["holds"], f"PENDING ({chk.get('source') or 'metrics'})")


# ---------------------------------------------------------------- intro and main results

def _intro(result: Mapping) -> list[str]:
    d, m = result["decision"], result["decision"].get("machine")
    bench = m["label"] if m else "none (criterion 5 and stop (c) wait for the benchmark)"
    state = "final" if d["final"] else "provisional"
    rc = result.get("recomputed_check") or {}
    return ["# Laya PoC: Phase 5 evaluation and decision", "",
            f"Generated {result['generated_at']} from the phase5_metrics output of "
            f"{result.get('metrics_generated_at') or 'n/a'} (recomputed macro-F1 matches the eval JSONs: "
            f"{fmt(rc.get('passed'))}). Judged benchmark machine: {bench}. Trap set: "
            f"{(result.get('traps') or {}).get('status') or 'pending'}. Post-T metrics unless marked pre; "
            "multi-seed groups show `mean (min to max)` over seeds. Metrics only: no FSQ rows.", "",
            f"Verdict: **{d['verdict']}** ({state}). {d['statement']}", ""]


def _gap_cell(r: Mapping, pool: str) -> str:
    if pool == "test_id":
        return "-"
    return fmt(r["gap"]) + (" (excluded)" if r["excluded"] else "")


def main_section(result: Mapping) -> list[str]:
    head = ["Arm", "Macro-F1 (mean ± range)", "Acc", f"ECE pre {ARROW} post", "Brier", "NLL", "Acc@80", "Acc@90",
            "Gap vs ID"]
    note = ("Mean ± range is shown as `mean (min to max)` over seeds. Gap vs ID = macro-F1(test_id) - macro-F1(pool) "
            "per run, then averaged; `(excluded)`: the pool is not judged for that model (`laya` cannot read Thai). "
            "B5 scores evaluation subsets. E6 train subsets are in the Phase 3/4 report.")
    out = ["## Main results (design §7.13)", "", note, ""]
    for pool in POOLS:
        rows = [[r["name"], fmt_stat(r["macro_f1"]), fmt(r["acc"]), f"{fmt(r['ece_pre'])} {ARROW} {fmt(r['ece_post'])}",
                 fmt(r["brier"]), fmt(r["nll"]), fmt(r["acc80"]), fmt(r["acc90"]), _gap_cell(r, pool)]
                for r in result["tables"]["main"][pool]]
        out += [f"### {pool}", "", *table(head, rows), ""]
    return out


def _trap_cell(r: Mapping) -> str:
    if r["trap_status"] != "ok":
        return f"pending (unannotated candidates: {fmt(r['trap_acc'])})"
    return f"{fmt(r['trap_acc'])} / {fmt(r['trap_multi'])}"


def _range(x: Any, r: Mapping, scale: float) -> str:
    return "-" if (r.get("n_runs") or 1) < 2 or x is None else f"{scale * x:.{1 if scale > 1 else 4}f}"


def _order_cell(r: Mapping) -> str:
    return "n/a" if r["order_inv"] is None else f"{fmt(r['order_inv'])} (all >= 0.99: {fmt(r['order_ok'])})"


def robustness_section(result: Mapping) -> list[str]:
    head = ["Arm", "Trap acc (all / multi)", "Stripped: abstain rate", "Stripped: false-confident",
            "Flip rate (test_id)", "Order invariance", "Seed range macro-F1 test_id (points)",
            "Seed range ECE test_id"]
    rows = [[r["name"], _trap_cell(r), fmt(r["abstain"]), fmt(r["false_conf"]), fmt(r["flip_rate"]),
             _order_cell(r), _range(r["range_f1"], r, 100),
             _range(r["range_ece"], r, 1)] for r in result["tables"]["robustness"]]
    note = ("Trap accuracy on the annotated keep set, multi-category items split out; before annotation the "
            "unannotated-candidate accuracy is shown for information. Stripped: no-evidence rows, abstention at the "
            "val tau and the share answered with confidence > 0.8. Flip rate: rows whose argmax differs across "
            "seeds. Order invariance: mean argmax agreement over shuffled field orders.")
    return ["## Traps, abstention and stability", "", note, "", *table(head, rows), ""]


# ---------------------------------------------------------------- CPU

def _onnx_line(result: Mapping) -> list[str]:
    acc = result["tables"].get("onnx_acceptance") or {}
    if not acc:
        return []
    word = {True: "passed", False: "FAILED (ONNX rows not judged)", None: "not recorded"}
    return ["ONNX acceptance check (judged machine; argmax agreement and max |dp| vs PyTorch fp32): "
            + "; ".join(f"{m} {word[v]}" for m, v in acc.items()) + ".", ""]


def cpu_section(result: Mapping) -> list[str]:
    rows = result["tables"]["cpu"]
    title = "## CPU (design §7.11)"
    if not rows:
        return [title, "", "PENDING: no benchmark JSON given (--bench): criterion 5 and stop (c) stay pending.", ""]
    head = ["Model", "Backend", "Threads", "Cold start (s)", "p50 (ms)", "p95 (ms)", "Batch rec/s", "Peak RAM (GB)",
            "Hardware"]
    body = [[r["model"], r["backend"], r["threads"], fmt(r["cold_s"], 2), fmt(r["p50_ms"], 1), fmt(r["p95_ms"], 1),
             fmt(r["batch_rps"], 1), fmt(r["peak_rss_gb"], 2), (r["hardware"] or "n/a") + (" (judged)" if r["judged"]
                                                                                             else "")]
            for r in rows]
    rel = [[r["model"], r["encoder"], r["threads"], fmt(r["laya_rps"], 1), fmt(r["encoder_rps"], 1),
            f"{r['ratio']:.2f}x"] for r in result["tables"]["relative_throughput"]]
    out = [title, "", "fp32; batch-1 latency and batched throughput (batch 32, length-sorted); the first --bench is "
           "the judged machine (criterion 5 at 4 threads, stop (c) at 8 threads with ONNX).", "", *table(head, body), "",
           *_onnx_line(result)]
    notes = result["tables"].get("bench_notes") or []
    out += [*(f"- {n}" for n in notes), ""] if notes else []
    if rel:
        out += ["Throughput relative to the language-matched small encoder (faster Laya backend):", "",
                *table(["Laya model", "Small encoder", "Threads", "Laya rec/s", "Encoder rec/s", "Ratio"], rel), ""]
    return out


# ---------------------------------------------------------------- decision checklist

def _c_number(c: Mapping) -> str:
    v = c["value"]
    if c["id"] == 1:
        if v is None:
            return "n/a"
        return (f"{pts(v)} pts ({c.get('b3')} {pts(c['leads']['B3'])}{ci(c['ci'].get('B3'))}; "
                f"{c['b4']} {pts(c['leads']['B4'])}{ci(c['ci'].get('B4'))})")
    if c["id"] == 2:
        gaps = ", ".join(f"{p} {pts(g, False)}" for p, g in c["gaps"].items())
        return "n/a" if v is None else f"{pts(v, False)} pts (worst {c['worst_pool']}; {gaps})"
    if c["id"] == 3:
        return "n/a" if v is None else (f"{v:.4f} (test_id {fmt(c['ece_test_id'])}; OOD avg "
                                        f"{fmt(c['ece_ood_avg'])})")
    if c["id"] == 4:
        return "n/a" if v is None else (f"{v:.4f} (multi {fmt(c.get('multi'))}) vs {c['best']} "
                                        f"{fmt(c['threshold'])}")
    return "n/a" if v is None else (f"p95 {v['p95_ms']:.0f} ms, {v['batch_rps']:.1f} rec/s ({v['backend']}, "
                                    f"{v['threads']} threads)")


def _threshold(c: Mapping, th: Mapping) -> str:
    return {1: f">= +{100 * th['lead']:.1f} pts over both", 2: f"<= {100 * th['max_gap']:.1f} pts on each pool",
            3: f"<= {th['max_ece']:.2f} on both", 4: ">= the best baseline",
            5: f"p95 <= {th['p95_ms']:.0f} ms and >= {th['min_rps']:.0f} rec/s"}[c["id"]]


def _check_number(chk: Mapping) -> str:
    if chk["id"] == "seed_range":
        return ", ".join(f"{k} {pts(chk['ranges'][k], False)}" for k in chk["flagged"]) or "none flagged"
    if chk["id"] == "c":
        return "n/a" if chk.get("p95_ms") is None else f"p95 {chk['p95_ms']:.0f} ms ({chk['threads']} threads)"
    if chk["id"] == "a" and chk["holds"] is not None:
        return f"below on test_id: {fmt(chk['below_on_id'])}; on the OOD average: {fmt(chk['below_on_ood'])}"
    if chk["id"] == "b":
        return f"test_id {fmt(chk.get('ece_test_id'))}; OOD avg {fmt(chk.get('ece_ood_avg'))}"
    if chk["id"] == "near_miss":
        near = set(chk.get("near") or [])
        fails = [f"C{i}" + (" (near)" if i in near else "") for i in chk.get("fails") or []]
        return "fails: " + (", ".join(fails) or "none")
    return ""


def candidate_block(c: Mapping, th: Mapping) -> list[str]:
    rows = [[f"C{x['id']} {x['name']}", _c_number(x), _threshold(x, th), status_cell(x), x["note"]]
            for x in c["criteria"]]
    s = c["c1_strict"]
    rows.insert(1, ["C1 strict (sensitivity): best B4 of either language", _c_number(s), _threshold(s, th),
                    status_cell(s), "reported only; not in the verdict"])
    rows += [[f"stop ({x['id']}) {x['name']}", _check_number(x), "", holds_cell(x), x["note"]] for x in c["stop"]]
    rows += [[f"investigate ({n}) {x['name']}", _check_number(x), "", holds_cell(x), x["note"]]
             for n, x in zip(("i", "ii", "iii"), c["investigate"])]
    excl = f"; {', '.join(c['excluded_pools'])} excluded: cannot read Thai by design" if c["excluded_pools"] else ""
    seeds = ", ".join(str(s) for s in c.get("seeds") or [])
    reach = "" if c["possible"] == [c["verdict"]] else f" (still reachable: {', '.join(c['possible'])})"
    return [f"### {c['label']} (seeds {seeds}; OOD pools {', '.join(c['pools'])}{excl})", "",
            *table(["Check", "Number", "Threshold", "Result", "Note"], rows), "",
            f"Candidate verdict: **{c['verdict']}**{reach}", ""]


def decision_section(result: Mapping) -> list[str]:
    d = result["decision"]
    note = ("Seed means. C1's B4 is the language-matched small encoder (config phase5.small_encoder_match); the strict "
            "variant is a sensitivity check. Brackets: bootstrap 95% CI of the lead. A near miss fails by less than "
            "half the criterion's margin. Stop is global (every candidate must hit one); the overall verdict is PASS "
            "if any candidate passes, else STOP, INVESTIGATE, or NO PASS (no formal stop condition met — LDS "
            "judgement).")
    out = ["## Decision checklist (design §5.12)", "", note, ""]
    for c in d["candidates"]:
        out += candidate_block(c, d["thresholds"])
    state = "final" if d["final"] else "provisional"
    out += ["### Overall", "", f"Verdict: **{d['verdict']}** ({state}). {d['statement']}", ""]
    if d["pending_inputs"]:
        rows = [[p["input"], ", ".join(p["items"]), fmt(p["can_change_verdict"])] for p in d["pending_inputs"]]
        out += [*table(["Pending input", "Items", "Can change the verdict"], rows), ""]
    return out


# ---------------------------------------------------------------- bootstrap and confusions

def bootstrap_section(result: Mapping) -> list[str]:
    lead = 100 * result["decision"]["thresholds"]["lead"]
    head = ["Candidate", "Baseline", "Pool", "Diff (points)", "95% CI (points)", "P(diff > 0)",
            f"P(diff >= {lead:.0f} points)", "Rows"]
    rows = [[c.get("candidate"), c.get("baseline"), c.get("pool") + (f" ({'+'.join(c['pools'])})" if c.get("pool") ==
                                                                     "ood_average" and c.get("pools") else ""),
             pts(c.get("diff")), ci(c).strip() or "n/a", fmt(c.get("p_gt_0")), fmt(c.get("p_ge_lead")),
             "baseline's evaluated ids" if c.get("restricted") else "all"] for c in result["tables"]["bootstrap"]]
    note = ("Paired item bootstrap of the seed-mean macro-F1 difference (candidate - baseline), design §5.11; "
            "ood_average uses the candidate's criterion-1 pools.")
    return ["## Bootstrap 95% CIs", "", note, "", *(table(head, rows) if rows else ["No bootstrap comparisons."]), ""]


def _prf(pc: Mapping | None) -> str:
    if not pc:
        return "n/a"
    return " / ".join(fmt((pc.get(k) or {}).get("mean"), 3) for k in ("p", "r", "f1"))


def confusion_sections(result: Mapping) -> list[str]:
    conf, best = result["tables"]["confusion"], result["tables"]["best_baseline"]
    gids = list(next(iter(conf.values()))["per_class"]) if conf else []
    out = ["## Per-class P / R / F1 (candidates and the best baseline)", "",
           f"Cells: P / R / F1, seed means. Best baseline: {best or 'n/a'} (highest test_id macro-F1).", ""]
    for pool, t in conf.items():
        rows = [[k, *(_prf(t["per_class"][g].get(k)) for g in gids)] for k in t["labels"]]
        out += [f"### {pool}", "", *table(["Class", *gids], rows), ""]
    out += ["## Look-alike confusions (design §5.11 pairs)", "",
            "Rate = confusions / true-class support over the group's runs (count/support).", ""]
    for pool, t in conf.items():
        cells = {g: {(x["true"], x["pred"]): x for x in t["lookalike"][g]} for g in gids}
        pairs = list(dict.fromkeys(k for g in gids for k in cells[g]))
        rows = [[f"{a} {ARROW} {b}", *(_look(cells[g].get((a, b))) for g in gids)] for a, b in pairs]
        out += [f"### {pool}", "", *table([f"True {ARROW} predicted", *gids], rows), ""]
    return out


def _look(x: Mapping | None) -> str:
    return "n/a" if not x else f"{fmt(x.get('rate'), 3)} ({x.get('count')}/{x.get('support')})"


def render_markdown(result: Mapping) -> str:
    lines = [*_intro(result), *main_section(result), *robustness_section(result), *cpu_section(result),
             *decision_section(result), *bootstrap_section(result), *confusion_sections(result)]
    return "\n".join(lines) + "\n"
