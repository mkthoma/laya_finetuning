"""Markdown rendering of the Phase 3 report (matrix_report.build_report's result). Pure string building.

Numbers are 4-dp values (macro-F1, ECE, rates); a group of several seeds shows `mean (min to max)`. Metrics only:
no FSQ rows ever reach the report. With Phase 4 baselines configured or present, the report also has the
decision criterion 1 early read and the Phase 4 exit check (matrix_decision); a Phase 3 only report is unchanged.
"""
from __future__ import annotations

from typing import Any, Sequence

from .matrix_decision import LEAD, NO_THAI_MODELS

POOLS = ("ood_country", "ood_script", "ood_brand")
LAYA = "laya"


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def fmt_stat(s: dict | None, nd: int = 4) -> str:
    """`mean` for one value, `mean (min to max)` for several."""
    if not s:
        return "n/a"
    if s["n"] == 1:
        return fmt(float(s["mean"]), nd)
    return f"{s['mean']:.{nd}f} ({s['min']:.{nd}f} to {s['max']:.{nd}f})"


def _pre_post(g: dict, pre: str, post: str) -> str:
    m = g["metrics"]
    return f"{fmt_stat(m.get(pre))} -> {fmt_stat(m.get(post))}"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def _seeds(g: dict) -> str:
    seeds = [s for s in g["seeds"] if s is not None]
    return "-" if g["zero_shot"] or not seeds else ", ".join(str(s) for s in seeds)


def _has_baselines(groups: Sequence[dict]) -> bool:
    return any(g.get("kind", LAYA) != LAYA for g in groups)


def accuracy_section(groups: Sequence[dict]) -> list[str]:
    head = ["Group", "Runs", "Seeds", "val", "val (9-class)", "test_id", *POOLS]
    rows = [[g["label"], g["n_runs"], _seeds(g), *(fmt_stat(g["metrics"].get(m)) for m in (
        "val_macro_f1", "val_macro_f1_9", "test_id_macro_f1", *(f"{p}_macro_f1" for p in POOLS)))] for g in groups]
    return ["## Accuracy: macro-F1 (post-T)", "", *table(head, rows), ""]


def _gap_cell(g: dict, pool: str) -> str:
    cell = fmt_stat(g["metrics"].get(f"gap_{pool}"))
    return cell + (" (excluded)" if pool == "ood_script" and g["model"] in NO_THAI_MODELS else "")


def gap_section(groups: Sequence[dict]) -> list[str]:
    rows = [[g["label"], *(_gap_cell(g, p) for p in POOLS)] for g in groups]
    note = ("Gap = macro-F1(test_id) - macro-F1(pool) per run, then averaged (design §5.11). The decision rule "
            "(§5.12, judged in Phase 5) wants <= 5 points on each included pool; `laya` cannot read Thai by design, "
            "so its ood_script gap is excluded there and only reported.")
    return ["## ID->OOD gap", "", note, "", *table(["Group", *POOLS], rows), ""]


def calibration_section(groups: Sequence[dict]) -> list[str]:
    rows = [[g["label"], fmt_stat(g["metrics"].get("T")), _pre_post(g, "val_ece_pre", "val_ece_post"),
             _pre_post(g, "test_id_ece_pre", "test_id_ece_post")] for g in groups]
    note = ("T: fitted on val by export_check for trained runs; the shipped temperature for zero-shot. ECE: 15 "
            "equal-width bins on answer_confidence, before and after the temperature.")
    if _has_baselines(groups):
        note += (" Baselines: T from calibration.json, fitted on val (B1 majority/prior: T = 1; B5 on its 500 val "
                 "rows).")
    return ["## Calibration: T and ECE pre -> post", "", note, "",
            *table(["Group", "T", "val ECE pre -> post", "test_id ECE pre -> post"], rows), ""]


def robustness_section(groups: Sequence[dict]) -> list[str]:
    head = ["Group", "Trap acc (unannotated candidates)", "Stripped: abstain rate at tau",
            "Stripped: false-confident (conf > 0.8)", "Order invariance (mean agreement)", "All >= 0.99"]
    rows = [[g["label"], *(fmt_stat(g["metrics"].get(m)) for m in (
        "trap_candidates_acc", "stripped_abstain_rate", "stripped_false_confident", "order_invariance")),
        fmt(g["order_invariance_all_99"])] for g in groups]
    note = ("Trap accuracy is on every trap CANDIDATE with its automatic label (annotation pending), so it is not "
            "the §5.11 trap metric; Phase 5 restricts it to the annotated keep set via fsq_place_id. tau is chosen "
            "on val (abstain.target_acc / min_coverage).")
    return ["## Traps, no evidence and field-order invariance", "", note, "", *table(head, rows), ""]


def variance_section(result: dict) -> list[str]:
    thr = result.get("seed_variance_threshold", 0.03)
    title = f"## Seed variance (design §5.12: Investigate when a macro-F1 range over seeds exceeds {100 * thr:.0f} points)"
    sv = result["seed_variance"]
    if not sv:
        return [title, "", "No group has more than one finished seed.", ""]
    rows = [[v["group"], v["metric"], v["n"], f"{100 * v['range']:.1f}", "INVESTIGATE" if v["flagged"] else "ok"]
            for v in sv]
    flagged = sorted({v["group"] for v in sv if v["flagged"]})
    verdict = f"Flagged: {', '.join(flagged)}." if flagged else "No group exceeds the threshold."
    return [title, "", verdict, "", *table(["Group", "Metric", "Seeds", "Range (points)", "Flag"], rows), ""]


def curve_section(points: Sequence[dict]) -> list[str]:
    title = "## Learning curve (E6: train subsets + the full-size run of the same seed)"
    if not points:
        return [title, "", "No train-subset run has finished.", ""]
    rows = [[p["curve"], fmt(p["n"]) + (" (full)" if p["full"] else ""), p["run_name"], fmt(p["val_macro_f1"]),
             fmt(p["test_id_macro_f1"])] for p in points]
    return [title, "", *table(["Curve", "n (train rows)", "Run", "val macro-F1", "test_id macro-F1"], rows), ""]


def _minutes(x: Any) -> str:
    return "n/a" if x is None else f"{x / 60:.1f}"


def runs_section(runs: Sequence[dict]) -> list[str]:
    head = ["Run", "Card", "Epochs", "Stop", "Best opt", "T", "val macro-F1", "test_id macro-F1", "Order inv.",
            "Train (min)", "Run (min)", "export_check"]
    rows = [[r["run_name"], fmt(r.get("card")), fmt(r.get("epochs_run"), 2), fmt(r.get("stop_reason")),
             fmt(r.get("best_opt_step")), fmt(r.get("T")) + (" CLAMPED" if r.get("clamped") else ""),
             fmt(r.get("val_macro_f1")), fmt(r.get("test_id_macro_f1")), fmt(r.get("order_invariance")),
             _minutes(r.get("train_seconds")), _minutes(r.get("run_seconds")),
             {True: "passed", False: "FAILED", None: "-"}.get(r.get("export_check_passed"), "-")] for r in runs]
    return ["## Runs", "", *table(head, rows), ""]


def _lead_cell(row: dict) -> str:
    return "n/a" if row["lead"] is None else f"{100 * row['lead']:+.1f}"


def _verdict_cell(row: dict) -> str:
    if row["laya_ood_avg"] is None:
        return "n/a (the arm has no macro-F1 on every pool)"
    if row["passed"] is None:
        return "n/a (no trained baseline of this scheme)"
    verdict = "PASS" if row["passed"] else "FAIL"
    return f"{verdict} (incomplete: no {' / '.join(row['missing'])} {row['scheme']})" if row["missing"] else verdict


def _compared(row: dict) -> str:
    return "; ".join(f"{b['label']} {fmt((b['ood_avg'] or {}).get('mean'))}" for b in row["baselines"]) or "none"


def decision_section(rows: Sequence[dict]) -> list[str]:
    title = "## Decision criterion 1 (early read)"
    note = (f"Information only: Phase 5 applies the decision rule (design §5.12). Criterion 1: on OOD, a "
            f"fine-tuned Laya checkpoint (mean over its seeds) must beat BOTH char TF-IDF+LR (B3) and the fine-tuned "
            f"small encoder (B4) by >= {100 * LEAD:.0f} macro-F1 points, averaged over the OOD pools. `laya` excludes "
            f"ood_script (it cannot read Thai by design) and every baseline is averaged over the same pools as the arm "
            f"it is compared with. The best seed-mean B3/B4 group of the same scheme is the bar (a +1.5 to +3 point "
            f"lead is the §5.12 Investigate band). Zero-shot (B2), head-only (E4) and train-subset (E6) runs are not "
            f"candidates.")
    if not rows:
        return [title, "", note, "", "No fine-tuned Laya arm has finished under the runs roots of this report "
                "(combine with the Phase 3 archive: `matrix report --runs-root <p3 runs> --runs-root <p4 runs>`).", ""]
    head = ["Laya arm", "Seeds", "OOD pools", "Laya OOD avg", "Best trained baseline", "Its OOD avg",
            "Lead (points)", f">= {100 * LEAD:.0f} points", "Compared (seed-mean OOD avg)"]
    body = [[r["label"], ", ".join(str(s) for s in r["seeds"]), ", ".join(r["pools"]), fmt_stat(r["laya_ood_avg"]),
             r["best"] or "n/a", fmt(r["best_ood_avg"]), _lead_cell(r), _verdict_cell(r), _compared(r)] for r in rows]
    return [title, "", note, "", *table(head, body), ""]


NOT_RUN_NOTE = ("not in these results: no run of this phase under the given runs roots (combine archives with "
                "`matrix report --runs-root A --runs-root B --archived`)")


def phase4_exit_section(ex: dict | None) -> list[str]:
    if not ex:
        return []
    if ex["verdict"] == "NOT RUN":
        return ["## Phase 4 exit check", "", f"Verdict: **NOT RUN**: {NOT_RUN_NOTE}.", ""]
    rows = [[r["run_name"], r["kind"] + (" (optional)" if r["optional"] else ""), fmt(r["done"]), fmt(r["preds"]),
             "optional: not run" if r["skipped"] else ("ok" if r["passed"] else "MISSING " + ", ".join(r["missing"]))]
            for r in ex["runs"]]
    skipped = f"; optional not run: {', '.join(ex['skipped_optional'])}" if ex["skipped_optional"] else ""
    return ["## Phase 4 exit check", "", f"Criterion: {ex['criterion']}.", "",
            f"Verdict: **{ex['verdict']}** ({ex['complete']}/{ex['total']} baseline runs complete{skipped})", "",
            *table(["Run", "Kind", "done.json", "preds (every split)", "Result"], rows), ""]


def exit_section(ex: dict) -> list[str]:
    if ex["verdict"] == "NOT RUN":
        return ["## Phase 3 exit check", "", f"Verdict: **NOT RUN**: {NOT_RUN_NOTE}.", ""]
    rows = [[r["run_name"], *(fmt(r[k]) for k in ("done", "best", "T", "val_metrics")),
             "ok" if r["passed"] else "MISSING " + ", ".join(r["missing"])] for r in ex["runs"]]
    return ["## Phase 3 exit check", "", f"Criterion: {ex['criterion']}.", "",
            f"Verdict: **{ex['verdict']}** ({ex['complete']}/{ex['total']} runs complete)", "",
            *table(["Run", "done.json", "best/", "T", "val metrics", "Result"], rows), ""]


def _intro(result: dict[str, Any], phase4: bool) -> list[str]:
    ex, p4 = result["exit_check"], result.get("phase4_exit_check")
    title = "# Laya PoC: Phase 3 seed matrix" + (" and Phase 4 baselines" if phase4 else "")
    p4_line = f" Phase 4 exit check: **{p4['verdict']}** ({p4['complete']}/{p4['total']})." if p4 else ""
    base_note = (" B1/B3/B4/B5 rows are the Phase 4 baselines (majority/prior, char TF-IDF+LR, fine-tuned small "
                 "encoders, the optional reference LLM on evaluation subsets)." if phase4 else "")
    return [title, "", f"Generated {result['generated_at']}; card(s) {', '.join(result['cards']) or 'n/a'}; "
            f"{len(result['runs'])} finished of {len(result['configured'])} configured runs. "
            f"Phase 3 exit check: **{ex['verdict']}** ({ex['complete']}/{ex['total']}).{p4_line}", "",
            "Post-T metrics unless marked pre; groups of several seeds show `mean (min to max)` over seeds. "
            f"B2 rows are the zero-shot hub checkpoints (shipped temperature).{base_note} Metrics only: no FSQ rows.",
            ""]


def render_markdown(result: dict[str, Any]) -> str:
    ex, groups = result["exit_check"], result["groups"]
    phase4 = bool(result.get("phase4_exit_check")) or _has_baselines(groups)
    lines = _intro(result, phase4)
    if not groups:
        where = ", ".join(result.get("runs_roots") or []) or "runs/p3"
        lines += [f"No finished runs yet (no finished runs under {where}).", ""]
    else:
        lines += [*accuracy_section(groups), *gap_section(groups), *calibration_section(groups),
                  *robustness_section(groups), *variance_section(result), *curve_section(result["learning_curve"]),
                  *(decision_section(result.get("decision_criterion_1") or []) if phase4 else []),
                  *runs_section(result["runs"])]
    tail = [*exit_section(ex), *phase4_exit_section(result.get("phase4_exit_check"))]
    return "\n".join([*lines, *tail]) + "\n"
