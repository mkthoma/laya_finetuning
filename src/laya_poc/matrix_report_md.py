"""Markdown rendering of the Phase 3 report (matrix_report.build_report's result). Pure string building.

Numbers are 4-dp values (macro-F1, ECE, rates); a group of several seeds shows `mean (min to max)`. Metrics only:
no FSQ rows ever reach the report.
"""
from __future__ import annotations

from typing import Any, Sequence

NO_THAI_MODELS = ("laya",)   # design §5.12 criterion 1: laya cannot read Thai; its ood_script is excluded
POOLS = ("ood_country", "ood_script", "ood_brand")


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
    return "-" if g["zero_shot"] else ", ".join(str(s) for s in g["seeds"])


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


def exit_section(ex: dict) -> list[str]:
    rows = [[r["run_name"], *(fmt(r[k]) for k in ("done", "best", "T", "val_metrics")),
             "ok" if r["passed"] else "MISSING " + ", ".join(r["missing"])] for r in ex["runs"]]
    return ["## Phase 3 exit check", "", f"Criterion: {ex['criterion']}.", "",
            f"Verdict: **{ex['verdict']}** ({ex['complete']}/{ex['total']} runs complete)", "",
            *table(["Run", "done.json", "best/", "T", "val metrics", "Result"], rows), ""]


def render_markdown(result: dict[str, Any]) -> str:
    ex, groups = result["exit_check"], result["groups"]
    lines = ["# Laya PoC: Phase 3 seed matrix", "",
             f"Generated {result['generated_at']}; card(s) {', '.join(result['cards']) or 'n/a'}; "
             f"{len(result['runs'])} finished of {len(result['configured'])} configured runs. "
             f"Phase 3 exit check: **{ex['verdict']}** ({ex['complete']}/{ex['total']}).", "",
             "Post-T metrics unless marked pre; groups of several seeds show `mean (min to max)` over seeds. "
             "B2 rows are the zero-shot hub checkpoints (shipped temperature). Metrics only: no FSQ rows.", ""]
    if not groups:
        lines += ["No finished runs yet (no finished runs under runs/p3).", ""]
    else:
        lines += [*accuracy_section(groups), *gap_section(groups), *calibration_section(groups),
                  *robustness_section(groups), *variance_section(result), *curve_section(result["learning_curve"]),
                  *runs_section(result["runs"])]
    return "\n".join([*lines, *exit_section(ex)]) + "\n"
