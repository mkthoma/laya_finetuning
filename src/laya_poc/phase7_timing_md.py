"""Markdown for laya_poc.phase7_timing: strings only (the values come from build_timing's result dict)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

NA = "n/a"
DAGGER = " †"


def rate(x: Any) -> str:
    """Rows per second: whole numbers from 100 up, one decimal from 10, two below."""
    if not isinstance(x, (int, float)):
        return NA
    return f"{x:,.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def num(x: Any, nd: int = 1) -> str:
    return f"{x:,.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else NA


def text(x: Any) -> str:
    if x is None or x == []:
        return NA
    return ", ".join(map(str, x)) if isinstance(x, list) else str(x)


def _cell(text: str) -> str:
    return " ".join(str(text).split("\n")).replace("|", "\\|")


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(map(_cell, header)) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(map(_cell, r)) + " |" for r in rows)]


def _header(res: Mapping) -> list[str]:
    d = res.get("definitions") or {}
    cards = "; ".join(f"{c} = {n}" for c, n in (res.get("cards") or {}).items()) or "not recorded"
    return [
        "# GPU scoring throughput and training time (Phase 3/4 Colab archives)", "",
        f"Generated {res.get('generated_at')} from {res.get('n_runs')} finished runs under "
        f"{', '.join(f'`{r}`' for r in res.get('roots') or [])}. Cards: {cards}. Metrics only: read from done.json, "
        "timing.json, eval/<split>.json, train/summary.json, train/log.jsonl and calibration.json; no FSQ row is "
        "read or written.", "",
        f"**One-pass throughput (rows/s)**: {d.get('throughput', NA)}. The summary takes the median over a group's "
        f"runs (seeds) and its eval splits with >= {res.get('min_scored_rows')} scored rows (val, test_id and the OOD "
        "pools; trap_candidates is smaller and stripped_test has no scored row). "
        f"**Train min**: median training minutes per run. **Total min**: {d.get('total', NA)}.", "",
        "Which field is one pass (read in the writers' code):", "",
        f"- Laya (E*, B2): {d.get('laya', NA)}.",
        f"- B4 small encoders: {d.get('small_encoder', NA)}.",
        f"- B5 LLM: {d.get('llm', NA)}.",
        f"- {d.get('cpu', NA)}.",
        f"- Training: {d.get('train', NA)}.", "",
        "† = derived with a caveat, see the notes column.", ""]


def _summary(res: Mapping) -> list[str]:
    rows = [[s["group"], text(s["card"]), text(s["device"]), text(s["precision"]), text(s["batch_size"]),
             str(s["n_runs"]), rate(s["rows_per_s_median"]) + (DAGGER if s["notes"] else ""),
             f"{rate(s['rows_per_s_min'])} to {rate(s['rows_per_s_max'])}", str(s["n_points"]),
             num(s["train_min_median"], 1), num(s["total_min_median"], 1), "; ".join(s["notes"]) or ""]
            for s in res.get("summary") or []]
    head = ["Group", "Card", "Device", "Precision", "Batch", "Runs", "One-pass rows/s (median)", "Range",
            "Points", "Train min (median/run)", "Total min (median/run)", "Notes"]
    return ["## Summary per group", "", *table(head, rows), ""]


def _throughput(res: Mapping) -> list[str]:
    rows = [[t["group"], t["split"], text(t["rows_forwarded"]), text(t["n_scored"]), str(t["n_runs"]),
             num(t["one_pass_s_median"], 2), rate(t["rows_per_s_median"]) + (DAGGER if t["notes"] else ""),
             text(t["batch_size"]), text(t["device"]), text(t["card"]), "; ".join(t["notes"])]
            for t in res.get("throughput") or []]
    head = ["Group", "Split", "Rows forwarded", "Scored", "Runs", "One-pass s (median)", "Rows/s (median)",
            "Batch", "Device", "Card", "Notes"]
    return ["## Scoring throughput per group and split", "", *table(head, rows), ""]


def _runs(res: Mapping) -> list[str]:
    rows = [[r["run"], r["group"], text(r["seed"]), text(r["card"]), text(r["device"]), text(r["train_precision"]),
             num(r["train_s"]), num(r["scoring_s"], 2), num(r["eval_step_s"]), num(r["export_check_s"]),
             num(r["load_s"]), num(r["total_s"])]
            for r in res.get("runs") or []]
    head = ["Run", "Group", "Seed", "Card", "Device", "Train precision", "Train s", "Scoring s (one pass, all splits)",
            "Eval step s", "Export check s", "Load s", "Total s"]
    return ["## Per run", "", "Eval step s: the Laya evaluate subprocess (both passes on every split plus order "
            "invariance, model load included); the baselines run one process, so only its total is recorded.", "",
            *table(head, rows), ""]


def render_markdown(res: Mapping) -> str:
    return "\n".join([*_header(res), *_summary(res), *_throughput(res), *_runs(res)])
