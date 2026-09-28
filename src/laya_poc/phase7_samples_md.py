"""Markdown rendering of the Phase 7 sample inferences (phase7_samples.SplitView list). Pure string building.

The page holds FSQ rows, so it starts with the FSQ NOTICE verbatim and a LOCAL ONLY line (design doc Appendix D);
phase7_samples refuses to write it anywhere git would track. Records render as `name · locality · country ·
other=value` with each value clipped and escaped for a markdown table cell; answers as `label (0.62) ✓`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .phase7_samples import Agreement, Options, Pred, SplitView

DESCRIPTIONS = {
    "test_id": "in-distribution test (countries and chains seen in training)",
    "ood_country": "unseen countries",
    "ood_script": "non-Latin script (Thai)",
    "ood_brand": "unseen chains/brands",
    "stripped_test": "no evidence (every field except the country removed; models should abstain)",
}
FRONT_FIELDS = ("name", "locality", "country")
MAX_VALUE = 70
RIGHT, WRONG = "✓", "✗"
LOCAL_ONLY = ("**LOCAL ONLY: contains FSQ-derived rows. Do not commit or share outside the team "
              "(design doc Appendix D).**")


# ---------------------------------------------------------------- cells

def clip(value: Any, limit: int = MAX_VALUE) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def md_escape(text: str) -> str:
    """Safe inside a markdown table cell: newlines collapsed, backslash, pipe and `<` escaped."""
    return " ".join(text.split()).replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;")


def render_record(state: Mapping[str, Any]) -> str:
    """`name · locality · country · other=value`, each value clipped to MAX_VALUE characters."""
    front = [clip(state[k]) for k in FRONT_FIELDS if state.get(k) not in (None, "")]
    rest = [f"{k}={clip(v)}" for k, v in state.items() if k not in FRONT_FIELDS and v not in (None, "")]
    return md_escape(" · ".join(front + rest))


def row_state(row: Mapping[str, Any]) -> dict:
    state = row.get("state") or {}
    return json.loads(state) if isinstance(state, str) else dict(state)


def answer(pred: Pred, labels: Sequence[str], truth: int, rank: int = 0) -> str:
    """`label (0.62) ✓` for the top answer, `label (0.21)` for a runner-up, `abstained` without p."""
    if pred.p is None:
        return "abstained"
    i = pred.ranked()[rank]
    mark = "" if rank else " " + (RIGHT if i == truth else WRONG)
    return f"{labels[i]} ({pred.p[i]:.2f}){mark}"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def run_label(view: SplitView, run: str) -> str:
    model = view.data.models[run]
    return run if model == run else f"{run} ({model})"


def run_header(view: SplitView, run: str) -> str:
    note = view.notes.get(run)
    return f"{run} ({note})" if note else run


# ---------------------------------------------------------------- labelled splits

def accuracy_line(view: SplitView, run: str) -> str:
    s, note = view.summaries[run], view.notes.get(run)
    acc = "n/a" if s["accuracy"] is None else f"{s['accuracy']:.4f}"
    tail = f"; {note}" if note else ""
    return (f"- {run_label(view, run)}: accuracy {acc} ({s['correct']}/{s['answered']} right, "
            f"{s['abstained']} abstained{tail})")


def two_by_two(ag: Agreement, focus: str, versus: str) -> list[str]:
    c = ag.counts()
    rows = [[f"**{focus} right**", c["both_right"], c["focus_only_right"]],
            [f"**{focus} wrong**", c["versus_only_right"],
             f"{c['both_wrong']} (same wrong label: {c['both_wrong_same_label']})"]]
    left_out = f" {c['either_abstained']} rows where either abstained are left out." if c["either_abstained"] else ""
    return [f"{focus} vs {versus}, rows both answered: {c['n'] - c['either_abstained']}.{left_out}", "",
            *table(["", f"**{versus} right**", f"**{versus} wrong**"], rows), ""]


def cell_titles(opts: Options) -> dict[str, str]:
    return {"both_right": "Both right", "focus_only_right": f"Only {opts.focus} right",
            "versus_only_right": f"Only {opts.versus} right", "both_wrong": "Both wrong"}


def example_table(view: SplitView, ids: Sequence[str], focus: str) -> list[str]:
    sd = view.data
    by_id = {str(r["id"]): r for r in sd.rows}
    runs = list(sd.preds)
    head = ["Record", "True label", *(run_header(view, r) for r in runs), f"{focus} runner-up"]
    rows = [[render_record(row_state(by_id[i])), sd.labels[sd.y[i]],
             *(answer(sd.preds[r][i], sd.labels, sd.y[i]) for r in runs),
             answer(sd.preds[focus][i], sd.labels, sd.y[i], rank=1)] for i in ids]
    return table(head, rows)


def labelled_section(view: SplitView, opts: Options) -> list[str]:
    sd = view.data
    out = [f"Rows: {len(sd.rows)}. Accuracy is top-1 over answered rows.", ""]
    out += [accuracy_line(view, r) for r in sd.preds] + [""]
    out += two_by_two(view.agreement, opts.focus, opts.versus)
    for cell, title in cell_titles(opts).items():
        ids, total = view.samples[cell], len(view.agreement.cells[cell])
        out += [f"### {title} ({total} rows; showing {len(ids)})", ""]
        out += (example_table(view, ids, opts.focus) if ids else ["_No rows._"]) + [""]
    return out


# ---------------------------------------------------------------- no-evidence splits

def stripped_line(view: SplitView, run: str) -> str:
    s, u = view.summaries[run], view.ungated.get(run)
    line = (f"- {run_label(view, run)}: abstained {s['abstained']}/{s['n']} "
            f"({s['abstained'] / max(s['n'], 1):.4f}); answered {s['answered']}")
    if s["accuracy"] is not None:
        line += f" (accuracy {s['accuracy']:.4f})"
    if u:
        line += f"; without the gate: accuracy {u['accuracy']:.4f}, mean confidence {u['mean_confidence']:.4f}"
    return line


def stripped_cell(view: SplitView, run: str, rid: str) -> str:
    sd = view.data
    gated = sd.preds[run][rid]
    cell = answer(gated, sd.labels, sd.y[rid])
    ungated = sd.ungated.get(run, {}).get(rid)
    if gated.p is None and ungated is not None and ungated.p is not None:
        cell += "; ungated: " + answer(ungated, sd.labels, sd.y[rid])
    return cell


def stripped_section(view: SplitView) -> list[str]:
    sd, ids = view.data, view.samples["rows"]
    out = [f"Rows: {len(sd.rows)}. The gate should abstain on every row; `ungated` is what the model would have "
           "answered without it (preds/no_gate), when recorded.", ""]
    out += [stripped_line(view, r) for r in sd.preds] + ["", f"Sampled rows ({len(ids)}):", ""]
    by_id = {str(r["id"]): r for r in sd.rows}
    rows = [[render_record(row_state(by_id[i])), sd.labels[sd.y[i]], *(stripped_cell(view, r, i) for r in sd.preds)]
            for i in ids]
    return out + table(["Record", "True label", *(run_header(view, r) for r in sd.preds)], rows) + [""]


# ---------------------------------------------------------------- page

def intro(notice: str, views: Sequence[SplitView], runs: Mapping[str, Path], opts: Options,
          generated_at: str) -> list[str]:
    models = views[0].data.models if views else {}
    legend = [f"- {name}: `{Path(d).name}` (model {models.get(name, '?')})" for name, d in runs.items()]
    how = (f"Each section samples real records from one evaluation split and shows what every model predicted. "
           f"`label (0.62)` is a model's top answer with its probability (after temperature scaling); {RIGHT} means "
           f"it matches the true label, {WRONG} that it does not. `{opts.focus} runner-up` is the focus model's "
           f"second choice. Rows are grouped by whether {opts.focus} (focus) and {opts.versus} (comparison) were "
           f"right, and up to {opts.per_cell} rows are sampled from each group, so the examples show agreement and "
           "disagreement, not overall accuracy: use each section's accuracy lines and 2x2 table for that. A record "
           "shows the fields the models saw: name · locality · country, then other fields as key=value (values cut "
           f"at {MAX_VALUE} characters).")
    return [notice.rstrip("\n"), "", LOCAL_ONLY, "", "# Laya PoC: sample inferences", "",
            f"Generated {generated_at} · seed {opts.seed} · {opts.per_cell} rows per cell.", "",
            "## How to read this", "", how, "", "Runs:", "", *legend, ""]


def render_markdown(notice: str, views: Sequence[SplitView], runs: Mapping[str, Path], opts: Options,
                    generated_at: str) -> str:
    out = intro(notice, views, runs, opts, generated_at)
    for view in views:
        split = view.data.split
        out += [f"## {split}: {DESCRIPTIONS.get(split, split)}", ""]
        out += stripped_section(view) if view.agreement is None else labelled_section(view, opts)
    return "\n".join(out).rstrip("\n") + "\n"
