"""Compare the two annotators' trap CSVs before `traps merge` (design doc §5.3 step 3; docs/trap_annotation_guide.md).

Read-only. Prints how far the annotation is, the agreement on the rows both annotators marked (share of equal marks
and Cohen's kappa), the projected kept set, and every row the lead must decide (the annotators disagree and
keep_lead is empty) as the spreadsheet row number in annotator 1's file with its pattern and fsq_place_id. Values
follow `laya_poc.traps` exactly: 1 = keep, 0 = drop, empty = not yet marked ("1.0"/"0.0" from spreadsheets pass).
Exit code 0 when the pair is ready for `traps merge`, 1 when rows are still open, 2 on a file problem.

    python tools/p5_trap_check.py data/trap_candidates_a1.csv data/trap_candidates_a2.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from laya_poc.config import load_config  # noqa: E402
from laya_poc.traps import read_candidates  # noqa: E402

KEY = ["pattern", "fsq_place_id"]
MARKS = {"1": 1, "1.0": 1, "0": 0, "0.0": 0, "": None}


def _marks(values, column: str, rows: Sequence[int]) -> list[int | None]:
    out, bad = [], []
    for value, row in zip(values, rows):
        v = str(value).strip()
        if v not in MARKS:
            bad.append(row)
        out.append(MARKS.get(v))
    if bad:
        raise ValueError(f"{column}: only 1, 0 or empty are allowed; fix spreadsheet rows {bad[:20]}")
    return out


def align(a1_csv: Path, a2_csv: Path) -> tuple[list[dict], bool]:
    """One record per candidate in annotator 1's row order, with both marks and the lead's; and whether annotator 2's
    file is in another order (the merge matches rows by pattern + fsq_place_id, but keep the original order)."""
    a1, a2 = read_candidates(a1_csv), read_candidates(a2_csv)
    for df, name, col in ((a1, a1_csv, "keep_a1"), (a2, a2_csv, "keep_a2")):
        missing = [c for c in (*KEY, "multi", col) if c not in df]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
    ids1, ids2 = list(zip(a1["pattern"], a1["fsq_place_id"])), list(zip(a2["pattern"], a2["fsq_place_id"]))
    if len(ids1) != len(ids2) or set(ids1) != set(ids2) or len(set(ids1)) != len(ids1):
        raise ValueError("the two CSVs do not hold the same candidates (pattern, fsq_place_id): both must be "
                         "unedited copies of data/trap_candidates.csv (no rows added, removed or changed)")
    other = a2.set_index(KEY).loc[ids1].reset_index()
    rows = [i + 2 for i in range(len(a1))]  # spreadsheet rows: the header is row 1
    lead1 = _marks(a1.get("keep_lead", [""] * len(a1)), "keep_lead", rows)
    lead2 = _marks(other.get("keep_lead", [""] * len(a1)), "keep_lead", rows)
    if any(x is not None and y is not None and x != y for x, y in zip(lead1, lead2)):
        raise ValueError("keep_lead differs between the two files: write the lead's decisions in one file only")
    records = [{"row": r, "pattern": p, "fsq_place_id": pid, "multi": str(m).strip(), "a1": x, "a2": y,
                "lead": l1 if l1 is not None else l2}
               for r, p, pid, m, x, y, l1, l2 in zip(rows, a1["pattern"], a1["fsq_place_id"], a1["multi"],
                                                    _marks(a1["keep_a1"], "keep_a1", rows),
                                                    _marks(other["keep_a2"], "keep_a2", rows), lead1, lead2)]
    return records, ids1 != ids2


def kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Cohen's kappa of two binary mark lists (None when empty; 1.0 when chance agreement is already 1)."""
    if not a:
        return None
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    pa, pb = sum(a) / len(a), sum(b) / len(b)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def summarise(records: Sequence[dict]) -> dict:
    """Counts with the merge's own rules: kept = keep_lead 1, else both annotators 1; a row is open while the lead
    has not decided it and an annotator has not marked it (unmarked) or the two disagree (need_lead)."""
    both = [r for r in records if r["a1"] is not None and r["a2"] is not None]
    kept = [r for r in records if r["lead"] == 1 or (r["lead"] is None and r["a1"] == r["a2"] == 1)]
    return {"candidates": len(records), "marked_a1": sum(r["a1"] is not None for r in records),
            "marked_a2": sum(r["a2"] is not None for r in records), "both_marked": len(both),
            "agreement": (sum(r["a1"] == r["a2"] for r in both) / len(both)) if both else None,
            "kappa": kappa([r["a1"] for r in both], [r["a2"] for r in both]),
            "unmarked": sum(r["lead"] is None and (r["a1"] is None or r["a2"] is None) for r in records),
            "need_lead": [r for r in both if r["a1"] != r["a2"] and r["lead"] is None],
            "lead_decided": sum(r["lead"] is not None for r in records),
            "kept": len(kept), "kept_multi": sum(r["multi"] in ("1", "1.0") for r in kept)}


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def report(s: dict, reordered: bool, target: Sequence[int] = (150, 300)) -> list[str]:
    lines = [f"candidates {s['candidates']}: marked by annotator 1 {s['marked_a1']}, annotator 2 {s['marked_a2']}, "
             f"both {s['both_marked']}; agreement {_num(s['agreement'])}, Cohen's kappa {_num(s['kappa'])}",
             f"not yet marked by both (and not decided by the lead): {s['unmarked']}; decided by the lead: "
             f"{s['lead_decided']}; still for the lead: {len(s['need_lead'])}",
             f"kept so far: {s['kept']} ({s['kept_multi']} multi); the target is {target[0]}-{target[1]} "
             "(config data.trap_target, design §5.3)"]
    lines += [f"  lead: row {r['row']} ({r['pattern']}, {r['fsq_place_id']}): a1 {r['a1']}, a2 {r['a2']}"
              for r in s["need_lead"]]
    if reordered:
        lines.append("warning: annotator 2's file is in another row order than annotator 1's (re-sorted?): the "
                     "merge still matches rows, but keep both in the original order")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="tools/p5_trap_check.py", description=__doc__.splitlines()[0])
    ap.add_argument("a1_csv", type=Path, help="annotator 1's copy (keep_a1, note; the lead's keep_lead)")
    ap.add_argument("a2_csv", type=Path, help="annotator 2's copy (keep_a2, note)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records, reordered = align(args.a1_csv, args.a2_csv)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    s = summarise(records)
    for line in report(s, reordered, load_config()["data"].get("trap_target", (150, 300))):
        print(line)
    ready = not s["unmarked"] and not s["need_lead"]
    print("ready for traps merge" if ready else "not ready: rows are still unmarked or undecided")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
