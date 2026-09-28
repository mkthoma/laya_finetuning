"""Phase 7 sample inferences: real records next to what each model predicted, per evaluation split.

Inputs: <data-dir>/<split>.jsonl rows {id, state (compact JSON string), label, has_evidence, ...}; per run (NAME=DIR)
<DIR>/eval/<split>.json (`labels` = the class order of p, `model`) and the gated <DIR>/preds/<split>.jsonl
{id, y, p, answer_confidence, abstained} joined by id (preds/no_gate/<split>.jsonl, when present, gives the ungated
answer shown on no-evidence rows). Every run must cover exactly the data's ids with the same label order.

Per labelled split: each run's top-1 accuracy over answered rows and abstentions, the 2x2 of --focus vs --versus
correctness (with how many "both wrong" rows share the wrong label) and --per-cell examples sampled from each cell.
A no-evidence split (every row has_evidence false, e.g. stripped_test) shows abstention rates and sampled rows.
Sampling is numpy default_rng seeded from (--seed, split name), so a split's sample does not depend on which other
splits are asked for; chosen ids are sorted.

Outputs: --out markdown (phase7_samples_md; FSQ rows: starts with the FSQ NOTICE, LOCAL ONLY; refused when git
would track the path, design doc Appendix D) and --stats-out JSON (metrics only: counts, accuracies, seed; no ids,
names or row content). Both written atomically. Torch-free (phase5_load / matrix_report_md import torch, so the
small readers here are local).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .env_check import run_cli, write_json
from .io_utils import read_jsonl
from .labels import SCHEMES, option_keys
from .notice import FSQ_NOTICE_NAME
from .phase7_samples_md import render_markdown

SCHEMA = "phase7_samples/1"
SPLITS = ("test_id", "ood_country", "ood_script", "ood_brand", "stripped_test")
# (eval JSON model, split) -> note. English `laya` cannot read Thai by design (phase5_load.NO_THAI_MODELS).
NOT_JUDGED = {("laya", "ood_script"): "not judged: English model"}
CELLS = ("both_right", "focus_only_right", "versus_only_right", "both_wrong")


# ---------------------------------------------------------------- data

@dataclass(frozen=True)
class Pred:
    """One run's answer on one row; p (probabilities in label order) is None when the run abstained."""
    y: int | None
    p: tuple[float, ...] | None

    def ranked(self) -> list[int]:
        """Label indices by falling probability (ties: the lower index first, as argmax)."""
        return [] if self.p is None else sorted(range(len(self.p)), key=lambda i: -self.p[i])

    @property
    def top(self) -> int | None:
        return self.ranked()[0] if self.p is not None else None


@dataclass(frozen=True)
class SplitData:
    split: str
    labels: tuple[str, ...]
    rows: tuple[dict, ...]                        # data rows, file order
    y: Mapping[str, int]                          # id -> true label index
    preds: Mapping[str, Mapping[str, Pred]]       # run -> id -> gated prediction
    ungated: Mapping[str, Mapping[str, Pred]]     # run -> id -> preds/no_gate prediction (runs that have one)
    models: Mapping[str, str]                     # run -> eval JSON model

    @property
    def ids(self) -> list[str]:
        return [str(r["id"]) for r in self.rows]

    @property
    def no_evidence(self) -> bool:
        return bool(self.rows) and all(r.get("has_evidence") is False for r in self.rows)


def read_labels(run_dir: Path, split: str) -> tuple[list[str], str]:
    """(labels, model) from <run>/eval/<split>.json; labels must be the scheme's option keys when it names one."""
    path = run_dir / "eval" / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"eval JSON not found: {path}")
    ev = json.loads(path.read_text(encoding="utf-8"))
    labels = ev.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{path}: no `labels` list")
    scheme = ev.get("scheme")
    if scheme in SCHEMES and labels != option_keys(scheme):
        raise ValueError(f"{path}: labels {labels} are not the {scheme} option keys {option_keys(scheme)}")
    return [str(x) for x in labels], str(ev.get("model") or run_dir.name)


def read_preds(path: Path, k: int) -> dict[str, Pred]:
    if not path.is_file():
        raise FileNotFoundError(f"predictions not found: {path}")
    out: dict[str, Pred] = {}
    for r in read_jsonl(path):
        rid = str(r["id"])
        if rid in out:
            raise ValueError(f"{path}: duplicate id {rid}")
        p = None if r.get("abstained") or r.get("p") is None else tuple(float(v) for v in r["p"])
        if p is not None and len(p) != k:
            raise ValueError(f"{path}: row {rid} has {len(p)} probabilities for {k} labels")
        out[rid] = Pred(None if r.get("y") is None else int(r["y"]), p)
    return out


def check_join(who: str, split: str, preds: Mapping[str, Pred], y: Mapping[str, int]) -> None:
    """The run covers exactly the data ids, and its y (when given) is the data label's index."""
    missing, extra = set(y) - set(preds), set(preds) - set(y)
    if missing or extra:
        raise ValueError(f"{who} {split}: prediction ids differ from the data ids ({len(missing)} missing, "
                         f"{len(extra)} extra; e.g. {sorted(missing or extra)[0]})")
    for rid, pr in preds.items():
        if pr.y is not None and pr.y != y[rid]:
            raise ValueError(f"{who} {split}: row {rid} y={pr.y} does not match the data label index {y[rid]}")


def true_indices(split: str, rows: Sequence[dict], labels: Sequence[str]) -> dict[str, int]:
    y: dict[str, int] = {}
    for r in rows:
        rid = str(r["id"])
        if rid in y:
            raise ValueError(f"{split}: duplicate id {rid} in the data")
        if r.get("label") not in labels:
            raise ValueError(f"{split}: row {rid} label {r.get('label')!r} is not one of {list(labels)}")
        y[rid] = list(labels).index(r["label"])
    return y


def load_split(data_dir: Path, runs: Mapping[str, Path], split: str) -> SplitData:
    rows = tuple(read_jsonl(Path(data_dir) / f"{split}.jsonl"))
    first, labels, models, preds, ungated = "", [], {}, {}, {}
    for name, d in runs.items():
        got, models[name] = read_labels(Path(d), split)
        if not first:
            first, labels = name, got
        elif got != labels:
            raise ValueError(f"{split}: {name} label order {got} differs from {first}'s {labels}")
        preds[name] = read_preds(Path(d) / "preds" / f"{split}.jsonl", len(got))
        no_gate = Path(d) / "preds" / "no_gate" / f"{split}.jsonl"
        if no_gate.is_file():
            ungated[name] = read_preds(no_gate, len(got))
    y = true_indices(split, rows, labels)
    for name in runs:
        check_join(name, split, preds[name], y)
        if name in ungated:
            check_join(f"{name} (no_gate)", split, ungated[name], y)
    return SplitData(split, tuple(labels), rows, y, preds, ungated, models)


# ---------------------------------------------------------------- metrics and sampling

def not_judged(sd: SplitData, run: str) -> str | None:
    return NOT_JUDGED.get((sd.models[run], sd.split))


def run_summary(sd: SplitData, run: str) -> dict[str, Any]:
    """Top-1 accuracy over answered rows, abstentions, and whether the split is judged for the run's model."""
    preds = sd.preds[run]
    answered = [rid for rid in sd.ids if preds[rid].p is not None]
    correct = sum(preds[rid].top == sd.y[rid] for rid in answered)
    return {"n": len(sd.rows), "answered": len(answered), "abstained": len(sd.rows) - len(answered),
            "correct": correct, "accuracy": correct / len(answered) if answered else None,
            "judged": not_judged(sd, run) is None}


def ungated_summary(sd: SplitData, run: str) -> dict[str, Any] | None:
    """What the run would have answered without the no-evidence gate (preds/no_gate); None when absent."""
    preds = sd.ungated.get(run)
    answered = [rid for rid in sd.ids if preds[rid].p is not None] if preds else []
    if not answered:
        return None
    correct = sum(preds[rid].top == sd.y[rid] for rid in answered)
    return {"answered": len(answered), "accuracy": correct / len(answered),
            "mean_confidence": float(np.mean([preds[rid].p[preds[rid].top] for rid in answered]))}


@dataclass(frozen=True)
class Agreement:
    """Focus vs versus correctness over the rows both answered; cells hold ids in data order."""
    cells: Mapping[str, tuple[str, ...]]
    same_wrong: int
    either_abstained: int

    def counts(self) -> dict[str, int]:
        cells = {c: len(self.cells[c]) for c in CELLS}
        return {"n": sum(cells.values()) + self.either_abstained, **cells,
                "both_wrong_same_label": self.same_wrong, "either_abstained": self.either_abstained}


def agreement(sd: SplitData, focus: str, versus: str) -> Agreement:
    cells: dict[str, list[str]] = {c: [] for c in CELLS}
    same = skipped = 0
    for rid in sd.ids:
        f, v = sd.preds[focus][rid], sd.preds[versus][rid]
        if f.p is None or v.p is None:
            skipped += 1
            continue
        cell = CELLS[2 * (f.top != sd.y[rid]) + (v.top != sd.y[rid])]
        cells[cell].append(rid)
        same += cell == "both_wrong" and f.top == v.top
    return Agreement({c: tuple(ids) for c, ids in cells.items()}, same, skipped)


def sample_ids(ids: Sequence[str], k: int, rng: np.random.Generator) -> list[str]:
    """k ids drawn without replacement (all of them when there are no more than k), sorted."""
    if k <= 0 or not ids:
        return []
    if len(ids) <= k:
        return sorted(ids)
    return sorted(ids[int(i)] for i in rng.choice(len(ids), size=k, replace=False))


def split_rng(seed: int, split: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(split.encode("utf-8"))])


@dataclass(frozen=True)
class Options:
    focus: str
    versus: str
    per_cell: int
    stripped_rows: int
    seed: int


@dataclass(frozen=True)
class SplitView:
    """One split's data with everything the page and the stats JSON show about it."""
    data: SplitData
    summaries: Mapping[str, dict]                 # run -> run_summary
    ungated: Mapping[str, dict | None]            # run -> ungated_summary
    notes: Mapping[str, str | None]               # run -> NOT_JUDGED note on this split
    agreement: Agreement | None                   # None on a no-evidence split
    samples: Mapping[str, list[str]]              # cell (or "rows" on a no-evidence split) -> sampled ids


def analyse_split(sd: SplitData, opts: Options) -> SplitView:
    rng = split_rng(opts.seed, sd.split)
    runs = list(sd.preds)
    base = ({r: run_summary(sd, r) for r in runs}, {r: ungated_summary(sd, r) for r in runs},
            {r: not_judged(sd, r) for r in runs})
    if sd.no_evidence:
        return SplitView(sd, *base, None, {"rows": sample_ids(sd.ids, opts.stripped_rows, rng)})
    ag = agreement(sd, opts.focus, opts.versus)
    return SplitView(sd, *base, ag, {c: sample_ids(ag.cells[c], opts.per_cell, rng) for c in CELLS})


# ---------------------------------------------------------------- stats JSON (metrics only)

def split_stats(view: SplitView) -> dict[str, Any]:
    sd = view.data
    runs = {r: {**s, **({"ungated": u} if (u := view.ungated.get(r)) else {})} for r, s in view.summaries.items()}
    return {"kind": "no_evidence" if view.agreement is None else "labelled", "n": len(sd.rows), "runs": runs,
            "agreement": view.agreement.counts() if view.agreement else None,
            "sampled": {c: len(ids) for c, ids in view.samples.items()}}


def build_stats(views: Sequence[SplitView], runs: Mapping[str, Path], opts: Options, generated_at: str) -> dict:
    models = views[0].data.models if views else {}
    return {"schema": SCHEMA, "generated_at": generated_at, "seed": opts.seed, "per_cell": opts.per_cell,
            "stripped_rows": opts.stripped_rows, "focus": opts.focus, "versus": opts.versus,
            "labels": list(views[0].data.labels) if views else [],
            "runs": {n: {"dir": Path(d).name, "model": models.get(n)} for n, d in runs.items()},
            "splits": {v.data.split: split_stats(v) for v in views}}


# ---------------------------------------------------------------- files and CLI

def read_notice(path: Path) -> str:
    """The FSQ NOTICE as kept with the data; Appendix D wants the "modified" statement with it."""
    if not path.is_file():
        raise FileNotFoundError(f"FSQ NOTICE not found: {path}")
    text = path.read_text(encoding="utf-8")
    if "modified" not in text:
        raise ValueError(f"{path} has no 'modified' statement (design doc Appendix D)")
    return text


def ensure_git_ignored(path: Path) -> None:
    """The markdown holds FSQ rows: refuse a path git would track (git check-ignore exit 1). Outside a work tree
    (exit 128) or without git, allow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rc = subprocess.run(["git", "check-ignore", "-q", str(path.resolve())], cwd=path.parent,
                            capture_output=True).returncode
    except OSError:
        return
    if rc == 1:
        raise ValueError(f"{path} is not git-ignored; the sample markdown holds FSQ rows and must stay local "
                         "(write it under runs/)")


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def parse_run(spec: str) -> tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=RUN_DIR, got {spec!r}")
    return name.strip(), Path(path.strip())


def check_runs(pairs: Sequence[tuple[str, Path]], focus: str, versus: str) -> dict[str, Path]:
    runs = dict(pairs)
    if len(runs) != len(pairs):
        raise ValueError("run names must be unique")
    for role, name in (("--focus", focus), ("--versus", versus)):
        if name not in runs:
            raise ValueError(f"{role} {name!r} is not one of the --run names {list(runs)}")
    if focus == versus:
        raise ValueError("--focus and --versus must differ")
    return runs


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.phase7_samples", description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", required=True, type=Path, help="directory with <split>.jsonl and NOTICE_FSQ.txt")
    p.add_argument("--run", action="append", required=True, type=parse_run, metavar="NAME=RUN_DIR",
                   help="a run directory (eval/ and preds/); repeatable, table columns follow this order")
    p.add_argument("--focus", required=True, help="run name whose correctness splits the examples (rows)")
    p.add_argument("--versus", required=True, help="run name it is compared with (columns of the 2x2)")
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--per-cell", type=int, default=3, help="examples sampled from each 2x2 cell")
    p.add_argument("--stripped-rows", type=int, default=6, help="rows sampled on a no-evidence split")
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--notice", type=Path, default=None, help=f"default: <data-dir>/{FSQ_NOTICE_NAME}")
    p.add_argument("--out", required=True, type=Path, help="markdown (FSQ rows: LOCAL ONLY, must be git-ignored)")
    p.add_argument("--stats-out", required=True, type=Path, help="metrics-only JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        runs = check_runs(args.run, args.focus, args.versus)
        opts = Options(args.focus, args.versus, args.per_cell, args.stripped_rows, args.seed)
        notice = read_notice(args.notice or args.data_dir / FSQ_NOTICE_NAME)
        ensure_git_ignored(args.out)
        views = [analyse_split(load_split(args.data_dir, runs, s), opts) for s in args.splits]
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_text(args.out, render_markdown(notice, views, runs, opts, generated_at))
        write_json(args.stats_out, build_stats(views, runs, opts, generated_at))
        print(f"phase7_samples: wrote {args.out} (LOCAL ONLY) and {args.stats_out}", flush=True)
        return 0

    return run_cli("phase7_samples", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
