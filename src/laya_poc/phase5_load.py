"""Phase 5 inputs (spec P5 §2): every finished run's eval JSONs and per-row predictions, grouped like the matrix
report (matrix_report.group_key: arm, model, scheme, train subset, head-only; seeds pooled).

Both layouts are the same (IMPL_SPEC_P3 §1 / P4 §1): <root>/<run>/eval/<split>.json and preds/<split>.jsonl
{id, y, p, answer_confidence, abstained}. Records come from matrix_results.run_record (the first root holding a run
name wins, as in the matrix report). Predictions become arrays once, in file order; nothing here prints a row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .labels import SCHEMES, option_keys
from .matrix_report import group_key, group_label
from .matrix_results import done_run_dirs, num, run_record

EVAL_DIR, PREDS_DIR = "eval", "preds"
POOLS = ("test_id", "ood_country", "ood_script", "ood_brand")
OOD_POOLS = POOLS[1:]
NO_THAI_MODELS = ("laya",)          # design §5.12 criterion 1: laya cannot read Thai; ood_script excluded
BOOTSTRAP_KINDS = ("majority", "tfidf_lr", "small_encoder", "llm")   # B1 majority, B3, B4, B5 (spec P5 §2)


# ---------------------------------------------------------------- statistics

def stat5(values: Sequence[Any]) -> dict[str, Any] | None:
    """{mean, min, max, range, n, values} over the finite values (values keeps run order, None for a gap);
    None when no value is finite."""
    vals = [num(v) for v in values]
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "range": max(xs) - min(xs), "n": len(xs),
            "values": vals}


def ood_pools(model: Any) -> tuple[str, ...]:
    """The OOD pools of criterion 1 for a model: `laya` excludes ood_script (it cannot read Thai by design)."""
    return tuple(p for p in OOD_POOLS if not (p == "ood_script" and model in NO_THAI_MODELS))


# ---------------------------------------------------------------- predictions

@dataclass(frozen=True)
class SplitPreds:
    """One run's predictions on one split, in file order. y = -1 for an unlabelled row; P and conf are NaN on
    abstained (no-evidence gate) rows."""
    ids: tuple[str, ...]
    y: np.ndarray
    P: np.ndarray
    conf: np.ndarray
    abstained: np.ndarray

    @property
    def scored(self) -> np.ndarray:
        """Answered AND labelled: the rows evaluate scores."""
        return ~self.abstained & (self.y >= 0)

    @property
    def yhat(self) -> np.ndarray:
        """argmax of p (the renormalised post-T probabilities, as evaluate) on answered rows, -1 elsewhere."""
        out = np.full(len(self.ids), -1, dtype=int)
        answered = ~self.abstained
        if answered.any():
            out[answered] = self.P[answered].argmax(1)
        return out


def read_preds(path: Path, k: int) -> SplitPreds:
    """preds/<split>.jsonl -> arrays. A row whose p has another width than the eval JSON's labels is an error."""
    ids, ys, ps, confs, abst = [], [], [], [], []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            gated = bool(r.get("abstained")) or r.get("p") is None
            p = [np.nan] * k if gated else [float(v) for v in r["p"]]
            if len(p) != k:
                raise ValueError(f"{path}: row {r.get('id')} has {len(p)} probabilities for {k} labels")
            ids.append(str(r["id"]))
            ys.append(-1 if r.get("y") is None else int(r["y"]))
            ps.append(p)
            confs.append(np.nan if gated else float(r["answer_confidence"]))
            abst.append(gated)
    return SplitPreds(tuple(ids), np.array(ys, dtype=int), np.array(ps, dtype=float).reshape(len(ids), k),
                      np.array(confs, dtype=float), np.array(abst, dtype=bool))


@dataclass(frozen=True)
class RunData:
    """A finished run: its matrix_results record, eval JSONs {split: payload} and predictions {split: arrays}."""
    record: dict
    evals: Mapping[str, dict]
    preds: Mapping[str, SplitPreds] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.record["run_name"])

    def labels(self, split: str) -> list[str] | None:
        ev = self.evals.get(split) or {}
        return list(ev["labels"]) if isinstance(ev.get("labels"), list) else None


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_labels(run: str, split: str, got: list, scheme: Any) -> None:
    """An eval JSON's labels are its scheme's option keys in option order (the preds' p and y index them)."""
    if scheme in SCHEMES and got != option_keys(scheme):
        raise ValueError(f"{run} {split}: labels {got} are not the {scheme} option keys {option_keys(scheme)}")


def load_run(run_dir: Path) -> RunData:
    """Record, every eval/<split>.json and the preds of each split whose eval JSON names its labels."""
    rec = run_record(run_dir)
    evals = {p.stem: ev for p in sorted((run_dir / EVAL_DIR).glob("*.json")) if (ev := _read_json(p)) is not None}
    preds = {}
    for split, ev in evals.items():
        path = run_dir / PREDS_DIR / f"{split}.jsonl"
        if isinstance(ev.get("labels"), list) and path.is_file():
            check_labels(str(rec["run_name"]), split, ev["labels"], rec.get("scheme"))
            preds[split] = read_preds(path, len(ev["labels"]))
    return RunData(rec, evals, preds)


def load_runs(roots: Iterable[Path]) -> list[RunData]:
    """Every done run over the roots (sorted by name; a name in several roots: the first root's)."""
    return [load_run(rd) for rd in done_run_dirs(list(roots))]


# ---------------------------------------------------------------- groups

def group_id(key: tuple) -> str:
    """Stable group id: "<arm> <model> <scheme>" + " n<subset>" + " head" (the report label adds notes)."""
    arm, model, scheme, subset, head = key
    return " ".join([str(arm), str(model), str(scheme), *([f"n{subset}"] if subset else []),
                     *(["head"] if head else [])])


def role(rec: Mapping[str, Any], candidates: Sequence[str], report_also: Sequence[str]) -> str:
    """candidate / report_also (full-size, fully fine-tuned Laya arms named in config phase5), baseline (B*),
    other (head-only, train subsets, ...)."""
    laya = (rec.get("kind") or "laya") == "laya"
    full = laya and not (rec.get("zero_shot") or rec.get("head_only") or rec.get("subset"))
    if full and rec.get("arm") in candidates:
        return "candidate"
    if full and rec.get("arm") in report_also:
        return "report_also"
    return "baseline" if str(rec.get("arm", "")).startswith("B") else "other"


@dataclass(frozen=True)
class Group:
    id: str
    key: tuple
    runs: tuple[RunData, ...]
    role: str

    @property
    def rec(self) -> dict:
        return self.runs[0].record

    def identity(self) -> dict[str, Any]:
        arm, model, scheme, subset, head = self.key
        kind = str(self.rec.get("kind") or "laya")
        return {"id": self.id, "label": group_label(self.key, kind), "arm": arm, "model": model, "scheme": scheme,
                "subset": subset, "head_only": head, "kind": kind, "zero_shot": bool(self.rec.get("zero_shot")),
                "eval_subset": any(r.record.get("eval_subset") for r in self.runs), "role": self.role,
                "runs": [r.name for r in self.runs], "seeds": [r.record.get("seed") for r in self.runs],
                "n_runs": len(self.runs), "ood_pools": list(ood_pools(model))}


def make_groups(runs: Sequence[RunData], candidates: Sequence[str] = (),
                report_also: Sequence[str] = ()) -> dict[str, Group]:
    """{group id: Group} in matrix-report order; runs within a group sorted by name."""
    by_key: dict[tuple, list[RunData]] = {}
    for r in sorted(runs, key=lambda r: r.name):
        by_key.setdefault(group_key(r.record), []).append(r)
    order = sorted(by_key, key=lambda k: (str(k[0]), str(k[1]), str(k[2]), k[3] or 0, k[4]))
    return {group_id(k): Group(group_id(k), k, tuple(by_key[k]), role(by_key[k][0].record, candidates, report_also))
            for k in order}
