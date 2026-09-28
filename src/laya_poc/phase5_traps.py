"""Trap accuracy on the annotated keep set (design §5.3 "Trap set construction", §5.11 "Traps"; spec P5 §2).

Every run scored all trap CANDIDATES in-session (preds/trap_candidates.jsonl, ids "trap_candidates-NNNNNN", written by
`variants traps` in candidate-CSV order). After the annotation, `traps merge` writes the kept items (data/trap.jsonl);
trap accuracy is the accuracy on the kept candidates, joined on fsq_place_id:
- preds id -> fsq_place_id: the candidate rows JSONL when available (it carries both), else the candidate CSV's row
  order, which is valid only when the build's data_eval/variant.json shows nothing was dropped and the CSV (with its
  annotation columns blanked) re-serialises to the candidates_sha256 that variant.json recorded;
- kept fsq_place_ids: the trap rows' own fsq_place_id; trap rows without one (traps.trap_rows does not write it) are
  joined by position with the kept in-scope candidates of the annotated CSV(s) (traps.merge_annotations +
  keep_in_scope: the same order trap_rows writes), verified row by row on pattern, multi and label.
Without --traps the status is "pending" and each group carries its accuracy on the UNANNOTATED candidates (the eval
JSON's), labelled as such: it is not the §5.11 trap metric.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .io_utils import iter_jsonl
from .phase5_load import SplitPreds, stat5
from .traps import ANNOTATION_COLUMNS, TRAP_CSV, eval_countries, keep_in_scope, merge_annotations, read_candidates

TRAP_SPLIT = "trap_candidates"
CANDIDATE_JSONL = f"{TRAP_SPLIT}.jsonl"
VARIANT_JSON = "variant.json"
PENDING_NOTE = ("annotation pending: accuracy on every UNANNOTATED trap candidate with its automatic label "
                "(not the §5.11 trap metric)")


def blank_sha256(csv_path: Path) -> str:
    """sha256 of the candidate CSV with the annotation columns blanked, serialised as traps.write_candidates does:
    equals variant.json's candidates_sha256 when only the annotation columns were edited."""
    df = read_candidates(csv_path)
    df = df.assign(**{c: "" for c in ANNOTATION_COLUMNS if c in df})
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return hashlib.sha256(buf.getvalue().encode("utf-8-sig")).hexdigest()


def ids_from_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """{preds id: {fsq_place_id, multi}} from the candidate rows JSONL (variants traps output)."""
    out = {}
    for r in iter_jsonl(path):
        if "fsq_place_id" not in r:
            raise ValueError(f"{path}: row {r.get('id')} carries no fsq_place_id (not a variants traps file)")
        out[str(r["id"])] = {"fsq_place_id": str(r["fsq_place_id"]), "multi": int(r.get("multi", 0))}
    return out


def ids_from_csv(csv_path: Path, variant_path: Path) -> dict[str, dict[str, Any]]:
    """{preds id: {fsq_place_id, multi}} from the candidate CSV's row order, checked against variant.json."""
    meta = json.loads(Path(variant_path).read_text(encoding="utf-8"))
    df = read_candidates(csv_path)
    dropped = sum(int(v) for v in (meta.get("dropped") or {}).values())
    want = (meta.get("params") or {}).get("candidates_sha256")
    rows = (meta.get("rows") or {}).get(CANDIDATE_JSONL)
    problems = [f"{dropped} candidates dropped" if dropped else "",
                f"{len(df)} CSV rows vs {rows} candidate rows" if rows != len(df) else "",
                "the CSV (annotation columns blanked) differs from the one the build scored"
                if want != blank_sha256(csv_path) else ""]
    problems = [p for p in problems if p]
    if problems:
        raise ValueError(f"cannot map {TRAP_SPLIT} ids to places by CSV order ({'; '.join(problems)}): pass "
                         f"--trap-candidates <data_eval>/{CANDIDATE_JSONL} from the session that scored them")
    return {f"{TRAP_SPLIT}-{i:06d}": {"fsq_place_id": pid, "multi": int(m)}
            for i, (pid, m) in enumerate(zip(df["fsq_place_id"], df["multi"]))}


def _check_positions(rows: Sequence[dict], kept: pd.DataFrame) -> None:
    if len(rows) != len(kept):
        raise ValueError(f"{len(rows)} trap rows vs {len(kept)} kept in-scope candidates in the annotated CSV")
    for i, (r, c) in enumerate(zip(rows, kept.to_dict("records"))):
        want = (c["pattern"], int(c["multi"]), str(c["label_key"]))
        if (r.get("pattern"), int(r.get("multi", 0)), r.get("label")) != want:
            raise ValueError(f"trap row {i} does not match kept candidate {i} (pattern, multi, label)")


def kept_places(trap_rows: Sequence[dict], csvs: Sequence[Path], data_dir: Path | None,
                cfg: Mapping[str, Any] | None) -> tuple[dict[str, int], str]:
    """({kept fsq_place_id: multi}, how they were joined)."""
    if trap_rows and all("fsq_place_id" in r for r in trap_rows):
        return {str(r["fsq_place_id"]): int(r.get("multi", 0)) for r in trap_rows}, "trap rows' fsq_place_id"
    if not csvs or cfg is None:
        raise ValueError("trap rows carry no fsq_place_id: pass the annotated CSV(s) (--trap-annotations) and the "
                         "config, or add fsq_place_id to traps.trap_rows and re-run traps merge")
    kept = merge_annotations(*csvs[:2])
    pool_path = None if data_dir is None else Path(data_dir) / "pool.parquet"
    pool = pd.read_parquet(pool_path, columns=["fsq_place_id", "name", "locality", "country"]) \
        if pool_path is not None and pool_path.is_file() else None
    in_scope, _ = keep_in_scope(kept, eval_countries(cfg["data"]), pool=pool)
    _check_positions(trap_rows, in_scope)
    return ({str(p): int(m) for p, m in zip(in_scope["fsq_place_id"], in_scope["multi"])},
            "position in the annotated CSV's kept in-scope candidates (verified on pattern, multi, label)")


def run_trap_acc(sp: SplitPreds, id_map: Mapping[str, dict], kept: Mapping[str, int]) -> dict[str, Any]:
    """Accuracy of one run on the kept candidates (answered rows), all / multi=1 / multi=0."""
    pid = [(id_map.get(i) or {}).get("fsq_place_id") for i in sp.ids]
    on = np.array([p in kept for p in pid], dtype=bool)
    multi = np.array([kept.get(p, 0) == 1 for p in pid], dtype=bool)
    correct = sp.yhat == sp.y

    def acc(mask: np.ndarray) -> float | None:
        m = mask & sp.scored
        return float(correct[m].mean()) if m.any() else None

    return {"acc": acc(on), "acc_multi": acc(on & multi), "acc_single": acc(on & ~multi), "n": int(on.sum()),
            "n_multi": int((on & multi).sum()), "n_answered": int((on & sp.scored).sum())}


def group_traps(per_run: Sequence[dict | None]) -> dict[str, Any]:
    first = next((r for r in per_run if r), None)
    return {"status": "ok", "basis": "annotated keep set",
            **{k: stat5([r[k] if r else None for r in per_run]) for k in ("acc", "acc_multi", "acc_single")},
            "n": None if first is None else first["n"], "n_multi": None if first is None else first["n_multi"],
            "n_answered": stat5([r["n_answered"] if r else None for r in per_run])}


def pending_traps(evals: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Pending: the eval JSON's post-T accuracy on the unannotated candidates, per run."""
    accs = [((ev or {}).get("post") or {}).get("acc") for ev in evals]
    return {"status": "pending", "basis": "unannotated candidates", "acc": stat5(accs), "acc_multi": None,
            "acc_single": None, "n": None, "n_multi": None, "n_answered": None}


def find_first(paths: Sequence[Path | None]) -> Path | None:
    return next((p for p in paths if p is not None and Path(p).is_file()), None)


def resolve_id_map(data_dir: Path | None, roots: Sequence[Path], trap_candidates: Path | None,
                   data_eval: Path | None) -> tuple[dict[str, dict], str]:
    """The preds id -> place map: an explicit/discovered candidate JSONL, else the CSV order (see module doc)."""
    evals = [data_eval] if data_eval else [Path(r).parent.parent / "data_eval" for r in roots]
    jsonl = find_first([trap_candidates, *[d / CANDIDATE_JSONL for d in evals],
                        None if data_dir is None else Path(data_dir) / CANDIDATE_JSONL])
    if jsonl is not None:
        return ids_from_jsonl(jsonl), f"ids from {jsonl.name}"
    variant = find_first([d / VARIANT_JSON for d in evals])
    csv = None if data_dir is None else Path(data_dir) / TRAP_CSV
    if variant is None or csv is None or not csv.is_file():
        raise FileNotFoundError(f"no {CANDIDATE_JSONL} and no {VARIANT_JSON} + {TRAP_CSV} to map the scored trap "
                                "candidates to places (pass --trap-candidates or --data-eval, and --data-dir)")
    return ids_from_csv(csv, variant), f"ids from {TRAP_CSV} order (checked against {VARIANT_JSON})"
