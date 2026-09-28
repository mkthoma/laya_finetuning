"""Trap-set candidates and the two-annotator merge (design doc §5.3 "Trap set construction", §7.4.5).

Candidates are real pool rows whose name misleads (a hospital word on a place FSQ files elsewhere, a
bank word on a bakery, ...) plus places with mixed level-1 categories. build_data removes EVERY
candidate (by fsq_place_id and by dedup key) from the pool before the splits are made, so whatever
the annotators keep is unseen by every run, and E1 need not wait for the annotation (config
data.trap_per_pattern). The annotation is a human task: two annotators fill keep_a1 / keep_a2
(1 = the FSQ level-1 label is plausibly right AND the name misleads), the lead fills keep_lead where
they disagree, then

    python -m laya_poc.traps merge --candidates data/trap_candidates.csv [annotator2.csv] --out data/trap.jsonl

The merge re-runs the unseen assertion against the build's split_*.parquet (--data-dir, default: the CSV's
dir) and refuses a CSV whose kept places or dedup keys are in a split (a CSV from another build). Kept
places outside the evaluation countries (ID + OOD; the extraction also pulls the KR fallback) are dropped
and counted.

Selection is deterministic everywhere: Python `re` (Unicode word boundaries; pandas' pyarrow string
methods would use RE2's ASCII ones), sha256 hash order, no set iteration order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import labels
from .io_utils import write_jsonl
from .normalise import clean
from .rows import assert_no_leakage, make_rows
from .serialise import make_record
from .splits import add_keys

TRAP_CSV = "trap_candidates.csv"
TRAP_JSONL = "trap.jsonl"
TRAP_SPLIT = "trap"
SPLIT_GLOB = "split_*.parquet"
KEY_FIELDS = ("name", "locality", "country")  # the dedup key's inputs (splits.add_keys)
MULTI_PATTERN = "multi_category"
ANNOTATION_COLUMNS = ("keep_a1", "keep_a2", "keep_lead", "note")
DEFAULT_KEEP_FIELDS = ("name", "address", "locality", "region", "postcode", "admin_region", "post_town", "po_box",
                       "country", "tel", "website", "email", "facebook_id", "instagram", "twitter")


@dataclass(frozen=True)
class TrapPattern:
    """A name pattern (§7.4.5) and the level-1 label a matching place must NOT have."""
    name: str
    regex: re.Pattern
    not_label: str


def _words(*words: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)


PATTERNS: tuple[TrapPattern, ...] = (
    TrapPattern("health_word_not_health", _words("hospital", "clinic", "pharmacy", "dental", "surgery"),
                "Health and Medicine"),
    TrapPattern("bank_word", _words("bank"), "Business and Professional Services"),
    TrapPattern("church_school_word", _words("church", "chapel", "school", "college", "abbey"),
                "Community and Government"),
    TrapPattern("museum_theatre_word", _words("museum", "theatre", "theater", "gallery", "cinema"),
                "Arts and Entertainment"),
    TrapPattern("park_garden_word", _words("park", "garden", "beach", "lake"), "Landmarks and Outdoors"),
    TrapPattern("station_hotel_word", _words("station", "hotel", "airport"), "Travel and Transportation"),
)


def pattern_names() -> list[str]:
    return [p.name for p in PATTERNS] + [MULTI_PATTERN]


def candidate_columns(keep_fields: Sequence[str] = DEFAULT_KEEP_FIELDS) -> list[str]:
    return ["pattern", "fsq_place_id", "key", *keep_fields, "label", "label_key", "multi", "n_l1", "l1s",
            *ANNOTATION_COLUMNS]


def hash_order(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rows sorted by sha256(fsq_place_id|seed), then id: deterministic on every OS and library version."""
    keys = [(hashlib.sha256(f"{pid}|{int(seed)}".encode("utf-8")).hexdigest(), str(pid), i)
            for i, pid in enumerate(df["fsq_place_id"])]
    return df.iloc[[i for _, _, i in sorted(keys)]]


def _matches(pat: TrapPattern, names: Sequence[Any], l1: Sequence[Any]) -> list[bool]:
    return [isinstance(n, str) and lab != pat.not_label and pat.regex.search(n) is not None
            for n, lab in zip(names, l1, strict=True)]


def _gold_l1(row: Mapping[str, Any]) -> str:
    """Gold level-1 name: the label, or for a multi-category place the first listed category's level-1."""
    first = row.get("first_l1")
    return first if int(row["n_l1"]) > 1 and isinstance(first, str) and first else row["label"]


def _l1s_text(row: Mapping[str, Any]) -> str:
    l1s = row.get("l1s")
    return "; ".join(str(x) for x in l1s) if pd.api.types.is_list_like(l1s) else str(row["label"])


def _candidate_frame(picked: pd.DataFrame, keep_fields: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in ("name", "locality", "country") if c in picked]
    keys = add_keys(picked[cols])["key"].tolist() if len(picked) else []
    out = []
    for row, key in zip(picked.to_dict("records"), keys, strict=True):
        gold = _gold_l1(row)
        out.append({"pattern": row["pattern"], "fsq_place_id": row["fsq_place_id"], "key": key,
                    **{f: clean(row.get(f)) for f in keep_fields}, "label": gold,
                    "label_key": labels.l1_to_key(gold, "c10"), "multi": int(int(row["n_l1"]) > 1),
                    "n_l1": int(row["n_l1"]), "l1s": _l1s_text(row), **{c: "" for c in ANNOTATION_COLUMNS}})
    return pd.DataFrame(out, columns=candidate_columns(keep_fields))


def trap_candidates(pool: pd.DataFrame, per_pattern: int, seed: int, *,
                    keep_fields: Sequence[str] = DEFAULT_KEEP_FIELDS) -> pd.DataFrame:
    """Up to per_pattern candidates per pattern (hash order). Name patterns look at n_l1 == 1 rows only;
    multi_category takes n_l1 > 1 rows. A place appears under the first pattern it matches."""
    if int(per_pattern) < 0:
        raise ValueError(f"per_pattern must be >= 0, got {per_pattern}")
    single = pool[pool["n_l1"] == 1]
    names, l1 = single["name"].tolist(), single["label"].tolist()
    taken: set[str] = set()
    parts = []
    for pat in PATTERNS:
        hits = single[_matches(pat, names, l1)]
        hits = hits[~hits["fsq_place_id"].isin(taken)]
        picked = hash_order(hits, seed).head(int(per_pattern))
        taken.update(picked["fsq_place_id"])
        parts.append(picked.assign(pattern=pat.name))
    parts.append(hash_order(pool[pool["n_l1"] > 1], seed).head(int(per_pattern)).assign(pattern=MULTI_PATTERN))
    return _candidate_frame(pd.concat(parts, ignore_index=True), keep_fields)


def set_aside(pool: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pool without every candidate place and every row sharing a candidate's dedup key (§5.3)."""
    keys = add_keys(pool[[c for c in ("name", "locality", "country") if c in pool]])["key"]
    by_id = pool["fsq_place_id"].isin(set(candidates["fsq_place_id"]))
    by_key = keys.isin(set(candidates["key"]))
    drop = by_id | by_key
    counts = candidates["pattern"].value_counts()
    info = {"candidates": len(candidates), "by_pattern": {p: int(counts.get(p, 0)) for p in pattern_names()},
            "removed_rows": int(drop.sum()), "removed_by_id": int(by_id.sum()),
            "removed_by_key_only": int((by_key & ~by_id).sum())}
    return pool[~drop].reset_index(drop=True), info


def set_aside_candidates(pool: pd.DataFrame, out: str | Path, *, per_pattern: int, seed: int,
                         keep_fields: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """FULL build step: candidates -> <out>/trap_candidates.csv; returns (pool without them, report, candidates)."""
    candidates = trap_candidates(pool, per_pattern, seed, keep_fields=keep_fields)
    written, warnings = write_candidates(candidates, Path(out) / TRAP_CSV)
    kept, info = set_aside(pool, candidates)
    return kept, {**info, "file": written.name, "annotation": "pending", "warnings": warnings}, candidates


def assert_unseen(splits: Mapping[str, pd.DataFrame], candidates: pd.DataFrame) -> None:
    """No trap candidate (place or dedup key) may reach any split (§5.3 step 5)."""
    ids, keys = set(candidates["fsq_place_id"]), set(candidates["key"])
    for name, df in splits.items():
        hits = int((df["fsq_place_id"].isin(ids) | df["key"].isin(keys)).sum()) if len(df) else 0
        if hits:
            raise AssertionError(f"trap candidate rows reached split {name} ({hits} rows by place or dedup key)")


def read_candidates(path: str | Path) -> pd.DataFrame:
    """Every cell as text ('' when empty): no NA guessing, so a place named "NA" stays a name."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _has_annotations(path: Path) -> bool:
    try:
        df = read_candidates(path)
    except Exception:  # unreadable: treat as precious, never overwrite
        return True
    cols = [c for c in ANNOTATION_COLUMNS if c in df]
    return bool(cols) and bool((df[cols].apply(lambda s: s.str.strip()) != "").any().any())


def write_candidates(candidates: pd.DataFrame, path: str | Path) -> tuple[Path, list[str]]:
    """Write the CSV (UTF-8 with BOM for spreadsheets, '\\n' newlines). An existing file that already
    holds annotations is never overwritten: the new candidates go to <stem>.new.csv instead."""
    path, warnings = Path(path), []
    target = path
    if path.exists() and _has_annotations(path):
        target = path.with_name(f"{path.stem}.new{path.suffix}")
        warnings.append(f"{path.name} already holds annotations: left untouched, new candidates written to "
                        f"{target.name}")
    tmp = target.with_suffix(target.suffix + ".tmp")
    candidates.to_csv(tmp, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(tmp, target)
    return target, warnings


def _flag(value: str, where: str) -> int | None:
    v = str(value).strip()
    if v in ("1", "1.0"):
        return 1
    if v in ("0", "0.0"):
        return 0
    if v == "":
        return None
    raise ValueError(f"{where}: annotation must be 1, 0 or empty, got {value!r}")


def _column(df: pd.DataFrame, col: str) -> list[str]:
    return [str(v).strip() for v in df[col]] if col in df else [""] * len(df)


def _combine(a1: pd.DataFrame, a2: pd.DataFrame) -> pd.DataFrame:
    """One frame from two annotator files: keep_a1 (+note) from the first, keep_a2 from the second."""
    for df, col, who in ((a1, "keep_a1", 1), (a2, "keep_a2", 2)):
        if not any(_column(df, col)):
            raise ValueError(f"annotator {who}'s CSV has no {col} values (annotator {who} fills {col})")
    ids = list(zip(a1["pattern"], a1["fsq_place_id"]))
    other = a2.set_index(["pattern", "fsq_place_id"])
    if len(ids) != len(other) or set(ids) != set(other.index):
        raise ValueError("the two annotator CSVs do not hold the same candidates (pattern, fsq_place_id)")
    other = other.loc[ids]
    lead1, lead2 = _column(a1, "keep_lead"), _column(other, "keep_lead")
    if any(x and y and x != y for x, y in zip(lead1, lead2)):
        raise ValueError("keep_lead differs between the two CSVs; the lead's decision must be in one place")
    notes = [" | ".join(n for n in (x, y) if n) for x, y in zip(_column(a1, "note"), _column(other, "note"))]
    return a1.assign(keep_a2=_column(other, "keep_a2"), keep_lead=[x or y for x, y in zip(lead1, lead2)],
                     note=notes)


def merge_annotations(csv_a1: str | Path, csv_a2: str | Path | None = None) -> pd.DataFrame:
    """Kept candidates (candidate order): keep_lead when set, else keep_a1 == keep_a2 == 1. Raises when
    any row is unannotated or the annotators disagree without a lead decision."""
    df = read_candidates(csv_a1)
    if csv_a2 is not None:
        df = _combine(df, read_candidates(csv_a2))
    missing = [c for c in ("pattern", "fsq_place_id", "multi", "keep_a1", "keep_a2") if c not in df]
    if missing:
        raise ValueError(f"{csv_a1}: missing columns {missing}")
    keep, open_rows = [], {"not annotated": [], "disagree": []}
    rows = zip(df["fsq_place_id"], df["keep_a1"], df["keep_a2"], _column(df, "keep_lead"))
    for i, (pid, a1, a2, lead) in enumerate(rows):
        where = f"row {i + 2} ({pid})"
        a1, a2, lead = _flag(a1, where), _flag(a2, where), _flag(lead, where)
        if lead is None and (a1 is None or a2 is None):
            open_rows["not annotated"].append(pid)
        elif lead is None and a1 != a2:
            open_rows["disagree"].append(pid)
        keep.append(lead == 1 if lead is not None else a1 == a2 == 1)
    problems = [f"{len(v)} rows {k} (e.g. {v[:3]})" for k, v in open_rows.items() if v]
    if problems:
        raise ValueError("annotation incomplete: " + "; ".join(problems) + " - fill keep_a1/keep_a2, and "
                         "keep_lead where the annotators disagree")
    return df[keep].reset_index(drop=True)


def eval_countries(data_cfg: Mapping[str, Any]) -> list[str]:
    """Countries of the evaluation pools (§5.3: the ID pool plus the OOD pools). The FULL extraction also
    pulls the KR script fallback, which no split uses while ood_script is another country."""
    return list(dict.fromkeys([*data_cfg["id_countries"], *data_cfg["ood_country"], data_cfg["ood_script"]]))


def _source_fields(kept: pd.DataFrame, pool: pd.DataFrame | None) -> pd.DataFrame:
    """name/locality/country of each kept candidate (row-aligned) as trap_rows writes them: the pool's
    values when given (spreadsheets edit CSVs), else the CSV's."""
    if pool is None:
        return kept[[c for c in KEY_FIELDS if c in kept]].reset_index(drop=True)
    cols = [c for c in KEY_FIELDS if c in pool]
    source = pool[["fsq_place_id", *cols]].drop_duplicates("fsq_place_id").set_index("fsq_place_id")
    ids = kept["fsq_place_id"].tolist()
    missing = [p for p in ids if p not in source.index]
    if missing:
        raise ValueError(f"{len(missing)} trap candidates are not in the pool (e.g. {missing[:3]}); merge against "
                         "the build's pool.parquet")
    return source.loc[ids, cols].reset_index(drop=True)


def keep_in_scope(kept: pd.DataFrame, countries: Sequence[str], *,
                  pool: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Kept candidates whose country (the pool's when given) is an evaluation country, and the number dropped
    per other country. Candidate selection itself stays as frozen (it decides which rows the splits lose)."""
    country = _source_fields(kept, pool)["country"].astype(str)
    inside = country.isin(set(countries)).to_numpy()
    counts = country[~inside].value_counts()
    return kept[inside].reset_index(drop=True), {str(c): int(counts[c]) for c in sorted(counts.index)}


def kept_identity(kept: pd.DataFrame, pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """fsq_place_id + dedup key of each kept candidate, for assert_unseen: the key of the values trap_rows
    writes and, when present, the CSV's own key column (a place may appear twice, once per key)."""
    ids = kept["fsq_place_id"].tolist()
    fresh = add_keys(_source_fields(kept, pool))["key"].tolist()
    csv_keys = kept["key"].tolist() if "key" in kept else []
    return pd.DataFrame({"fsq_place_id": ids + ids[:len(csv_keys)], "key": fresh + csv_keys})


def load_split_ids(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """{split name: fsq_place_id + dedup key} of every split_*.parquet a build wrote."""
    paths = sorted(Path(data_dir).glob(SPLIT_GLOB))
    if not paths:
        raise FileNotFoundError(f"no {SPLIT_GLOB} in {data_dir}: pass --data-dir <the build's out dir> so the kept "
                                "candidates are checked against every split")
    return {p.stem.removeprefix("split_"): pd.read_parquet(p, columns=["fsq_place_id", "key"]) for p in paths}


def check_unseen(kept: pd.DataFrame, splits: Mapping[str, pd.DataFrame],
                 pool: pd.DataFrame | None = None) -> None:
    """§5.3 step 5 again at merge time: the build sets aside only the candidates it wrote, so an annotated CSV
    from another build (e.g. kept across a rebuild, whose candidates went to *.new.csv) may hold split rows."""
    try:
        assert_unseen(splits, kept_identity(kept, pool))
    except AssertionError as exc:
        raise ValueError(f"{exc}: the CSV's kept candidates were not set aside by the build that wrote these "
                         "splits (rebuilt after the CSV was written?). Annotate that build's own candidates "
                         f"({TRAP_CSV}, or trap_candidates.new.csv when a rebuild kept an annotated file), "
                         "or merge against the build that wrote this CSV") from exc


def trap_rows(kept: pd.DataFrame, ctx: Any, *,
              pool: pd.DataFrame | None = None) -> tuple[list[dict], dict[str, Any]]:
    """§1.1 rows (split "trap", alphabetical keys, true label) plus `multi` and `pattern`. Field values
    and gold come from the pool when given (spreadsheets mangle ids, postcodes and phone numbers)."""
    source = pool.drop_duplicates("fsq_place_id").set_index("fsq_place_id") if pool is not None else None
    rows, rejected, compressed = [], [], 0
    for cand in kept.to_dict("records"):
        pid = cand["fsq_place_id"]
        if source is not None and pid not in source.index:
            raise ValueError(f"trap candidate {pid} is not in the pool; merge against the build's pool.parquet")
        row = {**source.loc[pid].to_dict(), "fsq_place_id": pid} if source is not None else cand
        label_key = labels.l1_to_key(_gold_l1(row), ctx.scheme) if source is not None else cand["label_key"]
        made, stats = make_rows(TRAP_SPLIT, [(make_record(row, ctx.keep_fields), label_key)], scheme=ctx.scheme,
                                smoothing=ctx.smoothing, fits=ctx.fits, compress_order=ctx.compress_order,
                                address_max_chars=ctx.address_max_chars, evidence_fields=ctx.evidence_fields,
                                count_tokens=ctx.count_tokens)
        if not made:
            rejected.append(pid)
            continue
        compressed += stats["compressed"]
        rows.append({**made[0], "id": f"{TRAP_SPLIT}-{len(rows):06d}", "multi": int(cand["multi"]),
                     "pattern": cand["pattern"]})
    assert_no_leakage(rows, ctx.keep_fields, set(ctx.category_ids), set(ctx.category_names))
    by_pattern = {p: sum(r["pattern"] == p for r in rows) for p in pattern_names()}
    return rows, {"written": len(rows), "rejected": len(rejected), "rejected_ids": rejected,
                  "compressed": compressed, "multi": sum(r["multi"] for r in rows), "by_pattern": by_pattern}


# ---- CLI -----------------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m laya_poc.traps", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("merge", help="annotated candidate CSV(s) -> trap JSONL")
    m.add_argument("--candidates", nargs="+", required=True,
                   help="one CSV with keep_a1+keep_a2 (+keep_lead), or annotator 1's then annotator 2's CSV")
    m.add_argument("--out", required=True, help="trap JSONL (e.g. data/trap.jsonl)")
    m.add_argument("--data-dir", help="the build's out dir, whose split_*.parquet every kept candidate is checked "
                                      "against (default: the --pool's dir, else the first CSV's dir)")
    m.add_argument("--pool", help="the build's pool.parquet (default: <data-dir>/pool.parquet, if present)")
    m.add_argument("--config", help="config.yaml (default: <project root>/config.yaml)")
    m.add_argument("--models", nargs="+", default=["laya", "laya_ml"], help="config model keys to budget for")
    m.add_argument("--init-tokenizer", help="checkpoint dir whose tokenizer/ replaces the Hub tokenizers (tests)")
    return p.parse_args(argv)


def row_context(cfg: dict, args: argparse.Namespace, category_ids: set[str]) -> Any:
    """The build's row context (token budgets of every model), so trap rows fit like eval rows."""
    from .build_data import _resolve, load_models, make_context

    models = load_models(cfg, args.models, _resolve(args.init_tokenizer))
    names = labels.expected_level1_names(cfg["data"].get("extra_level1_names", ()))
    return make_context(cfg, models, category_ids, set(names))


def _data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir)
    return Path(args.pool if args.pool else args.candidates[0]).resolve().parent


def _pool_for(args: argparse.Namespace, data_dir: Path) -> Path | None:
    if args.pool:
        return Path(args.pool)
    default = data_dir / "pool.parquet"
    return default if default.exists() else None


def _category_ids(pool: pd.DataFrame | None) -> set[str]:
    if pool is None or "fsq_category_ids" not in pool:
        return set()
    return {str(c) for ids in pool["fsq_category_ids"] if pd.api.types.is_list_like(ids) for c in ids}


def merge_main(args: argparse.Namespace) -> list[str]:
    from .config import load_config

    if len(args.candidates) > 2:
        raise ValueError("--candidates takes one merged CSV or two annotator CSVs")
    cfg = load_config(Path(args.config) if args.config else None)
    kept = merge_annotations(*args.candidates)
    total = len(read_candidates(args.candidates[0]))
    data_dir = _data_dir(args)
    splits = load_split_ids(data_dir)
    pool_path = _pool_for(args, data_dir)
    pool = pd.read_parquet(pool_path) if pool_path else None
    check_unseen(kept, splits, pool)  # before anything is written
    countries = eval_countries(cfg["data"])
    in_scope, dropped = keep_in_scope(kept, countries, pool=pool)
    rows, stats = trap_rows(in_scope, row_context(cfg, args, _category_ids(pool)), pool=pool)
    write_jsonl(args.out, rows)
    source = f"field values from {pool_path.name}" if pool_path else "field values from the CSV (no pool.parquet)"
    lines = [f"traps merge: kept {len(kept)} of {total} candidates, wrote {stats['written']} rows "
             f"({stats['multi']} multi, {stats['rejected']} over the token budget) to {args.out}; {source}",
             "per pattern: " + json.dumps(stats["by_pattern"]),
             f"unseen: no kept candidate (place or dedup key) is in any of the {len(splits)} splits "
             f"({', '.join(splits)}) in {data_dir}"]
    if dropped:
        lines.append(f"dropped {sum(dropped.values())} kept candidates outside the evaluation countries: "
                     f"{json.dumps(dropped)} (design 5.3: ID and OOD pools only, {' '.join(countries)})")
    lo, hi = cfg["data"].get("trap_target", [150, 300])
    if not lo <= stats["written"] <= hi:
        lines.append(f"warning: {stats['written']} trap items is outside data.trap_target [{lo}, {hi}]")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lines = merge_main(args)
    except Exception as exc:  # one-line error; nothing is written on failure
        print(" ".join(f"traps: error: {type(exc).__name__}: {exc}".split()), file=sys.stderr, flush=True)
        return 1
    for line in lines:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
