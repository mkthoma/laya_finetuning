"""Data variants of a frozen build for the Phase 3 matrix (design doc §5.14, §6.2; spec P3 §3).

    python -m laya_poc.variants c7 --data-dir D --out O [--models laya laya_ml] [--init-tokenizer <abs dir>]
    python -m laya_poc.variants subset --data-dir D --out O --n N [--seed S] [...]
    python -m laya_poc.variants traps --data-dir D --out O [...]          (all: [--config <yaml>] [--force])

c7: every D/*.jsonl relabelled with labels.C7_MAP: same rows and states, c7 question and gold; a gold uniform in
c10 stays uniform, so stripped_test keeps its TRUE (mapped) label as in c10. The option text changes, so every
state is re-checked against every model's c7 budget. subset: a stratified train subset through build_data's own
augmentation and serialisation, plus a byte copy of D/val.jsonl (training evals). traps: D/trap_candidates.csv
as eval rows (c10 and c7) scored in-session; trap accuracy is later restricted to the annotated keep set via
fsq_place_id. O/variant.json (source fingerprint, parameters, {file: sha256}) is written last, so it marks a
complete variant; outputs are byte-deterministic. O must differ from D; variant files already in O need --force;
a failed build removes what it wrote.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

from . import freeze, labels
from .build_data import (RowContext, _pool_category_ids, _redact, _resolve, load_models, make_context,
                         write_clean_train, write_train)
from .config import load_config, with_overrides
from .io_utils import iter_jsonl, sha256_file, write_jsonl
from .notice import FSQ_NOTICE_NAME
from .rows import STRIPPED_SPLIT, assert_no_leakage, make_rows
from .serialise import make_record
from .traps import TRAP_CSV, hash_order, pattern_names, read_candidates

VARIANT_JSON = "variant.json"
SOURCE_SCHEME, C7 = "c10", "c7"
C7_QUESTIONS = json.dumps(labels.question(C7), ensure_ascii=False)
TRAP_SPLIT = "trap_candidates"
TRAP_FILE, TRAP_C7_FILE = f"{TRAP_SPLIT}.jsonl", f"{TRAP_SPLIT}_c7.jsonl"
ROW_KEYS = ("id", "split", "state", "questions", "gold", "label")
CANDIDATE_COLUMNS = ("pattern", "fsq_place_id", "label_key", "multi")
_UNIFORM_TOL = 1e-9


# ---- pure: c7 relabelling and stratified subsets --------------------------------------------------------

@lru_cache(maxsize=None)
def _question_scheme(questions_json: str) -> str | None:
    q = json.loads(questions_json)
    return next((s for s in labels.SCHEMES if q == labels.question(s)), None)


@lru_cache(maxsize=None)
def _is_uniform(gold_json: str) -> bool:
    probs = list(json.loads(gold_json)[labels.QUESTION_NAME]["probabilities"].values())
    return max(probs) - min(probs) < _UNIFORM_TOL


@lru_cache(maxsize=None)
def _c7_gold(label: str | None, smoothing: float) -> str:
    return json.dumps(labels.gold(label, C7, smoothing), ensure_ascii=False)


def to_c7(row: Mapping[str, Any], smoothing: float) -> dict:
    """A c10 row as a c7 row: state and bookkeeping untouched, label collapsed, c7 question and gold. The gold
    is uniform exactly where the c10 one is: unlabelled rows and stripped_test (which keeps its true label)."""
    missing = [k for k in ROW_KEYS if k not in row]
    if missing:
        raise ValueError(f"{row.get('id', '?')}: not a data row (missing {missing})")
    rid, label, split = row["id"], row["label"], row["split"]
    if _question_scheme(row["questions"]) != SOURCE_SCHEME:
        raise ValueError(f"{rid}: not a c10 row (question differs from labels.question('c10')); c7 is made from "
                         "the c10 build")
    if label is not None and label not in labels.C7_MAP:
        raise ValueError(f"{rid}: label {label!r} is not a c10 key")
    uniform = label is None or split == STRIPPED_SPLIT
    if _is_uniform(row["gold"]) != uniform:
        raise ValueError(f"{rid}: gold {'is not' if uniform else 'is'} uniform for label {label!r} in split "
                         f"{split!r} (expected uniform gold only for unlabelled and {STRIPPED_SPLIT} rows)")
    label7 = labels.C7_MAP[label] if label is not None else None
    return {**row, "questions": C7_QUESTIONS, "gold": _c7_gold(None if uniform else label7, smoothing),
            "label": label7}


def assert_fits(rows: Iterable[Mapping[str, Any]], fits: Callable[[str], bool], name: str,
                budgets: Mapping[str, int]) -> None:
    """States are copied unchanged, but the question (and so the state room) differs: check every row."""
    for row in rows:
        if not fits(row["state"]):
            raise ValueError(f"{name}: {row['id']} state exceeds the c7 state token budget {dict(budgets)}")


def stratified_quotas(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Per-class sizes proportional to counts, summing to n: largest remainder in exact integer arithmetic,
    ties to the class name."""
    total = sum(int(v) for v in counts.values())
    if total <= 0 or not 0 <= int(n) <= total:
        raise ValueError(f"cannot pick {n} of {total} rows")
    classes = sorted(counts)
    base = {c: n * int(counts[c]) // total for c in classes}
    extra = set(sorted(classes, key=lambda c: (-(n * int(counts[c]) % total), c))[:n - sum(base.values())])
    return {c: base[c] + int(c in extra) for c in classes}


def stratified_subset(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n train rows stratified by `label`, taken per class in sha256(fsq_place_id|seed) order and returned in
    split order. Subsets of one seed nest wherever no class quota shrinks as n grows (true for the E6 sizes;
    largest remainder is not monotone in general: a class can lose one row at a few n)."""
    if not 0 < int(n) < len(train):
        raise ValueError(f"--n must be between 1 and {len(train) - 1} (the train split has {len(train)} rows; the "
                         f"full split is the E2 run), got {n}")
    train = train.reset_index(drop=True)
    quotas = stratified_quotas(train["label"].value_counts().to_dict(), int(n))
    keep: list[int] = []
    for label in sorted(quotas):
        keep.extend(hash_order(train[train["label"] == label], seed).head(quotas[label]).index)
    return train.loc[sorted(keep)].reset_index(drop=True)


def write_subset(ctx: RowContext, out: Path, subset: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, dict]:
    """build_data's train outputs for these split rows: train_e{k}.jsonl (augmentation and field order seeded
    by data.split_seed per epoch, as in the build) and the clean train.jsonl."""
    seed, epochs = int(cfg["data"]["split_seed"]), int(cfg["train"]["epochs"])
    results = write_train(ctx, out, subset, cfg["augment"], seed, epochs)
    return {**results, "train": write_clean_train(ctx, out, subset)}


def candidate_sources(cands: pd.DataFrame, data_dir: Path) -> tuple[list[dict], str]:
    """Field values per candidate in CSV order: the build's pool.parquet rows when present (a spreadsheet may
    have re-saved the CSV and mangled ids, postcodes or phone numbers), else the CSV's own cells."""
    pool_path = data_dir / "pool.parquet"
    if not pool_path.is_file():
        return cands.to_dict("records"), TRAP_CSV
    ids = cands["fsq_place_id"].tolist()
    pool = pd.read_parquet(pool_path, filters=[("fsq_place_id", "in", ids)])
    pool = pool.drop_duplicates("fsq_place_id").set_index("fsq_place_id", drop=False)
    missing = [p for p in ids if p not in pool.index]
    if missing:
        raise ValueError(f"{len(missing)} trap candidates are not in {pool_path.name} (e.g. {missing[:3]}): the "
                         f"{TRAP_CSV} comes from another build")
    return [pool.loc[p].to_dict() for p in ids], pool_path.name


def candidate_rows(cands: pd.DataFrame, sources: Sequence[Mapping[str, Any]],
                   ctx: RowContext) -> tuple[list[dict], dict[str, int]]:
    """Eval-style rows (alphabetical keys, the build's budgets and compression, label = label_key) in CSV
    order, plus fsq_place_id / multi / pattern; rows over the budget or without evidence are dropped."""
    keys, patterns = set(labels.option_keys(ctx.scheme)), set(pattern_names())
    kw = {k: getattr(ctx, k) for k in ("scheme", "smoothing", "fits", "compress_order", "address_max_chars",
                                         "evidence_fields", "count_tokens")}
    rows, dropped = [], {"rejected": 0, "no_evidence": 0}
    for cand, src in zip(cands.to_dict("records"), sources, strict=True):
        if cand["label_key"] not in keys or cand["pattern"] not in patterns:
            raise ValueError(f"candidate {cand['fsq_place_id']}: label_key {cand['label_key']!r} or pattern "
                             f"{cand['pattern']!r} unknown ({ctx.scheme} keys, traps.pattern_names())")
        made, _ = make_rows(TRAP_SPLIT, [(make_record(src, ctx.keep_fields), cand["label_key"])], **kw)
        if not made:
            dropped["rejected"] += 1
        elif not made[0]["has_evidence"]:
            dropped["no_evidence"] += 1
        else:
            rows.append({**made[0], "id": f"{TRAP_SPLIT}-{len(rows):06d}", "fsq_place_id": str(cand["fsq_place_id"]),
                         "multi": int(cand["multi"]), "pattern": cand["pattern"]})
    return rows, dropped


def category_ids(data_dir: Path) -> set[str]:
    """Category ids for the leakage check, from the build's pool.parquet (empty without one)."""
    pool = data_dir / "pool.parquet"
    if not pool.is_file() or "fsq_category_ids" not in pq.read_schema(pool).names:
        return set()
    return _pool_category_ids(pd.read_parquet(pool, columns=["fsq_category_ids"]))


def row_context(cfg: dict, data_dir: Path, model_names: Sequence[str], init_tokenizer: str | Path | None,
                scheme: str, *, leakage: bool = True) -> tuple[RowContext, dict[str, int]]:
    """build_data's row context for `scheme` and every model's exact state budget for it."""
    scfg = with_overrides(cfg, {"labels.scheme": scheme})
    models = load_models(scfg, model_names, Path(init_tokenizer) if init_tokenizer else None)
    names = labels.expected_level1_names(cfg["data"].get("extra_level1_names", ()))
    ctx = make_context(scfg, models, category_ids(data_dir) if leakage else set(), set(names))
    return ctx, {m["name"]: int(m["budget"]) for m in models}


def _check_budgets(budgets: Mapping[str, int], report: Mapping[str, Any]) -> None:
    """c10 rows must fit exactly the budgets the source build used, or compression would differ."""
    built = report.get("budgets") or {}
    diff = {n: [b, built[n]] for n, b in budgets.items() if n in built and b != built[n]}
    if diff:
        raise ValueError(f"state budgets differ from the source build's (now, then): {diff}; same tokenizers and "
                         "config serialise.token_margin are needed")


def source_fingerprint(data_dir: Path, report: Mapping[str, Any]) -> dict[str, str]:
    """The build's JSONL fingerprint digest: from data_report.json (FULL builds), else recomputed over D/*.jsonl."""
    if report.get("fingerprint_sha256"):
        return {"fingerprint_sha256": report["fingerprint_sha256"], "from": "data_report.json"}
    names = sorted(p.name for p in data_dir.glob("*.jsonl"))
    return {"fingerprint_sha256": freeze.fingerprint_digest(freeze.fingerprint(data_dir, names)), "from": "recomputed"}


def _params(cfg: Mapping[str, Any], args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    parts = {"labels": cfg["labels"], "serialise": cfg["serialise"], "augment": cfg["augment"],
             "epochs": cfg["train"]["epochs"], "split_seed": cfg["data"]["split_seed"]}
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {**extra, "models": list(args.models), "config_sha256": digest}


def _outputs(out: Path) -> list[Path]:
    """Variant files in O, the manifest first (a half-rebuilt dir must never look complete)."""
    return [p for p in [out / VARIANT_JSON] if p.exists()] + sorted(out.glob("*.jsonl"))


def _begin(args: argparse.Namespace, data_dir: Path, out: Path, report: dict, params: dict) -> dict[str, Any]:
    """The manifest head (variant, source fingerprint, parameters), after clearing O's variant files (--force)."""
    head = {"variant": args.command, "source": source_fingerprint(data_dir, report), "params": dict(params)}
    existing = _outputs(out)
    if existing and not args.force:
        raise FileExistsError(f"{out} already holds {[p.name for p in existing][:6]}; pass --force to delete its "
                              "*.jsonl and variant.json and rebuild (variant.json marks a complete variant)")
    for path in existing:
        path.unlink()
    out.mkdir(parents=True, exist_ok=True)
    return head


@contextmanager
def _building(out: Path) -> Iterator[None]:
    """A failed build removes what it wrote, so no partial variant survives (a rerun needs no --force)."""
    try:
        yield
    except BaseException:
        for path in _outputs(out):
            path.unlink(missing_ok=True)
        raise


def finish(data_dir: Path, out: Path, head: dict, rows: Mapping[str, int], **extra: Any) -> dict[str, Any]:
    """variant.json (written last) with {file: sha256}, their digest and row counts; the FSQ NOTICE is copied."""
    files = freeze.fingerprint(out, rows)
    meta = {**head, **extra, "files": files, "fingerprint_sha256": freeze.fingerprint_digest(files),
            "rows": dict(sorted(rows.items()))}
    if (data_dir / FSQ_NOTICE_NAME).is_file():
        shutil.copyfile(data_dir / FSQ_NOTICE_NAME, out / FSQ_NOTICE_NAME)
    (out / VARIANT_JSON).write_bytes((json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return meta


def _setup(args: argparse.Namespace) -> tuple[dict, Path, Path, dict[str, Any]]:
    cfg = load_config(_resolve(args.config))
    data_dir, out = _resolve(args.data_dir), _resolve(args.out)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir {data_dir} does not exist")
    is_build = (out / "data_report.json").exists()  # swapped arguments must never clear a build (even --force)
    if is_build or out.resolve() == data_dir.resolve() or (out.exists() and out.samefile(data_dir)):
        raise ValueError(f"--out {out} is the --data-dir or another build_data output (data_report.json): a variant "
                         "never writes into a build")
    path = data_dir / "data_report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    scheme = report.get("scheme", SOURCE_SCHEME)
    if cfg["labels"]["scheme"] != SOURCE_SCHEME or scheme != SOURCE_SCHEME:
        raise ValueError(f"variants start from a c10 build (config labels.scheme {cfg['labels']['scheme']!r}, "
                         f"source scheme {scheme!r})")
    unknown, built = [m for m in args.models if m not in cfg["model"]], report.get("budgets") or {}
    if unknown or (built and set(args.models) != set(built)):
        raise ValueError(f"--models {' '.join(args.models)}: the source build budgeted for {sorted(built)} (config "
                         f"models {sorted(cfg['model'])}); every state must fit the same models")
    return cfg, data_dir, out, report


def run_c7(args: argparse.Namespace) -> list[str]:
    cfg, data_dir, out, report = _setup(args)
    names = sorted(p.name for p in data_dir.glob("*.jsonl"))
    if not names:
        raise FileNotFoundError(f"no *.jsonl in {data_dir}")
    head = _begin(args, data_dir, out, report, _params(cfg, args))
    with _building(out):
        ctx, budgets = row_context(cfg, data_dir, args.models, _resolve(args.init_tokenizer), C7, leakage=False)
        counts = {}
        for name in names:
            rows = [to_c7(r, ctx.smoothing) for r in iter_jsonl(data_dir / name)]
            assert_fits(rows, ctx.fits, name, budgets)
            counts[name] = write_jsonl(out / name, rows)
        meta = finish(data_dir, out, head, counts, scheme=C7, schemes={n: C7 for n in names}, budgets=budgets)
    return [f"variants c7: {len(names)} files, {sum(counts.values())} rows relabelled c10 -> c7 (states unchanged; "
            f"all within the c7 state budgets {meta['budgets']}) -> {out}",
            f"variant.json fingerprint {meta['fingerprint_sha256'][:12]} (source "
            f"{head['source']['fingerprint_sha256'][:12]} from {head['source']['from']})"]


def run_subset(args: argparse.Namespace) -> list[str]:
    cfg, data_dir, out, report = _setup(args)
    for name in ("split_train.parquet", "val.jsonl"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"{data_dir / name} is missing (build_data writes it)")
    train = pd.read_parquet(data_dir / "split_train.parquet")
    seed, n = int(cfg["data"]["split_seed"]) if args.seed is None else int(args.seed), int(args.n)
    subset = stratified_subset(train, n, seed)
    params = _params(cfg, args, n=n, seed=seed, aug_seed=int(cfg["data"]["split_seed"]),
                     epochs=int(cfg["train"]["epochs"]))
    head = _begin(args, data_dir, out, report, params)
    with _building(out):
        ctx, budgets = row_context(cfg, data_dir, args.models, _resolve(args.init_tokenizer), SOURCE_SCHEME)
        _check_budgets(budgets, report)
        counts = {f"{k}.jsonl": r["stats"]["written"] for k, r in write_subset(ctx, out, subset, cfg).items()}
        shutil.copyfile(data_dir / "val.jsonl", out / "val.jsonl")
        counts["val.jsonl"] = sum(1 for _ in iter_jsonl(out / "val.jsonl"))
        classes = stratified_quotas(train["label"].value_counts().to_dict(), n)
        meta = finish(data_dir, out, head, counts, scheme=SOURCE_SCHEME, schemes={k: SOURCE_SCHEME for k in counts},
                      budgets=budgets, classes=classes)
    first = counts["train_e0.jsonl"]
    return [f"variants subset: n={n} of {len(train)} train rows (stratified over {len(classes)} classes, seed {seed})"
            f" -> {len(counts) - 2} epoch files ({first} rows in train_e0), train.jsonl, val.jsonl (copied) in {out}",
            f"variant.json fingerprint {meta['fingerprint_sha256'][:12]}"]


def _trap_files(args: argparse.Namespace, cfg: dict, data_dir: Path, out: Path, cands: pd.DataFrame,
                report: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    init = _resolve(args.init_tokenizer)
    ctx, budgets = row_context(cfg, data_dir, args.models, init, SOURCE_SCHEME)
    _check_budgets(budgets, report)
    sources, field_values = candidate_sources(cands, data_dir)
    rows, dropped = candidate_rows(cands, sources, ctx)
    assert_no_leakage(rows, ctx.keep_fields, set(ctx.category_ids), set(ctx.category_names))
    ctx7, budgets7 = row_context(cfg, data_dir, args.models, init, C7, leakage=False)
    rows7 = [to_c7(r, ctx.smoothing) for r in rows]
    assert_fits(rows7, ctx7.fits, TRAP_C7_FILE, budgets7)
    counts = {TRAP_FILE: write_jsonl(out / TRAP_FILE, rows), TRAP_C7_FILE: write_jsonl(out / TRAP_C7_FILE, rows7)}
    return counts, {"schemes": {TRAP_FILE: SOURCE_SCHEME, TRAP_C7_FILE: C7},
                    "budgets": {SOURCE_SCHEME: budgets, C7: budgets7},
                    "field_values": field_values, "candidates": len(cands), "dropped": dropped,
                    "by_pattern": {p: sum(r["pattern"] == p for r in rows) for p in pattern_names()}}


def run_traps(args: argparse.Namespace) -> list[str]:
    cfg, data_dir, out, report = _setup(args)
    csv = data_dir / TRAP_CSV
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} is missing (FULL build_data runs write {TRAP_CSV})")
    cands = read_candidates(csv)
    missing = [c for c in CANDIDATE_COLUMNS if c not in cands]
    if missing or cands.empty:
        raise ValueError(f"{csv.name}: {'missing columns ' + str(missing) if missing else 'no candidates'}")
    head = _begin(args, data_dir, out, report, _params(cfg, args, candidates_sha256=sha256_file(csv)))
    with _building(out):
        counts, extra = _trap_files(args, cfg, data_dir, out, cands, report)
        meta = finish(data_dir, out, head, counts, **extra)
    d = meta["dropped"]
    return [f"variants traps: {len(cands)} candidates -> {counts[TRAP_FILE]} rows ({d['rejected']} over the budget, "
            f"{d['no_evidence']} without evidence dropped; field values from {meta['field_values']}) -> "
            f"{TRAP_FILE} + {TRAP_C7_FILE} in {out}", f"variant.json fingerprint {meta['fingerprint_sha256'][:12]}"]


RUNNERS: dict[str, Callable[[argparse.Namespace], list[str]]] = {"c7": run_c7, "subset": run_subset, "traps": run_traps}
HELP = {"c7": "every *.jsonl of the build relabelled to the 7-class scheme",
        "subset": "stratified train subset through build_data's augmentation + serialisation (E6 learning curve)",
        "traps": "trap_candidates.csv -> trap_candidates.jsonl + trap_candidates_c7.jsonl eval rows"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m laya_poc.variants", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, text in HELP.items():
        s = sub.add_parser(name, help=text, description=text)
        s.add_argument("--data-dir", required=True, help="source build_data out dir (relative: under the project root)")
        s.add_argument("--out", required=True, help="variant dir (must differ from --data-dir)")
        s.add_argument("--config", help="config.yaml (default: <project root>/config.yaml)")
        s.add_argument("--models", nargs="+", default=["laya", "laya_ml"],
                       help="config model keys whose state budgets every row must fit (the source build's)")
        s.add_argument("--init-tokenizer", help="checkpoint dir whose tokenizer/ replaces the Hub tokenizers (tests)")
        s.add_argument("--force", action="store_true", help="rebuild even if --out already holds variant files")
        if name == "subset":
            s.add_argument("--n", type=int, required=True, help="train rows to keep (< the train split size)")
            s.add_argument("--seed", type=int, help="selection seed (default: config data.split_seed); the "
                                                    "augmentation always uses data.split_seed, as the build does")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lines = RUNNERS[args.command](args)
    except Exception as exc:  # one-line error for the notebook
        message = " ".join(f"variants {args.command}: error: {type(exc).__name__}: {exc}".split())
        print(_redact(message), file=sys.stderr, flush=True)
        return 1
    for line in lines:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
