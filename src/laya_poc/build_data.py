"""Build the data directory: pool -> splits -> eval and per-epoch train JSONL (spec §1.2, §2).

    python -m laya_poc.build_data --out <data_dir> [--smoke] [--config <yaml>] [--pool <pool.parquet>]
                                  [--models laya laya_ml] [--init-tokenizer <abs ckpt dir>]
                                  [--duckdb-memory 4GB] [--duckdb-threads N] [--duckdb-temp <dir>]
                                  [--verify-frozen] [--write-manifest <json>]

Without --pool the labelled pool is extracted from the gated FSQ release with DuckDB, using the
token in $HF_TOKEN (never printed or written). Every state must fit EVERY listed model's exact
state budget, so one data directory serves both checkpoints. Prints a short summary; full detail
goes to <data_dir>/data_report.json.

FULL mode (no --smoke) also sets every trap candidate aside before the splits (trap_candidates.csv),
writes train.jsonl (clean train split, for the baselines), applies the Event headline rule and
fingerprints the JSONL files; --verify-frozen fails unless they match config data.frozen_manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from . import extract, freeze, labels, traps
from .augment import augment_epoch
from .build_report import data_report, headline, summary_lines
from .config import load_config, project_root, split_spec
from .io_utils import sha256_file, write_jsonl, write_sha256sums
from .notice import FSQ_NOTICE_NAME, fsq_notice
from .rows import TRAIN_SPLIT, assert_no_leakage, check_reject_rate, make_rows, records_from_split, stripped_rows
from .serialise import fits_from_counters, state_budget, token_counter
from .splits import SPLIT_ORDER, make_splits_with_info

EVAL_SPLITS = ("val", "test_id")
OOD_SPLITS = ("ood_country", "ood_script", "ood_brand")
STALE_PATTERNS = ("train_e*.jsonl", "train.jsonl", "ood_*.jsonl", "val.jsonl", "test_id.jsonl",
                  "stripped_test.jsonl", "split_*.parquet", "data_report.json", "SHA256SUMS", "build_data_error.log",
                  FSQ_NOTICE_NAME)
MAX_STRIPPED = 1000
# The hf:// scan is I/O-bound: DuckDB threads set how many range requests run at once. The research
# timings (53 s for the 10 smoke files, fsq-data.md) used 8; the connect() default of 4 took ~3x longer.
EXTRACT_THREADS = 8
DUCKDB_RAM_FRACTION = 0.5  # FULL extraction: DuckDB gets half the RAM available at start, pandas the rest
SPILL_SUBDIR = "laya_duckdb_spill"


@dataclass(frozen=True)
class RowContext:
    """Everything make_rows and the leakage/reject checks need, fixed for one build."""
    scheme: str
    smoothing: float
    fits: Callable[[str], bool]
    count_tokens: Callable[[str], int]
    keep_fields: tuple[str, ...]
    evidence_fields: tuple[str, ...]
    compress_order: tuple[str, ...]
    address_max_chars: int
    max_reject_rate: float
    category_ids: frozenset[str]
    category_names: frozenset[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m laya_poc.build_data", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="data directory (relative paths resolve under the project root)")
    p.add_argument("--smoke", action="store_true", help="smoke-test sizes (config section `smoke`)")
    p.add_argument("--config", help="config.yaml (default: <project root>/config.yaml)")
    p.add_argument("--pool", help="use this pool parquet instead of extracting from the FSQ release")
    p.add_argument("--models", nargs="+", default=["laya", "laya_ml"], help="config model keys to budget for")
    p.add_argument("--init-tokenizer", help="checkpoint dir whose tokenizer/ replaces the Hub tokenizers (tests)")
    p.add_argument("--duckdb-memory", help="FULL extraction: DuckDB memory_limit, e.g. 4GB "
                                           f"(default: {int(DUCKDB_RAM_FRACTION * 100)}%% of the RAM available now)")
    p.add_argument("--duckdb-threads", type=int, help="FULL extraction: DuckDB threads (default: max("
                                                      f"{EXTRACT_THREADS}, CPU count); the hf:// scan is I/O-bound)")
    p.add_argument("--duckdb-temp", help=f"FULL extraction: spill under <dir>/{SPILL_SUBDIR} (default: <out>), "
                                         "removed afterwards")
    p.add_argument("--verify-frozen", action="store_true",
                   help="FULL: exit 1 unless the JSONL fingerprint matches config data.frozen_manifest")
    p.add_argument("--write-manifest", help="FULL: write the fingerprint manifest (hashes, counts, versions) here")
    return p.parse_args(argv)


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else project_root() / p


def _pool_category_ids(pool: pd.DataFrame) -> set[str]:
    if "fsq_category_ids" not in pool.columns:
        return set()
    ids: set[str] = set()
    for value in pool["fsq_category_ids"]:
        if pd.api.types.is_list_like(value):
            ids.update(str(v) for v in value)
    return ids


def _extract_pool(cfg: dict, args: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, dict, set[str], set[str]]:
    token = os.environ.get("HF_TOKEN", "")
    if not token.strip():
        raise ValueError("HF_TOKEN is not set (the FSQ dataset is gated); set it or pass --pool")
    if args.smoke:
        return extract.extract_release(cfg, smoke=True, token=token, settings={"threads": EXTRACT_THREADS})
    spill = (_resolve(args.duckdb_temp) if args.duckdb_temp else out) / SPILL_SUBDIR
    settings = {"threads": args.duckdb_threads or max(EXTRACT_THREADS, os.cpu_count() or 1),
                "memory_limit": args.duckdb_memory or extract.auto_memory_limit(DUCKDB_RAM_FRACTION) or "4GB",
                "temp_directory": str(spill)}
    try:
        pool, info, ids, names = extract.extract_release(cfg, smoke=False, token=token, settings=settings)
    finally:
        shutil.rmtree(spill, ignore_errors=True)
    duck = {k: v for k, v in settings.items() if k != "temp_directory"}
    return pool, {**info, "duckdb": duck}, ids, names


def load_pool(args: argparse.Namespace, cfg: dict, out: Path) -> tuple[pd.DataFrame, dict, set[str], set[str]]:
    """(pool, pool report, category ids, category names) and a copy of the pool at out/pool.parquet."""
    target = out / "pool.parquet"
    if args.pool:
        src = _resolve(args.pool)
        pool = pd.read_parquet(src)
        if src.resolve() != target.resolve():
            shutil.copyfile(src, target)
        names = set(labels.expected_level1_names(cfg["data"].get("extra_level1_names", ())))
        return pool, {"source": "file", "path": str(src), "rows": len(pool)}, _pool_category_ids(pool), names
    pool, info, ids, names = _extract_pool(cfg, args, out)
    pool.to_parquet(target, index=False)
    info = {"source": "extract", "path": str(target), "rows": len(pool), **info}
    return pool, info, ids | _pool_category_ids(pool), names


def _check_models(cfg: dict, names: Sequence[str]) -> None:
    unknown = [m for m in names if m not in cfg["model"]]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; expected keys of config `model`: {sorted(cfg['model'])}")


def load_models(cfg: dict, names: Sequence[str], init_tokenizer: Path | None) -> list[dict]:
    """Tokenizer, exact state budget and a cached token counter per model."""
    from . import hub
    from .items import assert_options_untrimmed, internal_question

    scheme, margin = cfg["labels"]["scheme"], int(cfg["serialise"]["token_margin"])
    q = internal_question(labels.QUESTION_NAME, labels.question(scheme)[labels.QUESTION_NAME])
    models = []
    for name in names:
        spec = hub.model_spec(cfg, name)
        tok = hub.load_tokenizer_dir(init_tokenizer) if init_tokenizer else hub.load_tokenizer(spec)
        assert_options_untrimmed(tok, q, spec.head_max_len)
        source = str(init_tokenizer) if init_tokenizer else f"{spec.repo_id}/{spec.subfolder or ''}@{spec.revision}"
        models.append({"name": name, "count": token_counter(tok), "max_len": spec.max_len,
                       "head_max_len": spec.head_max_len, "tokenizer": source,
                       "budget": state_budget(tok, scheme, spec.max_len, spec.head_max_len, margin)})
    return models


def make_context(cfg: dict, models: list[dict], category_ids: set[str], category_names: set[str]) -> RowContext:
    s = cfg["serialise"]
    counters = [m["count"] for m in models]
    return RowContext(
        scheme=cfg["labels"]["scheme"], smoothing=float(cfg["labels"]["smoothing"]),
        fits=fits_from_counters([(m["count"], m["budget"]) for m in models]),
        count_tokens=lambda state: max(c(state) for c in counters),
        keep_fields=tuple(s["keep_fields"]), evidence_fields=tuple(s["evidence_fields"]),
        compress_order=tuple(s["compress_order"]), address_max_chars=int(s["address_max_chars"]),
        max_reject_rate=float(s["max_reject_rate"]),
        category_ids=frozenset(category_ids), category_names=frozenset(category_names))


def emit(ctx: RowContext, out: Path, stem: str, split: str, records: list, **kw: Any) -> tuple[list[dict], dict]:
    """make_rows -> leakage and reject-rate checks -> <out>/<stem>.jsonl (written only if both pass)."""
    rows, stats = make_rows(split, records, scheme=ctx.scheme, smoothing=ctx.smoothing, fits=ctx.fits,
                            compress_order=ctx.compress_order, address_max_chars=ctx.address_max_chars,
                            evidence_fields=ctx.evidence_fields, count_tokens=ctx.count_tokens, **kw)
    assert_no_leakage(rows, ctx.keep_fields, ctx.category_ids, ctx.category_names)
    check_reject_rate(stats, ctx.max_reject_rate)
    write_jsonl(out / f"{stem}.jsonl", rows)
    return rows, stats


def _label_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(r["label"] if r["label"] is not None else "null" for r in rows)
    return dict(sorted(counts.items()))


def write_eval(ctx: RowContext, out: Path, splits: dict[str, pd.DataFrame], seed: int) -> dict[str, dict]:
    """val / test_id / non-empty ood_* with alphabetical keys, plus stripped_test from test_id."""
    results = {}
    for name in (*EVAL_SPLITS, *(n for n in OOD_SPLITS if len(splits[n]))):
        recs = records_from_split(splits[name], ctx.keep_fields, ctx.scheme)
        rows, stats = emit(ctx, out, name, name, recs)
        results[name] = {"stats": stats, "labels": _label_counts(rows), "rows": rows}
    test_rows = results["test_id"]["rows"]
    stripped = stripped_rows(test_rows, min(MAX_STRIPPED, len(test_rows)), seed)
    assert_no_leakage(stripped, ctx.keep_fields, ctx.category_ids, ctx.category_names)
    write_jsonl(out / "stripped_test.jsonl", stripped)
    results["stripped_test"] = {"stats": {"split": "stripped_test", "written": len(stripped), "rejected": 0},
                                "labels": _label_counts(stripped), "rows": stripped}
    return results


def write_train(ctx: RowContext, out: Path, train: pd.DataFrame, aug_cfg: dict, seed: int,
                epochs: int) -> dict[str, dict]:
    """train_e{e}.jsonl: augmentation and field order regenerated from seed*1000+e each epoch."""
    recs = records_from_split(train, ctx.keep_fields, ctx.scheme)
    results = {}
    for epoch in range(epochs):
        augmented = augment_epoch(recs, aug_cfg, ctx.evidence_fields, seed, epoch)
        rng = random.Random(seed * 1000 + epoch)
        rows, stats = emit(ctx, out, f"train_e{epoch}", "train", [(r, lab) for r, lab, _ in augmented],
                           rng=rng, shuffle=True, aug_tags=[tag for _, _, tag in augmented])
        results[f"train_e{epoch}"] = {"stats": stats, "labels": _label_counts(rows)}
    return results


def _clear_stale(out: Path) -> None:
    """Outputs of an earlier build (e.g. more epochs) would otherwise be picked up and checksummed.
    pool.parquet is kept: it may be the --pool input."""
    for pattern in STALE_PATTERNS:
        for path in out.glob(pattern):
            path.unlink()


def leakage_warnings(category_ids: set[str]) -> list[str]:
    """An empty id set makes the category-id substring check pass vacuously: say so in the report."""
    if category_ids:
        return []
    return ["category-id leakage check had no category ids (the --pool parquet has no fsq_category_ids "
            "column): states were NOT checked for category-id substrings"]


def smoke_capacity_warnings(cfg: dict, results: dict[str, dict], card: str = "T4") -> list[str]:
    """The smoke run wants smoke.micro_steps micro-batches from train_e0 alone (smoke.epochs = 1)."""
    mb = cfg["train"]["micro_batch"].get(card)
    first = results.get("train_e0")
    if not mb or first is None:
        return []
    needed = int(cfg["smoke"]["micro_steps"]) * int(mb)
    written = first["stats"]["written"]
    if written >= needed:
        return []
    return [f"train_e0 has {written} rows < smoke.micro_steps x {card} micro_batch = {needed}: "
            "the smoke run would need more than one epoch"]


def write_notice(out: Path, release: str) -> dict[str, str]:
    """<out>/NOTICE_FSQ.txt: the FSQ NOTICE verbatim plus the 'modified' statement (design doc Appendix D)."""
    path = out / FSQ_NOTICE_NAME
    path.write_bytes(fsq_notice(release).encode("utf-8"))  # bytes: no newline translation on Windows
    return {"file": FSQ_NOTICE_NAME, "sha256": sha256_file(path)}


def _save_splits(out: Path, splits: dict[str, pd.DataFrame]) -> None:
    for name in SPLIT_ORDER:
        if name in ("train", *EVAL_SPLITS) or len(splits[name]):
            splits[name].to_parquet(out / f"split_{name}.parquet", index=False)


def write_clean_train(ctx: RowContext, out: Path, train: pd.DataFrame) -> dict[str, Any]:
    """train.jsonl: the train split serialised like an eval split (alphabetical keys, no augmentation,
    true labels). The baselines (B1, B3) train on it; the trainer reads train_e{k}.jsonl only."""
    rows, stats = emit(ctx, out, "train", TRAIN_SPLIT, records_from_split(train, ctx.keep_fields, ctx.scheme))
    return {"stats": stats, "labels": _label_counts(rows)}


class FrozenDataError(RuntimeError):
    """The JSONL fingerprint differs from config data.frozen_manifest (--verify-frozen)."""


def _check_args(cfg: dict, args: argparse.Namespace) -> None:
    _check_models(cfg, args.models)
    if args.smoke and (args.verify_frozen or args.write_manifest):
        raise ValueError("--verify-frozen and --write-manifest apply to FULL builds (drop --smoke)")


def _duckdb_warning(cfg: dict, args: argparse.Namespace) -> str | None:
    """The pin is enforced where DuckDB samples the pool (FULL extraction); elsewhere it only warns."""
    expected = cfg["data"].get("duckdb_version")
    return extract.check_duckdb_version(expected, strict=not (args.smoke or args.pool)) if expected else None


def _set_aside_traps(cfg: dict, pool: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    return traps.set_aside_candidates(pool, out, per_pattern=int(cfg["data"]["trap_per_pattern"]),
                                      seed=int(cfg["data"]["split_seed"]),
                                      keep_fields=tuple(cfg["serialise"]["keep_fields"]))


def _full_fields(cfg: dict, out: Path, results: dict, trap_info: dict, split_info: dict) -> dict[str, Any]:
    names = [f"{k}.jsonl" for k in results]
    return {"traps": trap_info, **headline(cfg, split_info),
            **freeze.full_build_fields(out, names, cfg["data"].get("frozen_manifest"), project_root())}


def _finish_full(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """--verify-frozen first (a mismatching build must not overwrite a manifest), then --write-manifest."""
    check = report["frozen_check"]
    if args.verify_frozen and check["status"] not in ("match", "not_frozen"):
        raise FrozenDataError(freeze.mismatch_message(check))
    if args.write_manifest:
        freeze.write_manifest(_resolve(args.write_manifest), report)


def build(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(_resolve(args.config))
    _check_args(cfg, args)
    out = _resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _clear_stale(out)
    duck_warning = _duckdb_warning(cfg, args)
    pool, pool_info, cat_ids, cat_names = load_pool(args, cfg, out)
    full = not args.smoke
    if full:  # every trap candidate leaves the pool before any split exists (§5.3)
        pool, trap_info, candidates = _set_aside_traps(cfg, pool, out)
    spec = split_spec(cfg, args.smoke)
    splits, brands, split_info = make_splits_with_info(pool, spec)
    if full:
        traps.assert_unseen(splits, candidates)
    _save_splits(out, splits)
    models = load_models(cfg, args.models, _resolve(args.init_tokenizer))
    ctx = make_context(cfg, models, cat_ids, cat_names)
    seed = int(cfg["data"]["split_seed"])
    epochs = int(cfg["smoke"]["epochs"] if args.smoke else cfg["train"]["epochs"])
    results = {**write_eval(ctx, out, splits, seed), **write_train(ctx, out, splits["train"], cfg["augment"],
                                                                     seed, epochs)}
    if full:
        results["train"] = write_clean_train(ctx, out, splits["train"])
    warnings = [*spec.warnings, *leakage_warnings(cat_ids), *([duck_warning] if duck_warning else []),
                *(trap_info["warnings"] if full else smoke_capacity_warnings(cfg, results))]
    report = {**data_report(cfg, args.smoke, pool_info, splits, brands, warnings, models, ctx, results, epochs, out),
              "notice": write_notice(out, cfg["data"]["fsq_release"])}
    if full:
        report.update(_full_fields(cfg, out, results, trap_info, split_info))
    (out / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    write_sha256sums(out)
    if full:
        _finish_full(args, report)
    return report


def _redact(text: str) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    return text.replace(token, "<redacted>") if token else text


def _write_error_log(out: Path | None, exc: BaseException) -> None:
    if out is None or not out.is_dir():
        return
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    (out / "build_data_error.log").write_text(_redact(detail), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build(args)
    except Exception as exc:  # one-line error for the notebook; the traceback goes to a file
        message = " ".join(f"build_data: error: {type(exc).__name__}: {exc}".split())
        print(_redact(message), file=sys.stderr, flush=True)
        _write_error_log(_resolve(args.out), exc)
        return 1
    for line in summary_lines(report):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
