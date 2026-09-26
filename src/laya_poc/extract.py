"""FSQ OS Places pool extraction with DuckDB (design doc §7.4, corrected per fsq-data.md).

Verified facts this module relies on (release 2026-09-15, DuckDB 1.5.5):
- `read_parquet` on a `dt=` path adds a `dt` column unless `hive_partitioning=false`.
- `facebook_id` is int64 with values above 2**53: cast to VARCHAR here, never let pandas see it.
- Places files are geographic tiles, not country partitions: the smoke pool reads one "best"
  file per country (config `smoke.best_files`, release-specific; see `best_files_by_country`).
  File names change between releases (places_000089, places-00000.zstd, Spark part-*-c000.zstd),
  so a best-file entry is either a number (this release's places_NNNNNN.parquet) or a file name.
- `unresolved_flags` is NULL (not []) when a place has no flags.
- FULL mode scans every places file (~58M rows pass the filters): pass 1 reads only the narrow filter/label
  columns and keeps the cap smallest hash values per (country, label) as a top-N aggregate (8 bytes per kept
  place and thread instead of a 58M-row window), pass 2 reads the wide columns of the places with those
  hashes and applies the same capped ranking. `hash()` is DuckDB-version specific, so the version is
  pinned (`data.duckdb_version`, `check_duckdb_version`).
The HF token only ever goes into `create_hf_secret`, whose SQL is never logged or echoed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import PurePosixPath
from typing import Any, Iterable

import pandas as pd

from . import labels

FALLBACK_COUNTRY = "KR"  # OOD-script fallback (design §5.3), always extracted in full mode
_CAP_COLUMNS = frozenset({"country", "label"})
_TEXT_COLUMNS = ("fsq_place_id", "name", "address", "locality", "region", "postcode", "admin_region",
                 "post_town", "po_box", "country", "tel", "website", "email")
_WIDE_COLUMNS = (", ".join(_TEXT_COLUMNS) + ", CAST(facebook_id AS VARCHAR) AS facebook_id, instagram, twitter, "
                 "fsq_category_ids")
_NARROW_COLUMNS = "fsq_place_id, country"
_SELECTED = "_laya_selected_ids"


def _lit(value: Any) -> str:
    """SQL string literal with single quotes escaped."""
    return "'" + str(value).replace("'", "''") + "'"


def _list_lit(values: Iterable[Any]) -> str:
    return "[" + ", ".join(_lit(v) for v in values) + "]"


def connect(threads: int = 4, memory_limit: str = "8GB", *, httpfs: bool = True,
            temp_directory: str | None = None) -> Any:
    """In-memory DuckDB connection configured for the pool scan (httpfs for hf:// paths). A
    temp_directory lets operators spill to disk instead of failing at memory_limit."""
    import duckdb

    con = duckdb.connect()
    if httpfs:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit={_lit(memory_limit)}")
    if temp_directory:
        con.execute(f"SET temp_directory={_lit(temp_directory)}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")  # keep notebook logs sparse
    check_sql_features(con)
    return con


def check_sql_features(con: Any) -> None:
    """Fail fast if this DuckDB lacks what `pool_sql` relies on (map[key] -> value, x -> lambdas)."""
    import duckdb

    try:
        value, _ = con.execute("SELECT MAP(['a'], ['b'])['a'], list_transform([1], x -> x + 1)").fetchone()
    except duckdb.Error as exc:
        raise RuntimeError(f"duckdb {duckdb.__version__} cannot run the pool query (tested on 1.5.5): {exc}") from exc
    if value != "b":
        raise RuntimeError(f"duckdb {duckdb.__version__}: map[key] returned {value!r} instead of the value; "
                           "install duckdb>=1.2 (tested on 1.5.5)")


def create_hf_secret(con: Any, token: str | None) -> None:
    """Register the HF token as an in-memory DuckDB secret for hf:// reads."""
    if token is None or not str(token).strip():
        raise ValueError("HF token is empty: set HF_TOKEN (the FSQ dataset is gated)")
    sql = f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN {_lit(str(token).strip())})"
    try:
        con.execute(sql)
    except Exception as exc:
        # DuckDB parser errors echo the statement (and so the token): drop the original error.
        raise RuntimeError(f"could not create the DuckDB Hugging Face secret ({type(exc).__name__}); "
                           "is the httpfs extension loaded?") from None


def release_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Places/categories locations for the configured release (`data.hf_base` may be a local dir)."""
    d = cfg["data"]
    base = d["hf_base"].format(rel=d["fsq_release"]).rstrip("/")
    places_dir = f"{base}/places/parquet"
    return {
        "base": base,
        "places_dir": places_dir,
        "places": lambda file_no: f"{places_dir}/places_{int(file_no):06d}.parquet",
        "places_glob": f"{places_dir}/*.parquet",
        "categories_glob": f"{base}/categories/parquet/*.parquet",
    }


def load_categories(con: Any, glob: str) -> pd.DataFrame:
    sql = (f"SELECT category_id, level1_category_name, level2_category_name "
           f"FROM read_parquet({_lit(glob)}, hive_partitioning=false) ORDER BY category_id")
    cats = con.execute(sql).df()
    if cats.empty:
        raise ValueError(f"no categories found at {glob}")
    dup = cats["category_id"][cats["category_id"].duplicated()]
    if len(dup):
        raise ValueError(f"duplicate category ids in {glob}: {sorted(dup)[:5]}")
    return cats


def verify_level1(names: Iterable[str], expected: frozenset[str]) -> None:
    """The categories table must hold exactly the expected level-1 names (10 + known extras)."""
    names = set(names)
    if names != set(expected):
        raise ValueError(f"level-1 category names changed: missing {sorted(set(expected) - names)}, "
                         f"unexpected {sorted(names - set(expected))}; update labels.L1_TO_KEY or "
                         "data.extra_level1_names after checking the release")


def register_category_map(con: Any, cats: pd.DataFrame) -> None:
    """SET VARIABLE cmap = MAP(category_id -> level-1 name), used by `pool_sql`."""
    con.register("_laya_categories", cats[["category_id", "level1_category_name"]])
    try:
        con.execute("SET VARIABLE cmap = (SELECT map(list(category_id), list(level1_category_name)) "
                    "FROM _laya_categories)")
    finally:
        con.unregister("_laya_categories")


def _cap_partition(cap_col: str) -> str:
    parts = [p.strip() for p in str(cap_col).split(",")]
    if not parts or any(p not in _CAP_COLUMNS for p in parts):
        raise ValueError(f"cap_col must be 'country' or 'country, label', got {cap_col!r}")
    return ", ".join(parts)


def _check_query_args(files: list[str], countries: list[str], cap: int | None = None) -> None:
    if not files:
        raise ValueError("pool_sql needs at least one parquet file (files is empty)")
    if not countries:
        raise ValueError("pool_sql needs at least one country")
    if cap is not None and int(cap) < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")


def _labelled_sql(files: list[str], countries: list[str], exclude_flags: list[str], columns: str,
                  extra_where: str = "") -> str:
    """CTEs ending in `labelled`: the filtered places with l1s (sorted distinct level-1 names), label
    (= l1s[1]), n_l1 and first_l1 (level-1 of the first listed known category: the multi-category trap
    gold, §5.3). Shared by the one-query form and both FULL-mode passes so they cannot drift apart."""
    flags = f"AND NOT list_has_any(coalesce(unresolved_flags, []), {_list_lit(exclude_flags)})" if exclude_flags else ""
    return f"""
WITH p AS (
  SELECT {columns},
         list_filter(list_transform(fsq_category_ids, x -> getvariable('cmap')[x]), y -> y IS NOT NULL) AS l1_listed
  FROM read_parquet({_list_lit(files)}, hive_partitioning=false)
  WHERE country IN ({", ".join(_lit(c) for c in countries)})
    AND date_closed IS NULL
    {flags}
    AND len(coalesce(fsq_category_ids, [])) > 0
    AND name IS NOT NULL
    {extra_where}
), q AS (
  SELECT * EXCLUDE (l1_listed), list_sort(list_distinct(l1_listed)) AS l1s, l1_listed[1] AS first_l1 FROM p
), labelled AS (
  SELECT * EXCLUDE (first_l1), l1s[1] AS label, len(l1s) AS n_l1, first_l1 FROM q
  WHERE len(l1s) >= 1
    AND len(list_filter(l1s, y -> NOT list_contains({_list_lit(sorted(labels.L1_TO_KEY))}, y))) = 0
)"""


def _hash_key(seed: int) -> str:
    return f"hash(fsq_place_id || {_lit(int(seed))})"


def _capped_sql(files: list[str], countries: list[str], exclude_flags: list[str], partition: str, cap: int,
                seed: int, extra_where: str = "") -> str:
    return _labelled_sql(files, countries, exclude_flags, _WIDE_COLUMNS, extra_where) + f"""
SELECT * FROM labelled
QUALIFY row_number() OVER (PARTITION BY {partition}
                           ORDER BY {_hash_key(seed)}, fsq_place_id) <= {int(cap)}
ORDER BY fsq_place_id
"""


def pool_sql(files: list[str], countries: list[str], exclude_flags: list[str], cap_col: str, cap: int,
             seed: int) -> str:
    """Labelled, filtered, capped pool query. Needs `register_category_map` on the connection first.
    The per-(country, label) cap (design §7.4.3) makes the pool, and so val/test_id, close to class-balanced
    rather than natural prevalence (§5.3 wording); changing that means a deliberate re-freeze."""
    _check_query_args(files, countries, cap)
    return _capped_sql(files, countries, exclude_flags, _cap_partition(cap_col), cap, seed)


def select_sql(files: list[str], countries: list[str], exclude_flags: list[str], cap: int, seed: int) -> str:
    """FULL pass 1, narrow columns only: per (country, label) the `cap` smallest hash VALUES (min(h, cap)
    keeps an 8-byte heap entry per place; min_by over the id strings needed > 2 GB with 8 threads)."""
    _check_query_args(files, countries, cap)
    return _labelled_sql(files, countries, exclude_flags, _NARROW_COLUMNS + ", fsq_category_ids") + f"""
SELECT unnest(min({_hash_key(seed)}, {int(cap)})) AS h FROM labelled GROUP BY country, label
"""


def selected_pool_sql(files: list[str], countries: list[str], exclude_flags: list[str], table: str, cap: int,
                      seed: int) -> str:
    """FULL pass 2: wide columns of the places whose hash is in `table` (semi-join), capped exactly like
    `pool_sql`. Exact: every place of a group with hash <= its cap-th smallest hash is in the semi-join
    (ties included), so ranking that superset by (hash, id) keeps the same rows as ranking the group."""
    _check_query_args(files, countries, cap)
    return _capped_sql(files, countries, exclude_flags, "country, label", cap, seed,
                       f"AND {_hash_key(seed)} IN (SELECT h FROM {table})")


def two_pass_pool(con: Any, files: list[str], countries: list[str], exclude_flags: list[str], cap: int,
                  seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exactly `pool_sql(files, countries, exclude_flags, "country, label", cap, seed)`, in bounded memory."""
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TEMP TABLE {_SELECTED} AS {select_sql(files, countries, exclude_flags, cap, seed)}")
    selected = int(con.execute(f"SELECT count(*) FROM {_SELECTED}").fetchone()[0])
    t1 = time.time()
    try:
        sql = selected_pool_sql(files, countries, exclude_flags, _SELECTED, cap, seed)
        pool = con.execute(sql).df().reset_index(drop=True)
    finally:
        con.execute(f"DROP TABLE IF EXISTS {_SELECTED}")
    dup = pool["fsq_place_id"][pool["fsq_place_id"].duplicated()]
    if len(dup) or len(pool) != selected:
        raise RuntimeError(f"duplicate fsq_place_id in the release (e.g. {sorted(set(dup))[:3]}; pass 1 kept "
                           f"{selected} ids, pass 2 returned {len(pool)} rows): the split and trap bookkeeping "
                           "assume unique place ids")
    return pool, {"passes": 2, "selected": selected, "pass1_seconds": round(t1 - t0, 1),
                  "pass2_seconds": round(time.time() - t1, 1)}


def full_countries(data_cfg: dict[str, Any]) -> list[str]:
    """ID + OOD-country + OOD-script countries, plus the KR fallback (15 with the default config)."""
    ordered = [*data_cfg["id_countries"], *data_cfg["ood_country"], data_cfg["ood_script"], FALLBACK_COUNTRY]
    return list(dict.fromkeys(ordered))


def _plan(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    d, paths = cfg["data"], release_paths(cfg)
    if not smoke:
        return {"files": [paths["places_glob"]], "countries": full_countries(d), "cap_col": "country, label",
                "cap": int(d["pool_per_country_label"])}
    best = cfg["smoke"]["best_files"]
    countries = list(d["id_countries"])
    missing = [c for c in countries if c not in best]
    if missing:
        raise ValueError(f"smoke.best_files has no places file for {missing}; recompute the map with "
                         "extract.best_files_by_country() for this release")
    files = list(dict.fromkeys(_best_file_path(paths, c, best[c]) for c in countries))
    return {"files": files, "countries": countries, "cap_col": "country", "cap": int(cfg["smoke"]["per_country"])}


def _best_file_path(paths: dict[str, Any], country: str, entry: Any) -> str:
    """smoke.best_files entry -> path: a number is this release's places_NNNNNN.parquet, a string is a
    file name in places/parquet (what `best_files_by_country` returns, whatever the release's naming)."""
    if isinstance(entry, int) and not isinstance(entry, bool) and entry >= 0:
        return paths["places"](entry)
    is_bare_name = isinstance(entry, str) and "/" not in entry and "\\" not in entry
    if is_bare_name and entry.endswith(".parquet"):
        return f"{paths['places_dir']}/{entry}"
    raise ValueError(f"smoke.best_files[{country!r}] = {entry!r}: expected a file number >= 0 or a "
                     "places file name ending in .parquet (no directory)")


def extract_pool(con: Any, cfg: dict[str, Any], *, smoke: bool,
                 categories: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the pool query. Returns (pool sorted by fsq_place_id, info for data_report.json)."""
    d, plan = cfg["data"], _plan(cfg, smoke)
    cats = categories if categories is not None else load_categories(con, release_paths(cfg)["categories_glob"])
    register_category_map(con, cats)
    t0 = time.time()
    args = (plan["files"], plan["countries"], list(d["exclude_flags"]))
    if smoke:
        pool, passes = con.execute(pool_sql(*args, plan["cap_col"], plan["cap"], int(d["split_seed"]))).df(), {}
        pool = pool.reset_index(drop=True)
    else:
        pool, passes = two_pass_pool(con, *args, plan["cap"], int(d["split_seed"]))
    seconds = round(time.time() - t0, 1)
    counts = pool["country"].value_counts()
    rows_by_country = {c: int(counts.get(c, 0)) for c in plan["countries"]}
    empty = [c for c, n in rows_by_country.items() if n == 0]
    if empty:
        raise ValueError(f"no pool rows for {empty} (release {d['fsq_release']}): places files are geographic "
                         "tiles and smoke.best_files is release-specific; recompute it with "
                         "extract.best_files_by_country()")
    hist = pool["n_l1"].value_counts()
    info = {"mode": "smoke" if smoke else "full", "seconds": seconds, "files": plan["files"],
            "countries": plan["countries"], "cap_col": plan["cap_col"], "cap": plan["cap"], "rows": len(pool),
            "rows_by_country": rows_by_country,
            "n_l1_hist": {int(k): int(hist[k]) for k in sorted(hist.index)}, **passes}
    return pool, info


def extract_release(cfg: dict[str, Any], *, smoke: bool, token: str,
                    settings: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], set[str], set[str]]:
    """connect(**settings) -> HF secret -> categories + level-1 check -> pool.
    Returns (pool, info, category ids, level-1 and level-2 category names) for the leakage checks."""
    con = connect(**settings)
    try:
        create_hf_secret(con, token)
        cats = load_categories(con, release_paths(cfg)["categories_glob"])
        verify_level1(set(cats["level1_category_name"]),
                      labels.expected_level1_names(cfg["data"].get("extra_level1_names", ())))
        pool, info = extract_pool(con, cfg, smoke=smoke, categories=cats)
    finally:
        con.close()
    names = set(cats["level1_category_name"]) | set(cats["level2_category_name"].dropna())
    return pool, info, set(cats["category_id"]), names


def check_duckdb_version(expected: str, *, strict: bool = True) -> str | None:
    """hash()-ordered sampling is only reproducible on one DuckDB version: raise (strict, FULL
    extraction) or return a warning (smoke / --pool builds) when the installed one differs."""
    try:
        import duckdb
        found = duckdb.__version__
    except ImportError:
        found = "not installed"
    if found == str(expected):
        return None
    msg = (f"duckdb {found} is installed but data.duckdb_version is {expected}: the hash()-based pool "
           f"sample would differ from the frozen build; pip install duckdb=={expected}")
    if strict:
        raise RuntimeError(msg)
    return msg


def available_ram_bytes() -> int | None:
    """Physical memory available to a new process (Windows, /proc/meminfo, sysconf); None if unknown."""
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page", "avail_page",
                                                        "total_virtual", "avail_virtual", "avail_ext_virtual")]

        status = _MemoryStatus(dwLength=ctypes.sizeof(_MemoryStatus))
        return int(status.avail_phys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def auto_memory_limit(fraction: float = 0.5) -> str | None:
    """DuckDB memory_limit as a fraction of the RAM available now (the rest holds the pandas pool)."""
    ram = available_ram_bytes()
    return None if ram is None else f"{max(256, int(ram * fraction / 1e6))}MB"


def _file_name(path: str) -> str:
    return PurePosixPath(str(path).replace("\\", "/")).name


def best_files_by_country(con: Any, glob: str, countries: list[str]) -> dict[str, str]:
    """For each country, the name of the places file holding the most of its rows (countries with no
    rows are omitted; ties go to the first name). Names, not numbers: file naming changes between
    releases. The result can be pasted into config `smoke.best_files`. Reads only the `country`
    column chunks (~67 s for a full release)."""
    sql = (f"SELECT filename, country, count(*) AS n "
           f"FROM read_parquet({_lit(glob)}, filename=true, hive_partitioning=false) "
           f"WHERE country IN ({', '.join(_lit(c) for c in countries)}) GROUP BY ALL")
    hist = con.execute(sql).df()
    if hist.empty:
        return {}
    hist = hist.assign(file=hist["filename"].map(_file_name))
    ranked = hist.sort_values(["country", "n", "file"], ascending=[True, False, True])
    best = ranked.drop_duplicates("country").set_index("country")["file"]
    return {c: str(best[c]) for c in countries if c in best.index}
