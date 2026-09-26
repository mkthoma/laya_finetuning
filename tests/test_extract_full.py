"""FULL-mode extraction (Phase 2): two passes that equal the one-query form, the first listed level-1
(multi-category trap gold), the DuckDB version pin and the memory/spill settings. Local parquet only."""
from pathlib import Path

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from laya_poc import extract as X  # noqa: E402
from test_extract import (CATS, DINING, EXCLUDE, GROCERY, HEALTH, RESTAURANT, RETAIL, UNKNOWN, _cap_rows,  # noqa: E402
                          _local_cfg, _multi_country_release, _place, _run, _write_places, con, release)  # noqa: F401

# ---- Phase 2: FULL-mode two-pass extraction, first listed level-1, DuckDB pin and resources -----------------

def _mixed_rows(n=12):
    """Mixed-l1 places (trap candidates) competing for the same (country, label) cap as single-l1 ones."""
    rows = [_place(f"GB-m{i}", "GB", [RETAIL, HEALTH] if i % 2 else [UNKNOWN, HEALTH, DINING]) for i in range(n)]
    return rows + [_place(f"GB-h{i}", "GB", [HEALTH]) for i in range(n)]


def _full_release(release):
    _multi_country_release(release)
    _write_places(release / "places" / "parquet" / "places_000011.parquet",
                  _mixed_rows() + _cap_rows(9, countries=("GB",)))
    return release


def _full_cfg(cfg, release, cap=3):
    return _local_cfg(cfg, release, **{"data.id_countries": ["GB", "JP"], "data.ood_country": ["NL"],
                                       "data.ood_script": "TH", "data.pool_per_country_label": cap})


@pytest.mark.parametrize("cap", [1, 3, 50])
def test_full_two_pass_extraction_equals_the_one_query_form(cfg, release, cap):
    local = _full_cfg(cfg, _full_release(release), cap)
    c = duckdb.connect()
    two, info = X.extract_pool(c, local, smoke=False)
    glob = X.release_paths(local)["places_glob"]
    one = c.execute(X.pool_sql([glob], X.full_countries(local["data"]), EXCLUDE, "country, label", cap,
                               local["data"]["split_seed"])).df()
    pd.testing.assert_frame_equal(two, one)
    assert info["passes"] == 2 and info["selected"] == len(two) > 0
    assert info["pass1_seconds"] >= 0 and info["pass2_seconds"] >= 0
    assert two["n_l1"].max() > 1  # mixed-l1 rows stay in the pool (trap candidates)


def test_pass1_reads_only_the_narrow_columns_and_keeps_only_hash_values():
    sql = X.select_sql(["a.parquet"], ["GB"], EXCLUDE, 5, 1)
    for wide in ("address", "facebook_id", "website", "locality", "tel", "email", "instagram"):
        assert wide not in sql
    # min(h, cap): an 8-byte heap entry per kept place, not an id string (the live build ran out of memory
    # at 2 GB with min_by(id, {hash, id}, 8000) per (country, label) and 8 threads)
    assert "min(" in sql and "min_by(" not in sql and "fsq_category_ids" in sql


@pytest.mark.parametrize("cap", [1, 2, 5])
def test_two_pass_is_exact_under_forced_hash_ties(cfg, release, monkeypatch, cap):
    """Coarse hash (3 values) -> most places tie at the cap boundary; the id tie-break must still match."""
    real = X._hash_key
    monkeypatch.setattr(X, "_hash_key", lambda seed: f"({real(seed)} % 3)")
    local = _full_cfg(cfg, _full_release(release), cap)
    c = duckdb.connect()
    two, info = X.extract_pool(c, local, smoke=False)
    one = c.execute(X.pool_sql([X.release_paths(local)["places_glob"]], X.full_countries(local["data"]), EXCLUDE,
                               "country, label", cap, local["data"]["split_seed"])).df()
    pd.testing.assert_frame_equal(two, one)
    assert info["selected"] == len(two) and two.groupby(["country", "label"]).size().max() == cap


def test_first_l1_is_the_level1_of_the_first_listed_known_category(con, tmp_path):
    f = _write_places(tmp_path / "f" / "places_000001.parquet", [
        _place("a", "GB", [UNKNOWN, RETAIL, HEALTH]), _place("b", "GB", [HEALTH, GROCERY]),
        _place("c", "GB", [RESTAURANT])])
    df = _run(con, [f]).set_index("fsq_place_id")
    assert df.loc["a", "first_l1"] == "Retail" and df.loc["a", "label"] == "Health and Medicine"
    assert df.loc["b", "first_l1"] == "Health and Medicine" and df.loc["c", "first_l1"] == "Dining and Drinking"


def test_full_extraction_rejects_duplicate_place_ids(cfg, release):
    pdir = release / "places" / "parquet"
    _write_places(pdir / "places_000097.parquet", _cap_rows(2, countries=("JP", "NL", "TH", "KR")))
    _write_places(pdir / "places_000098.parquet", [_place("ok-retail", "GB", [GROCERY])])  # same id twice
    with pytest.raises(RuntimeError, match="duplicate fsq_place_id"):
        X.extract_pool(duckdb.connect(), _full_cfg(cfg, release, 50), smoke=False)


def test_check_duckdb_version(monkeypatch):
    assert X.check_duckdb_version(duckdb.__version__) is None
    monkeypatch.setattr(duckdb, "__version__", "1.4.0")
    with pytest.raises(RuntimeError, match=r"duckdb 1\.4\.0.*1\.5\.5.*duckdb==1\.5\.5"):
        X.check_duckdb_version("1.5.5")
    warning = X.check_duckdb_version("1.5.5", strict=False)
    assert "1.4.0" in warning and "1.5.5" in warning


def test_connect_sets_the_spill_directory(tmp_path):
    spill = tmp_path / "spill"
    c = X.connect(threads=1, memory_limit="1GB", httpfs=False, temp_directory=str(spill))
    assert Path(c.execute("SELECT current_setting('temp_directory')").fetchone()[0]) == spill


def test_auto_memory_limit_is_a_fraction_of_available_ram(monkeypatch):
    monkeypatch.setattr(X, "available_ram_bytes", lambda: 8_000_000_000)
    assert X.auto_memory_limit(0.5) == "4000MB"
    monkeypatch.setattr(X, "available_ram_bytes", lambda: None)
    assert X.auto_memory_limit(0.5) is None


def test_available_ram_bytes_on_this_machine():
    ram = X.available_ram_bytes()
    assert ram is None or ram > 100_000_000


def test_extract_release_runs_every_step_with_the_given_settings(cfg, release, monkeypatch):
    seen = {}
    monkeypatch.setattr(X, "connect", lambda **kw: seen.setdefault("kw", kw) and duckdb.connect())
    monkeypatch.setattr(X, "create_hf_secret", lambda con, tok: seen.setdefault("token", tok))
    local = _full_cfg(cfg, _full_release(release))
    pool, info, ids, names = X.extract_release(local, smoke=False, token="hf_x", settings={"threads": 3})
    assert seen == {"kw": {"threads": 3}, "token": "hf_x"}
    assert info["passes"] == 2 and len(pool) == info["rows"]
    assert set(CATS) <= ids and {"Retail", "Grocery Store"} <= names
