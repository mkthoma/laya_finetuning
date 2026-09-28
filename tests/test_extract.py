"""extract.py against small local parquet files that mirror the real FSQ schema (fsq-data.md §3)."""
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from laya_poc import extract as X  # noqa: E402
from laya_poc import labels as L  # noqa: E402
from laya_poc.config import with_overrides  # noqa: E402

REL = "2026-09-15"
EXCLUDE = ["closed", "doesnt_exist", "delete", "duplicate", "inappropriate", "privatevenue"]
BIG_FB = 122132281766001389  # > 2**53

# category_id -> (level1 name, level2 name); ids are 24-char hex like the real table
CATS = {
    "4d4b7105d754a06374d81259": ("Dining and Drinking", None),
    "4bf58dd8d48988d1c4941735": ("Dining and Drinking", "Restaurant"),
    "4d4b7105d754a06378d81259": ("Retail", None),
    "4bf58dd8d48988d118951735": ("Retail", "Grocery Store"),
    "63be6904847c3692a84b9bb9": ("Health and Medicine", None),
    "4d4b7105d754a06376d81259": ("Nightlife Spot", None),
    "4d4b7104d754a06370d81259": ("Arts and Entertainment", None),
    "4d4b7105d754a06373d81259": ("Event", None),
    "4d4b7105d754a06375d81259": ("Business and Professional Services", None),
    "63be6904847c3692a84b9b9a": ("Community and Government", None),
    "4d4b7105d754a06377d81259": ("Landmarks and Outdoors", None),
    "4f4528bc4b90abdf24c9de85": ("Sports and Recreation", None),
    "4d4b7105d754a06379d81259": ("Travel and Transportation", None),
}
DINING, RESTAURANT, RETAIL, GROCERY, HEALTH, NIGHTLIFE = list(CATS)[:6]
UNKNOWN = "ffffffffffffffffffffffff"

STR = pa.string()
LIST = pa.list_(STR)
PLACES_SCHEMA = pa.schema([
    ("fsq_place_id", STR), ("name", STR), ("latitude", pa.float64()), ("longitude", pa.float64()),
    ("address", STR), ("locality", STR), ("region", STR), ("postcode", STR), ("admin_region", STR),
    ("post_town", STR), ("po_box", STR), ("country", STR), ("date_created", STR), ("date_refreshed", STR),
    ("date_closed", STR), ("tel", STR), ("website", STR), ("email", STR), ("facebook_id", pa.int64()),
    ("instagram", STR), ("twitter", STR), ("fsq_category_ids", LIST), ("fsq_category_labels", LIST),
    ("placemaker_url", STR), ("unresolved_flags", LIST), ("geom", pa.binary()),
    ("bbox", pa.struct([("xmin", pa.float64()), ("ymin", pa.float64()), ("xmax", pa.float64()),
                        ("ymax", pa.float64())])),
])


def _place(pid, country, cats, **kw):
    row = {f.name: None for f in PLACES_SCHEMA}
    row.update(fsq_place_id=pid, name=f"Place {pid}", country=country, fsq_category_ids=cats,
               latitude=1.0, longitude=2.0, date_created="2020-01-01", date_refreshed="2025-01-01",
               geom=b"\x01", bbox={"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0})
    row.update(kw)
    return row


def _write_places(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=PLACES_SCHEMA), path)
    return path


def _write_categories(path: Path):
    rows = []
    for cid, (l1, l2) in CATS.items():
        row = {"category_id": cid, "category_level": 2 if l2 else 1, "category_name": l2 or l1,
               "category_label": f"{l1} > {l2}" if l2 else l1, "level1_category_id": cid,
               "level1_category_name": l1}
        for lv in range(2, 7):
            row[f"level{lv}_category_id"] = cid if (lv == 2 and l2) else None
            row[f"level{lv}_category_name"] = l2 if lv == 2 else None
        rows.append(row)
    schema = pa.schema([("category_id", STR), ("category_level", pa.int32()), ("category_name", STR),
                        ("category_label", STR), ("level1_category_id", STR), ("level1_category_name", STR)]
                       + [(f"level{lv}_category_id", STR) for lv in range(2, 7)]
                       + [(f"level{lv}_category_name", STR) for lv in range(2, 7)])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def _filter_rows():
    return [
        _place("ok-dining", "GB", [DINING, RESTAURANT], facebook_id=BIG_FB, website="http://www.a.com"),
        _place("ok-retail", "GB", [GROCERY]),
        _place("mixed", "GB", [RETAIL, HEALTH]),
        _place("unknown-ignored", "GB", [UNKNOWN, HEALTH]),
        _place("only-unknown", "GB", [UNKNOWN]),
        _place("nightlife", "GB", [DINING, NIGHTLIFE]),
        _place("closed", "GB", [DINING], date_closed="2023-04-10"),
        _place("flag-dup", "GB", [DINING], unresolved_flags=["duplicate"]),
        _place("flag-private", "GB", [DINING], unresolved_flags=["privatevenue"]),
        _place("flag-other", "GB", [DINING], unresolved_flags=["some_new_flag"]),
        _place("no-cats", "GB", None),
        _place("empty-cats", "GB", []),
        _place("no-name", "GB", [DINING], name=None),
        _place("wrong-country", "FR", [DINING]),
        _place("null-country", None, [DINING]),
    ]


@pytest.fixture()
def release(tmp_path):
    """Local copy of the release layout (with the dt= segment that triggers hive partitioning)."""
    base = tmp_path / "release" / f"dt={REL}"
    _write_categories(base / "categories" / "parquet" / "categories_000000.parquet")
    _write_places(base / "places" / "parquet" / "places_000089.parquet", _filter_rows())
    return base


@pytest.fixture()
def con(release):
    c = duckdb.connect()
    X.register_category_map(c, X.load_categories(c, _cats_glob(release)))
    yield c
    c.close()


def _cats_glob(base):
    return (base / "categories" / "parquet" / "*.parquet").as_posix()


def _run(con, files, countries=("GB",), cap_col="country", cap=100, seed=1):
    sql = X.pool_sql([Path(f).as_posix() for f in files], list(countries), EXCLUDE, cap_col, cap, seed)
    return con.execute(sql).df()


def test_pool_sql_filters_labels_and_types(con, release):
    df = _run(con, [release / "places" / "parquet" / "places_000089.parquet"])
    got = dict(zip(df["fsq_place_id"], df["label"], strict=True))
    assert got == {"ok-dining": "Dining and Drinking", "ok-retail": "Retail", "mixed": "Health and Medicine",
                   "unknown-ignored": "Health and Medicine", "flag-other": "Dining and Drinking"}
    by_id = df.set_index("fsq_place_id")
    assert by_id.loc["mixed", "n_l1"] == 2 and list(by_id.loc["mixed", "l1s"]) == ["Health and Medicine", "Retail"]
    assert by_id.loc["unknown-ignored", "n_l1"] == 1
    assert by_id.loc["ok-dining", "facebook_id"] == str(BIG_FB)
    assert isinstance(by_id.loc["ok-dining", "facebook_id"], str)
    assert "dt" not in df.columns  # hive_partitioning=false
    for col in ("latitude", "geom", "bbox", "placemaker_url", "date_closed", "unresolved_flags"):
        assert col not in df.columns
    expected = {"fsq_place_id", "name", "address", "locality", "region", "postcode", "admin_region", "post_town",
                "po_box", "country", "tel", "website", "email", "facebook_id", "instagram", "twitter",
                "l1s", "label", "n_l1"}
    assert expected <= set(df.columns)


def _cap_rows(n_per_label=30, countries=("GB", "JP")):
    rows = []
    for c in countries:
        for i in range(n_per_label):
            rows.append(_place(f"{c}-d{i}", c, [DINING]))
            rows.append(_place(f"{c}-r{i}", c, [RETAIL]))
    return rows


def test_pool_sql_caps_per_country_deterministically(con, tmp_path):
    f = _write_places(tmp_path / "cap" / "places_000001.parquet", _cap_rows())
    a = _run(con, [f], countries=("GB", "JP"), cap_col="country", cap=7, seed=5)
    b = _run(con, [f], countries=("GB", "JP"), cap_col="country", cap=7, seed=5)
    c = _run(con, [f], countries=("GB", "JP"), cap_col="country", cap=7, seed=6)
    assert a["country"].value_counts().to_dict() == {"GB": 7, "JP": 7}
    assert list(a["fsq_place_id"]) == list(b["fsq_place_id"])  # same rows, same order
    assert set(a["fsq_place_id"]) != set(c["fsq_place_id"])  # the seed changes the sample


def test_pool_sql_caps_per_country_and_label(con, tmp_path):
    f = _write_places(tmp_path / "cap" / "places_000001.parquet", _cap_rows())
    df = _run(con, [f], countries=("GB", "JP"), cap_col="country, label", cap=4)
    assert df.groupby(["country", "label"]).size().to_dict() == {
        ("GB", "Dining and Drinking"): 4, ("GB", "Retail"): 4, ("JP", "Dining and Drinking"): 4, ("JP", "Retail"): 4}


def test_pool_sql_rejects_unsafe_cap_column():
    with pytest.raises(ValueError, match="cap_col"):
        X.pool_sql(["a.parquet"], ["GB"], EXCLUDE, "country; DROP TABLE x", 5, 1)
    with pytest.raises(ValueError, match="cap"):
        X.pool_sql(["a.parquet"], ["GB"], EXCLUDE, "country", 0, 1)
    with pytest.raises(ValueError, match="files"):
        X.pool_sql([], ["GB"], EXCLUDE, "country", 5, 1)


def test_pool_sql_escapes_literals(con, release):
    df = _run(con, [release / "places" / "parquet" / "places_000089.parquet"], countries=("GB", "O'X"))
    assert len(df) == 5
    assert "'O''X'" in X.pool_sql(["a.parquet"], ["O'X"], ["it's"], "country", 5, 1)


def test_load_categories_and_verify_level1(release):
    c = duckdb.connect()
    cats = X.load_categories(c, _cats_glob(release))
    assert list(cats.columns) == ["category_id", "level1_category_name", "level2_category_name"]
    assert len(cats) == len(CATS)
    names = set(cats["level1_category_name"])
    X.verify_level1(names, L.expected_level1_names(["Nightlife Spot"]))
    with pytest.raises(ValueError, match="Nightlife Spot"):
        X.verify_level1(names, L.expected_level1_names())
    with pytest.raises(ValueError, match="Travel and Transportation"):
        X.verify_level1(names - {"Travel and Transportation"}, L.expected_level1_names(["Nightlife Spot"]))


def test_release_paths(cfg):
    p = X.release_paths(cfg)
    base = f"hf://datasets/foursquare/fsq-os-places/release/dt={cfg['data']['fsq_release']}"
    assert p["places"](89) == f"{base}/places/parquet/places_000089.parquet"
    assert p["places_glob"] == f"{base}/places/parquet/*.parquet"
    assert p["categories_glob"] == f"{base}/categories/parquet/*.parquet"


def _local_cfg(cfg, base, **extra):
    over = {"data.hf_base": str(base.parent / "dt={rel}").replace("\\", "/"), "data.fsq_release": REL}
    over.update(extra)
    return with_overrides(cfg, over)


def _multi_country_release(release):
    pdir = release / "places" / "parquet"
    _write_places(pdir / "places_000097.parquet", _cap_rows(20, countries=("JP",)))
    _write_places(pdir / "places_000084.parquet", _cap_rows(5, countries=("NL", "TH", "KR")) +
                  [_place("gb-in-nl-file", "GB", [RETAIL])])
    return release


def test_extract_pool_smoke_uses_best_files(cfg, release):
    _multi_country_release(release)
    c = duckdb.connect()
    local = _local_cfg(cfg, release, **{"data.id_countries": ["GB", "JP"],
                                        "smoke.best_files": {"GB": 89, "JP": 97}, "smoke.per_country": 10})
    pool, info = X.extract_pool(c, local, smoke=True)
    assert info["rows_by_country"] == {"GB": 5, "JP": 10}
    assert "gb-in-nl-file" not in set(pool["fsq_place_id"])  # only the best files are read
    assert info["n_l1_hist"] == {1: 14, 2: 1}
    assert len(info["files"]) == 2 and info["seconds"] >= 0
    assert list(pool["fsq_place_id"]) == sorted(pool["fsq_place_id"])  # stable order for seeded splits


def test_extract_pool_raises_on_country_with_no_rows(cfg, release):
    c = duckdb.connect()
    local = _local_cfg(cfg, release, **{"data.id_countries": ["GB", "DE"],
                                        "smoke.best_files": {"GB": 89, "DE": 89}})
    with pytest.raises(ValueError, match="DE"):
        X.extract_pool(c, local, smoke=True)


def test_extract_pool_full_reads_all_files_and_fallback_country(cfg, release):
    _multi_country_release(release)
    c = duckdb.connect()
    local = _local_cfg(cfg, release, **{"data.id_countries": ["GB", "JP"], "data.ood_country": ["NL"],
                                        "data.ood_script": "TH", "data.pool_per_country_label": 3})
    pool, info = X.extract_pool(c, local, smoke=False)
    assert set(info["rows_by_country"]) == {"GB", "JP", "NL", "TH", "KR"}
    assert info["rows_by_country"]["JP"] == 6  # 3 per (country, label)
    assert "gb-in-nl-file" in set(pool["fsq_place_id"])


def test_best_files_by_country(tmp_path):
    d = tmp_path / "places"
    _write_places(d / "places_000001.parquet", _cap_rows(3, ("GB",)) + _cap_rows(1, ("JP",)))
    _write_places(d / "places_000002.parquet", _cap_rows(2, ("JP",)))
    _write_places(d / "places_000003.parquet", _cap_rows(1, ("GB", "NL")))
    c = duckdb.connect()
    got = X.best_files_by_country(c, (d / "*.parquet").as_posix(), ["GB", "JP", "NL", "FR"])
    assert got == {"GB": "places_000001.parquet", "JP": "places_000002.parquet", "NL": "places_000003.parquet"}


SPARK_A = "part-00003-3f1c2a9b-1d2e-4c5f-9a8b-7c6d5e4f3a2b-c000.zstd.parquet"
SPARK_B = "part-00007-3f1c2a9b-1d2e-4c5f-9a8b-7c6d5e4f3a2b-c000.zstd.parquet"


def test_best_files_by_country_with_spark_part_names(tmp_path):
    """Release 2025-11-18 names places files part-*-c000.zstd.parquet: no file number to read."""
    d = tmp_path / "places"
    _write_places(d / SPARK_A, _cap_rows(1, ("GB",)) + _cap_rows(3, ("JP",)))
    _write_places(d / SPARK_B, _cap_rows(2, ("GB",)))
    got = X.best_files_by_country(duckdb.connect(), (d / "*.parquet").as_posix(), ["GB", "JP"])
    assert got == {"GB": SPARK_B, "JP": SPARK_A}


def test_extract_pool_smoke_accepts_file_names_in_best_files(cfg, release):
    _write_places(release / "places" / "parquet" / SPARK_B, _cap_rows(20, countries=("JP",)))
    local = _local_cfg(cfg, release, **{"data.id_countries": ["GB", "JP"], "smoke.per_country": 10,
                                        "smoke.best_files": {"GB": 89, "JP": SPARK_B}})
    pool, info = X.extract_pool(duckdb.connect(), local, smoke=True)
    assert info["rows_by_country"] == {"GB": 5, "JP": 10}
    assert info["files"][1].endswith(f"/places/parquet/{SPARK_B}")


@pytest.mark.parametrize("bad", ["sub/places_000001.parquet", "places_000001.csv", -1, True, 1.5])
def test_extract_pool_smoke_rejects_bad_best_file_entries(cfg, release, bad):
    local = _local_cfg(cfg, release, **{"data.id_countries": ["GB"], "smoke.best_files": {"GB": bad}})
    with pytest.raises(ValueError, match="smoke.best_files"):
        X.extract_pool(duckdb.connect(), local, smoke=True)


def test_connect_applies_settings_without_httpfs():
    c = X.connect(threads=2, memory_limit="1GB", httpfs=False)
    threads, mem, order = c.execute("SELECT current_setting('threads'), current_setting('memory_limit'), "
                                    "current_setting('preserve_insertion_order')").fetchone()
    assert threads == 2 and order is False
    assert mem.startswith("953.6")  # 1GB = 10**9 bytes, shown in MiB
    assert c.execute("SELECT current_setting('enable_progress_bar')").fetchone()[0] is False


class _FakeCon:
    def __init__(self, fail=False):
        self.sql, self.fail = [], fail

    def execute(self, sql):
        self.sql.append(sql)
        if self.fail:
            raise RuntimeError(f"Parser Error near: {sql}")
        return self


def test_create_hf_secret_escapes_quotes():
    fake = _FakeCon()
    X.create_hf_secret(fake, "hf_ab'cd")
    assert fake.sql == ["CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN 'hf_ab''cd')"]


def test_create_hf_secret_error_never_contains_token():
    token = "hf_SECRETSECRETSECRETSECRET"
    with pytest.raises(RuntimeError) as exc:
        X.create_hf_secret(_FakeCon(fail=True), token)
    assert token not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


@pytest.mark.parametrize("token", ["", "   ", None])
def test_create_hf_secret_rejects_empty_token(token):
    with pytest.raises(ValueError, match="HF_TOKEN"):
        X.create_hf_secret(_FakeCon(), token)


def test_create_hf_secret_real_duckdb_redacts():
    c = duckdb.connect()
    try:
        c.execute("LOAD httpfs")
    except Exception as exc:  # extension not installed and no network
        pytest.skip(f"httpfs unavailable: {exc}")
    X.create_hf_secret(c, "hf_notARealToken'x")
    rows = c.execute("SELECT name, type, secret_string FROM duckdb_secrets()").fetchall()
    assert rows[0][:2] == ("hf", "huggingface")
    assert "notARealToken" not in rows[0][2]


def test_httpfs_connection_backs_off_on_rate_limits():
    # A 48-core Colab host drew HTTP 429 from the Hub: the scan must retry with backoff, not fail.
    import duckdb
    try:
        con = X.connect(threads=2, memory_limit="1GB", httpfs=True)
    except duckdb.IOException as exc:  # httpfs extension not installable offline
        pytest.skip(f"httpfs unavailable: {exc}")
    get = lambda s: con.execute(f"SELECT current_setting('{s}')").fetchone()[0]
    assert get("http_retries") == X.HTTP_RETRIES >= 5
    assert get("http_retry_wait_ms") == X.HTTP_RETRY_WAIT_MS
    assert float(get("http_retry_backoff")) == X.HTTP_RETRY_BACKOFF
