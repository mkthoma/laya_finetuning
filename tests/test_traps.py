"""traps.py: name-pattern trap candidates (design §5.3, §7.4.5) and the two-annotator merge.

Annotation itself is a human task: the merge is tested on synthetic CSVs only.
"""
import json

import pandas as pd
import pytest

from laya_poc import labels as L
from laya_poc import traps as T
from laya_poc.build_data import RowContext
from laya_poc.splits import add_keys

KEEP = ("name", "address", "locality", "region", "postcode", "admin_region", "post_town", "po_box", "country",
        "tel", "website", "email", "facebook_id", "instagram", "twitter")
DINING, HEALTH, RETAIL, TRAVEL = ("Dining and Drinking", "Health and Medicine", "Retail",
                                  "Travel and Transportation")
PATTERNS = ["health_word_not_health", "bank_word", "church_school_word", "museum_theatre_word",
            "park_garden_word", "station_hotel_word", "multi_category"]


def _row(pid, name, label, n_l1=1, country="GB", **kw):
    row = {f: None for f in KEEP}
    row.update(fsq_place_id=pid, name=name, locality="Leeds", country=country, label=label, n_l1=n_l1,
               l1s=[label], fsq_category_ids=["4d4b7105d754a06374d81259"])
    row.update(kw)
    return row


def _pool(rows):
    return pd.DataFrame(rows)


def _basic_pool():
    return _pool([
        _row("h1", "St Mary's HOSPITAL Cafe", DINING),          # health word, not health
        _row("h2", "City Hospital", HEALTH),                     # health word, health label: not a trap
        _row("h3", "Hospitality House", DINING),                 # no word boundary
        _row("b1", "Kings Bank Bakery", DINING),
        _row("p1", "Central Park Hotel", DINING),                # park AND hotel: first pattern wins
        _row("p2", "Parking Lot 4", TRAVEL),                     # "parking" is not "park"
        _row("s1", "Grand Station", TRAVEL),                     # travel label: not a trap
        _row("m1", "Bank Pharmacy", RETAIL, n_l1=2, l1s=[HEALTH, RETAIL], first_l1=RETAIL),
        _row("m2", "Corner Shop", HEALTH, n_l1=2, l1s=[HEALTH, RETAIL], first_l1=HEALTH),
    ])


def test_patterns_follow_the_design_queries_in_order():
    assert [p.name for p in T.PATTERNS] + [T.MULTI_PATTERN] == PATTERNS
    assert T.pattern_names() == PATTERNS


def test_candidates_match_case_insensitive_whole_words_with_label_conditions():
    got = T.trap_candidates(_basic_pool(), per_pattern=10, seed=1)
    assert dict(zip(got["fsq_place_id"], got["pattern"])) == {
        "h1": "health_word_not_health", "b1": "bank_word", "p1": "park_garden_word",
        "m1": "multi_category", "m2": "multi_category"}
    assert got["fsq_place_id"].is_unique  # a place is a candidate under one pattern only


def test_candidate_columns_values_and_gold():
    raw = T.trap_candidates(_basic_pool().assign(postcode="08302", facebook_id="122132281766001389"),
                            per_pattern=10, seed=1)
    assert list(raw.columns) == ["pattern", "fsq_place_id", "key", *KEEP, "label", "label_key",
                                 "multi", "n_l1", "l1s", "keep_a1", "keep_a2", "keep_lead", "note"]
    got = raw.set_index("fsq_place_id")
    assert got.loc["h1", "label"] == DINING and got.loc["h1", "label_key"] == "dining"
    assert got.loc["h1", "multi"] == 0 and got.loc["h1", "n_l1"] == 1
    # multi-category gold: level-1 of the FIRST listed category, flagged multi=1
    assert got.loc["m1", "label"] == RETAIL and got.loc["m1", "label_key"] == "retail" and got.loc["m1", "multi"] == 1
    assert got.loc["m2", "label_key"] == "health" and got.loc["m1", "l1s"] == f"{HEALTH}; {RETAIL}"
    assert got.loc["h1", "key"] == add_keys(pd.DataFrame([{"name": "St Mary's HOSPITAL Cafe", "locality": "Leeds",
                                                           "country": "GB"}]))["key"][0]
    assert got.loc["h1", "facebook_id"] == "122132281766001389" and got.loc["h1", "postcode"] == "08302"
    assert (got[["keep_a1", "keep_a2", "keep_lead", "note"]] == "").all().all()


def test_multi_gold_falls_back_to_the_pool_label_without_first_l1():
    got = T.trap_candidates(_basic_pool().drop(columns=["first_l1"]), per_pattern=10, seed=1)
    m1 = got.set_index("fsq_place_id").loc["m1"]
    assert m1["label"] == RETAIL and m1["multi"] == 1  # pool label (alphabetical first l1) is Retail here


def _many(n=12):
    return _pool([_row(f"x{i:02d}", f"Blue Museum {i}", DINING) for i in range(n)])


def test_per_pattern_cap_is_hash_ordered_and_independent_of_input_order():
    a = T.trap_candidates(_many(), per_pattern=4, seed=7)
    b = T.trap_candidates(_many().iloc[::-1].reset_index(drop=True), per_pattern=4, seed=7)
    c = T.trap_candidates(_many(), per_pattern=4, seed=8)
    assert len(a) == 4 and list(a["fsq_place_id"]) == list(b["fsq_place_id"])
    assert set(a["fsq_place_id"]) != set(c["fsq_place_id"])
    assert T.trap_candidates(_many(), per_pattern=0, seed=7).empty
    with pytest.raises(ValueError, match="per_pattern"):
        T.trap_candidates(_many(), per_pattern=-1, seed=7)


def test_set_aside_removes_candidate_ids_and_keys():
    pool = pd.concat([_basic_pool(), _pool([_row("dup", "St Mary's Hospital Cafe!", DINING),   # same key as h1
                                            _row("other", "Other Place", DINING)])], ignore_index=True)
    cands = T.trap_candidates(pool, per_pattern=1, seed=1)
    kept, info = T.set_aside(pool, cands)
    assert "dup" not in set(kept["fsq_place_id"]) and "other" in set(kept["fsq_place_id"])
    assert not set(kept["fsq_place_id"]) & set(cands["fsq_place_id"])
    assert info["candidates"] == len(cands) and info["removed_rows"] == len(pool) - len(kept)
    assert info["removed_by_key_only"] >= 1 and info["removed_by_id"] == len(cands)
    assert list(info["by_pattern"]) == PATTERNS
    assert list(kept.index) == list(range(len(kept)))


def test_write_candidates_roundtrip_is_deterministic(tmp_path):
    cands = T.trap_candidates(_basic_pool().assign(postcode="08302", facebook_id="122132281766001389"), 10, 1)
    path, warnings = T.write_candidates(cands, tmp_path / "trap_candidates.csv")
    first = path.read_bytes()
    T.write_candidates(cands, path)
    assert path.read_bytes() == first and warnings == [] and b"\r\n" not in first
    back = T.read_candidates(path)
    assert list(back["fsq_place_id"]) == list(cands["fsq_place_id"])
    assert back.loc[0, "keep_a1"] == "" and back["postcode"].tolist()[0] == "08302"


def test_write_candidates_never_overwrites_annotations(tmp_path):
    cands = T.trap_candidates(_basic_pool(), 10, 1)
    path, _ = T.write_candidates(cands, tmp_path / "trap_candidates.csv")
    annotated = T.read_candidates(path).assign(keep_a1="1")
    annotated.to_csv(path, index=False)
    before = path.read_bytes()
    written, warnings = T.write_candidates(cands, path)
    assert path.read_bytes() == before
    assert written.name == "trap_candidates.new.csv" and "annotations" in warnings[0]


# ---- merge ---------------------------------------------------------------------------------------------

IDS = ["h1", "b1", "p1", "m1", "m2"]  # annotation values below are given in this order


def _by_id(cands, values):
    return [dict(zip(IDS, values, strict=True))[pid] for pid in cands["fsq_place_id"]]


def _annotated(tmp_path, a1, a2, lead=None, name="merged.csv"):
    cands = T.trap_candidates(_basic_pool(), 10, 1)
    assert sorted(cands["fsq_place_id"]) == sorted(IDS)
    df = cands.assign(keep_a1=_by_id(cands, a1), keep_a2=_by_id(cands, a2),
                      keep_lead=_by_id(cands, lead or [""] * len(IDS)))
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path, cands


def test_merge_keeps_agreed_and_lead_resolved_rows(tmp_path):
    path, cands = _annotated(tmp_path, ["1", "0", "1", "1", "1.0"], ["1", "0", "0", "1", "0"],
                             ["", "", "1", "0", "0"])
    kept = T.merge_annotations(path)
    assert list(kept["fsq_place_id"]) == ["h1", "p1"]  # agreed 1; disagreement resolved 1; lead 0 overrides


def test_merge_of_two_annotator_files(tmp_path):
    cands = T.trap_candidates(_basic_pool(), 10, 1)
    a1, a2 = tmp_path / "a1.csv", tmp_path / "a2.csv"
    cands.assign(keep_a1=_by_id(cands, ["1", "0", "0", "1", "0"]),
                 note=_by_id(cands, ["x", "", "", "", ""])).to_csv(a1, index=False)
    cands.assign(keep_a2=_by_id(cands, ["1", "0", "0", "1", "1"])).iloc[::-1].to_csv(a2, index=False)
    with pytest.raises(ValueError, match="disagree"):
        T.merge_annotations(a1, a2)
    cands.assign(keep_a2=_by_id(cands, ["1", "0", "0", "1", "0"])).iloc[::-1].to_csv(a2, index=False)
    kept = T.merge_annotations(a1, a2)
    assert list(kept["fsq_place_id"]) == ["h1", "m1"] and kept.iloc[0]["note"] == "x"
    with pytest.raises(ValueError, match="keep_a2"):
        T.merge_annotations(a1, a1)  # the second file must carry annotator 2's column
    cands.iloc[:3].assign(keep_a2="1").to_csv(a2, index=False)
    with pytest.raises(ValueError, match="same candidates"):
        T.merge_annotations(a1, a2)


@pytest.mark.parametrize("a1, a2, match", [
    (["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"], "disagree"),
    (["1", "", "1", "1", "1"], ["1", "1", "1", "1", "1"], "not annotated"),
    (["1", "yes", "1", "1", "1"], ["1", "1", "1", "1", "1"], "1, 0 or empty"),
])
def test_merge_refuses_incomplete_or_invalid_annotation(tmp_path, a1, a2, match):
    path, _ = _annotated(tmp_path, a1, a2)
    with pytest.raises(ValueError, match=match):
        T.merge_annotations(path)


def _ctx(fits=lambda s: len(s) <= 400):
    return RowContext(scheme="c10", smoothing=0.1, fits=fits, count_tokens=len, keep_fields=KEEP,
                      evidence_fields=tuple(f for f in KEEP if f != "country"),
                      compress_order=("twitter", "instagram", "facebook_id", "email"), address_max_chars=120,
                      max_reject_rate=0.005, category_ids=frozenset({"4d4b7105d754a06374d81259"}),
                      category_names=frozenset(L.expected_level1_names()))


def test_trap_rows_follow_the_row_schema_and_prefer_pool_values(tmp_path):
    pool = _basic_pool().assign(postcode="08302")
    path, _ = _annotated(tmp_path, ["1", "0", "1", "1", "1"], ["1", "0", "1", "1", "1"])
    csv = pd.read_csv(path, dtype=str, keep_default_na=False).assign(postcode="8302")  # a spreadsheet ate the 0
    csv.to_csv(path, index=False)
    rows, stats = T.trap_rows(T.merge_annotations(path), _ctx(), pool=pool)
    assert [r["id"] for r in rows] == ["trap-000000", "trap-000001", "trap-000002", "trap-000003"]
    assert set(rows[0]) == {"id", "split", "state", "questions", "gold", "label", "country", "aug", "has_evidence",
                            "multi", "pattern"}
    by_name = {json.loads(r["state"])["name"]: r for r in rows}
    r0, m1 = rows[0], by_name["Bank Pharmacy"]
    assert r0["split"] == "trap" and r0["label"] == "dining" and r0["aug"] == "none" and r0["multi"] == 0
    assert json.loads(r0["state"]) == {"country": "GB", "locality": "Leeds", "name": "St Mary's HOSPITAL Cafe",
                                       "postcode": "08302"}
    assert list(json.loads(r0["state"])) == sorted(json.loads(r0["state"]))
    assert r0["gold"] == json.dumps(L.gold("dining", "c10", 0.1), ensure_ascii=False)
    assert m1["multi"] == 1 and m1["pattern"] == "multi_category" and m1["label"] == "retail"
    assert stats["written"] == 4 and stats["rejected"] == 0 and stats["by_pattern"]["multi_category"] == 2
    rows_csv, _ = T.trap_rows(T.merge_annotations(path), _ctx())  # no pool: the CSV values are used as they are
    assert json.loads(rows_csv[0]["state"])["postcode"] == "8302"


def test_trap_rows_count_rejections_and_fail_on_leakage(tmp_path):
    path, _ = _annotated(tmp_path, ["1"] * 5, ["1"] * 5)
    rows, stats = T.trap_rows(T.merge_annotations(path), _ctx(fits=lambda s: "Kings" not in s))
    assert stats["rejected"] == 1 and stats["rejected_ids"] == ["b1"] and len(rows) == 4
    leaky = _basic_pool().assign(website="shop-4d4b7105d754a06374d81259.example.com")
    with pytest.raises(AssertionError, match="category id"):
        T.trap_rows(T.merge_annotations(path), _ctx(), pool=leaky)


def test_eval_countries_and_keep_in_scope(tmp_path):
    assert T.eval_countries({"id_countries": ["GB", "US"], "ood_country": ["NL", "GB"], "ood_script": "TH"}) == [
        "GB", "US", "NL", "TH"]
    path, _ = _annotated(tmp_path, ["1"] * 5, ["1"] * 5)
    kept = T.merge_annotations(path)
    pool = _basic_pool()
    pool.loc[pool["fsq_place_id"] == "m2", "country"] = "KR"  # the extracted-but-unused script fallback
    got, dropped = T.keep_in_scope(kept, ["GB", "TH"], pool=pool)  # the pool's country wins over the CSV's
    assert list(got["fsq_place_id"]) == [p for p in kept["fsq_place_id"] if p != "m2"] and dropped == {"KR": 1}
    assert list(got.index) == list(range(len(got)))
    got_csv, dropped_csv = T.keep_in_scope(kept, ["GB"])  # no pool: the CSV's country (GB for all)
    assert len(got_csv) == 5 and dropped_csv == {}
    with pytest.raises(ValueError, match="not in the pool"):
        T.keep_in_scope(kept, ["GB"], pool=pool[pool["fsq_place_id"] != "h1"])


def _split(pool, ids):
    """A split parquet frame as build_data writes it (pool rows plus the dedup key)."""
    return add_keys(pool.set_index("fsq_place_id").loc[ids].reset_index())


def _write_build(path, pool, **splits):
    """A build's out dir as `traps merge` reads it: pool.parquet and split_<name>.parquet."""
    path.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(path / "pool.parquet", index=False)
    for name, df in splits.items():
        df.to_parquet(path / f"split_{name}.parquet", index=False)
    return path


def _clean_build(path, pool=None):
    pool = _basic_pool() if pool is None else pool
    return _write_build(path, pool, train=_split(pool, ["h2", "h3"]), val=_split(pool, ["p2", "s1"]))


def _merge_cli(tmp_path, cfg, monkeypatch, csv, *extra):
    import yaml
    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(T, "row_context", lambda cfg, args, ids: _ctx())
    out = tmp_path / "trap.jsonl"
    return T.main(["merge", "--candidates", str(csv), "--out", str(out), "--config", str(conf), *extra]), out


def test_merge_cli_writes_trap_jsonl(tmp_path, cfg, monkeypatch, capsys):
    path, _ = _annotated(tmp_path, ["1", "0", "1", "1", "1"], ["1", "0", "1", "1", "1"])
    _clean_build(tmp_path)
    code, out = _merge_cli(tmp_path, cfg, monkeypatch, path)
    printed = capsys.readouterr().out
    assert code == 0 and out.exists()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4
    assert "kept 4 of 5" in printed and "pool.parquet" in printed
    assert "no kept candidate (place or dedup key) is in any of the 2 splits (train, val)" in printed
    assert "outside data.trap_target" in printed  # 4 << 150
    assert "evaluation countries" not in printed


@pytest.mark.parametrize("leak, csv_key", [
    ("id", None),          # a kept candidate itself is in val
    ("key", None),         # another place with a kept candidate's dedup key is in val
    ("key", "edited"),     # ... found from the pool's values even when a spreadsheet edited the CSV's key
])
def test_merge_cli_refuses_kept_candidates_that_reached_a_split(tmp_path, cfg, monkeypatch, capsys, leak, csv_key):
    # e.g. an annotated CSV kept across a rebuild whose candidates changed (write_candidates -> *.new.csv)
    path, _ = _annotated(tmp_path, ["1", "0", "1", "1", "1"], ["1", "0", "1", "1", "1"])
    if csv_key:
        T.read_candidates(path).assign(key=csv_key).to_csv(path, index=False)
    pool = _basic_pool()
    extra = _split(pool, ["h1"]) if leak == "id" else add_keys(_pool([_row("dup", "St Mary's Hospital Cafe!", DINING)]))
    _write_build(tmp_path, pool, train=_split(pool, ["h2", "h3"]),
                 val=pd.concat([_split(pool, ["p2"]), extra], ignore_index=True))
    code, out = _merge_cli(tmp_path, cfg, monkeypatch, path)
    err = capsys.readouterr().err.strip()
    assert code == 1 and len(err.splitlines()) == 1
    assert "split val" in err and "not set aside by the build" in err and "trap_candidates.new.csv" in err
    assert not out.exists()


def test_merge_cli_needs_the_builds_splits(tmp_path, cfg, monkeypatch, capsys):
    path, _ = _annotated(tmp_path, ["1"] * 5, ["1"] * 5)
    _basic_pool().to_parquet(tmp_path / "pool.parquet", index=False)  # no split_*.parquet next to the CSV
    code, out = _merge_cli(tmp_path, cfg, monkeypatch, path)
    err = capsys.readouterr().err.strip()
    assert code == 1 and "split_*.parquet" in err and "--data-dir" in err and not out.exists()


def test_merge_cli_reads_the_build_from_data_dir(tmp_path, cfg, monkeypatch, capsys):
    path, _ = _annotated(tmp_path, ["1"] * 5, ["1"] * 5)  # the CSV's own dir holds no build
    build = _clean_build(tmp_path / "build")
    code, out = _merge_cli(tmp_path, cfg, monkeypatch, path, "--data-dir", str(build))
    printed = capsys.readouterr().out
    assert code == 0 and "field values from pool.parquet" in printed and "any of the 2 splits" in printed
    assert len(out.read_text(encoding="utf-8").splitlines()) == 5


def test_merge_cli_drops_kept_candidates_outside_the_evaluation_countries(tmp_path, cfg, monkeypatch, capsys):
    path, _ = _annotated(tmp_path, ["1"] * 5, ["1"] * 5)
    pool = _basic_pool()
    pool.loc[pool["fsq_place_id"] == "m2", "country"] = "KR"  # KR is extracted (fallback) but in no split
    assert "KR" not in cfg["data"]["id_countries"] + cfg["data"]["ood_country"] + [cfg["data"]["ood_script"]]
    _clean_build(tmp_path, pool)
    code, out = _merge_cli(tmp_path, cfg, monkeypatch, path)
    printed = capsys.readouterr().out
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert code == 0 and len(rows) == 4 and all(r["country"] != "KR" for r in rows)
    assert 'dropped 1 kept candidates outside the evaluation countries: {"KR": 1}' in printed


def test_merge_cli_one_line_error(tmp_path, capsys):
    path, _ = _annotated(tmp_path, ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"])
    code = T.main(["merge", "--candidates", str(path), "--out", str(tmp_path / "t.jsonl")])
    err = capsys.readouterr().err.strip()
    assert code == 1 and len(err.splitlines()) == 1 and "disagree" in err
    assert not (tmp_path / "t.jsonl").exists()
