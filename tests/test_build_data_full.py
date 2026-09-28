"""FULL-mode build_data (Phase 2): trap set-aside, clean train.jsonl, Event headline rule, JSONL
fingerprint, frozen-manifest check, DuckDB pin and two-pass extraction settings. CPU, no network."""
import hashlib
import json

import pandas as pd
import pytest
import yaml

from laya_poc import labels as L
from laya_poc.config import with_overrides
from laya_poc.io_utils import read_jsonl, sha256_file

pytestmark = pytest.mark.torch
pytest.importorskip("laya")

from laya_poc import build_data as B  # noqa: E402
from laya_poc import traps as T  # noqa: E402
from test_build_data import _build, _pool, _report  # noqa: E402

FULL = {"data.id_countries": ["GB", "US", "DE", "JP"], "data.ood_country": ["NL"], "data.ood_script": "TH",
        "data.train_size": 60, "data.train_cap_per_class": 8, "data.val_size": 20, "data.test_size": 20,
        "data.ood_size": 10, "data.brand_top_n": 2, "data.trap_per_pattern": 2, "data.frozen_manifest": None,
        "train.epochs": 2}
ALL_JSONL = ["ood_brand.jsonl", "ood_country.jsonl", "ood_script.jsonl", "stripped_test.jsonl", "test_id.jsonl",
             "train.jsonl", "train_e0.jsonl", "train_e1.jsonl", "val.jsonl"]


def _like(base: pd.DataFrame, pid: str, name: str, country: str, l1: str, locality: str) -> dict:
    return {**base.iloc[0].to_dict(), "fsq_place_id": pid, "name": name, "locality": locality, "country": country,
            "label": l1, "l1s": [l1], "n_l1": 1}


def _full_pool(seed=0):
    """ID + OOD countries, two brand chains (ood_brand), trap-word names and mixed-l1 rows."""
    base = _pool(n_per_country=30, countries=("GB", "US", "DE", "JP", "NL", "TH"), seed=seed)
    extra = []
    for c in ("GB", "US", "DE", "JP"):
        for i in range(6):
            extra.append(_like(base, f"{c}mm{i}", "Mega Mart", c, "Retail", f"Town{i}"))
            extra.append(_like(base, f"{c}qc{i}", "Quick Cafe", c, "Dining and Drinking", f"Town{i}"))
        for i, (name, l1) in enumerate([("Station Cafe", "Dining and Drinking"), ("Kings Bank Bakery", "Retail"),
                                        ("Park Lane Pharmacy", "Health and Medicine")]):
            extra.append(_like(base, f"{c}trap{i}", f"{name} {c}", c, l1, "Leeds"))
    pool = pd.concat([base, pd.DataFrame(extra)], ignore_index=True)
    multi = (pool["n_l1"] == 2).tolist()
    return pool.assign(l1s=[["Health and Medicine", "Retail"] if m else ls for m, ls in zip(multi, pool["l1s"])],
                       first_l1=["Retail" if m else lab for m, lab in zip(multi, pool["label"])])


def _full(tmp_path, cfg, tiny_ckpt_dir, name="data", extra=(), pool=None, **over):
    pool_path = tmp_path / "full_pool.parquet"
    (pool if pool is not None else _full_pool()).to_parquet(pool_path, index=False)
    conf = tmp_path / f"config_{name}.yaml"
    conf.write_text(yaml.safe_dump(with_overrides(cfg, {**FULL, **over})), encoding="utf-8")
    out = tmp_path / name
    argv = ["--out", str(out), "--config", str(conf), "--pool", str(pool_path), "--models", "laya",
            "--init-tokenizer", str(tiny_ckpt_dir), *extra]
    return B.main(argv), out


def _jsonl_names(out):
    return sorted(p.name for p in out.glob("*.jsonl"))


def test_full_build_sets_trap_candidates_aside_before_the_splits(tmp_path, cfg, tiny_ckpt_dir, capsys):
    code, out = _full(tmp_path, cfg, tiny_ckpt_dir)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert len(captured.out.strip().splitlines()) <= 15 and "trap candidates" in captured.out
    cands = T.read_candidates(out / "trap_candidates.csv")
    assert {"station_hotel_word", "bank_word", "park_garden_word", "multi_category"} <= set(cands["pattern"])
    assert (cands.groupby("pattern").size() <= 2).all()
    for split in out.glob("split_*.parquet"):
        df = pd.read_parquet(split)
        assert not set(df["fsq_place_id"]) & set(cands["fsq_place_id"]), split.name
        assert not set(df["key"]) & set(cands["key"]), split.name
    traps = _report(out)["traps"]
    assert traps["candidates"] == len(cands) and traps["annotation"] == "pending"
    assert traps["removed_rows"] >= len(cands) and traps["file"] == "trap_candidates.csv"
    assert list(traps["by_pattern"]) == T.pattern_names()


def test_full_build_writes_a_clean_train_jsonl_for_the_baselines(tmp_path, cfg, tiny_ckpt_dir):
    code, out = _full(tmp_path, cfg, tiny_ckpt_dir)
    assert code == 0
    train = read_jsonl(out / "train.jsonl")
    split = pd.read_parquet(out / "split_train.parquet")
    assert len(train) == len(split) == _report(out)["stats"]["train"]["written"]
    assert all(r["aug"] == "none" and r["label"] in L.option_keys("c10") and r["has_evidence"] for r in train)
    assert all(list(json.loads(r["state"])) == sorted(json.loads(r["state"])) for r in train)
    assert sorted(r["label"] for r in train) == sorted(L.l1_to_key(x) for x in split["label"])
    assert _jsonl_names(out) == ALL_JSONL
    assert "train.jsonl" in (out / "SHA256SUMS").read_text(encoding="utf-8")


@pytest.mark.parametrize("minimum, classes", [(5, 10), (300, 9)])
def test_event_headline_rule(tmp_path, cfg, tiny_ckpt_dir, minimum, classes):
    code, out = _full(tmp_path, cfg, tiny_ckpt_dir, **{"labels.event_min_id_pool": minimum})
    report = _report(out)
    assert code == 0 and report["headline_classes"] == classes
    split = {n: pd.read_parquet(out / f"split_{n}.parquet") for n in ("train", "val", "test_id")}
    in_splits = sum(int((df["label"] == "Event").sum()) for df in split.values())
    assert 0 < in_splits <= report["event_id_pool"] == report["id_pool"]["id_pool_labels"]["Event"]


def test_full_build_fingerprint_covers_every_jsonl_and_is_reproducible(tmp_path, cfg, tiny_ckpt_dir):
    code_a, a = _full(tmp_path, cfg, tiny_ckpt_dir, "a")
    code_b, b = _full(tmp_path, cfg, tiny_ckpt_dir, "b")
    assert code_a == code_b == 0
    ra, rb = _report(a), _report(b)
    assert sorted(ra["fingerprint"]) == _jsonl_names(a) == ALL_JSONL
    assert all(ra["fingerprint"][n] == sha256_file(a / n) for n in ra["fingerprint"])
    lines = "".join(sorted(f"{sha}  {name}\n" for name, sha in ra["fingerprint"].items()))
    assert ra["fingerprint_sha256"] == hashlib.sha256(lines.encode("utf-8")).hexdigest()
    assert ra["fingerprint"] == rb["fingerprint"] and ra["fingerprint_sha256"] == rb["fingerprint_sha256"]
    for name in ALL_JSONL:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert ra["frozen_check"]["status"] == "not_frozen"


def test_a_later_trap_jsonl_does_not_change_the_fingerprint(tmp_path, cfg, tiny_ckpt_dir):
    code, a = _full(tmp_path, cfg, tiny_ckpt_dir, "a")
    (a / "trap.jsonl").write_text('{"id": "trap-000000"}\n', encoding="utf-8")
    first = _report(a)["fingerprint_sha256"]
    code, a = _full(tmp_path, cfg, tiny_ckpt_dir, "a")
    assert code == 0 and (a / "trap.jsonl").exists() and _report(a)["fingerprint_sha256"] == first


def _annotate_all(csv):
    T.read_candidates(csv).assign(keep_a1="1", keep_a2="1").to_csv(csv, index=False)


def test_trap_merge_refuses_an_annotated_csv_from_before_a_rebuild(tmp_path, cfg, tiny_ckpt_dir, capsys):
    """An annotated trap_candidates.csv survives a rebuild whose candidates changed (the new ones go to
    trap_candidates.new.csv); its kept places are no longer set aside and may be in a split (§5.3)."""
    code, out = _full(tmp_path, cfg, tiny_ckpt_dir)  # trap_per_pattern 2
    csv, new = out / "trap_candidates.csv", out / "trap_candidates.new.csv"
    assert code == 0
    _annotate_all(csv)
    code, out = _full(tmp_path, cfg, tiny_ckpt_dir, **{"data.trap_per_pattern": 1})
    assert code == 0 and new.exists()
    in_splits = set().union(*(set(pd.read_parquet(p)["fsq_place_id"]) for p in out.glob("split_*.parquet")))
    assert in_splits & set(T.read_candidates(csv)["fsq_place_id"])  # the stale CSV really leaks here
    trap = tmp_path / "trap.jsonl"
    merge = ["merge", "--out", str(trap), "--config", str(tmp_path / "config_data.yaml"), "--models", "laya",
             "--init-tokenizer", str(tiny_ckpt_dir)]
    capsys.readouterr()
    assert T.main([*merge, "--candidates", str(csv)]) == 1
    assert "not set aside by the build" in capsys.readouterr().err and not trap.exists()
    _annotate_all(new)  # the current build's candidates merge cleanly
    assert T.main([*merge, "--candidates", str(new)]) == 0
    assert len(read_jsonl(trap)) == len(T.read_candidates(new))


def test_write_manifest_then_verify_frozen(tmp_path, cfg, tiny_ckpt_dir, capsys):
    manifest = tmp_path / "frozen.json"
    code, a = _full(tmp_path, cfg, tiny_ckpt_dir, "a", extra=["--write-manifest", str(manifest)])
    assert code == 0
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    assert {"release", "created", "duckdb", "pandas", "pyarrow", "python", "platform", "files",
            "fingerprint_sha256", "split_sizes", "event_id_pool", "headline_classes"} <= set(frozen)
    assert frozen["files"] == _report(a)["fingerprint"]
    assert "state" not in manifest.read_text(encoding="utf-8")  # hashes and counts only, no FSQ rows

    code, b = _full(tmp_path, cfg, tiny_ckpt_dir, "b", extra=["--verify-frozen"],
                    **{"data.frozen_manifest": str(manifest)})
    assert code == 0 and _report(b)["frozen_check"]["status"] == "match"

    files = {**frozen["files"], "val.jsonl": "0" * 64}
    files.pop("ood_brand.jsonl")
    manifest.write_text(json.dumps({**frozen, "files": files}), encoding="utf-8")
    capsys.readouterr()
    code, c = _full(tmp_path, cfg, tiny_ckpt_dir, "c", extra=["--verify-frozen"],
                    **{"data.frozen_manifest": str(manifest)})
    err = capsys.readouterr().err.strip()
    assert code == 1 and len(err.splitlines()) == 1
    assert "val.jsonl" in err and "ood_brand.jsonl" in err and "re-freeze" in err and "train.jsonl" not in err
    assert _report(c)["frozen_check"]["status"] == "mismatch"  # the report is still written for debugging


def test_verify_frozen_fails_when_the_manifest_file_is_missing(tmp_path, cfg, tiny_ckpt_dir, capsys):
    code, _ = _full(tmp_path, cfg, tiny_ckpt_dir, extra=["--verify-frozen"],
                    **{"data.frozen_manifest": "docs/results/no_such_manifest.json"})
    err = capsys.readouterr().err
    assert code == 1 and "no_such_manifest.json" in err and "not found" in err


def test_full_mode_flags_are_rejected_in_smoke_mode(tmp_path, cfg, tiny_ckpt_dir, capsys):
    code, _ = _build(tmp_path, cfg, tiny_ckpt_dir, extra=["--verify-frozen"])
    assert code == 1 and "--verify-frozen" in capsys.readouterr().err


def test_smoke_build_has_no_full_mode_outputs(tmp_path, cfg, tiny_ckpt_dir):
    code, out = _build(tmp_path, cfg, tiny_ckpt_dir)
    report = _report(out)
    assert code == 0 and not (out / "train.jsonl").exists() and not (out / "trap_candidates.csv").exists()
    assert not {"fingerprint", "fingerprint_sha256", "traps", "event_id_pool", "headline_classes"} & set(report)


def test_duckdb_pin_only_warns_for_pool_builds(tmp_path, cfg, tiny_ckpt_dir, monkeypatch):
    import duckdb
    monkeypatch.setattr(duckdb, "__version__", "0.0.1")
    code, out = _full(tmp_path, cfg, tiny_ckpt_dir)
    assert code == 0 and any("duckdb 0.0.1" in w for w in _report(out)["warnings"])


def test_duckdb_pin_blocks_full_extraction(tmp_path, cfg, monkeypatch, capsys):
    import duckdb
    monkeypatch.setattr(duckdb, "__version__", "0.0.1")
    monkeypatch.setenv("HF_TOKEN", "hf_dummyTokenValue1234567890abcd")
    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump(with_overrides(cfg, FULL)), encoding="utf-8")
    code = B.main(["--out", str(tmp_path / "d"), "--config", str(conf)])
    err = capsys.readouterr().err
    assert code == 1 and "duckdb==" in err and "hf_dummy" not in err


def _local_full_release(tmp_path):
    import test_extract as TX
    base = tmp_path / "release" / f"dt={TX.REL}"
    TX._write_categories(base / "categories" / "parquet" / "categories_000000.parquet")
    cids = [c for c in TX.CATS if c != TX.NIGHTLIFE]
    rows = []
    for c in ("GB", "US", "DE", "JP", "NL", "TH", "KR"):
        rows += [TX._place(f"{c}-{i:03d}", c, [cids[i % len(cids)]], name=f"Blue Place {c} {i}",
                           locality=f"Town{i % 7}") for i in range(60)]
        rows += [TX._place(f"{c}-mm{i}", c, [TX.RETAIL], name="Mega Mart", locality=f"Town{i}") for i in range(4)]
        rows += [TX._place(f"{c}-m{i}", c, [TX.RETAIL, TX.HEALTH], name=f"Hospital Shop {c}{i}") for i in range(3)]
    pdir = base / "places" / "parquet"
    TX._write_places(pdir / "places_000001.parquet", rows[: len(rows) // 2])
    TX._write_places(pdir / "places_000002.parquet", rows[len(rows) // 2:])
    return base


def test_full_extraction_two_passes_settings_and_pool_rebuild(tmp_path, cfg, tiny_ckpt_dir, monkeypatch, capsys):
    import duckdb
    _local_full_release(tmp_path)
    seen = []
    monkeypatch.setenv("HF_TOKEN", "hf_dummyTokenValue1234567890abcd")
    monkeypatch.setattr(B.extract, "connect", lambda **kw: seen.append(kw) or duckdb.connect())
    monkeypatch.setattr(B.extract, "create_hf_secret", lambda con, tok: None)
    over = {**FULL, "data.hf_base": (tmp_path / "release" / "dt={rel}").as_posix(), "data.fsq_release": "2026-09-15",
            "data.pool_per_country_label": 8, "data.val_size": 10, "data.test_size": 10, "data.train_size": 30,
            "data.ood_size": 4}
    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump(with_overrides(cfg, over)), encoding="utf-8")
    spill, out = tmp_path / "spill", tmp_path / "data"
    spill.mkdir()
    (spill / "keep.txt").write_text("user file", encoding="utf-8")
    code = B.main(["--out", str(out), "--config", str(conf), "--models", "laya", "--init-tokenizer",
                   str(tiny_ckpt_dir), "--duckdb-memory", "2GB", "--duckdb-threads", "3", "--duckdb-temp", str(spill)])
    assert code == 0, capsys.readouterr().err
    assert seen == [{"threads": 3, "memory_limit": "2GB", "temp_directory": str(spill / "laya_duckdb_spill")}]
    assert (spill / "keep.txt").exists() and not (spill / "laya_duckdb_spill").exists()  # only ours is removed
    report = _report(out)
    assert report["extraction"]["passes"] == 2 and report["extraction"]["duckdb"]["memory_limit"] == "2GB"
    assert report["extraction"]["rows_by_country"]["KR"] > 0
    assert pd.read_parquet(out / "pool.parquet")["n_l1"].max() == 2  # mixed-l1 rows kept for the traps
    # Rebuilding from the extracted pool.parquet (object vs Arrow-backed strings) gives identical JSONL
    rebuilt = tmp_path / "rebuilt"
    code = B.main(["--out", str(rebuilt), "--config", str(conf), "--pool", str(out / "pool.parquet"), "--models",
                   "laya", "--init-tokenizer", str(tiny_ckpt_dir)])
    assert code == 0 and _report(rebuilt)["fingerprint"] == report["fingerprint"]
