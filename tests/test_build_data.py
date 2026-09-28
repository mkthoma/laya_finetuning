"""End-to-end build_data runs on CPU: a synthetic --pool, and a local "extraction" (no network)."""
import json
import random
from pathlib import Path

import pandas as pd
import pytest
import yaml

from laya_poc import labels as L
from laya_poc.config import with_overrides
from laya_poc.io_utils import read_jsonl

pytestmark = pytest.mark.torch
pytest.importorskip("laya")

from laya_poc import build_data as B  # noqa: E402

KEEP_EXTRA_CIDS = ["4bf58dd8d48988d1c4941735", "63be6904847c3692a84b9bb9"]
SMALL = {"smoke.train_questions": 60, "smoke.train_cap_per_class": 8, "smoke.val_size": 20,
         "smoke.test_size": 20, "smoke.brand_top_n": 2, "smoke.epochs": 2}


def _pool(n_per_country=30, countries=("GB", "US", "DE", "JP"), seed=0):
    rng = random.Random(seed)
    names = list(L.L1_TO_KEY)
    rows = []
    for c in countries:
        for i in range(n_per_country):
            l1 = names[i % len(names)]
            word = l1.split()[0]
            rows.append({
                "fsq_place_id": f"{c}{i:04d}", "name": f"{rng.choice(['Rosa', 'Blue', 'Kings'])} {word} {c}{i}",
                "address": f"{i} High St" if i % 2 else None, "locality": rng.choice(["Leeds", "Osaka", "Lyon"]),
                "region": None, "postcode": None, "admin_region": None, "post_town": None, "po_box": None,
                "country": c, "tel": f"0{i:03d} 111" if i % 3 else None,
                "website": f"http://www.site{c}{i}.com" if i % 4 == 0 else None,
                "email": None, "facebook_id": "122132281766001389" if i == 5 else None,
                "instagram": None, "twitter": None,
                "fsq_category_ids": [KEEP_EXTRA_CIDS[i % 2]], "l1s": [l1],
                "label": l1, "n_l1": 2 if i == 7 else 1})
    return pd.DataFrame(rows)


def _config(cfg, tmp_path, **over):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(with_overrides(cfg, {**SMALL, **over})), encoding="utf-8")
    return path


def _build(tmp_path, cfg, tiny_ckpt_dir, name="data", extra=(), pool=None, **over):
    pool_path = tmp_path / "pool_in.parquet"
    if pool is not None:
        pool.to_parquet(pool_path, index=False)
    elif not pool_path.exists():
        _pool().to_parquet(pool_path, index=False)
    out = tmp_path / name
    argv = ["--out", str(out), "--smoke", "--config", str(_config(cfg, tmp_path, **over)), "--pool",
            str(pool_path), "--models", "laya", "--init-tokenizer", str(tiny_ckpt_dir), *extra]
    return B.main(argv), out


def _report(out):
    return json.loads((out / "data_report.json").read_text(encoding="utf-8"))


def test_build_from_pool_writes_layout_and_valid_rows(tmp_path, cfg, tiny_ckpt_dir, capsys):
    code, out = _build(tmp_path, cfg, tiny_ckpt_dir)
    printed = capsys.readouterr().out
    assert code == 0
    assert len(printed.strip().splitlines()) <= 15
    for f in ["pool.parquet", "split_train.parquet", "split_val.parquet", "split_test_id.parquet",
              "train_e0.jsonl", "train_e1.jsonl", "val.jsonl", "test_id.jsonl", "stripped_test.jsonl",
              "data_report.json", "SHA256SUMS"]:
        assert (out / f).exists(), f
    assert not list(out.glob("ood_*.jsonl"))  # smoke: OOD pools are empty

    val = read_jsonl(out / "val.jsonl")
    assert len(val) == 20
    assert all(list(json.loads(r["state"])) == sorted(json.loads(r["state"])) for r in val)
    assert all(r["label"] in L.option_keys("c10") and r["aug"] == "none" for r in val)
    stripped = read_jsonl(out / "stripped_test.jsonl")
    assert len(stripped) == 20 and all(set(json.loads(r["state"])) <= {"country"} for r in stripped)

    e0, e1 = read_jsonl(out / "train_e0.jsonl"), read_jsonl(out / "train_e1.jsonl")
    assert [r["state"] for r in e0] != [r["state"] for r in e1]  # augmentation regenerated per epoch
    n_train = len(pd.read_parquet(out / "split_train.parquet"))
    assert len(e0) == n_train + round(0.07 * n_train)
    assert sum(r["aug"] == "stripped" for r in e0) == round(0.07 * n_train)
    assert all(r["label"] is None for r in e0 if r["aug"] in ("stripped", "emptied"))

    report = json.loads((out / "data_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "smoke" and report["pool"]["source"] == "file"
    assert report["split_sizes"]["val"] == 20 and report["split_sizes"]["ood_brand"] == 0
    assert set(report["stats"]) == {"val", "test_id", "stripped_test", "train_e0", "train_e1"}
    assert report["budgets"]["laya"] == report["models"]["laya"]["budget"] == 355  # EN tokenizer, margin 8
    assert report["leakage"]["category_ids"] == 2
    assert not any("leakage" in w for w in report["warnings"])
    assert any(w.startswith("train_e0 has") for w in report["warnings"])  # 60 rows << 250 x 8
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8")
    assert "train_e1.jsonl" in sums and "split_val.parquet" in sums


def test_build_writes_the_fsq_notice_and_records_its_sha256(tmp_path, cfg, tiny_ckpt_dir):
    """Appendix D: every kept FSQ-derived artefact carries the NOTICE verbatim plus a 'modified' statement."""
    from laya_poc.io_utils import sha256_file
    from laya_poc.notice import fsq_notice

    code, out = _build(tmp_path, cfg, tiny_ckpt_dir)
    path = out / "NOTICE_FSQ.txt"
    assert code == 0
    assert path.read_bytes() == fsq_notice(cfg["data"]["fsq_release"]).encode("utf-8")
    assert _report(out)["notice"] == {"file": "NOTICE_FSQ.txt", "sha256": sha256_file(path)}


def test_rows_feed_build_items(tmp_path, cfg, tiny_ckpt_dir):
    from laya_poc.hub import load_tokenizer_dir
    from laya_poc.items import build_items
    code, out = _build(tmp_path, cfg, tiny_ckpt_dir)
    assert code == 0
    tok = load_tokenizer_dir(tiny_ckpt_dir)
    for name in ("train_e0.jsonl", "val.jsonl", "stripped_test.jsonl"):
        for row in read_jsonl(out / name):
            items = build_items(tok, row, 512, 192)
            assert len(items) == 1 and len(items[0]["markers"]) == 10


def test_build_is_deterministic(tmp_path, cfg, tiny_ckpt_dir):
    code_a, a = _build(tmp_path, cfg, tiny_ckpt_dir, "a")
    code_b, b = _build(tmp_path, cfg, tiny_ckpt_dir, "b")
    assert code_a == code_b == 0
    for name in ("train_e0.jsonl", "train_e1.jsonl", "val.jsonl", "test_id.jsonl", "stripped_test.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_rebuild_removes_stale_epoch_files(tmp_path, cfg, tiny_ckpt_dir):
    out = tmp_path / "data"
    out.mkdir()
    (out / "train_e7.jsonl").write_text("{}\n", encoding="utf-8")
    code, out = _build(tmp_path, cfg, tiny_ckpt_dir)
    assert code == 0 and not (out / "train_e7.jsonl").exists()


def test_missing_hf_token_fails_with_one_line(tmp_path, cfg, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    code = B.main(["--out", str(tmp_path / "d"), "--smoke", "--config", str(_config(cfg, tmp_path))])
    err = capsys.readouterr().err.strip()
    assert code != 0
    assert len(err.splitlines()) == 1 and "HF_TOKEN" in err


def test_unknown_model_fails(tmp_path, cfg, tiny_ckpt_dir, capsys):
    code, _ = _build(tmp_path, cfg, tiny_ckpt_dir, extra=["--models", "nope"])
    assert code != 0 and "nope" in capsys.readouterr().err


def test_extraction_path_with_local_release(tmp_path, cfg, tiny_ckpt_dir, monkeypatch, capsys):
    import duckdb
    import test_extract as TX

    base = tmp_path / "release" / f"dt={TX.REL}"
    TX._write_categories(base / "categories" / "parquet" / "categories_000000.parquet")
    pdir = base / "places" / "parquet"
    TX._write_places(pdir / "places_000089.parquet", TX._filter_rows() + TX._cap_rows(30, ("GB",)))
    TX._write_places(pdir / "places_000097.parquet", TX._cap_rows(30, ("JP",)))
    token = "hf_dummyTokenValue1234567890abcd"
    seen, connect_kw = [], []
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(B.extract, "connect", lambda **kw: connect_kw.append(kw) or duckdb.connect())
    monkeypatch.setattr(B.extract, "create_hf_secret", lambda con, tok: seen.append(tok))
    conf = _config(cfg, tmp_path, **{"data.hf_base": (tmp_path / "release" / "dt={rel}").as_posix(),
                                     "data.fsq_release": TX.REL, "data.id_countries": ["GB", "JP"],
                                     "smoke.best_files": {"GB": 89, "JP": 97}, "smoke.per_country": 50})
    out = tmp_path / "data"
    code = B.main(["--out", str(out), "--smoke", "--config", str(conf), "--models", "laya",
                   "--init-tokenizer", str(tiny_ckpt_dir)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert seen == [token] and token not in captured.out + captured.err
    assert connect_kw == [{"threads": 8}]  # hf:// scan is I/O-bound; research timings used 8 threads
    report = json.loads((out / "data_report.json").read_text(encoding="utf-8"))
    assert report["pool"]["source"] == "extract"
    assert report["extraction"]["rows_by_country"] == {"GB": 50, "JP": 50}
    assert report["leakage"]["category_ids"] >= len(TX.CATS)
    pool = pd.read_parquet(out / "pool.parquet")
    assert "nightlife" not in set(pool["fsq_place_id"])
    assert token not in "".join(p.read_text(encoding="utf-8", errors="ignore")
                                for p in Path(out).iterdir() if p.suffix in (".json", ".jsonl"))


def test_smoke_capacity_warning(cfg):
    need = cfg["smoke"]["micro_steps"] * cfg["train"]["micro_batch"]["T4"]
    short = B.smoke_capacity_warnings(cfg, {"train_e0": {"stats": {"written": need - 1}}})
    assert len(short) == 1 and str(need) in short[0]
    assert B.smoke_capacity_warnings(cfg, {"train_e0": {"stats": {"written": need}}}) == []


def _outputs_absent(out):
    return not any((out / f).exists() for f in ("val.jsonl", "train_e0.jsonl", "data_report.json", "SHA256SUMS",
                                                  "NOTICE_FSQ.txt"))


def test_category_id_in_a_state_fails_the_build(tmp_path, cfg, tiny_ckpt_dir, capsys):
    pool = _pool().assign(website=f"shop-{KEEP_EXTRA_CIDS[0]}.example.com")
    code, out = _build(tmp_path, cfg, tiny_ckpt_dir, pool=pool)
    err = capsys.readouterr().err.strip()
    assert code == 1 and len(err.splitlines()) == 1
    assert f"category id {KEEP_EXTRA_CIDS[0]}" in err
    assert _outputs_absent(out)


@pytest.mark.parametrize("max_rate, code", [(0.0, 1), (0.9, 0)])
def test_reject_rate_is_enforced_per_config(tmp_path, cfg, tiny_ckpt_dir, capsys, max_rate, code):
    pool = _pool()
    pool = pool.assign(name=[f"{n} {'N' * 3000}" if i % 3 == 0 else n for i, n in enumerate(pool["name"])])
    got, out = _build(tmp_path, cfg, tiny_ckpt_dir, pool=pool, **{"serialise.max_reject_rate": max_rate})
    err = capsys.readouterr().err
    assert got == code, err
    if code:
        assert "max_reject_rate" in err and _outputs_absent(out)
    else:
        assert all(_report(out)["stats"][s]["rejected"] > 0 for s in ("val", "test_id", "train_e0"))


def test_every_written_split_goes_through_the_leakage_check(tmp_path, cfg, tiny_ckpt_dir, monkeypatch):
    checked = []
    real = B.assert_no_leakage

    def spy(rows, *args):
        rows = list(rows)
        checked.extend({r["split"] for r in rows})
        return real(rows, *args)

    monkeypatch.setattr(B, "assert_no_leakage", spy)
    code, out = _build(tmp_path, cfg, tiny_ckpt_dir)
    assert code == 0
    assert set(checked) == {"val", "test_id", "stripped_test", "train"}
    assert checked.count("train") == 2  # once per epoch


def test_pool_without_category_ids_warns_that_the_id_check_was_empty(tmp_path, cfg, tiny_ckpt_dir):
    code, out = _build(tmp_path, cfg, tiny_ckpt_dir, pool=_pool().drop(columns=["fsq_category_ids"]))
    report = _report(out)
    assert code == 0 and report["leakage"]["category_ids"] == 0
    assert report["warnings"][0].startswith("category-id leakage check had no category ids")
