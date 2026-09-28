import copy
import json
import random

import pandas as pd
import pytest

from laya_poc import labels as L
from laya_poc import rows as R
from laya_poc import serialise as S

KEEP = ["name", "address", "locality", "region", "postcode", "country", "tel", "website", "email",
        "facebook_id", "instagram", "twitter"]
EVIDENCE = [f for f in KEEP if f != "country"]
COMPRESS = ["twitter", "instagram", "facebook_id", "email"]
ROW_KEYS = {"id", "split", "state", "questions", "gold", "label", "country", "aug", "has_evidence"}


def _kw(**over):
    base = dict(scheme="c10", smoothing=0.1, fits=lambda s: len(s) <= 200, compress_order=COMPRESS,
                address_max_chars=120, evidence_fields=EVIDENCE, count_tokens=len)
    base.update(over)
    return base


def _recs():
    return [
        ({"country": "GB", "name": "Rosa's Trattoria", "tel": "0113 000"}, "dining"),
        ({"country": "US", "name": "Big Mart", "twitter": "t" * 190}, "retail"),       # compressed
        ({"country": "JP", "name": "N" * 400}, "health"),                              # rejected
        ({"country": "FR"}, None),                                                     # no evidence
    ]


def test_records_from_split_maps_labels_and_cleans():
    df = pd.DataFrame([{"name": " A ", "country": "GB", "label": "Dining and Drinking", "tel": None, "key": "x"},
                       {"name": "B", "country": "US", "label": "Event", "tel": "1", "key": "y"}])
    assert R.records_from_split(df, KEEP, "c10") == [({"country": "GB", "name": "A"}, "dining"),
                                                     ({"country": "US", "name": "B", "tel": "1"}, "event")]
    assert [lab for _, lab in R.records_from_split(df, KEEP, "c7")] == ["dining", "culture"]


def test_records_from_split_rejects_unknown_level1():
    df = pd.DataFrame([{"name": "A", "country": "GB", "label": "Nightlife Spot"}])
    with pytest.raises(KeyError, match="Nightlife"):
        R.records_from_split(df, KEEP, "c10")


def test_make_rows_schema_and_stats():
    recs = _recs()
    snapshot = copy.deepcopy(recs)
    rows, stats = R.make_rows("val", recs, **_kw())
    assert recs == snapshot
    assert [r["id"] for r in rows] == ["val-000000", "val-000001", "val-000002"]
    assert all(set(r) == ROW_KEYS and r["split"] == "val" and r["aug"] == "none" for r in rows)
    first = rows[0]
    assert first["state"] == '{"country":"GB","name":"Rosa\'s Trattoria","tel":"0113 000"}'
    assert first["questions"] == json.dumps(L.question("c10"), ensure_ascii=False)
    assert first["gold"] == json.dumps(L.gold("dining", "c10", 0.1), ensure_ascii=False)
    assert (first["label"], first["country"], first["has_evidence"]) == ("dining", "GB", True)
    assert "twitter" not in json.loads(rows[1]["state"])
    last = rows[2]
    assert last["label"] is None and last["has_evidence"] is False
    assert json.loads(last["gold"])[L.QUESTION_NAME]["probabilities"] == pytest.approx({k: 0.1 for k in L.option_keys("c10")})
    assert stats["written"] == 3 and stats["compressed"] == 1 and stats["rejected"] == 1
    assert stats["no_evidence"] == 1 and stats["total"] == 4
    lengths = sorted(len(r["state"]) for r in rows)
    assert stats["tokens_max"] == lengths[-1] and stats["tokens_p50"] == lengths[1]
    assert stats["tokens_p50"] <= stats["tokens_p95"] <= stats["tokens_max"]
    assert stats["aug"] == {"none": 3}


def test_make_rows_eval_keys_are_alphabetical():
    rec = {"tel": "1", "name": "A", "country": "GB"}
    rows, _ = R.make_rows("val", [(rec, "dining")], **_kw())
    assert list(json.loads(rows[0]["state"])) == ["country", "name", "tel"]


def test_make_rows_shuffle_is_seeded():
    rec = {f: f"v{i}" for i, f in enumerate(["country", "name", "tel", "locality", "address", "website"])}
    recs = [(dict(rec), "dining") for _ in range(10)]
    a, _ = R.make_rows("train", recs, **_kw(), rng=random.Random(7), shuffle=True)
    b, _ = R.make_rows("train", recs, **_kw(), rng=random.Random(7), shuffle=True)
    assert [r["state"] for r in a] == [r["state"] for r in b]
    orders = {tuple(json.loads(r["state"])) for r in a}
    assert len(orders) > 1  # field order varies row to row
    with pytest.raises(ValueError, match="rng"):
        R.make_rows("train", recs, **_kw(), shuffle=True)


def test_make_rows_aug_tags():
    recs = [({"country": "GB", "name": "A"}, "dining"), ({"country": "GB"}, None)]
    rows, stats = R.make_rows("train", recs, **_kw(), aug_tags=["conflict", "stripped"])
    assert [r["aug"] for r in rows] == ["conflict", "stripped"]
    assert stats["aug"] == {"conflict": 1, "stripped": 1}
    with pytest.raises(ValueError, match="aug_tags"):
        R.make_rows("train", recs, **_kw(), aug_tags=["none"])


def test_make_rows_empty_split():
    rows, stats = R.make_rows("ood_script", [], **_kw())
    assert rows == [] and stats["written"] == 0 and stats["tokens_max"] == 0
    assert stats["evidence_lost"] == 0


def _all_evidence_compressed():
    """Only `email` carries evidence and it is too long to fit: compression leaves country alone."""
    return [({"country": "GB", "email": "e" * 300}, "dining")], _kw(fits=lambda s: len(s) <= 100)


def test_make_rows_train_row_compressed_to_no_evidence_gets_uniform_target():
    recs, kw = _all_evidence_compressed()
    rows, stats = R.make_rows("train", recs, **kw, rng=random.Random(0), shuffle=True, aug_tags=["conflict"])
    (row,) = rows
    assert row["state"] == '{"country":"GB"}' and row["has_evidence"] is False
    assert row["label"] is None and row["aug"] == "emptied"  # §5.7/§5.8: never a class from country alone
    assert json.loads(row["gold"])[L.QUESTION_NAME]["probabilities"] == pytest.approx({k: 0.1 for k in L.option_keys("c10")})
    assert stats["evidence_lost"] == 1 and stats["no_evidence"] == 1 and stats["compressed"] == 1
    assert stats["aug"] == {"emptied": 1}
    rows, stats = R.make_rows("train", [({"country": "GB"}, "dining")], **_kw())  # no evidence to begin with
    assert (rows[0]["label"], rows[0]["aug"], stats["evidence_lost"]) == (None, "emptied", 0)


def test_make_rows_eval_row_compressed_to_no_evidence_keeps_true_label_and_is_counted():
    recs, kw = _all_evidence_compressed()
    rows, stats = R.make_rows("val", recs, **kw)
    (row,) = rows
    assert row["label"] == "dining" and row["has_evidence"] is False and row["aug"] == "none"
    assert stats["evidence_lost"] == 1 and stats["no_evidence"] == 1


def _clean_rows():
    rows, _ = R.make_rows("val", _recs()[:2], **_kw())
    return rows


def test_assert_no_leakage_passes_on_clean_rows():
    R.assert_no_leakage(_clean_rows(), KEEP, {"4bf58dd8d48988d1c4941735"}, {"Retail", "Dining and Drinking"})


def test_assert_no_leakage_rejects_unknown_key():
    rows = _clean_rows()
    rows[0] = {**rows[0], "state": S.dumps({"name": "A", "fsq_place_id": "abc"})}
    with pytest.raises(AssertionError, match="fsq_place_id"):
        R.assert_no_leakage(rows, KEEP, set(), set())


def test_assert_no_leakage_rejects_category_id_substring():
    cid = "4bf58dd8d48988d1c4941735"
    rows = _clean_rows()
    rows[1] = {**rows[1], "state": S.dumps({"name": "A", "website": f"x.com/{cid}/menu"})}
    with pytest.raises(AssertionError, match=cid):
        R.assert_no_leakage(rows, KEEP, {cid, "4d4b7105d754a06374d81259"}, set())


def test_assert_no_leakage_rejects_category_name_as_key():
    rows = [{**_clean_rows()[0], "state": S.dumps({"Retail": "yes", "name": "A"})}]
    with pytest.raises(AssertionError, match="Retail"):
        R.assert_no_leakage(rows, KEEP + ["Retail"], set(), {"Retail"})


def test_check_reject_rate():
    R.check_reject_rate({"split": "val", "written": 995, "rejected": 5}, 0.005)
    R.check_reject_rate({"split": "val", "written": 0, "rejected": 0}, 0.005)
    with pytest.raises(ValueError, match="val.*6"):
        R.check_reject_rate({"split": "val", "written": 994, "rejected": 6}, 0.005)


def test_stripped_rows_keep_true_label_and_country_only():
    rows, _ = R.make_rows("test_id", [(r, lab) for r, lab in _recs() if lab][:2] * 3, **_kw())
    out = R.stripped_rows(rows, 4, seed=3)
    assert [r["id"] for r in out] == [f"stripped_test-{i:06d}" for i in range(4)]
    for r in out:
        assert set(r) == ROW_KEYS and r["split"] == "stripped_test"
        assert r["aug"] == "stripped" and r["has_evidence"] is False
        assert json.loads(r["state"]) == {"country": r["country"]}
        assert r["label"] in {"dining", "retail"}  # the TRUE label, for no-evidence metrics
        probs = json.loads(r["gold"])[L.QUESTION_NAME]["probabilities"]
        assert len(set(probs.values())) == 1  # uniform target
    assert out == R.stripped_rows(rows, 4, seed=3)
    assert len(R.stripped_rows(rows, 100, seed=3)) == len(rows)


def test_stripped_rows_pairs_label_with_source_country():
    rows, _ = R.make_rows("test_id", _recs()[:2], **_kw())
    out = R.stripped_rows(rows, 2, seed=0)
    assert sorted((r["country"], r["label"]) for r in out) == [("GB", "dining"), ("US", "retail")]


@pytest.mark.torch
def test_rows_build_items_with_real_tokenizer(en_tokenizer):
    pytest.importorskip("laya")
    from laya_poc.items import build_items
    budget = S.state_budget(en_tokenizer, "c10", 512, 192, margin=8)
    fits = S.make_fits([(en_tokenizer, budget)])
    count = S.token_counter(en_tokenizer)
    recs = [({"country": "GB", "name": "Rosa's Trattoria", "address": "x " * 400}, "dining"),
            ({"country": "JP", "name": "和乃匠 鷲北", "website": "wise-core.com"}, "retail")]
    rows, stats = R.make_rows("val", recs, **_kw(fits=fits, count_tokens=count))
    assert stats["written"] == 2 and stats["compressed"] == 1  # address truncated to fit
    for row in rows:
        items = build_items(en_tokenizer, row, 512, 192)
        assert len(items[0]["markers"]) == 10
        assert count(row["state"]) <= budget
