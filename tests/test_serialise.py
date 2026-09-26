import copy
import json
import math
import random

import pandas as pd
import pytest

from laya_poc import serialise as S

KEEP = ["name", "address", "locality", "region", "postcode", "country", "tel", "website", "email",
        "facebook_id", "instagram", "twitter"]
EVIDENCE = [f for f in KEEP if f != "country"]


def test_make_record_cleans_orders_and_drops_missing():
    row = {"name": "  Rosa's   Trattoria ", "country": "GB", "tel": None, "email": "",
           "locality": float("nan"), "region": pd.NA, "postcode": "LS1 4AP", "latitude": 53.8}
    rec = S.make_record(row, KEEP)
    assert rec == {"country": "GB", "name": "Rosa's Trattoria", "postcode": "LS1 4AP"}
    assert list(rec) == sorted(rec)  # alphabetical by default
    assert "latitude" not in rec  # only keep_fields survive


@pytest.mark.parametrize("raw,expected", [
    ("http://www.example.com/menu", "example.com/menu"),
    ("https://example.com", "example.com"),
    ("www.example.com", "example.com"),
    ("HTTPS://WWW.Example.com", "Example.com"),
    ("example.com", "example.com"),
])
def test_make_record_strips_website_scheme_and_www(raw, expected):
    assert S.make_record({"website": raw}, KEEP) == {"website": expected}


def test_make_record_drops_website_that_is_only_a_scheme():
    assert S.make_record({"website": "http://www.", "name": "x"}, KEEP) == {"name": "x"}


def test_make_record_keeps_facebook_id_string_exact():
    big = "122132281766001389"  # > 2**53: a float would lose digits
    assert S.make_record({"facebook_id": big}, KEEP) == {"facebook_id": big}


def test_make_record_rejects_float_facebook_id():
    with pytest.raises(TypeError, match="facebook_id"):
        S.make_record({"facebook_id": 1.2213228176600139e17}, KEEP)


def test_make_record_accepts_pandas_row_and_does_not_mutate():
    row = pd.Series({"name": " A ", "country": "US", "tel": None})
    before = row.copy()
    assert S.make_record(row, KEEP) == {"country": "US", "name": "A"}
    pd.testing.assert_series_equal(row, before)
    d = {"name": " A ", "website": "http://a.com"}
    snapshot = copy.deepcopy(d)
    S.make_record(d, KEEP)
    assert d == snapshot


def test_make_record_shuffle_is_seeded_and_keeps_content():
    row = {f: f"v{i}" for i, f in enumerate(KEEP)}
    a = S.make_record(row, KEEP, rng=random.Random(3), shuffle=True)
    b = S.make_record(row, KEEP, rng=random.Random(3), shuffle=True)
    c = S.make_record(row, KEEP, rng=random.Random(4), shuffle=True)
    assert list(a) == list(b)
    assert a == c and list(a) != list(c)
    assert list(a) != sorted(a)


def test_make_record_shuffle_requires_rng():
    with pytest.raises(ValueError, match="rng"):
        S.make_record({"name": "x"}, KEEP, shuffle=True)


def test_order_fields_returns_new_dict():
    rec = {"b": "1", "a": "2"}
    out = S.order_fields(rec)
    assert list(out) == ["a", "b"] and list(rec) == ["b", "a"]


@pytest.mark.parametrize("rec,expected", [
    ({"country": "GB"}, False),
    ({}, False),
    ({"country": "GB", "name": "A"}, True),
    ({"country": "GB", "tel": "1"}, True),
    ({"country": "GB", "unknown_field": "1"}, False),
])
def test_has_evidence_country_alone_is_not_evidence(rec, expected):
    assert S.has_evidence(rec, EVIDENCE) is expected


def test_has_evidence_ignores_country_even_if_listed():
    assert S.has_evidence({"country": "GB"}, EVIDENCE + ["country"]) is False


def test_dumps_is_compact_non_ascii_and_order_preserving():
    s = S.dumps({"name": "和乃匠 鷲北", "country": "JP"})
    assert s == '{"name":"和乃匠 鷲北","country":"JP"}'


def _len_fits(limit):
    return lambda s: len(s) <= limit


def test_fit_state_returns_uncompressed_when_it_fits():
    rec = {"country": "GB", "name": "A"}
    assert S.fit_state(rec, _len_fits(1000), ["twitter"], 120) == (S.dumps(rec), False)


def test_fit_state_drops_fields_one_at_a_time_in_order():
    rec = {"country": "GB", "instagram": "i" * 20, "name": "A", "twitter": "t" * 20}
    limit = len(S.dumps({k: v for k, v in rec.items() if k != "twitter"}))
    state, compressed = S.fit_state(rec, _len_fits(limit), ["twitter", "instagram", "email"], 120)
    assert compressed is True
    assert json.loads(state) == {"country": "GB", "instagram": "i" * 20, "name": "A"}


def test_fit_state_truncates_address_last():
    rec = {"address": "x" * 300, "country": "GB", "email": "e@x.com", "name": "A"}
    limit = len(S.dumps({"address": "x" * 120, "country": "GB", "name": "A"}))
    state, compressed = S.fit_state(rec, _len_fits(limit), ["twitter", "email"], 120)
    assert compressed and json.loads(state) == {"address": "x" * 120, "country": "GB", "name": "A"}
    assert list(json.loads(state)) == ["address", "country", "name"]  # key order preserved


def test_fit_state_rejects_when_still_too_long_and_is_pure():
    rec = {"address": "x" * 300, "country": "GB", "name": "N" * 500, "twitter": "t"}
    snapshot = copy.deepcopy(rec)
    assert S.fit_state(rec, _len_fits(200), ["twitter"], 120) == (None, True)
    assert rec == snapshot


def test_fit_state_reject_without_any_compression_possible():
    rec = {"country": "GB", "name": "N" * 500}
    assert S.fit_state(rec, _len_fits(50), ["twitter"], 120) == (None, False)


class _WordTok:
    """Whitespace 'tokenizer' with the attributes items.count_state_tokens needs."""
    mask_token = "[MASK]"

    def __call__(self, text, **_):
        return {"input_ids": text.replace('"', " ").split()}


def test_make_fits_requires_every_budget():
    pytest.importorskip("laya")
    tok = _WordTok()
    state = S.dumps({"name": "a b c d"})
    n = len(tok(state)["input_ids"])
    assert S.make_fits([(tok, n), (tok, n + 5)])(state) is True
    assert S.make_fits([(tok, n + 5), (tok, n - 1)])(state) is False


def test_make_fits_rejects_bad_budgets():
    with pytest.raises(ValueError):
        S.make_fits([])
    with pytest.raises(ValueError, match="budget"):
        S.make_fits([(_WordTok(), 0)])


def test_token_counter_is_cached_and_counts():
    pytest.importorskip("laya")
    calls = []

    class Counting(_WordTok):
        def __call__(self, text, **kw):
            calls.append(text)
            return super().__call__(text, **kw)

    count = S.token_counter(Counting())
    assert count("a b") == count("a b") == 2
    assert len(calls) == 1


@pytest.mark.torch
def test_state_budget_matches_exact_room_minus_margin(en_tokenizer):
    # critique.md §0: C10 on the EN tokenizer leaves 363 state tokens at 512/192.
    assert S.state_budget(en_tokenizer, "c10", 512, 192, margin=8) == 363 - 8
    assert S.state_budget(en_tokenizer, "c10", 1024, 256, margin=0) == 875


@pytest.mark.torch
def test_make_fits_with_real_tokenizer(en_tokenizer):
    fits = S.make_fits([(en_tokenizer, 20)])
    assert fits(S.dumps({"country": "GB", "name": "Cafe"}))
    assert not fits(S.dumps({"name": "word " * 50}))


def test_nan_float_values_are_missing_not_text():
    assert S.make_record({"postcode": math.nan, "name": "A"}, KEEP) == {"name": "A"}
