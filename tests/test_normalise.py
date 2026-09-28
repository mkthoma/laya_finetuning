import math

import pandas as pd
import pytest

from laya_poc.normalise import clean, norm


@pytest.mark.parametrize("raw,expected", [
    ("  Joe's  PIZZA, Ltd. ", "joe s pizza"),
    ("Starbucks Coffee Co.", "starbucks coffee"),
    ("ＡＢＣ　Café", "abc café"),          # NFKC folds full-width forms
    ("Müller GmbH & Co. KG", "müller kg"),
    ("", ""),
])
def test_norm_examples(raw, expected):
    assert norm(raw) == expected


def test_norm_keeps_thai_combining_marks_intact():
    name = "ร้านกาแฟ"  # contains combining vowel/tone marks
    assert norm(name) == name
    assert " " not in norm(name)


def test_norm_non_string_is_empty():
    assert norm(None) == "" and norm(float("nan")) == "" and norm(12) == ""


def test_norm_does_not_strip_suffix_inside_words():
    assert norm("Costa Coffee") == "costa coffee"
    assert norm("Spartan Gym") == "spartan gym"


@pytest.mark.parametrize("v", [None, float("nan"), "", "   ", pd.NA])
def test_clean_missing_values_become_none(v):
    assert clean(v) is None


def test_clean_collapses_whitespace_and_stringifies():
    assert clean("  12\n High   St ") == "12 High St"
    assert clean(42) == "42"
    assert not math.isnan(len(clean("x")))
