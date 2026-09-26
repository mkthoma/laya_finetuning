import math

import pytest

from laya_poc import labels as L


def test_c10_has_ten_semantic_keys_in_canonical_order():
    keys = L.option_keys("c10")
    assert keys == ["arts", "services", "community", "dining", "event",
                    "health", "outdoors", "retail", "sports", "travel"]
    assert not set(keys) & L.BANNED_KEYS


def test_c7_collapse_covers_every_c10_key_and_yields_seven_classes():
    assert set(L.C7_MAP) == set(L.C10)
    assert set(L.C7_MAP.values()) == set(L.C7)
    assert len(L.C7) == 7


def test_every_l1_name_maps_to_a_c10_key():
    assert sorted(L.L1_TO_KEY.values()) == sorted(L.C10)
    assert L.l1_to_key("Dining and Drinking") == "dining"
    assert L.l1_to_key("Event", "c7") == "culture"
    assert L.l1_to_key("Landmarks and Outdoors", "c7") == "leisure"


def test_unknown_l1_name_raises():
    with pytest.raises(KeyError):
        L.l1_to_key("Events")


def test_descriptions_stay_short():
    # Proxy for the <=10-token rule; the notebook re-checks with the real tokenizer.
    for scheme in ("c10", "c7"):
        for desc in L.criteria(scheme).values():
            assert len(desc.split()) <= 10, desc


def test_question_shape_is_one_choice_question():
    q = L.question("c10")
    assert list(q) == [L.QUESTION_NAME]
    body = q[L.QUESTION_NAME]
    assert body["type"] == "choice"
    assert body["instructions"] == L.INSTRUCTIONS
    assert list(body["criteria"]) == L.option_keys("c10")


def test_question_returns_independent_copies():
    q1 = L.question("c10")
    q1[L.QUESTION_NAME]["criteria"]["arts"] = "mutated"
    assert L.question("c10")[L.QUESTION_NAME]["criteria"]["arts"] != "mutated"


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        L.question("c12")


def test_expected_level1_names_include_extras():
    names = L.expected_level1_names(["Nightlife Spot"])
    assert len(names) == 11 and "Nightlife Spot" in names and "Event" in names


def test_smoothed_gold_puts_0_9_on_true_class_and_sums_to_one():
    g = L.gold("dining", "c10", 0.1)[L.QUESTION_NAME]
    p = g["probabilities"]
    assert list(p) == L.option_keys("c10")
    assert p["dining"] == pytest.approx(0.9)
    assert p["arts"] == pytest.approx(0.1 / 9)
    assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-9)
    assert g["label"] == "dining" and g["confidence"] == pytest.approx(0.9)


def test_one_hot_gold_when_smoothing_zero():
    p = L.gold("retail", "c7", 0.0)[L.QUESTION_NAME]["probabilities"]
    assert p["retail"] == 1.0 and sum(v for k, v in p.items() if k != "retail") == 0.0


def test_no_evidence_gold_is_uniform():
    g = L.gold(None, "c10", 0.1)[L.QUESTION_NAME]
    assert all(v == pytest.approx(0.1) for v in g["probabilities"].values())
    assert g["label"] in L.option_keys("c10")


def test_gold_rejects_label_outside_scheme():
    with pytest.raises(KeyError):
        L.gold("culture", "c10", 0.1)


@pytest.mark.parametrize("bad", [-0.1, 1.0])
def test_gold_rejects_bad_smoothing(bad):
    with pytest.raises(ValueError):
        L.gold_probabilities("arts", L.option_keys("c10"), bad)
