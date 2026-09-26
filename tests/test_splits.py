import itertools

import pandas as pd
import pytest

from laya_poc import splits as S

LABELS = ["Dining and Drinking", "Retail", "Health and Medicine", "Travel and Transportation"]


def _pool(n_per_country=400, countries=("GB", "US", "DE", "NL", "PL", "MX", "TH"), seed=0):
    rows = []
    rng = pd.Series(range(10_000)).sample(frac=1.0, random_state=seed).tolist()
    it = itertools.count()
    for c in countries:
        for i in range(n_per_country):
            j = next(it)
            label = LABELS[j % len(LABELS)]
            # 10% of rows belong to two recurring chains -> brands
            if i % 10 == 0:
                name = "Mega Mart Ltd" if i % 20 == 0 else "Quick Cafe"
            else:
                name = f"Place {rng[j]}"
            rows.append({"fsq_place_id": f"id{j}", "name": name, "locality": f"Town{i % 50}",
                         "country": c, "label": label, "n_l1": 1})
    rows.append({"fsq_place_id": "mixed", "name": "Mixed Co", "locality": "X", "country": "GB",
                 "label": LABELS[0], "n_l1": 2})
    return pd.DataFrame(rows)


def _spec(**kw):
    base = dict(id_countries=("GB", "US", "DE"), ood_country=("NL", "PL", "MX"), ood_script="TH",
                train_size=600, train_cap_per_class=200, val_size=100, test_size=100, ood_size=90,
                brand_top_n=2, max_rows_per_name_country=5, seed=20260925)
    base.update(kw)
    return S.SplitSpec(**base)


def test_add_keys_builds_normalised_dedup_key():
    df = S.add_keys(pd.DataFrame([{"name": "Joe's Pizza Ltd", "locality": "  Leeds ", "country": "GB"}]))
    assert df.loc[0, "key"] == "joe s pizza|leeds|GB"


def test_make_splits_sizes_and_no_overlap():
    splits, brands = S.make_splits(_pool(), _spec())
    assert len(splits["train"]) == 600
    assert len(splits["val"]) == 100 and len(splits["test_id"]) == 100
    assert len(splits["ood_country"]) == 90 and len(splits["ood_script"]) == 90
    assert brands == {"mega mart", "quick cafe"}
    for a, b in itertools.combinations(S.SPLIT_ORDER, 2):
        assert not set(splits[a]["key"]) & set(splits[b]["key"])


def test_brands_only_in_ood_brand_and_mixed_label_rows_excluded():
    splits, brands = S.make_splits(_pool(), _spec())
    for n in ("train", "val", "test_id"):
        assert not set(splits[n]["nname"]) & brands
    assert set(splits["ood_brand"]["nname"]) <= brands
    all_ids = pd.concat(splits.values())["fsq_place_id"]
    assert "mixed" not in set(all_ids)


def test_train_respects_class_cap_and_id_countries():
    spec = _spec(train_cap_per_class=120, train_size=10_000)
    splits, _ = S.make_splits(_pool(), spec)
    tr = splits["train"]
    assert len(tr) == 120 * len(LABELS)
    assert tr.groupby("label").size().max() <= 120
    assert set(tr["country"]) <= {"GB", "US", "DE"}
    assert any("split train" in w for w in spec.warnings)  # short train only warns


def test_splits_are_deterministic_for_a_seed():
    a, _ = S.make_splits(_pool(), _spec())
    b, _ = S.make_splits(_pool(), _spec())
    for n in S.SPLIT_ORDER:
        assert list(a[n]["fsq_place_id"]) == list(b[n]["fsq_place_id"])


def test_strict_mode_raises_when_pool_too_small():
    with pytest.raises(ValueError, match="ood_script"):
        S.make_splits(_pool(countries=("GB", "US", "DE", "NL", "PL", "MX")), _spec())


def test_lenient_mode_records_warning_for_missing_pool():
    spec = _spec(strict=False)
    splits, _ = S.make_splits(_pool(countries=("GB", "US", "DE", "NL", "PL", "MX")), spec)
    assert len(splits["ood_script"]) == 0
    assert any("ood_script" in w for w in spec.warnings)


def test_cap_per_name_country_limits_chain_rows():
    df = S.add_keys(pd.DataFrame([{"name": "Chain", "locality": f"L{i}", "country": "GB"} for i in range(12)]))
    assert len(S.cap_per_name_country(df, 5, seed=1)) == 5


def test_assert_splits_detects_key_overlap():
    df = S.add_keys(pd.DataFrame([{"name": "A", "locality": "B", "country": "GB", "label": "Retail"}]))
    with pytest.raises(AssertionError, match="key overlap"):
        S.assert_splits({"train": df, "val": df}, set(), _spec())


def test_assert_splits_detects_brand_leak():
    df = S.add_keys(pd.DataFrame([{"name": "Mega Mart", "locality": "B", "country": "GB", "label": "Retail"}]))
    with pytest.raises(AssertionError, match="brand leak"):
        S.assert_splits({"val": df}, {"mega mart"}, _spec())
