"""Split construction: brand holdout, dedup, stratified train, OOD pools (design doc §5.3, §7.4.4).

Input is the labelled pool (one row per place, `label` = FSQ level-1 name, `n_l1` = number of
distinct level-1 names over the place's categories). All sampling is seeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .normalise import norm

SPLIT_ORDER = ("train", "val", "test_id", "ood_country", "ood_script", "ood_brand")


@dataclass(frozen=True)
class SplitSpec:
    id_countries: tuple[str, ...]
    ood_country: tuple[str, ...]
    ood_script: str
    train_size: int
    train_cap_per_class: int
    val_size: int
    test_size: int
    ood_size: int
    brand_top_n: int
    max_rows_per_name_country: int
    seed: int
    strict: bool = True  # smoke mode sets False: small/missing pools warn instead of raising
    warnings: list[str] = field(default_factory=list, compare=False)


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalised name/locality and the cross-split dedup key."""
    out = df.copy()
    out["nname"] = out["name"].map(norm)
    out["nloc"] = out["locality"].map(norm) if "locality" in out else ""
    out["key"] = out["nname"] + "|" + out["nloc"] + "|" + out["country"].astype(str)
    return out


def top_brands(id_pool: pd.DataFrame, n: int) -> set[str]:
    """The n most frequent normalised names in the ID pool (ties broken by name for determinism)."""
    counts = id_pool.loc[id_pool["nname"] != "", "nname"].value_counts()
    ranked = counts.reset_index().sort_values(["count", "nname"], ascending=[False, True])
    return set(ranked["nname"].head(n))


def cap_per_name_country(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Keep at most k rows per (nname, country) so chains cannot dominate."""
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled[shuffled.groupby(["nname", "country"]).cumcount() < k].sort_index()


def natural_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample at natural prevalence (without replacement)."""
    return df.sample(n=min(n, len(df)), random_state=seed)


def stratified_train(df: pd.DataFrame, cap_per_class: int, total: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(n=min(len(g), cap_per_class), random_state=seed) for _, g in df.groupby("label", sort=True)]
    train = pd.concat(parts) if parts else df.iloc[:0]
    return train.sample(n=total, random_state=seed) if len(train) > total else train


def _check_size(spec: SplitSpec, name: str, got: int, want: int) -> None:
    if got >= want:
        return
    msg = f"split {name}: {got} rows < requested {want}"
    # Train is "about" train_size by design: per-class caps can leave it short (small classes
    # contribute everything they have), so it only warns. Evaluation splits must be full.
    if spec.strict and name != "train":
        raise ValueError(msg)
    spec.warnings.append(msg)


def make_splits(pool: pd.DataFrame, spec: SplitSpec) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """Build every split from the labelled pool. Returns (splits, brand set)."""
    df = add_keys(pool[pool["n_l1"] == 1])
    idp = df[df["country"].isin(spec.id_countries)]
    brands = top_brands(idp, spec.brand_top_n)
    brand_pool = idp[idp["nname"].isin(brands)]
    idp = idp[~idp["nname"].isin(brands)]
    idp = cap_per_name_country(idp, spec.max_rows_per_name_country, spec.seed)
    idp = idp.drop_duplicates("key")

    test_id = natural_sample(idp, spec.test_size, spec.seed)
    rest = idp.drop(test_id.index)
    val = natural_sample(rest, spec.val_size, spec.seed)
    rest = rest.drop(val.index)
    train = stratified_train(rest, spec.train_cap_per_class, spec.train_size, spec.seed)

    per_country = spec.ood_size // max(1, len(spec.ood_country))
    ood_parts = [natural_sample(df[df["country"] == c], per_country, spec.seed) for c in spec.ood_country]
    splits = {
        "train": train,
        "val": val,
        "test_id": test_id,
        "ood_country": pd.concat(ood_parts) if ood_parts else df.iloc[:0],
        "ood_script": natural_sample(df[df["country"] == spec.ood_script], spec.ood_size, spec.seed),
        "ood_brand": natural_sample(brand_pool, spec.ood_size, spec.seed),
    }
    wanted = {"train": spec.train_size, "val": spec.val_size, "test_id": spec.test_size,
              "ood_country": per_country * len(spec.ood_country), "ood_script": spec.ood_size,
              "ood_brand": spec.ood_size}
    for name in SPLIT_ORDER:
        _check_size(spec, name, len(splits[name]), wanted[name])
    assert_splits(splits, brands, spec)
    return splits, brands


def assert_splits(splits: dict[str, pd.DataFrame], brands: set[str], spec: SplitSpec) -> None:
    """Leakage and composition assertions (§5.3 dedup rules 1-3)."""
    names = [n for n in SPLIT_ORDER if n in splits]
    keys = {n: set(splits[n]["key"]) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = keys[a] & keys[b]
            if overlap:
                raise AssertionError(f"key overlap {a}/{b}: {len(overlap)} keys, e.g. {sorted(overlap)[:3]}")
    for n in ("train", "val", "test_id"):
        if n in splits and set(splits[n]["nname"]) & brands:
            raise AssertionError(f"brand leak in {n}")
    train = splits.get("train")
    if train is not None and len(train):
        extra = set(train["country"]) - set(spec.id_countries)
        if extra:
            raise AssertionError(f"non-ID countries in train: {sorted(extra)}")
        if train.groupby("label").size().max() > spec.train_cap_per_class:
            raise AssertionError("train class cap exceeded")
        per_name = train.groupby(["nname", "country"]).size().max()
        if per_name > spec.max_rows_per_name_country:
            raise AssertionError(f"train has {per_name} rows for one (name, country)")
