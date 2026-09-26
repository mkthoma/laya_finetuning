"""Train-only augmentation, regenerated per epoch (design doc §5.7, §5.8).

Pure and deterministic for (seed, epoch): `random.Random(seed * 1000 + epoch)` drives every
draw, in a fixed order, so a resumed run rebuilds exactly the same epoch.
"""
from __future__ import annotations

import random
from typing import Iterable, Mapping, Sequence

from .serialise import has_evidence

PROTECTED = ("name", "country")  # never dropped by field dropout, never copied by conflicts
RATE_KEYS = ("field_dropout", "name_dropout", "stripped_rate", "conflict_rate")

Record = dict
Labelled = tuple[Record, str]
Augmented = tuple[Record, str | None, str]


def epoch_rng(seed: int, epoch: int) -> random.Random:
    return random.Random(seed * 1000 + epoch)


def _rates(aug_cfg: Mapping[str, float]) -> dict[str, float]:
    rates = {k: float(aug_cfg[k]) for k in RATE_KEYS}
    bad = {k: v for k, v in rates.items() if not 0.0 <= v <= 1.0}
    if bad:
        raise ValueError(f"augment rates must be in [0, 1]: {bad}")
    return rates


def _conflict_fields(rec: Mapping[str, str]) -> list[str]:
    return sorted(k for k in rec if k not in PROTECTED)


def _donor_pools(records: Sequence[Labelled]) -> dict[str, list[int]]:
    """label -> indices of records of OTHER classes that have at least one copyable field."""
    usable = [i for i, (rec, _) in enumerate(records) if _conflict_fields(rec)]
    labels = sorted({lab for _, lab in records})
    return {lab: [i for i in usable if records[i][1] != lab] for lab in labels}


def _augment_one(rec: Record, label: str, records: Sequence[Labelled], donors: Mapping[str, list[int]],
                 rates: Mapping[str, float], evidence_fields: Iterable[str],
                 rng: random.Random) -> tuple[Record, str | None, str]:
    out = dict(rec)
    tags = set()
    if rng.random() < rates["name_dropout"] and "name" in out:
        out.pop("name")
        tags.add("name_dropout")
    for field in sorted(k for k in rec if k not in PROTECTED):
        if rng.random() < rates["field_dropout"]:
            out.pop(field, None)
            tags.add("dropout")
    if rng.random() < rates["conflict_rate"] and donors.get(label):
        donor = records[rng.choice(donors[label])][0]
        field = rng.choice(_conflict_fields(donor))
        out[field] = donor[field]
        tags.add("conflict")
    if not has_evidence(out, evidence_fields):
        return out, None, "emptied"
    for tag in ("conflict", "name_dropout", "dropout"):  # precedence when several apply
        if tag in tags:
            return out, label, tag
    return out, label, "none"


def augment_epoch(records: list[Labelled], aug_cfg: Mapping[str, float], evidence_fields: Iterable[str],
                  seed: int, epoch: int) -> list[Augmented]:
    """(record, label or None, aug tag) for one epoch: per-record augmentations, then stripped
    copies (country only, uniform target), shuffled. Inputs are never mutated."""
    rates = _rates(aug_cfg)
    evidence_fields = list(evidence_fields)
    rng = epoch_rng(seed, epoch)
    donors = _donor_pools(records)
    out = [_augment_one(rec, lab, records, donors, rates, evidence_fields, rng) for rec, lab in records]
    n_stripped = round(rates["stripped_rate"] * len(records))
    for i in rng.sample(range(len(records)), n_stripped):
        rec = records[i][0]
        out.append(({"country": rec["country"]} if "country" in rec else {}, None, "stripped"))
    rng.shuffle(out)
    return out
