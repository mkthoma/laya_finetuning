import copy
from collections import Counter

import pytest

from laya_poc.augment import augment_epoch

EVIDENCE = ["name", "address", "locality", "region", "postcode", "tel", "website", "email"]
LABELS = ["dining", "retail", "health", "travel"]
ZERO = {"field_dropout": 0.0, "name_dropout": 0.0, "stripped_rate": 0.0, "conflict_rate": 0.0}


def _records(n=200):
    out = []
    for i in range(n):
        rec = {"country": "GB", "locality": f"loc-{i}", "name": f"name-{i}", "tel": f"tel-{i}"}
        if i % 3 == 0:
            rec["website"] = f"site-{i}.com"
        out.append((dict(sorted(rec.items())), LABELS[i % len(LABELS)]))
    return out


def _cfg(**kw):
    return {**ZERO, **kw}


def test_no_augmentation_returns_same_records_shuffled():
    recs = _records()
    out = augment_epoch(recs, _cfg(), EVIDENCE, seed=1, epoch=0)
    assert len(out) == len(recs)
    assert {t for _, _, t in out} == {"none"}
    as_set = sorted((tuple(r.items()), lab) for r, lab, _ in out)
    assert as_set == sorted((tuple(r.items()), lab) for r, lab in recs)
    assert [r for r, _, _ in out] != [r for r, _ in recs]  # order shuffled


def test_deterministic_for_seed_and_epoch():
    recs = _records()
    cfg = _cfg(field_dropout=0.4, name_dropout=0.1, stripped_rate=0.07, conflict_rate=0.05)
    a = augment_epoch(recs, cfg, EVIDENCE, seed=11, epoch=2)
    b = augment_epoch(recs, cfg, EVIDENCE, seed=11, epoch=2)
    c = augment_epoch(recs, cfg, EVIDENCE, seed=11, epoch=3)
    assert a == b
    assert a != c


def test_does_not_mutate_inputs():
    recs = _records()
    snapshot = copy.deepcopy(recs)
    cfg = _cfg(field_dropout=0.9, name_dropout=0.9, stripped_rate=0.5, conflict_rate=0.9)
    out = augment_epoch(recs, cfg, EVIDENCE, seed=0, epoch=0)
    assert recs == snapshot
    ids = {id(r) for r, _ in recs}
    assert not any(id(r) in ids for r, _, _ in out)


def test_stripped_copies_keep_only_country_with_no_label():
    recs = _records(200)
    out = augment_epoch(recs, _cfg(stripped_rate=0.07), EVIDENCE, seed=0, epoch=0)
    stripped = [(r, lab) for r, lab, t in out if t == "stripped"]
    assert len(stripped) == round(0.07 * 200)
    assert len(out) == 200 + len(stripped)
    assert all(r == {"country": "GB"} and lab is None for r, lab in stripped)


def test_name_dropout_removes_name_and_keeps_label():
    out = augment_epoch(_records(), _cfg(name_dropout=1.0), EVIDENCE, seed=0, epoch=0)
    assert all("name" not in r for r, _, _ in out)
    assert {t for _, _, t in out} == {"name_dropout"}
    assert all(lab is not None for _, lab, _ in out)


def test_field_dropout_keeps_name_and_country():
    out = augment_epoch(_records(), _cfg(field_dropout=1.0), EVIDENCE, seed=0, epoch=0)
    assert all(set(r) == {"country", "name"} for r, _, _ in out)
    assert {t for _, _, t in out} == {"dropout"}


def test_everything_dropped_is_emptied_with_no_label():
    out = augment_epoch(_records(), _cfg(field_dropout=1.0, name_dropout=1.0), EVIDENCE, seed=0, epoch=0)
    assert all(r == {"country": "GB"} for r, _, _ in out)
    assert {t for _, _, t in out} == {"emptied"}
    assert all(lab is None for _, lab, _ in out)


def test_conflict_copies_one_field_from_a_different_class():
    recs = _records()
    owner = {}  # value -> label of the record it came from
    for r, lab in recs:
        for k, v in r.items():
            if k not in ("name", "country"):
                owner[v] = lab
    out = augment_epoch(recs, _cfg(conflict_rate=1.0), EVIDENCE, seed=0, epoch=0)
    assert {t for _, _, t in out} == {"conflict"}
    originals = {r["name"]: (r, lab) for r, lab in recs}
    for r, lab, _ in out:
        orig, orig_label = originals[r["name"]]
        assert lab == orig_label  # conflicts keep the true label
        changed = [k for k in set(r) | set(orig) if r.get(k) != orig.get(k)]
        assert len(changed) == 1 and changed[0] not in ("name", "country")
        assert owner[r[changed[0]]] != lab  # donor has a different class


def test_conflict_tag_takes_precedence_over_dropouts():
    cfg = _cfg(conflict_rate=1.0, name_dropout=1.0, field_dropout=0.5)
    out = augment_epoch(_records(), cfg, EVIDENCE, seed=0, epoch=0)
    assert {t for _, _, t in out} == {"conflict"}


def test_name_dropout_tag_beats_field_dropout():
    out = augment_epoch(_records(), _cfg(name_dropout=1.0, field_dropout=0.5), EVIDENCE, seed=0, epoch=0)
    tags = Counter(t for _, _, t in out)
    assert set(tags) == {"name_dropout", "emptied"}
    assert tags["name_dropout"] > tags["emptied"]


def test_conflict_skipped_when_only_one_class():
    recs = [(r, "dining") for r, _ in _records(20)]
    out = augment_epoch(recs, _cfg(conflict_rate=1.0), EVIDENCE, seed=0, epoch=0)
    assert {t for _, _, t in out} == {"none"}


def test_rates_are_roughly_respected():
    recs = _records(4000)
    cfg = _cfg(field_dropout=0.4, name_dropout=0.1, stripped_rate=0.07, conflict_rate=0.05)
    out = augment_epoch(recs, cfg, EVIDENCE, seed=5, epoch=0)
    kept = [r for r, _, t in out if t != "stripped"]
    no_name = sum("name" not in r for r in kept) / len(kept)
    assert 0.08 < no_name < 0.12
    tel_kept = sum("tel" in r for r in kept) / len(kept)
    assert 0.55 < tel_kept < 0.66  # dropout 0.4, plus a few conflict re-additions
    assert 0.03 < Counter(t for _, _, t in out)["conflict"] / len(kept) < 0.07


def test_empty_input_and_invalid_rates():
    assert augment_epoch([], _cfg(stripped_rate=0.5), EVIDENCE, seed=0, epoch=0) == []
    with pytest.raises(ValueError, match="field_dropout"):
        augment_epoch(_records(4), _cfg(field_dropout=1.5), EVIDENCE, seed=0, epoch=0)
    with pytest.raises(KeyError):
        augment_epoch(_records(4), {"field_dropout": 0.1}, EVIDENCE, seed=0, epoch=0)
