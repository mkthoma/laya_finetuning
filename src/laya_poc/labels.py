"""Label keys, the Laya question object and gold targets (design doc §5.2, §5.6, §7.5).

The same question object, with the same key order, is used for training, calibration,
evaluation and benchmarking. Never build a question any other way.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

QUESTION_NAME = "place_category"
INSTRUCTIONS = "Which category best describes this place record?"

# FSQ level-1 category name -> 10-class key (Appendix B).
L1_TO_KEY: Mapping[str, str] = MappingProxyType({
    "Arts and Entertainment": "arts",
    "Business and Professional Services": "services",
    "Community and Government": "community",
    "Dining and Drinking": "dining",
    "Event": "event",
    "Health and Medicine": "health",
    "Landmarks and Outdoors": "outdoors",
    "Retail": "retail",
    "Sports and Recreation": "sports",
    "Travel and Transportation": "travel",
})

C10: Mapping[str, str] = MappingProxyType({
    "arts": "museums, theatres, cinemas, galleries, music venues",
    "services": "offices, trades, repairs, agencies, professional services",
    "community": "schools, worship, government offices, civic facilities",
    "dining": "restaurants, cafés, bars, bakeries, takeaways",
    "event": "festivals, markets, conferences, temporary events",
    "health": "clinics, dentists, pharmacies, hospitals, therapists",
    "outdoors": "parks, beaches, monuments, natural features, squares",
    "retail": "shops, supermarkets, boutiques, stores",
    "sports": "gyms, pitches, pools, sports clubs, leisure centres",
    "travel": "hotels, stations, airports, car parks, transport",
})

C7_MAP: Mapping[str, str] = MappingProxyType({
    "arts": "culture", "event": "culture",
    "sports": "leisure", "outdoors": "leisure",
    "community": "civic_services", "services": "civic_services",
    "dining": "dining", "retail": "retail", "health": "health", "travel": "travel",
})

C7: Mapping[str, str] = MappingProxyType({
    "culture": "arts venues, entertainment, festivals and events",
    "leisure": "sport, recreation, parks, landmarks and outdoors",
    "civic_services": "government, community, business and professional services",
    "dining": C10["dining"],
    "retail": C10["retail"],
    "health": C10["health"],
    "travel": C10["travel"],
})

SCHEMES: Mapping[str, Mapping[str, str]] = MappingProxyType({"c10": C10, "c7": C7})

# Boolean-looking keys get followed instead of their descriptions (README); never use them.
BANNED_KEYS = frozenset({"yes", "no", "true", "false", "other"})


def expected_level1_names(extra: tuple[str, ...] | list[str] = ()) -> frozenset[str]:
    """Level-1 names the categories table must contain: the 10 taxonomy names plus known extras
    (release 2026-09-15 also has "Nightlife Spot", which is outside the taxonomy)."""
    return frozenset(L1_TO_KEY) | frozenset(extra)


def criteria(scheme: str) -> dict[str, str]:
    """Option key -> description for a label scheme, in canonical order."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown label scheme {scheme!r}; expected one of {sorted(SCHEMES)}")
    crit = dict(SCHEMES[scheme])
    banned = set(crit) & BANNED_KEYS
    if banned:
        raise ValueError(f"banned option keys in {scheme}: {sorted(banned)}")
    return crit


def option_keys(scheme: str) -> list[str]:
    return list(criteria(scheme))


def question(scheme: str) -> dict:
    """The single `choice` question posed for every record."""
    return {QUESTION_NAME: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria(scheme)}}


def l1_to_key(l1_name: str, scheme: str = "c10") -> str:
    """Map an FSQ level-1 category name to the label key of a scheme."""
    if l1_name not in L1_TO_KEY:
        raise KeyError(f"unknown FSQ level-1 name {l1_name!r}")
    key10 = L1_TO_KEY[l1_name]
    return key10 if scheme == "c10" else collapse_key(key10, scheme)


def collapse_key(key10: str, scheme: str) -> str:
    if scheme == "c10":
        return key10
    if scheme == "c7":
        return C7_MAP[key10]
    raise ValueError(f"unknown label scheme {scheme!r}")


def gold_probabilities(label_key: str | None, keys: list[str], smoothing: float) -> dict[str, float]:
    """Target distribution: label-smoothed one-hot, or uniform when there is no evidence (§5.7)."""
    n = len(keys)
    if n < 2:
        raise ValueError("need at least two options")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
    if label_key is None:
        return {k: 1.0 / n for k in keys}
    if label_key not in keys:
        raise KeyError(f"label {label_key!r} not in options {keys}")
    off = smoothing / (n - 1)
    return {k: (1.0 - smoothing) if k == label_key else off for k in keys}


def gold(label_key: str | None, scheme: str, smoothing: float) -> dict:
    """Typed-decisions `gold` object for the single question."""
    keys = option_keys(scheme)
    probs = gold_probabilities(label_key, keys, smoothing)
    top = max(probs.values())
    # For the uniform (no-evidence) target every option ties; report the first key in canonical order.
    label = label_key if label_key is not None else keys[0]
    return {QUESTION_NAME: {"type": "choice", "label": label, "confidence": top, "probabilities": probs}}
