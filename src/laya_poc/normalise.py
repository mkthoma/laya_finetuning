"""String normalisation for dedup keys and brand detection (design doc §5.3, §7.4.4).

These keys never reach the model; they only decide which rows share a split.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

# Legal-form suffixes stripped from names (applied after punctuation removal).
_SUFFIX = re.compile(r"\b(ltd|limited|inc|llc|gmbh|sa|sas|srl|spa|plc|co|corp|bv|ltda|pt|kk)\b")
_WS = re.compile(r"\s+")


def _strip_punct(s: str) -> str:
    # The doc uses [^\w\s]; Python's \w excludes combining marks (Thai vowels, Devanagari matras),
    # which would shred non-Latin names. Replace only Unicode punctuation (P*) and symbols (S*).
    return "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in s)


def norm(s: Any) -> str:
    """Lower-case, NFKC, strip punctuation and legal suffixes, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = _SUFFIX.sub(" ", _strip_punct(s))
    return _WS.sub(" ", s).strip()


def clean(v: Any) -> str | None:
    """Field value -> trimmed single-spaced string, or None when missing/empty."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        import pandas as pd  # local import: pandas NA types
        if v is pd.NA or v is pd.NaT:
            return None
    except ImportError:  # pragma: no cover
        pass
    out = _WS.sub(" ", str(v)).strip()
    return out or None
