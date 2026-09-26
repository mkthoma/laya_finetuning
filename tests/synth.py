"""Synthetic FSQ-shaped records and typed-decisions rows for tests (no gated data needed).

Names carry class words so a tiny model can learn something in a few steps; other fields are
sparse, as in the real data.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from laya_poc import labels as L
from laya_poc.io_utils import write_jsonl

_WORDS = {
    "arts": ["Museum", "Theatre", "Cinema", "Gallery"], "services": ["Plumbing", "Law Office", "Agency", "Repairs"],
    "community": ["School", "Church", "Town Hall", "Library"], "dining": ["Pizza", "Cafe", "Bar", "Bakery"],
    "event": ["Festival", "Christmas Market", "Conference", "Fair"], "health": ["Clinic", "Dental", "Pharmacy", "Hospital"],
    "outdoors": ["Park", "Beach", "Monument", "Square"], "retail": ["Supermarket", "Boutique", "Store", "Shop"],
    "sports": ["Gym", "Pool", "Football Club", "Leisure Centre"], "travel": ["Hotel", "Station", "Airport", "Car Park"],
}
_COUNTRIES = ["GB", "US", "DE", "FR", "ES", "IT", "BR", "IN", "JP", "ID"]


def synthetic_records(n: int, seed: int = 0) -> list[tuple[dict, str]]:
    rng = random.Random(seed)
    keys = L.option_keys("c10")
    out = []
    for i in range(n):
        label = keys[i % len(keys)]
        rec = {"country": rng.choice(_COUNTRIES), "name": f"{rng.choice(['Rosa', 'Sun', 'Blue', 'Kings'])} "
                                                           f"{rng.choice(_WORDS[label])} {i}"}
        if rng.random() < 0.7:
            rec["locality"] = rng.choice(["Leeds", "Austin", "Berlin", "Lyon", "Osaka"])
        if rng.random() < 0.5:
            rec["tel"] = f"0{rng.randint(100, 999)} {rng.randint(100000, 999999)}"
        if rng.random() < 0.3:
            rec["website"] = f"example{i}.com"
        out.append((dict(sorted(rec.items())), label))
    return out


def rows_from_records(records: list[tuple[dict, str | None]], split: str, scheme: str = "c10",
                      smoothing: float = 0.1) -> list[dict]:
    q = json.dumps(L.question(scheme), ensure_ascii=False)
    return [{"id": f"{split}-{i:06d}", "split": split,
             "state": json.dumps(rec, ensure_ascii=False, separators=(",", ":")),
             "questions": q,
             "gold": json.dumps(L.gold(label, scheme, smoothing), ensure_ascii=False),
             "label": label}
            for i, (rec, label) in enumerate(records)]


def write_synthetic_data_dir(out: Path, n_train: int = 64, n_val: int = 32, epochs: int = 1, seed: int = 0) -> Path:
    """data dir with train_e{k}.jsonl, val.jsonl and test_id.jsonl in the build_data layout."""
    out.mkdir(parents=True, exist_ok=True)
    for e in range(epochs):
        write_jsonl(out / f"train_e{e}.jsonl", rows_from_records(synthetic_records(n_train, seed + e), "train"))
    write_jsonl(out / "val.jsonl", rows_from_records(synthetic_records(n_val, seed + 100), "val"))
    write_jsonl(out / "test_id.jsonl", rows_from_records(synthetic_records(n_val, seed + 200), "test_id"))
    return out
