"""Helper functions tools/build_p3_notebook.py embeds verbatim in the Phase 3 notebook's Step 1, after the Phase 1
and E1 helpers (tools/notebook_helpers.py, tools/e1_helpers.py).

Same rules as there: each is copied with inspect.getsource, so it may use only json, os, shutil, subprocess,
time, Path and the other helpers (Step 1 imports and defines those); module constants here do not exist in the
notebook.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from e1_helpers import trainer_pids  # an E1 helper: Step 1 defines it before these


def elapsed(t0) -> str:
    """Wall-clock time since t0, for the one-line duration every step prints at its end."""
    seconds = time.time() - t0
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def card_note(env_json, card):
    """A warning when Step 4 found another GPU than the card profile CARD trains with, else None (also for the
    local CPU dry run, which trains on the CPU profile by design)."""
    path = Path(env_json)
    env = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if card == "CPU" or env.get("card") == card:
        return None
    gpu = env.get("gpu_name") or "no GPU"
    return (f"WARNING: this runtime has {gpu} (card profile {env.get('card')}) but every run trains with "
            f"--card {card}. Timings and the plan assume a G4. Set CARD in Step 1 to this card's profile "
            "(config train.micro_batch) before the first training step, or Remove Server and ask for a G4.")


def busy_pids() -> list:
    """Matrix or trainer processes still running (e.g. from before a kernel restart): a second one must never
    start on the same run directories."""
    return sorted(set(trainer_pids("laya_poc.matrix")) | set(trainer_pids("laya_poc.train_single")))


def show_runs(csv_path, arm=None, cols=("run_name", "val_macro_f1", "test_id_macro_f1", "T", "order_invariance",
                                        "run_seconds")) -> list:
    """Print one line per finished run in results/runs.csv (only `arm`'s when given); return those rows."""
    import csv

    def cell(col, value):
        try:
            return value if "." not in value else f"{float(value):.{0 if col.endswith('seconds') else 4}f}"
        except ValueError:
            return value

    path = Path(csv_path)
    if not path.exists():
        print(f"no {path} yet: no run has finished")
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if arm is None or r.get("arm") == arm]
    for row in rows:
        print("  " + " | ".join(f"{c} {cell(c, row.get(c) or '')}" for c in cols))
    return rows


P3_HELPERS = (elapsed, card_note, busy_pids, show_runs)
