"""Helper functions tools/build_p4_notebook.py embeds verbatim in the Phase 4 notebook's Step 1, after the Phase 1,
E1 and Phase 3 helpers (tools/notebook_helpers.py, e1_helpers.py, p3_helpers.py).

Same rules as there: each is copied with inspect.getsource, so it may use only json, os, shutil, subprocess,
time, Path and the other helpers (Step 1 imports and defines those); module constants here do not exist in the
notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

from e1_helpers import trainer_pids  # an E1 helper: Step 1 defines it before these


def baseline_pids() -> list:
    """The matrix or a baseline process still running (e.g. from before a kernel restart): a second matrix must
    never start on the same run directories."""
    modules = ("laya_poc.matrix", "laya_poc.baseline_runs", "laya_poc.small_encoder", "laya_poc.llm_baseline")
    return sorted({pid for module in modules for pid in trainer_pids(module)})


def p4_card_note(env_json, card):
    """A warning when Step 4 found another GPU than CARD, else None (also for the local CPU dry run)."""
    path = Path(env_json)
    env = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if card == "CPU" or env.get("card") == card:
        return None
    gpu = env.get("gpu_name") or "no GPU"
    return (f"WARNING: this runtime has {gpu} (card profile {env.get('card')}), not a {card}. B1 and B3 run on the "
            "CPU anyway; B4 and B5 still run, slower than the estimates, and B5 (Qwen3-4B, about 8 GB of weights) "
            "needs a GPU with at least 16 GB. CARD is recorded with every run (runs.csv): set it in Step 1 to "
            "this card's profile, or Remove Server and ask for a G4.")


def phase4_verdict(report_json) -> str:
    """One line with the Phase 4 exit check of a matrix report JSON. The report's Phase 3 exit check counts the
    Laya runs, which are not part of a Phase 4 session: it FAILs there by design and is not this step's verdict."""
    path = Path(report_json)
    ex = (json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}).get("phase4_exit_check") or {}
    if not ex:
        return f"Phase 4 exit check: not found ({path.name} missing or without it)"
    skipped = ex.get("skipped_optional") or []
    note = f"; optional, not run: {', '.join(skipped)}" if skipped else ""
    return (f"Phase 4 exit check: {ex.get('verdict')} ({ex.get('complete')}/{ex.get('total')} baseline runs "
            f"complete{note}). The Phase 3 exit check above counts the Laya runs of the Phase 3 session, not these.")


P4_HELPERS = (baseline_pids, p4_card_note, phase4_verdict)
