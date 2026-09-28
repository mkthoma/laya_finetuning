"""Attribution texts that travel with FSQ-derived data and fine-tuned Laya checkpoints (design doc Appendix D).

FSQ OS Places is Apache-2.0 with a NOTICE file: any derived artefact we keep carries that text verbatim
plus a statement that the data was modified. Laya is Apache-2.0: a fine-tuned checkpoint keeps its
LICENSE and is marked as modified. The NOTICE text is embedded (no download at run time, so building
the data needs no extra network call).
"""
from __future__ import annotations

import importlib.metadata as md
import json

FSQ_NOTICE_NAME = "NOTICE_FSQ.txt"
CHECKPOINT_NOTICE_NAME = "NOTICE.md"
LAYA_REPO = "https://github.com/NandhaKishorM/laya"

# NOTICE.txt at the root of the foursquare/fsq-os-places dataset repo, byte for byte (582 bytes, no final newline).
FSQ_NOTICE = (
    "Copyright 2024 Foursquare Labs, Inc. All rights reserved.\n"
    "\n"
    'Licensed under the Apache License, Version 2.0 (the "License");\n'
    "you may not use this file except in compliance with the License.\n"
    "You may obtain a copy of the License at\n"
    "\n"
    "    http://www.apache.org/licenses/LICENSE-2.0\n"
    "\n"
    "Unless required by applicable law or agreed to in writing, software\n"
    'distributed under the License is distributed on an "AS IS" BASIS,\n'
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "See the License for the specific language governing permissions and\n"
    "limitations under the License."
)


def fsq_notice(release: str | None) -> str:
    """The FSQ NOTICE verbatim, then the statement that the data was modified."""
    rel = f"release {release}" if release else "(release not recorded)"
    return (f"{FSQ_NOTICE}\n\nDerived from Foursquare Open Source Places {rel}; "
            "modified: fields removed, records filtered and sampled.\n")


def checkpoint_notice(*, model_name: str, base_repo: str | None = None, base_revision: str | None = None,
                      laya_commit: str | None = None) -> str:
    """NOTICE.md for a fine-tuned checkpoint: what it was derived from, its licence and its training data."""
    base = base_repo or "not recorded"
    rev = base_revision or "not recorded"
    licence = f"{LAYA_REPO}/blob/{laya_commit or 'main'}/LICENSE"
    commit_note = "" if laya_commit else " (the Laya commit was not recorded)"
    return (f"# {model_name}\n\n"
            f"- Base checkpoint: {base} (Hub revision: {rev})\n"
            "- modified: fine-tuned (weights and rl_agent_config.json differ from the base checkpoint)\n"
            f"- Licence: Laya code and weights are Apache-2.0 (Convai Innovations): {licence}{commit_note}; "
            "a copy is in LICENSE when the installed laya package ships one\n"
            f"- Data: trained on data derived from FSQ OS Places (see {FSQ_NOTICE_NAME})\n")


def installed_laya_commit() -> str | None:
    """Commit of the installed laya (PEP 610 direct_url.json of a git install); None otherwise."""
    try:
        info = json.loads(md.distribution("laya").read_text("direct_url.json") or "{}")
    except (md.PackageNotFoundError, ValueError):
        return None
    return (info.get("vcs_info") or {}).get("commit_id")


def laya_license_text() -> str | None:
    """Laya's LICENSE as shipped in the installed package's metadata; None when absent."""
    try:
        files = md.distribution("laya").files or []
    except md.PackageNotFoundError:
        return None
    for f in files:
        if f.name.upper() in ("LICENSE", "LICENSE.TXT", "LICENSE.MD"):
            path = f.locate()
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return None
    return None
