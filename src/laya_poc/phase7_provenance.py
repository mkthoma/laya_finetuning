"""The workbook's Provenance sheet: what produced the numbers (design §6.2 Phase 7: "archived with configs, hashes and
seeds").

Rows (item, value, source): the Laya commit and Hub revision and the pinned baseline revisions (config.yaml), the FSQ
release and the frozen data fingerprint (its manifest JSON), the seeds, the decision thresholds and CPU budget, the
sha256 of config.yaml and of every input JSON, the generation times of the Phase 5 report and the GPU timing, and the
notebook code-bundle sha256 of the Colab runs where a record of it exists: env*.json / *.log in the run archives
(full sha256), and report headers in docs/results ("bundle <hex>", often an 8-hex prefix). What is not found is said
in the cell, never guessed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io_utils import sha256_file
from .phase7_xlsx import Sheet, kv_table

LOG_BUNDLE = re.compile(r"bundle\W{0,30}(?:sha-?256\W{0,5})?([0-9a-f]{64})\b", re.IGNORECASE)
NOTE_BUNDLE = re.compile(r"\bbundle ([0-9a-f]{8,64})\b")
HEX = re.compile(r"^[0-9a-f]{8,64}$")
HEADER_LINES = 3
Row = tuple[str, Any, Any]


def rel(p: Path | str | None) -> str | None:
    """A path relative to the working directory when under it, else its name (no user paths in the workbook)."""
    if p is None:
        return None
    p = Path(p)
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.name


def config_rows(cfg: Mapping) -> list[Row]:
    laya, p4 = cfg.get("laya") or {}, cfg.get("phase4") or {}
    rows: list[Row] = [(f"Laya {k.replace('_', ' ')}", laya.get(k), "config.yaml laya")
                       for k in ("repo", "version", "commit", "hub_revision")]
    for section in ("small_encoders", "llms"):
        rows += [(f"Baseline revision: {k}", f"{v.get('id')}@{v.get('revision')}", f"config.yaml phase4.{section}")
                 for k, v in (p4.get(section) or {}).items()]
    return rows


def data_rows(cfg: Mapping, fingerprint: Mapping | None, fp_path: Path | None) -> list[Row]:
    src = rel(fp_path) if fingerprint else None
    fp = fingerprint or {}
    rows: list[Row] = [("FSQ release", fp.get("release") or (cfg.get("data") or {}).get("fsq_release"),
                        src or "config.yaml data.fsq_release"),
                       ("Frozen data fingerprint sha256",
                        fp.get("fingerprint_sha256") or f"not found ({rel(fp_path) or 'no manifest given'})", src),
                       ("Frozen data manifest created", fp.get("created"), src)]
    rows += [(f"Frozen split size: {k}", v, src) for k, v in (fp.get("split_sizes") or {}).items()]
    return rows


def seed_rows(cfg: Mapping) -> list[Row]:
    arms = [*((cfg.get("phase3") or {}).get("arms") or []), *((cfg.get("phase4") or {}).get("arms") or [])]
    seeds = sorted({s for a in arms for s in a.get("seeds") or []})
    get = lambda *ks: _dig(cfg, *ks)  # noqa: E731
    return [("Training seeds (Phase 3 / 4 arms)", ", ".join(map(str, seeds)), "config.yaml phase3/phase4 arms"),
            ("Split seed", get("data", "split_seed"), "config.yaml data.split_seed"),
            ("LLM eval subset seed", get("phase4", "llm_eval", "subset_seed"), "config.yaml phase4.llm_eval"),
            ("Bootstrap seed", get("phase5", "bootstrap", "seed"), "config.yaml phase5.bootstrap"),
            ("Bootstrap resamples", get("phase5", "bootstrap", "resamples"), "config.yaml phase5.bootstrap"),
            ("Order-invariance seed", get("phase3", "order_invariance", "seed"), "config.yaml phase3")]


def threshold_rows(cfg: Mapping) -> list[Row]:
    rows: list[Row] = [(f"Decision: {k}", v, "config.yaml phase5.decision")
                       for k, v in (_dig(cfg, "phase5", "decision") or {}).items()]
    rows += [(f"CPU budget: {k}", v, "config.yaml cpu_budget") for k, v in (cfg.get("cpu_budget") or {}).items()]
    onnx = _dig(cfg, "phase5", "bench", "onnx") or {}
    rows += [(f"ONNX acceptance: {k}", onnx[k], "config.yaml phase5.bench.onnx")
             for k in ("min_argmax_agree", "max_dp", "check_n") if k in onnx]
    return rows


def _dig(obj: Any, *keys: str) -> Any:
    for k in keys:
        obj = obj.get(k) if isinstance(obj, Mapping) else None
    return obj


# ---------------------------------------------------------------- bundle sha256

def _json_bundles(obj: Any) -> list[str]:
    if isinstance(obj, Mapping):
        return [v for k, v in obj.items() if "bundle" in str(k).lower() and isinstance(v, str) and HEX.match(v)] + \
            [x for v in obj.values() for x in _json_bundles(v)]
    return [x for v in obj for x in _json_bundles(v)] if isinstance(obj, list) else []


def archive_bundles(archives: Iterable[Path]) -> list[tuple[str, str]]:
    """(sha256, source file) recorded in env*.json (a key naming a bundle) or *.log ("bundle ... <sha256>")."""
    found: dict[str, str] = {}
    for a in archives:
        for p in sorted(Path(a).rglob("env*.json")):
            try:
                hits = _json_bundles(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                hits = []
            for h in hits:
                found.setdefault(h, rel(p))
        for p in sorted(Path(a).rglob("*.log")):
            for m in LOG_BUNDLE.finditer(p.read_text(encoding="utf-8", errors="replace")):
                found.setdefault(m.group(1).lower(), rel(p))
    return sorted(found.items(), key=lambda kv: kv[1])


def note_bundles(notes_dir: Path | None) -> list[tuple[str, str]]:
    """(hex, file) from the first lines of the result reports ("... notebook ..., bundle fae1ee11)")."""
    if notes_dir is None or not Path(notes_dir).is_dir():
        return []
    out = []
    for p in sorted(Path(notes_dir).glob("*.md")):
        head = "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[:HEADER_LINES])
        out += [(m.group(1), rel(p)) for m in NOTE_BUNDLE.finditer(head)]
    return out


def bundle_rows(archives: Sequence[Path], notes_dir: Path | None) -> list[Row]:
    item = "Notebook bundle sha256 (archives)"
    found = archive_bundles(archives)
    names = ", ".join(rel(a) for a in archives) or "none given or found"
    rows: list[Row] = [(item, sha, src) for sha, src in found] or [
        (item, f"not recorded in the run archives (searched env*.json and *.log under: {names})", None)]
    for h, src in note_bundles(notes_dir):
        size = "full sha256" if len(h) == 64 else f"{len(h)}-hex prefix; the full sha256 is not recorded"
        rows.append(("Notebook bundle (report header)", f"{h} ({size})", src))
    return rows


# ---------------------------------------------------------------- the sheet

def run_rows(report: Mapping, gpu: Mapping | None) -> list[Row]:
    rc = report.get("recomputed_check") or {}
    cards = "; ".join(f"{c} = {n}" for c, n in ((gpu or {}).get("cards") or {}).items()) or None
    return [("Phase 5 report generated at", report.get("generated_at"), "report"),
            ("Phase 5 metrics generated at", report.get("metrics_generated_at"), "report"),
            ("Recomputed macro-F1 check (preds vs eval JSONs)",
             f"passed {rc.get('passed')} on {rc.get('compared')} run-splits" if rc else None, "report"),
            ("GPU timing generated at", (gpu or {}).get("generated_at"), "gpu_timing"),
            ("Colab cards", cards, "gpu_timing (run archives env.json)")]


def input_rows(paths: Mapping[str, Path | None], config_path: Path) -> list[Row]:
    rows: list[Row] = [("config.yaml sha256", sha256_file(config_path),
                        f"{rel(config_path)} (the copy read when this workbook was generated)")]
    rows += [(f"Input sha256: {k}", sha256_file(p) if p else "not given", rel(p)) for k, p in paths.items()]
    return rows


def provenance_sheet(*, cfg: Mapping, config_path: Path, report: Mapping, gpu: Mapping | None,
                     fingerprint: Mapping | None, fp_path: Path | None, inputs: Mapping[str, Path | None],
                     archives: Sequence[Path], notes_dir: Path | None) -> Sheet:
    rows = [*config_rows(cfg), *data_rows(cfg, fingerprint, fp_path), *seed_rows(cfg), *threshold_rows(cfg),
            *input_rows(inputs, config_path), *run_rows(report, gpu), *bundle_rows(archives, notes_dir)]
    return Sheet("Provenance", [kv_table(None, rows, ("Item", "Value", "Source"))])
