"""JSONL fingerprint and the frozen-data manifest (design doc §6.2, Phase 2 entry: "full splits frozen,
the data/ hash recorded").

Only the JSONL files a build writes are fingerprinted: they are exactly what training, calibration,
evaluation and the baselines read, and their bytes are reproducible across runs, OSes and library
versions. Parquet files are not (pyarrow writes its version into the metadata). A later `trap.jsonl`
(annotation output) is not part of the frozen build either. The manifest holds hashes, counts and
versions only: never FSQ rows.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import unicodedata
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io_utils import sha256_file

MANIFEST_REPORT_KEYS = ("release", "split_sizes", "event_id_pool", "headline_classes")


def fingerprint(directory: str | Path, names: Iterable[str]) -> dict[str, str]:
    """{file name: sha256} for the given JSONL files, sorted by name."""
    directory = Path(directory)
    return {name: sha256_file(directory / name) for name in sorted(set(names))}


def fingerprint_digest(files: Mapping[str, str]) -> str:
    """sha256 over the sorted '<sha256>  <name>' lines (one '\\n'-terminated line per file)."""
    lines = "".join(sorted(f"{sha}  {name}\n" for name, sha in files.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def full_build_fields(out: str | Path, names: Iterable[str], manifest_path: str | None,
                      root: Path) -> dict[str, Any]:
    """data_report fields of a FULL build: fingerprint, its digest, the frozen check, the versions."""
    files = fingerprint(out, names)
    digest = fingerprint_digest(files)
    return {"fingerprint": files, "fingerprint_sha256": digest,
            "frozen_check": frozen_check(manifest_path, root, files, digest), "environment": environment()}


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    """Versions that could change the bytes: DuckDB hash(), pandas/numpy sampling, the tokenizers
    (token budgets decide compression), Python's Unicode tables (name normalisation, trap regexes)."""
    return {"duckdb": _version("duckdb"), "pandas": _version("pandas"), "pyarrow": _version("pyarrow"),
            "numpy": _version("numpy"), "tokenizers": _version("tokenizers"),
            "transformers": _version("transformers"), "python": platform.python_version(),
            "unicodedata": unicodedata.unidata_version, "platform": platform.platform(),
            "byteorder": sys.byteorder}


def manifest(report: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen-build manifest (docs/results/data_fingerprint_<release>.json) from a FULL data_report."""
    env = report.get("environment") or environment()
    return {"release": report["release"], "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **env, "files": dict(report["fingerprint"]), "fingerprint_sha256": report["fingerprint_sha256"],
            **{k: report[k] for k in MANIFEST_REPORT_KEYS if k != "release"}}


def write_manifest(path: str | Path, report: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(manifest(report), indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


def differing_files(files: Mapping[str, str], frozen: Mapping[str, str]) -> list[str]:
    """Every file whose hash differs, or that only one side has, with the reason."""
    out = [f"{n} (changed)" for n in sorted(set(files) & set(frozen)) if files[n] != frozen[n]]
    out += [f"{n} (not in this build)" for n in sorted(set(frozen) - set(files))]
    out += [f"{n} (not in the manifest)" for n in sorted(set(files) - set(frozen))]
    return out


def frozen_check(manifest_path: str | None, root: Path, files: Mapping[str, str], digest: str) -> dict[str, Any]:
    """Compare a build's fingerprint with config data.frozen_manifest (a path relative to the project root)."""
    if not manifest_path:
        return {"manifest": None, "status": "not_frozen", "differing": []}
    path = Path(manifest_path)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        return {"manifest": str(manifest_path), "status": "missing_manifest", "differing": []}
    frozen = json.loads(path.read_text(encoding="utf-8"))
    differing = differing_files(files, frozen.get("files", {}))
    same = not differing and frozen.get("fingerprint_sha256") == digest
    return {"manifest": str(manifest_path), "status": "match" if same else "mismatch", "differing": differing,
            "frozen_sha256": frozen.get("fingerprint_sha256"), "frozen_versions": {
                k: frozen.get(k) for k in ("duckdb", "pandas", "pyarrow", "python", "platform")}}


def mismatch_message(check: Mapping[str, Any]) -> str:
    """One line for the notebook: what differs and how to re-freeze deliberately."""
    m = check["manifest"]
    if check["status"] == "missing_manifest":
        return (f"frozen manifest {m} not found (config data.frozen_manifest): the data cannot be verified; on "
                "Colab the manifest must be bundled with the code")
    listed = ", ".join(check["differing"]) or "fingerprint_sha256 only"
    return (f"data differs from the frozen manifest {m}: {listed}. Check the pinned versions against the "
            f"manifest ({check.get('frozen_versions')}); if the change is intended, re-freeze: rebuild with "
            f"--write-manifest {m}, review and commit it, and rerun every experiment on the new data")
