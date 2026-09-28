"""JSONL rows, SHA256 manifests and the per-run JSON-lines event log (design doc §7.3)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write rows atomically (temp file then rename). Returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_sha256sums(directory: str | Path, patterns: tuple[str, ...] = ("*.jsonl", "*.parquet")) -> Path:
    """Freeze a data directory: one `<sha256>  <name>` line per file, sorted by name."""
    directory = Path(directory)
    files = sorted({p for pat in patterns for p in directory.glob(pat)}, key=lambda p: p.name)
    out = directory / "SHA256SUMS"
    out.write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files), encoding="utf-8")
    return out


class RunLog:
    """Append-only JSON-lines event log: one object per event with a wall-clock `t`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields: Any) -> dict:
        event = {"t": round(time.time(), 3), **fields}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=_jsonable) + "\n")
            fh.flush()
        return event

    def events(self) -> list[dict]:
        return read_jsonl(self.path) if self.path.exists() else []


def _jsonable(o: Any) -> Any:
    if hasattr(o, "tolist"):  # numpy / torch scalars and arrays of any size
        return o.tolist()
    return str(o)
