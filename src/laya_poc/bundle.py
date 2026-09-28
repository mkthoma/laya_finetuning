"""Pack the project's code into a notebook cell so a Colab runtime needs no git remote or Drive.

The zip is deterministic (sorted entries, fixed timestamps), so its SHA-256 only changes when
the code changes; the notebook records it and the unpack cell verifies it.
"""
from __future__ import annotations

import base64
import hashlib
import io
import zipfile
from pathlib import Path

DEFAULT_INCLUDE = ("config.yaml", "pyproject.toml", "src/laya_poc/**/*.py")
_FIXED_DATE = (2026, 1, 1, 0, 0, 0)


def collect_files(root: Path, patterns: tuple[str, ...] = DEFAULT_INCLUDE) -> list[Path]:
    files = {p for pat in patterns for p in root.glob(pat) if p.is_file() and "__pycache__" not in p.parts}
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def make_zip(root: Path, files: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files:
            info = zipfile.ZipInfo(f.relative_to(root).as_posix(), date_time=_FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, f.read_bytes().replace(b"\r\n", b"\n"))
    return buf.getvalue()


def bundle(root: Path, patterns: tuple[str, ...] = DEFAULT_INCLUDE) -> tuple[str, str, list[str]]:
    """Return (base64 zip, sha256 of the zip, member names)."""
    files = collect_files(root, patterns)
    if not files:
        raise FileNotFoundError(f"nothing to bundle under {root} for {patterns}")
    raw = make_zip(root, files)
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest(), \
        [f.relative_to(root).as_posix() for f in files]


def unpack(b64: str, sha256: str, dest: Path) -> list[str]:
    """Verify and extract a bundle, overwriting code files only (data/ and runs/ are untouched)."""
    raw = base64.b64decode(b64)
    got = hashlib.sha256(raw).hexdigest()
    if got != sha256:
        raise ValueError(f"bundle checksum mismatch: {got} != {sha256}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        for name in names:
            target = (dest / name).resolve()
            if dest.resolve() not in target.parents:
                raise ValueError(f"unsafe path in bundle: {name}")
        zf.extractall(dest)
    return names


UNPACK_SOURCE = '''\
import base64, hashlib, io, pathlib, zipfile
def _unpack(b64, sha256, dest):
    raw = base64.b64decode(b64)
    assert hashlib.sha256(raw).hexdigest() == sha256, "bundle checksum mismatch"
    dest = pathlib.Path(dest); dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for n in zf.namelist():
            assert dest.resolve() in (dest / n).resolve().parents, f"unsafe path {n}"
        zf.extractall(dest)
        return zf.namelist()
'''
