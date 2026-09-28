import base64
import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from laya_poc import bundle as B

ROOT = Path(__file__).resolve().parents[1]


def _tree(tmp_path):
    (tmp_path / "src" / "laya_poc").mkdir(parents=True)
    (tmp_path / "src" / "laya_poc" / "a.py").write_bytes(b"x = 1\r\n")
    (tmp_path / "src" / "laya_poc" / "__pycache__").mkdir()
    (tmp_path / "src" / "laya_poc" / "__pycache__" / "a.cpython.pyc").write_bytes(b"\0")
    (tmp_path / "config.yaml").write_text("k: v\n")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    return tmp_path


def test_bundle_is_deterministic_and_skips_pycache(tmp_path):
    root = _tree(tmp_path)
    b1, s1, names = B.bundle(root)
    b2, s2, _ = B.bundle(root)
    assert (b1, s1) == (b2, s2)
    assert names == ["config.yaml", "pyproject.toml", "src/laya_poc/a.py"]


def test_bundle_normalises_line_endings(tmp_path):
    b64, _, _ = B.bundle(_tree(tmp_path))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as zf:
        assert zf.read("src/laya_poc/a.py") == b"x = 1\n"


def test_unpack_round_trip_preserves_other_files(tmp_path):
    b64, sha, _ = B.bundle(_tree(tmp_path / "src_root"))
    dest = tmp_path / "dest"
    (dest / "data").mkdir(parents=True)
    (dest / "data" / "keep.jsonl").write_text("{}")
    names = B.unpack(b64, sha, dest)
    assert (dest / "src" / "laya_poc" / "a.py").read_text() == "x = 1\n"
    assert (dest / "data" / "keep.jsonl").exists()
    assert "config.yaml" in names


def test_unpack_rejects_checksum_mismatch(tmp_path):
    b64, _, _ = B.bundle(_tree(tmp_path / "r"))
    with pytest.raises(ValueError, match="checksum"):
        B.unpack(b64, "0" * 64, tmp_path / "d")


def test_unpack_rejects_path_traversal(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.py", "x")
    raw = buf.getvalue()
    with pytest.raises(ValueError, match="unsafe"):
        B.unpack(base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest(), tmp_path / "d")


def test_embedded_unpack_source_matches_library_behaviour(tmp_path):
    b64, sha, _ = B.bundle(_tree(tmp_path / "r"))
    ns: dict = {}
    exec(B.UNPACK_SOURCE, ns)
    names = ns["_unpack"](b64, sha, str(tmp_path / "d"))
    assert "src/laya_poc/a.py" in names


def test_empty_bundle_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        B.bundle(tmp_path)


def test_repo_bundles_real_package():
    _, _, names = B.bundle(ROOT)
    assert "src/laya_poc/labels.py" in names and "config.yaml" in names
