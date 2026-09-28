"""`build_notebook.py --with-token`: a gitignored notebook variant that needs no token prompt."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_notebook as BN  # noqa: E402

FAKE = "hf_" + "A1b2C3d4E5" * 3  # token-shaped, not a real token


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def test_read_token_prefers_environment_over_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"# comment\nOTHER=1\nHF_TOKEN = '{FAKE[:-1]}Z'\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", FAKE)
    assert BN.read_local_token(tmp_path) == FAKE
    monkeypatch.delenv("HF_TOKEN")
    assert BN.read_local_token(tmp_path) == FAKE[:-1] + "Z"


def test_read_token_missing_or_malformed_raises_without_echo(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match=r"\.env"):
        BN.read_local_token(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_short\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        BN.read_local_token(tmp_path)
    assert "hf_short" not in str(exc.value)


def test_token_variant_embeds_the_token_once_in_step_3_only(tmp_path):
    out, _ = BN.build(ROOT, tmp_path / "nb.local.ipynb", token=FAKE)
    cells = _cells(out)
    holders = [s for s in cells if FAKE in s]
    assert len(holders) == 1 and holders[0].startswith("# Step 3")
    assert holders[0].count(FAKE) == 1
    assert "del PRESET_TOKEN" in holders[0]


def test_default_build_has_an_empty_preset_and_no_token(tmp_path):
    out, _ = BN.build(ROOT, tmp_path / "nb.ipynb")
    step3 = next(s for s in _cells(out) if s.startswith("# Step 3"))
    assert 'PRESET_TOKEN = ""' in step3
    assert not BN.TOKEN_RE.search(out.read_text(encoding="utf-8"))


def test_token_variant_refuses_the_committed_notebook_path():
    with pytest.raises(ValueError, match="gitignored"):
        BN.build(ROOT, BN.DEFAULT_OUT, token=FAKE)


def test_local_variant_is_gitignored():
    path = "notebooks/smoke_test.local.ipynb"
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert r.returncode == 0, f"{path} is not gitignored"


def test_preset_token_is_used_without_prompting(monkeypatch):
    # Execute the rendered Step 3 cell with a preset and a login stub: no getpass, token exported.
    src = BN.render_token_cell(FAKE)
    calls = {}
    ns = {"os": __import__("os"), "getpass": type("G", (), {"getpass": staticmethod(
        lambda *a, **k: pytest.fail("prompted despite a preset token"))}),
        "restore_hf_token": lambda: False}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "login", lambda token: calls.setdefault("token", token))
    exec(src, ns)
    assert calls["token"] == FAKE
    assert ns["os"].environ["HF_TOKEN"] == FAKE
    assert "PRESET_TOKEN" not in ns and "token" not in ns
