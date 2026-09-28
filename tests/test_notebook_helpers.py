"""Step 1 helpers that the notebook defines in the kernel (tools/notebook_helpers.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import notebook_helpers as nh  # noqa: E402


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_run_logged_tees_output_and_returns_code(tmp_path, capsys):
    log = tmp_path / "logs" / "a.log"
    rc = nh.run_logged(_py("print('hello'); print('world')"), log)
    assert rc == 0
    assert "hello" in capsys.readouterr().out
    text = log.read_text(encoding="utf-8")
    assert "hello" in text and "world" in text and text.startswith("$ ")


def test_run_logged_raises_with_log_path_and_hint_on_unexpected_code(tmp_path):
    log = tmp_path / "b.log"
    with pytest.raises(RuntimeError, match=r"exited with 3") as exc:
        nh.run_logged(_py("import sys; sys.exit(3)"), log, hint="check X")
    assert str(log) in str(exc.value) and "check X" in str(exc.value)


def test_run_logged_accepts_expected_codes_or_none(tmp_path):
    assert nh.run_logged(_py("import sys; sys.exit(3)"), tmp_path / "c.log", expect_returncode=(-9, 3)) == 3
    assert nh.run_logged(_py("import sys; sys.exit(5)"), tmp_path / "d.log", expect_returncode=None) == 5


def test_run_logged_closes_stdin(tmp_path, capsys):
    nh.run_logged(_py("import sys; print('got', repr(sys.stdin.read()))"), tmp_path / "e.log")
    assert "got ''" in capsys.readouterr().out


def test_already_done_respects_redo(tmp_path, monkeypatch):
    marker = tmp_path / "done.json"
    assert not nh.already_done(marker)
    marker.write_text("{}")
    assert nh.already_done(marker)
    monkeypatch.setattr(nh, "REDO", True, raising=False)
    assert not nh.already_done(marker)


def test_fresh_dir_and_write_json(tmp_path, capsys):
    d = tmp_path / "run"
    nh.write_json(d / "x" / "summary.json", {"a": 1, "per_row": [1, 2, 3], "b": {"cm": [[1]], "c": 2}})
    nh.show_json(d / "x" / "summary.json")
    out = capsys.readouterr().out
    assert "a: 1" in out and "per_row" not in out and "cm" not in out
    nh.fresh_dir(d)
    assert d.is_dir() and not any(d.iterdir())


def test_show_json_filters_keys_and_falls_back_to_all(tmp_path, capsys):
    path = tmp_path / "s.json"
    nh.write_json(path, {"a": 1, "b": 2})
    assert nh.show_json(path, ["b", "zzz"]) == {"a": 1, "b": 2}
    assert capsys.readouterr().out.split() == ["b:", "2"]
    nh.show_json(path, ["zzz"])
    assert "a: 1" in capsys.readouterr().out


def test_resume_blocker_without_a_crash_run(tmp_path):
    assert "Step 9" in nh.resume_blocker(tmp_path / "runs")
