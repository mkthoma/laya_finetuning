"""matrix_exec helpers: the tee'd subprocess runner (sparse console, full log), log naming across attempts,
train/ cleanup and the per-step timings."""
from __future__ import annotations

import io
import json
import sys

from laya_poc import matrix_exec as X


def test_cleanup_keeps_the_configured_dirs_and_every_file(tmp_path):
    train = tmp_path / "train"
    for d in ("best", "ckpt", "final", "best.prev"):
        (train / d).mkdir(parents=True)
        (train / d / "f").write_text("x", encoding="utf-8")
    for f in ("log.jsonl", "summary.json", "config.yaml"):
        (train / f).write_text("x", encoding="utf-8")
    assert X.cleanup_train(train, ("best",)) == ["best.prev", "ckpt", "final"]
    assert sorted(p.name for p in train.iterdir()) == ["best", "config.yaml", "log.jsonl", "summary.json"]
    assert X.cleanup_train(tmp_path / "absent", ("best",)) == []


def test_next_log_keeps_earlier_attempts(tmp_path):
    assert X.next_log(tmp_path, "train") == tmp_path / "train.log"
    (tmp_path / "train.log").write_text("", encoding="utf-8")
    (tmp_path / "train_2.log").write_text("", encoding="utf-8")
    assert X.next_log(tmp_path, "train") == tmp_path / "train_3.log"


def test_tee_run_streams_to_the_log_and_thins_micro_lines(tmp_path, capsys):
    script = ("import sys\n"
              "for i in range(50): print(f'micro {i}/50 | loss 1.0')\n"
              "for i in range(4): print(f'evaluate val post-T {i + 1}/4 rows, {i}s')\n"
              "print('done: all good')\n"
              "sys.exit(3)\n")
    log = tmp_path / "logs" / "x.log"
    rc = X.tee_run([sys.executable, "-c", script], log)
    assert rc == 3
    text = log.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("$ ") and sum(x.startswith("micro ") for x in lines) == 50
    assert lines[-1] == "done: all good"
    out = capsys.readouterr().out
    assert out.count("micro ") == 1 and out.count(" rows, ") == 1  # console: first tick of each kind per minute
    assert "done: all good" in out and "1/4 rows" in out


def test_add_timing_accumulates(tmp_path):
    X.add_timing(tmp_path, "train", 10.0)
    X.add_timing(tmp_path, "train", 5.5)
    X.add_timing(tmp_path, "evaluate", 2.0)
    assert json.loads((tmp_path / "timing.json").read_text(encoding="utf-8")) == {"train": 15.5, "evaluate": 2.0}


def test_console_echo_survives_a_non_utf8_console(monkeypatch):

    class Cp1252Out(io.StringIO):
        def write(self, s):
            s.encode("cp1252")  # raises on characters cp1252 lacks, like a Windows pipe
            return super().write(s)

    out = Cp1252Out()
    monkeypatch.setattr(sys, "stdout", out)
    X._SparsePrinter()("evaluate ✓ done\n")
    assert out.getvalue() == "    evaluate ? done\n"
