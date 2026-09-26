import numpy as np

from laya_poc import io_utils as IO


def test_jsonl_round_trip_keeps_unicode(tmp_path):
    rows = [{"id": "a", "state": '{"name":"Café Ñandú"}'}, {"id": "b", "n": 2}]
    p = tmp_path / "sub" / "x.jsonl"
    assert IO.write_jsonl(p, rows) == 2
    assert IO.read_jsonl(p) == rows
    assert "Café" in p.read_text(encoding="utf-8")
    assert not (tmp_path / "sub" / "x.jsonl.tmp").exists()


def test_sha256sums_lists_files_sorted(tmp_path):
    IO.write_jsonl(tmp_path / "b.jsonl", [{"x": 1}])
    IO.write_jsonl(tmp_path / "a.jsonl", [{"x": 2}])
    out = IO.write_sha256sums(tmp_path)
    lines = out.read_text().splitlines()
    assert [l.split("  ")[1] for l in lines] == ["a.jsonl", "b.jsonl"]
    assert lines[0].split("  ")[0] == IO.sha256_file(tmp_path / "a.jsonl")


def test_runlog_appends_events_with_timestamp_and_numpy_values(tmp_path):
    log = IO.RunLog(tmp_path / "run" / "log.jsonl")
    log.log(step=1, loss=np.float32(2.5))
    log.log(step=2, arr=np.array([1, 2]))
    ev = log.events()
    assert [e["step"] for e in ev] == [1, 2]
    assert ev[0]["loss"] == 2.5 and ev[1]["arr"] == [1, 2]
    assert all("t" in e for e in ev)


def test_runlog_events_empty_when_missing(tmp_path):
    assert IO.RunLog(tmp_path / "none.jsonl").events() == []
