"""Phase 5 report (design §7.13 templates; spec P5 §3) on synthetic metrics / bench JSONs: every template section,
the decision checklist, the bootstrap and confusion tables, the CLI (md + json, bench optional), metrics only."""
from __future__ import annotations

import json
import re

import pytest

from laya_poc import decision as D
from laya_poc import phase5_report as R
from laya_poc import phase5_report_md as M

from test_decision_eval import bench_json, metrics_json


@pytest.fixture(scope="module")
def cfg():
    from laya_poc.config import load_config
    from conftest import ROOT
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="module")
def pending(cfg):
    return R.build_report(metrics_json(), [], cfg)


@pytest.fixture(scope="module")
def complete(cfg):
    return R.build_report(metrics_json(traps_annotated=True), [bench_json()], cfg)


def _section(md: str, title: str) -> str:
    start = md.index(title)
    nxt = md.find("\n## ", start + len(title))
    return md[start: nxt if nxt > 0 else len(md)]


def test_main_results_one_table_per_pool_in_template_order(pending):
    md = M.render_markdown(pending)
    for pool in ("test_id", "ood_country", "ood_script", "ood_brand"):
        assert f"### {pool}" in md
    sec = _section(md, "### test_id")
    assert ("| Arm | Macro-F1 (mean ± range) | Acc | ECE pre → post | Brier | NLL | Acc@80 | Acc@90 | Gap vs ID |"
            in sec)
    names = ["Majority (B1)", "Zero-shot `laya` (B2)", "TF-IDF+LR (B3)", "ModernBERT-base (B4)", "mmBERT-small (B4)",
             "Head-only `laya` (E4)", "FT `laya` (E2)", "FT `laya-multilingual` (E3)", "Reference LLM"]
    idx = [sec.index(n) for n in names]
    assert idx == sorted(idx)
    assert "0.5639 (0.5602 to 0.5688)" in sec                         # E2 test_id seed mean (min to max)
    e2_script = next(ln for ln in _section(md, "### ood_script").splitlines() if "FT `laya` (E2)" in ln)
    assert "(excluded)" in e2_script                                  # laya cannot read Thai: gap excluded


def test_traps_abstention_stability(pending, complete):
    sec = _section(M.render_markdown(pending), "## Traps, abstention and stability")
    assert "Trap acc (all / multi)" in sec and "pending" in sec and "unannotated" in sec
    line = next(ln for ln in _section(M.render_markdown(complete), "## Traps, abstention and stability").splitlines()
                if "FT `laya` (E2)" in ln)
    assert "0.3800 / 0.3300" in line and "0.1200" in line          # trap all / multi; flip rate


def test_cpu_table_and_relative_throughput(pending, complete):
    assert "PENDING" in _section(M.render_markdown(pending), "## CPU")
    sec = _section(M.render_markdown(complete), "## CPU")
    assert "| Model | Backend | Threads | Cold start (s) | p50 (ms) | p95 (ms) | Batch rec/s | Peak RAM (GB) | Hardware |" in sec
    assert "Test CPU (8C/16T, 32 GB RAM)" in sec and "modernbert_base" in sec
    rel = next(r for r in complete["tables"]["relative_throughput"] if r["model"] == "laya")
    assert rel["encoder"] == "modernbert_base" and rel["ratio"] == pytest.approx(60 / 120)


def test_decision_checklist(pending, complete):
    sec = _section(M.render_markdown(pending), "## Decision checklist")
    for text in ("C1", "C2", "C3", "C4", "C5", "stop (a)", "stop (b)", "stop (c)", "investigate (i)",
                 "investigate (ii)", "investigate (iii)", "C1 strict (sensitivity)"):
        assert text in sec
    assert "PENDING (traps)" in sec and "PENDING (bench)" in sec and "FAIL (near miss)" in sec
    assert "+4.3 pts" in sec and "B4 modernbert_base c10" in sec          # E2 primary lead, matched B4
    assert f"**{D.INVESTIGATE}**" in sec and "PASS is out of reach" in sec
    final = _section(M.render_markdown(complete), "## Decision checklist")
    assert "PENDING" not in final.split("Overall")[0] and "final" in final


def test_verdict_at_the_top(pending):
    md = M.render_markdown(pending)
    head = md[: md.index("## Main results")]
    assert f"Verdict: **{D.INVESTIGATE}**" in head and "provisional" in head
    assert head.count("Verdict") == 1  # the statement after the bold verdict does not restate it


def test_bootstrap_table(pending):
    sec = _section(M.render_markdown(pending), "## Bootstrap")
    assert "| Candidate | Baseline | Pool | Diff (points) | 95% CI (points) | P(diff > 0) | P(diff >= 3 points) |" in sec
    row = next(ln for ln in sec.splitlines() if "E3 laya_ml c10" in ln and "B4 mmbert_small c10" in ln
               and "ood_average" in ln)
    assert "-0.4" in row and "[-2.4, +1.6]" in row


def test_per_class_and_lookalike_tables(pending):
    md = M.render_markdown(pending)
    sec = _section(md, "## Per-class")
    assert "B4 mmbert_small c10" in sec                             # best baseline by test_id macro-F1
    assert "| dining | 0.600 / 0.550 / 0.570 |" in sec
    look = _section(md, "## Look-alike")
    assert "dining → arts" in look and "arts → dining" in look and "0.034 (2/58)" in look


def test_report_has_no_fsq_rows(complete):
    md = M.render_markdown(complete)
    text = md + json.dumps(complete)
    assert "fsq_place_id" not in text and '"state"' not in text and not re.search(r"hf_[A-Za-z0-9]{20,}", text)


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_cli_writes_md_and_json(tmp_path, capsys):
    metrics = _write(tmp_path, "m.json", metrics_json(traps_annotated=True))
    bench = _write(tmp_path, "b.json", bench_json())
    out = tmp_path / "rep" / "phase5_report.md"
    assert R.main(["--metrics", str(metrics), "--bench", str(bench), "--out", str(out)]) == 0
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["decision"]["verdict"] == D.INVESTIGATE and data["decision"]["final"] is True
    assert data["inputs"]["bench"] == [str(bench)]
    assert "## Decision checklist" in out.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert "INVESTIGATE" in printed and len(printed.splitlines()) <= 5


def test_cli_without_bench_and_with_a_bad_metrics_file(tmp_path, capsys):
    metrics = _write(tmp_path, "m.json", metrics_json())
    out = tmp_path / "r.md"
    assert R.main(["--metrics", str(metrics), "--out", str(out)]) == 0
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["decision"]["final"] is False
    assert R.main(["--metrics", str(tmp_path / "absent.json"), "--out", str(out)]) == 1
    assert "absent.json" in capsys.readouterr().err


def test_cpu_notes_for_skipped_threads_and_failures(cfg):
    bench = {**bench_json(with_eight=False), "threads_skipped": [{"threads": 8, "reason": "above 4 physical cores"}],
             "errors": [{"model": "laya", "backend": "onnx", "threads": 2, "error": "timed out"}]}
    sec = _section(M.render_markdown(R.build_report(metrics_json(), [bench], cfg)), "## CPU")
    assert "8 threads skipped (above 4 physical cores)" in sec and "laya/onnx 2 threads failed (timed out)" in sec
