"""Phase 2 gate (laya_poc.gate): design doc §6.2 gate table on synthetic E1 artefacts.

Every input is written by the test (no GPU, no model): a passing E1 run, then one change per test.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from laya_poc import gate as G
from laya_poc.config import load_config

ROOT = Path(__file__).resolve().parents[1]
GATE_CFG = {"min_over_zero_shot": 0.10, "min_over_majority": 0.20, "max_below_tfidf": 0.05, "min_scale": 1.0}
CRIT = {"e2e": G.END_TO_END, "num": G.NUMERICS, "beats": G.BEATS_TRIVIAL, "near": G.NEAR_TFIDF}


def _log_events() -> list[dict]:
    """A crash at micro-step 2002, a resume from the micro-step-2000 checkpoint, one skipped fp16 step."""
    ev = [{"t": 1000.0, "event": "start", "fp16": True, "micro_batch": 8, "grad_accum": 4},
          {"t": 1001.0, "event": "micro", "micro_step": 1, "loss": 2.3, "loss_ce": 2.3, "finite": True},
          {"t": 1002.0, "event": "opt", "opt_step": 1, "scale": 65536.0, "grad_norm": 3.1},
          {"t": 1003.0, "event": "opt", "opt_step": 2, "scale": 32768.0, "grad_norm": float("inf")},
          {"t": 1004.0, "event": "opt", "opt_step": 500, "scale": 32768.0, "grad_norm": 1.2,
           "vram_reserved_gb": 8.4},
          {"t": 1005.0, "event": "ckpt", "opt_step": 500, "micro_step": 2000, "seconds": 70.0},
          {"t": 1864.0, "event": "crash_injected", "micro_step": 2002},
          {"t": 1900.0, "event": "start", "fp16": True},
          {"t": 1901.0, "event": "resumed", "from_micro_step": 2000, "from_opt_step": 500, "path": "x/step0000500.pt"},
          {"t": 1902.0, "event": "opt", "opt_step": 501, "scale": 32768.0, "grad_norm": 0.9,
           "vram_reserved_gb": 8.6},
          {"t": 1903.0, "event": "eval", "opt_step": 750, "val_macro_f1": 0.70},
          {"t": 1904.0, "event": "best", "opt_step": 750, "val_macro_f1": 0.70},
          {"t": 8100.0, "event": "done", "micro_steps": 13376, "opt_steps": 3344, "stop_reason": "epochs"}]
    return ev


def _metrics(f1: float, f1_9: float, **extra) -> dict:
    return {"n": 3000, "macro_f1": f1, "macro_f1_9": f1_9, "acc": f1 + 0.05, "ece": 0.04, "brier": 0.4, "nll": 0.9,
            "acc@80": 0.9, "acc@90": 0.86, **extra}


def _inputs() -> dict[str, object]:
    summary = {"stop_reason": "epochs", "micro_steps": 13376, "opt_steps": 3344, "nonfinite": 0,
               "nonfinite_grad_applied": 0, "opt_steps_skipped": 1, "min_scale": 32768.0, "evals": 13,
               "best_opt_step": 750, "best_eval": {"val_macro_f1": 0.70, "val_acc": 0.74, "n": 3000},
               "sec_per_micro_mean": 0.431, "sec_per_micro_median": 0.42, "peak_vram_reserved_gb": 8.6,
               "peak_vram_alloc_gb": 8.3, "epochs": 4, "device": "cuda", "truncated_items": 0,
               "padding_ratio": 1.08, "model": "laya", "seed": 11, "resumed_from": 2000}
    return {
        "train/summary.json": summary,
        "train/log.jsonl": _log_events(),
        "crash_exit.json": {"returncode": -9},
        "export_check.json": {"passed": True, "T": 1.21, "clamped": False, "pre": {"ece": 0.06}, "post": {"ece": 0.02}},
        "eval_e1_val.json": {"model": "laya", "n": 3000, "n_unlabelled": 0, "n_abstained_no_evidence": 2,
                             "temperature": 1.21, "pre": _metrics(0.70, 0.72, ece=0.06),
                             "post": _metrics(0.70, 0.72, ece=0.02), "cpu_fallback": False},
        "eval_zeroshot_val.json": {"model": "laya", "pre": _metrics(0.40, 0.41), "post": _metrics(0.40, 0.41)},
        "baselines.json": {"n_train": 25000, "schemes": {"c10": {
            "b1_majority": {"val": _metrics(0.03, 0.035)}, "b1_prior": {"val": _metrics(0.03, 0.035)},
            "b3_tfidf_lr": {"C": 4, "T": 0.9, "val": {"pre": _metrics(0.72, 0.74), "post": _metrics(0.72, 0.74)}}}}},
        "bench_cpu.json": {"results": [
            {"threads": 1, "cold_s": 5.1, "p50_ms": 120.0, "p95_ms": 180.0, "batch_rps": 9.5, "peak_rss_gb": 1.9},
            {"threads": 2, "cold_s": 4.8, "p50_ms": 80.0, "p95_ms": 110.0, "batch_rps": 15.0, "peak_rss_gb": 1.9}]},
        "data/data_report.json": {"headline_classes": 10, "event_id_pool": 450, "fingerprint_sha256": "ab" * 32,
                                  "split_sizes": {"train": 25000, "val": 3000}},
    }


def _write(root: Path, files: dict[str, object]) -> Path:
    for rel, obj in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".jsonl"):
            path.write_text("".join(json.dumps(e) + "\n" for e in obj), encoding="utf-8")
        else:
            path.write_text(json.dumps(obj), encoding="utf-8")
    return root


def _gate(tmp_path: Path, files: dict[str, object] | None = None, cfg_over: dict | None = None) -> dict:
    """Write the artefacts (runs/e1 + data) and a config; run the gate in-process; return gate_report.json."""
    files = _inputs() if files is None else files
    runs, data = tmp_path / "runs" / "e1", tmp_path / "data"
    _write(runs, {k: v for k, v in files.items() if not k.startswith("data/")})
    _write(tmp_path, {k: v for k, v in files.items() if k.startswith("data/")})
    data.mkdir(exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    for dotted, value in (cfg_over or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = runs / "gate_report.md"
    assert G.main(["--run-root", str(runs), "--data-dir", str(data), "--out", str(out), "--config", str(cfg_path)]) == 0
    return json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))


def _crit(report: dict, key: str) -> dict:
    return next(c for c in report["criteria"] if c["name"] == CRIT[key])


def _without(key: str) -> dict:
    files = _inputs()
    del files[key]
    return files


def _edit(path: str, fn) -> dict:
    files = copy.deepcopy(_inputs())
    fn(files[path])
    return files


# ---------------------------------------------------------------- verdict

def test_a_complete_e1_run_passes_every_criterion(tmp_path, capsys):
    rep = _gate(tmp_path)
    assert rep["verdict"] == "PASS", rep["criteria"]
    assert [c["name"] for c in rep["criteria"]] == list(G.CRITERIA)
    assert all(c["passed"] for c in rep["criteria"])
    md = (tmp_path / "runs" / "e1" / "gate_report.md").read_text(encoding="utf-8")
    assert "Verdict: **PASS**" in md and "Debug checklist" not in md
    assert "gate: PASS" in capsys.readouterr().out


@pytest.mark.parametrize("missing, failed", [
    ("train/summary.json", {"e2e", "num"}),
    ("train/log.jsonl", {"e2e", "num"}),
    ("export_check.json", {"e2e"}),
    ("eval_e1_val.json", {"e2e", "beats", "near"}),
    ("bench_cpu.json", {"e2e"}),
    ("eval_zeroshot_val.json", {"beats"}),
    ("baselines.json", {"beats", "near"}),
    ("data/data_report.json", {"beats", "near"}),
])
def test_a_missing_input_fails_its_criteria_with_a_note_and_never_crashes(tmp_path, missing, failed):
    rep = _gate(tmp_path, _without(missing))
    assert rep["verdict"] == "FAIL"
    got = {k for k in CRIT if not _crit(rep, k)["passed"]}
    assert got == failed
    name = Path(missing).name
    assert all(name in _crit(rep, k)["note"] for k in failed), [_crit(rep, k)["note"] for k in failed]


def test_a_failed_gate_prints_and_renders_the_debug_checklist(tmp_path, capsys):
    rep = _gate(tmp_path, _edit("eval_e1_val.json", lambda e: e["post"].update(macro_f1=0.60)))
    assert rep["verdict"] == "FAIL" and rep["debug_checklist"]
    md = (tmp_path / "runs" / "e1" / "gate_report.md").read_text(encoding="utf-8")
    for phrase in ("leakage asserts", "label mapping", "question-key order", "truncation counts", "learning rates",
                   "loss sign"):
        assert phrase in md.lower(), phrase
    out = capsys.readouterr().out
    assert "gate: FAIL" in out and "stop and debug" in out


def test_malformed_input_fails_the_criterion_instead_of_crashing(tmp_path):
    rep = _gate(tmp_path, _edit("eval_e1_val.json", lambda e: e.update(post="garbage")))
    assert not _crit(rep, "beats")["passed"] and not _crit(rep, "near")["passed"]
    assert "post" in _crit(rep, "beats")["note"]


# ---------------------------------------------------------------- 1. end-to-end

@pytest.mark.parametrize("reason", [None, "crashed", ""])
def test_training_must_have_finished(tmp_path, reason):
    rep = _gate(tmp_path, _edit("train/summary.json", lambda s: s.update(stop_reason=reason)))
    assert not _crit(rep, "e2e")["passed"] and "stop_reason" in _crit(rep, "e2e")["note"]


@pytest.mark.parametrize("reason", ["epochs", "early_stop", "max_micro_steps"])
def test_every_normal_stop_reason_counts_as_finished(tmp_path, reason):
    assert _crit(_gate(tmp_path, _edit("train/summary.json", lambda s: s.update(stop_reason=reason))), "e2e")["passed"]


def test_a_failed_temperature_fit_fails_end_to_end(tmp_path):
    rep = _gate(tmp_path, _edit("export_check.json", lambda e: e.update(passed=False)))
    assert not _crit(rep, "e2e")["passed"] and "export_check" in _crit(rep, "e2e")["note"]


def test_a_benchmark_without_thread_settings_fails_end_to_end(tmp_path):
    rep = _gate(tmp_path, _edit("bench_cpu.json", lambda b: b.update(results=[])))
    assert not _crit(rep, "e2e")["passed"] and "thread" in _crit(rep, "e2e")["note"]


def test_a_run_without_a_resume_fails_end_to_end(tmp_path):
    files = _inputs()
    files["train/log.jsonl"] = [e for e in files["train/log.jsonl"] if e["event"] != "resumed"]
    rep = _gate(tmp_path, files)
    assert not _crit(rep, "e2e")["passed"] and "resume" in _crit(rep, "e2e")["note"]


def test_the_note_says_the_resume_was_forced(tmp_path):
    assert "forced" in _crit(_gate(tmp_path), "e2e")["note"]


# ---------------------------------------------------------------- 2. numerics

def test_a_non_finite_loss_fails_numerics(tmp_path):
    files = _inputs()
    files["train/log.jsonl"].append({"t": 1.0, "event": "micro", "micro_step": 7, "loss": float("nan"),
                                     "loss_ce": 2.0, "finite": False})
    rep = _gate(tmp_path, files)
    assert not _crit(rep, "num")["passed"] and "non-finite loss" in _crit(rep, "num")["note"]


def test_an_applied_non_finite_gradient_fails_numerics(tmp_path):
    rep = _gate(tmp_path, _edit("train/summary.json", lambda s: s.update(nonfinite_grad_applied=1)))
    assert not _crit(rep, "num")["passed"]


def test_a_collapsed_scaler_scale_fails_numerics(tmp_path):
    rep = _gate(tmp_path, _edit("train/summary.json", lambda s: s.update(min_scale=0.5)))
    assert not _crit(rep, "num")["passed"] and "0.5" in _crit(rep, "num")["note"]


def test_skipped_fp16_steps_are_reported_not_failed(tmp_path):
    crit = _crit(_gate(tmp_path), "num")
    assert crit["passed"] and "1 fp16 step" in crit["note"]


def test_scaler_accounting_from_the_log_when_the_summary_lacks_it():
    events = _log_events()
    assert G.scaler_steps(events) == (1, 0)  # opt 2: inf grad norm, scale halved -> skipped, not applied
    events.append({"event": "opt", "opt_step": 502, "scale": 32768.0, "grad_norm": float("nan")})
    assert G.scaler_steps(events) == (1, 1)  # NaN grad norm without a scale drop -> applied
    first = [{"event": "start", "fp16": True}, {"event": "opt", "opt_step": 1, "scale": 32768.0,
                                                "grad_norm": float("inf")}]
    assert G.scaler_steps(first) == (1, 0)  # the first step is compared with GradScaler's init scale 2**16
    cpu = [{"event": "start", "fp16": False}, {"event": "opt", "opt_step": 1, "scale": 1.0, "grad_norm": float("inf")}]
    assert G.scaler_steps(cpu) == (0, 1)  # no GradScaler on CPU: nothing is ever skipped


def test_log_derived_applied_gradients_fail_numerics(tmp_path):
    files = _inputs()
    del files["train/summary.json"]["nonfinite_grad_applied"]
    files["train/log.jsonl"].append({"t": 1.0, "event": "opt", "opt_step": 502, "scale": 32768.0,
                                     "grad_norm": float("inf")})
    assert not _crit(_gate(tmp_path, files), "num")["passed"]


def test_a_cpu_run_passes_numerics_with_the_disabled_scaler(tmp_path):
    files = _inputs()
    files["train/summary.json"].update(device="cpu", min_scale=1.0, opt_steps_skipped=0)
    files["train/log.jsonl"] = [dict(e, **({"scale": 1.0, "grad_norm": 1.0} if e["event"] == "opt" else {}),
                                     **({"fp16": False} if e["event"] == "start" else {}))
                                for e in files["train/log.jsonl"]]
    assert _crit(_gate(tmp_path, files), "num")["passed"]


# ---------------------------------------------------------------- 3-4. accuracy criteria

@pytest.mark.parametrize("zs, maj, ok", [(0.60, 0.03, True), (0.601, 0.03, False), (0.40, 0.50, True),
                                          (0.40, 0.501, False)])
def test_beats_trivial_needs_ten_points_over_zero_shot_and_twenty_over_majority(tmp_path, zs, maj, ok):
    files = _inputs()
    files["eval_zeroshot_val.json"]["post"]["macro_f1"] = zs
    files["baselines.json"]["schemes"]["c10"]["b1_majority"]["val"]["macro_f1"] = maj
    assert _crit(_gate(tmp_path, files), "beats")["passed"] is ok


@pytest.mark.parametrize("b3, ok", [(0.75, True), (0.7501, False), (0.60, True)])
def test_near_tfidf_allows_at_most_five_points_below_b3(tmp_path, b3, ok):
    files = _inputs()
    files["baselines.json"]["schemes"]["c10"]["b3_tfidf_lr"]["val"]["post"]["macro_f1"] = b3
    crit = _crit(_gate(tmp_path, files), "near")
    assert crit["passed"] is ok
    if not ok:
        assert "stop and debug" in crit["note"]


def test_the_nine_class_headline_is_used_when_event_is_too_small(tmp_path):
    files = _inputs()
    files["data/data_report.json"].update(headline_classes=9, event_id_pool=120)
    files["eval_e1_val.json"]["post"]["macro_f1"] = 0.10  # the 10-class value would fail both criteria
    rep = _gate(tmp_path, files)
    assert _crit(rep, "beats")["passed"] and _crit(rep, "near")["passed"]
    assert _crit(rep, "beats")["value"] == pytest.approx(0.72)
    assert "9-class" in rep["info"]["rows"]["Headline metric"]


def test_thresholds_come_from_the_config(tmp_path):
    rep = _gate(tmp_path, cfg_over={"gate.min_over_zero_shot": 0.40})
    assert not _crit(rep, "beats")["passed"] and "+40.0" in _crit(rep, "beats")["threshold"]


# ---------------------------------------------------------------- information rows

def test_info_rows_cover_the_run(tmp_path):
    rows = _gate(tmp_path)["info"]["rows"]
    wall = rows["E1 wall-clock"]
    assert "2 processes" in wall and "1.96 h" in wall  # (1864-1000) + (8100-1900) seconds of work
    assert "0.431" in rows["Seconds per micro-step; peak VRAM"] and "8.6" in rows["Seconds per micro-step; peak VRAM"]
    assert "750" in rows["Best checkpoint"]
    assert "1.21" in rows["Temperature; ECE pre -> post (val)"] and "0.06" in rows["Temperature; ECE pre -> post (val)"]
    assert "acc@80 0.9000" in rows["E1 val (post-T)"] and "0 truncated" in rows["Epochs / stop"]
    assert "C 4" in rows["TF-IDF+LR (B3)"] and "T 0.9" in rows["TF-IDF+LR (B3)"]
    assert rows["Zero-shot laya-multilingual (val)"] == "not run"
    assert "10-class" in rows["Headline metric"] and "450" in rows["Headline metric"]


def test_zero_shot_multilingual_is_shown_when_present(tmp_path):
    files = _inputs()
    files["eval_zeroshot_ml_val.json"] = {"model": "laya_ml", "post": _metrics(0.33, 0.34)}
    assert "0.3300" in _gate(tmp_path, files)["info"]["rows"]["Zero-shot laya-multilingual (val)"]


def test_cpu_bench_table_compares_with_the_budget_as_information(tmp_path):
    files = _edit("bench_cpu.json", lambda b: b["results"][0].update(p95_ms=900.0))
    rep = _gate(tmp_path, files)
    table = rep["info"]["bench"]
    assert [r["threads"] for r in table] == [1, 2]
    assert table[0]["within_budget"] is False and table[1]["within_budget"] is True
    assert rep["verdict"] == "PASS"  # the CPU budget is Phase 5's decision rule, not the gate
    md = (tmp_path / "runs" / "e1" / "gate_report.md").read_text(encoding="utf-8")
    assert "| 1 |" in md and "900" in md


@pytest.mark.parametrize("obj", [
    [{"threads": 1, "p95_ms": 10.0, "rps": 5.0}],
    {"runs": [{"threads": 1, "p95_ms": 10.0, "records_per_s": 5.0}]},
    {"threads": [1], "results": {"1": {"p95_ms": 10.0, "batch_rps": 5.0}}},
])
def test_bench_rows_accepts_the_plausible_layouts(obj):
    rows = G.bench_rows(obj)
    assert len(rows) == 1 and rows[0]["threads"] == 1 and rows[0]["p95_ms"] == 10.0 and rows[0]["rps"] == 5.0


def test_trap_candidates_are_counted_and_annotation_is_pending(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "trap_candidates.csv").write_text(
        'pattern,name\nbank_word,"Bank Cafe"\nmulti_category,"Line\nbreak Ltd"\n', encoding="utf-8")
    row = _gate(tmp_path)["info"]["rows"]["Trap candidates"]
    assert row.startswith("2 candidates") and "annotation pending" in row
    (tmp_path / "data" / "trap.jsonl").write_text('{"id": "trap-000000"}\n', encoding="utf-8")
    assert "1 trap items merged" in _gate(tmp_path)["info"]["rows"]["Trap candidates"]


def test_data_fingerprint_is_checked_against_the_frozen_manifest(tmp_path):
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({"fingerprint_sha256": "ab" * 32}), encoding="utf-8")
    row = _gate(tmp_path, cfg_over={"data.frozen_manifest": str(manifest)})["info"]["rows"]["Data fingerprint"]
    assert "matches the frozen manifest" in row
    manifest.write_text(json.dumps({"fingerprint_sha256": "cd" * 32}), encoding="utf-8")
    row = _gate(tmp_path, cfg_over={"data.frozen_manifest": str(manifest)})["info"]["rows"]["Data fingerprint"]
    assert "DIFFERS" in row
    unfrozen = _gate(tmp_path, cfg_over={"data.frozen_manifest": None})  # independent of the project config
    assert "not frozen" in unfrozen["info"]["rows"]["Data fingerprint"]


def test_wall_clock_sums_process_segments():
    active, span, n = G.wall_clock(_log_events())
    assert n == 2 and active == pytest.approx(864 + 6200) and span == pytest.approx(7100)


def test_gate_module_stays_under_400_lines():
    n = len((ROOT / "src" / "laya_poc" / "gate.py").read_text(encoding="utf-8").splitlines())
    assert n < 400, n
