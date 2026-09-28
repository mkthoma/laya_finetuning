"""Phase 7 GPU timing (laya_poc.phase7_timing) on synthetic run archives: which field is one scoring pass per run
kind, rows through the model, the per-group summary (median over seeds and splits with >= 1000 scored rows), the
training-time bases, and the CLI (md + json, metrics only)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from laya_poc import phase7_timing as T

LABELS = ["arts", "services", "community", "dining", "event", "health", "outdoors", "retail", "sports", "travel"]


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def ev(n: int, *, seconds: float, post: float = 0.0, pre: float = 0.0, gated: int = 0, device: str = "cuda",
       batch: int | None = 64, **extra) -> dict:
    return {"model": "m", "split": "s", "labels": LABELS, "n": n, "n_unlabelled": 0,
            "n_abstained_no_evidence": gated, "n_scored": n - gated, "seconds_post": post, "seconds_pre": pre,
            "device": device, "batch_size": batch, "cpu_fallback": False, "seconds": seconds,
            "rows": "/content/laya_poc/data/s.jsonl", **extra}


def make_run(root: Path, name: str, *, arm: str, model: str, seed=None, kind: str | None = None,
             timing: dict, seconds: float, evals: dict, summary: dict | None = None,
             calibration: dict | None = None, start: dict | None = None, head_only: bool = False) -> Path:
    d = root / name
    done = {"run_name": name, "arm": arm, "model": model, "scheme": "c10", "seed": seed, "subset": None,
            "head_only": head_only, "card": "G4", "finished_at": "2026-09-27T18:43:52+00:00", "seconds": seconds}
    if kind:
        done["kind"] = kind
    _write(d / "done.json", done)
    _write(d / "timing.json", timing)
    for split, payload in evals.items():
        _write(d / "eval" / f"{split}.json", {**payload, "split": split})
    if summary is not None:
        _write(d / "train" / "summary.json", summary)
    if calibration is not None:
        _write(d / "calibration.json", calibration)
    if start is not None:
        (d / "train").mkdir(parents=True, exist_ok=True)
        (d / "train" / "log.jsonl").write_text(json.dumps({"t": 1.0, "event": "start", **start}) + "\n",
                                                encoding="utf-8")
    return d


def laya_run(root: Path, seed: int, post: float) -> Path:
    return make_run(root, f"fsq-c10-E3-laya_ml-s{seed}", arm="E3", model="laya_ml", seed=seed,
                    timing={"train": 222.6, "export_check": 9.4, "evaluate": 31.9}, seconds=264.0,
                    start={"fp16": True, "device": "cuda"},
                    evals={"val": ev(3000, seconds=4.97, post=post, pre=post),
                           "test_id": ev(3000, seconds=4.88, post=post, pre=2.49),
                           "trap_candidates": ev(700, seconds=1.11, post=0.1, pre=0.1),
                           "stripped_test": ev(1000, seconds=0.62, gated=1000)})


def b4_run(root: Path, seed: int, sec: float) -> Path:
    return make_run(root, f"fsq-c10-B4-mmbert_small-s{seed}", arm="B4", model="mmbert_small", seed=seed,
                    kind="small_encoder", timing={"small_encoder": 92.7}, seconds=92.7,
                    summary={"seconds": 79.79, "fp16": True, "device": "cuda", "gpu": "NVIDIA RTX PRO 6000"},
                    evals={"val": ev(3000, seconds=sec), "test_id": ev(3000, seconds=sec),
                           "stripped_test": ev(1000, seconds=0.1, gated=1000)})


def b5_run(root: Path) -> Path:
    extra = {"scoring": "per_key", "batch_size": 64, "dtype": "bfloat16", "device": "cuda", "load_seconds": 21.7,
             "seconds": 817.9, "gpu": "NVIDIA RTX PRO 6000"}
    return make_run(root, "fsq-c10-B5-qwen3_4b", arm="B5", model="qwen3_4b", kind="llm",
                    timing={"llm_baseline": 819.9}, seconds=819.9, calibration={"T": 1.2, "extra": extra},
                    evals={"val": ev(500, seconds=39.65, subset=True), "test_id": ev(2000, seconds=160.0, subset=True),
                           "ood_brand": ev(2000, seconds=200.0, subset=True)})


def b3_run(root: Path) -> Path:
    extra = {"kind": "tfidf_lr", "tfidf": {"seconds": 4.87}, "seconds_fit": 20.04}
    return make_run(root, "fsq-c10-B3-tfidf_lr", arm="B3", model="tfidf_lr", kind="tfidf_lr",
                    timing={"baseline": 29.2}, seconds=29.2, calibration={"T": 1.1, "extra": extra},
                    evals={"test_id": ev(3000, seconds=0.03, post=0.45, device="cpu", batch=None)})


def b1_run(root: Path) -> Path:
    return make_run(root, "fsq-c10-B1-majority", arm="B1", model="majority", kind="majority",
                    timing={"baseline": 1.2}, seconds=1.2, calibration={"T": 1.0, "extra": {}},
                    evals={"test_id": ev(3000, seconds=0.02, device="cpu", batch=None)})


@pytest.fixture()
def roots(tmp_path):
    p3, p4 = tmp_path / "p3", tmp_path / "p4"
    laya_run(p3, 11, 2.33)
    laya_run(p3, 22, 2.50)
    _write(p3 / "env.json", {"card": "G4", "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"})
    (p3 / "logs").mkdir()
    b4_run(p4, 11, 0.48)
    b4_run(p4, 22, 0.50)
    b5_run(p4)
    b3_run(p4)
    b1_run(p4)
    return [p3, p4]


def _split(result: dict, run: str, split: str) -> dict:
    return next(r for r in result["splits"] if r["run"] == run and r["split"] == split)


def _summary(result: dict, group: str) -> dict:
    return next(s for s in result["summary"] if s["group"] == group)


def test_laya_precision_unknown_without_evals_is_not_guessed(tmp_path):
    run = T.load_run(make_run(tmp_path, "fsq-c10-E2-laya-s11", arm="E2", model="laya", seed=11,
                              timing={"train": 1.0}, seconds=2.0, evals={}))
    assert run.device is None and T.precision(run) is None


def test_first_start_skips_bad_and_non_object_lines(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"event": "step"\n[1, 2]\n7\n{"event": "start", "fp16": true}\n', encoding="utf-8")
    assert T.first_start(log) == {"event": "start", "fp16": True}


def test_markdown_table_escapes_pipes_and_newlines():
    from laya_poc.phase7_timing_md import table
    out = table(["a", "b"], [["x|y", "one\ntwo"]])
    assert out[-1] == "| x\\|y | one two |"


def test_laya_one_pass_is_seconds_post_over_answered_rows(roots):
    res = T.build_timing(roots)
    row = _split(res, "fsq-c10-E3-laya_ml-s11", "test_id")
    assert row["one_pass_s"] == 2.33 and row["rows_forwarded"] == 3000
    assert row["rows_per_s"] == pytest.approx(3000 / 2.33, abs=0.1)
    gated = _split(res, "fsq-c10-E3-laya_ml-s11", "stripped_test")
    assert gated["rows_forwarded"] == 0 and gated["rows_per_s"] is None


def test_small_encoder_one_pass_is_seconds_over_every_row(roots):
    res = T.build_timing(roots)
    row = _split(res, "fsq-c10-B4-mmbert_small-s11", "test_id")
    assert row["one_pass_s"] == 0.48 and row["rows_forwarded"] == 3000
    assert row["rows_per_s"] == pytest.approx(6250.0, abs=0.1)
    assert row["precision"] == "fp16 autocast"
    stripped = _split(res, "fsq-c10-B4-mmbert_small-s11", "stripped_test")
    assert stripped["rows_forwarded"] == 1000 and stripped["n_scored"] == 0     # scored, then gated


def test_llm_one_pass_is_seconds_with_val_t_fit_and_per_key_noted(roots):
    res = T.build_timing(roots)
    row = _split(res, "fsq-c10-B5-qwen3_4b", "test_id")
    assert row["rows_per_s"] == pytest.approx(12.5) and row["precision"] == "bfloat16"
    assert any("temperature" in n for n in _split(res, "fsq-c10-B5-qwen3_4b", "val")["notes"])
    notes = " ".join(_summary(res, "B5 qwen3_4b c10")["notes"])
    assert "per_key" in notes and "6 items" in notes


def test_cpu_baselines_use_seconds_post_and_say_cpu(roots):
    res = T.build_timing(roots)
    row = _split(res, "fsq-c10-B3-tfidf_lr", "test_id")
    assert row["device"] == "cpu" and row["one_pass_s"] == 0.45
    assert row["rows_per_s"] == pytest.approx(3000 / 0.45, abs=0.1)
    b1 = _split(res, "fsq-c10-B1-majority", "test_id")
    assert b1["rows_per_s"] is None and any("resolution" in n for n in b1["notes"])


def test_run_rows_train_scoring_and_total(roots):
    runs = {r["run"]: r for r in T.build_timing(roots)["runs"]}
    laya = runs["fsq-c10-E3-laya_ml-s11"]
    assert laya["train_s"] == 222.6 and laya["eval_step_s"] == 31.9 and laya["export_check_s"] == 9.4
    assert laya["total_s"] == 264.0 and laya["train_precision"] == "fp16 AMP"
    assert laya["scoring_s"] == pytest.approx(2.33 + 2.33 + 0.1)             # one pass per scored split
    assert runs["fsq-c10-B4-mmbert_small-s11"]["train_s"] == 79.79
    assert runs["fsq-c10-B3-tfidf_lr"]["train_s"] == pytest.approx(24.91)
    assert runs["fsq-c10-B3-tfidf_lr"]["device"] == "cpu"
    assert runs["fsq-c10-B5-qwen3_4b"]["train_s"] is None
    assert runs["fsq-c10-B5-qwen3_4b"]["load_s"] == 21.7


def test_summary_medians_over_seeds_and_large_splits(roots):
    res = T.build_timing(roots)
    laya = _summary(res, "E3 laya_ml c10")
    # val and test_id (3000 scored) of both seeds; trap_candidates (700) and stripped_test (0 scored) left out
    expected = sorted([3000 / 2.33, 3000 / 2.33, 3000 / 2.50, 3000 / 2.50])
    assert laya["n_points"] == 4 and laya["splits_used"] == ["test_id", "val"]
    assert laya["rows_per_s_median"] == pytest.approx((expected[1] + expected[2]) / 2, abs=0.1)
    assert laya["train_min_median"] == pytest.approx(222.6 / 60, abs=0.01)
    assert laya["n_runs"] == 2 and laya["seeds"] == [11, 22] and laya["card"] == "G4"
    b5 = _summary(res, "B5 qwen3_4b c10")
    assert b5["splits_used"] == ["ood_brand", "test_id"]                   # val subset (500) excluded
    assert b5["rows_per_s_median"] == pytest.approx((12.5 + 10.0) / 2)
    assert _summary(res, "B4 mmbert_small c10")["short_pass"] is True        # sub-second passes flagged


def test_throughput_per_group_and_split(roots):
    res = T.build_timing(roots)
    row = next(t for t in res["throughput"] if t["group"] == "B4 mmbert_small c10" and t["split"] == "test_id")
    assert row["n_runs"] == 2 and row["one_pass_s_median"] == pytest.approx(0.49)
    assert row["rows_per_s_median"] == pytest.approx((6250.0 + 6000.0) / 2, abs=0.1)
    assert row["batch_size"] == 64 and row["device"] == "cuda" and row["card"] == "G4"


def test_card_names_from_env(roots):
    assert T.build_timing(roots)["cards"] == {"G4": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}


def test_cli_writes_md_and_json_metrics_only(roots, tmp_path):
    out = tmp_path / "phase7" / "gpu_timing.md"
    args = [a for r in roots for a in ("--runs-root", str(r))] + ["--out", str(out)]
    assert T.main(args) == 0
    md, js = out.read_text(encoding="utf-8"), json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert js["schema"] == T.SCHEMA and js["min_scored_rows"] == 1000
    assert "not batch-1 latency" in md and "seconds_post" in md
    assert "E3 laya_ml c10" in md and "B4 mmbert_small c10" in md
    assert "| B3 tfidf_lr c10 |" in md and "cpu" in md
    both = md + json.dumps(js)
    assert not re.search(r"\b[a-z_]+-\d{6}\b", both)                          # no row ids


def test_cli_error_without_runs(tmp_path, capsys):
    out = tmp_path / "t.md"
    assert T.main(["--runs-root", str(tmp_path / "missing"), "--out", str(out)]) == 1
    assert "no finished run" in capsys.readouterr().err
