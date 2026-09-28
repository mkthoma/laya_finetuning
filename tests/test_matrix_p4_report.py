"""Phase 4 results and report (spec P4 §5) on synthetic run directories (no model): baseline records (T from
calibration.json, B4 train facts, B5 subsets), the runs.csv `kind` column, several runs roots merged, baseline
groups in every table, "Decision criterion 1 (early read)" and the Phase 4 exit check; a Phase 3 only config
still renders the Phase 3 report unchanged."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest
import yaml

from laya_poc import matrix as M
from laya_poc import matrix_baselines as B
from laya_poc import matrix_decision as D
from laya_poc import matrix_plan as P
from laya_poc import matrix_report as R
from laya_poc import matrix_results as S
from laya_poc.io_utils import write_jsonl

from test_matrix_report import SPEC_COLUMNS, _eval, _json

SPLITS = ("val", "test_id", "ood_country", "ood_script", "ood_brand", "stripped_test", "trap_candidates")
DEFAULT_F1 = {"val": 0.55, "test_id": 0.55, "ood_country": 0.45, "ood_script": 0.40, "ood_brand": 0.50,
              "trap_candidates": 0.30}


@pytest.fixture()
def p4cfg(cfg):
    """A small Phase 3 matrix (every kind of Laya group) plus the configured Phase 4 baselines (13 runs)."""
    c = copy.deepcopy(cfg)
    c["phase3"]["arms"] = [{"id": "E2", "model": "laya", "scheme": "c10", "seeds": [11, 22, 33]},
                           {"id": "E3", "model": "laya_ml", "scheme": "c10", "seeds": [11]},
                           {"id": "E4", "model": "laya", "scheme": "c10", "seeds": [11], "freeze_encoder": True},
                           {"id": "E5", "model": "laya", "scheme": "c7", "seeds": [11]},
                           {"id": "E6", "model": "laya", "scheme": "c10", "seeds": [11], "train_subset": [1000]}]
    c["phase3"]["zero_shot"] = ["laya"]
    return c


def spec(cfg: dict, name: str) -> P.RunSpec:
    return next(s for s in B.expand_all(cfg) if s.name == name)


def _stripped(T: float, extra: dict) -> dict:
    return {"temperature": T, "pre": None, "post": None, "false_confident_rate": 0.0, "abstain_rate_at_tau": 1.0,
            **extra}


def _train_files(rd: Path, T: float) -> None:
    (rd / "train" / "best").mkdir(parents=True, exist_ok=True)
    (rd / "train" / "best" / "model.safetensors").write_bytes(b"w")
    _json(rd / "train" / "summary.json", {"stop_reason": "epochs", "best_opt_step": 400, "micro_steps": 16})
    _json(rd / "export_check.json", {"T": T, "clamped": False, "passed": True})


def finished(root: Path, s: P.RunSpec, *, f1: dict | None = None, T: float = 1.1, drop_preds: tuple = (),
             done: bool = True) -> Path:
    """A run directory in the Phase 3 / Phase 4 layout with the given per-split post-T macro-F1."""
    rd, f1, extra = root / s.name, {**DEFAULT_F1, **(f1 or {})}, ({"subset": True} if s.kind == "llm" else {})
    for split in SPLITS:
        ev = _stripped(T, extra) if split == "stripped_test" else _eval(f1[split], T=T, **extra)
        _json(rd / "eval" / f"{split}.json", ev)
        if split not in drop_preds:
            write_jsonl(rd / "preds" / f"{split}.jsonl", [{"id": "x", "y": 0}])
    if P.computes_order_invariance(s):
        _json(rd / "order_invariance.json", {"split": "test_id", "n": 100, "perms": 5, "mean_agreement": 0.95,
                                             "passed_99": False})
    if s.kind == "laya" and not s.zero_shot:
        _train_files(rd, T)
    elif s.kind != "laya":
        _json(rd / "calibration.json", {"T": T, "clamped": False, "fitted_on": "val", "n": 100, "extra": {}})
    if s.kind == "small_encoder":
        _json(rd / "train" / "summary.json", {"epochs": 3, "epochs_run": 3.0, "best_epoch": 3, "seconds": 300.0})
    if done:
        _json(rd / "done.json", {"run_name": s.name, "arm": s.arm, "model": s.model, "scheme": s.scheme,
                                 "seed": s.seed, "subset": s.subset, "head_only": s.head_only, "card": "G4",
                                 **({} if s.kind == "laya" else {"kind": s.kind}),
                                 "finished_at": "2026-09-28T10:00:00+00:00", "seconds": 100.0})
    return rd


def finish_all(root: Path, cfg: dict, phase: int, **over) -> None:
    for s in B.expand_all(cfg):
        if s.phase == phase:
            finished(root, s, **over.get(s.name, {}))


def build(tmp_path: Path, cfg: dict, *roots: Path) -> dict:
    csv_path = S.refresh_runs_csv(roots, tmp_path / "results" / "runs.csv")
    return R.build_report(cfg, B.expand_all(cfg), roots, csv_path)


# ---------------------------------------------------------------- records and runs.csv

def test_baseline_records_take_T_from_calibration_json(tmp_path, p4cfg):
    rec = S.run_record(finished(tmp_path, spec(p4cfg, "fsq-c7-B3-tfidf_lr"), T=0.95))
    assert (rec["kind"], rec["T"], rec["clamped"], rec["export_check_passed"]) == ("tfidf_lr", 0.95, False, None)
    assert rec["epochs_run"] is None and rec["order_invariance"] == 0.95 and not rec["eval_subset"]
    b5 = S.run_record(finished(tmp_path, spec(p4cfg, "fsq-c10-B5-qwen3_4b")))
    assert b5["eval_subset"] is True and b5["order_invariance"] is None
    b4 = S.run_record(finished(tmp_path, spec(p4cfg, "fsq-c10-B4-mmbert_small-s22")))
    assert (b4["epochs_run"], b4["train_seconds"], b4["best_epoch"], b4["seed"]) == (3.0, 300.0, 3, 22)
    laya = S.run_record(finished(tmp_path, spec(p4cfg, "fsq-c10-E2-laya-s11"), T=1.3))
    assert laya["kind"] == "laya" and laya["T"] == 1.3


def test_runs_csv_gains_kind_at_the_end_only_with_a_baseline_row(tmp_path, p4cfg):
    root, out = tmp_path / "runs", tmp_path / "runs.csv"
    finished(root, spec(p4cfg, "fsq-c10-E2-laya-s11"))
    with S.refresh_runs_csv(root, out).open(encoding="utf-8", newline="") as fh:
        assert csv.DictReader(fh).fieldnames == SPEC_COLUMNS  # Laya only: the Phase 3 header exactly
    finished(root, spec(p4cfg, "fsq-c10-B1-prior"))
    with S.refresh_runs_csv(root, out).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = {r["run_name"]: r for r in reader}
        assert reader.fieldnames == [*SPEC_COLUMNS, "kind"]
    assert rows["fsq-c10-B1-prior"]["kind"] == "prior" and rows["fsq-c10-E2-laya-s11"]["kind"] == "laya"
    assert rows["fsq-c10-B1-prior"]["seed"] == "" and rows["fsq-c10-B1-prior"]["T"] == "1.1"


def test_several_runs_roots_merge_and_the_first_holding_a_name_wins(tmp_path, p4cfg):
    a, b = tmp_path / "a", tmp_path / "b"
    finished(a, spec(p4cfg, "fsq-c10-B1-prior"), f1={"val": 0.2})
    finished(b, spec(p4cfg, "fsq-c10-B1-prior"), f1={"val": 0.9})
    finished(b, spec(p4cfg, "fsq-c10-E2-laya-s11"))
    recs = {r["run_name"]: r for r in S.done_records((a, b))}
    assert sorted(recs) == ["fsq-c10-B1-prior", "fsq-c10-E2-laya-s11"]
    assert recs["fsq-c10-B1-prior"]["val_macro_f1"] == 0.2
    assert S.locate_run((a, b), "fsq-c10-E2-laya-s11") == b / "fsq-c10-E2-laya-s11"
    assert S.locate_run((a, b), "nowhere") == a / "nowhere"


# ---------------------------------------------------------------- groups

def test_baselines_are_grouped_like_the_laya_arms(tmp_path, p4cfg):
    finish_all(tmp_path / "p3", p4cfg, 3)
    finish_all(tmp_path / "p4", p4cfg, 4, **{f"fsq-c10-B4-modernbert_base-s{s}": {"f1": {"val": v}}
                                             for s, v in ((11, 0.50), (22, 0.52), (33, 0.57))})
    res = build(tmp_path, p4cfg, tmp_path / "p3", tmp_path / "p4")
    labels = [g["label"] for g in res["groups"]]
    assert labels[:4] == ["B1 majority c10", "B1 majority c7", "B1 prior c10", "B1 prior c7"]
    assert labels.index("B2 laya c10 (zero-shot, shipped T)") < labels.index("B3 tfidf_lr c10") < \
        labels.index("B4 mmbert_small c10") < labels.index("B5 qwen3_4b c10 (zero-shot reference, eval subsets)") < \
        labels.index("E2 laya c10")
    b4 = next(g for g in res["groups"] if g["label"] == "B4 modernbert_base c10")
    assert b4["kind"] == "small_encoder" and b4["seeds"] == [11, 22, 33]
    assert b4["metrics"]["val_macro_f1"] == pytest.approx({"mean": (0.50 + 0.52 + 0.57) / 3, "min": 0.50,
                                                           "max": 0.57, "n": 3})
    flagged = {(v["group"], v["metric"]) for v in res["seed_variance"] if v["flagged"]}
    assert ("B4 modernbert_base c10", "val_macro_f1") in flagged  # 7 points over seeds
    md = R.render_markdown(res)
    assert "| B1 prior c10 | 1 | - |" in md and "| B4 modernbert_base c10 | 3 | 11, 22, 33 |" in md
    assert "calibration.json" in md and "B1/B3/B4/B5 rows are the Phase 4 baselines" in md
    assert "# Laya PoC: Phase 3 seed matrix and Phase 4 baselines" in md


# ---------------------------------------------------------------- decision criterion 1

POOL_F1 = {  # per run: (ood_country, ood_script, ood_brand)
    "fsq-c10-E2-laya-s11": (0.43, 0.30, 0.59), "fsq-c10-E2-laya-s22": (0.44, 0.31, 0.59),
    "fsq-c10-E2-laya-s33": (0.45, 0.32, 0.59),                                    # laya: country+brand = 0.515
    "fsq-c10-E3-laya_ml-s11": (0.49, 0.50, 0.50),                                 # all three = 0.4967
    "fsq-c7-E5-laya-s11": (0.50, 0.30, 0.60),                                     # 0.55 vs B3 c7 0.475
    "fsq-c10-B3-tfidf_lr": (0.34, 0.28, 0.45),                                    # 0.395 / 0.3567
    **{f"fsq-c10-B4-modernbert_base-s{s}": (0.45, 0.20, 0.52) for s in (11, 22, 33)},  # 0.485 / 0.39
    "fsq-c10-B4-mmbert_small-s11": (0.47, 0.46, 0.48),                            # 0.475 / 0.47
    "fsq-c10-B4-mmbert_small-s22": (0.47, 0.46, 0.48), "fsq-c10-B4-mmbert_small-s33": (0.47, 0.46, 0.48),
    "fsq-c10-E4-laya-s11-head": (0.99, 0.99, 0.99), "fsq-c10-E6-laya-s11-n1000": (0.99, 0.99, 0.99),
    "fsq-c10-B2-laya-zs": (0.99, 0.99, 0.99),                                     # not candidates
}


def _pool_over() -> dict:
    return {name: {"f1": dict(zip(("ood_country", "ood_script", "ood_brand"), v))} for name, v in POOL_F1.items()}


def test_decision_criterion_1_against_the_best_trained_baseline_on_the_same_pools(tmp_path, p4cfg):
    finish_all(tmp_path / "p3", p4cfg, 3, **_pool_over())
    finish_all(tmp_path / "p4", p4cfg, 4, **_pool_over())
    rows = {r["label"]: r for r in build(tmp_path, p4cfg, tmp_path / "p3", tmp_path / "p4")["decision_criterion_1"]}
    assert sorted(rows) == ["E2 laya c10", "E3 laya_ml c10", "E5 laya c7"]  # no B2 / E4 / E6
    e2 = rows["E2 laya c10"]
    assert e2["pools"] == ["ood_country", "ood_brand"] and e2["seeds"] == [11, 22, 33]
    assert e2["laya_ood_avg"]["mean"] == pytest.approx(0.515)
    assert {b["label"]: round(b["ood_avg"]["mean"], 4) for b in e2["baselines"]} == {
        "B3 tfidf_lr c10": 0.395, "B4 mmbert_small c10": 0.475, "B4 modernbert_base c10": 0.485}
    assert e2["best"] == "B4 modernbert_base c10" and e2["lead"] == pytest.approx(0.03)
    assert e2["passed"] is True and e2["missing"] == []  # exactly 3 points passes
    e3 = rows["E3 laya_ml c10"]
    assert e3["pools"] == ["ood_country", "ood_script", "ood_brand"] and e3["best"] == "B4 mmbert_small c10"
    assert e3["lead"] == pytest.approx(0.4967 - 0.47, abs=1e-4) and e3["passed"] is False
    e5 = rows["E5 laya c7"]  # c7: B3 c7 exists, no B4 for c7
    assert e5["best"] == "B3 tfidf_lr c7" and e5["missing"] == ["B4"] and e5["passed"] is True
    md = R.render_markdown(build(tmp_path, p4cfg, tmp_path / "p3", tmp_path / "p4"))
    assert "## Decision criterion 1 (early read)" in md and "Information only" in md
    assert "| E2 laya c10 | 11, 22, 33 | ood_country, ood_brand | 0.5150 (0.5100 to 0.5200) | B4 modernbert_base c10 " \
           "| 0.4850 | +3.0 | PASS |" in md
    assert "| FAIL |" in md and "PASS (incomplete: no B4 c7)" in md


def test_decision_without_a_trained_baseline_is_not_applicable(tmp_path, p4cfg):
    finished(tmp_path / "p3", spec(p4cfg, "fsq-c10-E3-laya_ml-s11"))
    finished(tmp_path / "p4", spec(p4cfg, "fsq-c10-B1-prior"))  # B1 is not a trained baseline
    row, = build(tmp_path, p4cfg, tmp_path / "p3", tmp_path / "p4")["decision_criterion_1"]
    assert row["passed"] is None and row["lead"] is None and row["missing"] == ["B3", "B4"]
    assert D.ood_average({"ood_country_macro_f1": 0.5}, D.ood_pools("laya")) is None  # a pool is missing


# ---------------------------------------------------------------- Phase 4 exit check

def test_phase4_exit_passes_with_every_baseline_and_the_optional_b5_not_run(tmp_path, p4cfg):
    finish_all(tmp_path / "p4", p4cfg, 4, **{"fsq-c10-B5-qwen3_4b": {"done": False}})
    import shutil
    shutil.rmtree(tmp_path / "p4" / "fsq-c10-B5-qwen3_4b")
    ex = build(tmp_path, p4cfg, tmp_path / "p4")["phase4_exit_check"]
    assert ex["verdict"] == "PASS" and (ex["complete"], ex["total"]) == (12, 13)
    assert ex["skipped_optional"] == ["fsq-c10-B5-qwen3_4b"] and ex["splits"] == list(SPLITS)
    b5 = next(r for r in ex["runs"] if r["run_name"] == "fsq-c10-B5-qwen3_4b")
    assert b5["skipped"] and b5["passed"] and b5["missing"] == []


def test_phase4_exit_fails_on_missing_preds_or_an_unfinished_run(tmp_path, p4cfg):
    finish_all(tmp_path / "p4", p4cfg, 4, **{"fsq-c7-B3-tfidf_lr": {"drop_preds": ("ood_script",)},
                                             "fsq-c10-B4-mmbert_small-s33": {"done": False}})
    ex = build(tmp_path, p4cfg, tmp_path / "p4")["phase4_exit_check"]
    bad = {r["run_name"]: r["missing"] for r in ex["runs"] if not r["passed"]}
    assert ex["verdict"] == "FAIL" and ex["complete"] == 11 and bad == {
        "fsq-c7-B3-tfidf_lr": ["preds/ood_script.jsonl"], "fsq-c10-B4-mmbert_small-s33": ["done.json"]}
    md = R.render_markdown(build(tmp_path, p4cfg, tmp_path / "p4"))
    assert "## Phase 4 exit check" in md and "Verdict: **FAIL** (11/13" in md
    assert "MISSING preds/ood_script.jsonl" in md


def test_a_phase3_only_config_renders_the_phase3_report_unchanged(tmp_path, p4cfg):
    c = {k: v for k, v in p4cfg.items() if k != "phase4"}
    finish_all(tmp_path / "p3", c, 3)
    res = build(tmp_path, c, tmp_path / "p3")
    md = R.render_markdown(res)
    assert res["phase4_exit_check"] is None and res["exit_check"]["verdict"] == "PASS"
    assert md.startswith("# Laya PoC: Phase 3 seed matrix\n") and "Phase 4" not in md
    assert "Decision criterion" not in md and "calibration.json" not in md


# ---------------------------------------------------------------- CLI

def _cfg_file(cfg: dict, tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_report_cli_merges_a_phase3_and_a_phase4_archive(tmp_path, p4cfg, capsys):
    work = tmp_path / "work"
    finish_all(work / "runs" / "p3_colab" / "runs" / "p3", p4cfg, 3, **_pool_over())
    finish_all(work / "runs" / "p4_colab" / "runs" / "p4", p4cfg, 4, **_pool_over())
    argv = ["report", "--config", str(_cfg_file(p4cfg, tmp_path)), "--work", str(work), "--runs-root",
            "runs/p3_colab/runs/p3", "--runs-root", "runs/p4_colab/runs/p4", "--out", "results/phase4_report.md"]
    assert M.main(argv) == 0
    out = capsys.readouterr().out
    assert "Phase 3 exit PASS (8/8 runs complete); Phase 4 exit PASS (13/13 baseline runs complete)" in out
    data = json.loads((work / "results" / "phase4_report.json").read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS" and data["phase4_exit_check"]["passed"] is True and len(data["runs"]) == 21
    assert len(data["runs_roots"]) == 2 and len(data["decision_criterion_1"]) == 3
    with (work / "results" / "runs.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames[-1] == "kind" and len(list(reader)) == 21


def test_report_of_a_phase4_session_alone(tmp_path, p4cfg, capsys):
    work = tmp_path / "work"
    finish_all(work / "runs" / "p4", p4cfg, 4)
    assert M.main(["report", "--config", str(_cfg_file(p4cfg, tmp_path)), "--work", str(work)]) == 0
    assert "Phase 3 exit NOT RUN (0/8 runs complete); Phase 4 exit PASS (13/13" in capsys.readouterr().out
    md = (work / "results" / "phase3_report.md").read_text(encoding="utf-8")
    assert "No fine-tuned Laya arm has finished" in md and "--runs-root <p3 runs>" in md
