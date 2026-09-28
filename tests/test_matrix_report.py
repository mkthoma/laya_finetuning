"""Phase 3 results: run records, results/runs.csv (spec §1 header) and `matrix report` (per-arm mean and range
over seeds, ID->OOD gaps, ECE pre->post, seed-variance flag, E6 learning curve, B2 rows, Phase 3 exit check),
all on synthetic run directories (no model)."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest
import yaml

from laya_poc import matrix as M
from laya_poc import matrix_plan as P
from laya_poc import matrix_report as R
from laya_poc import matrix_results as S

SPEC_COLUMNS = ["run_name", "arm", "model", "scheme", "seed", "subset", "head_only", "card", "epochs_run",
                "stop_reason", "best_opt_step", "T", "clamped", "val_macro_f1", "val_macro_f1_9", "val_acc",
                "val_ece_pre", "val_ece_post", "test_id_macro_f1", "ood_country_macro_f1", "ood_script_macro_f1",
                "ood_brand_macro_f1", "trap_candidates_acc", "stripped_false_confident", "order_invariance",
                "train_seconds", "run_seconds"]
POOLS = ("test_id", "ood_country", "ood_script", "ood_brand")


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _eval(f1: float, *, ece_pre: float = 0.09, ece_post: float = 0.04, T: float = 1.2, acc: float | None = None,
          **extra) -> dict:
    side = {"n": 100, "macro_f1": f1, "macro_f1_9": f1 + 0.02, "acc": f1 + 0.03 if acc is None else acc}
    return {"temperature": T, "pre": {**side, "ece": ece_pre}, "post": {**side, "ece": ece_post}, **extra}


def write_run(runs: Path, spec: P.RunSpec, *, val: float = 0.6, gap: float = 0.05, T: float = 1.2,
              best: bool = True, done: bool = True, micro_steps: int = 16, events: list | None = None) -> Path:
    """A finished run directory in the spec §1 layout with the given headline numbers."""
    rd = runs / spec.name
    evals = {"val": _eval(val, T=T), "test_id": _eval(val - 0.01, ece_pre=0.1, ece_post=0.05, T=T),
             **{p: _eval(val - 0.01 - gap, T=T) for p in POOLS[1:]},
             "stripped_test": {"temperature": T, "pre": None, "post": None, "false_confident_rate": 0.02,
                               "abstain_rate_at_tau": 0.9},
             "trap_candidates": _eval(0.3, acc=0.35, T=T)}
    for split, ev in evals.items():
        _json(rd / "eval" / f"{split}.json", ev)
    _json(rd / "order_invariance.json", {"split": "test_id", "n": 100, "perms": 5, "mean_agreement": 0.995,
                                         "passed_99": True})
    if not spec.zero_shot:
        n = spec.subset or 25000
        log = events or [{"t": 0.0, "event": "start", "n_items_per_epoch": [n] * 4, "micro_batch": 32},
                         {"t": 600.0, "event": "done"}]
        (rd / "train").mkdir(parents=True, exist_ok=True)
        (rd / "train" / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log), encoding="utf-8")
        _json(rd / "train" / "summary.json", {"stop_reason": "epochs", "best_opt_step": 400, "epochs": 4,
                                              "micro_steps": micro_steps})
        if best:
            (rd / "train" / "best").mkdir(parents=True, exist_ok=True)
            (rd / "train" / "best" / "model.safetensors").write_bytes(b"w")
        _json(rd / "export_check.json", {"T": T, "clamped": False, "passed": True})
    if done:
        _json(rd / "done.json", {"run_name": spec.name, "arm": spec.arm, "model": spec.model, "scheme": spec.scheme,
                                 "seed": spec.seed, "subset": spec.subset, "head_only": spec.head_only, "card": "G4",
                                 "finished_at": "2026-09-28T10:00:00+00:00", "seconds": 700.0})
    return rd


def spec(name: str, cfg) -> P.RunSpec:
    return next(s for s in P.expand_specs(cfg) if s.name == name)


@pytest.fixture()
def small_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["phase3"]["arms"] = [{"id": "E2", "model": "laya", "scheme": "c10", "seeds": [11, 22, 33]},
                           {"id": "E4", "model": "laya", "scheme": "c10", "seeds": [11], "freeze_encoder": True},
                           {"id": "E5", "model": "laya_ml", "scheme": "c7", "seeds": [11]},
                           {"id": "E6", "model": "laya", "scheme": "c10", "seeds": [11], "train_subset": [1000, 3000]}]
    c["phase3"]["zero_shot"] = ["laya"]
    return c


def complete_matrix(runs: Path, cfg, **over) -> None:
    vals = {"fsq-c10-E2-laya-s11": 0.60, "fsq-c10-E2-laya-s22": 0.62, "fsq-c10-E2-laya-s33": 0.66,
            "fsq-c10-E6-laya-s11-n1000": 0.40, "fsq-c10-E6-laya-s11-n3000": 0.50, "fsq-c10-B2-laya-zs": 0.30}
    for s in P.expand_specs(cfg):
        write_run(runs, s, val=vals.get(s.name, 0.55), **over.get(s.name, {}))


# ---------------------------------------------------------------- records and runs.csv

def test_csv_header_is_the_spec_contract():
    assert list(S.CSV_COLUMNS) == SPEC_COLUMNS


def test_run_record_of_a_trained_run(tmp_path, small_cfg):
    events = [{"t": 0.0, "event": "start", "n_items_per_epoch": [100, 100, 100, 100], "micro_batch": 10},
              {"t": 100.0, "event": "micro"}, {"t": 1000.0, "event": "start", "n_items_per_epoch": [100] * 4,
                                                "micro_batch": 10}, {"t": 1050.0, "event": "done"}]
    rd = write_run(tmp_path, spec("fsq-c10-E2-laya-s22", small_cfg), val=0.62, gap=0.07, micro_steps=25,
                   events=events)
    rec = S.run_record(rd)
    assert (rec["run_name"], rec["arm"], rec["seed"], rec["card"]) == ("fsq-c10-E2-laya-s22", "E2", 22, "G4")
    assert rec["epochs_run"] == 2.5 and rec["train_seconds"] == 150.0 and rec["run_seconds"] == 700.0
    assert (rec["T"], rec["clamped"], rec["stop_reason"], rec["best_opt_step"]) == (1.2, False, "epochs", 400)
    assert rec["val_macro_f1"] == 0.62 and rec["val_ece_pre"] == 0.09 and rec["val_ece_post"] == 0.04
    assert rec["ood_brand_macro_f1"] == pytest.approx(0.62 - 0.01 - 0.07)
    assert rec["trap_candidates_acc"] == 0.35 and rec["stripped_false_confident"] == 0.02
    assert rec["order_invariance"] == 0.995 and rec["has_best"] is True and rec["n_train"] == 100
    row = S.csv_row(rec)
    assert list(row) == SPEC_COLUMNS and row["head_only"] == "false" and row["subset"] == "" and row["T"] == "1.2"


def test_run_record_of_a_zero_shot_run_takes_the_shipped_T(tmp_path, small_cfg):
    rec = S.run_record(write_run(tmp_path, spec("fsq-c10-B2-laya-zs", small_cfg), T=0.8))
    assert rec["zero_shot"] and rec["T"] == 0.8 and rec["clamped"] is None
    row = S.csv_row(rec)
    assert row["epochs_run"] == "" and row["train_seconds"] == "" and row["seed"] == "" and row["T"] == "0.8"


def test_runs_csv_has_one_row_per_done_run_sorted(tmp_path, small_cfg):
    runs = tmp_path / "runs"
    complete_matrix(runs, small_cfg)
    write_run(runs, spec("fsq-c7-E5-laya_ml-s11", small_cfg), done=False)  # overwrite: not done
    (runs / "fsq-c7-E5-laya_ml-s11" / "done.json").unlink()
    out = S.refresh_runs_csv(runs, tmp_path / "results" / "runs.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        names = [r["run_name"] for r in reader]
        assert reader.fieldnames == SPEC_COLUMNS
    assert names == sorted(names) and "fsq-c7-E5-laya_ml-s11" not in names and len(names) == 7


# ---------------------------------------------------------------- aggregation

def _group(result: dict, label_part: str) -> dict:
    return next(g for g in result["groups"] if g["label"].startswith(label_part))


def build(tmp_path: Path, cfg) -> dict:
    runs = tmp_path / "runs"
    csv_path = S.refresh_runs_csv(runs, tmp_path / "results" / "runs.csv")
    return R.build_report(cfg, P.expand_specs(cfg), runs, csv_path)


def test_groups_average_over_seeds_with_the_range(tmp_path, small_cfg):
    complete_matrix(tmp_path / "runs", small_cfg)
    res = build(tmp_path, small_cfg)
    e2 = _group(res, "E2 laya c10")
    assert e2["seeds"] == [11, 22, 33] and e2["n_runs"] == 3
    assert e2["metrics"]["val_macro_f1"] == pytest.approx({"mean": (0.60 + 0.62 + 0.66) / 3, "min": 0.60,
                                                           "max": 0.66, "n": 3})
    assert e2["metrics"]["gap_ood_country"]["mean"] == pytest.approx(0.05)
    assert e2["metrics"]["test_id_ece_pre"]["mean"] == pytest.approx(0.1)
    labels = [g["label"] for g in res["groups"]]
    assert labels[0].startswith("B2") and any("head-only" in x for x in labels) and any("n=1000" in x for x in labels)


def test_seed_variance_flag_above_three_points(tmp_path, small_cfg):
    complete_matrix(tmp_path / "runs", small_cfg)
    res = build(tmp_path, small_cfg)
    flags = {(v["group"], v["metric"]): v for v in res["seed_variance"]}
    assert flags[("E2 laya c10", "val_macro_f1")]["flagged"] is True  # 0.66 - 0.60 = 6 points
    assert flags[("E2 laya c10", "val_macro_f1")]["range"] == pytest.approx(0.06)
    assert all(v["group"] == "E2 laya c10" for v in res["seed_variance"])  # single-seed groups have no variance


def test_seed_variance_not_flagged_within_three_points(tmp_path, small_cfg):
    runs = tmp_path / "runs"
    for name, v in (("fsq-c10-E2-laya-s11", 0.60), ("fsq-c10-E2-laya-s22", 0.61), ("fsq-c10-E2-laya-s33", 0.63)):
        write_run(runs, spec(name, small_cfg), val=v)
    res = build(tmp_path, small_cfg)
    assert not any(v["flagged"] for v in res["seed_variance"])


def test_learning_curve_adds_the_full_size_seed_11_run(tmp_path, small_cfg):
    complete_matrix(tmp_path / "runs", small_cfg)
    curve = build(tmp_path, small_cfg)["learning_curve"]
    assert [(p["n"], p["run_name"]) for p in curve] == [(1000, "fsq-c10-E6-laya-s11-n1000"),
                                                        (3000, "fsq-c10-E6-laya-s11-n3000"),
                                                        (25000, "fsq-c10-E2-laya-s11")]
    assert [p["val_macro_f1"] for p in curve] == [0.40, 0.50, 0.60] and curve[-1]["full"] is True


# ---------------------------------------------------------------- exit check

def test_exit_check_passes_when_every_run_has_best_T_and_val_metrics(tmp_path, small_cfg):
    complete_matrix(tmp_path / "runs", small_cfg)
    ex = build(tmp_path, small_cfg)["exit_check"]
    assert ex["verdict"] == "PASS" and ex["passed"] is True and ex["complete"] == ex["total"] == 8
    zs = next(r for r in ex["runs"] if r["run_name"] == "fsq-c10-B2-laya-zs")
    assert zs["best"] is None and zs["passed"]  # zero-shot: no best/ to check


def test_exit_check_fails_on_a_missing_run_best_or_T(tmp_path, small_cfg):
    runs = tmp_path / "runs"
    complete_matrix(runs, small_cfg, **{"fsq-c10-E4-laya-s11-head": {"best": False}})
    import shutil
    shutil.rmtree(runs / "fsq-c10-E2-laya-s33")
    ec = runs / "fsq-c7-E5-laya_ml-s11" / "export_check.json"
    ec.write_text(json.dumps({"T": None, "passed": False}), encoding="utf-8")
    ex = build(tmp_path, small_cfg)["exit_check"]
    assert ex["verdict"] == "FAIL" and ex["passed"] is False and ex["complete"] == 5 and ex["total"] == 8
    bad = {r["run_name"]: r["missing"] for r in ex["runs"] if not r["passed"]}
    assert bad == {"fsq-c10-E2-laya-s33": ["done.json", "best/", "T", "val metrics"],
                   "fsq-c10-E4-laya-s11-head": ["best/"], "fsq-c7-E5-laya_ml-s11": ["T"]}


# ---------------------------------------------------------------- markdown and CLI

def test_markdown_has_every_section(tmp_path, small_cfg):
    complete_matrix(tmp_path / "runs", small_cfg)
    md = R.render_markdown(build(tmp_path, small_cfg))
    for text in ("# Laya PoC: Phase 3 seed matrix", "Phase 3 exit check", "**PASS**", "ID->OOD gap",
                 "ECE pre -> post", "unannotated candidates", "Seed variance", "Investigate", "Learning curve",
                 "B2 laya c10 (zero-shot", "0.6267 (0.6000 to 0.6600)", "head-only", "false-confident"):
        assert text in md, text
    assert "cannot read Thai" in md


def test_report_cli_writes_md_json_and_runs_csv_under_work(tmp_path, small_cfg, capsys):
    work = tmp_path / "work"
    complete_matrix(work / "runs" / "p3", small_cfg)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(small_cfg, sort_keys=False), encoding="utf-8")
    assert M.main(["report", "--config", str(cfg_path), "--work", str(work)]) == 0
    md, js = work / "results" / "phase3_report.md", work / "results" / "phase3_report.json"
    assert md.is_file() and (work / "results" / "runs.csv").is_file()
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["exit_check"]["passed"] is True and data["verdict"] == "PASS" and len(data["runs"]) == 8
    assert "Phase 3 exit PASS (8/8" in capsys.readouterr().out


def test_report_on_an_empty_matrix_is_not_run_and_renders(tmp_path, small_cfg):
    res = build(tmp_path, small_cfg)
    # nothing ran: NOT RUN (never PASS); passed is None, not True
    assert res["exit_check"]["verdict"] == "NOT RUN" and res["exit_check"]["passed"] is None and res["groups"] == [] and res["learning_curve"] == []
    assert "no finished runs" in R.render_markdown(res)
