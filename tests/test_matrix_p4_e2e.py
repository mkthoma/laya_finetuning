"""One tiny real Phase 4 matrix on CPU: B1 (majority + prior) and B3 (char TF-IDF + LR) through real
`python -m laya_poc.baseline_runs` subprocesses -> done.json -> runs.csv (with `kind`) -> report with the Phase 4
exit check. Plumbing only; the numbers mean nothing. B4/B5 need a model download or a tiny stand-in and are
covered by the mocked tests and the notebook's local dry run (tools/dry_run_p4_local.py)."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import shutil

import pytest
import yaml

from laya_poc import matrix as M

from synth import write_synthetic_data_dir

pytest.importorskip("sklearn")
if importlib.util.find_spec("laya_poc.baseline_runs") is None:
    pytest.skip("laya_poc.baseline_runs does not exist yet", allow_module_level=True)


def test_tiny_baseline_matrix_end_to_end_on_cpu(cfg, tmp_path):
    work = tmp_path / "work"
    data = write_synthetic_data_dir(work / "data", n_train=48, n_val=24)
    shutil.copyfile(data / "train_e0.jsonl", data / "train.jsonl")  # the clean train split B1/B3 fit on
    c = copy.deepcopy(cfg)
    c["phase3"] = {**c["phase3"], "arms": [], "zero_shot": [], "eval_splits": ["val", "test_id"],
                   "order_invariance": {"split": "test_id", "n": 8, "perms": 2, "seed": 1}}
    c["phase4"] = {**c["phase4"], "arms": [{"id": "B1", "kind": "majority", "schemes": ["c10"]},
                                           {"id": "B3", "kind": "tfidf_lr", "schemes": ["c10"]}]}
    c["baselines"] = {**c["baselines"], "lr": {**c["baselines"]["lr"], "C_grid": [1.0]}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    common = ["--config", str(cfg_path), "--work", str(work)]
    rc = M.main(["run", "--only", "all", *common, "--device", "cpu", "--card", "CPU"])
    runs = work / "runs" / "p4"
    logs = {p.relative_to(runs).as_posix(): p.read_text(encoding="utf-8")[-800:] for p in runs.glob("*/logs/*.log")}
    assert rc == 0, logs
    names = ["fsq-c10-B1-majority", "fsq-c10-B1-prior", "fsq-c10-B3-tfidf_lr"]
    assert all(json.loads((runs / n / "done.json").read_text(encoding="utf-8"))["kind"] for n in names)
    assert (runs / "fsq-c10-B3-tfidf_lr" / "order_invariance.json").is_file()
    assert not (runs / "fsq-c10-B1-prior" / "order_invariance.json").exists()
    with (work / "results" / "runs.csv").open(encoding="utf-8", newline="") as fh:
        rows = {r["run_name"]: r for r in csv.DictReader(fh)}
    assert sorted(rows) == names and rows["fsq-c10-B1-prior"]["T"] == "1.0"
    assert rows["fsq-c10-B3-tfidf_lr"]["kind"] == "tfidf_lr" and rows["fsq-c10-B3-tfidf_lr"]["val_macro_f1"]
    assert M.main(["report", *common, "--out", "results/phase4_report.md"]) == 0
    report = json.loads((work / "results" / "phase4_report.json").read_text(encoding="utf-8"))
    assert report["phase4_exit_check"]["passed"] is True and report["phase4_exit_check"]["complete"] == 3
