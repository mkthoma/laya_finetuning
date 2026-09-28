"""One tiny real Phase 3 matrix on CPU: train_single -> export_check -> evaluate (multi-split + order invariance)
-> cleanup -> done.json -> runs.csv -> report, through real subprocesses, with the tiny checkpoint as --init.
Plumbing only; the numbers mean nothing. B2 (evaluate only) is covered by the mocked tests and the notebook's
local dry run (tools/dry_run_p3_local.py); leaving it out keeps this to three subprocesses."""
from __future__ import annotations

import copy
import csv
import json

import pytest
import yaml

pytestmark = pytest.mark.torch
pytest.importorskip("torch")
pytest.importorskip("laya")

from laya_poc import matrix as M  # noqa: E402

from synth import write_synthetic_data_dir  # noqa: E402


def test_tiny_matrix_end_to_end_on_cpu(cfg, tiny_ckpt_dir, tmp_path):
    work = tmp_path / "work"
    write_synthetic_data_dir(work / "data", n_train=32, n_val=16)
    c = {k: v for k, v in copy.deepcopy(cfg).items() if k != "phase4"}  # Phase 3 only: `all` = the one Laya run
    c["train"]["epochs"] = 1
    c["phase3"] = {**c["phase3"], "arms": [{"id": "E2", "model": "laya", "scheme": "c10", "seeds": [11]}],
                   "zero_shot": [], "eval_splits": ["val", "test_id"],
                   "order_invariance": {"split": "test_id", "n": 8, "perms": 2, "seed": 1}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    common = ["--config", str(cfg_path), "--work", str(work)]
    rc = M.main(["run", "--only", "all", *common, "--init", str(tiny_ckpt_dir), "--device", "cpu", "--card", "CPU",
                 "--train-extra=--max-micro-steps 6 --eval-every-opt-steps 2"])
    runs = work / "runs" / "p3"
    logs = sorted(p.name for p in runs.glob("*/logs/*.log"))
    assert rc == 0, logs
    rd = runs / "fsq-c10-E2-laya-s11"
    assert (rd / "train" / "best" / "model.safetensors").is_file() and not (rd / "train" / "ckpt").exists()
    assert json.loads((rd / "export_check.json").read_text(encoding="utf-8"))["T"] > 0
    oi = json.loads((rd / "order_invariance.json").read_text(encoding="utf-8"))
    assert oi["perms"] == 2 and 0.0 <= oi["mean_agreement"] <= 1.0
    rows = list(csv.DictReader((work / "results" / "runs.csv").open(encoding="utf-8")))
    assert [r["run_name"] for r in rows] == ["fsq-c10-E2-laya-s11"]
    assert all(r["T"] and r["val_macro_f1"] and r["order_invariance"] for r in rows)
    assert M.main(["report", *common]) == 0
    report = json.loads((work / "results" / "phase3_report.json").read_text(encoding="utf-8"))
    assert report["exit_check"]["passed"] is True
