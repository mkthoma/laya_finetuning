"""Phase 4 baselines through the matrix (spec P4 §1, §5), plan layer: phase4.arms expansion and run names,
selection, data inputs per kind, run-dir status and `matrix plan` with both phases (runs/p3 and runs/p4).
The run layer (dispatch, mocked subprocesses) is in test_matrix_p4_run.py."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from laya_poc import matrix as M
from laya_poc import matrix_baselines as B
from laya_poc import matrix_plan as P

from test_matrix_run import _data_dir, _rows

SPLITS = ("val", "test_id", "ood_country", "ood_script", "ood_brand", "stripped_test", "trap_candidates")
P4_NAMES = ["fsq-c10-B1-majority", "fsq-c10-B1-prior", "fsq-c7-B1-majority", "fsq-c7-B1-prior",
            "fsq-c10-B3-tfidf_lr", "fsq-c7-B3-tfidf_lr",
            "fsq-c10-B4-modernbert_base-s11", "fsq-c10-B4-modernbert_base-s22", "fsq-c10-B4-modernbert_base-s33",
            "fsq-c10-B4-mmbert_small-s11", "fsq-c10-B4-mmbert_small-s22", "fsq-c10-B4-mmbert_small-s33",
            "fsq-c10-B5-qwen3_4b"]


def _names(specs) -> list[str]:
    return [s.name for s in specs]


def _spec(cfg, name: str) -> P.RunSpec:
    return next(s for s in B.expand_all(cfg) if s.name == name)


# ---------------------------------------------------------------- expansion and names

def test_phase4_arms_expand_to_the_contract_run_names(cfg):
    specs = B.expand_phase4(cfg)
    assert _names(specs) == P4_NAMES
    assert [s.kind for s in specs[:6]] == ["majority", "prior", "majority", "prior", "tfidf_lr", "tfidf_lr"]
    assert {s.kind for s in specs[6:12]} == {"small_encoder"} and specs[12].kind == "llm"
    assert [s.optional for s in specs] == [False] * 12 + [True]
    assert all(s.phase == 4 for s in specs) and all(s.seed is None for s in specs[:6])
    b4 = specs[7]
    assert (b4.arm, b4.model, b4.scheme, b4.seed, b4.subset, b4.head_only) == ("B4", "modernbert_base", "c10", 22,
                                                                                None, False)


def test_expand_all_is_the_phase3_matrix_then_the_baselines(cfg):
    specs = B.expand_all(cfg)
    assert _names(specs[:14]) == _names(P.expand_specs(cfg)) and _names(specs[14:]) == P4_NAMES
    assert all(s.phase == 3 and s.kind == "laya" for s in specs[:14])


def test_no_phase4_section_means_no_baselines(cfg):
    c = {k: v for k, v in cfg.items() if k != "phase4"}
    assert B.expand_phase4(c) == () and _names(B.expand_all(c)) == _names(P.expand_specs(c))


@pytest.mark.parametrize("arm,msg", [
    ({"id": "B9", "kind": "svm", "scheme": "c10"}, "kind"),
    ({"id": "B1", "kind": "majority", "schemes": ["c99"]}, "scheme"),
    ({"id": "B1", "kind": "majority", "schemes": []}, "scheme"),
    ({"id": "B4", "kind": "small_encoder", "model": "bert_tiny", "scheme": "c10", "seeds": [1]}, "small_encoders"),
    ({"id": "B4", "kind": "small_encoder", "model": "mmbert_small", "scheme": "c10", "seeds": []}, "seeds"),
    ({"id": "B5", "kind": "llm", "model": "gpt", "scheme": "c10"}, "llms"),
    ({"kind": "tfidf_lr", "scheme": "c10"}, "id"),
])
def test_bad_phase4_arms_are_rejected_with_the_field_named(cfg, arm, msg):
    c = copy.deepcopy(cfg)
    c["phase4"]["arms"] = [arm]
    with pytest.raises(ValueError, match=msg):
        B.expand_phase4(c)


def test_duplicate_baseline_runs_are_rejected(cfg):
    c = copy.deepcopy(cfg)
    c["phase4"]["arms"] = [{"id": "B1", "kind": "majority", "schemes": ["c10"]},
                           {"id": "B1", "kind": "prior", "schemes": ["c10"]}]
    with pytest.raises(ValueError, match="duplicate.*fsq-c10-B1-prior"):
        B.expand_all(c)


def test_select_baselines_by_arm_arm_model_and_name(cfg):
    specs = B.expand_all(cfg)
    assert _names(P.select(specs, ["B1"])) == P4_NAMES[:4]
    assert _names(P.select(specs, ["b3"])) == P4_NAMES[4:6]
    assert _names(P.select(specs, ["B1-prior"])) == ["fsq-c10-B1-prior", "fsq-c7-B1-prior"]
    assert _names(P.select(specs, ["B4-mmbert_small"])) == P4_NAMES[9:12]
    assert _names(P.select(specs, ["fsq-c10-B5-qwen3_4b", "E4"])) == ["fsq-c10-E4-laya-s11-head",
                                                                     "fsq-c10-B5-qwen3_4b"]
    assert len(P.select(specs, ["all"])) == 27


# ---------------------------------------------------------------- data inputs and status

def test_baseline_data_dirs_follow_the_scheme_and_the_kind(cfg, tmp_path):
    b1c7 = P.data_paths(_spec(cfg, "fsq-c7-B1-prior"), tmp_path, SPLITS)
    assert (b1c7.train, b1c7.eval) == (tmp_path / "data_c7", tmp_path / "data_c7")
    assert dict(b1c7.extras) == {"trap_candidates": tmp_path / "data_eval" / "trap_candidates_c7.jsonl"}
    b5 = P.data_paths(_spec(cfg, "fsq-c10-B5-qwen3_4b"), tmp_path, SPLITS)
    assert b5.train is None and b5.eval == tmp_path / "data"
    assert P.train_inputs(_spec(cfg, "fsq-c10-B3-tfidf_lr"), 4) == ["train.jsonl"]
    b4 = _spec(cfg, "fsq-c10-B4-mmbert_small-s11")
    assert B.epochs(cfg, b4, P.TrainOverrides(), P.TrainOverrides()) == 3  # phase4.small_encoder_train.epochs
    assert B.epochs(cfg, b4, P.TrainOverrides(), P.TrainOverrides(epochs=1)) == 1
    assert P.train_inputs(b4, 3) == ["train_e0.jsonl", "train_e1.jsonl", "train_e2.jsonl", "val.jsonl"]


def test_check_data_names_the_missing_train_split_and_the_c7_variant(cfg, tmp_path):
    _data_dir(tmp_path / "data")
    _rows(tmp_path / "data_eval" / "trap_candidates.jsonl", 4, "trap_candidates")
    b3 = _spec(cfg, "fsq-c10-B3-tfidf_lr")
    with pytest.raises(FileNotFoundError, match="train.jsonl.*build_data"):
        P.check_data(b3, P.data_paths(b3, tmp_path, SPLITS), tmp_path, SPLITS, epochs=0)
    _rows(tmp_path / "data" / "train.jsonl", 8, "train")
    P.check_data(b3, P.data_paths(b3, tmp_path, SPLITS), tmp_path, SPLITS, epochs=0)
    b1 = _spec(cfg, "fsq-c7-B1-majority")
    with pytest.raises(FileNotFoundError, match="variants c7"):
        P.check_data(b1, P.data_paths(b1, tmp_path, SPLITS), tmp_path, SPLITS, epochs=0)


def _touch(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _evaluated(run: Path, splits) -> None:
    for s in splits:
        _touch(run / "eval" / f"{s}.json")
        _touch(run / "preds" / f"{s}.jsonl", "")


def test_stage_of_baseline_run_dirs(cfg, tmp_path):
    splits = ("val", "test_id")
    b1, b3, b4 = (_spec(cfg, n) for n in ("fsq-c10-B1-prior", "fsq-c10-B3-tfidf_lr", "fsq-c10-B4-mmbert_small-s22"))
    for spec in (b1, b3, b4):
        assert P.stage(tmp_path / spec.name, spec, splits) == "todo"
        _touch(tmp_path / spec.name / "calibration.json", json.dumps({"T": 1.0}))
        assert P.stage(tmp_path / spec.name, spec, splits) == "calibrated"
        _evaluated(tmp_path / spec.name, splits)
    assert P.stage(tmp_path / b1.name, b1, splits) == "evaluated"   # B1 has no order invariance
    assert P.stage(tmp_path / b3.name, b3, splits) == "calibrated"  # B3 still needs order_invariance.json
    _touch(tmp_path / b3.name / "order_invariance.json")
    assert P.stage(tmp_path / b3.name, b3, splits) == "evaluated"
    fresh = tmp_path / "b4" / b4.name
    _touch(fresh / "train" / "summary.json")
    assert (P.stage(fresh, b4, splits), P.status(fresh, b4, splits)) == ("trained", "partial")


def _write_cfg(cfg: dict, tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_plan_shows_the_baselines_in_their_own_table_under_runs_p4(cfg, tmp_path, capsys):
    work = tmp_path / "work"
    _touch(work / "runs" / "p4" / "fsq-c10-B1-prior" / "done.json", json.dumps({"run_name": "x"}))
    _touch(work / "runs" / "p4" / "fsq-c10-B4-mmbert_small-s11" / "train" / "log.txt", "")
    _touch(work / "runs" / "p3" / "fsq-c10-B1-majority" / "done.json")  # the wrong root: not this run's
    assert M.main(["plan", "--config", str(_write_cfg(cfg, tmp_path)), "--work", str(work)]) == 0
    out = capsys.readouterr().out
    lines = {line.split()[0]: line for line in out.splitlines() if line.startswith("fsq-")}
    assert " done" in lines["fsq-c10-B1-prior"] and "todo" in lines["fsq-c10-B1-majority"]
    assert "partial" in lines["fsq-c10-B4-mmbert_small-s11"] and "training" in lines["fsq-c10-B4-mmbert_small-s11"]
    assert "small_encoder" in lines["fsq-c10-B4-mmbert_small-s11"] and lines["fsq-c10-B5-qwen3_4b"].endswith(
        "(optional)")
    assert "Phase 4 baselines" in out and "13 baseline runs: 1 done, 1 partial, 11 todo" in out
    assert str(work / "runs" / "p4") in out and "14 runs: 0 done, 0 partial, 14 todo" in out


def test_plan_and_run_take_one_runs_root(cfg, tmp_path, capsys):
    cfg_path = str(_write_cfg(cfg, tmp_path))
    assert M.main(["plan", "--config", cfg_path, "--work", str(tmp_path), "--runs-root", "a", "--runs-root", "b"]) == 1
    assert "one runs root" in capsys.readouterr().err
    assert M.main(["plan", "--config", cfg_path, "--work", str(tmp_path), "--runs-root", "elsewhere"]) == 0
    assert f"under {tmp_path / 'elsewhere'}" in capsys.readouterr().out
