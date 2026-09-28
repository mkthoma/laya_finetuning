"""Phase 3 seed matrix, plan layer (laya_poc.matrix_plan + `matrix plan`): run specs and names (design doc
§7.3, spec §1), selection, data directories per scheme/subset, the eval cadence of small subsets and the
done / partial / todo status of run directories. No model, no subprocess."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from laya_poc import matrix as M
from laya_poc import matrix_plan as P

from synth import rows_from_records, synthetic_records
from laya_poc.io_utils import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("val", "test_id", "ood_country", "ood_script", "ood_brand", "stripped_test", "trap_candidates")


def _names(specs) -> list[str]:
    return [s.name for s in specs]


def _write_rows(path: Path, n: int, split: str = "train") -> None:
    write_jsonl(path, rows_from_records(synthetic_records(n), split))


# ---------------------------------------------------------------- expansion and names

def test_config_matrix_expands_to_12_trained_runs_and_2_zero_shot(cfg):
    specs = P.expand_specs(cfg)
    trained = [s for s in specs if not s.zero_shot]
    zs = [s for s in specs if s.zero_shot]
    assert len(trained) == 3 + 3 + 1 + 1 + 1 + 3 and len(zs) == 2
    assert len(set(_names(specs))) == len(specs)
    by_arm = {arm: [s for s in trained if s.arm == arm] for arm in ("E2", "E3", "E4", "E5", "E6")}
    assert [len(v) for v in by_arm.values()] == [3, 3, 1, 2, 3]


def test_run_names_follow_the_contract(cfg):
    names = set(_names(P.expand_specs(cfg)))
    for expected in ("fsq-c10-E2-laya-s11", "fsq-c10-E2-laya-s22", "fsq-c10-E2-laya-s33", "fsq-c10-E3-laya_ml-s33",
                     "fsq-c10-E4-laya-s11-head", "fsq-c7-E5-laya-s11", "fsq-c7-E5-laya_ml-s11",
                     "fsq-c10-E6-laya-s11-n1000", "fsq-c10-E6-laya-s11-n3000", "fsq-c10-E6-laya-s11-n10000",
                     "fsq-c10-B2-laya-zs", "fsq-c10-B2-laya_ml-zs"):
        assert expected in names


def test_zero_shot_runs_come_first_then_arms_in_config_order(cfg):
    names = _names(P.expand_specs(cfg))
    assert names[:2] == ["fsq-c10-B2-laya-zs", "fsq-c10-B2-laya_ml-zs"]
    assert names[2:5] == ["fsq-c10-E2-laya-s11", "fsq-c10-E2-laya-s22", "fsq-c10-E2-laya-s33"]


def test_spec_fields(cfg):
    by_name = {s.name: s for s in P.expand_specs(cfg)}
    head = by_name["fsq-c10-E4-laya-s11-head"]
    assert (head.arm, head.model, head.scheme, head.seed, head.subset, head.head_only) == ("E4", "laya", "c10", 11,
                                                                                            None, True)
    sub = by_name["fsq-c10-E6-laya-s11-n3000"]
    assert (sub.subset, sub.head_only, sub.zero_shot) == (3000, False, False)
    zs = by_name["fsq-c10-B2-laya_ml-zs"]
    assert (zs.arm, zs.model, zs.scheme, zs.seed, zs.zero_shot) == ("B2", "laya_ml", "c10", None, True)


def test_zero_shot_scheme_can_be_configured(cfg):
    c = copy.deepcopy(cfg)
    c["phase3"]["zero_shot"] = ["laya", {"model": "laya", "scheme": "c7"}]
    assert _names(P.expand_specs(c))[:2] == ["fsq-c10-B2-laya-zs", "fsq-c7-B2-laya-zs"]


@pytest.mark.parametrize("arm,msg", [
    ({"id": "E9", "model": "nope", "scheme": "c10", "seeds": [1]}, "model"),
    ({"id": "E9", "model": "laya", "scheme": "c99", "seeds": [1]}, "scheme"),
    ({"id": "E9", "model": "laya", "scheme": "c10", "seeds": []}, "seeds"),
    ({"id": "E9", "model": "laya", "scheme": "c10", "seeds": [1], "train_subset": [0]}, "train_subset"),
    ({"model": "laya", "scheme": "c10", "seeds": [1]}, "id"),
    ({"id": "B2", "model": "laya", "scheme": "c10", "seeds": [1]}, "B2"),
])
def test_bad_arms_are_rejected_with_the_field_named(cfg, arm, msg):
    c = copy.deepcopy(cfg)
    c["phase3"]["arms"] = [arm]
    with pytest.raises(ValueError, match=msg):
        P.expand_specs(c)


def test_duplicate_runs_are_rejected(cfg):
    c = copy.deepcopy(cfg)
    c["phase3"]["arms"] = [{"id": "E2", "model": "laya", "scheme": "c10", "seeds": [11, 11]}]
    with pytest.raises(ValueError, match="duplicate"):
        P.expand_specs(c)


def test_missing_phase3_section_is_a_clear_error(cfg):
    c = {k: v for k, v in cfg.items() if k != "phase3"}
    with pytest.raises(ValueError, match="phase3"):
        P.expand_specs(c)


# ---------------------------------------------------------------- selection

def test_select_by_arm_run_name_and_all(cfg):
    specs = P.expand_specs(cfg)
    assert _names(P.select(specs, ["E2"])) == ["fsq-c10-E2-laya-s11", "fsq-c10-E2-laya-s22", "fsq-c10-E2-laya-s33"]
    assert _names(P.select(specs, ["B2"])) == ["fsq-c10-B2-laya-zs", "fsq-c10-B2-laya_ml-zs"]
    assert _names(P.select(specs, ["fsq-c7-E5-laya-s11"])) == ["fsq-c7-E5-laya-s11"]
    assert len(P.select(specs, ["all"])) == len(specs)
    assert _names(P.select(specs, ["e4"])) == ["fsq-c10-E4-laya-s11-head"]  # arm ids are case-insensitive


def test_select_keeps_plan_order_and_dedupes(cfg):
    specs = P.expand_specs(cfg)
    got = _names(P.select(specs, ["E4", "B2-laya", "fsq-c10-E4-laya-s11-head", "B2"]))
    assert got == ["fsq-c10-B2-laya-zs", "fsq-c10-B2-laya_ml-zs", "fsq-c10-E4-laya-s11-head"]


def test_select_b2_model_shorthand(cfg):
    assert _names(P.select(P.expand_specs(cfg), ["B2-laya_ml"])) == ["fsq-c10-B2-laya_ml-zs"]


def test_select_unknown_names_the_choices(cfg):
    with pytest.raises(ValueError, match="E2.*E6"):
        P.select(P.expand_specs(cfg), ["E7"])


# ---------------------------------------------------------------- data directories

def test_data_paths_per_scheme_and_subset(cfg, tmp_path):
    by_name = {s.name: s for s in P.expand_specs(cfg)}
    c10 = P.data_paths(by_name["fsq-c10-E2-laya-s22"], tmp_path, SPLITS)
    assert (c10.train, c10.eval) == (tmp_path / "data", tmp_path / "data")
    assert dict(c10.extras) == {"trap_candidates": tmp_path / "data_eval" / "trap_candidates.jsonl"}
    c7 = P.data_paths(by_name["fsq-c7-E5-laya_ml-s11"], tmp_path, SPLITS)
    assert (c7.train, c7.eval) == (tmp_path / "data_c7", tmp_path / "data_c7")
    assert dict(c7.extras) == {"trap_candidates": tmp_path / "data_eval" / "trap_candidates_c7.jsonl"}
    lc = P.data_paths(by_name["fsq-c10-E6-laya-s11-n1000"], tmp_path, SPLITS)
    assert (lc.train, lc.eval) == (tmp_path / "data_lc1000", tmp_path / "data")  # evaluated on the full splits
    zs = P.data_paths(by_name["fsq-c10-B2-laya-zs"], tmp_path, SPLITS)
    assert zs.train is None and zs.eval == tmp_path / "data"
    assert P.data_paths(by_name["fsq-c10-E2-laya-s22"], tmp_path, ("val", "test_id")).extras == ()


def _full_data(root: Path, name: str = "data", epochs: int = 4, n: int = 8) -> Path:
    d = root / name
    for e in range(epochs):
        _write_rows(d / f"train_e{e}.jsonl", n)
    for split in SPLITS[:-1]:
        _write_rows(d / f"{split}.jsonl", n, split)
    return d


def _traps(root: Path, *suffixes: str) -> None:
    for s in suffixes or ("",):
        _write_rows(root / "data_eval" / f"trap_candidates{s}.jsonl", 4, "trap_candidates")


def test_check_data_passes_on_a_complete_layout(cfg, tmp_path):
    _full_data(tmp_path)
    _traps(tmp_path)
    spec = P.select(P.expand_specs(cfg), ["fsq-c10-E2-laya-s11"])[0]
    P.check_data(spec, P.data_paths(spec, tmp_path, SPLITS), tmp_path, SPLITS, epochs=4)


@pytest.mark.parametrize("run,missing,hint", [
    ("fsq-c7-E5-laya-s11", "data_c7", "python -m laya_poc.variants c7 --data-dir {root}/data --out {root}/data_c7"),
    ("fsq-c10-E6-laya-s11-n3000", "data_lc3000",
     "python -m laya_poc.variants subset --data-dir {root}/data --n 3000 --out {root}/data_lc3000"),
    ("fsq-c10-E2-laya-s11", "data_eval", "python -m laya_poc.variants traps --data-dir {root}/data --out "
                                         "{root}/data_eval"),
])
def test_a_missing_variant_names_the_command_to_run(cfg, tmp_path, run, missing, hint):
    _full_data(tmp_path)
    if missing != "data_eval":
        _traps(tmp_path, "", "_c7")
    spec = P.select(P.expand_specs(cfg), [run])[0]
    with pytest.raises(FileNotFoundError) as info:
        P.check_data(spec, P.data_paths(spec, tmp_path, SPLITS), tmp_path, SPLITS, epochs=4)
    msg = str(info.value)
    assert "\n" not in msg and missing in msg
    assert hint.format(root=tmp_path.as_posix()) in msg.replace("\\", "/")


def test_check_data_names_a_missing_split_file_and_missing_epochs(cfg, tmp_path):
    d = _full_data(tmp_path)
    _traps(tmp_path)
    spec = P.select(P.expand_specs(cfg), ["fsq-c10-E2-laya-s11"])[0]
    paths = P.data_paths(spec, tmp_path, SPLITS)
    (d / "ood_brand.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="ood_brand.jsonl"):
        P.check_data(spec, paths, tmp_path, SPLITS, epochs=4)
    _write_rows(d / "ood_brand.jsonl", 4, "ood_brand")
    with pytest.raises(FileNotFoundError, match="train_e4.jsonl"):
        P.check_data(spec, paths, tmp_path, SPLITS, epochs=5)


def test_zero_shot_needs_no_train_files(cfg, tmp_path):
    d = _full_data(tmp_path, epochs=0)
    _traps(tmp_path)
    spec = P.select(P.expand_specs(cfg), ["B2-laya"])[0]
    P.check_data(spec, P.data_paths(spec, tmp_path, SPLITS), tmp_path, SPLITS, epochs=4)
    assert not (d / "train_e0.jsonl").exists()


# ---------------------------------------------------------------- eval cadence of subsets

@pytest.mark.parametrize("total,expected", [(128, 10), (376, 31), (1252, 104), (5000, 250), (16, 10), (0, 10)])
def test_eval_every_formula(total, expected):
    assert P.eval_every_from_total(total, evals_per_run=12, cap=250) == expected


def test_eval_every_counts_the_subset_epoch_files_with_the_card_batching(cfg, tmp_path):
    by_name = {s.name: s for s in P.expand_specs(cfg)}
    d = tmp_path / "data_lc1000"
    for e in range(4):
        _write_rows(d / f"train_e{e}.jsonl", 1000)
    spec = by_name["fsq-c10-E6-laya-s11-n1000"]
    # G4: micro-batch 32 x accumulation 1 -> 4 x ceil(1000 / 32) = 128 opt steps -> 128 // 12 = 10
    assert P.eval_every(cfg, spec, d, "G4", P.TrainOverrides()) == 10
    # T4: 8 x 4 -> 4 x ceil(125 / 4) = 128 opt steps as well
    assert P.eval_every(cfg, spec, d, "T4", P.TrainOverrides()) == 10
    # --train-extra micro/effective batch and epochs change the plan: 2 epochs x ceil(1000 / 4) = 500 -> 41
    over = P.parse_train_extra(("--epochs", "2", "--micro-batch", "2", "--effective-batch", "4"))
    assert P.eval_every(cfg, spec, d, "CPU", over) == 41
    assert P.eval_every(cfg, by_name["fsq-c10-E2-laya-s11"], tmp_path / "data", "G4", P.TrainOverrides()) is None


def test_parse_train_extra_ignores_unrelated_flags():
    over = P.parse_train_extra(("--max-micro-steps", "20", "--print-every", "5", "--initial-eval"))
    assert over == P.TrainOverrides(max_micro_steps=20)


# ---------------------------------------------------------------- status

def _touch(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_stage_and_status_of_run_dirs(cfg, tmp_path):
    spec = P.select(P.expand_specs(cfg), ["fsq-c10-E2-laya-s11"])[0]
    run = tmp_path / spec.name
    splits = ("val", "test_id")
    assert (P.stage(run, spec, splits), P.status(run, spec, splits)) == ("todo", "todo")
    _touch(run / "train" / "ckpt" / "step0000010.pt")
    assert (P.stage(run, spec, splits), P.status(run, spec, splits)) == ("training", "partial")
    _touch(run / "train" / "summary.json")
    _touch(run / "train" / "best" / "model.safetensors", "")
    assert P.stage(run, spec, splits) == "trained"
    _touch(run / "export_check.json")
    assert P.stage(run, spec, splits) == "calibrated"
    for s in splits:
        _touch(run / "eval" / f"{s}.json")
        _touch(run / "preds" / f"{s}.jsonl", "")
    assert P.stage(run, spec, splits) == "calibrated"  # order invariance still missing
    _touch(run / "order_invariance.json")
    assert P.stage(run, spec, splits) == "evaluated"
    _touch(run / "done.json")
    assert (P.stage(run, spec, splits), P.status(run, spec, splits)) == ("done", "done")


def test_zero_shot_stage_skips_training(cfg, tmp_path):
    spec = P.select(P.expand_specs(cfg), ["B2-laya"])[0]
    run = tmp_path / spec.name
    run.mkdir()
    assert P.stage(run, spec, ("val",)) == "started"
    _touch(run / "eval" / "val.json")
    _touch(run / "preds" / "val.jsonl", "")
    _touch(run / "order_invariance.json")
    assert P.stage(run, spec, ("val",)) == "evaluated"


# ---------------------------------------------------------------- `matrix plan`

def _write_cfg(cfg: dict, tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_plan_prints_every_run_with_its_status(cfg, tmp_path, capsys):
    work = tmp_path / "work"
    _touch(work / "runs" / "p3" / "fsq-c10-E2-laya-s11" / "done.json", json.dumps({"run_name": "x"}))
    _touch(work / "runs" / "p3" / "fsq-c10-E2-laya-s22" / "train" / "ckpt" / "step0000001.pt", "")
    assert M.main(["plan", "--config", str(_write_cfg(cfg, tmp_path)), "--work", str(work)]) == 0
    out = capsys.readouterr().out
    lines = {line.split()[0]: line for line in out.splitlines() if line.startswith("fsq-")}
    phase3 = [n for n in lines if not n.startswith(("fsq-c10-B1", "fsq-c7-B1", "fsq-c10-B3", "fsq-c7-B3",
                                                     "fsq-c10-B4", "fsq-c10-B5"))]
    assert len(phase3) == 14 and len(lines) == 14 + 13  # the Phase 4 baselines follow in their own table
    assert " done" in lines["fsq-c10-E2-laya-s11"]
    assert "partial" in lines["fsq-c10-E2-laya-s22"] and "training" in lines["fsq-c10-E2-laya-s22"]
    assert "todo" in lines["fsq-c10-E3-laya_ml-s11"]
    assert "14 runs: 1 done, 1 partial, 12 todo" in out


def test_plan_rejects_a_relative_work_dir(cfg, tmp_path, capsys):
    assert M.main(["plan", "--config", str(_write_cfg(cfg, tmp_path)), "--work", "relative/work"]) == 1
    assert "absolute" in capsys.readouterr().err
