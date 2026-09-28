from pathlib import Path

import pytest
import yaml

from laya_poc import config as C

ROOT = Path(__file__).resolve().parents[1]


def test_repo_config_loads_and_validates():
    cfg = C.load_config(ROOT / "config.yaml")
    assert cfg["laya"]["commit"] and len(cfg["laya"]["commit"]) == 40
    assert cfg["model"]["laya"]["head_max_len"] == 192


def test_project_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv(C.PROJECT_ROOT_ENV, str(tmp_path))
    assert C.project_root() == tmp_path


@pytest.mark.parametrize("gpu,card", [("Tesla T4", "T4"), ("NVIDIA L4", "L4"), ("NVIDIA A10G", "A10"),
                                      ("NVIDIA A10", "A10"),
                                      ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "G4")])
def test_card_from_gpu_name(gpu, card):
    assert C.card_from_gpu_name(gpu) == card


def test_unknown_gpu_raises():
    with pytest.raises(ValueError):
        C.card_from_gpu_name("NVIDIA A100-SXM4-40GB")


def test_accumulation_for_t4_is_8_by_4():
    cfg = C.load_config(ROOT / "config.yaml")
    assert C.accumulation(cfg, "T4") == (8, 4)
    assert C.accumulation(cfg, "L4") == (32, 1)


def test_validate_rejects_indivisible_batch(tmp_path):
    cfg = C.load_config(ROOT / "config.yaml")
    bad = C.with_overrides(cfg, {"train.effective_batch": 30})
    with pytest.raises(ValueError, match="not divisible"):
        C.validate_config(bad)


def test_validate_rejects_missing_section():
    with pytest.raises(ValueError, match="missing"):
        C.validate_config({"laya": {}})


def test_validate_rejects_kill_after_end():
    cfg = C.load_config(ROOT / "config.yaml")
    with pytest.raises(ValueError, match="kill_at_micro_step"):
        C.validate_config(C.with_overrides(cfg, {"smoke.kill_at_micro_step": 999}))


def test_with_overrides_does_not_mutate_original():
    cfg = C.load_config(ROOT / "config.yaml")
    new = C.with_overrides(cfg, {"train.epochs": 1})
    assert new["train"]["epochs"] == 1 and cfg["train"]["epochs"] == 4


def test_split_spec_smoke_vs_full():
    cfg = C.load_config(ROOT / "config.yaml")
    full = C.split_spec(cfg)
    smoke = C.split_spec(cfg, smoke=True)
    assert full.strict and not smoke.strict
    assert full.train_size == 25000 and smoke.train_size == cfg["smoke"]["train_questions"]
    assert smoke.ood_country == () and full.ood_country == ("NL", "PL", "MX")


def test_load_config_from_explicit_yaml(tmp_path):
    cfg = C.load_config(ROOT / "config.yaml")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert C.load_config(p) == cfg


def test_g4_profile_is_the_l4_recipe_without_grad_ckpt():
    cfg = C.load_config(ROOT / "config.yaml")
    assert C.accumulation(cfg, "G4") == (32, 1)
    assert cfg["train"]["grad_ckpt"]["G4"] is False
