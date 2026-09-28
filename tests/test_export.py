import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("laya")
from laya_poc import export as X  # noqa: E402
from laya_poc import labels as L  # noqa: E402

pytestmark = pytest.mark.torch

STATES = [json.dumps(s, ensure_ascii=False, separators=(",", ":")) for s in (
    {"country": "GB", "locality": "Leeds", "name": "Rosa's Trattoria"},
    {"country": "JP", "name": "Blue Station 4", "tel": "0312 345678"},
    {"country": "DE", "name": "Kings Gym 9", "website": "example9.com"},
)]


@pytest.fixture(scope="module")
def source_agent(tiny_ckpt_dir):
    import laya
    return laya.load(str(tiny_ckpt_dir), device="cpu")


@pytest.fixture(scope="module")
def saved(source_agent, tmp_path_factory):
    out = tmp_path_factory.mktemp("export") / "final"
    path = X.save_laya_checkpoint(source_agent.model, source_agent.tok, source_agent.cfg, out,
                                  max_len=512, head_max_len=192, model_name="laya-poc-laya-s11")
    return path


def _probs(agent, states=STATES):
    out = agent.predict_batch(states, L.question("c10"), batch_size=4)
    return [out[i]["answers"][L.QUESTION_NAME]["probabilities"] for i in range(len(states))]


def test_saved_checkpoint_layout_and_config(saved, source_agent):
    assert (saved / "model.safetensors").exists()
    assert (saved / "encoder" / "config.json").exists()
    assert (saved / "tokenizer" / "tokenizer_config.json").exists()
    assert not saved.with_name(saved.name + ".partial").exists()
    cfg = json.loads((saved / "rl_agent_config.json").read_text(encoding="utf-8"))
    base = source_agent.cfg
    assert cfg["fine_tuned"] is True and cfg["model_name"] == "laya-poc-laya-s11"
    assert cfg["max_len"] == 512 and cfg["head_max_len"] == 192
    assert "temperature_by_options" not in cfg
    # T=None: choice T = 1.0 (raw logits). The inherited temperature[0] (1.637 on the English
    # root) was never applied to 10-option questions there (bucket choice:6-10 = 1.0000159) and is
    # stale for fine-tuned weights anyway; score/noul values are kept as the spec says.
    assert base["temperature"][0] != 1.0 and base.get("temperature_by_options")
    assert cfg["temperature"] == [1.0, base["temperature"][1], base["temperature"][2]]
    for key in ("encoder", "head_layers", "act_costs"):
        assert cfg[key] == base[key]


def test_checkpoint_carries_notice_licence_and_fsq_notice(source_agent, tmp_path):
    """Appendix D: keep Laya's LICENSE with a fine-tuned checkpoint, mark it modified, attribute the FSQ data."""
    from laya_poc.notice import fsq_notice
    out = X.save_laya_checkpoint(source_agent.model, source_agent.tok, source_agent.cfg, tmp_path / "o",
                                 max_len=512, head_max_len=192, model_name="laya-poc-laya-s11",
                                 base_repo="convaiinnovations/laya", base_revision="55cf4c4e",
                                 laya_commit="4066d5d5", fsq_release="2026-09-15")
    text = (out / "NOTICE.md").read_text(encoding="utf-8")
    assert "laya-poc-laya-s11" in text and "convaiinnovations/laya" in text and "55cf4c4e" in text
    assert "modified: fine-tuned" in text and "https://github.com/NandhaKishorM/laya/blob/4066d5d5/LICENSE" in text
    assert "see NOTICE_FSQ.txt" in text
    assert (out / "NOTICE_FSQ.txt").read_bytes() == fsq_notice("2026-09-15").encode("utf-8")
    assert "Apache License" in (out / "LICENSE").read_text(encoding="utf-8")  # copied from laya's dist-info


def test_notice_without_provenance_args_still_written(saved):
    text = (saved / "NOTICE.md").read_text(encoding="utf-8")
    assert "not recorded" in text and "modified: fine-tuned" in text
    assert (saved / "NOTICE_FSQ.txt").exists()


def test_base_config_is_not_mutated(source_agent, tmp_path):
    before = copy.deepcopy(source_agent.cfg)
    X.save_laya_checkpoint(source_agent.model, source_agent.tok, source_agent.cfg, tmp_path / "o",
                           max_len=512, head_max_len=192, model_name="m", temperature_choice=1.5)
    assert source_agent.cfg == before


def test_weights_are_fp16_without_wrapper_prefixes(saved):
    from safetensors import safe_open
    with safe_open(str(saved / "model.safetensors"), framework="pt") as fh:
        keys = list(fh.keys())
        dtypes = {fh.get_tensor(k).dtype for k in keys}
    assert dtypes == {torch.float16}
    assert all(k.split(".", 1)[0] in {"encoder", "head", "type_emb", "scorer", "act_head", "temperature"}
               for k in keys)


def test_saved_weights_equal_the_model_weights_in_fp16(saved, source_agent):
    # The tiny model predicts exactly uniform probabilities, so predictions cannot detect wrong weights.
    from safetensors.torch import load_file
    exported, source = load_file(str(saved / "model.safetensors")), source_agent.model.state_dict()
    assert set(exported) == set(source)
    for k, v in source.items():
        assert torch.equal(exported[k], v.half() if v.is_floating_point() else v), k


def test_reloaded_agent_carries_the_exported_weights(saved, source_agent):
    import laya
    reloaded, source = laya.load(str(saved), device="cpu").model.state_dict(), source_agent.model.state_dict()
    assert set(reloaded) == set(source)
    for k, v in source.items():                 # tiny ckpt weights are fp16-exact, so fp32 == fp32
        assert torch.equal(reloaded[k], v), k


def test_saved_checkpoint_reloads_and_predicts_identically(saved, source_agent):
    import laya
    agent = laya.load(str(saved), device="cpu")
    X.neutralise_temperatures(agent)
    X.neutralise_temperatures(source_agent)
    ours, theirs = _probs(agent), _probs(source_agent)
    for a, b in zip(ours, theirs):
        assert set(a) == set(L.option_keys("c10"))
        assert max(abs(a[k] - b[k]) for k in a) <= 1e-4   # tiny ckpt weights are already fp16


def test_temperature_choice_is_written_first(source_agent, tmp_path):
    out = X.save_laya_checkpoint(source_agent.model, source_agent.tok, {"encoder": "x", "head_layers": 2},
                                 tmp_path / "o", max_len=256, head_max_len=128, model_name="m",
                                 temperature_choice=2.5)
    cfg = json.loads((out / "rl_agent_config.json").read_text(encoding="utf-8"))
    assert cfg["temperature"] == [2.5, 1.0, 1.0]        # defaults [1, 1, 1] when missing
    assert (cfg["max_len"], cfg["head_max_len"]) == (256, 128)


def test_overwrites_an_existing_output_directory(source_agent, tmp_path):
    out = tmp_path / "final"
    out.mkdir()
    (out / "stale.txt").write_text("old", encoding="utf-8")
    X.save_laya_checkpoint(source_agent.model, source_agent.tok, source_agent.cfg, out,
                           max_len=512, head_max_len=192, model_name="m")
    assert not (out / "stale.txt").exists() and (out / "model.safetensors").exists()


def test_extra_json_lands_inside_the_checkpoint_and_laya_still_loads_it(source_agent, tmp_path):
    import laya
    marker = {"opt_step": 250, "final_eval": {"val_acc": 0.5, "n": 3}, "name": "Café"}
    out = X.save_laya_checkpoint(source_agent.model, source_agent.tok, source_agent.cfg, tmp_path / "best",
                                 max_len=512, head_max_len=192, model_name="m",
                                 extra_json={"train_eval.json": marker})
    assert json.loads((out / "train_eval.json").read_text(encoding="utf-8")) == marker
    assert not (tmp_path / "best.partial").exists()
    agent = laya.load(str(out), device="cpu")
    assert len(_probs(agent)) == len(STATES)


@pytest.mark.parametrize("name", ["model.safetensors", "rl_agent_config.json", "../x.json", "sub/x.json", "x.txt"])
def test_extra_json_cannot_clobber_checkpoint_files_or_escape(source_agent, tmp_path, name):
    with pytest.raises(ValueError, match="extra_json"):
        X.save_laya_checkpoint(source_agent.model, source_agent.tok, source_agent.cfg, tmp_path / "o",
                               max_len=512, head_max_len=192, model_name="m", extra_json={name: {}})
    assert not (tmp_path / "o").exists()


def test_rejects_wrapped_model_state_dict(source_agent, tmp_path):
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    with pytest.raises(ValueError, match="prefix"):
        X.save_laya_checkpoint(Wrapper(source_agent.model), source_agent.tok, source_agent.cfg, tmp_path / "o",
                               max_len=512, head_max_len=192, model_name="m")
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_temperature(source_agent, tmp_path, bad):
    with pytest.raises(ValueError, match="temperature"):
        X.save_laya_checkpoint(source_agent.model, source_agent.tok, source_agent.cfg, tmp_path / "o",
                               max_len=512, head_max_len=192, model_name="m", temperature_choice=bad)


def test_write_choice_temperature_round_trips_through_laya_load(saved, tmp_path):
    import shutil

    import laya
    ckpt = tmp_path / "calibrated"
    shutil.copytree(saved, ckpt)
    cfg_path = ckpt / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["temperature_by_options"] = {"choice:6-10": 1.0}        # an inherited bucket must be removed
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    new = X.write_choice_temperature(ckpt, 1.8)
    assert new["temperature"][0] == 1.8 and "temperature_by_options" not in new
    assert json.loads(cfg_path.read_text(encoding="utf-8")) == new
    assert not list(ckpt.glob("*.tmp"))
    agent = laya.load(str(ckpt), device="cpu")
    assert agent.temperature_raw[0] == 1.8
    assert agent.temperature_by_options_raw == {}


def test_write_choice_temperature_validates(tmp_path):
    with pytest.raises(FileNotFoundError):
        X.write_choice_temperature(tmp_path, 1.2)
    (tmp_path / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        X.write_choice_temperature(tmp_path, 0.0)
    assert X.write_choice_temperature(tmp_path, 1.2)["temperature"] == [1.2, 1.0, 1.0]


def test_neutralise_temperatures(source_agent):
    X.neutralise_temperatures(source_agent)
    assert source_agent.temperature == [1.0, 1.0, 1.0]
    assert source_agent.temperature_by_options == {} and source_agent.lang_temperatures == {}


@pytest.mark.parametrize("init, model, repo, revision", [
    ("hub", "laya", "convaiinnovations/laya", True),
    ("hub", "laya_ml", "convaiinnovations/laya/multilingual", True),
    ("/abs/ckpt", "laya", "/abs/ckpt", False),  # a local --init: its own path, no Hub revision
])
def test_checkpoint_provenance_names_base_revision_commit_and_release(init, model, repo, revision):
    from laya_poc.config import load_config
    cfg = load_config()
    prov = X.checkpoint_provenance(init, cfg, model)
    assert prov == {"base_repo": repo, "base_revision": cfg["laya"]["hub_revision"] if revision else None,
                    "laya_commit": cfg["laya"]["commit"], "fsq_release": cfg["data"]["fsq_release"]}
