"""Shared fixtures.

`en_tokenizer` / `tiny_ckpt_dir` need the real Laya tokenizer from the public Hub checkpoint
(pinned revision, ~4 MB, cached after the first run). Tests using them skip when laya/torch
are not installed or the Hub is unreachable.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# A 3-layer, 64-wide ModernBERT: same architecture family as ModernBERT-large, trains in seconds.
TINY_ENCODER = dict(hidden_size=64, intermediate_size=96, num_attention_heads=4, num_hidden_layers=3,
                    layer_types=["full_attention", "sliding_attention", "sliding_attention"])


@pytest.fixture(scope="session")
def cfg():
    from laya_poc.config import load_config
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def en_snapshot(cfg):
    pytest.importorskip("torch")
    pytest.importorskip("laya")
    from laya_poc import hub
    try:
        return hub.snapshot(hub.model_spec(cfg, "laya"), files=("tokenizer/*", "encoder/*", "rl_agent_config.json"))
    except Exception as exc:  # offline, rate-limited, ...
        pytest.skip(f"Hub unavailable: {exc}")


@pytest.fixture(scope="session")
def en_tokenizer(en_snapshot):
    from laya_poc import hub
    return hub.load_tokenizer_dir(en_snapshot)


def build_tiny_checkpoint(src: Path, dst: Path, seed: int = 0, init_scale: float = 0.05) -> Path:
    """Write a randomly initialised tiny Laya checkpoint directory that `laya.load` accepts.

    At the default init_scale the encoder attends almost uniformly, and ModernBERT has no absolute
    position embedding, so every option's [MASK] marker gets the same features: option logits are
    equal (std ~3e-5) and loss_ce is pinned at ln(K). Fine for plumbing tests; use a larger
    init_scale where a test needs the model to be able to learn.
    """
    import torch
    from laya.common import build_model
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "tokenizer", dst / "tokenizer", dirs_exist_ok=True)
    enc_cfg = json.loads((src / "encoder" / "config.json").read_text(encoding="utf-8"))
    enc_cfg.update(TINY_ENCODER)
    (dst / "encoder").mkdir(exist_ok=True)
    (dst / "encoder" / "config.json").write_text(json.dumps(enc_cfg), encoding="utf-8")
    rl_cfg = json.loads((src / "rl_agent_config.json").read_text(encoding="utf-8"))
    (dst / "rl_agent_config.json").write_text(json.dumps(rl_cfg, indent=2), encoding="utf-8")

    torch.manual_seed(seed)
    model = build_model(rl_cfg, encoder_dir=str(dst / "encoder"), pretrained=False)
    state = {}
    for k, v in model.state_dict().items():
        if not v.is_floating_point():
            state[k] = v
        elif k.endswith("norm.weight") or k.endswith("temperature"):
            state[k] = torch.ones_like(v)
        elif k.endswith("bias") or k.endswith("norm.bias"):
            state[k] = torch.zeros_like(v)
        else:
            state[k] = torch.randn_like(v) * init_scale
    save_file({k: v.half().contiguous() if v.is_floating_point() else v.contiguous() for k, v in state.items()},
              str(dst / "model.safetensors"))
    return dst


@pytest.fixture(scope="session")
def tiny_ckpt_dir(en_snapshot, tmp_path_factory):
    return build_tiny_checkpoint(en_snapshot, tmp_path_factory.mktemp("tiny_ckpt") / "ckpt")


@pytest.fixture()
def synthetic_data_dir(tmp_path):
    from synth import write_synthetic_data_dir
    return write_synthetic_data_dir(tmp_path / "data")
