"""Laya checkpoint directories: load one for training, write and patch them (upstream save,
cell08.py:251-263, plus fixes).

Layout that `laya.load` accepts: model.safetensors (fp16 state dict of the *unwrapped* model),
encoder/ (config only), tokenizer/, rl_agent_config.json; plus NOTICE.md, NOTICE_FSQ.txt and Laya's
LICENSE, which the loader ignores (design doc Appendix D). The saved config carries the
max_len/head_max_len the training items were built with (upstream saved 1024/256 after building
items at 512/192), and drops `temperature_by_options`: inherited bucket values take precedence
over `temperature` at inference and would silently mask a new fit (agent.py:768).
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from . import notice as N

CONFIG_NAME = "rl_agent_config.json"
DEFAULT_TEMPERATURE = (1.0, 1.0, 1.0)  # [choice, score, noul]
_BAD_PREFIXES = ("module.", "_orig_mod.")


def _check_temperature(t: float) -> float:
    t = float(t)
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f"temperature must be a finite number > 0, got {t}")
    return t


def _temperatures(cfg: dict) -> list[float]:
    """[choice, score, noul], padded with 1.0 (Laya indexes all three; a short list raises there)."""
    return (list(cfg.get("temperature") or DEFAULT_TEMPERATURE) + list(DEFAULT_TEMPERATURE))[:3]


def load_for_training(init: str, cfg: dict, s: Any) -> tuple[Any, Any, dict]:
    """Model, tokenizer and base config through the library load path (critique §A 7.6.3-1), on the
    training device in train() mode; `s` is schedule.Settings. Refuses non-fp32 weights (GradScaler)."""
    from . import hub

    source = hub.model_spec(cfg, s.model) if init == "hub" else Path(init)
    agent = hub.load_agent(source, device="cpu")
    model, tok, base_cfg = agent.model, agent.tok, copy.deepcopy(agent.cfg)
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    if dtypes != ["torch.float32"]:
        raise RuntimeError(f"GradScaler training needs fp32 parameters, found {dtypes}")
    model.to(s.device).train()  # the Agent leaves it in eval()
    if s.grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    return model, tok, base_cfg


def fp16_state_dict(model: Any) -> dict:
    """Upstream's fp16 cast (floating tensors only), contiguous, on CPU; refuses wrapped models."""
    sd = model.state_dict()
    bad = [k for k in sd if k.startswith(_BAD_PREFIXES) or "._orig_mod." in k]
    if bad:
        raise ValueError(f"state_dict keys carry a DDP/compile wrapper prefix (e.g. {bad[0]!r}); "
                         "save the unwrapped model, Laya's loader rejects these keys")
    return {k: (v.half() if v.is_floating_point() else v).contiguous().cpu() for k, v in sd.items()}


def export_config(base_cfg: dict, *, max_len: int, head_max_len: int, model_name: str,
                  temperature_choice: float | None = None) -> dict:
    """New rl_agent_config.json content; `base_cfg` is not modified.

    Without a fitted T the choice temperature is 1.0 (raw logits), not the inherited temperature[0]:
    that value is stale for fine-tuned weights, and on the English root (1.637) it was never even
    applied to our 10-option question, which used the bucket choice:6-10 = 1.0000159 dropped here.
    """
    cfg = copy.deepcopy(base_cfg)
    old = _temperatures(cfg)
    t_choice = 1.0 if temperature_choice is None else _check_temperature(temperature_choice)
    cfg.update(fine_tuned=True, model_name=model_name, max_len=int(max_len), head_max_len=int(head_max_len),
               temperature=[t_choice, old[1], old[2]])
    cfg.pop("temperature_by_options", None)
    return cfg


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


_RESERVED = {CONFIG_NAME, "model.safetensors", "LICENSE", N.CHECKPOINT_NOTICE_NAME, N.FSQ_NOTICE_NAME}


def _check_extra(extra: Mapping[str, dict]) -> dict[str, dict]:
    """Sidecar names: plain *.json file names that do not replace a file the loader or licence needs."""
    bad = [n for n in extra if n in _RESERVED or not n.endswith(".json") or Path(n).name != n or n.startswith(".")]
    if bad:
        raise ValueError(f"extra_json names must be plain new *.json file names, got {bad}")
    return dict(extra)


def checkpoint_provenance(init: str, cfg: dict, model: str) -> dict[str, str | None]:
    """save_laya_checkpoint's provenance kwargs: the Hub repo[/subfolder] at the pinned revision
    (init 'hub') or the local --init path (no revision), the pinned Laya commit and the FSQ release."""
    if init == "hub":
        m = cfg["model"][model]
        base = f"{m['id']}/{m['subfolder']}" if m.get("subfolder") else m["id"]
        revision = cfg["laya"].get("hub_revision")
    else:
        base, revision = str(init), None
    return {"base_repo": base, "base_revision": revision, "laya_commit": cfg["laya"].get("commit"),
            "fsq_release": cfg["data"].get("fsq_release")}


def write_notices(ckpt_dir: Path, *, model_name: str, base_repo: str | None = None, base_revision: str | None = None,
                  laya_commit: str | None = None, fsq_release: str | None = None) -> None:
    """NOTICE.md (base, 'modified: fine-tuned', licence, data), NOTICE_FSQ.txt and Laya's LICENSE if shipped."""
    text = N.checkpoint_notice(model_name=model_name, base_repo=base_repo, base_revision=base_revision,
                               laya_commit=laya_commit or N.installed_laya_commit())
    (ckpt_dir / N.CHECKPOINT_NOTICE_NAME).write_bytes(text.encode("utf-8"))
    (ckpt_dir / N.FSQ_NOTICE_NAME).write_bytes(N.fsq_notice(fsq_release).encode("utf-8"))
    licence = N.laya_license_text()
    if licence:
        (ckpt_dir / "LICENSE").write_bytes(licence.encode("utf-8"))


def save_laya_checkpoint(model: Any, tok: Any, base_cfg: dict, out_dir: str | Path, *, max_len: int,
                         head_max_len: int, model_name: str, temperature_choice: float | None = None,
                         base_repo: str | None = None, base_revision: str | None = None,
                         laya_commit: str | None = None, fsq_release: str | None = None,
                         extra_json: Mapping[str, dict] | None = None) -> Path:
    """Write a complete Laya checkpoint directory; replaces `out_dir` only once fully written.

    The optional provenance (base checkpoint repo and Hub revision, pinned Laya commit, FSQ release)
    goes into NOTICE.md / NOTICE_FSQ.txt; without it they say "not recorded" (the Laya commit falls
    back to the installed package's PEP 610 record). `extra_json` ({file name: object}) adds sidecar
    files that must appear together with the weights (e.g. best/'s train_eval.json marker).
    """
    from safetensors.torch import save_file

    extra = _check_extra(extra_json or {})
    cfg = export_config(base_cfg, max_len=max_len, head_max_len=head_max_len, model_name=model_name,
                        temperature_choice=temperature_choice)
    sd = fp16_state_dict(model)
    out = Path(out_dir)
    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    save_file(sd, str(partial / "model.safetensors"))
    model.encoder.config.save_pretrained(str(partial / "encoder"))
    tok.save_pretrained(str(partial / "tokenizer"))
    _write_json_atomic(partial / CONFIG_NAME, cfg)
    write_notices(partial, model_name=model_name, base_repo=base_repo, base_revision=base_revision,
                  laya_commit=laya_commit, fsq_release=fsq_release)
    for name, obj in extra.items():
        _write_json_atomic(partial / name, obj)
    if out.exists():
        shutil.rmtree(out)
    os.replace(partial, out)
    return out


def write_choice_temperature(ckpt_dir: str | Path, T: float) -> dict:
    """Set temperature[0] (choice) = T and drop temperature_by_options, atomically. Returns the config."""
    path = Path(ckpt_dir) / CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(f"no {CONFIG_NAME} in {ckpt_dir}")
    T = _check_temperature(T)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    new = {**cfg, "temperature": [T, *_temperatures(cfg)[1:]]}
    new.pop("temperature_by_options", None)
    _write_json_atomic(path, new)
    return new


def neutralise_temperatures(agent: Any) -> Any:
    """Make a loaded Agent output softmax(raw logits). Mutates the agent on purpose: these are
    plain attributes read on every decode, and assigning them bypasses the load-time clamp."""
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.lang_temperatures = {}
    return agent
