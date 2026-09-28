"""Which models the CPU benchmark knows, which backends each supports, and where their weights live.

Two kinds of model:
- "laya": a config `model:` key (laya, laya_ml), benchmarked with backends `torch` (the PyTorch Agent, fp32) and `onnx`
  (the exported graph on ONNX Runtime);
- "hf": a config `phase4.small_encoders` key (modernbert_base, mmbert_small; the B4 baselines), benchmarked with
  backend `hf` (AutoModelForSequenceClassification on PyTorch, fp32). `torch` is accepted as an alias of `hf` for
  them (config phase5.bench.backends lists them as `[torch]`: they are PyTorch models).

Latency does not depend on fine-tuned weights, so `hub` (the pinned base checkpoints: laya.hub_revision and
phase4.small_encoders.*.revision) is the default and a valid source. Every source is resolved to an ABSOLUTE LOCAL
directory in the parent process (downloading first), so no timed worker ever touches the network.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

LAYA, HF = "laya", "hf"
BACKENDS = {LAYA: ("torch", "onnx"), HF: ("hf",)}
HF_ALIASES = {"torch": "hf", "hf": "hf"}
HF_FILES = ("*.json", "*.safetensors", "*.txt", "*.model")  # config, tokenizer and weights (no onnx/ or *.bin)
HF_BIN_FILES = ("pytorch_model*.bin",)  # fallback weights for repos without safetensors (mmBERT-small@abc32620)
HUB = "hub"


@dataclass(frozen=True)
class Source:
    model: str
    kind: str        # LAYA | HF
    path: Path       # absolute local directory
    label: str       # repo@revision[/subfolder], or the local path

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": str(self.path), "label": self.label}


def small_encoders(cfg: Mapping[str, Any]) -> dict[str, dict]:
    return dict((cfg.get("phase4") or {}).get("small_encoders") or {})


def model_kind(cfg: Mapping[str, Any], key: str) -> str:
    if key in (cfg.get("model") or {}):
        return LAYA
    if key in small_encoders(cfg):
        return HF
    known = sorted([*(cfg.get("model") or {}), *small_encoders(cfg)])
    raise ValueError(f"unknown model {key!r}: expected one of {known} (config model / phase4.small_encoders)")


def backends_for(cfg: Mapping[str, Any], key: str, requested: Sequence[str] | None) -> list[str]:
    """The backends to run for `key`: the requested ones it supports (default: config phase5.bench.backends)."""
    kind = model_kind(cfg, key)
    if requested is None:
        requested = ((cfg.get("phase5") or {}).get("bench") or {}).get("backends", {}).get(key) or BACKENDS[kind]
    names = [HF_ALIASES.get(b, b) if kind == HF else b for b in requested]
    unknown = sorted({b for b in names if b not in BACKENDS[LAYA] + BACKENDS[HF]})
    if unknown:
        raise ValueError(f"unknown backend(s) {unknown}: expected torch, onnx or hf")
    return list(dict.fromkeys(b for b in names if b in BACKENDS[kind]))


def parse_ckpt(values: Sequence[str] | None, models: Sequence[str]) -> dict[str, str]:
    """--ckpt values -> {model: 'hub' | path}: 'hub', one path (only with one model), or MODEL=PATH entries."""
    values = list(values or [HUB])
    out = {m: HUB for m in models}
    pairs = [v for v in values if "=" in v]
    plain = [v for v in values if "=" not in v]
    if len(plain) > 1 or (plain and pairs):
        raise ValueError("--ckpt takes 'hub', one path, or MODEL=PATH entries (not a mix)")
    if plain and plain[0] != HUB:
        if len(models) != 1:
            raise ValueError("a single --ckpt path needs exactly one model; use MODEL=/abs/dir entries")
        out[models[0]] = plain[0]
    for entry in pairs:
        key, _, path = entry.partition("=")
        if key not in out:
            raise ValueError(f"--ckpt {entry}: {key!r} is not among the benchmarked models {list(models)}")
        out[key] = path
    return out


def local_label(path: Path, root: Path | None = None) -> str:
    """A local checkpoint's label with no user path: repo-relative inside the project, else the folder name. The
    label is copied into bench JSON rows and from there into committed reports; the full path stays in `path`."""
    from .config import project_root

    base = Path(root) if root is not None else project_root()
    try:
        return f"local:{Path(path).resolve().relative_to(base.resolve()).as_posix()}"
    except ValueError:
        return f"local:{Path(path).name}"


def _absolute_dir(path_str: str, marker: str, what: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        raise ValueError(f"checkpoint paths must be absolute ('hub' for the pinned Hub model), got {path_str!r}")
    if not (path / marker).exists():
        raise FileNotFoundError(f"not a {what} directory (no {marker}): {path}")
    return path


def _laya_hub(cfg: Mapping[str, Any], key: str) -> tuple[Path, str]:
    from .hub import model_spec, snapshot

    spec = model_spec(dict(cfg), key)
    sub = f"/{spec.subfolder}" if spec.subfolder else ""
    return snapshot(spec).resolve(), f"{spec.repo_id}@{(spec.revision or 'default')[:12]}{sub}"


def _hf_hub(cfg: Mapping[str, Any], key: str) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    spec = small_encoders(cfg)[key]
    token = os.environ.get("HF_TOKEN") or None
    root = Path(snapshot_download(spec["id"], revision=spec.get("revision"), allow_patterns=list(HF_FILES),
                                  token=token))
    if not any(root.glob("*.safetensors")):
        root = Path(snapshot_download(spec["id"], revision=spec.get("revision"), allow_patterns=list(HF_BIN_FILES),
                                      token=token))
    return root.resolve(), f"{spec['id']}@{(spec.get('revision') or 'default')[:12]}"


def resolve_source(cfg: Mapping[str, Any], key: str, ckpt: str) -> Source:
    """`hub` -> the pinned checkpoint downloaded to the local cache; else an absolute existing directory."""
    kind = model_kind(cfg, key)
    if ckpt == HUB:
        path, label = _laya_hub(cfg, key) if kind == LAYA else _hf_hub(cfg, key)
    elif kind == LAYA:
        path = _absolute_dir(ckpt, "rl_agent_config.json", "Laya checkpoint")
        label = local_label(path)
    else:
        path = _absolute_dir(ckpt, "config.json", "Hugging Face model")
        label = local_label(path)
    return Source(model=key, kind=kind, path=path, label=label)
