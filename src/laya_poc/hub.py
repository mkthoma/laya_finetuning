"""Load Laya checkpoints and tokenizers at the pinned Hub revision (config: laya.hub_revision).

Only public checkpoint files are fetched here; the gated FSQ data goes through `extract.py`.
Local checkpoint directories must be absolute paths: Laya treats a missing relative path as a
Hub repo id (agent.py:287-307).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    subfolder: str | None
    revision: str | None
    max_len: int
    head_max_len: int


def model_spec(cfg: dict[str, Any], key: str) -> ModelSpec:
    m = cfg["model"][key]
    return ModelSpec(key=key, repo_id=m["id"], subfolder=m.get("subfolder"),
                     revision=cfg["laya"].get("hub_revision"),
                     max_len=int(m["max_len"]), head_max_len=int(m["head_max_len"]))


def snapshot(spec: ModelSpec, files: tuple[str, ...] = CHECKPOINT_FILES) -> Path:
    """Download (or reuse from cache) the checkpoint files; returns the checkpoint directory."""
    from huggingface_hub import snapshot_download

    prefix = f"{spec.subfolder}/" if spec.subfolder else ""
    root = snapshot_download(spec.repo_id, revision=spec.revision,
                             allow_patterns=[prefix + f for f in files],
                             token=os.environ.get("HF_TOKEN") or None)
    return Path(root) / spec.subfolder if spec.subfolder else Path(root)


def load_tokenizer_dir(ckpt_dir: str | Path) -> Any:
    """Tokenizer of a local checkpoint directory, loaded exactly as the Laya runtime does."""
    from laya.agent import _fix_tokenizer_config, _load_tokenizer

    ckpt_dir = Path(ckpt_dir)
    _fix_tokenizer_config(str(ckpt_dir))
    return _load_tokenizer(str(ckpt_dir / "tokenizer"), {})


def load_tokenizer(spec: ModelSpec) -> Any:
    return load_tokenizer_dir(snapshot(spec, files=("tokenizer/*",)))


def load_agent(source: ModelSpec | str | Path, device: str = "cpu") -> Any:
    """A Laya Agent from the pinned Hub checkpoint (ModelSpec) or an absolute local directory."""
    import laya

    if isinstance(source, ModelSpec):
        return laya.load(source.repo_id, subfolder=source.subfolder, revision=source.revision, device=device)
    path = Path(source)
    if not path.is_absolute():
        raise ValueError(f"local checkpoint paths must be absolute, got {source!r}")
    if not (path / "rl_agent_config.json").exists():
        raise FileNotFoundError(f"not a Laya checkpoint directory: {path}")
    return laya.load(str(path), device=device)
