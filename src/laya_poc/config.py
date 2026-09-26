"""Config loading and derived settings (design doc §7.3)."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .splits import SplitSpec

PROJECT_ROOT_ENV = "LAYA_POC_ROOT"
KNOWN_CARDS = ("T4", "L4", "A10")


def project_root() -> Path:
    """Repo root: $LAYA_POC_ROOT if set, else two levels above this file (src/laya_poc/..)."""
    env = os.environ.get(PROJECT_ROOT_ENV)
    return Path(env) if env else Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else project_root() / "config.yaml"
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = ("laya", "data", "serialise", "labels", "model", "train", "smoke")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config missing sections: {missing}")
    tc = cfg["train"]
    for card, mb in tc["micro_batch"].items():
        if tc["effective_batch"] % mb:
            raise ValueError(f"effective_batch {tc['effective_batch']} not divisible by micro_batch {mb} ({card})")
    sm = cfg["smoke"]
    if sm["kill_at_micro_step"] >= sm["micro_steps"]:
        raise ValueError("smoke.kill_at_micro_step must be before smoke.micro_steps")
    if sm["kill_at_micro_step"] <= sm["ckpt_every_micro_steps"]:
        raise ValueError("smoke.kill_at_micro_step must come after the first checkpoint")
    for card, mb in tc["micro_batch"].items():
        acc = tc["effective_batch"] // mb
        if sm["ckpt_every_micro_steps"] % acc:
            raise ValueError(f"smoke.ckpt_every_micro_steps must be a multiple of grad accumulation {acc} ({card})")


_CARD_PATTERNS = {"T4": re.compile(r"\bT4\b"), "L4": re.compile(r"\bL4\b"), "A10": re.compile(r"\bA10G?\b")}


def card_from_gpu_name(name: str) -> str:
    """Map a CUDA device name to a card profile in config.train.micro_batch (A100 is not an A10)."""
    upper = name.upper()
    for card, pat in _CARD_PATTERNS.items():
        if pat.search(upper):
            return card
    raise ValueError(f"no card profile for GPU {name!r}; set CARD to one of {KNOWN_CARDS}")


def accumulation(cfg: dict[str, Any], card: str) -> tuple[int, int]:
    """(micro_batch, grad-accumulation steps) for a card."""
    tc = cfg["train"]
    mb = tc["micro_batch"][card]
    return mb, tc["effective_batch"] // mb


def split_spec(cfg: dict[str, Any], smoke: bool = False) -> SplitSpec:
    d = cfg["data"]
    sm = cfg["smoke"] if smoke else {}
    return SplitSpec(
        id_countries=tuple(d["id_countries"]),
        ood_country=tuple(d["ood_country"]) if not smoke or sm["ood_size"] else (),
        ood_script=d["ood_script"],
        train_size=sm.get("train_questions", d["train_size"]),
        train_cap_per_class=sm.get("train_cap_per_class", d["train_cap_per_class"]),
        val_size=sm.get("val_size", d["val_size"]),
        test_size=sm.get("test_size", d["test_size"]),
        ood_size=sm.get("ood_size", d["ood_size"]),
        brand_top_n=sm.get("brand_top_n", d["brand_top_n"]),
        max_rows_per_name_country=d["max_rows_per_name_country"],
        seed=d["split_seed"],
        strict=not smoke,
    )


def with_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with dotted-key overrides applied, e.g. {"train.epochs": 1}."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return out
