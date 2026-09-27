"""How a training run is laid out: exploration sigma, learning rate, parameter groups, seeds,
batching, epochs, the items of each epoch file and the exact number of micro/optimiser steps;
train_single's options resolved into Settings, and the fingerprint a resumed run must share.

Sigma follows upstream exactly (a per-epoch step function, cell08.py:133-134). The learning-rate
schedule and the no-decay parameter groups are the design doc's deliberate deviations (§5.9):
linear warm-up then linear decay to 0, and no weight decay on biases, norms and 1-D parameters.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import card_from_gpu_name
from .io_utils import iter_jsonl
from .items import build_items, count_state_tokens, internal_question, state_room

# micro_seed layout: seed * 10**9 + epoch * 10**6 + micro_idx (collision-free inside these bounds).
_MAX_EPOCHS = 1000
_MAX_MICRO_PER_EPOCH = 10 ** 6
_MAX_SEED = (2 ** 63 - 1) // 10 ** 9 - 1


def sigma_for_epoch(epoch: int, epochs: int, start: float, end: float) -> float:
    """Upstream noise scale: 0.4/0.3/0.2/0.1 over 4 epochs, constant `start` for 1 epoch."""
    if epochs < 1 or not 0 <= epoch < epochs:
        raise ValueError(f"epoch {epoch} outside range(0, {epochs})")
    progress = epoch / max(1, epochs - 1)
    return start + (end - start) * progress


def linear_warmup_lambda(total_steps: int, warmup_frac: float) -> Callable[[int], float]:
    """LambdaLR factor: linear warm-up over int(warmup_frac * total) steps, then linear decay to 0.

    LambdaLR evaluates f(n) for the (n+1)-th optimiser step, so the first step runs at
    1/warm of the base rate and the last one at 1/(total - warm), reaching 0 after it.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if not 0.0 <= warmup_frac < 1.0:
        raise ValueError(f"warmup_frac must be in [0, 1), got {warmup_frac}")
    warm = int(warmup_frac * total_steps)

    def factor(step: int) -> float:
        if step < warm:
            return min(1.0, (step + 1) / max(1, warm))
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    return factor


def _no_decay(name: str, param) -> bool:
    return param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()


def param_groups(model, lr_encoder: float, lr_head: float, weight_decay: float) -> list[dict]:
    """AdamW groups: encoder (names starting "encoder.") vs head, each split into decay / no-decay.

    Every group carries `name` and `group` ("encoder" | "head") so the trainer can log one
    learning rate per side. Empty groups are dropped; frozen parameters are skipped.
    """
    buckets: dict[str, list] = {"encoder": [], "encoder_no_decay": [], "head": [], "head_no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        side = "encoder" if name.startswith("encoder.") else "head"
        buckets[side + ("_no_decay" if _no_decay(name, p) else "")].append(p)
    groups = []
    for key, params in buckets.items():
        if not params:
            continue
        side = key.split("_", 1)[0]
        groups.append({"params": params, "name": key, "group": side,
                       "lr": lr_encoder if side == "encoder" else lr_head,
                       "weight_decay": 0.0 if key.endswith("_no_decay") else weight_decay})
    return groups


def micro_seed(seed: int, epoch: int, micro_idx: int) -> int:
    """Seed for one micro-step: resume then needs no RNG state, only (epoch, micro_idx)."""
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed}")
    if not 0 <= epoch < _MAX_EPOCHS:
        raise ValueError(f"epoch must be in [0, {_MAX_EPOCHS}), got {epoch}")
    if not 0 <= micro_idx < _MAX_MICRO_PER_EPOCH:
        raise ValueError(f"micro_idx must be in [0, {_MAX_MICRO_PER_EPOCH}), got {micro_idx}")
    return seed * 10 ** 9 + epoch * _MAX_MICRO_PER_EPOCH + micro_idx


def plan_steps(n_items_per_epoch: Sequence[int], micro_batch: int, grad_accum: int,
               max_micro_steps: int | None) -> dict[str, int]:
    """Micro and optimiser steps the trainer will run.

    Per epoch there are ceil(n / micro_batch) micro-batches (sampler.bucketed_batches) and an
    optimiser step every `grad_accum` of them, plus one on the partial last window of the epoch
    (cell08.py:178) and one at the `max_micro_steps` cap, where the run stops.
    """
    if micro_batch < 1 or grad_accum < 1:
        raise ValueError(f"micro_batch and grad_accum must be >= 1, got {micro_batch}, {grad_accum}")
    if max_micro_steps is not None and max_micro_steps < 1:
        raise ValueError(f"max_micro_steps must be >= 1 or None, got {max_micro_steps}")
    total_micro = total_opt = 0
    for n in n_items_per_epoch:
        n_batches = math.ceil(n / micro_batch)
        if max_micro_steps is not None:
            n_batches = min(n_batches, max_micro_steps - total_micro)
        total_micro += n_batches
        total_opt += math.ceil(n_batches / grad_accum)
        if max_micro_steps is not None and total_micro >= max_micro_steps:
            break
    return {"total_micro": total_micro, "total_opt": total_opt}


# ---- run layout: batching, epochs and item counts (what plan_steps is fed) ----

CPU_BATCHING = (2, 4)  # (micro-batch, effective batch) for local CPU runs


def batching(cfg: dict, card: str, micro_batch: int | None = None,
             effective_batch: int | None = None) -> tuple[int, int]:
    """(micro-batch, grad accumulation) from CLI overrides, else the CPU default or card profile."""
    tc = cfg["train"]
    mb0, eff0 = CPU_BATCHING if card == "CPU" else (tc["micro_batch"][card], tc["effective_batch"])
    mb, eff = micro_batch or mb0, effective_batch or eff0
    if mb < 1 or eff < mb or eff % mb:
        raise ValueError(f"effective batch {eff} must be a positive multiple of micro-batch {mb}")
    return mb, eff // mb


def available_epochs(data_dir: Path) -> int:
    """Consecutive train_e{k}.jsonl files present (build_data writes one per planned epoch)."""
    n = 0
    while (Path(data_dir) / f"train_e{n}.jsonl").exists():
        n += 1
    return n


def resolve_epochs(requested: int | None, data_dir: Path) -> int:
    """--epochs, else every epoch file build_data wrote (1 for the smoke data)."""
    have = available_epochs(data_dir)
    if have == 0:
        raise FileNotFoundError(f"no train_e0.jsonl in {data_dir}; run build_data first")
    epochs = requested or have
    if epochs > have:
        raise FileNotFoundError(f"--epochs {epochs} needs train_e0..train_e{epochs - 1}.jsonl; {data_dir} has {have}")
    return epochs


def count_items(path: Path) -> int:
    """Items a JSONL file yields (one per question per row) without tokenising anything."""
    return sum(len(json.loads(row["questions"])) for row in iter_jsonl(path))


def load_items(path: Path, tok: Any, max_len: int, head_max_len: int, *,
               labelled_only: bool = False) -> tuple[list[dict], int]:
    """Training items of one JSONL file, and how many of them had their state truncated."""
    items, truncated, rooms = [], 0, {}
    for row in iter_jsonl(path):
        if labelled_only and row.get("label") is None:
            continue
        row_items = build_items(tok, row, max_len, head_max_len)
        if any(len(it["markers"]) != len(it["target"]) for it in row_items):
            raise ValueError(f"row {row.get('id')}: marker count differs from the number of options")
        qkey = row["questions"]
        if qkey not in rooms:
            rooms[qkey] = [state_room(tok, internal_question(qid, q), max_len, head_max_len)
                           for qid, q in json.loads(qkey).items()]
        n_tokens = count_state_tokens(tok, row["state"])
        truncated += sum(n_tokens > room for room in rooms[qkey])
        items.extend(row_items)
    return items, truncated


# ---- train_single's options, resolved into run settings (CLI options + config + hardware) ----

TRAIN_DESCRIPTION = "Single-process port of upstream train_ddp.py (upstream_nb/cell08.py main()), resumable."


def _count(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    """train_single's command line (resolve_settings turns the parsed options into Settings)."""
    p = argparse.ArgumentParser(prog="python -m laya_poc.train_single", description=TRAIN_DESCRIPTION)
    p.add_argument("--run-dir", required=True, help="absolute run directory (log.jsonl, ckpt/, final/)")
    p.add_argument("--data-dir", required=True, help="absolute data directory (train_e{k}.jsonl, val.jsonl)")
    p.add_argument("--config", default=None, help="config.yaml (default: <project root>/config.yaml)")
    p.add_argument("--model", choices=("laya", "laya_ml"), default="laya")
    p.add_argument("--seed", type=_count, default=11)
    p.add_argument("--init", default="hub", help="'hub' (pinned revision) or an absolute Laya checkpoint dir")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--card", choices=("auto", "T4", "L4", "A10", "G4", "CPU"), default="auto")
    p.add_argument("--grad-ckpt", choices=("auto", "on", "off"), default="auto")
    for flag in ("--epochs", "--max-micro-steps", "--crash-at-micro-step", "--micro-batch", "--effective-batch",
                 "--ckpt-every-micro-steps", "--eval-every-opt-steps"):
        p.add_argument(flag, type=_count, default=None, help="0 disables" if "every" in flag else None)
    p.add_argument("--keep-last", type=_count, default=None, help="checkpoints kept, >= 1 (default: config)")
    p.add_argument("--ckpt-every-min", type=float, default=None, help="0 disables (default: config)")
    p.add_argument("--max-nonfinite", type=_count, default=10)
    p.add_argument("--print-every", type=_count, default=10)
    p.add_argument("--initial-eval", action="store_true", help="eval val at opt 0, before training (not on resume)")
    p.add_argument("--final-eval", action="store_true")
    p.add_argument("--save-final", action="store_true")
    p.add_argument("--save-best", action="store_true", help="export the best val macro-F1 weights to <run-dir>/best")
    p.add_argument("--mirror-dir", default=None, help="absolute directory for a copy of each checkpoint")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="head-only (E4, design §7.10 Option B): freeze the encoder, train the decision head alone")
    return p


def abs_path(value: str | None, flag: str) -> Path | None:
    if value is not None and not Path(value).is_absolute():
        raise ValueError(f"{flag} must be an absolute path, got {value!r}")
    return None if value is None else Path(value)


@dataclass(frozen=True)
class Settings:
    model: str
    seed: int
    device: str
    card: str
    micro_batch: int
    grad_accum: int
    grad_ckpt: bool
    fp16: bool
    epochs: int
    max_len: int
    head_max_len: int
    max_micro_steps: int | None
    ckpt_every_micro_steps: int | None
    ckpt_every_min: float
    eval_every_opt_steps: int | None
    keep_last: int
    eval_batch_size: int = 32
    head_only: bool = False  # --freeze-encoder: the encoder is frozen, the optimiser holds the head alone


def _or(value: Any, default: Any) -> Any:
    return default if value is None else value


def resolve_device_and_card(device: str, card: str) -> tuple[str, str]:
    """Fail fast on a CPU runtime; card=auto maps the GPU name to a config profile (CPU on cpu)."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available (CPU runtime?); connect a GPU or use --device cpu")
    if card != "auto":
        return device, card
    return device, "CPU" if device == "cpu" else card_from_gpu_name(torch.cuda.get_device_name(0))


def resolve_settings(opts: Any, cfg: dict, data_dir: Path) -> Settings:
    """Settings from train_single's parsed options (None = take the config default)."""
    tc, mc = cfg["train"], cfg["model"][opts.model]
    device, card = resolve_device_and_card(opts.device, opts.card)
    mb, acc = batching(cfg, card, opts.micro_batch, opts.effective_batch)
    grad_ckpt = {"on": True, "off": False}.get(opts.grad_ckpt, bool(tc["grad_ckpt"].get(card, False)))
    keep_last = int(_or(opts.keep_last, tc["keep_last"]))
    if keep_last < 1:  # else save_train_ckpt fails only at the first checkpoint, after the compute is spent
        raise ValueError(f"--keep-last / train.keep_last must be >= 1 (checkpoints to keep), got {keep_last}")
    return Settings(
        model=opts.model, seed=opts.seed, device=device, card=card, micro_batch=mb, grad_accum=acc,
        grad_ckpt=grad_ckpt, fp16=device == "cuda" and bool(tc["fp16"]), epochs=resolve_epochs(opts.epochs, data_dir),
        max_len=int(mc["max_len"]), head_max_len=int(mc["head_max_len"]), max_micro_steps=opts.max_micro_steps,
        ckpt_every_micro_steps=opts.ckpt_every_micro_steps or None,
        ckpt_every_min=float(_or(opts.ckpt_every_min, tc["ckpt_every_min"])),
        eval_every_opt_steps=_or(opts.eval_every_opt_steps, tc["eval_every_opt_steps"]) or None,
        keep_last=keep_last, eval_batch_size=int(cfg.get("eval", {}).get("batch_size", 32)),
        head_only=bool(opts.freeze_encoder))


# ---- resume identity: the settings a checkpoint must have been written with ----

def run_fingerprint(s: Settings, n_items_per_epoch: Sequence[int]) -> dict:
    """What a resumed run must share with its checkpoint: the batches, the step plan (and so the LR schedule)
    and the optimiser's parameter groups (head_only). Hardware and checkpoint/eval cadence may change."""
    return {"model": s.model, "seed": s.seed, "micro_batch": s.micro_batch, "grad_accum": s.grad_accum,
            "epochs": s.epochs, "max_micro_steps": s.max_micro_steps, "n_items_per_epoch": list(n_items_per_epoch),
            "head_only": s.head_only}


def _as_current(saved: Any) -> dict | None:
    """A fingerprint in today's form: one written before --freeze-encoder existed was full fine-tuning."""
    return {"head_only": False, **saved} if isinstance(saved, dict) else None


def same_run(saved: Any, current: dict) -> bool:
    """True when a checkpoint's fingerprint (`saved`, possibly None or older) is this run's."""
    return _as_current(saved) == current


def check_same_run(path: Path, saved: Any, current: dict) -> None:
    """Refuse to resume from a checkpoint written with other settings, naming the ones that differ."""
    old = _as_current(saved)
    if old == current:
        return
    if old is None:
        diff = "the checkpoint has no settings fingerprint"
    else:
        keys = [k for k in current if old.get(k) != current[k]] + [k for k in old if k not in current]
        diff = ", ".join(f"{k}: {old.get(k)!r} in the checkpoint, {current.get(k)!r} now" for k in keys)
    raise ValueError(f"{path} was written with different settings ({diff}); rerun with the checkpoint's "
                     "settings or use a new --run-dir")


def saved_fingerprint(path: Path) -> dict | None:
    """The fingerprint inside a training checkpoint, or None when it cannot be read. Reads the whole file with
    weights-only unpickling (the loop state holds plain Python values; a type outside torch's allow-list gives
    None): used only to explain a restore that failed, whose own error is then re-raised."""
    import torch

    try:
        fingerprint = torch.load(str(path), map_location="cpu", weights_only=True)["state"]["fingerprint"]
    except Exception:  # unreadable or not a checkpoint: the caller re-raises the restore's own error
        return None
    return fingerprint if isinstance(fingerprint, dict) else None


def load_same_run(path: Path, current: dict, **objs: Any) -> dict:
    """ckpt.load_train_ckpt (model, optimizer, scheduler, scaler restored in place) for a checkpoint of THIS run.

    Another run's checkpoint may fail to load before its fingerprint can be compared (a head-only optimiser has
    other parameter groups than a full one): the settings that differ are then named instead of torch's error.
    """
    from . import ckpt

    try:
        saved = ckpt.load_train_ckpt(path, **objs)
    except (ValueError, RuntimeError, KeyError):
        fingerprint = saved_fingerprint(path)
        if fingerprint is not None:
            check_same_run(path, fingerprint, current)
        raise
    check_same_run(path, saved.get("fingerprint"), current)
    return saved
