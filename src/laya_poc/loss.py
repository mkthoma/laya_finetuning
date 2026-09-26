"""The upstream training objective, ported verbatim from upstream_nb/cell08.py:152-173.

Soft-target cross-entropy plus a REINFORCE term on Gaussian-perturbed logits scored by Laya's
proper scoring rule. Every operation, and its order, matches the notebook so that under the same
RNG state the loss and its gradients are bit-for-bit identical. The noise is drawn from the
*global* RNG on purpose: the trainer reseeds it per micro-step, which makes resume exact.

`loss_rl`'s value is zero-mean-ish noise (about +/-2 for 10 options), so progress must be read
from `ce`, never from `total` (critique §B.11). `evaluate_items` measures the same CE on held-out
items, plus accuracy and macro-F1 on the raw-logit argmax.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from laya.common import collate_items, proper_reward

from .metrics import macro_f1

MODEL_INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


@dataclass(frozen=True)
class LossParts:
    total: torch.Tensor   # (rl + ce_weight * ce) / grad_accum + 0 * act.sum(); call backward on this
    ce: torch.Tensor      # soft-target cross-entropy, mean over the micro-batch
    rl: torch.Tensor      # policy-gradient term, mean over group_size x micro-batch
    reward: torch.Tensor  # mean proper-score reward of the noisy samples (no grad)


def _validate(logits, mask, target, sigma, group_size, grad_accum) -> None:
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2 (advantages are centred across the group), got {group_size}")
    if grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {grad_accum}")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be a bool tensor (collate_items marker_mask), got {mask.dtype}")
    if logits.shape != mask.shape or target.shape != mask.shape:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)}, mask {tuple(mask.shape)}, "
                         f"target {tuple(target.shape)}")


def upstream_loss(logits: torch.Tensor, act: torch.Tensor, mask: torch.Tensor, target: torch.Tensor,
                  qtype: torch.Tensor, sigma: float, *, group_size: int = 4, w_sph: float = 0.75,
                  w_rps: float = 1.0, ce_weight: float = 1.0, grad_accum: int = 1) -> LossParts:
    """cell08.py:152-173 with GROUP_SIZE, w_sph, w_rps, the CE weight and GRAD_ACCUM as arguments.

    `+ 0.0 * act.sum()` is kept: it gives act_head zero (not None) grads, so AdamW still applies
    weight decay to it exactly as upstream does.
    """
    _validate(logits, mask, target, sigma, group_size, grad_accum)
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()

    # 1. Sample G noisy logit distributions with zero-mean projection
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    # 2. Evaluate proper scoring reward
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    # 3. Policy gradient loss + soft cross-entropy guidance
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = (loss_rl + ce_weight * loss_ce) / grad_accum + 0.0 * act.sum()
    return LossParts(total=loss, ce=loss_ce, rl=loss_rl, reward=r.mean())


def autocast(device: str, fp16: bool) -> Any:
    """fp16 autocast around the forward only, on CUDA only (cell08.py:143); a no-op on CPU."""
    return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" and fp16 else nullcontext()


def forward(model: Any, batch: dict, device: str, fp16: bool) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(option logits, act logits, the batch tensors on `device`) for a collate_items batch."""
    t = {k: batch[k].to(device) for k in (*MODEL_INPUTS, "target")}
    with autocast(device, fp16):
        logits, act = model(*(t[k] for k in MODEL_INPUTS))
    return logits, act, t


def evaluate_items(model: Any, items: list[dict], *, device: str, fp16: bool, pad_id: int,
                   batch_size: int = 32) -> dict:
    """Soft-target CE, accuracy and macro-F1 (raw-logit argmax vs item["label"]).

    Runs in eval() under no_grad and restores the previous mode: left in eval(), dropout and
    head checkpointing would silently stay off for the rest of training (common.py:189).
    """
    if not items:
        raise ValueError("no labelled items to evaluate")
    was_training, ce, pred = model.training, [], []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                batch = collate_items([items[start:start + batch_size]], pad_id)
                logits, _, t = forward(model, batch, device, fp16)
                logits = logits.float().masked_fill(~t["marker_mask"], -1e4)
                ce.append(-(t["target"] * torch.log_softmax(logits, -1)).sum(-1).cpu())
                pred.append(logits.argmax(-1).cpu())
    finally:
        model.train(was_training)
    y, yhat = np.array([it["label"] for it in items]), torch.cat(pred).numpy()
    k = max(len(it["markers"]) for it in items)
    return {"val_ce": float(torch.cat(ce).mean()), "val_acc": float((yhat == y).mean()),
            "val_macro_f1": macro_f1(y, yhat, labels=list(range(k))), "n": len(items)}
