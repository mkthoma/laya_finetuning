"""Head-only training (E4, design §7.10 Option B): the encoder is frozen and only the decision head trains.

Freezing is `requires_grad=False` on every encoder parameter, done before the optimiser is built, so
schedule.param_groups (which skips frozen parameters) hands AdamW the head groups alone, with lr_head and
the usual weight-decay rules. Frozen parameters never get a .grad, so the optimiser, the GradScaler's
unscale_ and clip_grad_norm_ see only the head. Nothing backpropagates into the encoder, so its activations
are not kept and there is nothing for gradient checkpointing to recompute: it is switched off for the encoder,
while the head's checkpointing stays as the card profile sets it. Loss, schedule, eval, save-best and resume
are the full fine-tuning code paths; the encoder's dropouts are 0.0 in both Laya checkpoints, so leaving it in
train() mode changes nothing. The frozen encoder is still saved in every checkpoint and export, unchanged.
"""
from __future__ import annotations

from typing import Any


def freeze_encoder(model: Any) -> int:
    """Freeze `model.encoder` in place (a torch module is set up by mutation); returns the frozen parameter count."""
    encoder = model.encoder
    if getattr(encoder, "is_gradient_checkpointing", False):
        encoder.gradient_checkpointing_disable()
    frozen = 0
    for p in encoder.parameters():
        p.requires_grad_(False)
        frozen += p.numel()
    return frozen


def trainable_parameters(model: Any) -> int:
    """Parameters the optimiser updates (logged in the start event: the evidence that a freeze took effect)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
