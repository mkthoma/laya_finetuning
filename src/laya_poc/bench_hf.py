"""Small-encoder rows of the CPU benchmark (design doc §7.11 "run the fine-tuned small encoder the same way for the
relative-throughput row"): the B4 models (config phase4.small_encoders) as AutoModelForSequenceClassification.

Loaded exactly as B4 trains them (small_encoder_train.load_encoder: fp32, SDPA attention, a fresh `len(option_keys)`
head seeded 0 on the pinned base weights; latency does not depend on the head's values) and scored through B4's own
forward path (small_encoder_data.score_states: the rows' STATE strings, the model's tokenizer with special tokens,
truncation at config phase4.small_encoder_train.max_length = 256, length-sorted padded batches, no grad). Batch-1
latency is score_states on one state; throughput is score_states over the batch rows with `batch_size`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .bench_worker import Runtime

HEAD_SEED = 0
DEFAULT_MAX_LENGTH = 256


def max_length(cfg: dict) -> int:
    return int(((cfg.get("phase4") or {}).get("small_encoder_train") or {}).get("max_length", DEFAULT_MAX_LENGTH))


def load_model(cfg: dict, key: str, path: Path) -> tuple[Any, Any]:
    """(fp32 eval-mode sequence classifier, tokenizer) from a LOCAL model directory (the resolved pinned snapshot)."""
    from .labels import option_keys
    from .small_encoder_train import encoder_source, load_encoder

    model, tok = load_encoder(encoder_source(cfg, key, init=path), len(option_keys(cfg["labels"]["scheme"])),
                              HEAD_SEED)
    model.eval()
    return model, tok


def hf_loader(cfg: dict, key: str, path: Path, *, batch_size: int) -> Callable[[], Runtime]:
    length = max_length(cfg)

    def load() -> Runtime:
        import torch
        from .small_encoder_data import score_states

        model, tok = load_model(cfg, key, Path(path))
        kw = {"max_length": length, "device": "cpu", "fp16": False}
        return Runtime(
            predict_one=lambda s: score_states(model, tok, [s], batch_size=1, **kw),
            predict_many=lambda ss: score_states(model, tok, ss, batch_size=batch_size, **kw),
            info={"framework": "pytorch", "library": "transformers", "device": "cpu", "amp": False,
                  "dtype": str(next(model.parameters()).dtype).replace("torch.", ""), "max_length": length,
                  "attn_implementation": getattr(model.config, "_attn_implementation", None),
                  "num_labels": int(model.config.num_labels), "torch": torch.__version__})
    return load
