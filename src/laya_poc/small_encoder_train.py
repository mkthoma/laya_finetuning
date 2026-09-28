"""B4 fine-tuning: a small encoder with a softmax head on the rows Laya trains on (design §5.13 item 4, §7.10).

`AutoModelForSequenceClassification` at the pinned Hub revision (config phase4.small_encoders; ModernBERT-base and
mmBERT-small are both ModernBERT-architecture, so both load as ModernBertForSequenceClassification), fp32 weights,
SDPA attention. Input: the row's compact JSON state STRING through the model's own tokenizer, truncated at
max_length. Target: the row's gold probabilities, so label-smoothed rows and the uniform no-evidence rows train with
the same soft cross-entropy as Laya's CE term (HF's own loss would treat float labels as multi-label BCE).
Recipe (config phase4.small_encoder_train): AdamW lr 3e-5, weight decay 0.01 with none on biases, norms and 1-D
parameters (schedule.param_groups), batch 32, linear warm-up over 6% of the steps then linear decay to 0
(schedule.linear_warmup_lambda), gradients clipped at 1.0 (the HF Trainer default the doc's recipe assumes), fp16
autocast + GradScaler on CUDA only. Epoch k trains on train_e{k}.jsonl, the SAME augmented rows Laya saw, in
length-bucketed batches (sampler, seed*1000+k); torch is reseeded before every step (schedule.micro_seed) and before
the new head is initialised, so a seed fixes the run. After each epoch: val macro-F1 on the labelled val rows
(raw-logit argmax). The best epoch's weights (first on ties) are copied to CPU memory, restored at the end and
saved to <train_dir>/best (HF model + tokenizer). log.jsonl shares train_single's event names where they overlap
(`start` with micro_batch and n_items_per_epoch, `eval`, `done`), so the matrix reads epochs and seconds alike.
Tokenisation, soft targets, batching and scoring live in small_encoder_data.py.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .env_check import write_json
from .io_utils import RunLog, iter_jsonl
from .metrics import macro_f1
from .run_files import micro_timing, vram_gb
from .sampler import bucketed_batches, padding_ratio
from .schedule import linear_warmup_lambda, micro_seed, param_groups, plan_steps
from .small_encoder_data import ValSet, autocast, collate, load_epoch, score_states, soft_ce

MAX_GRAD_NORM = 1.0      # HF Trainer default; config phase4.small_encoder_train.max_grad_norm overrides
MAX_NONFINITE = 10       # non-finite losses tolerated before the run aborts (train_single's default)
PRINTS_PER_RUN = 10      # sparse progress: about this many step lines per run
BEST_DIR = "best"


# ---------------------------------------------------------------- settings

@dataclass(frozen=True)
class EncoderSource:
    key: str                 # phase4.small_encoders entry, e.g. modernbert_base
    repo_id: str
    revision: str | None
    init: Path | None = None  # an absolute local HF model dir replaces the Hub (tests, dry runs)

    @property
    def label(self) -> str:
        return str(self.init) if self.init else f"{self.repo_id}@{self.revision}"


def encoder_source(cfg: dict, key: str, init: str | Path | None = None) -> EncoderSource:
    encoders = (cfg.get("phase4") or {}).get("small_encoders") or {}
    if key not in encoders:
        raise ValueError(f"unknown small encoder {key!r}; config phase4.small_encoders has {sorted(encoders)}")
    if init is not None and not Path(init).is_absolute():
        raise ValueError(f"--init must be an absolute local model dir, got {str(init)!r}")
    if init is not None and not Path(init).is_dir():
        raise FileNotFoundError(f"--init {init}: no such directory")
    spec = encoders[key]
    return EncoderSource(key=key, repo_id=spec["id"], revision=spec.get("revision"),
                         init=None if init is None else Path(init))


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    lr: float
    batch_size: int
    epochs: int
    warmup_frac: float
    weight_decay: float
    max_length: int
    fp16: bool               # effective: config fp16 AND device == cuda
    max_grad_norm: float
    eval_batch_size: int
    max_steps: int | None = None


def train_config(cfg: dict, *, seed: int, device: str, epochs: int | None = None,
                 max_steps: int | None = None) -> TrainConfig:
    st = (cfg.get("phase4") or {}).get("small_encoder_train")
    if not st:
        raise ValueError("config has no phase4.small_encoder_train section")
    for name, v in (("--epochs", epochs), ("--max-steps", max_steps)):
        if v is not None and v < 1:
            raise ValueError(f"{name} must be >= 1, got {v}")
    return TrainConfig(seed=int(seed), device=device, lr=float(st["lr"]), batch_size=int(st["batch_size"]),
                       epochs=int(epochs or st["epochs"]), warmup_frac=float(st["warmup_frac"]),
                       weight_decay=float(st["weight_decay"]), max_length=int(st["max_length"]),
                       fp16=bool(st.get("fp16", True)) and device == "cuda",
                       max_grad_norm=float(st.get("max_grad_norm", MAX_GRAD_NORM)),
                       eval_batch_size=int((cfg.get("eval") or {}).get("batch_size", 64)), max_steps=max_steps)


def load_encoder(src: EncoderSource, num_labels: int, seed: int) -> tuple[Any, Any]:
    """(fp32 sequence classifier with a new `num_labels` head, tokenizer) on the CPU. Seeded first: the new head's
    initialisation is part of what a seed varies."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name, where = (str(src.init), {}) if src.init else (
        src.repo_id, {"revision": src.revision, "token": os.environ.get("HF_TOKEN") or None})
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels, dtype=torch.float32,
                                                               attn_implementation="sdpa", **where)
    tok = AutoTokenizer.from_pretrained(name, **where)
    if tok.pad_token_id is None:
        raise ValueError(f"{src.label}: the tokenizer has no pad token")
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.float32}:
        raise ValueError(f"{src.label}: expected fp32 parameters, got {sorted(map(str, dtypes))}")
    return model, tok


# ---------------------------------------------------------------- epoch selection

def val_metrics(model: Any, tok: Any, val: ValSet, tc: TrainConfig) -> dict[str, Any]:
    """Epoch selection metrics: macro-F1 over every option, accuracy and soft CE against the gold targets."""
    import torch

    Z = score_states(model, tok, val.states, max_length=tc.max_length, batch_size=tc.eval_batch_size,
                     device=tc.device, fp16=tc.fp16)
    yhat = Z.argmax(1)
    ce = float(soft_ce(torch.from_numpy(Z), torch.from_numpy(val.targets)).item())
    return {"val_macro_f1": macro_f1(val.y, yhat, labels=list(range(Z.shape[1]))),
            "val_acc": float((yhat == val.y).mean()), "val_ce": ce, "n": len(val.y)}


def cpu_state(model: Any) -> dict:
    """A CPU copy of the weights (copy=True: training on a CPU model must not change the saved best)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _count_rows(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def _gpu_name(device: str) -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if device == "cuda" else None


# ---------------------------------------------------------------- training

class EncoderTrainer:
    """One fine-tuning run: optimiser stack, per-epoch data, best-epoch tracking and the event log."""

    def __init__(self, model: Any, tok: Any, tc: TrainConfig, epoch_files: Sequence[Path], val: ValSet,
                 train_dir: Path, meta: dict):
        import torch

        self.model, self.tok, self.tc, self.files, self.val = model, tok, tc, list(epoch_files), val
        self.dir, self.meta, self.t0 = train_dir, dict(meta), time.perf_counter()
        self.n_rows = [_count_rows(f) for f in self.files]
        self.plan = plan_steps(self.n_rows, tc.batch_size, 1, tc.max_steps)
        self.natural_steps = plan_steps(self.n_rows, tc.batch_size, 1, None)["total_opt"]
        self.optimizer = torch.optim.AdamW(param_groups(model, tc.lr, tc.lr, tc.weight_decay))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, linear_warmup_lambda(self.plan["total_opt"], tc.warmup_frac))
        self.scaler = torch.amp.GradScaler("cuda", enabled=tc.fp16)
        self.log = RunLog(train_dir / "log.jsonl")
        self.print_every = max(1, self.plan["total_opt"] // PRINTS_PER_RUN)
        self.step, self.nonfinite, self.min_scale, self.truncated, self.pad_ratio = 0, 0, None, 0, None
        self.skipped = 0  # steps whose update was not applied (fp16 overflow, or a non-finite fp32 loss)
        self.epochs: list[dict] = []
        self.timings: list[float] = []
        self.best: tuple[int, float, dict] | None = None  # (epoch index, val macro-F1, CPU weights)

    def _at_cap(self) -> bool:
        return self.step >= self.plan["total_opt"]

    def log_start(self) -> None:
        tc, plan = self.tc, self.plan
        self.log.log(event="start", **self.meta, seed=tc.seed, device=tc.device, gpu=_gpu_name(tc.device),
                     fp16=tc.fp16, micro_batch=tc.batch_size, grad_accum=1, n_items_per_epoch=self.n_rows,
                     total_opt_steps=plan["total_opt"], total_micro_steps=plan["total_micro"], epochs=tc.epochs,
                     max_steps=tc.max_steps, max_length=tc.max_length, lr=tc.lr, warmup_frac=tc.warmup_frac,
                     weight_decay=tc.weight_decay, max_grad_norm=tc.max_grad_norm)
        print(f"start: B4 {self.meta.get('model')} seed {tc.seed} on {tc.device} (fp16 {tc.fp16}) | batch "
              f"{tc.batch_size} | {self.n_rows[0]} rows x {tc.epochs} epochs | {plan['total_opt']} steps", flush=True)

    def train(self) -> None:
        for epoch in range(self.tc.epochs):
            if self._at_cap():
                break
            self._run_epoch(epoch)

    def _run_epoch(self, epoch: int) -> None:
        t0, tc = time.perf_counter(), self.tc
        data = load_epoch(self.files[epoch], self.tok, self.val.keys, tc.max_length)
        lengths = [len(x) for x in data.ids]
        batches = bucketed_batches(lengths, tc.batch_size, tc.seed * 1000 + epoch)
        self.truncated += data.truncated
        self.pad_ratio = self.pad_ratio if self.pad_ratio is not None else padding_ratio(lengths, batches)
        self.model.train()
        losses = []
        for bi, batch in enumerate(batches):
            if self._at_cap():
                break
            losses.append(self._step(epoch, bi, [data.ids[i] for i in batch], data.targets[batch]))
        self._end_epoch(epoch, losses, time.perf_counter() - t0)

    def _step(self, epoch: int, bi: int, ids: list[list[int]], targets: np.ndarray) -> float:
        import torch

        tc, t0 = self.tc, time.perf_counter()
        torch.manual_seed(micro_seed(tc.seed, epoch, bi))  # dropout (if any) replays exactly
        input_ids, mask = collate(ids, self.tok.pad_token_id)
        with autocast(tc.device, tc.fp16):
            logits = self.model(input_ids=input_ids.to(tc.device), attention_mask=mask.to(tc.device)).logits
        loss = soft_ce(logits, torch.as_tensor(targets, dtype=torch.float32, device=tc.device))
        finite = bool(torch.isfinite(loss).item())
        if finite or self.scaler.is_enabled():  # GradScaler detects the overflow and skips the update itself
            self.scaler.scale(loss).backward()
            grad_norm, applied = self._optimizer_step()
        else:  # fp32 with a non-finite loss: no backward, no update (NaN grads would stick)
            self.optimizer.zero_grad(set_to_none=True)
            grad_norm, applied = None, False
        self.step += 1
        self.skipped += not applied
        value = float(loss.item())
        self._after_step(epoch, value, finite, grad_norm, time.perf_counter() - t0)
        return value

    def _optimizer_step(self) -> tuple[float, bool]:
        """(pre-clip grad norm, applied). An update GradScaler skipped (inf/NaN grads: the scale drops) does not
        advance the LR schedule, as the HF Trainer does, so every planned rate goes to an applied update."""
        import torch

        before = float(self.scaler.get_scale())
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale = float(self.scaler.get_scale())
        applied = scale >= before
        if applied:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.min_scale = scale if self.min_scale is None else min(self.min_scale, scale)
        return float(norm), applied

    def _after_step(self, epoch: int, loss: float, finite: bool, grad_norm: float | None, dt: float) -> None:
        self.timings.append(dt)
        if not finite:
            self.nonfinite += 1
            if self.nonfinite > MAX_NONFINITE:
                raise RuntimeError(f"{self.nonfinite} non-finite losses (> {MAX_NONFINITE}): aborting B4 training")
        if self.step % self.print_every and self.step != self.plan["total_opt"]:
            return
        lr = float(self.optimizer.param_groups[0]["lr"])
        self.log.log(event="step", step=self.step, epoch=epoch, loss=loss, finite=finite, lr=lr, grad_norm=grad_norm,
                     scale=float(self.scaler.get_scale()), sec_per_step=dt, **vram_gb(self.tc.device))
        print(f"step {self.step}/{self.plan['total_opt']} | epoch {epoch + 1} | loss {loss:.4f} | lr {lr:.2e} | "
              f"{dt:.3f} s/step", flush=True)

    def _end_epoch(self, epoch: int, losses: list[float], train_seconds: float) -> None:
        t0 = time.perf_counter()
        m = val_metrics(self.model, self.tok, self.val, self.tc)
        improved = self.best is None or m["val_macro_f1"] > self.best[1]
        if improved:
            self.best = (epoch, m["val_macro_f1"], cpu_state(self.model))
        ev = round(time.perf_counter() - t0, 2)
        rec = {"epoch": epoch + 1, "step": self.step, "train_loss": float(np.mean(losses)) if losses else None,
               "val_macro_f1": m["val_macro_f1"], "val_acc": m["val_acc"], "val_ce": m["val_ce"], "n_val": m["n"],
               "train_seconds": round(train_seconds, 2), "eval_seconds": ev, "seconds": round(train_seconds + ev, 2)}
        self.epochs.append(rec)
        self.log.log(event="eval", opt_step=self.step, epoch=epoch + 1, improved=improved, **m, seconds=ev)
        print(f"eval epoch {epoch + 1} @ step {self.step}: val macro-F1 {m['val_macro_f1']:.4f} acc {m['val_acc']:.3f}"
              f" ce {m['val_ce']:.4f} (n={m['n']}, {ev:.1f} s){' [best]' if improved else ''}", flush=True)

    def finish(self) -> dict:
        """Restore and save the best epoch's weights, then write summary.json (last: it marks success)."""
        if self.best is None:
            raise RuntimeError("no epoch was trained and evaluated")
        self.model.load_state_dict(self.best[2])
        best_dir = self.dir / BEST_DIR
        self.model.save_pretrained(str(best_dir))
        self.tok.save_pretrained(str(best_dir))
        summary = self.summary()
        write_json(self.dir / "summary.json", summary)
        self.log.log(event="done", micro_steps=self.step, opt_steps=self.step - self.skipped, stop_reason=summary["stop_reason"],
                     best_epoch=summary["best_epoch"])
        print(f"done: {self.step} steps ({summary['stop_reason']}) | best epoch {summary['best_epoch']} val macro-F1 "
              f"{summary['best_val_macro_f1']:.4f} | {summary['seconds']:.0f} s | {best_dir}", flush=True)
        return summary

    def _epochs_run(self) -> float:
        per = [np.ceil(n / self.tc.batch_size) for n in self.n_rows]
        steps = np.diff([0] + [e["step"] for e in self.epochs])
        return round(float(sum(s / p for s, p in zip(steps, per))), 2)

    def summary(self) -> dict[str, Any]:
        tc, (best_i, best_f1, _) = self.tc, self.best
        cut = tc.max_steps is not None and self.step < self.natural_steps
        vram = vram_gb(tc.device)
        return {**self.meta, "seed": tc.seed, "device": tc.device, "gpu": _gpu_name(tc.device), "fp16": tc.fp16,
                "epochs": tc.epochs, "epochs_run": self._epochs_run(), "micro_steps": self.step,
                "opt_steps": self.step - self.skipped, "opt_steps_skipped": self.skipped,
                "total_opt_steps": self.plan["total_opt"],
                "stop_reason": "max_steps" if cut else "epochs", "best_epoch": best_i + 1,
                "best_opt_step": self.epochs[best_i]["step"], "best_val_macro_f1": best_f1,
                "per_epoch": [{**e, "best": i == best_i} for i, e in enumerate(self.epochs)],
                "seconds": round(time.perf_counter() - self.t0, 2), **micro_timing(self.timings),
                "peak_vram_alloc_gb": vram["vram_alloc_gb"], "peak_vram_reserved_gb": vram["vram_reserved_gb"],
                "n_train_rows_per_epoch": self.n_rows, "truncated_rows": self.truncated,
                "padding_ratio": self.pad_ratio, "nonfinite": self.nonfinite, "min_scale": self.min_scale,
                "hparams": {"lr": tc.lr, "batch_size": tc.batch_size, "warmup_frac": tc.warmup_frac,
                            "weight_decay": tc.weight_decay, "max_length": tc.max_length,
                            "max_grad_norm": tc.max_grad_norm, "max_steps": tc.max_steps},
                "best_dir": str(self.dir / BEST_DIR)}


def train_encoder(model: Any, tok: Any, tc: TrainConfig, epoch_files: Sequence[Path], val: ValSet, train_dir: Path,
                  *, meta: dict) -> dict:
    """Fine-tune `model` in place on the epoch files; it ends holding the best epoch's weights. Returns the summary
    written to <train_dir>/summary.json (best weights in <train_dir>/best)."""
    missing = [str(f) for f in epoch_files if not Path(f).is_file()]
    if missing:
        raise FileNotFoundError(f"missing train epoch file(s): {', '.join(missing)}")
    if len(epoch_files) != tc.epochs:
        raise ValueError(f"{len(epoch_files)} epoch files for {tc.epochs} epochs")
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    model.to(tc.device)
    trainer = EncoderTrainer(model, tok, tc, epoch_files, val, Path(train_dir), meta)
    trainer.log_start()
    trainer.train()
    return trainer.finish()
