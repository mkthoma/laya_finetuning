"""Single-process port of upstream train_ddp.py (upstream_nb/cell08.py main()), resumable.

Kept from upstream: fp16 autocast around the forward only; the verbatim loss (loss.py);
unscale -> clip -> step -> update -> scheduler.step -> zero_grad at every accumulation boundary
and on the partial last window of each epoch; zero_grad at every epoch start; per-epoch sigma.
Deliberate deviations (config.yaml, doc §5.9): AdamW groups without decay on biases/norms,
linear warm-up + linear decay, effective batch from the config. Resume is exact: checkpoints
are written only at optimiser boundaries and every micro-step reseeds the global RNG.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from laya.common import collate_items

from . import ckpt as C
from . import export, hub
from .config import load_config
from .io_utils import RunLog, iter_jsonl
from .items import build_items, count_state_tokens, internal_question, state_room
from .loss import evaluate_items, forward, upstream_loss
from .sampler import bucketed_batches, padding_ratio
from .schedule import (Settings, count_items, linear_warmup_lambda, micro_seed, param_groups, plan_steps,
                       resolve_settings, sigma_for_epoch)

SKIP_TIMING = 5  # first micro-steps of each process (warm-up) are excluded from the s/micro stats
GB = 10 ** 9     # decimal GB, the unit of the doc's "<= 14 GB" cap (smoke.exit.max_vram_gb); not GiB


def load_model(init: str, cfg: dict, s: Settings) -> tuple[Any, Any, dict]:
    """Model, tokenizer and base config through the library load path (critique §A 7.6.3-1)."""
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


def vram_gb(device: str) -> dict:
    if device != "cuda":
        return {"vram_alloc_gb": None, "vram_reserved_gb": None}
    return {"vram_alloc_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            "vram_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3)}


class Trainer:
    """Mutable loop context: model, optimiser stack, the resumable `state` and the event log."""

    def __init__(self, args: argparse.Namespace, cfg: dict, s: Settings, model: Any, tok: Any, run_dir: Path,
                 data_dir: Path):
        self.args, self.cfg, self.s, self.model, self.tok = args, cfg, s, model, tok
        self.run_dir, self.data_dir, self.lc, tc = run_dir, data_dir, cfg["train"]["loss"], cfg["train"]
        self.n_items = [count_items(data_dir / f"train_e{e}.jsonl") for e in range(s.epochs)]
        self.plan = plan_steps(self.n_items, s.micro_batch, s.grad_accum, s.max_micro_steps)
        self.optimizer = torch.optim.AdamW(param_groups(model, tc["lr_encoder"], tc["lr_head"], tc["weight_decay"]))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, linear_warmup_lambda(self.plan["total_opt"], tc["warmup_frac"]))
        self.scaler = torch.amp.GradScaler("cuda", enabled=s.fp16)
        self.log = RunLog(run_dir / "log.jsonl")
        fingerprint = {"model": s.model, "seed": s.seed, "micro_batch": s.micro_batch, "grad_accum": s.grad_accum,
                       "epochs": s.epochs, "max_micro_steps": s.max_micro_steps,
                       "n_items_per_epoch": list(self.n_items)}
        self.state = {"epoch": 0, "batch_idx": 0, "micro_step": 0, "opt_step": 0, "best": -1.0, "bad": 0,
                      "nonfinite": 0, "min_scale": None, "fingerprint": fingerprint}
        self.resumed_from: tuple[str, int, int] | None = None
        self.timings: list[float] = []
        self.window: list[float] = []
        self.truncated, self.pad_ratio, self.stop_reason, self.initial_eval = 0, None, None, None
        self._epoch_cache: tuple[int, tuple] | None = None
        self._val_items: list[dict] | None = None
        self._last_ckpt_t, self._prev_boundary_micro = time.monotonic(), 0

    def _at_cap(self) -> bool:
        return self.s.max_micro_steps is not None and self.state["micro_step"] >= self.s.max_micro_steps

    def restore(self, mirror_dir: str | None) -> None:
        path = C.latest_ckpt(self.run_dir / "ckpt", mirror_dir)
        if path is None:
            return
        saved = C.load_train_ckpt(path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                                  scaler=self.scaler)
        if saved.get("fingerprint") != self.state["fingerprint"]:
            raise ValueError(f"{path} was written with different settings ({saved.get('fingerprint')} vs "
                             f"{self.state['fingerprint']}); use a new --run-dir")
        self.state = saved
        self.resumed_from = (str(path), saved["micro_step"], saved["opt_step"])
        self._prev_boundary_micro = saved["micro_step"]

    def epoch_data(self, epoch: int) -> tuple[list[dict], list[list[int]]]:
        if self._epoch_cache and self._epoch_cache[0] == epoch:
            return self._epoch_cache[1]
        s = self.s
        items, truncated = load_items(self.data_dir / f"train_e{epoch}.jsonl", self.tok, s.max_len, s.head_max_len)
        if len(items) != self.n_items[epoch]:
            raise RuntimeError(f"train_e{epoch}.jsonl changed while training ({len(items)} != {self.n_items[epoch]})")
        lengths = [len(it["ids"]) for it in items]
        batches = bucketed_batches(lengths, s.micro_batch, s.seed * 1000 + epoch)  # same order on resume
        self.truncated += truncated
        self.pad_ratio = padding_ratio(lengths, batches) if self.pad_ratio is None else self.pad_ratio
        self._epoch_cache = (epoch, (items, batches))
        return items, batches

    def log_start(self) -> None:
        s, st, plan = self.s, self.state, self.plan
        self.epoch_data(min(st["epoch"], s.epochs - 1))
        self.log.log(event="start", model=s.model, seed=s.seed, device=s.device, card=s.card,
                     micro_batch=s.micro_batch, grad_accum=s.grad_accum, n_items=sum(self.n_items),
                     n_items_per_epoch=self.n_items, total_opt_steps=plan["total_opt"],
                     total_micro_steps=plan["total_micro"], padding_ratio=self.pad_ratio,
                     truncated_items=self.truncated, grad_ckpt=s.grad_ckpt, fp16=s.fp16, epochs=s.epochs,
                     max_len=s.max_len, head_max_len=s.head_max_len)
        print(f"start: {s.model} seed {s.seed} on {s.device}/{s.card} | MB {s.micro_batch} x ACC {s.grad_accum}"
              f" | {sum(self.n_items)} items | {plan['total_micro']} micro / {plan['total_opt']} opt steps"
              f" | padding {self.pad_ratio:.2f} | truncated {self.truncated}", flush=True)
        if self.resumed_from:
            path, micro, opt = self.resumed_from
            self.log.log(event="resumed", path=path, from_micro_step=micro, from_opt_step=opt, epoch=st["epoch"])
            print(f"resumed from {path} (micro-step {micro}, opt-step {opt})", flush=True)

    def train(self) -> None:
        for epoch in range(self.state["epoch"], self.s.epochs):
            if self._at_cap() or self.stop_reason:
                break
            self._run_epoch(epoch)
        self.stop_reason = self.stop_reason or ("max_micro_steps" if self._at_cap() else "epochs")

    def _run_epoch(self, epoch: int) -> None:
        items, batches = self.epoch_data(epoch)
        sigma = sigma_for_epoch(epoch, self.s.epochs, self.lc["sigma_start"], self.lc["sigma_end"])
        start = self.state["batch_idx"] if epoch == self.state["epoch"] else 0
        self.optimizer.zero_grad(set_to_none=True)
        for bi in range(start, len(batches)):
            t0 = time.perf_counter()
            finite, ce, rl = self._micro_step(epoch, bi, [items[i] for i in batches[bi]], sigma)
            last = bi == len(batches) - 1
            boundary = (bi + 1) % self.s.grad_accum == 0 or last or self._at_cap()
            stepped = self._optimizer_step() if boundary else None
            self._after_micro(time.perf_counter() - t0, finite, ce, rl)  # timed incl. the optimiser step
            if stepped is not None:
                self._after_boundary(epoch, bi, last, *stepped)
            if self._at_cap() or self.stop_reason:
                return

    def _micro_step(self, epoch: int, bi: int, chunk: list[dict], sigma: float) -> tuple[bool, float, float]:
        lc, st = self.lc, self.state
        torch.manual_seed(micro_seed(self.s.seed, epoch, bi))  # noise + dropout replay exactly on resume
        logits, act, t = forward(self.model, collate_items([chunk], self.tok.pad_token_id), self.s.device, self.s.fp16)
        parts = upstream_loss(logits, act, t["marker_mask"], t["target"], t["qtype"], sigma,
                              group_size=lc["group_size"], w_sph=lc["w_sph"], w_rps=lc["w_rps"],
                              ce_weight=lc["ce_weight"], grad_accum=self.s.grad_accum)
        finite = bool(torch.isfinite(parts.total).item())
        if finite or self.scaler.is_enabled():  # GradScaler skips such steps; without it NaN grads would stick
            self.scaler.scale(parts.total).backward()
        st["micro_step"] += 1
        ce, rl = parts.ce.item(), parts.rl.item()  # float() on a grad tensor warns
        self.log.log(event="micro", micro_step=st["micro_step"], epoch=epoch, loss=rl + lc["ce_weight"] * ce,
                     loss_ce=ce, loss_rl=rl, reward=float(parts.reward), sigma=sigma, finite=finite)
        return finite, ce, rl

    def _after_micro(self, dt: float, finite: bool, ce: float, rl: float) -> None:
        st, args = self.state, self.args
        self.timings.append(dt)
        self.window.append(dt)
        if args.print_every > 0 and st["micro_step"] % args.print_every == 0:
            print(f"micro {st['micro_step']}/{self.plan['total_micro']} | opt {st['opt_step']} | loss_ce {ce:.4f}"
                  f" | loss_rl {rl:+.3f} | {dt:.2f} s/micro", flush=True)
        if not finite:
            st["nonfinite"] += 1
            if st["nonfinite"] > args.max_nonfinite:
                raise RuntimeError(f"{st['nonfinite']} non-finite losses (> --max-nonfinite {args.max_nonfinite})")
        if args.crash_at_micro_step and st["micro_step"] == args.crash_at_micro_step:
            self.log.log(event="crash_injected", micro_step=st["micro_step"])
            print(f"crash injected at micro-step {st['micro_step']}", flush=True)
            C.hard_kill()

    def _optimizer_step(self) -> tuple[dict, float]:
        """cell08.py:179-184. Returns the learning rates this step used and the pre-clip grad norm."""
        lrs = {g["group"]: g["lr"] for g in reversed(self.optimizer.param_groups)}
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["max_grad_norm"])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()  # also on scaler-skipped steps, as upstream
        self.optimizer.zero_grad(set_to_none=True)
        return lrs, float(grad_norm)

    def _after_boundary(self, epoch: int, bi: int, last_in_epoch: bool, lrs: dict, grad_norm: float) -> None:
        st = self.state
        st["opt_step"] += 1
        st["epoch"], st["batch_idx"] = (epoch + 1, 0) if last_in_epoch else (epoch, bi + 1)
        scale = float(self.scaler.get_scale())
        st["min_scale"] = scale if st["min_scale"] is None else min(st["min_scale"], scale)
        self.log.log(event="opt", opt_step=st["opt_step"], micro_step=st["micro_step"], lr_enc=lrs.get("encoder"),
                     lr_head=lrs.get("head"), scale=scale, grad_norm=grad_norm, **vram_gb(self.s.device),
                     sec_per_micro=statistics.fmean(self.window) if self.window else None)
        self.window = []
        self._maybe_eval()
        self._maybe_checkpoint()

    def _maybe_checkpoint(self) -> None:
        st, s = self.state, self.s
        micro_due = C.ckpt_due(self._prev_boundary_micro, st["micro_step"], s.ckpt_every_micro_steps)
        time_due = s.ckpt_every_min > 0 and time.monotonic() - self._last_ckpt_t >= s.ckpt_every_min * 60
        self._prev_boundary_micro = st["micro_step"]
        if not (micro_due or time_due):
            return
        t0 = time.monotonic()
        path = C.save_train_ckpt(self.run_dir / "ckpt", model=self.model, optimizer=self.optimizer,
                                 scheduler=self.scheduler, scaler=self.scaler, state=st, keep_last=s.keep_last,
                                 mirror_dir=self.args.mirror_dir)
        self._last_ckpt_t = time.monotonic()
        self.log.log(event="ckpt", opt_step=st["opt_step"], micro_step=st["micro_step"], path=str(path),
                     seconds=round(self._last_ckpt_t - t0, 3))

    def _maybe_eval(self) -> None:
        every, st, tc = self.s.eval_every_opt_steps, self.state, self.cfg["train"]
        if not every or st["opt_step"] % every:
            return
        f1 = self.evaluate()["val_macro_f1"]
        if f1 > st["best"] + tc["min_delta"]:
            st["best"], st["bad"] = f1, 0
        else:
            st["bad"] += 1
        if st["bad"] >= tc["patience"]:
            self.stop_reason = "early_stop"

    def evaluate(self, *, initial: bool = False) -> dict:
        """Val metrics plus the seconds they took (the report scales them to E1); logged as an eval event."""
        s = self.s
        if self._val_items is None:
            path = self.data_dir / "val.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"evaluation needs {path}")
            self._val_items = load_items(path, self.tok, s.max_len, s.head_max_len, labelled_only=True)[0]
        t0 = time.perf_counter()
        m = evaluate_items(self.model, self._val_items, device=s.device, fp16=s.fp16, pad_id=self.tok.pad_token_id)
        m["seconds"] = round(time.perf_counter() - t0, 3)  # evaluate_items ends on .cpu(): CUDA work is done
        self.log.log(event="eval", opt_step=self.state["opt_step"], **m, **({"initial": True} if initial else {}))
        print(f"eval @ opt {self.state['opt_step']}{' (initial)' if initial else ''}: val_ce {m['val_ce']:.4f} acc"
              f" {m['val_acc']:.3f} macro-F1 {m['val_macro_f1']:.3f} (n={m['n']}, {m['seconds']:.1f} s)", flush=True)
        return m

    def summary(self, final_eval: dict | None) -> dict:
        s, st, v, timed = self.s, self.state, vram_gb(self.s.device), self.timings[SKIP_TIMING:]
        return {"micro_steps": st["micro_step"], "opt_steps": st["opt_step"],
                "resumed_from": self.resumed_from[1] if self.resumed_from else None,
                "peak_vram_alloc_gb": v["vram_alloc_gb"], "peak_vram_reserved_gb": v["vram_reserved_gb"],
                "sec_per_micro_median": statistics.median(timed) if timed else None,
                "sec_per_micro_mean": statistics.fmean(timed) if timed else None,
                "nonfinite": st["nonfinite"], "min_scale": st["min_scale"], "initial_eval": self.initial_eval,
                "final_eval": final_eval, "n_train_items": sum(self.n_items), "truncated_items": self.truncated,
                "padding_ratio": self.pad_ratio, "device": s.device, "card": s.card, "micro_batch": s.micro_batch,
                "grad_accum": s.grad_accum, "model": s.model, "seed": s.seed, "epochs": s.epochs,
                "total_opt_steps": self.plan["total_opt"], "stop_reason": self.stop_reason}

    def finish(self, base_cfg: dict) -> dict:
        s, st = self.s, self.state
        if self.args.save_final:  # before the final eval: an eval failure (e.g. OOM) must not lose the weights
            export.save_laya_checkpoint(self.model, self.tok, base_cfg, self.run_dir / "final", max_len=s.max_len,
                                        head_max_len=s.head_max_len, model_name=f"laya-poc-{s.model}-s{s.seed}",
                                        **export.checkpoint_provenance(self.args.init, self.cfg, s.model))
        final_eval = self.evaluate() if self.args.final_eval else None
        summary = self.summary(final_eval)
        tmp = self.run_dir / "summary.json.tmp"
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        os.replace(tmp, self.run_dir / "summary.json")
        self.log.log(event="done", micro_steps=st["micro_step"], opt_steps=st["opt_step"], stop_reason=self.stop_reason)
        med = summary["sec_per_micro_median"]
        print(f"done: micro {st['micro_step']} opt {st['opt_step']} ({self.stop_reason}) | nonfinite {st['nonfinite']}"
              f" | min scale {st['min_scale']} | {med if med is None else round(med, 3)} s/micro"
              f" | {self.run_dir / 'summary.json'}", flush=True)
        return summary


def _abs_path(value: str | None, flag: str) -> Path | None:
    if value is not None and not Path(value).is_absolute():
        raise ValueError(f"{flag} must be an absolute path, got {value!r}")
    return None if value is None else Path(value)


def run(args: argparse.Namespace) -> dict:
    """Train (or resume) one run; returns the summary written to <run_dir>/summary.json."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before any CUDA use
    run_dir, data_dir = _abs_path(args.run_dir, "--run-dir"), _abs_path(args.data_dir, "--data-dir")
    _abs_path(args.mirror_dir, "--mirror-dir")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir {data_dir} does not exist")
    cfg = load_config(args.config)
    s = resolve_settings(args, cfg, data_dir)
    if (args.initial_eval or args.final_eval or s.eval_every_opt_steps) and not (data_dir / "val.jsonl").is_file():
        raise FileNotFoundError(f"evaluation needs {data_dir / 'val.jsonl'} (--initial/final-eval, eval_every)")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {"config": cfg, "args": vars(args), "settings": asdict(s)}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")
    model, tok, base_cfg = load_model(args.init, cfg, s)
    trainer = Trainer(args, cfg, s, model, tok, run_dir, data_dir)
    trainer.restore(args.mirror_dir)
    trainer.log_start()
    if args.initial_eval and trainer.resumed_from is None:  # eval opt 0; a resumed run holds other weights
        trainer.initial_eval = trainer.evaluate(initial=True)
    trainer.train()
    return trainer.finish(base_cfg)


def _count(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m laya_poc.train_single", description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True, help="absolute run directory (log.jsonl, ckpt/, final/)")
    p.add_argument("--data-dir", required=True, help="absolute data directory (train_e{k}.jsonl, val.jsonl)")
    p.add_argument("--config", default=None, help="config.yaml (default: <project root>/config.yaml)")
    p.add_argument("--model", choices=("laya", "laya_ml"), default="laya")
    p.add_argument("--seed", type=_count, default=11)
    p.add_argument("--init", default="hub", help="'hub' (pinned revision) or an absolute Laya checkpoint dir")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--card", choices=("auto", "T4", "L4", "A10", "CPU"), default="auto")
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
    p.add_argument("--mirror-dir", default=None, help="absolute directory for a copy of each checkpoint")
    return p


def _record_error(run_dir: str, exc: BaseException) -> None:
    path = Path(run_dir)
    if path.is_absolute() and path.is_dir():
        with contextlib.suppress(OSError), (path / "error.log").open("a", encoding="utf-8") as fh:
            fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:  # one line for the notebook; the traceback goes to <run_dir>/error.log
        _record_error(args.run_dir, exc)
        msg = (str(exc).strip().splitlines() or [""])[0]
        print(f"train_single: {type(exc).__name__}: {msg}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
