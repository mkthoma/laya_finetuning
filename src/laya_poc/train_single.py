"""Single-process port of upstream train_ddp.py (upstream_nb/cell08.py main()), resumable.

Kept from upstream: fp16 autocast around the forward only; the verbatim loss (loss.py);
unscale -> clip -> step -> update -> scheduler.step -> zero_grad at every accumulation boundary
and on the partial last window of each epoch; zero_grad at every epoch start; per-epoch sigma.
Deliberate deviations (config.yaml, doc §5.9): AdamW groups without decay on biases/norms,
linear warm-up + linear decay, effective batch from the config. Resume is exact: checkpoints
are written only at optimiser boundaries and every micro-step reseeds the global RNG.
Phase 2 (E1): --save-best keeps the best-val-macro-F1 weights in <run_dir>/best (consistent with the
checkpointed early-stopping state across kills, and mirrored with --mirror-dir), and summary.json carries the
gate's accounting. A run never reports success with a best/ that is not the best summary.json names.
Phase 3 (E4): --freeze-encoder trains the decision head alone (head_only.py); the resume fingerprint carries it.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from laya.common import collate_items

from . import ckpt as C
from . import export
from .config import load_config
from .head_only import freeze_encoder, trainable_parameters
from .io_utils import RunLog
from .loss import evaluate_items, forward, upstream_loss
from .run_files import micro_timing, record_error, vram_gb, write_snapshot, write_summary
from .sampler import bucketed_batches, padding_ratio
from .schedule import (Settings, abs_path, build_parser, count_items, linear_warmup_lambda, load_items, load_same_run,
                       micro_seed, param_groups, plan_steps, resolve_settings, run_fingerprint, sigma_for_epoch)


def make_scaler(enabled: bool) -> Any:
    """fp16 loss scaler; disabled (CPU) it never skips a step and get_scale() is 1.0."""
    return torch.amp.GradScaler("cuda", enabled=enabled)


class Trainer:
    """Mutable loop context: model, optimiser stack, the resumable `state` and the event log."""

    def __init__(self, args: argparse.Namespace, cfg: dict, s: Settings, loaded: tuple[Any, Any, dict],
                 run_dir: Path, data_dir: Path):
        self.args, self.cfg, self.s, (self.model, self.tok, self.base_cfg) = args, cfg, s, loaded
        self.run_dir, self.data_dir, self.lc, tc = run_dir, data_dir, cfg["train"]["loss"], cfg["train"]
        if s.head_only:
            freeze_encoder(self.model)  # before the optimiser: param_groups leaves frozen parameters out
        self.n_items = [count_items(data_dir / f"train_e{e}.jsonl") for e in range(s.epochs)]
        self.plan = plan_steps(self.n_items, s.micro_batch, s.grad_accum, s.max_micro_steps)
        self.optimizer = torch.optim.AdamW(param_groups(self.model, tc["lr_encoder"], tc["lr_head"],
                                                        tc["weight_decay"]))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, linear_warmup_lambda(self.plan["total_opt"], tc["warmup_frac"]))
        self.scaler = make_scaler(s.fp16)
        self.log = RunLog(run_dir / "log.jsonl")
        self.state = C.initial_state(run_fingerprint(s, self.n_items))
        self.resumed_from: tuple[str, int, int] | None = None
        self.timings: list[float] = []
        self.window: list[float] = []
        self.pad_ratio, self.stop_reason, self.reconciled = None, None, None
        self._epoch_cache: tuple[int, tuple] | None = None
        self._val_items: list[dict] | None = None
        self._last_ckpt_t, self._prev_boundary_micro = time.monotonic(), 0
        self._committed: int | None = None  # opt step of the newest checkpoint: the state best.prev/ guards

    def _at_cap(self) -> bool:
        return self.s.max_micro_steps is not None and self.state["micro_step"] >= self.s.max_micro_steps

    def restore(self, mirror_dir: str | None) -> None:
        path = C.latest_ckpt(self.run_dir / "ckpt", mirror_dir)
        if path is not None:
            saved = load_same_run(path, self.state["fingerprint"], model=self.model, optimizer=self.optimizer,
                                  scheduler=self.scheduler, scaler=self.scaler)
            # a Phase 1 checkpoint lacks the newer keys; an older fingerprint is carried on in today's form
            self.state = {**self.state, **saved, "fingerprint": self.state["fingerprint"]}
            self.resumed_from = (str(path), saved["micro_step"], saved["opt_step"])
            self._prev_boundary_micro, self._committed = saved["micro_step"], saved["opt_step"]
            if self.state["bad"] >= self.cfg["train"]["patience"]:
                self.stop_reason = "early_stop"  # checkpointed at the stopping boundary, killed before the end
        if self.args.save_best:
            self.reconciled = C.reconcile_best(self.run_dir, self.state["best_opt_step"], mirror_dir)
            if self.reconciled == "missing":  # training on would end without the best summary.json names
                places = [str(self.run_dir / C.BEST_DIR), C.BEST_PREV + "/"] + ([mirror_dir] if mirror_dir else [])
                raise RuntimeError(f"--save-best: the checkpoint's best (opt {self.state['best_opt_step']}) is not in "
                                   f"{' or '.join(places)}; put it back, or start a new --run-dir")

    def epoch_data(self, epoch: int) -> tuple[list[dict], list[list[int]]]:
        if self._epoch_cache and self._epoch_cache[0] == epoch:
            return self._epoch_cache[1]
        s = self.s
        items, truncated = load_items(self.data_dir / f"train_e{epoch}.jsonl", self.tok, s.max_len, s.head_max_len)
        if len(items) != self.n_items[epoch]:
            raise RuntimeError(f"train_e{epoch}.jsonl changed while training ({len(items)} != {self.n_items[epoch]})")
        lengths = [len(it["ids"]) for it in items]
        batches = bucketed_batches(lengths, s.micro_batch, s.seed * 1000 + epoch)  # same order on resume
        self.state["truncated"] = {**self.state["truncated"], epoch: truncated}  # every epoch, across resumes
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
                     max_len=s.max_len, head_max_len=s.head_max_len, save_best=self.args.save_best,
                     head_only=s.head_only, trainable_params=trainable_parameters(self.model))
        print(f"start: {s.model}{' head-only' if s.head_only else ''} seed {s.seed} on {s.device}/{s.card}"
              f" | MB {s.micro_batch} x ACC {s.grad_accum}"
              f" | {sum(self.n_items)} items | {plan['total_micro']} micro / {plan['total_opt']} opt steps"
              f" | padding {self.pad_ratio:.2f} | truncated {self.truncated}", flush=True)
        if self.resumed_from:
            path, micro, opt = self.resumed_from
            self.log.log(event="resumed", path=path, from_micro_step=micro, from_opt_step=opt, epoch=st["epoch"])
            print(f"resumed from {path} (micro-step {micro}, opt-step {opt})", flush=True)
        if self.reconciled not in (None, "kept"):  # best/ put back in line with the restored state
            self.log.log(event="best_reconciled", outcome=self.reconciled, best_opt_step=st["best_opt_step"])
            print(f"best/: {self.reconciled} (state best at opt {st['best_opt_step']})", flush=True)

    @property
    def truncated(self) -> int:
        return sum(self.state["truncated"].values())

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

    def _optimizer_step(self) -> tuple[dict, float, float]:
        """cell08.py:179-184. Returns the learning rates this step used, the pre-clip grad norm and the
        loss scale before the step (a drop across step+update means GradScaler skipped it)."""
        lrs = {g["group"]: g["lr"] for g in reversed(self.optimizer.param_groups)}
        scale_before = float(self.scaler.get_scale())
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["max_grad_norm"])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()  # also on scaler-skipped steps, as upstream
        self.optimizer.zero_grad(set_to_none=True)
        return lrs, float(grad_norm), scale_before

    def _after_boundary(self, epoch: int, bi: int, last_in_epoch: bool, lrs: dict, grad_norm: float,
                        scale_before: float) -> None:
        st = self.state
        st["opt_step"] += 1
        st["epoch"], st["batch_idx"] = (epoch + 1, 0) if last_in_epoch else (epoch, bi + 1)
        scale = float(self.scaler.get_scale())
        skipped, applied = C.opt_step_outcome(self.scaler.is_enabled(), scale_before, scale, grad_norm)
        st["skipped"], st["nonfinite_applied"] = st["skipped"] + skipped, st["nonfinite_applied"] + applied
        st["min_scale"] = scale if st["min_scale"] is None else min(st["min_scale"], scale)
        self.log.log(event="opt", opt_step=st["opt_step"], micro_step=st["micro_step"], lr_enc=lrs.get("encoder"),
                     lr_head=lrs.get("head"), scale=scale, grad_norm=grad_norm, skipped=skipped,
                     **vram_gb(self.s.device), sec_per_micro=statistics.fmean(self.window) if self.window else None)
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
                                 scheduler=self.scheduler, scaler=self.scaler, state=st, keep_last=s.keep_last)
        if self.args.mirror_dir is not None:
            self._mirror(path)
        self._last_ckpt_t, self._committed = time.monotonic(), st["opt_step"]
        if self.args.save_best:
            C.commit_best(self.run_dir)  # this checkpoint's state records the current best/
        self.log.log(event="ckpt", opt_step=st["opt_step"], micro_step=st["micro_step"], path=str(path),
                     seconds=round(self._last_ckpt_t - t0, 3))

    def _mirror(self, path: Path) -> None:
        """Mirror a checkpoint (after the local write). With --save-best the best it names is mirrored first,
        so the mirror's newest checkpoint always finds its best there (ckpt.mirror_best)."""
        mirror, best = self.args.mirror_dir, self.state["best_opt_step"]
        if self.args.save_best:
            C.mirror_best(self.run_dir, mirror, best)
        C.mirror_ckpt(path, mirror, self.s.keep_last)
        if self.args.save_best:
            C.prune_mirror_best(mirror, best)

    def _maybe_eval(self) -> None:
        every, st = self.s.eval_every_opt_steps, self.state
        if not every or st["opt_step"] % every:
            return
        self._track(self.evaluate())
        if st["bad"] >= self.cfg["train"]["patience"]:
            self.stop_reason = "early_stop"

    def _track(self, m: dict) -> None:
        """Early-stopping bookkeeping of a periodic or final eval; exports a new best (--save-best)."""
        st = self.state
        st["evals"], st["last_eval"] = st["evals"] + 1, (st["opt_step"], m)
        if not m["val_macro_f1"] > st["best"] + self.cfg["train"]["min_delta"]:
            st["bad"] += 1
            return
        st.update(best=m["val_macro_f1"], bad=0, best_opt_step=st["opt_step"], best_eval=m)
        if self.args.save_best:
            self._export_best(m)

    def _export_best(self, m: dict | None) -> None:
        st, t0 = self.state, time.monotonic()
        C.stage_best(self.run_dir, self._committed)
        marker = C.best_marker(opt_step=st["opt_step"], micro_step=st["micro_step"], model=self.s.model,
                               seed=self.s.seed, eval_result=m)
        out = self._export(C.BEST_DIR, {C.BEST_INFO: marker})
        f1, dt = marker["val_macro_f1"], round(time.monotonic() - t0, 3)
        self.log.log(event="best", opt_step=st["opt_step"], val_macro_f1=f1, seconds=dt)
        print(f"best @ opt {st['opt_step']}: val macro-F1 {f1} -> {out} ({dt:.1f} s)", flush=True)

    def _export(self, name: str, extra: dict | None = None) -> Path:
        s = self.s
        return export.save_laya_checkpoint(
            self.model, self.tok, self.base_cfg, self.run_dir / name, max_len=s.max_len,
            head_max_len=s.head_max_len, model_name=f"laya-poc-{s.model}-s{s.seed}", extra_json=extra,
            **export.checkpoint_provenance(self.args.init, self.cfg, s.model))

    def evaluate(self, *, initial: bool = False) -> dict:
        """Val metrics plus the seconds they took (the report scales them to E1); logged as an eval event."""
        s = self.s
        if self._val_items is None:
            path = self.data_dir / "val.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"evaluation needs {path}")
            self._val_items = load_items(path, self.tok, s.max_len, s.head_max_len, labelled_only=True)[0]
        t0 = time.perf_counter()
        m = {**self._val_metrics(self._val_items)}
        m["seconds"] = round(time.perf_counter() - t0, 3)  # evaluate_items ends on .cpu(): CUDA work is done
        self.log.log(event="eval", opt_step=self.state["opt_step"], **m, **({"initial": True} if initial else {}))
        print(f"eval @ opt {self.state['opt_step']}{' (initial)' if initial else ''}: val_ce {m['val_ce']:.4f} acc"
              f" {m['val_acc']:.3f} macro-F1 {m['val_macro_f1']:.3f} (n={m['n']}, {m['seconds']:.1f} s)", flush=True)
        return m

    def _val_metrics(self, items: list[dict]) -> dict:
        s = self.s
        return evaluate_items(self.model, items, device=s.device, fp16=s.fp16, pad_id=self.tok.pad_token_id,
                              batch_size=s.eval_batch_size)

    def summary(self, final_eval: dict | None) -> dict:
        s, st, v = self.s, self.state, vram_gb(self.s.device)
        return {"micro_steps": st["micro_step"], "opt_steps": st["opt_step"],
                "resumed_from": self.resumed_from[1] if self.resumed_from else None,
                "peak_vram_alloc_gb": v["vram_alloc_gb"], "peak_vram_reserved_gb": v["vram_reserved_gb"],
                **micro_timing(self.timings), "nonfinite": st["nonfinite"], "min_scale": st["min_scale"],
                "initial_eval": st["initial_eval"], "final_eval": final_eval, "n_train_items": sum(self.n_items),
                "truncated_items": self.truncated, "padding_ratio": self.pad_ratio, "device": s.device, "card": s.card,
                "micro_batch": s.micro_batch,
                "grad_accum": s.grad_accum, "model": s.model, "seed": s.seed, "epochs": s.epochs,
                "head_only": s.head_only, "total_opt_steps": self.plan["total_opt"], "stop_reason": self.stop_reason,
                "opt_steps_skipped": st["skipped"], "nonfinite_grad_applied": st["nonfinite_applied"],
                "evals": st["evals"], "best_opt_step": st["best_opt_step"], "best_eval": st["best_eval"]}

    def finish(self) -> dict:
        st = self.state
        if self.args.save_final:  # before the final eval: an eval failure (e.g. OOM) must not lose the weights
            self._export("final")
        final_eval = self._final_eval() if self.args.final_eval else None
        if self.args.save_best and st["best_opt_step"] is None:
            msg = "no val eval scored these weights: exporting the final weights as best/"
            self.log.log(event="warning", message=msg)
            print(f"WARNING: {msg}", flush=True)
            self._export_best(None)
            st["best_opt_step"] = st["opt_step"]
        if self.args.save_best:
            self._check_best()
        summary = self.summary(final_eval)
        write_summary(self.run_dir, summary)
        if self.args.save_best:
            C.commit_best(self.run_dir)  # the run is complete: summary.json now records best/
        self.log.log(event="done", micro_steps=st["micro_step"], opt_steps=st["opt_step"], stop_reason=self.stop_reason)
        med = summary["sec_per_micro_median"]
        print(f"done: micro {st['micro_step']} opt {st['opt_step']} ({self.stop_reason}) | nonfinite {st['nonfinite']}"
              f" | min scale {st['min_scale']} | {med if med is None else round(med, 3)} s/micro"
              f" | {self.run_dir / 'summary.json'}", flush=True)
        return summary

    def _check_best(self) -> None:
        """Before summary.json: best/ must hold the best it will name, or the run must not report success."""
        on_disk, best = C.best_step(self.run_dir / C.BEST_DIR), self.state["best_opt_step"]
        if on_disk != best:
            raise RuntimeError(f"--save-best: {self.run_dir / C.BEST_DIR} holds opt {on_disk}, not the run's best "
                               f"(opt {best}); summary.json not written")

    def _final_eval(self) -> dict:
        """--final-eval; a periodic eval of these very weights (e.g. at an early stop) is reused."""
        last = self.state["last_eval"]
        if last is not None and last[0] == self.state["opt_step"]:
            return last[1]
        m = self.evaluate()
        self._track(m)
        return m


def run(args: argparse.Namespace) -> dict:
    """Train (or resume) one run; returns the summary written to <run_dir>/summary.json."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before any CUDA use
    run_dir, data_dir = abs_path(args.run_dir, "--run-dir"), abs_path(args.data_dir, "--data-dir")
    abs_path(args.mirror_dir, "--mirror-dir")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir {data_dir} does not exist")
    cfg = load_config(args.config)
    s = resolve_settings(args, cfg, data_dir)
    if (args.initial_eval or args.final_eval or s.eval_every_opt_steps) and not (data_dir / "val.jsonl").is_file():
        raise FileNotFoundError(f"evaluation needs {data_dir / 'val.jsonl'} (--initial/final-eval, eval_every)")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot(run_dir, cfg, args, s)
    trainer = Trainer(args, cfg, s, export.load_for_training(args.init, cfg, s), run_dir, data_dir)
    trainer.restore(args.mirror_dir)
    trainer.log_start()
    if args.initial_eval and trainer.resumed_from is None:  # eval opt 0; a resumed run holds other weights
        trainer.state["initial_eval"] = trainer.evaluate(initial=True)  # in the state: survives a resume
    trainer.train()
    return trainer.finish()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:  # one line for the notebook; the traceback goes to <run_dir>/error.log
        record_error(args.run_dir, exc)
        msg = (str(exc).strip().splitlines() or [""])[0]
        print(f"train_single: {type(exc).__name__}: {msg}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
