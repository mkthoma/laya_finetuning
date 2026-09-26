"""Information rows and markdown for the smoke report (split from smoke_report to keep both modules small).

The §7.13 "Smoke test" template filled in, extra information rows, and the extrapolated E1 wall-clock.
Every row is computed under `_safe`: a malformed input gives "n/a (...)" in its cell, never a crash.
"""
from __future__ import annotations

import json
import math
from functools import partial
from statistics import fmean
from typing import Any, Callable, Sequence

from .smoke_report import (DATA_REPORT, INPUT_FILES, NUMERICS, PARITY_MODELS, RESUME_LOSS, RESUME_POINT, VRAM,
                           control_evals, fmt, micro_losses)

TEMPLATE_WINDOW = 10  # §7.13 loss row: mean loss_ce of 10 micro-steps at each end, 20 around the resume point


def e1_micro_steps(train_size: int, stripped_rate: float, epochs: int, micro_batch: int) -> int:
    """Micro-steps of the full E1 run: train items per epoch include the stripped copies."""
    return math.ceil(round(train_size * (1 + stripped_rate), 6) / micro_batch) * epochs


def _real(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _e1_opt_steps(e1: dict[str, Any], mb: int, acc: Any) -> int | None:
    """Optimiser steps of E1 at the control run's grad_accum (plus the partial last window of each epoch)."""
    if not (_real(acc) and acc >= 1 and acc == int(acc)):
        return None
    return math.ceil(e1_micro_steps(e1["train_size"], e1["stripped_rate"], 1, mb) / int(acc)) * e1["epochs"]


def _sec_per_micro(ctrl: Any) -> tuple[Any, str]:
    """(s/micro-step, statistic): the mean, which includes the 1-in-grad_accum optimiser steps; else the median."""
    if not isinstance(ctrl, dict):
        return None, "mean"
    if ctrl.get("sec_per_micro_mean") is not None:
        return ctrl["sec_per_micro_mean"], "mean"
    return ctrl.get("sec_per_micro_median"), "median"


def _eval_term(final_eval: dict | None, e1: dict[str, Any], e1_opt: int | None) -> tuple[float, str]:
    """E1 val evaluations (every eval_every_opt_steps, plus a final one) at the control final eval's s/row."""
    secs, n = (final_eval or {}).get("seconds"), (final_eval or {}).get("n")
    if not (_real(secs) and _real(n) and n > 0):
        return 0.0, "eval time excluded (no timed final eval in control/log.jsonl)"
    every, rows = e1.get("eval_every_opt_steps"), e1.get("val_size")
    if not (_real(rows) and _real(every) and every >= 0 and e1_opt is not None):
        return 0.0, "eval time excluded (no E1 eval plan)"
    n_evals = (e1_opt // int(every) if every else 0) + 1
    total = secs / n * rows * n_evals
    return total, (f"eval {total / 3600:.2f} h ({n_evals} evals × {rows} rows at {1000 * secs / n:.1f} ms/row, "
                   "from the control final eval)")


def _ckpt_term(ckpt_secs: Sequence[float], every_min: Any, run_s: float) -> tuple[float, str]:
    """E1 time-based checkpoints (one per ckpt_every_min of run time) at the smoke runs' mean save time."""
    if not ckpt_secs or not _real(every_min):
        why = "no timed ckpt events in the smoke logs" if not ckpt_secs else "no E1 ckpt_every_min"
        return 0.0, f"checkpoint time excluded ({why})"
    if every_min <= 0:
        return 0.0, "no checkpoints (ckpt_every_min 0)"
    n, mean = int(run_s // (every_min * 60)), fmean(ckpt_secs)
    return n * mean, f"checkpoints {n * mean / 3600:.2f} h ({n} checkpoints × {mean:.1f} s, one per {every_min:g} min)"


def e1_estimate(ctrl: Any, e1: dict[str, Any], *, final_eval: dict | None = None,
                ckpt_secs: Sequence[float] = ()) -> dict[str, Any]:
    """s/micro-step, micro-batch, E1 micro-steps, hours (train + eval + checkpoints) and a text; all None (the
    text says why) unless control/summary.json holds a finite s/micro-step and an integer micro_batch >= 1.
    Eval and checkpoint time are added when the logs time them; otherwise the text says they are excluded."""
    sec, stat = _sec_per_micro(ctrl)
    mb = ctrl.get("micro_batch") if isinstance(ctrl, dict) else None
    if not (_real(sec) and _real(mb) and mb >= 1 and mb == int(mb)):
        return {**dict.fromkeys(("sec", "stat", "mb", "steps", "hours", "train_hours")),
                "text": f"n/a: control/summary.json has sec_per_micro_{stat} {sec!r}, micro_batch {mb!r}"}
    steps = e1_micro_steps(e1["train_size"], e1["stripped_rate"], e1["epochs"], int(mb))
    train_s = sec * steps
    eval_s, eval_text = _eval_term(final_eval, e1, _e1_opt_steps(e1, int(mb), ctrl.get("grad_accum")))
    ckpt_s, ckpt_text = _ckpt_term(ckpt_secs, e1.get("ckpt_every_min"), train_s + eval_s)
    hours = (train_s + eval_s + ckpt_s) / 3600
    text = (f"{hours:.2f} h = train {train_s / 3600:.2f} h ({sec:.3f} s {stat} × {steps} micro-steps (ceil("
            f"{e1['train_size']} × {1 + e1['stripped_rate']:g} / {int(mb)}) × {e1['epochs']} epochs)) + {eval_text}"
            f" + {ckpt_text}")
    return {"sec": sec, "stat": stat, "mb": int(mb), "steps": steps, "hours": hours, "train_hours": train_s / 3600,
            "text": text}


def _safe(fn: Callable[[], str]) -> str:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - an info row must never break the report
        return f"n/a ({type(exc).__name__}: {exc})"


def _need(inputs: dict, key: str) -> Any:
    if inputs.get(key) is None:
        raise LookupError(f"missing {INPUT_FILES.get(key, DATA_REPORT)}")
    return inputs[key]


def _window(losses: dict[int, float], lo: int, hi: int) -> str | None:
    got = [s for s in range(lo, hi + 1) if s in losses]
    return f"{got[0]}..{got[-1]}: {fmean(losses[s] for s in got):.4f}" if got else None


def _loss_points(inputs: dict, crit: dict, kill_at: int) -> str:
    """Window means of control loss_ce (a single 8-item micro-batch swings by about ±0.5 with the batch drawn):
    micro 1..10, the 20 steps around the resume point (111..130), the last 10; then val_ce initial -> final."""
    events = _need(inputs, "control_log")
    losses, w = micro_losses(events), TEMPLATE_WINDOW
    steps = sorted(losses)
    mid = crit[RESUME_POINT]["value"] or kill_at
    spans = ((steps[0], steps[0] + w - 1), (mid - w + 1, mid + w), (steps[-1] - w + 1, steps[-1]))
    means = [m for lo, hi in spans if (m := _window(losses, lo, hi))]
    init, final = (e.get("val_ce") if e else None for e in control_evals(events))
    nan = "none" if crit[NUMERICS]["passed"] else crit[NUMERICS]["value"]
    return (f"mean loss_ce micro {' / '.join(means)}; val_ce {fmt(init)} (initial) -> {fmt(final)} (final)"
            f"; NaN/inf: {nan}")


def _template_rows(inputs: dict, crit: dict, kill_at: int, *, est: dict[str, Any]) -> dict[str, str]:
    """The design doc §7.13 "Smoke test" table, filled in."""
    env = partial(_need, inputs, "env")
    par = {m: f"{crit[f'parity {m}']['value']} ({crit[f'parity {m}']['note']})" for m in PARITY_MODELS}
    rows = {
        "GPU / CC / driver": lambda: f"{env()['gpu_name'] or 'no CUDA GPU'} / "
                                     f"{'.'.join(map(str, env()['capability'] or ['n/a']))} / {env()['driver']}",
        "Laya commit / version; transformers; torch": lambda: f"{(env()['laya_commit'] or '?')[:12]} / "
            f"{env()['laya_version']}; {env()['transformers']}; {env()['torch']} (CUDA {env()['torch_cuda']})",
        "Parity `laya`: max |Δp|, argmax agreement": lambda: par["laya"],
        "Parity `laya-multilingual`": lambda: par["laya_ml"],
        "Zero-shot on 20 records: correct / 20 (each checkpoint)": lambda: ", ".join(
            f"{m} {r['correct']}/{r['n']}" for m, r in _need(inputs, "zeroshot").items()),
        "Loss at step 0 / 120 / 250; NaN?": lambda: _loss_points(inputs, crit, kill_at),
        "Resume: step resumed from; Δloss vs control": lambda: f"micro-step {crit[RESUME_POINT]['value']}; "
            f"relative Δ {fmt(crit[RESUME_LOSS]['value'])} ({crit[RESUME_LOSS]['note']})",
        "Peak VRAM (GB); s/step at MB 8": lambda: f"{fmt(crit[VRAM]['value'], 2)} GB (10^9 B) reserved "
            f"({fmt(_need(inputs, 'control_summary').get('peak_vram_alloc_gb'), 2)} allocated); "
            f"{fmt(est['sec'], 3)} s/micro-step ({est['stat']}) at MB {est['mb']}",
        "Extrapolated E1 wall-clock (h) = s/step × steps": lambda: est["text"],
        "Optional typed-decisions 1-epoch accuracy": lambda: "not run",
    }
    return {k: _safe(v) for k, v in rows.items()}


def _check_text(name: str, check: Any) -> str:
    if not isinstance(check, dict):
        return ""
    word, note = {True: "ok", False: "FAIL"}.get(check.get("ok"), "skipped"), check.get("note")
    return f"; {name} {word}" + (f" ({note})" if word != "ok" and note else "")


def _export_text(ec: dict) -> str:
    """PASS only when the round trip passed AND the fitted T is the T the runtime applies (not clamped)."""
    calibrated = ec.get("calibration_ok", not ec.get("clamped"))
    status = "FAIL" if not ec.get("passed") else "PASS" if calibrated else "WARN"
    clamp = "" if calibrated else f" outside Laya's [0.5, 5]: the runtime applies {fmt(ec.get('T_applied'))}"
    return (f"{status}: T={fmt(ec.get('T'))}{clamp}; round-trip max|dp| {fmt(ec.get('roundtrip_max_dp'), 5)}; "
            f"ECE {fmt(ec['pre']['ece'])} -> {fmt(ec['post']['ece'])}"
            + _check_text("training eval", ec.get("train_eval")) + _check_text("budgets", ec.get("budgets")))


def _extra_rows(inputs: dict) -> dict[str, str]:
    x = partial(_need, inputs)
    rows = {
        "Description tokens": lambda: "C10 option descriptions tokenise to up to 15 tokens (the doc assumed <= 10); "
            "the ten options total 134 tokens incl. [MASK], under head_max_len - 16, so Laya never trims them",
        "Export / calibration round trip": lambda: _export_text(x("export_check")),
        "Smoke data (split sizes)": lambda: json.dumps(x("data_report")["split_sizes"], ensure_ascii=False),
        "Environment warnings": lambda: "; ".join(x("env").get("warnings") or []) or "none",
    }
    return {k: _safe(v) for k, v in rows.items()}


def _log(inputs: dict, key: str) -> list:
    return inputs[key] if isinstance(inputs.get(key), list) else []


def _timings(inputs: dict) -> tuple[dict | None, list[float]]:
    """The control run's final eval event and every timed checkpoint save of the smoke runs."""
    secs = [e["seconds"] for key in ("control_log", "resumed_log") for e in _log(inputs, key)
            if isinstance(e, dict) and e.get("event") == "ckpt" and _real(e.get("seconds"))]
    return control_evals(_log(inputs, "control_log"))[1], secs


def collect_info(inputs: dict[str, Any], criteria: list[dict[str, Any]], *, e1: dict[str, Any],
                 kill_at: int) -> dict[str, Any]:
    """The §7.13 template rows, extra information rows, and the E1 wall-clock extrapolation."""
    final_eval, ckpt_secs = _timings(inputs)
    est = e1_estimate(inputs.get("control_summary"), e1, final_eval=final_eval, ckpt_secs=ckpt_secs)
    crit = {r["name"]: r for r in criteria}
    return {"rows": _template_rows(inputs, crit, kill_at, est=est), "extra": _extra_rows(inputs),
            "e1_hours": est["hours"], "e1_train_hours": est["train_hours"], "e1_micro_steps": est["steps"],
            "sec_per_micro": est["sec"]}


def _cell(x: Any) -> str:
    return (fmt(x) if not isinstance(x, str) else x).replace("|", "\\|").replace("\n", " ")


def render_markdown(criteria: list[dict[str, Any]], info: dict[str, Any], verdict_: str, notes: list[str]) -> str:
    n_ok = sum(r["passed"] for r in criteria)
    out = ["# Laya PoC: Phase 1 smoke test report", "",
           f"Verdict: **{verdict_}** ({n_ok}/{len(criteria)} exit criteria met)", "",
           "## Smoke test (design doc §7.13)", "", "| Item | Value |", "|---|---|"]
    out += [f"| {_cell(k)} | {_cell(v)} |" for k, v in info["rows"].items()]
    out += ["", "## Exit checklist (design doc §6.2 Phase 1, restated per review §D.2)", "",
            "| # | Criterion | Value | Threshold | Result | Note |", "|---|---|---|---|---|---|"]
    out += [f"| {i} | {_cell(r['name'])} | {_cell(r['value'])} | {_cell(r['threshold'])} | "
            f"{'PASS' if r['passed'] else '**FAIL**'} | {_cell(r['note'])} |" for i, r in enumerate(criteria, 1)]
    out += ["", "## Additional information", "", "| Item | Value |", "|---|---|"]
    out += [f"| {_cell(k)} | {_cell(v)} |" for k, v in info["extra"].items()]
    if notes:
        out += ["", "## Input notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(out) + "\n"
