"""Phase 1 smoke-test report: exit criteria, the §7.13 template filled in, and a verdict.

Exit criteria: design doc §6.2 Phase 1 restated per critique §D.2 (resume compared on 20-micro-step means of
loss_ce, since CUDA is not bit-deterministic, plus the exact lr / GradScaler scale of every later opt step).
The loss drop is graded on the control run's held-out val CE in eval mode (--initial-eval at opt 0 vs
--final-eval): per-micro-batch training loss depends on which batches land in the windows (uniform-target
batches cannot fall), so the training-window means are information only.
A missing or unreadable input makes its criterion FAIL with a note; the report itself never crashes.
The information rows and the markdown are built in `smoke_render`.

    python -m laya_poc.smoke_report --run-root <runs/smoke> [--data-dir <data>] [--config yaml] --out <md>
                                    [--kill-at N --ckpt-every N --compare-steps N]   (default: config.smoke)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from .config import load_config
from .env_check import run_cli, write_json

INPUT_FILES = {"env": "env.json", "zeroshot": "zeroshot.json", "parity_laya": "parity_laya.json",
               "parity_laya_ml": "parity_laya_ml.json", "control_log": "control/log.jsonl",
               "control_summary": "control/summary.json", "resumed_log": "resumed/log.jsonl",
               "resumed_summary": "resumed/summary.json", "crash_exit": "crash_exit.json",
               "export_check": "export_check.json"}
DATA_REPORT = "data_report.json"
OPTIONAL_INPUTS = frozenset({"export_check"})
PARITY_MODELS = ("laya", "laya_ml")
TRAIN_INFO_WINDOW = 30  # micro-steps averaged at each end of the control run (information only)
LR_REL_TOL = 1e-12      # resumed vs control learning rates: the schedule is deterministic
MISSING_EVALS = "missing initial/final eval (control must run with --initial-eval --final-eval)"

CRASH = "crash run exited non-zero"
RESUME_POINT = "resumed from last checkpoint"
RESUME_LOSS = "resume vs control (loss_ce, lr, scaler scale)"
LOSS_DROP = "val_ce drop (control, initial -> final eval)"
NUMERICS = "numerics (finite loss, scaler scale >= 1)"
VRAM = "peak VRAM reserved"
CRITERIA = (*(f"parity {m}" for m in PARITY_MODELS), CRASH, RESUME_POINT, RESUME_LOSS, LOSS_DROP, NUMERICS, VRAM)


def read_events(path: Path) -> tuple[list[dict], int]:
    """JSON-lines events, skipping unparsable lines (a SIGKILL can leave half a line)."""
    events, bad = [], 0
    for line in filter(str.strip, path.read_text(encoding="utf-8").splitlines()):
        try:
            events.append(json.loads(line))
        except ValueError:
            bad += 1
    return events, bad


def load_inputs(run_root: Path, data_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """(inputs keyed like INPUT_FILES plus data_report, notes); missing/unreadable inputs are None."""
    paths = {**{k: (run_root / rel, rel) for k, rel in INPUT_FILES.items()},
             "data_report": (data_dir / DATA_REPORT, DATA_REPORT)}
    inputs: dict[str, Any] = dict.fromkeys(paths)
    notes: list[str] = []
    for key, (path, rel) in paths.items():
        if not path.exists():
            notes.append(f"{rel}: not run (optional)" if key in OPTIONAL_INPUTS else f"missing {rel}")
            continue
        try:
            inputs[key], bad = (read_events(path) if path.suffix == ".jsonl"
                                else (json.loads(path.read_text(encoding="utf-8")), 0))
        except (OSError, ValueError) as exc:
            notes.append(f"unreadable {rel}: {exc}")
            continue
        if bad:
            notes.append(f"{rel}: skipped {bad} unreadable line(s)")
    return inputs, notes


def micro_losses(events: list[dict]) -> dict[int, float]:
    """micro-step -> loss_ce (a later event for the same step wins)."""
    return {int(e["micro_step"]): float(e["loss_ce"]) for e in events if e.get("event") == "micro"}


def opt_events(events: list[dict]) -> dict[int, dict]:
    """opt_step -> its "opt" event (a later event for the same step wins)."""
    return {int(e["opt_step"]): e for e in events if e.get("event") == "opt"}


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def control_evals(events: list[Any]) -> tuple[dict | None, dict | None]:
    """(initial, final) eval events of a run log: the last --initial-eval one (opt 0, "initial": true) and the
    last other one, kept only when it comes after that initial eval and at or after the last opt step."""
    evs = [(i, e) for i, e in enumerate(events) if isinstance(e, dict) and e.get("event") in ("eval", "opt")]
    last_opt = max((e["opt_step"] for _, e in evs if e["event"] == "opt" and _is_int(e.get("opt_step"))), default=0)
    inits = [(i, e) for i, e in evs if e["event"] == "eval" and e.get("initial") is True and e.get("opt_step") == 0]
    finals = [(i, e) for i, e in evs if e["event"] == "eval" and e.get("initial") is not True]
    init, final = (inits or [None])[-1], (finals or [None])[-1]
    if final and not (_is_int(final[1].get("opt_step")) and final[1]["opt_step"] >= last_opt
                      and (init is None or final[0] > init[0])):
        final = None
    return (init[1] if init else None), (final[1] if final else None)


def last_index(events: list[dict], name: str) -> int | None:
    return next((i for i in range(len(events) - 1, -1, -1) if events[i].get("event") == name), None)


def expected_resume_step(kill_at: int, ckpt_every: int) -> int:
    """Last checkpoint written before the kill: the kill fires right after micro-step kill_at's
    backward, i.e. before that step's own boundary checkpoint (122, 40 -> 120; 120, 40 -> 80)."""
    return (kill_at - 1) // ckpt_every * ckpt_every


def _res(value: Any, threshold: str, passed: bool, note: str = "") -> dict[str, Any]:
    return {"value": value, "threshold": threshold, "passed": bool(passed), "note": note}


def _missing(inputs: dict[str, Any], *keys: str) -> str | None:
    gone = [INPUT_FILES.get(k, DATA_REPORT) for k in keys if inputs.get(k) is None]
    return "missing " + ", ".join(gone) if gone else None


def _parity(inputs: dict, exit_cfg: dict, plan: dict, *, model: str) -> dict[str, Any]:
    thr, min_agree = exit_cfg["parity_max_dp"], exit_cfg["parity_min_agree"]
    threshold = f"max|dp| <= {thr}, agree >= {min_agree}, no NaN, padded max|dp| <= {thr}"
    if gone := _missing(inputs, f"parity_{model}"):
        return _res(None, threshold, False, gone)
    p = inputs[f"parity_{model}"]
    max_dp, agree, padded = p.get("max_dp"), p.get("argmax_agree"), p.get("padded_max_dp")
    checks = [(max_dp is not None and max_dp <= thr, f"max|dp| {max_dp}"),
              (agree is not None and agree >= min_agree, f"argmax agreement {agree}"),
              (not (p.get("nan") or p.get("nan_ref") or p.get("nan_test")), "NaN probabilities"),
              (not p.get("padded_nan") and padded is not None and padded <= thr,  # as parity.passed
               f"padded vs unpadded max|dp| {padded}"),
              (p.get("device_test") == "cuda", f"test agent ran on {p.get('device_test')}: not a GPU parity check"),
              (not p.get("cpu_fallback"), "Laya answered some GPU batches on CPU (CUDA OOM fallback)")]
    problems = [msg for ok, msg in checks if not ok]
    value = f"max|dp| {fmt(max_dp)}, agree {fmt(agree, 3)}, padded {fmt(padded)}"
    detail = f"{p.get('n')} rows, test {p.get('device_test')} {p.get('dtype_test')}"
    return _res(value, threshold, not problems, "; ".join(problems) or detail)


def _crash(inputs: dict, exit_cfg: dict, plan: dict) -> dict[str, Any]:
    """A drill only when the injected kill fired: an exception (rc 1) or the OOM-killer (-9) is a real failure."""
    threshold = f"returncode != 0 (-9 or 137 expected) after crash_injected at micro-step {plan['kill_at']}"
    if gone := _missing(inputs, "crash_exit", "resumed_log"):
        return _res(None, threshold, False, gone)
    rc, log = inputs["crash_exit"].get("returncode"), inputs["resumed_log"]
    crash = last_index(log, "crash_injected")
    at = None if crash is None else log[crash].get("micro_step")
    checks = [(rc is not None and int(rc) != 0, f"returncode {rc}: crash injection did not fire"),
              (crash is not None, "no crash_injected event in resumed/log.jsonl: the run died for another reason"),
              (crash is None or at == plan["kill_at"], f"crash injected at micro-step {at}, plan {plan['kill_at']}")]
    problems = [msg for ok, msg in checks if not ok]
    return _res(rc, threshold, not problems, "; ".join(problems))


def _resume_point(inputs: dict, exit_cfg: dict, plan: dict) -> dict[str, Any]:
    expected = expected_resume_step(plan["kill_at"], plan["ckpt_every"])
    threshold = f"== {expected} (kill at {plan['kill_at']}, ckpt every {plan['ckpt_every']})"
    if gone := _missing(inputs, "resumed_log"):
        return _res(None, threshold, False, gone)
    log = inputs["resumed_log"]
    idx = last_index(log, "resumed")
    if idx is None:
        return _res(None, threshold, False, "no 'resumed' event in resumed/log.jsonl: the run restarted from scratch")
    got = log[idx].get("from_micro_step")
    return _res(got, threshold, got == expected, f"from {log[idx].get('path')}")


def _same_lr(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=LR_REL_TOL, abs_tol=0.0)
    return a == b


def _opt_mismatch(control: dict[int, dict], resumed: dict[int, dict], after: int) -> tuple[int, str | None]:
    """(opt steps compared, first mismatch) over the opt steps after `after` that both runs logged: a lost
    optimizer/scheduler/GradScaler state shows in lr_enc, lr_head or scale, not in a 20-step loss mean."""
    common = sorted(k for k in resumed if k > after and k in control)
    for k in common:
        c, r = control[k], resumed[k]
        for key in ("lr_enc", "lr_head", "scale"):
            same = _same_lr(c.get(key), r.get(key)) if key != "scale" else c.get(key) == r.get(key)
            if not same:
                return len(common), f"opt {k} {key}: resumed {r.get(key)!r} vs control {c.get(key)!r}"
    return len(common), None


def _resume_loss(inputs: dict, exit_cfg: dict, plan: dict) -> dict[str, Any]:
    tol = exit_cfg["resume_max_rel_dev"]
    threshold = (f"|mean resumed - mean control| / mean control <= {tol}; lr_enc, lr_head (rel {LR_REL_TOL:g}) "
                 "and GradScaler scale equal at every opt step after the resume point")
    if gone := _missing(inputs, "control_log", "resumed_log"):
        return _res(None, threshold, False, gone)
    log = inputs["resumed_log"]
    idx = last_index(log, "resumed")
    if idx is None:
        return _res(None, threshold, False, "no 'resumed' event in resumed/log.jsonl")
    control, resumed = micro_losses(inputs["control_log"]), micro_losses(log[idx + 1:])
    start = int(log[idx]["from_micro_step"]) + 1
    end = min(start + plan["compare_steps"] - 1, max(control, default=0))
    steps = list(range(start, end + 1))
    gaps = [s for s in steps if s not in control or s not in resumed]
    if not steps or gaps:
        return _res(None, threshold, False, f"missing micro-steps {gaps[:5] or 'all'} in window {start}..{end}")
    mc, mr = fmean(control[s] for s in steps), fmean(resumed[s] for s in steps)
    if not (math.isfinite(mc) and math.isfinite(mr)) or mc == 0:
        return _res(None, threshold, False, f"non-finite or zero means: resumed {mr}, control {mc}")
    dev = abs(mr - mc) / abs(mc)
    clipped = " (window clipped to the run length)" if len(steps) < plan["compare_steps"] else ""
    n_opt, mismatch = _opt_mismatch(opt_events(inputs["control_log"]), opt_events(log[idx + 1:]),
                                    int(log[idx].get("from_opt_step") or 0))
    opt_note = (f"optimiser state differs at {mismatch}" if mismatch else
                f"lr_enc, lr_head, scale match at {n_opt} opt steps" if n_opt else
                "no opt step after the resume point logged by both runs")
    return _res(dev, threshold, dev <= tol and n_opt > 0 and not mismatch,
                f"micro {start}..{end}: resumed {mr:.4f} vs control {mc:.4f}{clipped}; {opt_note}")


def _train_windows(events: list[dict]) -> str:
    """Information only: mean training loss_ce over the first and last TRAIN_INFO_WINDOW micro-steps."""
    steps, w = sorted(losses := micro_losses(events)), TRAIN_INFO_WINDOW
    if len(steps) < 2 * w:
        return f"info: train loss_ce windows n/a ({len(steps)} micro-steps logged)"
    head, tail = steps[:w], steps[-w:]
    return (f"info (not graded): train mean loss_ce micro {head[0]}..{head[-1]} {fmean(losses[s] for s in head):.4f}"
            f" -> {tail[0]}..{tail[-1]} {fmean(losses[s] for s in tail):.4f}")


def _loss_drop(inputs: dict, exit_cfg: dict, plan: dict) -> dict[str, Any]:
    """Held-out val CE in eval mode before vs after the control run: deterministic, no uniform-target batches."""
    min_drop = exit_cfg["min_loss_drop"]
    threshold = f"(initial val_ce - final val_ce) / initial val_ce >= {min_drop} (control, held-out val)"
    if gone := _missing(inputs, "control_log"):
        return _res(None, threshold, False, gone)
    info = _train_windows(inputs["control_log"])
    init, final = control_evals(inputs["control_log"])
    if init is None or final is None:
        return _res(None, threshold, False, f"{MISSING_EVALS}; {info}")
    a, b = float(init["val_ce"]), float(final["val_ce"])
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0:
        return _res(None, threshold, False, f"non-finite or non-positive val_ce: initial {a}, final {b}; {info}")
    drop = (a - b) / a
    note = f"val_ce {a:.4f} (opt 0, n={init.get('n')}) -> {b:.4f} (opt {final.get('opt_step')}, n={final.get('n')})"
    if (doc := exit_cfg.get("doc_loss_drop")) is not None:
        note += f"; doc §6.2 target {doc:.0%} (info): {'met' if drop >= doc else 'not met'}"
    return _res(drop, threshold, drop >= min_drop, f"{note}; {info}")


def _bad_micro(e: dict) -> bool:
    values = [e[k] for k in ("loss", "loss_ce") if k in e]
    return e.get("finite") is False or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _numerics(inputs: dict, exit_cfg: dict, plan: dict) -> dict[str, Any]:
    threshold = "0 non-finite losses; min GradScaler scale >= 1"
    if gone := _missing(inputs, "control_log", "resumed_log"):
        return _res(None, threshold, False, gone)
    events = inputs["control_log"] + inputs["resumed_log"]
    summaries = [s for s in (inputs.get("control_summary"), inputs.get("resumed_summary")) if s]
    bad_log = sum(1 for e in events if e.get("event") == "micro" and _bad_micro(e))
    nonfinite = max([bad_log] + [int(s.get("nonfinite") or 0) for s in summaries])
    scales = [float(e["scale"]) for e in events if e.get("event") == "opt" and e.get("scale") is not None]
    scales += [float(s["min_scale"]) for s in summaries if s.get("min_scale") is not None]
    min_scale = min(scales, default=None)
    on_cpu = bool(summaries) and all(s.get("device") == "cpu" for s in summaries)
    checks = [(not nonfinite, f"{nonfinite} non-finite losses"),
              (min_scale is not None or on_cpu, "no GradScaler scale recorded"),
              (min_scale is None or min_scale >= 1, f"GradScaler scale fell to {min_scale}")]
    problems = [msg for ok, msg in checks if not ok]
    return _res(f"{nonfinite} non-finite, min scale {min_scale}", threshold, not problems, "; ".join(problems))


def _vram(inputs: dict, exit_cfg: dict, plan: dict) -> dict[str, Any]:
    cap = exit_cfg["max_vram_gb"]
    threshold = f"<= {cap} GB (10^9 bytes), torch.cuda.max_memory_reserved"
    if gone := _missing(inputs, "control_summary", "resumed_summary"):
        return _res(None, threshold, False, gone)
    peaks = [inputs[k].get("peak_vram_reserved_gb") for k in ("control_summary", "resumed_summary")]
    if any(p is None for p in peaks):
        return _res(None, threshold, False, "no peak VRAM recorded (CPU run?)")
    peak = max(float(p) for p in peaks)
    return _res(peak, threshold, peak <= cap, f"control {peaks[0]}, resumed {peaks[1]}")


def evaluate_exit(inputs: dict[str, Any], exit_cfg: dict[str, float], *, kill_at: int, ckpt_every: int,
                  compare_steps: int) -> list[dict[str, Any]]:
    """One row per exit criterion (CRITERIA order): name, value, threshold, passed, note."""
    plan = {"kill_at": kill_at, "ckpt_every": ckpt_every, "compare_steps": compare_steps}
    checks: list[tuple[str, Callable[..., dict]]] = [(f"parity {m}", partial(_parity, model=m)) for m in PARITY_MODELS]
    checks += [(CRASH, _crash), (RESUME_POINT, _resume_point), (RESUME_LOSS, _resume_loss), (LOSS_DROP, _loss_drop),
               (NUMERICS, _numerics), (VRAM, _vram)]
    rows = []
    for name, check in checks:
        try:
            rows.append({"name": name, **check(inputs, exit_cfg, plan)})
        except Exception as exc:  # noqa: BLE001 - malformed input must fail the criterion, not the report
            rows.append({"name": name, **_res(None, "", False, f"could not evaluate: {type(exc).__name__}: {exc}")})
    return rows


def verdict(criteria: list[dict[str, Any]]) -> str:
    return "PASS" if criteria and all(r["passed"] for r in criteria) else "FAIL"


def fmt(x: Any, nd: int = 4) -> str:
    """A table value: floats to nd decimals, None as n/a."""
    return "n/a" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from .smoke_render import collect_info, render_markdown  # it imports this module's constants

    cfg = load_config(args.config)
    sm = cfg["smoke"]
    plan = {"kill_at": args.kill_at or sm["kill_at_micro_step"],
            "ckpt_every": args.ckpt_every or sm["ckpt_every_micro_steps"],
            "compare_steps": args.compare_steps or sm["resume_compare_steps"]}
    run_root = Path(args.run_root)
    data_dir = Path(args.data_dir) if args.data_dir else run_root.parent.parent / "data"
    inputs, notes = load_inputs(run_root, data_dir)
    criteria = evaluate_exit(inputs, sm["exit"], **plan)
    tc = cfg["train"]
    e1 = {"train_size": cfg["data"]["train_size"], "stripped_rate": cfg["augment"]["stripped_rate"],
          "epochs": tc["epochs"], "val_size": cfg["data"]["val_size"],
          "eval_every_opt_steps": tc["eval_every_opt_steps"], "ckpt_every_min": tc["ckpt_every_min"]}
    info = collect_info(inputs, criteria, e1=e1, kill_at=plan["kill_at"])
    result = {"verdict": verdict(criteria), "criteria": criteria, "info": info, "notes": notes, "plan": plan,
              "run_root": str(run_root), "data_dir": str(data_dir), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out = Path(args.out)
    write_json(out.with_suffix(".json"), result)  # also creates the directory
    out.write_text(render_markdown(criteria, info, result["verdict"], notes), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.smoke_report", description=__doc__.splitlines()[0])
    p.add_argument("--run-root", required=True, help="runs/smoke directory holding every step's output")
    p.add_argument("--data-dir", default=None, help="data dir with data_report.json (default: <run-root>/../../data)")
    p.add_argument("--config", default=None)
    p.add_argument("--out", required=True, help="markdown report path; smoke_report.json is written beside it")
    p.add_argument("--kill-at", type=int, default=None, help="override smoke.kill_at_micro_step (dry runs)")
    p.add_argument("--ckpt-every", type=int, default=None, help="override smoke.ckpt_every_micro_steps")
    p.add_argument("--compare-steps", type=int, default=None, help="override smoke.resume_compare_steps")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        res = run(args)
        failed = [r["name"] for r in res["criteria"] if not r["passed"]]
        print(f"smoke_report: {res['verdict']} ({len(res['criteria']) - len(failed)}/{len(res['criteria'])} criteria"
              f"{'; failed: ' + ', '.join(failed) if failed else ''}) -> {args.out}")
        return 0

    return run_cli("smoke_report", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
