"""Phase 2 gate on the E1 run (design doc §6.2 "Gate (end of Day 3)"): PASS/FAIL per criterion and a report.

Thresholds: config `gate` (`thresholds()`). Accuracy uses the headline val macro-F1: 10-class, or 9-class when
data_report says Event is too small (§5.2). Skipped fp16 steps are expected and only reported. A missing or
malformed input fails its criteria with a note; the report never crashes.

    python -m laya_poc.gate --run-root <runs/e1> --data-dir <data> --out <gate_report.md> [--config yaml]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .config import load_config, project_root
from .env_check import run_cli, write_json
from .smoke_report import fmt, opt_events, read_events, verdict

INPUT_FILES = {"summary": "train/summary.json", "log": "train/log.jsonl", "crash_exit": "crash_exit.json",
               "export_check": "export_check.json", "eval_e1": "eval_e1_val.json",
               "eval_zeroshot": "eval_zeroshot_val.json", "eval_zeroshot_ml": "eval_zeroshot_ml_val.json",
               "baselines": "baselines.json", "bench": "bench_cpu.json"}
DATA_REPORT = "data_report.json"
NAMES = {**INPUT_FILES, "data_report": DATA_REPORT}
OPTIONAL_INPUTS = frozenset({"crash_exit", "eval_zeroshot_ml"})
STOP_REASONS = ("epochs", "early_stop", "max_micro_steps")
INIT_SCALE = 65536.0  # torch.amp.GradScaler's default init_scale (2**16): the reference for opt step 1
EPS = 1e-9            # float slack on the >= thresholds (0.60 + 0.10 must equal 0.70)

END_TO_END = "end-to-end (train -> T fit -> val eval -> CPU bench; resume)"
NUMERICS = "numerics (finite loss and gradients, GradScaler scale)"
BEATS_TRIVIAL = "beats trivial (zero-shot, majority)"
NEAR_TFIDF = "near TF-IDF+LR"
CRITERIA = (END_TO_END, NUMERICS, BEATS_TRIVIAL, NEAR_TFIDF)
STOP = "more than {pts} points below TF-IDF+LR: stop and debug before any spend (design doc §6.2)"
DEBUG_CHECKLIST = (  # design doc §6.2, with where to look
    "Leakage asserts: data_report.json `leakage` and `warnings` (build_data asserts on every build)",
    "Label mapping: labels.L1_TO_KEY vs Appendix B; data_report label_distributions vs row_label_distributions",
    "Question-key order: one labels.question(scheme) object (same criteria order) in train, val and evaluate",
    "Truncation counts: summary.json truncated_items (also the `start` event); data_report stats compressed/rejected",
    "Learning rates: config train.lr_encoder / lr_head vs lr_enc / lr_head in the `opt` events (warm-up, decay)",
    "Loss sign: loss_ce falls over the `micro` events and val_ce over the `eval` events; loss_ce is never negative")


class Missing(LookupError):
    """A missing input or metric: the criterion fails with this message as its note."""


def load_inputs(run_root: Path, data_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """(inputs keyed like INPUT_FILES plus data_report, notes); missing or unreadable inputs are None."""
    paths = {**{k: run_root / rel for k, rel in INPUT_FILES.items()}, "data_report": data_dir / DATA_REPORT}
    inputs: dict[str, Any] = dict.fromkeys(paths)
    notes: list[str] = []
    for key, path in paths.items():
        try:
            inputs[key], bad = (read_events(path) if path.suffix == ".jsonl"
                                else (json.loads(path.read_text(encoding="utf-8")), 0))
        except FileNotFoundError:
            notes.append(f"{NAMES[key]}: not run (optional)" if key in OPTIONAL_INPUTS else f"missing {NAMES[key]}")
        except (OSError, ValueError) as exc:
            notes.append(f"unreadable {NAMES[key]}: {exc}")
        else:
            notes += [f"{NAMES[key]}: skipped {bad} unreadable line(s)"] if bad else []
    return inputs, notes


def _real(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _need(inputs: dict[str, Any], *keys: str) -> Any:
    """inputs[keys[0]]; Missing naming every absent (or non-JSON-object) input."""
    if gone := [NAMES[k] for k in keys if not isinstance(inputs.get(k), (dict, list))]:
        raise Missing("missing " + ", ".join(gone))
    return inputs[keys[0]]


def _metric(inputs: dict[str, Any], key: str, *path: str) -> float:
    """A finite number at inputs[key][path...]; Missing naming the file and path otherwise."""
    node = inputs.get(key)
    for p in path:
        if not isinstance(node, dict) or p not in node:
            raise Missing(f"{NAMES[key]} has no {'.'.join(path)}")
        node = node[p]
    if not _real(node):
        raise Missing(f"{NAMES[key]} {'.'.join(path)} is {node!r}")
    return float(node)


def _baseline(inputs: dict[str, Any], scheme: str, arm: str, metric: str, split: str = "val") -> float:
    """B1 reports are {split: report}; B3 {split: {pre, post}}: take post when present."""
    node = ((inputs.get("baselines") or {}).get("schemes") or {}).get(scheme, {}).get(arm, {}).get(split)
    path = ("schemes", scheme, arm, split, *(("post",) if isinstance(node, dict) and "post" in node else ()), metric)
    return _metric(inputs, "baselines", *path)


def headline(inputs: dict[str, Any], cfg: dict) -> tuple[str, str]:
    """(metric key, description): 9-class macro-F1 when the ID pool has too few Event places (§5.2)."""
    dr = inputs.get("data_report") or {}
    pool, floor = dr.get("event_id_pool"), cfg["labels"].get("event_min_id_pool")
    n = dr.get("headline_classes") or ((9 if pool < floor else 10) if _real(pool) and _real(floor) else None)
    if n not in (9, 10):
        raise Missing(f"{DATA_REPORT} has no headline_classes (10 or 9) and no event_id_pool")
    rule = f"Event: {pool} labelled ID places, {'<' if n == 9 else '>='} labels.event_min_id_pool {floor}"
    return ("macro_f1_9", f"9-class macro-F1 (event excluded; {rule})") if n == 9 else \
        ("macro_f1", f"10-class macro-F1 ({rule})")


def _pts(x: float, sign: bool = False) -> str:
    return f"{100 * x:{'+' if sign else ''}.1f}"


def thresholds(g: dict) -> dict[str, str]:
    """The pass condition of each criterion, from config `gate` (points are macro-F1 x 100)."""
    return {END_TO_END: f"stop_reason in {STOP_REASONS}; export_check passed; val eval; >= 1 bench thread; "
                        ">= 1 resume",
            NUMERICS: f"0 non-finite losses; nonfinite_grad_applied == 0; min GradScaler scale >= {g['min_scale']}",
            BEATS_TRIVIAL: f"E1 >= zero-shot laya {_pts(g['min_over_zero_shot'], True)} pts and >= B1 majority "
                           f"{_pts(g['min_over_majority'], True)} pts (val)",
            NEAR_TFIDF: f"E1 >= TF-IDF+LR (B3) - {_pts(g['max_below_tfidf'])} pts (val)"}


def _end_to_end(inputs: dict, g: dict, cfg: dict) -> tuple[Any, bool, str]:
    s, ec, ev, log = (inputs.get(k) or {} for k in ("summary", "export_check", "eval_e1", "log"))
    threads = [r["threads"] for r in bench_rows(inputs.get("bench"))]
    resumes = [e for e in log if e.get("event") == "resumed"]
    crash = next((e for e in log if e.get("event") == "crash_injected"), None)
    rc = (inputs.get("crash_exit") or {}).get("returncode")
    how = f"forced crash at micro-step {crash.get('micro_step')}, exit {rc}" if crash else "observed, no injected crash"
    stages = [("summary", s.get("stop_reason") in STOP_REASONS, f"training stop_reason {s.get('stop_reason')!r}"),
              ("export_check", ec.get("passed") is True,
               f"temperature fit: export_check passed={ec.get('passed')}, T {fmt(ec.get('T'))}"),
              ("eval_e1", isinstance(ev.get("post"), dict), f"val eval n={ev.get('n')}"),
              ("bench", bool(threads), f"CPU bench thread settings {threads}"),
              ("log", bool(resumes), f"{len(resumes)} resume(s) in train/log.jsonl ({how})")]
    texts = [f"missing {NAMES[k]}" if inputs.get(k) is None else text for k, _, text in stages]
    problems = [text for (_, ok, _), text in zip(stages, texts) if not ok]
    return f"{sum(ok for _, ok, _ in stages)}/{len(stages)} stages", not problems, "; ".join(problems or texts)


def scaler_steps(events: list[dict]) -> tuple[int, int]:
    """(opt steps GradScaler skipped, opt steps that applied a non-finite gradient) from the log alone. With fp16,
    GradScaler skips a step with inf/NaN gradients and halves its scale, so the scale logged after update() drops
    below the previous step's (step 1: below the init scale). Without it (CPU) every non-finite grad is applied."""
    fp16 = any(e.get("fp16") for e in events if e.get("event") == "start")
    opts = opt_events(events)
    skipped = applied = 0
    prev = INIT_SCALE
    for k in sorted(opts):
        scale, finite = opts[k].get("scale"), _real(opts[k].get("grad_norm"))
        dropped = fp16 and _real(scale) and scale < prev
        skipped, applied = skipped + bool(dropped), applied + (not finite and not dropped)
        prev = scale if _real(scale) else prev
    return skipped, applied


def _numerics(inputs: dict, g: dict, cfg: dict) -> tuple[Any, bool, str]:
    s, events = _need(inputs, "summary", "log"), inputs["log"]
    bad = [e for e in events if e.get("event") == "micro"
           and (e.get("finite") is False or not all(_real(e[k]) for k in ("loss", "loss_ce") if k in e))]
    losses = max(len(bad), int(s.get("nonfinite") or 0))
    skipped_log, applied_log = scaler_steps(events)
    applied = max(applied_log, int(s.get("nonfinite_grad_applied") or 0))
    scales = [float(e["scale"]) for e in opt_events(events).values() if _real(e.get("scale"))]
    low = min(scales + ([float(s["min_scale"])] if _real(s.get("min_scale")) else []), default=None)
    checks = [(losses == 0, f"{losses} non-finite loss(es)"),
              (applied == 0, f"{applied} optimiser step(s) applied a non-finite gradient"),
              (low is not None or s.get("device") == "cpu", "no GradScaler scale recorded"),
              (low is None or low >= g["min_scale"], f"GradScaler scale fell to {low}")]
    problems = [msg for ok, msg in checks if not ok]
    info = f"{s.get('opt_steps_skipped', skipped_log)} fp16 step(s) skipped by GradScaler (expected, not a failure)"
    value = f"{losses} non-finite losses, {applied} non-finite grads applied, min scale {low}"
    return value, not problems, "; ".join([*problems, info])


def _accuracy_inputs(inputs: dict, cfg: dict, *keys: str) -> tuple[str, str, float]:
    """(metric key, headline text, E1 post-T val value)."""
    _need(inputs, "eval_e1", "data_report", *keys)
    key, text = headline(inputs, cfg)
    return key, text, _metric(inputs, "eval_e1", "post", key)


def _beats_trivial(inputs: dict, g: dict, cfg: dict) -> tuple[Any, bool, str]:
    key, text, e1 = _accuracy_inputs(inputs, cfg, "eval_zeroshot", "baselines")
    zs = _metric(inputs, "eval_zeroshot", "post", key)
    maj = _baseline(inputs, cfg["labels"]["scheme"], "b1_majority", key)
    ok = e1 + EPS >= zs + g["min_over_zero_shot"] and e1 + EPS >= maj + g["min_over_majority"]
    return e1, ok, (f"{text}: E1 {e1:.4f}; zero-shot {zs:.4f} ({_pts(e1 - zs, True)} pts); majority {maj:.4f} "
                    f"({_pts(e1 - maj, True)} pts)")


def _near_tfidf(inputs: dict, g: dict, cfg: dict) -> tuple[Any, bool, str]:
    key, text, e1 = _accuracy_inputs(inputs, cfg, "baselines")
    b3 = _baseline(inputs, cfg["labels"]["scheme"], "b3_tfidf_lr", key)
    ok = e1 + EPS >= b3 - g["max_below_tfidf"]
    note = f"{text}: E1 {e1:.4f} vs TF-IDF+LR {b3:.4f} ({_pts(e1 - b3, True)} pts)"
    return e1, ok, note if ok else f"{note}; {STOP.format(pts=_pts(g['max_below_tfidf']))}"


def evaluate_gate(inputs: dict[str, Any], cfg: dict) -> list[dict[str, Any]]:
    """One row per criterion (CRITERIA order); each check returns (value, passed, note) or raises Missing."""
    rows, limits = [], thresholds(cfg["gate"])
    for name, check in zip(CRITERIA, (_end_to_end, _numerics, _beats_trivial, _near_tfidf)):
        try:
            value, passed, note = check(inputs, cfg["gate"], cfg)
        except Missing as exc:
            value, passed, note = None, False, str(exc)
        except Exception as exc:  # noqa: BLE001 - malformed input must fail the criterion, not the report
            value, passed, note = None, False, f"could not evaluate: {type(exc).__name__}: {exc}"
        rows.append({"name": name, "value": value, "threshold": limits[name], "passed": bool(passed), "note": note})
    return rows


_BENCH_KEYS = {"threads": ("threads", "n_threads"), "cold_s": ("cold_s", "cold_start_s", "load_s"),
               "p50_ms": ("p50_ms",), "p95_ms": ("p95_ms",), "rps": ("batch_rps", "rps", "records_per_s"),
               "peak_rss_gb": ("peak_rss_gb", "rss_gb")}


def bench_rows(obj: Any) -> list[dict[str, Any]]:
    """One row per thread setting of bench_cpu.json: a list of results, or a dict holding one under
    results/runs/settings/threads, as a list or keyed by the thread count."""
    candidates = [obj] if isinstance(obj, list) else \
        [obj[k] for k in ("results", "runs", "settings", "threads") if isinstance(obj, dict) and k in obj]
    for items in candidates:
        if isinstance(items, dict):
            items = [{"threads": k, **v} for k, v in items.items() if isinstance(v, dict)]
        rows = [{name: next((it[a] for a in aliases if a in it), None) for name, aliases in _BENCH_KEYS.items()}
                for it in (items if isinstance(items, list) else []) if isinstance(it, dict)]
        if rows := [{**r, "threads": int(r["threads"])} for r in rows
                    if _real(r["threads"]) or str(r["threads"]).isdigit()]:
            return rows
    return []


def wall_clock(events: list[dict]) -> tuple[float, float, int]:
    """(seconds of work summed over the training processes, span first start -> last event, processes).
    A process runs from its `start` event to its last event (the crash, or `done`)."""
    segments: list[list[float]] = []
    for e in (e for e in events if _real(e.get("t"))):
        if e.get("event") == "start":
            segments.append([e["t"], e["t"]])
        elif segments:
            segments[-1][1] = e["t"]
    if not segments:
        raise LookupError("no timed start event in train/log.jsonl")
    return sum(b - a for a, b in segments), segments[-1][1] - segments[0][0], len(segments)


def _wall(inputs: dict) -> str:
    work, span, n = wall_clock(_need(inputs, "log"))
    return f"{work / 3600:.2f} h of work over {n} processes ({span / 3600:.2f} h from the first start to the end)"


def _safe(fn: Callable[[], str]) -> str:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - an info row must never break the report
        return f"n/a ({type(exc).__name__}: {exc})"


def _speed(inputs: dict) -> str:
    s, log = _need(inputs, "summary"), _need(inputs, "log")
    vram = [e.get("vram_reserved_gb") for e in log if e.get("event") == "opt"] + [s.get("peak_vram_reserved_gb")]
    return (f"{fmt(s.get('sec_per_micro_mean'), 3)} s mean ({fmt(s.get('sec_per_micro_median'), 3)} median) at MB "
            f"{s.get('micro_batch', '?')}; peak VRAM {fmt(max(filter(_real, vram), default=None), 2)} GB reserved")


def _f1s(node: Any) -> str:
    post = node.get("post", node) if isinstance(node, dict) else {}
    return (f"macro-F1 {fmt(post.get('macro_f1'))} (9-class {fmt(post.get('macro_f1_9'))}), acc {fmt(post.get('acc'))}"
            f", ECE {fmt(post.get('ece'))}")


def _trap_row(data_dir: Path) -> str:
    """data/trap_candidates.csv (FSQ rows: gitignored, never archived) and the merged data/trap.jsonl, if any."""
    path, merged = data_dir / "trap_candidates.csv", data_dir / "trap.jsonl"
    with path.open(encoding="utf-8", newline="") as fh:
        n = max(0, sum(1 for _ in csv.reader(fh)) - 1)  # csv: names may hold quoted line breaks
    lines = merged.read_text(encoding="utf-8").splitlines() if merged.exists() else None
    status = (f"{sum(1 for x in lines if x.strip())} trap items merged (data/trap.jsonl)" if lines is not None else
              "annotation pending: data/trap_candidates.csv -> 2 annotators -> python -m laya_poc.traps merge")
    return f"{n} candidates; {status}"


def _frozen_row(dr: dict, cfg: dict) -> str:
    fp, rel = dr.get("fingerprint_sha256"), cfg["data"].get("frozen_manifest")
    if not rel:
        return f"{fp or 'n/a'}; not frozen (config data.frozen_manifest is null)"
    want = json.loads((project_root() / rel).read_text(encoding="utf-8")).get("fingerprint_sha256")  # abs rel wins
    return f"{fp}; {'matches the frozen manifest' if fp and fp == want else 'DIFFERS from the frozen manifest'} {rel}"


def collect_info(inputs: dict[str, Any], cfg: dict, data_dir: Path) -> dict[str, Any]:
    """Information rows (no pass/fail) and the CPU benchmark table."""
    s, ev, ec = (lambda k=k: _need(inputs, k) for k in ("summary", "eval_e1", "export_check"))
    b3 = lambda: _need(inputs, "baselines")["schemes"][cfg["labels"]["scheme"]]["b3_tfidf_lr"]  # noqa: E731
    rows = {
        "Headline metric": lambda: headline(inputs, cfg)[1],
        "E1 val (post-T)": lambda: f"{_f1s(ev())}; acc@80 {fmt(ev()['post'].get('acc@80'))}, acc@90 "
            f"{fmt(ev()['post'].get('acc@90'))}; n {ev().get('n')}, unlabelled {ev().get('n_unlabelled')}, "
            f"abstained without evidence {ev().get('n_abstained_no_evidence')}",
        "E1 wall-clock": lambda: _wall(inputs),
        "Seconds per micro-step; peak VRAM": lambda: _speed(inputs),
        "Epochs / stop": lambda: f"{s().get('stop_reason')}; {s().get('epochs')} planned epochs; "
            f"{s().get('micro_steps')} micro / {s().get('opt_steps')} opt steps; {s().get('evals')} evals; "
            f"{s().get('truncated_items')} truncated train items; padding ratio {fmt(s().get('padding_ratio'), 2)}",
        "Best checkpoint": lambda: f"opt step {s().get('best_opt_step')}: val macro-F1 "
                                   f"{fmt((s().get('best_eval') or {}).get('val_macro_f1'))} (training eval)",
        "Temperature; ECE pre -> post (val)": lambda: f"T {fmt(ev().get('temperature'))} (export_check T "
            f"{fmt(ec().get('T'))}, clamped {ec().get('clamped')}); ECE {fmt(ev()['pre'].get('ece'))} -> "
            f"{fmt(ev()['post'].get('ece'))}",
        "TF-IDF+LR (B3)": lambda: f"C {b3().get('C')}, T {fmt(b3().get('T'))}; val {_f1s(b3().get('val'))}",
        "Zero-shot laya (val)": lambda: _f1s(_need(inputs, "eval_zeroshot")),
        "Zero-shot laya-multilingual (val)": lambda: _f1s(_need(inputs, "eval_zeroshot_ml"))
        if inputs.get("eval_zeroshot_ml") is not None else "not run",
        "Data fingerprint": lambda: _frozen_row(_need(inputs, "data_report"), cfg),
        "Trap candidates": lambda: _trap_row(data_dir),
    }
    budget = cfg["cpu_budget"]
    bench = [{**r, "within_budget": bool(_real(r["p95_ms"]) and _real(r["rps"]) and r["p95_ms"] <= budget["p95_ms"]
                                         and r["rps"] >= budget["min_rps"])} for r in bench_rows(inputs.get("bench"))]
    return {"rows": {k: _safe(v) for k, v in rows.items()}, "bench": bench}


def _table(header: list[str], rows: list[list[Any]], nd: tuple[int, ...] = ()) -> list[str]:
    def cell(v: Any, d: int) -> str:
        return (fmt(v, d) if not isinstance(v, str) else v).replace("|", "\\|").replace("\n", " ")
    nd = nd or (4,) * len(header)
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(cell(v, d) for v, d in zip(r, nd)) + " |" for r in rows)]


def render_markdown(result: dict[str, Any], cfg: dict) -> str:
    crit, info, b = result["criteria"], result["info"], cfg["cpu_budget"]
    out = ["# Laya PoC: Phase 2 gate (E1)", "",
           f"Verdict: **{result['verdict']}** ({sum(r['passed'] for r in crit)}/{len(crit)} gate criteria met)", "",
           "## Gate (design doc §6.2)", ""]
    out += _table(["#", "Criterion", "Value", "Threshold", "Result", "Note"],
                  [[str(i), r["name"], r["value"], r["threshold"], "PASS" if r["passed"] else "**FAIL**", r["note"]]
                   for i, r in enumerate(crit, 1)])
    if result["debug_checklist"]:
        out += ["", "## Debug checklist (the gate failed: stop and debug before any spend, design doc §6.2)", ""]
        out += [f"- [ ] {item}" for item in result["debug_checklist"]]
    out += ["", "## E1 run (information)", ""] + _table(["Item", "Value"], [[k, v] for k, v in info["rows"].items()])
    out += ["", f"## CPU benchmark (info: p95 <= {b['p95_ms']} ms, >= {b['min_rps']} rec/s is judged in Phase 5)", ""]
    out += _table(["Threads", "Cold start (s)", "p50 (ms)", "p95 (ms)", "Batch rec/s", "Peak RSS (GB)", "In budget"],
                  [[str(r["threads"]), r["cold_s"], r["p50_ms"], r["p95_ms"], r["rps"], r["peak_rss_gb"],
                    "yes" if r["within_budget"] else "no"] for r in info["bench"]], (0, 1, 1, 1, 1, 2, 0))
    if result["notes"]:
        out += ["", "## Input notes", ""] + [f"- {n}" for n in result["notes"]]
    return "\n".join(out) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    run_root, data_dir = Path(args.run_root), Path(args.data_dir)
    inputs, notes = load_inputs(run_root, data_dir)
    criteria = evaluate_gate(inputs, cfg)
    result = {"verdict": verdict(criteria), "criteria": criteria, "info": collect_info(inputs, cfg, data_dir),
              "notes": notes, "thresholds": cfg["gate"], "run_root": str(run_root), "data_dir": str(data_dir),
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    result["debug_checklist"] = list(DEBUG_CHECKLIST) if result["verdict"] != "PASS" else []
    out = Path(args.out)
    write_json(out.with_suffix(".json"), result)  # also creates the directory
    out.write_text(render_markdown(result, cfg), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.gate", description=__doc__.splitlines()[0])
    p.add_argument("--run-root", required=True, help="runs/e1: train/, crash_exit.json, eval/baseline/bench JSON")
    p.add_argument("--data-dir", required=True, help="data dir with data_report.json (and trap_candidates.csv)")
    p.add_argument("--out", required=True, help="markdown report; gate_report.json is written beside it")
    p.add_argument("--config", default=None, help="config.yaml (default: <project root>/config.yaml)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        res = run(args)
        failed = [r for r in res["criteria"] if not r["passed"]]
        print(f"gate: {res['verdict']} ({len(res['criteria']) - len(failed)}/{len(res['criteria'])} criteria) -> "
              f"{args.out}", *(f"gate: FAIL {r['name']}: {r['note']}" for r in failed), sep="\n")
        if failed:
            print("gate: stop and debug before any spend; check:", *(f"  - {c}" for c in DEBUG_CHECKLIST), sep="\n")
        return 0

    return run_cli("gate", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
