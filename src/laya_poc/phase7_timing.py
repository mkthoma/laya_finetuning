"""GPU scoring throughput and training time of the Phase 3/4 Colab runs, read from their archives (Phase 7 write-up):
<out>.md and <out>.json.

    python -m laya_poc.phase7_timing --runs-root runs/p3_colab/runs/p3 --runs-root runs/p4_colab/runs/p4
                                     --out runs/phase7/gpu_timing.md

Per finished run (done.json; the first root holding a run name wins): done.json, timing.json (one wall clock per
orchestrator step), eval/<split>.json, train/summary.json, train/log.jsonl (its start event), calibration.json.
Which eval field is ONE scoring pass, as the writers record it (read in their code):
- Laya, E* and B2 (evaluate.evaluate_rows): `seconds_post` = one post-T predict_batch pass over the ANSWERED rows
  (rows the no-evidence gate abstains on are never forwarded); `seconds` covers both passes plus metrics.
- B4 (small_encoder.evaluate_run): `seconds` = one score_states pass over EVERY row (tokenise, length-sorted batches,
  fp16 autocast on CUDA), timed before T is fitted; `seconds_post` / `seconds_pre` are 0.
- B5 (llm_baseline.score_and_write): `seconds` = one scoring pass over every row of its eval subset; on val it also
  includes the T fit. The scoring method (packed / per_key) and dtype are in calibration.json `extra`.
- B3 / B1 (baseline_runs.write_splits): `seconds_post` = one scoring pass on the Colab host CPU; `seconds` is only
  baseline_eval's metric computation.
Rows per second = rows forwarded / one-pass seconds. The per-group summary is the median over the group's runs
(seeds) and its splits with >= MIN_SCORED scored rows. Metrics only: no FSQ row is read or written. Torch-free: the
matrix readers import torch (via matrix_plan), so the few readers needed here are small local copies (group ids as
phase5_load.group_id, run discovery as matrix_results.done_run_dirs).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .env_check import run_cli, write_json
from .phase7_timing_md import render_markdown

SCHEMA = "phase7_gpu_timing/1"
MIN_SCORED = 1000
SHORT_PASS_S = 1.0            # eval JSONs round seconds to 0.01: a pass under 1 s carries >= 1 % rounding
LAYA, ENCODER, LLM = "laya", "small_encoder", "llm"
CPU_KINDS = ("tfidf_lr", "majority", "prior")
ONE_PASS_FIELD = {LAYA: "seconds_post", ENCODER: "seconds", LLM: "seconds",
                  **{k: "seconds_post" for k in CPU_KINDS}}
DEFINITIONS = {
    "throughput": "rows forwarded in ONE batched scoring pass of an eval split / that pass's wall-clock seconds, "
                  "as the run's eval JSON recorded it: batched scoring at the eval batch size on the Colab card, "
                  "end to end incl. tokenisation, batching and host-device copies (warm model, one process); "
                  "not batch-1 latency",
    LAYA: "seconds_post: one post-T predict_batch pass over the answered rows (gated rows are not forwarded); "
          "`seconds` covers both passes and is not used. Precision: the laya package's CUDA autocast, fp16 "
          "because the Colab notebooks' setup cell sets LAYA_CUDA_AMP=fp16 (the run files do not record it)",
    ENCODER: "seconds: one score_states pass over every row (tokenise + length-sorted batches, fp16 autocast), "
             "timed before T is fitted; seconds_post/pre are 0",
    LLM: "seconds: one scoring pass over every row of the eval subset (tokenise + teacher-forced key scoring); "
         "on val it also includes the T fit",
    "cpu": "B3/B1 seconds_post: one scoring pass on the Colab host CPU (the GPU is idle); `seconds` is only the "
           "metric computation",
    "train": "Laya: timing.json `train` (train_single subprocess: imports, checkpoint load, training, in-training "
             "val evals, best save); B4: train/summary.json `seconds` (training loop incl. per-epoch val scoring, "
             "no model load); B3: TF-IDF fit + C grid on CPU (calibration.json); B1, B2, B5 train nothing",
    "total": "done.json `seconds`: the run's whole wall clock (every orchestrator step, subprocess start included)",
}


# ---------------------------------------------------------------- readers (torch-free)

def _obj(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def read_json(path: Path) -> Any:
    """Parsed JSON, or None when the file is absent or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def num(x: Any) -> float | None:
    """A finite real number as float, else None."""
    ok = isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    return float(x) if ok else None


def group_id(done: Mapping) -> str:
    """"<arm> <model> <scheme>" + " n<subset>" + " head", as phase5_load.group_id(matrix_report.group_key(done))."""
    subset, head = done.get("subset"), bool(done.get("head_only"))
    return " ".join([str(done.get("arm")), str(done.get("model")), str(done.get("scheme")),
                     *([f"n{subset}"] if subset else []), *(["head"] if head else [])])


def as_roots(roots: Path | str | Iterable[Path]) -> tuple[Path, ...]:
    return (Path(roots),) if isinstance(roots, (str, Path)) else tuple(Path(r) for r in roots)


def done_run_dirs(roots: Sequence[Path]) -> list[Path]:
    """Run directories with a done.json over the roots, sorted by name; a name in several roots: the first."""
    found: dict[str, Path] = {}
    for root in roots:
        if root.is_dir():
            for p in root.iterdir():
                if (p / "done.json").is_file():
                    found.setdefault(p.name, p)
    return [found[n] for n in sorted(found)]


# ---------------------------------------------------------------- one run


def first_start(log: Path) -> dict:
    """The first `start` event of a train/log.jsonl ({} when absent or unreadable)."""
    try:
        with log.open(encoding="utf-8") as fh:
            lines = list(fh)
    except (OSError, ValueError):
        return {}
    for line in lines:  # one bad or non-object line skips only itself
        try:
            event = json.loads(line) if line.strip() else {}
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event") == "start":
            return event
    return {}


@dataclass(frozen=True)
class Run:
    dir: Path
    done: dict
    timing: dict
    evals: dict
    summary: dict
    extra: dict
    start: dict

    @property
    def name(self) -> str:
        return str(self.done.get("run_name") or self.dir.name)

    @property
    def kind(self) -> str:
        return str(self.done.get("kind") or LAYA)

    @property
    def group(self) -> str:
        return group_id(self.done)

    @property
    def device(self) -> str | None:
        devices = sorted({str(e["device"]) for e in self.evals.values() if e.get("device")})
        return ", ".join(devices) or None


def load_run(run_dir: Path) -> Run:
    evals = {p.stem: ev for p in sorted((run_dir / "eval").glob("*.json")) if isinstance(ev := read_json(p), dict)}
    return Run(run_dir, _obj(read_json(run_dir / "done.json")), _obj(read_json(run_dir / "timing.json")), evals,
               _obj(read_json(run_dir / "train" / "summary.json")),
               _obj(_obj(read_json(run_dir / "calibration.json")).get("extra")),
               first_start(run_dir / "train" / "log.jsonl"))


def precision(run: Run) -> str | None:
    """Scoring precision per kind: recorded by B4 / B5; for Laya, the notebooks' LAYA_CUDA_AMP (not in the run)."""
    if run.kind == LAYA:
        if run.device is None:  # no eval JSON read: unknown, not guessed
            return None
        return "fp16 autocast (LAYA_CUDA_AMP, notebook)" if run.device == "cuda" else "fp32"
    if run.kind == ENCODER:
        fp16 = run.summary.get("fp16")
        return None if fp16 is None else "fp16 autocast" if fp16 and run.device == "cuda" else "fp32"
    if run.kind == LLM:
        return run.extra.get("dtype")
    return "cpu (sklearn / numpy)" if run.kind in CPU_KINDS else None


def train_precision(run: Run) -> str | None:
    fp16 = run.start.get("fp16") if run.kind == LAYA else run.summary.get("fp16") if run.kind == ENCODER else None
    return None if fp16 is None else "fp16 AMP" if fp16 else "fp32 / no fp16 AMP"


def train_seconds(run: Run) -> float | None:
    """Training time per the DEFINITIONS["train"] basis; None for runs that train nothing."""
    if run.kind == LAYA:
        return num(run.timing.get("train"))
    if run.kind == ENCODER:
        return num(run.summary.get("seconds"))
    if run.kind == "tfidf_lr":
        parts = [num(_obj(run.extra.get("tfidf")).get("seconds")), num(run.extra.get("seconds_fit"))]
        return round(sum(parts), 2) if None not in parts else None
    return None


def run_notes(run: Run) -> list[str]:
    if run.kind in CPU_KINDS:
        return ["scored on the Colab host CPU (the GPU is idle)"]
    if run.kind != LLM or not run.extra.get("scoring"):
        return []
    bs = num(run.extra.get("batch_size"))
    k = next((len(e["labels"]) for e in run.evals.values() if isinstance(e.get("labels"), list)), None)
    if run.extra["scoring"] == "per_key" and bs and k:
        items = max(1, int(bs) // k)
        return [f"per_key scoring (the packed check failed): {items} items ({items * k} sequences) per forward "
                f"at batch_size {int(bs)}"]
    return [f"{run.extra['scoring']} scoring: {int(bs) if bs else '?'} items per forward"]


# ---------------------------------------------------------------- rows

def rows_forwarded(kind: str, ev: Mapping) -> int | None:
    """Rows through the model: Laya skips gated rows; the baselines score every row and gate afterwards."""
    n = ev.get("n")
    if not isinstance(n, int):
        return None
    return n - int(ev.get("n_abstained_no_evidence") or 0) if kind == LAYA else n


def split_row(run: Run, split: str, ev: Mapping) -> dict[str, Any]:
    field = ONE_PASS_FIELD.get(run.kind)
    secs, fwd = (num(ev.get(field)) if field else None), rows_forwarded(run.kind, ev)
    rate = fwd / secs if secs and fwd else None
    notes = [] if field else [f"unknown run kind {run.kind!r}: one-pass field not verified"]
    notes += ["no row forwarded (the gate abstains on every row)"] if fwd == 0 else []
    notes += ["pass below the 0.01 s timer resolution: no rate"] if secs == 0 and fwd else []
    notes += ["one-pass seconds include the temperature fit on this split"] if run.kind == LLM and split == "val" \
        else []
    return {"run": run.name, "group": run.group, "arm": run.done.get("arm"), "model": run.done.get("model"),
            "kind": run.kind, "seed": run.done.get("seed"), "card": run.done.get("card"), "split": split,
            "device": ev.get("device"), "precision": precision(run), "batch_size": ev.get("batch_size"),
            "n": ev.get("n"), "n_scored": ev.get("n_scored"), "rows_forwarded": fwd, "one_pass_field": field,
            "one_pass_s": secs, "rows_per_s": None if rate is None else round(rate, 2),
            "short_pass": bool(secs and secs < SHORT_PASS_S), "subset": bool(ev.get("subset")), "notes": notes}


def run_row(run: Run, splits: Sequence[Mapping]) -> dict[str, Any]:
    passes = [s["one_pass_s"] for s in splits if s["one_pass_s"] is not None]
    return {"run": run.name, "group": run.group, "arm": run.done.get("arm"), "model": run.done.get("model"),
            "kind": run.kind, "scheme": run.done.get("scheme"), "seed": run.done.get("seed"),
            "card": run.done.get("card"), "device": run.device, "precision": precision(run),
            "train_precision": train_precision(run), "train_s": train_seconds(run),
            "scoring_s": round(sum(passes), 2) if passes else None, "n_splits": len(splits),
            "eval_step_s": num(run.timing.get("evaluate")), "export_check_s": num(run.timing.get("export_check")),
            "load_s": num(run.extra.get("load_seconds")), "total_s": num(run.done.get("seconds")),
            "timing_steps": dict(run.timing), "notes": run_notes(run)}


# ---------------------------------------------------------------- aggregates

def _median(xs: Iterable[float | None], nd: int) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(statistics.median(vals), nd) if vals else None


def _one(values: Iterable[Any]) -> Any:
    """The common value, a comma-joined list when they differ, None when there is none."""
    vals = list(dict.fromkeys(v for v in values if v is not None))
    return vals[0] if len(vals) == 1 else ", ".join(map(str, vals)) if vals else None


def _by(rows: Sequence[Mapping], *keys: str) -> dict[tuple, list[Mapping]]:
    out: dict[tuple, list[Mapping]] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def throughput(splits: Sequence[Mapping]) -> list[dict[str, Any]]:
    """Per (group, split): medians over the group's runs."""
    return [{"group": g, "split": s, "n_runs": len(rs), "rows_forwarded": _one(r["rows_forwarded"] for r in rs),
             "n_scored": _one(r["n_scored"] for r in rs),
             "one_pass_s_median": _median((r["one_pass_s"] for r in rs), 3),
             "rows_per_s_median": _median((r["rows_per_s"] for r in rs), 2),
             "batch_size": _one(r["batch_size"] for r in rs), "device": _one(r["device"] for r in rs),
             "card": _one(r["card"] for r in rs), "precision": _one(r["precision"] for r in rs),
             "short_pass": any(r["short_pass"] for r in rs), "notes": list(dict.fromkeys(
                 n for r in rs for n in r["notes"]))}
            for (g, s), rs in _by(splits, "group", "split").items()]


def summary_row(group: str, runs: Sequence[Mapping], splits: Sequence[Mapping]) -> dict[str, Any]:
    used = [s for s in splits if (s["n_scored"] or 0) >= MIN_SCORED and s["rows_per_s"] is not None]
    rates = [s["rows_per_s"] for s in used]
    notes = list(dict.fromkeys(n for r in runs for n in r["notes"]))
    short = any(s["short_pass"] for s in used)
    notes += ["one-pass times under 1 s (recorded to 0.01 s): rates carry about 1-2 % rounding"] if short else []
    first = runs[0]
    return {"group": group, "arm": first["arm"], "model": first["model"], "kind": first["kind"],
            "card": _one(r["card"] for r in runs), "device": _one(r["device"] for r in runs),
            "precision": _one(r["precision"] for r in runs), "batch_size": _one(s["batch_size"] for s in splits),
            "n_runs": len(runs), "seeds": sorted(r["seed"] for r in runs if r["seed"] is not None),
            "rows_per_s_median": _median(rates, 2), "rows_per_s_min": min(rates) if rates else None,
            "rows_per_s_max": max(rates) if rates else None, "n_points": len(rates),
            "splits_used": sorted({s["split"] for s in used}), "short_pass": short,
            "train_min_median": _median((None if r["train_s"] is None else r["train_s"] / 60 for r in runs), 2),
            "total_min_median": _median((None if r["total_s"] is None else r["total_s"] / 60 for r in runs), 2),
            "notes": notes}


def summarise(runs: Sequence[Mapping], splits: Sequence[Mapping]) -> list[dict[str, Any]]:
    by_split = _by(splits, "group")
    return [summary_row(g, rs, by_split.get((g,), [])) for (g,), rs in _by(runs, "group").items()]


def card_names(roots: Sequence[Path], runs: Sequence[Run]) -> dict[str, str]:
    """{card: GPU name} from each root's env.json, else a B4 summary / B5 calibration `gpu` field."""
    out: dict[str, str] = {}
    for root in roots:
        env = _obj(read_json(root / "env.json"))
        if env.get("card") and env.get("gpu_name"):
            out.setdefault(str(env["card"]), str(env["gpu_name"]))
    for r in runs:
        gpu = r.summary.get("gpu") or r.extra.get("gpu")
        if r.done.get("card") and gpu:
            out.setdefault(str(r.done["card"]), str(gpu))
    return out


def build_timing(roots: Path | Iterable[Path]) -> dict[str, Any]:
    roots = as_roots(roots)
    dirs = done_run_dirs(roots)
    if not dirs:
        raise FileNotFoundError(f"no finished run (done.json) under {', '.join(map(str, roots))}")
    runs = sorted((load_run(d) for d in dirs), key=lambda r: (r.group, r.name))
    per_run = {r.name: [split_row(r, s, ev) for s, ev in r.evals.items()] for r in runs}
    splits = [row for r in runs for row in per_run[r.name]]
    run_rows = [run_row(r, per_run[r.name]) for r in runs]
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "roots": [str(r) for r in roots], "min_scored_rows": MIN_SCORED, "cards": card_names(roots, runs),
            "definitions": DEFINITIONS, "n_runs": len(runs), "summary": summarise(run_rows, splits),
            "throughput": throughput(splits), "runs": run_rows, "splits": splits}


# ---------------------------------------------------------------- CLI

def write_outputs(result: Mapping, out_md: Path) -> tuple[Path, Path]:
    """<out>.md and <out>.json, each written atomically."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_md.with_name(out_md.name + ".tmp")
    tmp.write_text(render_markdown(result), encoding="utf-8")
    tmp.replace(out_md)
    return out_md, write_json(out_md.with_suffix(".json"), dict(result))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.phase7_timing", description=__doc__.splitlines()[0])
    p.add_argument("--runs-root", action="append", required=True,
                   help="an archive's runs directory (repeatable; the first root holding a run name wins)")
    p.add_argument("--out", required=True, help="markdown path (<out>.json is written next to it)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        result = build_timing([Path(r) for r in args.runs_root])
        md, js = write_outputs(result, Path(args.out))
        print(f"phase7_timing: {result['n_runs']} runs, {len(result['summary'])} groups; wrote {md} and {js}",
              flush=True)
        return 0

    return run_cli("phase7_timing", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
