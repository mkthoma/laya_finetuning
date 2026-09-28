"""data_report.json content and the short console summary of a build_data run (design doc §7.4-§7.5).

Split out of build_data so the build steps and what is reported about them stay readable.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .notice import FSQ_NOTICE_NAME
from .splits import SPLIT_ORDER

HEADLINE_CLASSES, HEADLINE_CLASSES_NO_EVENT = 10, 9


def data_report(cfg: dict, smoke: bool, pool_info: dict, splits: dict[str, pd.DataFrame], brands: set[str],
                warnings: list[str], models: list[dict], ctx: Any, results: dict[str, dict], epochs: int,
                out: Path) -> dict[str, Any]:
    """data_report.json: everything needed to review the build without re-running it."""
    extraction = {k: v for k, v in pool_info.items() if k not in ("source", "path", "rows")}
    return {
        "release": cfg["data"]["fsq_release"], "mode": "smoke" if smoke else "full",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "out": str(out),
        "pool": {k: pool_info[k] for k in ("source", "path", "rows")},
        "extraction": extraction if pool_info["source"] == "extract" else None,
        "split_sizes": {n: len(splits[n]) for n in SPLIT_ORDER},
        "label_distributions": {n: {str(k): int(v) for k, v in splits[n]["label"].value_counts().sort_index().items()}
                                for n in SPLIT_ORDER},
        "row_label_distributions": {k: r["labels"] for k, r in results.items()},
        "stats": {k: r["stats"] for k, r in results.items()},
        "warnings": list(warnings), "brands": len(brands), "epochs": epochs,
        "seed": int(cfg["data"]["split_seed"]), "scheme": ctx.scheme,
        "models": {m["name"]: {k: m[k] for k in ("max_len", "head_max_len", "budget", "tokenizer")} for m in models},
        "budgets": {m["name"]: m["budget"] for m in models},
        "leakage": {"category_ids": len(ctx.category_ids), "category_names": len(ctx.category_names)},
    }


def headline(cfg: dict, split_info: Mapping[str, Any]) -> dict[str, Any]:
    """§5.2: with fewer than labels.event_min_id_pool labelled Event places in the ID pool (after the trap
    set-aside, brand holdout, per-name cap and dedup), the headline macro-F1 is the 9-class one."""
    event = int(split_info["id_pool_labels"].get("Event", 0))
    minimum = int(cfg["labels"].get("event_min_id_pool", 300))
    classes = HEADLINE_CLASSES if event >= minimum else HEADLINE_CLASSES_NO_EVENT
    rule = "10-class macro-F1" if classes == HEADLINE_CLASSES else "9-class macro-F1 (Event reported, not averaged)"
    return {"id_pool": dict(split_info), "event_id_pool": event, "headline_classes": classes,
            "headline_rule": f"Event has {event} labelled places in the ID pool (minimum {minimum}): {rule}"}


def summary_lines(report: dict[str, Any]) -> list[str]:
    """At most ~14 lines: Colab / VS Code truncate long cell outputs."""
    ext = report["extraction"]
    src = f"extracted in {ext['seconds']}s" if ext else f"from {report['pool']['path']}"
    lines = [f"build_data: {report['mode']} release={report['release']} pool={report['pool']['rows']} rows ({src})",
             "splits: " + " ".join(f"{k}={v}" for k, v in report["split_sizes"].items() if v),
             "budgets (state tokens): " + " ".join(f"{k}={v}" for k, v in report["budgets"].items())]
    train = {k: v for k, v in report["stats"].items() if k.startswith("train_e")}
    for name, st in report["stats"].items():
        if name not in train:
            lines.append(_stats_line(name, st))
    lines.append(f"train x{len(train)} epochs: " + "; ".join(
        f"{st['written']} rows, {st['compressed']} compressed, {st['rejected']} rejected" for st in train.values()))
    if "fingerprint_sha256" in report:
        lines.append(_full_line(report))
    if report["warnings"]:
        lines.append(f"warnings ({len(report['warnings'])}): {report['warnings'][0]}")
    lines.append(f"wrote {report['out']} (data_report.json, SHA256SUMS, {FSQ_NOTICE_NAME})")
    return lines


def _full_line(report: dict[str, Any]) -> str:
    t = report["traps"]
    return (f"full: {t['candidates']} trap candidates -> {t['file']} (annotation pending), {t['removed_rows']} pool "
            f"rows set aside; Event ID pool {report['event_id_pool']} -> {report['headline_classes']}-class "
            f"headline; fingerprint {report['fingerprint_sha256'][:12]} frozen={report['frozen_check']['status']}")


def _stats_line(name: str, st: dict) -> str:
    base = f"{name}: {st['written']} rows"
    if "compressed" not in st:
        return base
    return (f"{base}, {st['compressed']} compressed, {st['rejected']} rejected, no-evidence {st['no_evidence']}, "
            f"tokens p50/p95/max {st['tokens_p50']:.0f}/{st['tokens_p95']:.0f}/{st['tokens_max']}")
