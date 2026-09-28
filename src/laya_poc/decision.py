"""The design §5.12 decision rule (spec P5 §3): criteria, stop and investigate checks, verdicts.

Pure functions over plain numbers; decision_eval.py feeds them from the phase5_metrics JSON, the bench_cpu JSON
(may be absent) and config phase5.decision / cpu_budget; `outlook` enumerates the pending inputs (trap annotation,
the benchmark) to say which verdicts are still reachable. Rates are fractions (0.03 = 3 macro-F1 points). Per
candidate (seed means):

  C1 OOD lead >= lead_points over B3 (char TF-IDF+LR) AND over the language-matched B4 (primary; the value is
     the smaller lead). `laya` averages ood_country + ood_brand (ood_script excluded: it cannot read Thai).
  C2 ID->OOD gap <= max_gap_points on each included pool (value: the largest gap).
  C3 post-T ECE <= max_ece on test_id AND on the OOD average of the included pools (value: the larger).
  C4 trap accuracy >= the best baseline's; PENDING until the trap set is annotated.
  C5 on cpu_budget.vcpus threads, fp32: batch-1 p95 <= p95_ms AND batched rps >= min_rps on ONE backend
     ("whichever of ONNX or PyTorch is faster" = the backend that meets it; ONNX only when its acceptance check
     did not fail); PENDING without the bench JSON or a row at that thread count.

Stop: (a) below B3 AND the matched B4 on test_id AND on the OOD average; (b) post-T ECE > stop_ece on test_id or
the OOD average; (c) p95 > stop_p95_ms with ONNX on 8 threads. Investigate: (i) exactly one criterion fails, by
less than investigate_fraction_of_margin of its margin, the margin being the criterion's own threshold (C1 3
points: a lead in (1.5, 3); C2 5 points: a gap in (5, 7.5); C3 0.05: ECE in (0.05, 0.075); C5 relative: p95
< 1.5 x budget and rps > 0.5 x floor; C4 has no margin, so a trap miss is never near). The doc's own example of
this case is a +1.5-3 point OOD lead, so "passes on accuracy" cannot exclude a C1 near miss: the single miss may
be C1. (ii) a seed range of a headline macro-F1 > investigate_seed_range_points. (iii) the gate failed once and a
bug was fixed: not applicable (the Phase 2 gate passed first time), recorded false.

Verdict per candidate: PASS (all five pass) > STOP (any stop condition) > INVESTIGATE > NO PASS. Overall: PASS
if any candidate passes (§5.12 "for at least one fine-tuned checkpoint"); STOP only if EVERY candidate stops
(the stop is global: one checkpoint clear of every stop condition keeps Laya alive); else INVESTIGATE if any
candidate investigates; else NO_PASS, for which the doc has no rule.
"""
from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

PASS, FAIL, PENDING = "PASS", "FAIL", "PENDING"
STOP, INVESTIGATE = "STOP", "INVESTIGATE"
NO_PASS = "NO PASS (no formal stop condition met — LDS judgement)"
VERDICT_ORDER = (PASS, STOP, INVESTIGATE, NO_PASS)
EPS = 1e-9                         # float slack: 0.53 - 0.50 is exactly 3 points
STOP_THREADS = 8                   # stop (c): "even with ONNX on 8 threads"
GATE_NOTE = "not applicable: the Phase 2 gate passed first time (PASS 4/4); no gate failure was fixed"


def thresholds(cfg: Mapping[str, Any]) -> dict[str, float]:
    """Config phase5.decision (points -> fractions) and cpu_budget as one flat dict."""
    d, b = (cfg.get("phase5") or {}).get("decision"), cfg.get("cpu_budget")
    if not d or not b:
        raise ValueError("config needs phase5.decision and cpu_budget (design §5.12 thresholds)")
    return {"lead": d["lead_points"] / 100, "max_gap": d["max_gap_points"] / 100, "max_ece": float(d["max_ece"]),
            "stop_ece": float(d["stop_ece"]), "stop_p95_ms": float(d["stop_p95_ms"]),
            "seed_range": d["investigate_seed_range_points"] / 100,
            "fraction": float(d["investigate_fraction_of_margin"]),
            "p95_ms": float(b["p95_ms"]), "min_rps": float(b["min_rps"]), "vcpus": int(b.get("vcpus", 4))}


def judge(value: float | None, threshold: float, *, higher: bool, margin: float | None,
          fraction: float) -> tuple[str, bool | None]:
    """(status, near_miss): near_miss only for a FAIL, True when it misses by less than fraction x margin."""
    if value is None:
        return PENDING, None
    miss = threshold - value if higher else value - threshold
    if miss <= EPS:
        return PASS, None
    return FAIL, margin is not None and miss < fraction * margin - EPS


def _criterion(cid: int, name: str, value: Any, status: str, near: bool | None, threshold: Any, *,
               note: str = "", source: str | None = None, **detail: Any) -> dict[str, Any]:
    return {"id": cid, "name": name, "status": status, "value": value, "threshold": threshold, "near_miss": near,
            "note": note, "source": source if status == PENDING else None, **detail}


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def criterion_1(cand: float | None, b3: float | None, b4: float | None, th: Mapping[str, float], *,
                b4_label: str = "B4", note: str = "") -> dict[str, Any]:
    """OOD-average lead over B3 and over the (matched or best) B4; the value is the smaller lead."""
    leads = {"B3": _diff(cand, b3), "B4": _diff(cand, b4)}
    missing = [k for k, v in (("candidate", cand), ("B3", b3), (b4_label, b4)) if v is None]
    value = None if missing else min(leads.values())
    status, near = judge(value, th["lead"], higher=True, margin=th["lead"], fraction=th["fraction"])
    why = f"missing OOD average: {', '.join(missing)}" if missing else note
    return _criterion(1, "OOD lead over B3 and B4", value, status, near, th["lead"], note=why, source="metrics",
                      leads=leads, b4=b4_label)


def criterion_2(gaps: Mapping[str, float | None], th: Mapping[str, float]) -> dict[str, Any]:
    """ID->OOD gap on each included pool; the value is the largest gap."""
    missing = [p for p, g in gaps.items() if g is None]
    known = {p: g for p, g in gaps.items() if g is not None}
    worst = max(known, key=known.__getitem__) if known and not missing else None
    value = known[worst] if worst else None
    status, near = judge(value, th["max_gap"], higher=False, margin=th["max_gap"], fraction=th["fraction"])
    note = f"missing gap: {', '.join(missing)}" if missing else ("no included OOD pool" if not gaps else "")
    return _criterion(2, "ID->OOD gap on each included pool", value, status, near, th["max_gap"], note=note,
                      source="metrics", gaps=dict(gaps), worst_pool=worst)


def criterion_3(ece_id: float | None, ece_ood: float | None, th: Mapping[str, float]) -> dict[str, Any]:
    """Post-T ECE on test_id and on the OOD average; the value is the larger."""
    value = None if ece_id is None or ece_ood is None else max(ece_id, ece_ood)
    status, near = judge(value, th["max_ece"], higher=False, margin=th["max_ece"], fraction=th["fraction"])
    note = "missing post-T ECE" if value is None else ""
    return _criterion(3, "post-T ECE on test_id and the OOD average", value, status, near, th["max_ece"],
                      note=note, source="metrics", ece_test_id=ece_id, ece_ood_avg=ece_ood)


def criterion_4(cand: float | None, baselines: Mapping[str, float | None], annotated: bool,
                th: Mapping[str, float]) -> dict[str, Any]:
    """Trap accuracy (all items) >= the best baseline's; no margin, so never a near miss."""
    if not annotated:
        return _criterion(4, "trap accuracy >= best baseline", None, PENDING, None, None, source="traps",
                          note="trap set not annotated yet (traps merge -> data/trap.jsonl)", best=None)
    known = {k: v for k, v in baselines.items() if v is not None}
    best = max(known, key=known.__getitem__) if known else None
    bar = known.get(best) if best else None
    value = None if cand is None or bar is None else cand
    status, near = judge(value, bar if bar is not None else 0.0, higher=True, margin=None,
                         fraction=th["fraction"])
    note = "" if value is not None else "missing trap accuracy (candidate or every baseline)"
    return _criterion(4, "trap accuracy >= best baseline", value, status, near, bar, note=note,
                      source="metrics", best=best, baselines=dict(baselines))


def model_rows(rows: Sequence[Mapping[str, Any]] | None, model: str) -> list[dict[str, Any]] | None:
    return None if rows is None else [dict(r) for r in rows if r.get("model") == model]


def _meets(r: Mapping[str, Any], p95: float, rps: float) -> bool:
    return r["p95_ms"] <= p95 + EPS and r["batch_rps"] >= rps - EPS


def criterion_5(rows: Sequence[Mapping[str, Any]] | None, model: str, th: Mapping[str, float], *,
                onnx_ok: bool | None = None) -> dict[str, Any]:
    """CPU budget at th['vcpus'] threads on the backend that meets it (else the lowest-p95 one)."""
    name, bar = "CPU budget (4 threads, fp32)", {"p95_ms": th["p95_ms"], "min_rps": th["min_rps"]}
    mine = model_rows(rows, model)
    if mine is None:
        return _criterion(5, name, None, PENDING, None, bar, source="bench", note="bench JSON absent")
    usable = [r for r in mine if r.get("threads") == th["vcpus"] and not (r.get("backend") == "onnx"
                                                                          and onnx_ok is False)]
    note = "ONNX rows ignored: the ONNX acceptance check failed" if onnx_ok is False else ""
    if not usable:
        return _criterion(5, name, None, PENDING, None, bar, source="bench",
                          note=f"no {th['vcpus']}-thread row for {model}" + (f"; {note}" if note else ""))
    usable.sort(key=lambda r: (not _meets(r, th["p95_ms"], th["min_rps"]), r["p95_ms"]))
    best, f = usable[0], th["fraction"]
    ok = _meets(best, th["p95_ms"], th["min_rps"])
    near = None if ok else any(_meets(r, th["p95_ms"] * (1 + f) - 2 * EPS, th["min_rps"] * (1 - f) + 2 * EPS)
                               for r in usable)
    value = {k: best.get(k) for k in ("backend", "threads", "p95_ms", "batch_rps")}
    return _criterion(5, name, value, PASS if ok else FAIL, near, bar, note=note,
                      backends={r.get("backend"): {"p95_ms": r["p95_ms"], "batch_rps": r["batch_rps"]}
                                for r in usable})


# ---------------------------------------------------------------- stop checks

def _check(cid: str, name: str, holds: bool | None, *, note: str = "", source: str | None = None,
           **detail: Any) -> dict[str, Any]:
    return {"id": cid, "name": name, "holds": holds, "note": note,
            "source": (source or "metrics") if holds is None else None, **detail}


def stop_below_baselines(cand: tuple, b3: tuple, b4: tuple) -> dict[str, Any]:
    """(a) below both trained baselines on ID and on OOD; each argument is (test_id F1, OOD average)."""
    name = "below B3 and the matched B4 on test_id and the OOD average"
    values = [*cand, *b3, *b4]
    if any(v is None for v in values):
        return _check("a", name, None, note="missing macro-F1 (candidate, B3 or matched B4)")
    below = [cand[i] < b3[i] - EPS and cand[i] < b4[i] - EPS for i in (0, 1)]
    return _check("a", name, all(below), below_on_id=below[0], below_on_ood=below[1])


def stop_ece(ece_id: float | None, ece_ood: float | None, th: Mapping[str, float]) -> dict[str, Any]:
    """(b) post-T ECE above stop_ece on test_id or on the OOD average."""
    name = f"post-T ECE > {th['stop_ece']:.2f} (test_id or OOD average)"
    over = [v > th["stop_ece"] + EPS for v in (ece_id, ece_ood) if v is not None]
    values = {"ece_test_id": ece_id, "ece_ood_avg": ece_ood}
    if any(over):
        return _check("b", name, True, **values)
    return _check("b", name, None if len(over) < 2 else False, note="" if len(over) == 2 else "missing ECE",
                  **values)


def stop_latency(rows: Sequence[Mapping[str, Any]] | None, model: str, th: Mapping[str, float]) -> dict[str, Any]:
    """(c) p95 above stop_p95_ms even with ONNX on 8 threads. Without an 8-thread row (threads are capped at the
    physical cores), the best measured ONNX setting can refute the stop but never confirm it."""
    name = f"p95 > {th['stop_p95_ms']:.0f} ms with ONNX on {STOP_THREADS} threads"
    onnx = [r for r in model_rows(rows, model) or [] if r.get("backend") == "onnx"]
    if rows is None or not onnx:
        return _check("c", name, None, source="bench", threads=None, p95_ms=None,
                      note="bench JSON absent" if rows is None else f"no ONNX row for {model}")
    top = max(r["threads"] for r in onnx)
    p95 = min(r["p95_ms"] for r in onnx if r["threads"] == top)
    over = p95 > th["stop_p95_ms"] + EPS
    if top == STOP_THREADS:
        return _check("c", name, over, threads=top, p95_ms=p95)
    note = (f"ONNX on {STOP_THREADS} threads not measured (threads capped at the physical cores); the most "
            f"threads measured: {top}, p95 {p95:.0f} ms" + ("" if over else ", already within the stop bound"))
    return _check("c", name, None if over else False, source="bench", threads=top, p95_ms=p95, note=note)


# ---------------------------------------------------------------- investigate checks

def near_miss_check(criteria: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """(i) exactly one criterion fails, by less than half its margin (None while pending inputs can decide)."""
    name = "misses a single criterion by less than half its margin"
    fails = [c for c in criteria if c["status"] == FAIL]
    pending = [c for c in criteria if c["status"] == PENDING]
    detail = {"fails": [c["id"] for c in fails], "near": [c["id"] for c in fails if c["near_miss"]]}
    if len(fails) > 1 or (len(fails) == 1 and not fails[0]["near_miss"]):
        return _check("near_miss", name, False, **detail)
    if pending:
        sources = ", ".join(sorted({c.get("source") or "metrics" for c in pending}))
        return _check("near_miss", name, None, source=sources,
                      note="pending: " + ", ".join(f"C{c['id']}" for c in pending), **detail)
    return _check("near_miss", name, len(fails) == 1, **detail)


def seed_range_check(ranges: Mapping[str, float | None], th: Mapping[str, float]) -> dict[str, Any]:
    """(ii) seed range (max - min over seeds) of a headline macro-F1 above the threshold."""
    name = f"seed range of a headline macro-F1 > {100 * th['seed_range']:.0f} points"
    known = {k: v for k, v in ranges.items() if v is not None}
    flagged = [k for k, v in known.items() if v > th["seed_range"] + EPS]
    holds = True if flagged else (None if not known else False)
    return _check("seed_range", name, holds, flagged=flagged, ranges=dict(ranges),
                  note="" if known else "no seed ranges")


def gate_check() -> dict[str, Any]:
    """(iii) the gate failed once and a clear bug was fixed: not applicable here."""
    return _check("gate_fixed", "the gate failed once and a clear bug was fixed", False, note=GATE_NOTE)


# ---------------------------------------------------------------- verdicts (resolved inputs only)

def candidate_verdict(crits: Sequence[tuple[str, bool | None]], stops: Sequence[bool], seed: bool) -> str:
    """crits: (status, near_miss) per criterion, all PASS or FAIL; stops / seed: resolved booleans."""
    if all(s == PASS for s, _ in crits):
        return PASS
    if any(stops):
        return STOP
    fails = [bool(near) for s, near in crits if s == FAIL]
    return INVESTIGATE if seed or fails == [True] else NO_PASS


def overall_verdict(verdicts: Sequence[str]) -> str:
    if PASS in verdicts:
        return PASS
    if verdicts and all(v == STOP for v in verdicts):
        return STOP
    return INVESTIGATE if INVESTIGATE in verdicts else NO_PASS


# ---------------------------------------------------------------- outlook (pending inputs enumerated)

CRITERION_OUTCOMES = ((PASS, None), (FAIL, True), (FAIL, False))   # pass, near miss, clear miss
NOT_MET = {"crit": (FAIL, False), "stop": False, "seed": False}      # "now": a pending item counts as not met


def _slots(cands: Sequence[Mapping[str, Any]]) -> list[tuple]:
    """(candidate, field, index, options, source, item) for every unresolved criterion or check."""
    out = []
    for i, c in enumerate(cands):
        for j, cr in enumerate(c["criteria"]):
            if cr["status"] == PENDING:
                opts = CRITERION_OUTCOMES if cr["id"] != 4 else CRITERION_OUTCOMES[::2]   # C4 has no margin
                out.append((i, "crit", j, opts, cr.get("source") or "metrics", f"{c['label']} C{cr['id']}"))
        for j, s in enumerate(c["stop"]):
            if s["holds"] is None:
                out.append((i, "stop", j, (True, False), s.get("source") or "metrics",
                            f"{c['label']} stop ({s['id']})"))
        seed = next(x for x in c["investigate"] if x["id"] == "seed_range")
        if seed["holds"] is None:
            out.append((i, "seed", 0, (True, False), seed.get("source") or "metrics", f"{c['label']} seed range"))
    return out


def _resolve(cands: Sequence[Mapping[str, Any]], slots: Sequence[tuple], combo: Sequence[Any]) -> tuple:
    """(overall verdict, per-candidate verdicts) with the slots set to `combo`."""
    states = [{"crit": [(cr["status"], cr["near_miss"]) for cr in c["criteria"]],
               "stop": [s["holds"] for s in c["stop"]],
               "seed": [next(x for x in c["investigate"] if x["id"] == "seed_range")["holds"]]} for c in cands]
    for (i, field, j, *_), value in zip(slots, combo):
        states[i][field][j] = value
    per = tuple(candidate_verdict(s["crit"], s["stop"], bool(s["seed"][0])) for s in states)
    return overall_verdict(per), per


def _ordered(verdicts: set[str]) -> list[str]:
    return [v for v in VERDICT_ORDER if v in verdicts]


def _can_change(results: Sequence[tuple], slots: Sequence[tuple], source: str) -> bool:
    """True when, for some values of the OTHER pending items, this input's items alone change the verdict."""
    others = [k for k, s in enumerate(slots) if s[4] != source]
    seen: dict[tuple, set[str]] = {}
    for combo, verdict, _ in results:
        seen.setdefault(tuple(combo[k] for k in others), set()).add(verdict)
    return any(len(v) > 1 for v in seen.values())


def outlook(cands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The verdict now (pending items counted as not met), every verdict still reachable, and which pending
    inputs can still change it. Inputs shared by candidates are enumerated independently (an over-estimate of
    what is reachable, never an under-estimate)."""
    slots = _slots(cands)
    results = [(combo, *_resolve(cands, slots, combo)) for combo in itertools.product(*(s[3] for s in slots))]
    now, now_per = _resolve(cands, slots, [NOT_MET[s[1]] for s in slots])
    possible = _ordered({v for _, v, _ in results})
    per = [{"label": c["label"], "verdict": now_per[i], "possible": _ordered({p[i] for *_, p in results})}
           for i, c in enumerate(cands)]
    inputs = [{"input": src, "items": [s[5] for s in slots if s[4] == src],
               "can_change_verdict": _can_change(results, slots, src)}
              for src in sorted({s[4] for s in slots})]
    out = {"verdict": now, "final": len(possible) == 1, "possible": possible, "per_candidate": per,
           "pending_inputs": inputs}
    return {**out, "statement": statement(out, cands)}


def _definite_fails(c: Mapping[str, Any]) -> list[str]:
    return [f"C{cr['id']}" for cr in c["criteria"] if cr["status"] == FAIL]


def statement(out: Mapping[str, Any], cands: Sequence[Mapping[str, Any]]) -> str:
    """One paragraph after the verdict (which it does not restate): whether it is final and what the pending inputs
    can still do."""
    inputs = out["pending_inputs"]
    if not inputs:
        text = "Final: every input is in (no pending input)."
    elif out["final"]:
        text = f"Final: the pending inputs ({', '.join(p['input'] for p in inputs)}) cannot change it."
    else:
        can = [f"{p['input']} ({', '.join(p['items'])})" for p in inputs if p["can_change_verdict"]]
        cannot = [p["input"] for p in inputs if not p["can_change_verdict"]]
        text = (f"Counted with the pending items as not met; not final: still reachable: "
                f"{', '.join(out['possible'])}. Pending inputs that can change it: {'; '.join(can)}."
                + (f" Pending inputs that cannot: {', '.join(cannot)}." if cannot else ""))
    if PASS not in out["possible"] and inputs:
        why = "; ".join(f"{c['label']}: {', '.join(_definite_fails(c))}" for c in cands if _definite_fails(c))
        text += f" PASS is out of reach whatever the pending inputs show: every candidate already fails ({why})."
    return text
