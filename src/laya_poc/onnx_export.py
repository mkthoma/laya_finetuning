"""Export a Laya checkpoint to ONNX as laya's scripts/export_onnx.py does, and the §7.11 acceptance check.

Export (`export`), replicating the pinned upstream script (laya repo scripts/export_onnx.py, not installed): the
checkpoint's model on a CPU agent in fp32 (compile off: parity.load_reference, which also forces fp32 weights and
amp off), traced with a (batch 1, seq 16) dummy (input ids in [0, 100), full mask, markers at [1, 5], qtype 0),
`torch.onnx.export(model, inputs, path, export_params=True, opset_version=18, do_constant_folding=True,
input_names=INPUT_NAMES, output_names=OUTPUT_NAMES, dynamic_axes=DYNAMIC_AXES)`. Every exported graph must pass a
dynamic-shape probe on ONNX Runtime (batch 3, seq 21, 4 markers, one padded row) or the next attempt runs:
  1. upstream verbatim: torch's default exporter (dynamo on torch >= 2.9; needs onnxscript), batch-1 dummy;
  2. `dynamo=False` (the TorchScript exporter upstream was written against), batch-1 dummy;
  3. the default exporter with a batch-2 dummy (else identical to 1).
Found on torch 2.14 / onnxruntime 1.30: (1) specialises the batch axis to the dummy's 1 (torch.export's 0/1
specialisation), so it only runs batch-1 inputs; (2) bakes the dummy's seq_len 16 into the Laya head's
nn.MultiheadAttention reshape, so it runs no real input; (3) keeps batch, sequence and markers symbolic. The attempt
used (`attempt`, `dummy_batch`), the failed attempts and the de-duplicated warnings are recorded. The dummy ids come
from a seeded generator (upstream: unseeded; the graph does not depend on the values).

Acceptance (`check`, design §7.11: argmax agreement with PyTorch fp32 >= 999/1000 on test_id, max |dp| <= 1e-3): the
first n STRING states, encoded ONCE by the PyTorch agent's own `_encode_state`, collated in length-sorted batches
(`collate_items`, dynamic batch and sequence shapes) and fed IDENTICALLY to the fp32 PyTorch model and to the ONNX
session; the unrounded post-temperature choice probabilities (the checkpoint's own temperature bucket) are compared.

    python -m laya_poc.onnx_export --model laya|laya_ml --ckpt hub|<abs Laya dir> --out <abs file.onnx>
        [--opset 18] [--exporter auto|dynamo|torchscript] [--check-rows <test_id.jsonl>] [--check-n 1000]
        [--batch-size 32] [--reuse] [--config yaml]

Writes <out> (+ <out>.data, the dynamo exporter's external weights) and <out>.json {model, ckpt, ckpt_dir, onnx_path,
identity, export, acceptance, passed, seconds}. Exit 0 when the export succeeded and the acceptance check (if
requested) passed. Run it with PYTHONIOENCODING=utf-8 on Windows (the dynamo exporter prints emoji progress lines).
"""
from __future__ import annotations

import argparse
import inspect
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import load_config
from .env_check import package_version, run_cli, write_json

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]
DYNAMIC_AXES = {  # verbatim from upstream scripts/export_onnx.py
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
    "marker_pos": {0: "batch_size", 1: "num_markers"},
    "marker_mask": {0: "batch_size", 1: "num_markers"},
    "qtype": {0: "batch_size"},
    "logits": {0: "batch_size", 1: "num_markers"},
    "act_logits": {0: "batch_size"},
}
# --exporter -> the (dynamo argument, dummy batch) attempts, in order (None = torch's default exporter)
EXPORTERS = {"auto": ((None, 1), (False, 1), (None, 2)), "dynamo": ((True, 1), (True, 2)), "torchscript": ((False, 1),)}
PROBE_SHAPE = (3, 21, 4)  # batch, seq_len, num_markers: all different from the (1, 16, 2) dummy
MAX_WARNINGS = 25
DUMMY_SEED = 0
ORT_LOG_FATAL = 4  # SessionOptions.log_severity_level: a failing probe raises; no duplicate ORT log


# ---------------------------------------------------------------- export

def dummy_inputs(batch: int = 1) -> tuple[Any, ...]:
    """Upstream's dummy (batch 1, seq 16, markers [1, 5], qtype 0), repeated `batch` times."""
    import torch

    g = torch.Generator().manual_seed(DUMMY_SEED)
    return (torch.randint(0, 100, (batch, 16), dtype=torch.long, generator=g),
            torch.ones((batch, 16), dtype=torch.long), torch.tensor([[1, 5]] * batch, dtype=torch.long),
            torch.tensor([[True, True]] * batch, dtype=torch.bool), torch.tensor([0] * batch, dtype=torch.long))


def torch_default_dynamo() -> bool | None:
    import torch

    p = inspect.signature(torch.onnx.export).parameters.get("dynamo")
    return None if p is None else bool(p.default)


def _exporter_name(dynamo: bool | None) -> str:
    effective = torch_default_dynamo() if dynamo is None else dynamo
    return "dynamo" if effective else "torchscript"


def _warning_lines(caught: Sequence[warnings.WarningMessage]) -> list[str]:
    seen = {}
    for w in caught:
        where = f"{Path(w.filename).name}:{w.lineno}"
        seen.setdefault(f"{w.category.__name__}: {str(w.message).splitlines()[0][:160]} ({where})", None)
    return list(seen)[:MAX_WARNINGS]


def _remove_outputs(out_path: Path) -> None:
    for p in (out_path, Path(f"{out_path}.data")):
        p.unlink(missing_ok=True)


def _export_once(model: Any, out_path: Path, opset: int, dynamo: bool | None, batch: int) -> list[str]:
    import torch

    kwargs: dict[str, Any] = dict(export_params=True, opset_version=opset, do_constant_folding=True,
                                  input_names=INPUT_NAMES, output_names=OUTPUT_NAMES, dynamic_axes=DYNAMIC_AXES)
    if dynamo is not None:
        kwargs["dynamo"] = dynamo
    _remove_outputs(out_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(model, dummy_inputs(batch), str(out_path), **kwargs)
    return _warning_lines(caught)


def probe(onnx_path: str | Path, shape: tuple[int, int, int] = PROBE_SHAPE) -> dict[str, Any]:
    """Run the graph on ONNX Runtime at a batch/sequence/marker shape unlike the dummy; raise unless it follows."""
    import onnxruntime as ort
    from .bench_onnx import CPU_PROVIDER, session_options

    b, s, k = shape
    rng = np.random.default_rng(DUMMY_SEED)
    feed = {"input_ids": rng.integers(0, 100, (b, s), dtype=np.int64), "attention_mask": np.ones((b, s), np.int64),
            "marker_pos": np.tile(np.arange(1, 1 + k, dtype=np.int64), (b, 1)),
            "marker_mask": np.ones((b, k), dtype=bool), "qtype": np.zeros(b, dtype=np.int64)}
    feed["attention_mask"][0, s - 3:] = 0  # one padded row
    so = session_options(None)
    so.log_severity_level = ORT_LOG_FATAL
    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=[CPU_PROVIDER])
    logits, act = session.run(OUTPUT_NAMES, feed)
    if logits.shape != (b, k) or act.shape[0] != b or not (np.isfinite(logits).all() and np.isfinite(act).all()):
        raise ValueError(f"dynamic-shape probe failed: logits {logits.shape} (want {(b, k)}), act {act.shape}, "
                         f"finite {bool(np.isfinite(logits).all())}")
    return {"shape": list(shape), "logits_shape": list(logits.shape), "act_shape": list(act.shape)}


def _attempt(model: Any, out_path: Path, opset: int, dynamo: bool | None, batch: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    warns = _export_once(model, out_path, opset, dynamo, batch)
    return {"exporter": _exporter_name(dynamo), "dynamo_arg": dynamo, "dummy_batch": batch, "warnings": warns,
            "probe": probe(out_path), "seconds": round(time.perf_counter() - t0, 2), "bytes": _size(out_path)}


def export_model(model: Any, out_path: str | Path, opset: int = 18, exporter: str = "auto") -> dict[str, Any]:
    """Export `model` with the first of EXPORTERS[exporter]'s attempts whose graph passes the dynamic-shape probe."""
    out_path, failed = Path(out_path), []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan = EXPORTERS[exporter]
    for number, (dynamo, batch) in enumerate(plan, start=1):
        try:
            rec = _attempt(model, out_path, opset, dynamo, batch)
        except Exception as exc:  # noqa: BLE001 - recorded, then the next attempt runs
            failed.append({"attempt": number, "exporter": _exporter_name(dynamo), "dynamo_arg": dynamo,
                           "dummy_batch": batch, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"})
            continue
        return {**rec, "attempt": number, "attempts_planned": len(plan), "request": exporter, "opset": opset,
                "upstream_verbatim": number == 1 and exporter == "auto", "failed_attempts": failed,
                "torch_default_dynamo": torch_default_dynamo()}
    _remove_outputs(out_path)
    raise RuntimeError("ONNX export failed: " + "; ".join(
        f"#{a['attempt']} {a['exporter']} (dummy batch {a['dummy_batch']}): {a['error']}" for a in failed))


def _size(out_path: Path) -> int:
    return sum(p.stat().st_size for p in (out_path, Path(f"{out_path}.data")) if p.exists())


def export(ckpt_source: Any, out_path: str | Path, opset: int = 18, *, exporter: str = "auto",
           agent: Any = None) -> dict[str, Any]:
    """Load the fp32 CPU agent (unless given) and export its model; returns the export record."""
    from .parity import load_reference

    agent = agent if agent is not None else load_reference(ckpt_source)
    return export_model(agent.model, out_path, opset, exporter)


# ---------------------------------------------------------------- acceptance check (§7.11)

def check_states(rows: Any, n: int) -> tuple[list[str], str | None]:
    """The first n STATE strings of a JSONL path, of row dicts, or of strings (and the path, when one was given)."""
    import itertools
    from .io_utils import iter_jsonl

    if isinstance(rows, (str, Path)):
        states, where = [r["state"] for r in itertools.islice(iter_jsonl(rows), n)], str(rows)
    else:
        states, where = [r["state"] if isinstance(r, dict) else r for r in list(rows)[:n]], None
    if not states:
        raise ValueError(f"no rows to check in {where or 'the given rows'}")
    if any(not isinstance(s, str) for s in states):
        raise TypeError("states must be the compact JSON strings from the rows, never parsed dicts")
    return states, where


def choice_temperature(agent: Any, qtype: int, k: int) -> float:
    """The temperature the runtime applies to a k-option question of this type (bucket, else the per-type value)."""
    from laya.common import temp_bucket

    return float(agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype]))


def paired_logits(agent: Any, session: Any, encoded: list[list[dict]], batch_size: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """[n, k] logits of the (single) question from the fp32 PyTorch model and from ONNX, on identical batches."""
    import torch
    from laya.common import collate_items
    from .bench_onnx import onnx_forward

    k = len(encoded[0][0]["markers"])
    order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i][0]["ids"]), i))
    z_torch, z_onnx = np.full((len(encoded), k), np.nan), np.full((len(encoded), k), np.nan)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        b = collate_items([encoded[i] for i in idx], agent.tok.pad_token_id)
        with torch.inference_mode():
            zt, _ = agent._forward(b)
        zo, _ = onnx_forward(session, b)
        z_torch[idx], z_onnx[idx] = zt[:, :k], zo[:, :k]
    return z_torch, z_onnx


def check(ckpt_source: Any, onnx_path: str | Path, rows: Any, n: int, *, question: dict, max_len: int,
          head_max_len: int, batch_size: int = 32, min_argmax_agree: float = 0.999, max_dp: float = 1e-3,
          agent: Any = None, threads: int | None = None) -> dict[str, Any]:
    """ONNX vs PyTorch fp32 on the first n rows: argmax agreement and max |dp| against the §7.11 thresholds."""
    from laya.common import QTYPES
    from .bench_onnx import encode_states, make_session, prepare_questions, softmax
    from .parity import compare, load_reference

    t0 = time.perf_counter()
    states, where = check_states(rows, n)
    ids, internal = prepare_questions(question)
    if len(ids) != 1:
        raise ValueError(f"the acceptance check compares one question, got {len(ids)}")
    agent = agent if agent is not None else load_reference(ckpt_source)
    encoded = encode_states(agent, states, ids, internal, max_len=max_len, head_max_len=head_max_len)
    z_torch, z_onnx = paired_logits(agent, make_session(onnx_path, threads), encoded, batch_size)
    T = choice_temperature(agent, QTYPES[internal[ids[0]]["t"]], z_torch.shape[1])
    cmp = compare(softmax(z_torch / T), softmax(z_onnx / T))
    ok = (cmp["max_dp"] is not None and cmp["max_dp"] <= max_dp and cmp["argmax_agree"] >= min_argmax_agree
          and not (cmp["nan_ref"] or cmp["nan_test"]))
    lengths = [len(e[0]["ids"]) for e in encoded]
    return {"n": len(states), "rows": where, "batch_size": batch_size, "options": int(z_torch.shape[1]),
            "temperature": T, **cmp, "n_disagree": int(round((1.0 - cmp["argmax_agree"]) * len(states))),
            "max_dlogit": float(np.nanmax(np.abs(z_torch - z_onnx))), "seq_len_range": [min(lengths), max(lengths)],
            "min_argmax_agree": min_argmax_agree, "max_dp_threshold": max_dp, "passed": bool(ok),
            "reference": "PyTorch fp32 CPU agent (amp off), identical collated batches",
            "compared": "unrounded post-temperature choice probabilities",
            "seconds": round(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------- CLI

def _sig(path: Path) -> list[int] | None:
    return [path.stat().st_size, path.stat().st_mtime_ns] if path.exists() else None


def identity(args: argparse.Namespace, cfg: dict, ckpt_dir: Path) -> dict[str, Any]:
    """Everything the export and the check depend on: a matching earlier <out>.json may be reused (--reuse)."""
    mc, rows = cfg["model"][args.model], Path(args.check_rows) if args.check_rows else None
    check_cfg = None if rows is None or args.check_n <= 0 else {
        "rows": str(rows), "rows_sig": _sig(rows), "n": args.check_n, "batch_size": args.batch_size,
        "max_len": int(mc["max_len"]), "head_max_len": int(mc["head_max_len"]),
        "min_argmax_agree": args.min_argmax_agree, "max_dp": args.max_dp, "scheme": cfg["labels"]["scheme"]}
    return {"model": args.model, "ckpt_dir": str(ckpt_dir), "weights": _sig(ckpt_dir / "model.safetensors"),
            "opset": args.opset, "exporter": args.exporter, "torch": package_version("torch"),
            "onnx": package_version("onnx"), "onnxruntime": package_version("onnxruntime"), "check": check_cfg}


def _reusable(meta_path: Path, out: Path, ident: dict[str, Any]) -> dict[str, Any] | None:
    import json

    if not (meta_path.is_file() and out.is_file()):
        return None
    try:
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return prev if prev.get("identity") == ident else None


def _print_acceptance(acc: dict[str, Any]) -> None:
    print(f"onnx_export: acceptance on {acc['n']} rows: argmax agreement {acc['argmax_agree']:.4f} (>= "
          f"{acc['min_argmax_agree']}), max |dp| {acc['max_dp']:.2e} (<= {acc['max_dp_threshold']:g}), max |dlogit| "
          f"{acc['max_dlogit']:.2e}: {'PASS' if acc['passed'] else 'FAIL'}", flush=True)


def _acceptance(args: argparse.Namespace, cfg: dict, out: Path, agent: Any) -> dict[str, Any]:
    from .labels import question

    mc = cfg["model"][args.model]
    acc = check(None, out, args.check_rows, args.check_n, question=question(cfg["labels"]["scheme"]),
                max_len=int(mc["max_len"]), head_max_len=int(mc["head_max_len"]), batch_size=args.batch_size,
                min_argmax_agree=args.min_argmax_agree, max_dp=args.max_dp, agent=agent)
    _print_acceptance(acc)
    return acc


def run(args: argparse.Namespace) -> dict[str, Any]:
    from .bench_sources import LAYA, resolve_source
    from .parity import load_reference

    cfg, t0, out = load_config(args.config), time.perf_counter(), Path(args.out)
    if not out.is_absolute():
        raise ValueError(f"--out must be an absolute path, got {args.out!r}")
    source = resolve_source(cfg, args.model, args.ckpt)
    if source.kind != LAYA:
        raise ValueError(f"{args.model!r} is not a Laya model: only Laya checkpoints are exported to ONNX")
    meta_path, ident = Path(f"{out}.json"), identity(args, cfg, source.path)
    if args.reuse and (prev := _reusable(meta_path, out, ident)) is not None:
        print(f"onnx_export: reusing {out} (same checkpoint, opset, versions and check)", flush=True)
        return prev
    meta_path.unlink(missing_ok=True)
    agent = load_reference(source.path)
    exp = export(None, out, args.opset, exporter=args.exporter, agent=agent)
    print(f"onnx_export: {args.model} -> {out} ({exp['exporter']} exporter, dummy batch {exp['dummy_batch']}, opset "
          f"{exp['opset']}, attempt {exp['attempt']}/{exp['attempts_planned']}, {exp['bytes'] / 2**20:.0f} MiB, "
          f"{exp['seconds']:.0f}s)", flush=True)
    acc = _acceptance(args, cfg, out, agent) if ident["check"] else None
    meta = {"model": args.model, "ckpt": source.label, "ckpt_dir": str(source.path), "onnx_path": str(out),
            "identity": ident, "export": exp, "acceptance": acc, "passed": acc is None or bool(acc["passed"]),
            "seconds": round(time.perf_counter() - t0, 2)}
    write_json(meta_path, meta)
    return meta


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    p5 = (load_config(pre.parse_known_args(argv)[0].config).get("phase5") or {}).get("bench") or {}
    onnx_cfg = p5.get("onnx") or {}
    p = argparse.ArgumentParser(prog="python -m laya_poc.onnx_export", description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="config model key: laya | laya_ml")
    p.add_argument("--ckpt", default="hub", help="'hub' (pinned revision) or an absolute Laya checkpoint dir")
    p.add_argument("--out", required=True, help="absolute path of the .onnx file (<out>.json is written next to it)")
    p.add_argument("--opset", type=int, default=int(onnx_cfg.get("opset", 18)))
    p.add_argument("--exporter", choices=sorted(EXPORTERS), default="auto",
                   help="auto (default): upstream verbatim, then dynamo=False, then a batch-2 dummy")
    p.add_argument("--check-rows", default=None, help="JSONL rows (test_id.jsonl) for the acceptance check")
    p.add_argument("--check-n", type=int, default=int(onnx_cfg.get("check_n", 1000)), help="0 skips the check")
    p.add_argument("--batch-size", type=int, default=int(p5.get("batch_size", 32)))
    p.add_argument("--min-argmax-agree", type=float, default=float(onnx_cfg.get("min_argmax_agree", 0.999)))
    p.add_argument("--max-dp", type=float, default=float(onnx_cfg.get("max_dp", 1e-3)))
    p.add_argument("--reuse", action="store_true", help="skip when <out>.json matches this export and check")
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_cli("onnx_export", lambda: 0 if run(args)["passed"] else 1,
                   args.out if Path(args.out).is_absolute() else None)


if __name__ == "__main__":
    sys.exit(main())
