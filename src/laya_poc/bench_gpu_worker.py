"""The GPU-benchmark worker: ONE model in a fresh process, batch-1 latency once, then each batch size (bench_gpu.py).

Timing core: bench_worker.time_latency (cold start = seconds to a ready-to-predict runtime on the device, weights on
local disk, no network; one recorded first call; `warmup` batch-1 calls; batch-1 latency p50/p95/mean/max over the
latency STRING states) and bench_worker.time_batch (ONE batched call over the batch states for records/s). Batch-1
latency is measured ONCE per model and repeated on each of its (model, batch size) rows. Per batch size: the CUDA
cache emptied and the peak-memory counters reset, an untimed warm-up call over the first BATCH_WARMUP_BATCHES
batches (new batch shapes load kernels lazily on CUDA), then the timed call, then the peak VRAM (max allocated and
reserved, GiB) of that batch size. The batch-1 phase's peak is read before the first reset (a fresh process counts
from zero); nothing touches CUDA before the load, so the cold start includes the CUDA context.
Runtimes, end to end from the state string to probabilities on the host (tokenisation and decoding included):
- `torch` (Laya): loaded as evaluation loads it on Colab (evaluate.load_checked: hub.load_agent(path, device)), with
  the Agent's own mixed precision on CUDA (autocast fp16 below compute capability 8, else LAYA_CUDA_AMP or the
  checkpoint's amp_dtype; the notebooks set LAYA_CUDA_AMP=fp16); batch-1 `agent.predict`, batched
  `agent.predict_batch(batch_size, sort_by_length=True)`, config model.<key>.max_len / head_max_len;
- `hf` (the B4 small encoders): bench_hf.load_model (fp32 weights) moved to the device and scored as Phase 4 scored
  them (small_encoder_data.score_states with train_config's max_length and fp16 = config fp16 AND cuda: fp16
  autocast on CUDA), batch-1 = one state, batched = length-sorted batches of `batch_size`.
Every predict call is wrapped in torch.cuda.synchronize() before and after (outputs already come back to the host;
the sync makes the boundary explicit). Laya answers a CUDA OOM by retrying on the CPU and says so on stdout only: a
FallbackGuard turns that into an error for the batch size (or the whole model, in the latency phase), never a
number. Output: one MARKER line {"results": [row per batch size], "errors": [{batch_size, error}]}.
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .bench_machine import peak_rss_gb
from .bench_worker import MARKER, time_batch, time_latency
from .env_check import nvidia_driver
from .fallback_guard import FallbackGuard

GIB = 2 ** 30
BATCH_WARMUP_BATCHES = 3  # untimed batches per batch size before its timed call

__all__ = ["MARKER", "GpuRuntime", "measure_model", "worker_main"]


@dataclass(frozen=True)
class GpuRuntime:
    predict_one: Callable[[str], Any]
    predict_batch: Callable[[list[str], int], Any]   # (states, batch_size)
    info: dict[str, Any] = field(default_factory=dict)


def _dtype(value: Any) -> str:
    return str(value).replace("torch.", "")


def cuda_sync(device: str) -> Callable[[], None]:
    if device != "cuda":
        return lambda: None
    import torch

    return torch.cuda.synchronize


def synced(fn: Callable[..., Any], sync: Callable[[], None]) -> Callable[..., Any]:
    """fn with the device synchronised before (nothing queued leaks in) and after (all its work is done)."""
    def call(*args: Any, **kwargs: Any) -> Any:
        sync()
        out = fn(*args, **kwargs)
        sync()
        return out
    return call


# ---------------------------------------------------------------- VRAM and GPU facts

def reset_vram(device: str) -> None:
    if device != "cuda":
        return
    import torch

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_vram(device: str, prefix: str = "") -> dict[str, float | None]:
    """Peak allocated / reserved CUDA memory in GiB since the last reset (None off CUDA)."""
    if device != "cuda":
        return {f"{prefix}peak_vram_gb": None, f"{prefix}peak_vram_reserved_gb": None}
    import torch

    return {f"{prefix}peak_vram_gb": round(torch.cuda.max_memory_allocated() / GIB, 3),
            f"{prefix}peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / GIB, 3)}


def gpu_facts() -> dict[str, Any]:
    """The GPU this process sees: name, compute capability, total VRAM (GiB), CUDA/cuDNN, SMs, driver."""
    import torch

    facts: dict[str, Any] = {"gpu": None, "capability": None, "vram_total_gb": None, "cuda": torch.version.cuda,
                             "cudnn": None, "multiprocessors": None, "driver": nvidia_driver()}
    if not torch.cuda.is_available():
        return facts
    major, minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    return {**facts, "gpu": torch.cuda.get_device_name(0), "capability": f"{major}.{minor}",
            "vram_total_gb": round(props.total_memory / GIB, 2), "cudnn": torch.backends.cudnn.version(),
            "multiprocessors": getattr(props, "multi_processor_count", None)}


# ---------------------------------------------------------------- runtimes

def laya_gpu_loader(path: Path, question: dict, *, device: str, max_len: int,
                    head_max_len: int) -> Callable[[], GpuRuntime]:
    def load() -> GpuRuntime:
        import torch
        from .evaluate import load_checked

        agent = load_checked(Path(path), device)
        budget, sync = {"max_len": max_len, "head_max_len": head_max_len}, cuda_sync(device)
        return GpuRuntime(
            predict_one=synced(lambda s: agent.predict(s, question, **budget), sync),
            predict_batch=synced(lambda ss, bs: agent.predict_batch(ss, question, batch_size=bs, sort_by_length=True,
                                                                    **budget), sync),
            info={"backend": "torch", "framework": "pytorch", "device": agent.device.type,
                  "amp": bool(agent.amp_enabled), "dtype": _dtype(agent.dtype),
                  "param_dtype": _dtype(next(agent.model.parameters()).dtype),
                  "laya_cuda_amp": os.environ.get("LAYA_CUDA_AMP"), "max_len": max_len,
                  "head_max_len": head_max_len, "torch": torch.__version__})
    return load


def hf_gpu_loader(cfg: dict, key: str, path: Path, *, device: str) -> Callable[[], GpuRuntime]:
    def load() -> GpuRuntime:
        import torch
        from .bench_hf import HEAD_SEED, load_model
        from .small_encoder_data import score_states
        from .small_encoder_train import train_config

        tc = train_config(cfg, seed=HEAD_SEED, device=device)
        model, tok = load_model(cfg, key, Path(path))
        model.to(device)
        param = next(model.parameters())
        if param.device.type != device:
            raise RuntimeError(f"{key} landed on {param.device.type}, not {device}")
        kw, sync = {"max_length": tc.max_length, "device": device, "fp16": tc.fp16}, cuda_sync(device)
        attn = getattr(model.config, "_attn_implementation", None)
        return GpuRuntime(
            predict_one=synced(lambda s: score_states(model, tok, [s], batch_size=1, **kw), sync),
            predict_batch=synced(lambda ss, bs: score_states(model, tok, ss, batch_size=bs, **kw), sync),
            info={"backend": "hf", "framework": "pytorch", "library": "transformers", "device": device,
                  "amp": tc.fp16, "dtype": "float16" if tc.fp16 else "float32", "param_dtype": _dtype(param.dtype),
                  "max_length": tc.max_length, "attn_implementation": attn,
                  "num_labels": int(model.config.num_labels), "torch": torch.__version__})
    return load


# ---------------------------------------------------------------- measurement

def _fallback_error(where: str) -> RuntimeError:
    return RuntimeError(f"Laya fell back to the CPU {where} (GPU out of memory): its timing would be a CPU number; "
                        "use smaller --batch-sizes")


def time_batch_size(rt: GpuRuntime, batch: Sequence[str], batch_size: int, device: str) -> dict[str, Any]:
    """Warm-up then ONE timed batched call at `batch_size`, with that batch size's peak VRAM."""
    reset_vram(device)
    run = lambda ss: rt.predict_batch(ss, batch_size)  # noqa: E731
    warm = list(batch[:batch_size * BATCH_WARMUP_BATCHES])
    guard = FallbackGuard()
    with guard:
        run(warm)
        timed = time_batch(run, batch)
    if guard.cpu_fallback:
        raise _fallback_error(f"at batch size {batch_size}")
    return {**timed, "batch_warmup_rows": len(warm), **peak_vram(device)}


def measure_model(load: Callable[[], GpuRuntime], latency: Sequence[str], batch: Sequence[str], *,
                  batch_sizes: Sequence[int], warmup: int, device: str) -> dict[str, list[dict]]:
    """{"results": one row per batch size (the shared batch-1 numbers + its throughput), "errors": [...]}. A batch
    size that fails (OOM, CPU fallback) is an error row; a failure while loading or at batch 1 raises."""
    guard = FallbackGuard()
    with guard:
        rt, lat = time_latency(load, latency, warmup)
    if guard.cpu_fallback:
        raise _fallback_error("during the batch-1 latency")
    b1 = peak_vram(device, prefix="b1_")
    rss, how = peak_rss_gb()
    gpu = gpu_facts() if device == "cuda" else {}
    shared = {**lat, **b1, "device": device, **rt.info, "gpu": gpu.get("gpu"), "capability": gpu.get("capability"),
              "cpu_fallback": False, "peak_rss_gb": None if rss is None else round(rss, 3), "peak_rss_source": how}
    results, errors = [], []
    for bs in batch_sizes:
        try:
            results.append({"batch_size": bs, **shared, **time_batch_size(rt, batch, bs, device)})
        except Exception as exc:  # noqa: BLE001 - one batch size failing (OOM) must not lose the others
            errors.append({"batch_size": bs, "error": f"{type(exc).__name__}: {exc}"})
    return {"results": results, "errors": errors}


# ---------------------------------------------------------------- worker entry (called by bench_gpu.main)

def _loader(args: Any, cfg: dict, source: Any) -> Callable[[], GpuRuntime]:
    from .bench_sources import LAYA
    from .labels import question

    if source.kind != LAYA:
        return hf_gpu_loader(cfg, source.model, source.path, device=args.device)
    mc = cfg["model"][source.model]
    return laya_gpu_loader(source.path, question(cfg["labels"]["scheme"]), device=args.device,
                           max_len=int(mc["max_len"]), head_max_len=int(mc["head_max_len"]))


def worker_main(args: Any, cfg: dict, load_states: Callable[[Any, int], list[str]]) -> int:
    """Measure one model (args.models[0], args.ckpt[0] a local dir) and print its rows as one MARKER line."""
    from .bench_sources import resolve_source

    source = resolve_source(cfg, args.models[0], args.ckpt[0])
    states = load_states(args.rows, max(args.latency_n, args.batch_n))
    out = measure_model(_loader(args, cfg, source), states[:args.latency_n], states[:args.batch_n],
                        batch_sizes=args.batch_sizes, warmup=args.warmup, device=args.device)
    extra = {"pid": os.getpid(), "platform": platform.platform(), "cpu_count": os.cpu_count()}
    payload = {"results": [{**r, **extra} for r in out["results"]], "errors": out["errors"]}
    print(MARKER + json.dumps(payload), flush=True)
    return 0
