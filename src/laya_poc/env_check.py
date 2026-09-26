"""Record and check the runtime before anything expensive runs (design doc §6.2 Phase 1 task 1).

The VS Code Colab extension can hand out a CPU (Auto Connect) or silently substitute another GPU,
and Colab's "Latest" image moves (colab-vscode §6), so the smoke test records what it actually got
and fails early on what would make every later number meaningless or the run die half-way: no CUDA
(or no kernels for the GPU) when a GPU is required, a Laya install that is not the pinned commit, and
too little free disk for the resumable checkpoints. Every error names its own fix, so the notebook
can print the CLI's message as is.

    python -m laya_poc.env_check --out <env.json> [--require-gpu] [--expect-card T4] [--config yaml]
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import card_from_gpu_name, load_config

GIB = 1024 ** 3
COLAB_MARKER = "/var/colab/hostname"  # Colab's own runtime marker (colabtools drive.py)
_NVIDIA_SMI_TIMEOUT_S = 20
_ARCH_RE = re.compile(r"^(sm|compute)_(\d{2,3})([a-z]?)$")  # torch.cuda.get_arch_list() entries
# Steps 8-10 at peak: the crash run keeps up to 3 x 5 GB resumable checkpoints, plus final/ (0.84 GB)
# and ~3 GB of Hub cache. Below DISK_MIN_GB the run would die mid-training; below DISK_WARN_GB it is tight.
DISK_MIN_GB = 20.0
DISK_WARN_GB = 30.0
NEW_T4_SERVER = "Remove Server, then Select Kernel > Colab > New Colab Server > GPU > T4, not Auto Connect"
STEP2_FIX = "re-run Step 2 (Install) and read logs/02_install_laya.log"
Problems = tuple[list[str], list[str]]  # (errors, warnings)


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def commit_from_direct_url(text: str | None) -> str | None:
    """Commit of a VCS install from PEP 610 direct_url.json; None for local-dir or unknown installs."""
    if not text:
        return None
    try:
        info = json.loads(text)
    except ValueError:
        return None
    return (info.get("vcs_info") or {}).get("commit_id")


def laya_commit() -> str | None:
    try:
        return commit_from_direct_url(md.distribution("laya").read_text("direct_url.json"))
    except md.PackageNotFoundError:
        return None


def nvidia_driver() -> str | None:
    """Host driver version from nvidia-smi (not published for Colab images); None when absent."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        res = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=_NVIDIA_SMI_TIMEOUT_S, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = res.stdout.strip().splitlines()
    return lines[0].strip() if res.returncode == 0 and lines else None


def card_or_none(gpu_name: str | None) -> str | None:
    if not gpu_name:
        return None
    try:
        return card_from_gpu_name(gpu_name)
    except ValueError:
        return None


def ram_gb() -> float | None:
    try:
        import psutil  # present on Colab
        return round(psutil.virtual_memory().total / GIB, 2)
    except ImportError:
        pass
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB, 2)
    except (AttributeError, ValueError, OSError):
        return None


def disk_free_gb(path: str | Path | None = None) -> float | None:
    path = path or ("/content" if os.path.isdir("/content") else Path.cwd())
    try:
        return round(shutil.disk_usage(path).free / GIB, 2)
    except OSError:
        return None


def torch_info() -> dict[str, Any]:
    """GPU facts from torch; every field is None/False on a CPU-only machine or without torch."""
    info: dict[str, Any] = {"torch": None, "torch_cuda": None, "cuda_available": False, "gpu_name": None,
                            "capability": None, "arch_list": [], "vram_total_gb": None}
    try:
        import torch
    except ImportError:
        return info
    info.update(torch=torch.__version__, torch_cuda=torch.version.cuda)
    info["arch_list"] = list(torch.cuda.get_arch_list()) if torch.version.cuda else []
    if not torch.cuda.is_available():
        return info
    props = torch.cuda.get_device_properties(0)
    info.update(cuda_available=True, gpu_name=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)),
                vram_total_gb=round(props.total_memory / GIB, 2))
    return info


def collect_env(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything the smoke report prints under "GPU / CC / driver" and "versions"."""
    cfg = cfg if cfg is not None else load_config()
    env = {"python": platform.python_version(), "platform": platform.platform(), **torch_info()}
    env["has_sm75"] = "sm_75" in env["arch_list"]
    env["driver"] = nvidia_driver()
    env.update({name: package_version(name) for name in ("transformers", "huggingface_hub", "duckdb")})
    env.update(laya_version=package_version("laya"), laya_commit=laya_commit(),
               laya_commit_expected=cfg["laya"]["commit"], on_colab=os.path.exists(COLAB_MARKER),
               ram_gb=ram_gb(), disk_free_gb=disk_free_gb(), card=card_or_none(env["gpu_name"]))
    return env


def arch_supported(capability: Sequence[int], arch_list: Sequence[str]) -> bool:
    """True when torch ships kernels that run on this GPU (CUDA compatibility rules):
    sm_XY SASS runs on the same major with minor >= Y (sm_86 on an L4's sm_89); an arch-specific
    sm_XYa/f only on exactly XY; compute_XY PTX is JIT-compiled for any capability >= XY."""
    dev = (int(capability[0]), int(capability[1]))
    return any(_runs_on(m.groups(), dev) for m in map(_ARCH_RE.match, map(str, arch_list)) if m)


def _runs_on(entry: tuple[str, str, str], dev: tuple[int, int]) -> bool:
    kind, num, suffix = entry
    cc = (int(num[:-1]), int(num[-1]))
    if kind == "compute":
        return not suffix and cc <= dev
    return cc == dev or (not suffix and cc[0] == dev[0] and cc[1] <= dev[1])


def evaluate_env(env: dict[str, Any], *, require_gpu: bool, expect_card: str | None) -> Problems:
    """(errors, warnings). Errors stop the notebook; warnings are recorded in the report. Each error
    carries its own fix hint."""
    found = (_gpu_problems(env, require_gpu, expect_card), _laya_problems(env), _disk_problems(env, require_gpu))
    return [e for errors, _ in found for e in errors], [w for _, warnings in found for w in warnings]


def _gpu_problems(env: dict[str, Any], require_gpu: bool, expect_card: str | None) -> Problems:
    errors: list[str] = []
    warnings: list[str] = []
    if require_gpu and not env.get("cuda_available"):
        errors.append(f"no CUDA GPU (Auto Connect gives a CPU): {NEW_T4_SERVER}")
    if env.get("cuda_available") and env.get("capability"):
        arch = "sm_%d%d" % tuple(env["capability"][:2])
        if not arch_supported(env["capability"], env.get("arch_list") or []):
            # critique §D.4: without kernels for the GPU every CUDA op fails with "no kernel image is available"
            (errors if require_gpu else warnings).append(
                f"torch {env.get('torch')} has no kernels for this GPU ({arch}; arch list {env.get('arch_list')}): "
                "Remove Server and start a New Colab Server on another runtime version, e.g. 2026.07 (picked in "
                "the server picker), or install a torch build that includes it")
    if expect_card and env.get("cuda_available") and env.get("card") != expect_card:
        warnings.append(f"got {env.get('gpu_name')} (card {env.get('card')}), expected {expect_card}; "
                        "micro-batch, precision and the VRAM budget assume a T4")
    return errors, warnings


def _laya_problems(env: dict[str, Any]) -> Problems:
    if not env.get("laya_version"):
        return [f"laya is not installed: {STEP2_FIX}"], []
    commit, expected = env.get("laya_commit"), env.get("laya_commit_expected")
    if commit and expected and commit != expected:
        return [f"laya commit {commit} != pinned {expected}: {STEP2_FIX} (it installs the pinned commit)"], []
    if not commit:
        return [], ["laya commit unknown (not a git install); cannot verify the pin"]
    return [], []


def _disk_problems(env: dict[str, Any], require_gpu: bool) -> Problems:
    """Too little disk is an error on the GPU run only: a local dry run writes tiny checkpoints."""
    free = env.get("disk_free_gb")
    if free is None or free >= DISK_WARN_GB:
        return [], []
    if free >= DISK_MIN_GB:
        return [], [f"only {free} GiB free disk (< {DISK_WARN_GB:g} GiB): tight for Steps 8-10, which keep up "
                       "to 3 x 5 GB checkpoints at once"]
    msg = (f"only {free} GiB free disk, need >= {DISK_MIN_GB:g} GiB (the crash run keeps up to 3 x 5 GB resumable "
           "checkpoints, plus final/ 0.84 GB and ~3 GB of Hub cache): delete old checkpoints (Step 13b, "
           "CLEANUP_CKPTS = True) or Remove Server and start a New Colab Server")
    return ([msg], []) if require_gpu else ([], [msg])


def run_cli(prog: str, body: Callable[[], int], detail_path: str | Path | None) -> int:
    """Run a CLI body: on an exception print ONE line and keep the traceback next to the output."""
    try:
        return body()
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, do not crash the notebook cell
        note = ""
        if detail_path is not None:
            tb_path = Path(f"{detail_path}.traceback.txt")
            try:
                tb_path.parent.mkdir(parents=True, exist_ok=True)
                tb_path.write_text(traceback.format_exc(), encoding="utf-8")
                note = f" (traceback: {tb_path})"
            except OSError:
                pass
        print(f"{prog}: error: {type(exc).__name__}: {exc}{note}", file=sys.stderr)
        return 1


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _print_summary(env: dict[str, Any], warnings: list[str]) -> None:
    gpu = f"{env['gpu_name']} cc={env['capability']} vram={env['vram_total_gb']}GiB" if env["cuda_available"] \
        else "no CUDA"
    print(f"env: python {env['python']}, torch {env['torch']} (cuda {env['torch_cuda']}), "
          f"transformers {env['transformers']}, laya {env['laya_version']}@{(env['laya_commit'] or '?')[:8]}")
    print(f"env: {gpu}, driver {env['driver']}, card {env['card']}, colab={env['on_colab']}, "
          f"ram {env['ram_gb']}GiB, disk free {env['disk_free_gb']}GiB")
    for w in warnings:
        print(f"env: WARNING: {w}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya_poc.env_check", description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="where to write env.json")
    p.add_argument("--config", default=None, help="config.yaml (default: project root)")
    p.add_argument("--require-gpu", action="store_true", help="fail when CUDA is unavailable")
    p.add_argument("--expect-card", default=None, help="warn (not fail) when the GPU is another card")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def body() -> int:
        env = collect_env(load_config(args.config))
        errors, warnings = evaluate_env(env, require_gpu=args.require_gpu, expect_card=args.expect_card)
        write_json(args.out, {**env, "errors": errors, "warnings": warnings, "passed": not errors})
        _print_summary(env, warnings)
        for e in errors:  # one line per error, each with its own fix
            print(f"env_check: FAIL: {e}", file=sys.stderr)
        if errors:
            return 1
        print(f"env_check: OK -> {args.out}")
        return 0

    return run_cli("env_check", body, args.out)


if __name__ == "__main__":
    sys.exit(main())
