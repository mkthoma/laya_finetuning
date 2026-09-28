"""Machine facts for the CPU benchmark (design doc §7.11 / §7.13 "Hardware" column): logical CPUs, PHYSICAL cores
(the thread sweep is capped at them), CPU model, RAM, library versions, and the peak resident set size of a worker.

Everything degrades to None with the method recorded, never an exception: psutil is optional (present on Colab,
absent from the laptop venv), so physical cores fall back to /proc/cpuinfo (Linux), GetLogicalProcessorInformation
(Windows) or sysctl (macOS), and peak RSS to resource / GetProcessMemoryInfo / tracemalloc.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

from .env_check import package_version, ram_gb

_PMC_SIZE_FIELDS = ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
_RELATION_PROCESSOR_CORE = 0  # LOGICAL_PROCESSOR_RELATIONSHIP.RelationProcessorCore
_SYSCTL_TIMEOUT_S = 10
VERSION_PACKAGES = ("torch", "transformers", "onnx", "onnxruntime", "laya", "numpy", "tokenizers")


# ---------------------------------------------------------------- physical cores

def _linux_physical_cores(text: str) -> int | None:
    """Distinct (physical id, core id) pairs of a /proc/cpuinfo text; None when the file lacks core ids."""
    pairs, phys = set(), "0"
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            phys = value.strip()
        elif key == "core id":
            pairs.add((phys, value.strip()))
    return len(pairs) or None


def _windows_physical_cores() -> int | None:
    import ctypes
    from ctypes import wintypes

    class Info(ctypes.Structure):  # SYSTEM_LOGICAL_PROCESSOR_INFORMATION (the union is 16 bytes)
        _fields_ = [("mask", ctypes.c_size_t), ("relationship", wintypes.DWORD),
                    ("union", ctypes.c_ulonglong * 2)]

    get_info = ctypes.WinDLL("kernel32").GetLogicalProcessorInformation
    size = wintypes.DWORD(0)
    get_info(None, ctypes.byref(size))
    buf = (Info * (size.value // ctypes.sizeof(Info)))()
    if not size.value or not get_info(buf, ctypes.byref(size)):
        return None
    return sum(1 for x in buf if x.relationship == _RELATION_PROCESSOR_CORE) or None


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=_SYSCTL_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def physical_cores() -> tuple[int | None, str]:
    """(physical core count, how it was found)."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n), "psutil"
    except ImportError:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                return _linux_physical_cores(f.read()), "/proc/cpuinfo"
        if sys.platform == "win32":
            return _windows_physical_cores(), "GetLogicalProcessorInformation"
        if sys.platform == "darwin":
            v = _sysctl("hw.physicalcpu")
            return (int(v) if v and v.isdigit() else None), "sysctl hw.physicalcpu"
    except (OSError, ValueError, AttributeError):
        pass
    return None, "unavailable"


# ---------------------------------------------------------------- CPU model, versions, machine summary

def _linux_cpu_model() -> str | None:
    with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
        for line in f:
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    return None


def _windows_cpu_model() -> str | None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()


def cpu_model() -> str | None:
    try:
        if sys.platform.startswith("linux"):
            name = _linux_cpu_model()
        elif sys.platform == "win32":
            name = _windows_cpu_model()
        else:
            name = _sysctl("machdep.cpu.brand_string")
    except (OSError, ImportError):
        name = None
    return name or platform.processor() or None


def _windows_ram_gb() -> float | None:
    import ctypes
    from ctypes import wintypes

    class Status(ctypes.Structure):  # MEMORYSTATUSEX
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    *((name, ctypes.c_ulonglong) for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                              "ullAvailPageFile", "ullTotalVirtual",
                                                              "ullAvailVirtual", "ullAvailExtendedVirtual"))]

    st = Status()
    st.dwLength = ctypes.sizeof(st)
    ok = ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(st))
    return round(st.ullTotalPhys / 2**30, 2) if ok else None


def total_ram_gb() -> float | None:
    """Physical RAM in GiB: env_check.ram_gb (psutil / sysconf), else GlobalMemoryStatusEx on Windows."""
    gb = ram_gb()
    if gb is None and sys.platform == "win32":
        try:
            gb = _windows_ram_gb()
        except (OSError, AttributeError):
            gb = None
    return gb


def library_versions() -> dict[str, str | None]:
    return {name: package_version(name) for name in VERSION_PACKAGES}


def machine_info() -> dict[str, Any]:
    """The benchmark host: every field a results table's Hardware column needs."""
    cores, how = physical_cores()
    return {"cpu_count": os.cpu_count(), "physical_cores": cores, "physical_cores_source": how,
            "cpu_model": cpu_model(), "ram_gb": total_ram_gb(), "platform": platform.platform(),
            "python": platform.python_version(), "versions": library_versions()}


def hardware_label(m: dict[str, Any]) -> str:
    """One line for the §7.13 CPU table: model, physical cores / logical CPUs, RAM."""
    ram = f", {m['ram_gb']:g} GB RAM" if m.get("ram_gb") else ""
    return f"{m.get('cpu_model') or 'unknown CPU'} ({m.get('physical_cores') or '?'}C/{m.get('cpu_count')}T{ram})"


# ---------------------------------------------------------------- peak RSS (moved from bench_cpu, Phase 2)

def _windows_peak_bytes() -> int | None:
    """PeakWorkingSetSize via GetProcessMemoryInfo (psutil's peak_wset without psutil)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    *((name, ctypes.c_size_t) for name in _PMC_SIZE_FIELDS)]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    ok = get_info(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return int(c.PeakWorkingSetSize) if ok else None


def peak_rss_gb() -> tuple[float | None, str]:
    """(peak resident set size in GiB, how it was measured) for this process."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        if getattr(info, "peak_wset", None):
            return info.peak_wset / 2**30, "psutil.peak_wset"
    except ImportError:
        info = None
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (2**30 if sys.platform == "darwin" else 2**20), "resource.ru_maxrss"
    except ImportError:
        pass
    peak = _windows_peak_bytes()
    if peak:
        return peak / 2**30, "GetProcessMemoryInfo.PeakWorkingSetSize"
    if info is not None:
        return info.rss / 2**30, "psutil.rss (current, not peak)"
    import tracemalloc
    if tracemalloc.is_tracing():
        return tracemalloc.get_traced_memory()[1] / 2**30, "tracemalloc peak (Python heap only)"
    return None, "unavailable"
