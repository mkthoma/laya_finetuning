"""Catch Laya's per-call GPU -> CPU fallback during inference (final review: OOM CPU-fallback guard).

On a CUDA OOM inside Agent._infer (laya/agent.py at the pinned commit) Laya prints
"Warning: GPU memory exceeded during inference. Retrying this request on CPU...", answers that batch
on CPU in fp32, then moves the model back, so agent.device reads "cuda" again afterwards and a GPU
number (parity, zero-shot, export round trip) can silently come from the CPU. When the move back
fails it prints "... could not move the model back to <device> after the CPU retry ..." and stays on
CPU. That stdout line is the only trace, so the harness tees stdout around every GPU inference call:

    guard = FallbackGuard()
    with guard:
        agent.predict_batch(...)
    result["cpu_fallback"] = guard.cpu_fallback
"""
from __future__ import annotations

import contextlib
import sys
from typing import Any, Callable, TextIO

FALLBACK_MARKERS = ("GPU memory exceeded", "after the CPU retry")


class _Tee:
    """A stdout stand-in: echoes every write to the real stream and hands each complete line to `on_line`."""

    def __init__(self, echo: TextIO, on_line: Callable[[str], None]) -> None:
        self._echo, self._on_line, self._pending = echo, on_line, ""

    def write(self, s: str) -> int:
        self._echo.write(s)
        *lines, self._pending = (self._pending + s).replace("\r", "\n").split("\n")
        for line in lines:
            self._on_line(line)
        return len(s)

    def flush(self) -> None:
        self._echo.flush()

    def flush_pending(self) -> None:
        """Scan a last line that never got its newline."""
        if self._pending:
            self._on_line(self._pending)
            self._pending = ""

    def __getattr__(self, name: str) -> Any:  # encoding, isatty, fileno, ... of the real stream
        return getattr(self._echo, name)


class FallbackGuard:
    """Reusable (not nestable) context manager; `cpu_fallback` stays True once any guarded call fell back."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._active: tuple[_Tee, contextlib.redirect_stdout] | None = None

    @property
    def cpu_fallback(self) -> bool:
        return bool(self.messages)

    def _scan(self, line: str) -> None:
        if any(marker in line for marker in FALLBACK_MARKERS):
            self.messages.append(line.strip())

    def __enter__(self) -> FallbackGuard:
        if self._active is not None:
            raise RuntimeError("FallbackGuard is already active; it does not nest")
        tee = _Tee(sys.stdout, self._scan)
        redirect = contextlib.redirect_stdout(tee)
        redirect.__enter__()
        self._active = (tee, redirect)
        return self

    def __exit__(self, *exc: Any) -> None:
        tee, redirect = self._active
        self._active = None
        try:
            tee.flush_pending()
        finally:
            redirect.__exit__(*exc)
