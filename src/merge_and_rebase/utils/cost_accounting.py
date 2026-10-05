"""Uniform per-phase cost accounting for rebase methods (Direct Residual, THESEUS, BiCo).

Every method's cost is reported under exactly three phases:

* ``activation_collection`` -- forward (and, for gradient statistics, backward) passes
  that capture activations or gradients;
* ``transformation`` -- turning captured quantities into the transform: accumulating
  statistics (cross-covariances, ridge Grams), Procrustes/ridge solves, target
  construction, and any mount-and-measure sweeps used to rescale the result;
* ``transport`` -- applying the result to produce the target task vector.

A `PhaseCostRecorder` is activated for one task with `recording(recorder)`; instrumented
code marks its work with `cost_phase(name)`, which is a no-op when no recorder is active
(so the default path and every numerical result are unchanged -- the recorder only
synchronizes and reads counters). Phases never nest: entering a phase closes the current
segment of whichever phase was open and reopens it on exit, so each instant of wall time is
charged to exactly one phase, and interleaved work (e.g. a streaming loop alternating
capture and accumulation per batch) is split per segment.

Per phase the recorder reports the summed wall seconds (CUDA synchronized at every segment
boundary), the CUDA peak allocated bytes (``torch.cuda.reset_peak_memory_stats`` at segment
start, max over segments) and the host peak RSS. The host peak is kernel-exact: the RSS
high-water mark is reset at segment start by writing ``5`` to ``/proc/self/clear_refs`` and
read back as ``VmHWM`` at segment end (absolute bytes, comparable to SLURM MaxRSS, and the
delta over the RSS at segment start). If that is unavailable it falls back to a ~20 ms
sampler thread over ``VmRSS``; the method used is recorded. Once the high-water mark has
been reset, ``getrusage(RUSAGE_SELF).ru_maxrss`` is no longer a lifetime maximum for this
process; job-level lifetime peaks come from SLURM (``sacct MaxRSS``) instead. Only the
calling process is measured (DataLoader worker processes are not).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import threading
import time
from typing import Any

import torch

PHASES = ("activation_collection", "transformation", "transport")

_ACTIVE: contextvars.ContextVar[PhaseCostRecorder | None] = contextvars.ContextVar("phase_cost_recorder", default=None)


def _read_status_kb(field: str) -> int | None:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _reset_hwm() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


class _RssSampler:
    """Fallback host-peak sampler (VmRSS every ~20 ms) when VmHWM cannot be reset."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.peak_kb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.peak_kb = _read_status_kb("VmRSS") or 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            rss = _read_status_kb("VmRSS")
            if rss is not None and rss > self.peak_kb:
                self.peak_kb = rss

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        rss = _read_status_kb("VmRSS") or 0
        return max(self.peak_kb, rss)


class PhaseCostRecorder:
    """Accumulates wall time and peak memory per phase for one task (see module docstring)."""

    def __init__(self, device: Any = None) -> None:
        self.device = device
        self._cuda = torch.cuda.is_available() and device is not None and str(device) != "cpu"
        self.phases: dict[str, dict[str, Any]] = {
            name: {
                "seconds": 0.0,
                "cuda_peak_allocated_bytes": 0.0,
                "host_peak_rss_bytes": 0.0,
                "host_peak_rss_delta_bytes": 0.0,
                "segments": 0,
            }
            for name in PHASES
        }
        self.host_peak_method = "vmhwm_reset"
        self._stack: list[str] = []
        self._segment: dict[str, Any] | None = None
        self._locked = 0  # > 0 inside an exclusive phase: inner phases are charged to it
        self._suspended = 0  # > 0 inside cost_excluded(): nothing is charged
        self._sampler: _RssSampler | None = None
        # Every closed segment's (cuda_peak_bytes, host_peak_bytes), phases and the
        # "_unattributed"/"_excluded" filler segments alike, so a bracket can read the exact
        # peak since its own mark (see mark()/peak_since()) even though segments reset the
        # CUDA/host high-water marks.
        self._timeline: list[tuple[float, float]] = []
        self.filler_seconds = {"_unattributed": 0.0, "_excluded": 0.0}
        self._started = time.perf_counter()
        self._open_segment("_unattributed")

    # -- segments --------------------------------------------------------------
    def _sync(self) -> None:
        if self._cuda:
            torch.cuda.synchronize()

    def _open_segment(self, name: str) -> None:
        self._sync()
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()
        baseline_kb = _read_status_kb("VmRSS") or 0
        if self.host_peak_method == "vmhwm_reset" and not _reset_hwm():
            self.host_peak_method = "sampler"
        if self.host_peak_method == "sampler":
            self._sampler = _RssSampler()
            self._sampler.start()
        self._segment = {"name": name, "t0": time.perf_counter(), "baseline_kb": baseline_kb}

    def _close_segment(self) -> None:
        segment = self._segment
        if segment is None:
            return
        self._sync()
        seconds = time.perf_counter() - segment["t0"]
        if self.host_peak_method == "sampler" and self._sampler is not None:
            peak_kb = self._sampler.stop()
            self._sampler = None
        else:
            peak_kb = _read_status_kb("VmHWM") or 0
        cuda_peak = float(torch.cuda.max_memory_allocated()) if self._cuda else 0.0
        host_peak = float(peak_kb * 1024)
        self._timeline.append((cuda_peak, host_peak))
        self._segment = None
        if segment["name"] in self.filler_seconds:
            self.filler_seconds[segment["name"]] += seconds
            return
        stats = self.phases[segment["name"]]
        stats["seconds"] += seconds
        stats["segments"] += 1
        stats["cuda_peak_allocated_bytes"] = max(stats["cuda_peak_allocated_bytes"], cuda_peak)
        stats["host_peak_rss_bytes"] = max(stats["host_peak_rss_bytes"], host_peak)
        stats["host_peak_rss_delta_bytes"] = max(
            stats["host_peak_rss_delta_bytes"], float(max(peak_kb - segment["baseline_kb"], 0) * 1024)
        )

    def _current_name(self) -> str:
        if self._suspended:
            return "_excluded"
        return self._stack[-1] if self._stack else "_unattributed"

    def _reopen(self) -> None:
        self._open_segment(self._current_name())

    def mark(self) -> int:
        """Start a bracket: returns a timeline index for `cuda_peak_since`/`host_peak_since`."""
        self._close_segment()
        self._reopen()
        return len(self._timeline)

    def peaks_since(self, mark: int) -> tuple[float, float]:
        self._close_segment()
        self._reopen()
        window = self._timeline[mark:]
        if not window:
            return 0.0, 0.0
        return max(c for c, _ in window), max(h for _, h in window)

    def cuda_peak_since(self, mark: int) -> float:
        """Exact CUDA peak allocated bytes since `mark()` (0 on CPU)."""
        return self.peaks_since(mark)[0]

    def host_peak_since(self, mark: int) -> float:
        """Exact host peak RSS bytes since `mark()`."""
        return self.peaks_since(mark)[1]

    @contextlib.contextmanager
    def phase(self, name: str, *, exclusive: bool = False):
        if name not in self.phases:
            raise ValueError(f"unknown cost phase {name!r}; expected one of {PHASES}")
        if self._suspended or self._locked:
            # Excluded work is charged to nothing; inside an exclusive phase every inner
            # phase is charged to the exclusive one (e.g. tv_scaling's measurement
            # forward passes count as transformation, not collection).
            yield
            return
        if exclusive:
            with self.phase(name):
                self._locked += 1
                try:
                    yield
                finally:
                    self._locked -= 1
            return
        if self._stack and self._stack[-1] == name:
            # Re-entering the phase that is already open (e.g. a helper that marks itself
            # as transformation, called from transformation code) extends the segment.
            self._stack.append(name)
            try:
                yield
            finally:
                self._stack.pop()
            return
        self._close_segment()
        self._stack.append(name)
        self._open_segment(name)
        try:
            yield
        finally:
            self._close_segment()
            self._stack.pop()
            self._reopen()

    @contextlib.contextmanager
    def excluded(self):
        """Charge the enclosed work to no phase (analysis-only diagnostics)."""
        if self._suspended or self._locked:
            yield
            return
        self._close_segment()
        self._suspended += 1
        self._reopen()
        try:
            yield
        finally:
            self._close_segment()
            self._suspended -= 1
            self._reopen()

    def close(self) -> None:
        """Close the open segment (end of recording); summary() stays valid."""
        self._close_segment()

    def summary(self) -> dict[str, Any]:
        if self._segment is not None:
            self._close_segment()
            self._reopen()
        total = time.perf_counter() - self._started
        return {
            "phases": {name: dict(stats) for name, stats in self.phases.items()},
            "host_peak_method": self.host_peak_method,
            "total_seconds": total,
            "unattributed_seconds": self.filler_seconds["_unattributed"],
            "excluded_seconds": self.filler_seconds["_excluded"],
            "max_host_peak_rss_bytes": max(p["host_peak_rss_bytes"] for p in self.phases.values()),
            "max_cuda_peak_allocated_bytes": max(p["cuda_peak_allocated_bytes"] for p in self.phases.values()),
        }


@contextlib.contextmanager
def recording(recorder: PhaseCostRecorder):
    """Make ``recorder`` the active recorder for `cost_phase` calls in this context."""
    token = _ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.reset(token)
        recorder.close()


@contextlib.contextmanager
def cost_phase(name: str, *, exclusive: bool = False):
    """Charge the enclosed work to phase ``name`` of the active recorder (no-op if none).

    ``exclusive=True`` charges everything inside, including code that marks itself with
    another phase, to ``name``.
    """
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
        return
    with recorder.phase(name, exclusive=exclusive):
        yield


@contextlib.contextmanager
def cost_excluded():
    """Exclude the enclosed work from every phase of the active recorder (no-op if none)."""
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
        return
    with recorder.excluded():
        yield


def cost_phase_decorator(name: str, *, exclusive: bool = False):
    """Decorator form of `cost_phase` (no-op when no recorder is active)."""

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if _ACTIVE.get() is None:
                return fn(*args, **kwargs)
            with cost_phase(name, exclusive=exclusive):
                return fn(*args, **kwargs)

        return inner

    return wrap
