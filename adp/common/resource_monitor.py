from __future__ import annotations

import os
import resource
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self


_MIB = 1024.0 * 1024.0


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Aggregated wall-clock time and resident-memory samples."""

    elapsed_sec: float
    rss_start_mib: float
    rss_min_mib: float
    rss_mean_mib: float
    rss_max_mib: float
    rss_peak_delta_mib: float
    samples: int
    source: str

    def to_dict(self, prefix: str) -> dict[str, float | int | str]:
        """Return flat columns suitable for experiment CSV tables."""

        return {
            f"{prefix}_time_sec": self.elapsed_sec,
            f"{prefix}_rss_start_mib": self.rss_start_mib,
            f"{prefix}_rss_min_mib": self.rss_min_mib,
            f"{prefix}_rss_mean_mib": self.rss_mean_mib,
            f"{prefix}_rss_max_mib": self.rss_max_mib,
            f"{prefix}_rss_peak_delta_mib": self.rss_peak_delta_mib,
            f"{prefix}_memory_samples": self.samples,
            f"{prefix}_memory_source": self.source,
        }


class ResourceMonitor:
    """Sample process RSS in a background thread and aggregate it online."""

    def __init__(self, *, sample_interval_sec: float = 0.01) -> None:
        if sample_interval_sec <= 0.0:
            raise ValueError("sample_interval_sec must be positive")
        self.sample_interval_sec = float(sample_interval_sec)
        self._read_rss_bytes, self._source = _rss_reader()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._start_rss_mib = 0.0
        self._count = 0
        self._sum_mib = 0.0
        self._min_mib = float("inf")
        self._max_mib = 0.0
        self._usage: ResourceUsage | None = None

    @property
    def usage(self) -> ResourceUsage:
        if self._usage is None:
            raise RuntimeError("ResourceMonitor has not finished")
        return self._usage

    def __enter__(self) -> Self:
        if self._thread is not None:
            raise RuntimeError("ResourceMonitor cannot be reused while active")
        self._stop.clear()
        self._usage = None
        self._count = 0
        self._sum_mib = 0.0
        self._min_mib = float("inf")
        self._max_mib = 0.0
        self._started = time.perf_counter()
        self._start_rss_mib = self._sample()
        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="adp-resource-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        elapsed = time.perf_counter() - self._started
        with self._lock:
            self._usage = ResourceUsage(
                elapsed_sec=elapsed,
                rss_start_mib=self._start_rss_mib,
                rss_min_mib=self._min_mib,
                rss_mean_mib=self._sum_mib / self._count,
                rss_max_mib=self._max_mib,
                rss_peak_delta_mib=max(0.0, self._max_mib - self._start_rss_mib),
                samples=self._count,
                source=self._source,
            )
        self._thread = None

    def _sampling_loop(self) -> None:
        while not self._stop.wait(self.sample_interval_sec):
            self._sample()

    def _sample(self) -> float:
        rss_mib = float(self._read_rss_bytes()) / _MIB
        with self._lock:
            self._count += 1
            self._sum_mib += rss_mib
            self._min_mib = min(self._min_mib, rss_mib)
            self._max_mib = max(self._max_mib, rss_mib)
        return rss_mib


def _rss_reader() -> tuple[Callable[[], int], str]:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return lambda: int(process.memory_info().rss), "psutil"
    except (ImportError, OSError):
        pass

    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        page_size = os.sysconf("SC_PAGE_SIZE")

        def read_procfs() -> int:
            fields = statm_path.read_text(encoding="ascii").split()
            return int(fields[1]) * page_size

        return read_procfs, "procfs"

    scale = 1 if sys.platform == "darwin" else 1024
    return (
        lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale,
        "resource",
    )
