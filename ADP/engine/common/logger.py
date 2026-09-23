"""Профилирование времени и Python-аллокаций по этапам ADP."""

# ruff: noqa: RUF002

from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import utils


@dataclass
class _Tracker:
    started_at: float
    baseline_memory: int
    owns_tracing: bool
    times: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    memory: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    peak_memory: int = 0
    result: dict | None = None


def start_tracking() -> _Tracker:
    """Начать сбор времени и tracemalloc-метрик для последовательности этапов."""
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    current, _ = tracemalloc.get_traced_memory()
    return _Tracker(time.perf_counter(), current, owns_tracing, peak_memory=current)


@contextmanager
def track_stage(tracker: _Tracker, name: str):
    """Измерить один именованный этап и добавить его метрики в tracker."""
    utils.require(
        tracker.result is None,
        "tracking has already finished",
        RuntimeError,
    )

    start_memory, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    started_at = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started_at
        _, peak_memory = tracemalloc.get_traced_memory()
        tracker.times[name] += elapsed
        tracker.memory[name] += max(0, peak_memory - start_memory)
        tracker.peak_memory = max(tracker.peak_memory, peak_memory)


def finish_tracking(tracker: _Tracker) -> dict:
    """Закрыть tracker и вернуть итоговые stage/time/memory-диагностики."""
    if tracker.result is not None:
        return tracker.result

    _, peak_memory = tracemalloc.get_traced_memory()
    tracker.peak_memory = max(tracker.peak_memory, peak_memory)
    total_time = time.perf_counter() - tracker.started_at
    measured_time = sum(tracker.times.values())
    measured_memory = sum(tracker.memory.values())
    max_memorpy = max(tracker.memory.values())
    stages = {}
    for name in tracker.times:
        stages[name] = {
            "time_seconds": tracker.times[name],
            "time_fraction": (
                tracker.times[name] / measured_time if measured_time else 0.0
            ),
            "memory_bytes": tracker.memory[name],
            "memory_fraction": (
                tracker.memory[name] / measured_memory if measured_memory else 0.0
            ),
        }

    tracker.result = {
        "stages": stages,
        "total_time_seconds": total_time,
        "stage_memory_bytes": measured_memory,
        "peak_memory_bytes": max(0, tracker.peak_memory - tracker.baseline_memory),
        "max_memory": max_memorpy,
    }
    if tracker.owns_tracing:
        tracemalloc.stop()
    return tracker.result


def format_profile(profile: dict) -> str:
    """Преобразовать результат ``finish_tracking`` в компактный текстовый отчёт."""
    labels = {
        "initialization": "инициализация",
        "directions": "направления",
        "statistics": "статистики",
        "solver": "солвер",
        "update": "обновление",
    }
    lines = [
        "этап             время, s  доля времени  память, MiB  доля памяти",
    ]
    for name, values in profile["stages"].items():
        lines.append(
            f"{labels.get(name, name):<16}"
            f"{values['time_seconds']:>9.6f}"
            f"{values['time_fraction'] * 100:>14.2f}"
            f"{values['memory_bytes'] / 2**20:>13.4f}"
            f"{values['memory_fraction'] * 100:>13.2f}"
        )
    lines.append(
        f"итого: {profile['total_time_seconds']:.6f} s; "
        f"Максимум памяти: {profile['max_memory'] / 2**20:.4f} MiB; "
        f"пик fit: {profile['peak_memory_bytes'] / 2**20:.4f} MiB"
    )
    return "\n".join(lines)


class IndexProfiler:
    """Общий stage-profiler CLI и моделей без зависимости engine от CLI."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, float]] = {}
        self._started = time.perf_counter()
        self._owns_trace = not tracemalloc.is_tracing()
        if self._owns_trace:
            tracemalloc.start()
        self._baseline, _ = tracemalloc.get_traced_memory()
        self._peak = self._baseline
        self._stage_memory: dict[str, float] = defaultdict(float)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Записать время и peak; для прежнего model profile также сумму peaks."""
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        try:
            yield
        finally:
            _, peak = tracemalloc.get_traced_memory()
            self._peak = max(self._peak, peak)
            self._stage_memory[name] += max(0, peak - current)
            record = self.records.setdefault(
                name,
                {"time_seconds": 0.0, "traced_peak_bytes": 0.0},
            )
            record["time_seconds"] += time.perf_counter() - started
            record["traced_peak_bytes"] = max(
                record["traced_peak_bytes"],
                float(max(0, peak - current)),
            )

    def finish(self) -> None:
        """Закрыть измерение, сохранив формат stage records CLI."""
        peak = max(
            (record["traced_peak_bytes"] for record in self.records.values()),
            default=0.0,
        )
        self.records["total"] = {
            "time_seconds": time.perf_counter() - self._started,
            "traced_peak_bytes": peak,
        }
        if self._owns_trace:
            tracemalloc.stop()

    def model_profile(self) -> dict:
        """Сохранить публичный формат profile_ для format_profile моделей."""
        records = {name: row for name, row in self.records.items() if name != "total"}
        measured_time = sum(row["time_seconds"] for row in records.values())
        measured_memory = sum(self._stage_memory.values())
        return {
            "stages": {
                name: {
                    "time_seconds": row["time_seconds"],
                    "time_fraction": row["time_seconds"] / measured_time
                    if measured_time
                    else 0.0,
                    "memory_bytes": self._stage_memory[name],
                    "memory_fraction": self._stage_memory[name] / measured_memory
                    if measured_memory
                    else 0.0,
                }
                for name, row in records.items()
            },
            "total_time_seconds": self.records["total"]["time_seconds"],
            "stage_memory_bytes": measured_memory,
            "peak_memory_bytes": max(0, self._peak - self._baseline),
            "max_memory": max(self._stage_memory.values(), default=0.0),
        }
