import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


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
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    current, _ = tracemalloc.get_traced_memory()
    return _Tracker(time.perf_counter(), current, owns_tracing, peak_memory=current)


@contextmanager
def track_stage(tracker: _Tracker, name: str):
    if tracker.result is not None:
        raise RuntimeError("tracking has already finished")

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
    labels = {
        "initialization": "инициализация",
        "directions": "направления",
        "statistics": "статистики",
        "solver": "солвер",
        "update": "обновление",
    }
    lines = [
        "этап             время, с  доля времени  память, MiB  доля памяти",
    ]
    for name, values in profile["stages"].items():
        lines.append(
            f"{labels.get(name, name):<16}"
            f"{values['time_seconds']:>9.6f}"
            f"{values['time_fraction']*100:>14.2f}"
            f"{values['memory_bytes'] / 2**20:>13.4f}"
            f"{values['memory_fraction']*100:>13.2f}"
        )
    lines.append(
        f"итого: {profile['total_time_seconds']:.6f} с; "
        f"Максимум памяти: {profile['max_memory'] / 2**20:.4f} MiB; "
        f"пик fit: {profile['peak_memory_bytes'] / 2**20:.4f} MiB"
    )
    return "\n".join(lines)
