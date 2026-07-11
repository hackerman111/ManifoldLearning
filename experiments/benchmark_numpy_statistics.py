from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adp import ADP, ADPConfig


@dataclass(frozen=True, slots=True)
class StatisticsBenchmarkCase:
    name: str
    n: int
    d: int
    n_centers: int
    n_directions: int
    h_multiplier: float


DEFAULT_CASES = (
    StatisticsBenchmarkCase("sparser", 1000, 100, 150, 16, 0.75),
    StatisticsBenchmarkCase("primary", 1000, 100, 150, 16, 1.00),
    StatisticsBenchmarkCase("denser", 1000, 100, 150, 16, 1.25),
)


def run_case(
    case: StatisticsBenchmarkCase,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    config = ADPConfig(
        n_centers=case.n_centers,
        n_directions=case.n_directions,
        min_neighbors=16.0,
        chunk_size=32,
        kernel="epanechnikov",
        backend="numpy",
        dtype="float64",
        center_noise_scale=0.1,
        use_neighbor_index=False,
        show_progress=False,
        random_state=seed,
    )
    model = ADP.create("new", config)
    data = model.generate_data(
        n=case.n,
        d=case.d,
        noise=0.05,
        sigma_x=1.0,
        corr=0.0,
        link="linear",
    )
    if data.directions is None:
        raise RuntimeError("new ADP benchmark requires random directions")

    selected_h = model._select_isotropic_bandwidth(data.X, data.centers, None)
    h = float(selected_h * case.h_multiplier)
    norm2 = model._cached_pairwise_norm2(data.X, data.centers)
    active_fraction = float(np.mean(np.asarray(norm2) / (h * h) < 1.0))

    statistics_result = model._compute_statistics(
        data.X,
        data.y,
        data.centers,
        h,
        data.beta,
        data.directions,
        None,
    )
    times: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        statistics_result = model._compute_statistics(
            data.X,
            data.y,
            data.centers,
            h,
            data.beta,
            data.directions,
            None,
        )
        times.append(time.perf_counter() - started)

    tracemalloc.start()
    model._compute_statistics(
        data.X,
        data.y,
        data.centers,
        h,
        data.beta,
        data.directions,
        None,
    )
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "name": case.name,
        "case": asdict(case),
        "shape": {
            "n": case.n,
            "d": case.d,
            "J": case.n_centers,
            "P": case.n_directions,
        },
        "h": h,
        "active_fraction": active_fraction,
        "repetitions": repetitions,
        "times_sec": times,
        "median_sec": float(statistics.median(times)),
        "min_sec": float(min(times)),
        "peak_memory_kib": float(peak_memory_bytes / 1024.0),
        "statistics_shapes": {
            "imav": list(statistics_result.imav.shape),
            "S": list(statistics_result.S.shape) if statistics_result.S is not None else None,
            "U": list(statistics_result.U.shape) if statistics_result.U is not None else None,
            "N": list(statistics_result.N.shape) if statistics_result.N is not None else None,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark warmed NumPy ADP statistics.")
    parser.add_argument(
        "--case",
        choices=("all",) + tuple(case.name for case in DEFAULT_CASES),
        default="all",
    )
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = (
        DEFAULT_CASES
        if args.case == "all"
        else tuple(case for case in DEFAULT_CASES if case.name == args.case)
    )
    records = [
        run_case(case, repetitions=args.repetitions, seed=args.seed)
        for case in cases
    ]
    payload = {"records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
