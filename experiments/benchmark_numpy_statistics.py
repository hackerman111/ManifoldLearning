from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adp import ADP, ADPConfig
from adp.common.experiment_log import CSVTable
from adp.common.resource_monitor import ResourceMonitor, ResourceUsage


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
    statistics_workers: int = 1,
) -> list[dict[str, object]]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    config = ADPConfig(
        n_centers=case.n_centers,
        n_directions=case.n_directions,
        min_neighbors=16.0,
        chunk_size=32,
        statistics_workers=statistics_workers,
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
    records: list[dict[str, object]] = []
    for repetition in range(repetitions):
        monitor = ResourceMonitor()
        with monitor:
            statistics_result = model._compute_statistics(
                data.X,
                data.y,
                data.centers,
                h,
                data.beta,
                data.directions,
                None,
            )
        records.append(
            {
                "name": case.name,
                "n": case.n,
                "d": case.d,
                "n_centers": case.n_centers,
                "n_directions": case.n_directions,
                "h_multiplier": case.h_multiplier,
                "h": h,
                "active_fraction": active_fraction,
                "repetition": repetition,
                "repetitions": repetitions,
                "statistics_workers": statistics_workers,
                **_usage_record(monitor.usage),
                "imav_rows": int(statistics_result.imav.shape[0]),
                "imav_cols": int(statistics_result.imav.shape[1]),
                "s_rows": int(statistics_result.S.shape[0]) if statistics_result.S is not None else 0,
                "s_cols": int(statistics_result.S.shape[1]) if statistics_result.S is not None else 0,
                "u_rows": int(statistics_result.U.shape[0]) if statistics_result.U is not None else 0,
                "u_cols": int(statistics_result.U.shape[1]) if statistics_result.U is not None else 0,
                "u_depth": int(statistics_result.U.shape[2]) if statistics_result.U is not None else 0,
                "n_rows": int(statistics_result.N.shape[0]) if statistics_result.N is not None else 0,
            }
        )
    times = [float(record["elapsed_sec"]) for record in records]
    for record in records:
        record["median_sec"] = float(statistics.median(times))
        record["min_sec"] = float(min(times))
    return records


def _usage_record(usage: ResourceUsage) -> dict[str, object]:
    return {
        "elapsed_sec": usage.elapsed_sec,
        "rss_start_mib": usage.rss_start_mib,
        "rss_min_mib": usage.rss_min_mib,
        "rss_mean_mib": usage.rss_mean_mib,
        "rss_max_mib": usage.rss_max_mib,
        "rss_peak_delta_mib": usage.rss_peak_delta_mib,
        "memory_samples": usage.samples,
        "memory_source": usage.source,
    }


def write_records_csv(records: list[dict[str, object]], path: Path) -> Path:
    ordered_fields: list[str] = []
    for record in records:
        for key in record:
            if key not in ordered_fields:
                ordered_fields.append(key)
    table = CSVTable(path, ("schema_version", *ordered_fields))
    table.path.unlink(missing_ok=True)
    if records:
        table.append_many(records)
    else:
        table.write_header()
    return table.path


def csv_output_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError("--output must use a .csv suffix")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark warmed NumPy ADP statistics.")
    parser.add_argument(
        "--case",
        choices=("all",) + tuple(case.name for case in DEFAULT_CASES),
        default="all",
    )
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--statistics-workers", type=int, default=1)
    parser.add_argument("--output", type=csv_output_path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = (
        DEFAULT_CASES
        if args.case == "all"
        else tuple(case for case in DEFAULT_CASES if case.name == args.case)
    )
    records = [
        record
        for case in cases
        for record in run_case(
            case,
            repetitions=args.repetitions,
            seed=args.seed,
            statistics_workers=args.statistics_workers,
        )
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_records_csv(records, args.output)
    print(f"records: {len(records)}")
    print(f"csv: {args.output}")
    if records:
        print(f"median_sec: {statistics.median(float(row['elapsed_sec']) for row in records):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
