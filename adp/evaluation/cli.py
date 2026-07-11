from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import get_args

from . import stress
from .reports import benchmark_summary, ensure_matplotlib_config_dir, save_benchmark_report
from .runner import run_benchmark_suite
from .scenarios import BenchmarkMethod, default_scenarios, grid_scenarios
from .single_index import (
    PROFILE_IDS,
    SingleIndexSeriesConfig,
    run_single_index_benchmark,
)


DEFAULT_METHODS: tuple[BenchmarkMethod, ...] = (
    "adp_new",
    "statsmodels_sir",
    "statsmodels_save",
    "statsmodels_phd",
    "sklearn_pls",
)


def main(argv: list[str] | None = None) -> int:
    """Запускает общий CLI для benchmark и stress-профилей."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "stress":
        return stress.main(args[1:])
    if args and args[0] == "benchmark":
        return run_benchmark_command(args[1:])
    if args and args[0] == "single-index":
        return run_single_index_command(args[1:])
    if args and args[0] in {"-h", "--help"}:
        build_top_parser().parse_args(args)
        return 0
    return run_benchmark_command(args)


def build_top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ADP command line tools: benchmarks and stress profiles.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("benchmark", help="Run benchmark scenarios and save CSV/PNG reports.")
    subparsers.add_parser(
        "single-index",
        help="Run the reproducible single-index ADP experiment series.",
    )
    subparsers.add_parser("stress", help="Run ADP single-index stress profiles.")
    return parser


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ADP benchmarks against ready EDR baselines.",
    )
    parser.add_argument("--quick", action="store_true", help="Use smaller default scenarios.")
    parser.add_argument("--grid", action="store_true", help="Use explicit d x P scenario grid.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_outputs"), help="Output directory.")
    parser.add_argument("--prefix", default="adp_benchmark", help="Output filename prefix.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods. Choices: " + ", ".join(get_args(BenchmarkMethod)),
    )
    parser.add_argument("--show-progress", action="store_true", help="Show ADP progress bars.")
    parser.add_argument("--allow-failures", action="store_true", help="Return zero even if a method failed.")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Limit scenario count after construction.")
    parser.add_argument("--grid-d", default="10,25,50,100,200", help="Comma-separated d values for --grid.")
    parser.add_argument("--grid-directions", default="5,10,20,40", help="Comma-separated P values for --grid.")
    parser.add_argument("--grid-n", type=int, default=360, help="Sample size for --grid.")
    parser.add_argument("--grid-centers", type=int, default=90, help="Center count for --grid.")
    parser.add_argument("--grid-trials", type=int, default=5, help="Trial count for --grid.")
    parser.add_argument("--grid-outer-steps", type=int, default=4, help="Outer ADP steps for --grid.")
    parser.add_argument("--grid-inner-steps", type=int, default=10, help="Inner ADP steps for --grid.")
    return parser


def build_single_index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the normalized single-index ADP benchmark series.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_IDS),
        default="smoke",
        help="Scenario profile (default: smoke).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_outputs/single_index"),
        help="Root directory for a new series.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=1,
        help="Process workers (default: 1).",
    )
    parser.add_argument(
        "--statistics-workers",
        type=positive_int,
        default=1,
        help="NumPy statistics workers inside one ADP fit (default: safe serial 1).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Existing series directory to resume.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Replace and rerun failed commit markers.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=positive_int,
        default=None,
        help="Limit the selected profile after registry expansion.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("adp_D1_data"),
        help="Manifest-driven D01-D04 package (default: adp_D1_data).",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Compatibility flag; D01-D04 still require the local manifest package.",
    )
    return parser


def run_single_index_command(argv: list[str] | None = None) -> int:
    args = build_single_index_parser().parse_args(argv)
    config = SingleIndexSeriesConfig(
        profile=args.profile,
        base_seed=args.seed,
        jobs=args.jobs,
        statistics_workers=args.statistics_workers,
        retry_failed=args.retry_failed,
        max_scenarios=args.max_scenarios,
        data_dir=str(args.data_dir),
        allow_download=args.allow_download,
    )
    saved = run_single_index_benchmark(
        config,
        args.output,
        resume=args.resume,
    )
    print(f"series: {saved['series'].parent}")
    for name in ("runs", "iterations", "summary", "failures", "artifacts"):
        print(f"{name}: {saved[name]}")
    return 0


def run_benchmark_command(argv: list[str] | None = None) -> int:
    parser = build_benchmark_parser()
    args = parser.parse_args(argv)
    try:
        methods = parse_methods(args.methods)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    scenarios = build_benchmark_scenarios(args)
    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]

    ensure_matplotlib_config_dir()
    frame = run_benchmark_suite(
        scenarios,
        methods=methods,
        random_state=args.seed,
        show_progress=args.show_progress,
    )
    saved = save_benchmark_report(frame, args.output, prefix=args.prefix)
    summary = benchmark_summary(frame)
    summary_path = args.output / f"{args.prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"rows: {len(frame)}")
    print(f"csv: {saved['csv']}")
    print(f"summary: {summary_path}")
    for key, path in saved.items():
        if key != "csv":
            print(f"{key}: {path}")

    has_failures = bool(frame.get("failed", False).any())
    if has_failures and not args.allow_failures:
        return 1
    return 0


def build_benchmark_scenarios(args: argparse.Namespace):
    if args.grid:
        return grid_scenarios(
            d_values=parse_int_tuple(args.grid_d, name="grid-d"),
            direction_values=parse_int_tuple(args.grid_directions, name="grid-directions"),
            n=args.grid_n,
            n_centers=args.grid_centers,
            trials=args.grid_trials,
            outer_steps=args.grid_outer_steps,
            inner_steps=args.grid_inner_steps,
        )
    return default_scenarios(quick=args.quick)


def parse_int_tuple(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated list of integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one integer")
    return parsed


def parse_methods(value: str) -> tuple[BenchmarkMethod, ...]:
    allowed = set(get_args(BenchmarkMethod))
    methods = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [method for method in methods if method not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError("Unknown benchmark methods: " + ", ".join(unknown))
    if not methods:
        raise argparse.ArgumentTypeError("At least one benchmark method is required")
    return methods  # type: ignore[return-value]


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed
