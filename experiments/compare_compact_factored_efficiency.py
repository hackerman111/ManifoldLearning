from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adp import ADP, ADPConfig
from adp.evaluation.single_index.scenarios import (
    full_parameter_grid,
    parse_seed_selection,
    smoke_parameter_grid,
)
from experiments.compare_model_efficiency import (
    compare_models,
    pair_model_runs,
    parse_args,
    write_comparison_artifacts,
)


MODEL_NAMES = ("random_projection", "cpu_compact_factored")


def build_models() -> tuple[Any, Any]:
    """Build the current and factored compact statistics implementations."""

    common = {
        "statistics_workers": 1,
        "show_progress": False,
        "record_telemetry": True,
        "renew_directions": False,
        "random_state": 0,
    }
    baseline = ADP.create(
        "new",
        ADPConfig(**common),
        stages={"statistics_builder": MODEL_NAMES[0]},
    )
    candidate = ADP.create(
        "new",
        ADPConfig(**common),
        stages={"statistics_builder": MODEL_NAMES[1]},
    )
    return baseline, candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    grid = (
        full_parameter_grid("2")
        if args.profile == "full"
        else smoke_parameter_grid("2")
    )
    seeds = (
        parse_seed_selection(args.seeds)
        if args.seeds is not None
        else (tuple(range(100)) if args.profile == "full" else (0,))
    )
    runs = compare_models(
        *build_models(),
        model_names=MODEL_NAMES,
        parameter_grid=grid,
        seeds=seeds,
        sample_interval_sec=args.sample_interval,
        jobs=args.jobs,
        show_progress=not args.no_progress,
    )
    artifacts = write_comparison_artifacts(
        runs,
        args.output,
        model_names=MODEL_NAMES,
        dpi=args.dpi,
    )
    paired = pair_model_runs(runs, model_names=MODEL_NAMES)
    print(f"runs: {len(runs)}")
    print(f"median_time_speedup: {paired['time_speedup'].median():.6f}")
    print(
        "median_peak_delta_memory_ratio: "
        f"{paired['peak_delta_memory_ratio'].median():.6f}"
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
