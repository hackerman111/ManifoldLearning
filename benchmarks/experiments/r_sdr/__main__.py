"""CLI: python -m experiments.r_sdr --help."""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import math
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .design import METHODS, SUPPORTED_METHODS, catalog


def minimum_ten(value: str) -> int:
    count = int(value)
    if count < 10:
        raise argparse.ArgumentTypeError("at least 10 repetitions are required")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired ADP vs R MAVE / classical SIR / density MAVE / refined MAVE"
    )
    parser.add_argument(
        "--profile",
        choices=(
            "smoke",
            "standard",
            "full",
            "stress",
            "multi-rank",
            "high-dim-dm",
            "d25-n5000",
            "seven-families",
            "oscillation-seven",
            "fourteen-d50-n4000",
            "fourteen-d100-n8000",
        ),
        default="full",
    )
    parser.add_argument("--runs", type=minimum_ten, default=10)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--cases", default="*", help="comma-separated case name globs")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="per-process seconds, including startup",
    )
    parser.add_argument(
        "--memory-mib", type=float, default=4096, help="sampled worker RSS cap (Linux)"
    )
    parser.add_argument("--recovery-threshold", type=float, default=0.1)
    parser.add_argument("--r-iterations", type=int, default=25)
    parser.add_argument("--r-tolerance", type=float, default=1e-5)
    parser.add_argument("--sir-slices", type=int, default=10)
    parser.add_argument("--solver-steps", type=int, default=80)
    parser.add_argument(
        "--grassman-steps",
        type=int,
        default=5,
        help="Grassmann core-GN steps per ADP scale (default: 5)",
    )
    parser.add_argument(
        "--adp-config", type=Path, help="JSON object of scalar ADP_Config overrides"
    )
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--keep-inputs",
        action="store_true",
        help="save compressed X,y,truth for every repetition",
    )
    parser.add_argument("--plots", action="store_true", help="requires matplotlib")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--analyze", type=Path, help="rebuild reports only from a saved run"
    )
    args = parser.parse_args()
    if args.analyze:
        from .report import write_report

        write_report(args.analyze, plots=args.plots)
        return 0
    if (
        min(
            args.threads,
            args.r_iterations,
            args.solver_steps,
            args.grassman_steps,
            args.sir_slices,
        )
        < 1
        or not all(
            math.isfinite(v) and v > 0
            for v in (args.timeout, args.memory_mib, args.r_tolerance)
        )
        or not 0 < args.recovery_threshold <= 1
        or args.seed < 0
    ):
        parser.error(
            "counts/limits must be positive; threshold in (0,1]; seed nonnegative"
        )
    if not sys.platform.startswith("linux"):
        parser.error("RSS supervision currently requires Linux /proc")
    methods = args.methods.split(",")
    if len(set(methods)) != len(methods) or any(
        m not in SUPPORTED_METHODS for m in methods
    ):
        parser.error(f"methods must be unique members of {SUPPORTED_METHODS}")
    patterns = args.cases.split(",")
    cases = [
        c
        for c in catalog(args.profile)
        if any(fnmatch.fnmatchcase(c.name, p) for p in patterns)
    ]
    if not cases:
        parser.error("no cases matched")
    if args.sir_slices <= max(c.m for c in cases) or args.sir_slices > min(
        c.n for c in cases
    ):
        parser.error("SIR slices must satisfy max(m) < slices <= min(n)")
    planned = len(cases) * len(methods) * args.runs
    print(
        f"{len(cases)} cases x {args.runs} repetitions x "
        f"{len(methods)} methods = {planned} fits",
        flush=True,
    )
    if args.dry_run:
        for c in cases:
            print(
                f"{c.name}: n={c.n}, d={c.d}, m={c.m}, link={c.link}, "
                f"rho={c.rho}, noise={c.noise}, X={c.distribution}"
                f", frequency={c.frequency}, data_key={c.data_key}"
            )
        print(
            f"Per-fit limits: {args.timeout}s, {args.memory_mib}MiB; "
            f"worst-case timeout budget {planned * args.timeout / 3600:.1f} hours"
        )
        return 0
    rscript = shutil.which(args.rscript)
    if rscript is None:
        parser.error("Rscript not found; install R or use --rscript /path/to/Rscript")
    args.rscript = rscript
    if args.resume and args.output is None:
        parser.error("--resume requires --output and the same original options")
    if args.output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = (
            Path("benchmarks/results/outputs/r_sdr") / f"{timestamp}-{args.profile}"
        )
    if args.plots and importlib.util.find_spec("matplotlib") is None:
        parser.error("--plots requires matplotlib (uv sync --group bench)")
    from .runner import run

    print(f"Output: {args.output}", flush=True)
    try:
        return run(args, cases, methods)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Interrupted; saved rows remain. "
            "Resume with the same options and --resume.\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
