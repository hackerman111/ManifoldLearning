"""Paired end-to-end CLI comparison of the LSMR and Grassmann solvers.

Run with ``rtk proxy python -m benchmarks.lsmr_grassman_comparison``.
Each pair uses identical generated data, initialization seed, and ADP settings.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from ADP.cli.main import _run, build_parser


WORKLOADS = ((120, 10, 3), (200, 20, 4))
SEEDS = (0, 1, 2)
SOLVERS = ("grassman", "lsmr")
OUTPUT = Path("benchmarks/results/outputs/lsmr_vs_grassman.json")


def _args(solver: str, n: int, d: int, index_dim: int, seed: int):
    return build_parser().parse_args(
        [
            "--mode", "multi",
            "--solver", solver,
            "--n", str(n),
            "--d", str(d),
            "--index-dim", str(index_dim),
            "--noise", "0.05",
            "--data-seed", str(seed),
            "--seed", str(seed),
            "--N_loc", "8",
            "--N_lin", "30",
            "--N_J", "40",
            "--N_phi", "5",
            "--outer_steps", "1",
            "--lambda-penalty", "0.05",
            "--h-min", "1000000",
            "--batch-size", "32",
            "--index-init", "random",
            "--select-step", "last",
            "--solver-tol", "1e-6",
            "--solver-max-steps", "20",
            "--theta", "0.1",
        ]
    )


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for n, d, index_dim in WORKLOADS:
        for seed in SEEDS:
            for solver in SOLVERS:
                row: dict[str, object] = {
                    "n": n,
                    "d": d,
                    "index_dim": index_dim,
                    "seed": seed,
                    "solver": solver,
                }
                try:
                    _, _, profile, metadata = _run(
                        _args(solver, n, d, index_dim, seed)
                    )
                    diagnostic = metadata["trace"][-1]["solver"]
                    row.update(
                        status="ok",
                        quality=metadata["last_quality"],
                        stop_reason=metadata["stop_reason"],
                        solver_time_sec=profile["solver"]["time_seconds"],
                        total_time_sec=profile["total"]["time_seconds"],
                        solver_converged=diagnostic.get("converged"),
                        solver_iterations=diagnostic.get(
                            "accepted_steps", diagnostic.get("iterations")
                        ),
                        solver_loss=diagnostic.get("loss"),
                        solver_diagnostics=diagnostic,
                    )
                except Exception as error:  # Keep failed fits in the paired record.
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                rows.append(row)

    summaries: list[dict[str, object]] = []
    for n, d, index_dim in WORKLOADS:
        for solver in SOLVERS:
            selected = [
                row for row in rows
                if row["n"] == n
                and row["d"] == d
                and row["index_dim"] == index_dim
                and row["solver"] == solver
            ]
            successful = [row for row in selected if row["status"] == "ok"]
            summaries.append(
                {
                    "n": n,
                    "d": d,
                    "index_dim": index_dim,
                    "solver": solver,
                    "successes": len(successful),
                    "failures": len(selected) - len(successful),
                    "median_solver_time_sec": (
                        statistics.median(row["solver_time_sec"] for row in successful)
                        if successful else None
                    ),
                    "median_total_time_sec": (
                        statistics.median(row["total_time_sec"] for row in successful)
                        if successful else None
                    ),
                    "median_quality": (
                        statistics.median(row["quality"] for row in successful)
                        if successful else None
                    ),
                    "converged": sum(
                        row["solver_converged"] is True for row in successful
                    ),
                }
            )
    return {
        "protocol": {
            "mode": "multi",
            "index_dims": tuple(item[2] for item in WORKLOADS),
            "workloads_n_d": WORKLOADS,
            "seeds": SEEDS,
            "paired": True,
            "selection": "last",
            "outer_steps": 1,
            "solver_max_steps": 20,
            "solver_tol": 1e-6,
            "index_init": "random",
            "note": (
                "One outer fit per pair. Grassmann lambda_prox is GN damping; "
                "LSMR lambda_prox regularizes its correction subproblem."
            ),
        },
        "source_sha256": {
            "lsmr": _fingerprint(Path("ADP/solver/LSMR.py")),
            "grassman": _fingerprint(Path("ADP/solver/grassman.py")),
            "cli": _fingerprint(Path("ADP/cli/main.py")),
        },
        "summaries": summaries,
        "rows": rows,
    }


def main() -> int:
    results = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved {OUTPUT}")
    for row in results["summaries"]:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
