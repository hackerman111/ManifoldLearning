from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Парные полные запуски исходного и hybrid солвера, отдельный процесс."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--feasible", action="store_true")
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--no-tracemalloc", action="store_true")
    parser.add_argument("--hybrid-inner-rtol", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cases = {
        "multi2": [
            "--mode",
            "multi",
            "--n",
            "1000",
            "--d",
            "100",
            "--index-dim",
            "2",
            "--N_J",
            "100",
            "--N_phi",
            "20",
            "--N_loc",
            "20",
            "--N_lin",
            "150",
            "--outer_steps",
            "3",
            "--solver-max-steps",
            "3",
        ],
        "multi10": [
            "--mode",
            "multi",
            "--n",
            "1000",
            "--d",
            "100",
            "--index-dim",
            "10",
            "--N_J",
            "100",
            "--N_phi",
            "20",
            "--N_loc",
            "20",
            "--N_lin",
            "150",
            "--outer_steps",
            "3",
            "--solver-max-steps",
            "3",
        ],
        "manifold20": [
            "--mode",
            "manifold",
            "--n",
            "400",
            "--d",
            "20",
            "--index-dim",
            "2",
            "--N_J",
            "40",
            "--N_phi",
            "12",
            "--N_loc",
            "20",
            "--N_lin",
            "40",
            "--N_manifold",
            "8",
            "--sync-steps",
            "1",
            "--h_min",
            "1000000",
        ],
        "manifold_adaptive": [
            "--mode",
            "manifold",
            "--n",
            "150",
            "--d",
            "5",
            "--index-dim",
            "2",
            "--N_J",
            "15",
            "--N_phi",
            "10",
            "--N_loc",
            "20",
            "--N_lin",
            "30",
            "--N_manifold",
            "5",
            "--sync-steps",
            "1",
            "--h_min",
            "0.8",
            "--scale-boundary",
            "stop",
        ],
    }
    cases["manifold150"] = [
        "--mode",
        "manifold",
        "--n",
        "600",
        "--d",
        "150",
        "--index-dim",
        "2",
        "--N_J",
        "20",
        "--N_phi",
        "30",
        "--N_loc",
        "100",
        "--N_lin",
        "400",
        "--N_manifold",
        "8",
        "--sync-steps",
        "1",
        "--h_min",
        "1000000",
        "--cg-maxiter",
        "1000",
    ]
    cases["multi1000"] = [
        "--mode",
        "multi",
        "--n",
        "10000",
        "--d",
        "1000",
        "--index-dim",
        "10",
        "--N_J",
        "100",
        "--N_phi",
        "20",
        "--N_loc",
        "100",
        "--N_lin",
        "150",
        "--index-init",
        "random",
        "--outer_steps",
        "1",
        "--solver-max-steps",
        "2",
    ]
    cases["multi_large"] = [
        "--mode",
        "multi",
        "--n",
        "10000",
        "--d",
        "100",
        "--index-dim",
        "10",
        "--N_J",
        "500",
        "--N_phi",
        "20",
        "--N_loc",
        "20",
        "--N_lin",
        "150",
        "--outer_steps",
        "3",
        "--solver-max-steps",
        "3",
    ]
    if args.feasible:
        cases["manifold20"] += ["--N_lin", "100"]
        cases["manifold_adaptive"] += [
            "--N_loc",
            "30",
            "--N_lin",
            "60",
            "--N_J",
            "20",
            "--N_phi",
            "20",
            "--N_manifold",
            "8",
        ]
    for name, flags in cases.items():
        if args.cases and name not in args.cases:
            continue
        baseline = (
            "reference"
            if args.reference_root is not None
            else ("cg" if name.startswith("manifold") else "lsmr")
        )
        for seed in args.seeds:
            order = (baseline, "hybrid") if seed % 2 else ("hybrid", baseline)
            for solver in order:
                label = f"{name}_{seed}_{solver}"
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.multi_training",
                    "--output",
                    str(args.output / f"{label}.json"),
                    "--label",
                    label,
                    *flags,
                    "--solver",
                    "hybrid" if solver == "reference" else solver,
                    "--seed",
                    str(seed + 100),
                    "--data-seed",
                    str(seed),
                ]
                if solver == "reference":
                    command += ["--reference-root", str(args.reference_root)]
                if args.no_tracemalloc:
                    command.append("--no-tracemalloc")
                if (
                    solver == "hybrid"
                    and name.startswith("multi")
                    and args.hybrid_inner_rtol is not None
                ):
                    command += ["--hybrid-inner-rtol", str(args.hybrid_inner_rtol)]
                try:
                    subprocess.run(command, env=env, check=True, timeout=args.timeout)
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as error:
                    # Отказы процесса тоже остаются в таблице результатов.
                    (args.output / f"{label}.json").write_text(
                        json.dumps(
                            {
                                "label": label,
                                "arguments": command,
                                "error": f"{type(error).__name__}: {error}",
                            },
                            indent=2,
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    main()
