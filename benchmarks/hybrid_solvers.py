from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
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
        baseline = "cg" if name.startswith("manifold") else "lsmr"
        for seed in (1, 2, 3):
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
                    solver,
                    "--seed",
                    str(seed + 100),
                    "--data-seed",
                    str(seed),
                ]
                subprocess.run(command, env=env, check=True, timeout=120)


if __name__ == "__main__":
    main()
