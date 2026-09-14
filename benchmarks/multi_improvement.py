from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Парные полные запуски: code reference и отдельный local-cv estimator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--suite", choices=("speed", "quality"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cases = (
        [(3000, 100, 2, 0.05), (10000, 100, 10, 0.05)]
        if args.suite == "speed"
        else [(1000, 30, 2, 0.05), (1000, 30, 2, 1.0), (3000, 100, 2, 0.5)]
    )
    for n, d, m, noise in cases:
        for seed in (1, 2, 3):
            variants = (
                ["before", "fixed"] if args.suite == "speed" else ["fixed", "local-cv"]
            )
            if seed % 2 == 0:
                variants.reverse()
            for variant in variants:
                label = f"n{n}_d{d}_m{m}_noise{noise}_seed{seed}_{variant}"
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.multi_training",
                    "--output",
                    str(args.output / f"{label}.json"),
                    "--label",
                    label,
                    "--n",
                    str(n),
                    "--d",
                    str(d),
                    "--index-dim",
                    str(m),
                    "--noise",
                    str(noise),
                    "--N_J",
                    str(max(100, n // 20)),
                    "--N_phi",
                    "20",
                    "--N_loc",
                    "20",
                    "--N_lin",
                    str(max(40, d * 3 // 2)),
                    "--outer-steps",
                    "6",
                    "--solver-max-steps",
                    "3",
                    "--data-seed",
                    str(seed),
                    "--seed",
                    str(seed + 100),
                ]
                if variant == "before":
                    command += ["--reference-root", str(args.reference_root)]
                elif variant == "local-cv":
                    command += ["--index-init", "local-cv"]
                subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
