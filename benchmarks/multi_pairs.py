from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Парные замеры LSMR: одинаковые данные, чередование порядка A/B."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initialization-reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for m in (2, 10):
        for seed in (1, 2, 3):
            order = ("before", "after") if seed % 2 else ("after", "before")
            for label in order:
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.multi_training",
                    "--output",
                    str(args.output / f"m{m}_seed{seed}_{label}.json"),
                    "--label",
                    label,
                    "--n",
                    "1000",
                    "--d",
                    "100",
                    "--index-dim",
                    str(m),
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
                    "--solver",
                    "lsmr",
                    "--data-seed",
                    str(seed),
                    "--seed",
                    str(seed + 100),
                ]
                if label == "before":
                    command += ["--solver-source", str(args.reference)]
                    if args.initialization_reference is not None:
                        command += [
                            "--initialization-source",
                            str(args.initialization_reference),
                        ]
                subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
