from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Сетка качества; один процесс и один BLAS-поток на точку, включая отказы."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--n", type=int, nargs="+", default=[500, 1000, 3000])
    parser.add_argument("--d", type=int, nargs="+", default=[10, 30, 100])
    settings = parser.parse_args()
    settings.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for n in settings.n:
        for d in settings.d:
            for seed in (1, 2, 3):
                label = f"n{n}_d{d}_seed{seed}"
                output = settings.output / f"{label}.json"
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.multi_training",
                    "--output",
                    str(output),
                    "--label",
                    label,
                    "--n",
                    str(n),
                    "--d",
                    str(d),
                    "--index-dim",
                    "2",
                    "--data-seed",
                    str(seed),
                    "--seed",
                    str(seed + 100),
                    "--N_J",
                    str(max(100, (n + 19) // 20)),
                    "--N_phi",
                    "20",
                    "--N_loc",
                    "20",
                    "--N_lin",
                    str(max(40, 3 * d // 2)),
                    "--outer_steps",
                    "6",
                    "--solver-max-steps",
                    "3",
                    "--solver",
                    "lsmr",
                ]
                try:
                    subprocess.run(
                        command, check=True, env=env, timeout=settings.timeout
                    )
                except (
                    subprocess.TimeoutExpired,
                    subprocess.CalledProcessError,
                ) as error:
                    output.write_text(
                        json.dumps(
                            {
                                "label": label,
                                "error": str(error),
                                "command": command,
                            },
                            indent=2,
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    main()
