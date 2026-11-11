"""Reuse the paired benchmark's exact datasets/configs in fresh workers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def worker(case, seed, variant):
    import ADP.solver.SVD as svd
    from benchmarks.svd_vs_hybrid import CASES, _worker

    CASES["spokoini_d100"] = {**CASES["spokoini_m2_dhigh_30s"], "d": 100}
    original = svd.solve
    if variant == "correction":

        def correction(*args, **kwargs):
            return original(*args, **kwargs, low_rank_target="correction")

        svd.solve = correction
    elif variant == "full":
        spec = importlib.util.spec_from_file_location(
            "prototype", ROOT / "prototype.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        svd.solve = module.solve
    result = _worker(case, seed, "svd")
    result["variant"] = variant
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", default="small")
    parser.add_argument("--seed", type=int, default=73000)
    parser.add_argument("--variant", default="full")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["small", "spokoini_m2_dhigh_30s", "spokoini_d100"],
    )
    parser.add_argument(
        "--variants", nargs="+", default=["matrix", "correction", "full"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[73000])
    parser.add_argument("--output", default="pilot.jsonl")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.case, args.seed, args.variant)))
        return
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": ".",
    }
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    for case in args.cases:
        for seed in args.seeds:
            for variant in args.variants:
                command = [
                    sys.executable,
                    str(Path(__file__)),
                    "--worker",
                    "--case",
                    case,
                    "--seed",
                    str(seed),
                    "--variant",
                    variant,
                ]
                try:
                    result = subprocess.run(
                        command, env=env, capture_output=True, text=True, timeout=240
                    )
                    if result.returncode:
                        raise RuntimeError(result.stderr[-4000:])
                    row = json.loads(result.stdout)
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    row = {
                        "case": case,
                        "seed": seed,
                        "variant": variant,
                        "status": "error",
                        "error": str(error),
                    }
                with output.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k
                            in {
                                "case",
                                "seed",
                                "variant",
                                "status",
                                "error",
                                "fit_time_sec",
                                "projector_distance",
                                "peak_rss_kib",
                            }
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
