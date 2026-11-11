"""Reproduce one untimed frozen solver profile on a saved snapshot."""

import argparse
import cProfile
import json
import pstats
from pathlib import Path

import numpy as np

from ADP.solver import grassman
from ADP.solver.LSMR import _normalize_index
from experiments.metric_spokoiny_2026_10_15.diagnostics import projector_metrics
from experiments.metric_spokoiny_2026_10_15.workers import _solver

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=("core", "metric2"), required=True)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
with np.load(args.snapshot) as source:
    B, U, I, mass, truth = (
        source[k].copy() for k in ("index_init", "U", "I", "mass", "truth_basis")
    )
profile = cProfile.Profile()
result = profile.runcall(
    _solver(args.arm, 100).method, B, U, I, mass=mass, lambda_prox=0.05
)
prefix = args.output / f"{args.label}_{args.arm}"
paths = [prefix.with_suffix(suffix) for suffix in (".pstats", ".txt", ".json")]
if any(path.exists() for path in paths):
    raise FileExistsError(f"refusing to overwrite profile artifact for {args.label}")
profile.dump_stats(str(paths[0]))
with paths[1].open("x") as stream:
    stats = pstats.Stats(profile, stream=stream).strip_dirs()
    stream.write("Top 25 by cumulative time\n")
    stats.sort_stats("cumulative").print_stats(25)
    stream.write("\nTop 25 by self time\n")
    stats.sort_stats("tottime").print_stats(25)
Y = _normalize_index(result.index).T
state = grassman._profile(grassman._project(U, Y), I, mass, 0.0)
gradient = grassman._gradient(Y, U, state, mass)
scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
certificate = {
    "arm": args.arm,
    "snapshot_sha256": __import__("hashlib")
    .sha256(args.snapshot.read_bytes())
    .hexdigest(),
    "objective": float(state.value),
    "smooth": bool(state.smooth),
    "live_gradient_certificate": float(np.linalg.norm(gradient) / scale),
    "local_rank_min": int(np.min(state.ranks)),
    **projector_metrics(Y, truth),
}
with paths[2].open("x") as stream:
    json.dump(certificate, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(certificate, sort_keys=True))
