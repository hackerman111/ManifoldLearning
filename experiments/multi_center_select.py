"""C2: изолированный отбор multi-index центров по обучающему pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from ADP.core.ADP_Config import ADP_Config
from ADP.engine.common.calculus import pairwise_distance2, search_bandwidth

from .data import _generate_data, _GeneratedData, _make_seed_bundle
from .diagnostic import _code_fingerprint
from .multi_center_diagnostic import SEEDS, _local_spectrum, _weight_metrics
from .multiv2_quality import SENTINELS, _baseline
from .runner import _effective_config


@dataclass(frozen=True, slots=True)
class CenterSelection:
    selected_ids: np.ndarray | None
    pilot_basis: np.ndarray
    feasible_count: int
    checkpoints: tuple[dict[str, float | int], ...]
    reason: str


def _cross_covariances(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Cov_w(X,Y) через GEMM; рабочая память O(Jd+nd)."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean()
    mass = W.sum(axis=1)
    mean_X = (W @ Xc) / mass[:, None]
    mean_Y = (W @ Yc) / mass
    return (W @ (Xc * Yc[:, None])) / mass[:, None] - mean_X * mean_Y[:, None]


def _greedy(
    vectors: np.ndarray,
    support: np.ndarray,
    ids: np.ndarray,
    J0: int,
) -> tuple[np.ndarray | None, tuple[dict[str, float | int], ...], str]:
    """Coverage-first, затем точный D-optimal marginal; tie по ID."""
    M, m = vectors.shape
    if M == 0:
        return None, (), "no feasible centers"
    reference = np.eye(m) + (J0 / M) * (vectors.T @ vectors)
    sign, reference_logdet = np.linalg.slogdet(reference)
    if sign <= 0 or reference_logdet <= 1e-12:
        return None, (), "uninformative pilot"
    inverse = np.eye(m)
    information = np.zeros((m, m))
    covered = np.zeros(support.shape[1], dtype=bool)
    available = np.ones(M, dtype=bool)
    selected: list[int] = []
    checkpoints: list[dict[str, float | int]] = []
    counts = tuple(dict.fromkeys((J0 // 4, J0 // 2, 3 * J0 // 4, J0)))
    for _ in range(min(J0, M)):
        gains = np.log1p(np.einsum("jd,dk,jk->j", vectors, inverse, vectors))
        gains[~available] = -np.inf
        if np.mean(covered) < 0.99:
            new_coverage = np.count_nonzero(support[:, ~covered], axis=1)
            new_coverage[~available] = -1
            order = np.lexsort((ids, -gains, -new_coverage))
        else:
            order = np.lexsort((ids, -gains))
        j = int(order[0])
        available[j] = False
        selected.append(j)
        covered |= support[j]
        vector = vectors[j]
        direction = inverse @ vector
        denominator = 1.0 + float(vector @ direction)
        inverse -= np.outer(direction, direction) / denominator
        information += np.outer(vector, vector)
        k = len(selected)
        if k in counts:
            sign, logdet = np.linalg.slogdet(np.eye(m) + (J0 / k) * information)
            if sign <= 0:
                raise RuntimeError("non-positive pilot information")
            checkpoints.append(
                {
                    "J": k,
                    "coverage": float(np.mean(covered)),
                    "information_ratio": float(logdet / reference_logdet),
                }
            )
    for row in checkpoints:
        if (
            row["J"] < J0
            and row["coverage"] >= 0.99
            and row["information_ratio"] >= 0.95
        ):
            return (
                ids[np.array(selected[: int(row["J"])])],
                tuple(checkpoints),
                "selected",
            )
    return None, tuple(checkpoints), "no smaller J passes coverage and information"


def select_centers(
    X: np.ndarray,
    Y: np.ndarray,
    center_ids: np.ndarray,
    config: ADP_Config,
    m: int,
) -> CenterSelection:
    """Вернуть обучающий candidate; исходные строки X не переставляются."""
    if config.N_lin is None or config.N_J is None or config.h_min is None:
        raise ValueError("selection requires explicit N_lin, N_J and h_min")
    if config.center_displacement != 0 or config.training_set != "all":
        raise ValueError(
            "selection requires center_displacement=0 and training_set=all"
        )
    if len(center_ids) != config.N_J or len(np.unique(center_ids)) != len(center_ids):
        raise ValueError("center_ids must be a J0-sized set without replacement")
    distance2 = pairwise_distance2(X, X[center_ids])
    h_lin = search_bandwidth(
        distance2, config.N_lin, config.kernel, lower=np.finfo(float).eps
    )
    h_outer = search_bandwidth(
        distance2, config.N_loc, config.kernel, lower=config.h_min
    )
    W_lin = config.kernel(distance2 / h_lin**2)
    W_outer = config.kernel(distance2 / h_outer**2)
    mass_lin, _, _ = _weight_metrics(W_lin)
    _, n_eff_outer, support_outer = _weight_metrics(W_outer)
    rank, _ = _local_spectrum(X, W_lin, mass_lin)
    feasible = (rank == X.shape[1]) & (n_eff_outer >= m + 2) & (support_outer >= m + 2)
    gradients = _cross_covariances(X, Y, W_lin)
    _, values, right = np.linalg.svd(
        np.sqrt(mass_lin)[:, None] * gradients, full_matrices=False
    )
    if (
        len(values) < m
        or values[m - 1] <= np.finfo(float).eps * max(gradients.shape) * values[0]
    ):
        raise RuntimeError("pilot does not identify the requested index dimension")
    pilot_basis = right[:m]
    projected = gradients @ pilot_basis.T
    scale = (
        float(np.median(np.linalg.norm(projected[feasible], axis=1)))
        if np.any(feasible)
        else 0.0
    )
    if not np.isfinite(scale) or scale <= 0:
        return CenterSelection(
            None, pilot_basis, int(np.count_nonzero(feasible)), (), "zero pilot scale"
        )
    vectors = (
        projected[feasible]
        * np.sqrt(n_eff_outer[feasible] / config.N_loc)[:, None]
        / scale
    )
    selected_ids, checkpoints, reason = _greedy(
        vectors, W_outer[feasible] > 0, center_ids[feasible], config.N_J
    )
    return CenterSelection(
        selected_ids, pilot_basis, int(np.count_nonzero(feasible)), checkpoints, reason
    )


def make_case(name: str, seed: int) -> tuple[_GeneratedData, ADP_Config, np.ndarray]:
    point = replace(_baseline(name), center_displacement=0.0, training_set="all")
    bundle = _make_seed_bundle(f"multi-center-{name}", point, seed)
    data = _generate_data(f"multi-center-{name}", point, bundle, seed)
    config = _effective_config(ADP_Config(), point, bundle.init)
    if config.N_J is None:
        raise RuntimeError("expected explicit N_J")
    center_seed = np.random.SeedSequence(config.seed).spawn(3)[0]
    center_ids = np.random.default_rng(center_seed).choice(
        len(data.X), config.N_J, replace=False
    )
    return data, config, center_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/diagnostic/multi_center_c2_20260925"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print("C2: d10,n1000 x seeds 4000-4009; 20 pilot selections")
        return 0
    root = args.output_dir
    if root.exists():
        parser.error(f"output directory already exists: {root}")
    root.mkdir(parents=True)
    manifest = {
        "status": "running",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "code_fingerprint": _code_fingerprint(),
        "points": {
            name: asdict(replace(_baseline(name), center_displacement=0.0))
            for name in SENTINELS
        },
        "seeds": list(SEEDS),
        "seed_scheme": (
            "_make_seed_bundle(multi-center-{name}, point, seed); "
            "model=seeds.init; centers=SeedSequence(model).spawn(3)[0]"
        ),
        "rule": "docs/experiments/multi_center_selection_2026-09-25/c2_rule.md",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fields: tuple[str, ...] = (
        "point",
        "seed",
        "status",
        "reason",
        "J",
        "feasible",
        "wall_sec",
        "peak_rss_mib",
        "checkpoints",
        "selected_ids",
    )
    with (root / "selections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer: csv.DictWriter[str] = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in SENTINELS:
            for seed in SEEDS:
                started = perf_counter()
                row: dict[str, object] = {"point": name, "seed": seed}
                try:
                    data, config, ids = make_case(name, seed)
                    result = select_centers(data.X, data.Y, ids, config, 2)
                    row.update(
                        status="success"
                        if result.selected_ids is not None
                        else "no_candidate",
                        reason=result.reason,
                        J=len(result.selected_ids)
                        if result.selected_ids is not None
                        else None,
                        feasible=result.feasible_count,
                        checkpoints=json.dumps(result.checkpoints),
                        selected_ids=json.dumps(result.selected_ids.tolist())
                        if result.selected_ids is not None
                        else None,
                    )
                except Exception as error:  # Keep every failed seed in the gate.
                    row.update(
                        status="numerical_failure",
                        reason=f"{type(error).__name__}: {error}",
                    )
                row["wall_sec"] = perf_counter() - started
                row["peak_rss_mib"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                )
                writer.writerow(row)
                stream.flush()
                print(
                    f"{name} seed={seed} {row['status']} J={row.get('J')}", flush=True
                )
    manifest["status"] = "complete"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
