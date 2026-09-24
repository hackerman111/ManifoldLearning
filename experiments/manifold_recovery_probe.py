"""Воспроизводимая диагностика фаз manifold ADP на фиксированных seed.

Запуск: python -m experiments.manifold_recovery_probe --seeds 0:10 --out DIR
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.sparse import csr_matrix

from ADP import ADP_Manifold
from ADP.engine.common.statistic import _dense_moments

from .data import _generate_data, _make_seed_bundle
from .diagnostic import make_case
from .runner import _local_subspace_metrics


class Probe(ADP_Manifold):
    """Снять компактные состояния фаз через существующие private hooks."""

    def __init__(
        self,
        *,
        variant: str = "baseline",
        true_basis: np.ndarray,
        true_gradient: np.ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(1, **kwargs)
        self.variant = variant
        self.true_basis_by_sample = true_basis
        self.true_gradient_by_sample = true_gradient
        self.snapshots: list[dict[str, Any]] = []
        self.latest_statistics: dict[str, Any] = {}
        self.linear_diagnostics: list[dict[str, float | int]] = []

    def _local_gradients(
        self, X: np.ndarray, Y: np.ndarray, centers: np.ndarray, h: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = super()._local_gradients(X, Y, centers, h)
        self.center_indices_probe = np.array(
            [np.argmin(np.sum(np.square(X - center), axis=1)) for center in centers]
        )
        if "quad" in self.variant:
            d = X.shape[1]
            pairs = [(a, b) for a in range(d) for b in range(a, d)]
            quadratic_gradients = np.empty_like(result[0])
            self.quadratic_hessians = np.empty((len(centers), d, d))
            for j, center in enumerate(centers):
                distance2 = np.sum(np.square(X - center), axis=1)
                neighbors = np.argpartition(distance2, 60)[:60]
                delta = X[neighbors] - center
                design = np.column_stack(
                    (
                        np.ones(len(neighbors)),
                        delta,
                        *(
                            (0.5 if a == b else 1.0) * delta[:, a] * delta[:, b]
                            for a, b in pairs
                        ),
                    )
                )
                coefficient, _, rank, _ = np.linalg.lstsq(
                    design, Y[neighbors], rcond=None
                )
                if rank != design.shape[1]:
                    raise RuntimeError(f"rank-deficient quadratic pilot at center {j}")
                quadratic_gradients[j] = coefficient[1 : d + 1]
                hessian = np.zeros((d, d))
                for index, (a, b) in enumerate(pairs, start=d + 1):
                    hessian[a, b] = hessian[b, a] = coefficient[index]
                self.quadratic_hessians[j] = hessian
            result = (quadratic_gradients, result[1], result[2])
        if "oracle_gradients" in self.variant:
            result = (
                self.true_gradient_by_sample[self.center_indices_probe].copy(),
                result[1],
                result[2],
            )
        self.gradient_snapshot = result
        for j in range(len(centers)):
            weight = self._weight_block(X, centers, None, None, h, 1.0, j)[0]
            mean = (weight @ X) / weight.sum()
            design = (X - mean) * np.sqrt(weight)[:, None]
            singular = np.linalg.svd(design, compute_uv=False)
            self.linear_diagnostics.append(
                {
                    "linear_rank": int(np.linalg.matrix_rank(design)),
                    "linear_condition": float(singular[0] / singular[-1]),
                    "linear_support": int(np.count_nonzero(weight)),
                }
            )
        return result

    def _build_manifold_graph(
        self,
        centers: np.ndarray,
        projectors: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        h: float,
        alpha: float,
    ) -> csr_matrix:
        graph = super()._build_manifold_graph(
            centers, projectors, eigenvalues, h, alpha
        )
        if not any(token in self.variant for token in ("knn1", "knn2", "knn3")):
            return graph
        neighbors = 1 if "knn1" in self.variant else 2 if "knn2" in self.variant else 3
        dense = graph.toarray()
        for row in dense:
            keep = np.argsort(row)[-neighbors:]
            mask = np.ones(len(row), dtype=bool)
            mask[keep] = False
            row[mask] = 0
        return csr_matrix(dense)

    def _calculate_statistics(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        centers: np.ndarray,
        directions: np.ndarray,
        projectors: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        h: float,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        result = super()._calculate_statistics(
            X, Y, centers, directions, projectors, eigenvalues, h, alpha
        )
        local_bandwidth = np.full(len(centers), h)
        if "local_mass" in self.variant:
            floor = 12 if "local_mass12" in self.variant else 6
            I, U, mass, n_eff, edges = result
            for j in np.flatnonzero(mass < floor):
                center = centers[j : j + 1]
                basis = None if projectors is None else projectors[j : j + 1]
                spectrum = None if eigenvalues is None else eigenvalues[j : j + 1]

                def local_weight(
                    bandwidth: float,
                    center: np.ndarray = center,
                    basis: np.ndarray | None = basis,
                    spectrum: np.ndarray | None = spectrum,
                ) -> np.ndarray:
                    return self._weight_block(
                        X, center, basis, spectrum, bandwidth, alpha, 0
                    )[0]

                old_support = np.count_nonzero(local_weight(h))
                low, high = h, h
                for _ in range(30):
                    if local_weight(high).sum() >= floor:
                        break
                    high *= 2
                else:
                    raise RuntimeError("per-center mass cannot reach the floor")
                for _ in range(35):
                    middle = (low + high) / 2
                    if local_weight(middle).sum() >= floor:
                        high = middle
                    else:
                        low = middle
                weight = local_weight(high)
                local_bandwidth[j] = high
                mass[j] = weight.sum()
                normalized = weight / mass[j]
                moment = _dense_moments(
                    X,
                    Y,
                    normalized[None, :],
                    directions[j : j + 1],
                    mass[j : j + 1],
                    True,
                    include_eta=False,
                )
                I[j] = moment.I[0]
                U[j] = moment.U[0]
                n_eff[j] = 1 / np.square(normalized).sum()
                edges += int(np.count_nonzero(weight) - old_support)
            result = (I, U, mass, n_eff, edges)
        if "quad_moments" in self.variant:
            I, U, mass, n_eff, edges = result
            for j, center in enumerate(centers):
                basis = None if projectors is None else projectors[j : j + 1]
                spectrum = None if eigenvalues is None else eigenvalues[j : j + 1]
                weight = self._weight_block(
                    X, centers[j : j + 1], basis, spectrum, local_bandwidth[j], alpha, 0
                )[0]
                delta = X - center
                quadratic = 0.5 * np.einsum(
                    "nd,de,ne->n", delta, self.quadratic_hessians[j], delta
                )
                correction = _dense_moments(
                    X,
                    quadratic,
                    (weight / weight.sum())[None, :],
                    directions[j : j + 1],
                    mass[j : j + 1],
                    True,
                    include_eta=False,
                )
                I[j] -= correction.I[0]
            result = (I, U, mass, n_eff, edges)
        if "oracle_moments" in self.variant or "oracle_both" in self.variant:
            oracle_I = np.einsum(
                "jpd,jd->jp",
                result[1],
                self.true_gradient_by_sample[self.center_indices_probe],
                optimize=True,
            )
            result = (oracle_I, *result[1:])
        self.latest_statistics = {
            "mass": result[2].copy(),
            "n_eff": result[3].copy(),
            "h": h,
            "alpha": alpha,
            "function_edges": result[4],
        }
        return result

    def _one_step(
        self,
        I: np.ndarray,
        U: np.ndarray,
        mass: np.ndarray,
        graph: csr_matrix,
        projectors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        if (
            "oracle_init" in self.variant or "oracle_both" in self.variant
        ) and not self.snapshots:
            projectors = self.true_basis_by_sample[self.center_indices_probe]
        if not self.snapshots:
            self.snapshots.append(
                {
                    "phase": "init",
                    "projectors": projectors.copy(),
                    "graph": graph.copy(),
                }
            )
        slope_min_singular = np.full(len(projectors), np.inf)
        slope_rank_zero = np.zeros(len(projectors), dtype=int)
        for l in range(len(projectors)):
            sources = graph.indices[graph.indptr[l] : graph.indptr[l + 1]]
            for j in sources:
                singular = np.linalg.svd(U[j] @ projectors[l].T, compute_uv=False)
                slope_min_singular[l] = min(slope_min_singular[l], singular[-1])
                slope_rank_zero[l] += int(singular[-1] <= np.finfo(float).eps)
        snapshot = {
            "phase": "step",
            "projectors": projectors.copy(),
            "graph": graph.copy(),
            "mass": mass.copy(),
            "n_eff": self.latest_statistics["n_eff"],
            "h": self.latest_statistics["h"],
            "alpha": self.latest_statistics["alpha"],
            "slope_min_singular": slope_min_singular,
            "slope_rank_zero": slope_rank_zero,
        }
        try:
            result = super()._one_step(I, U, mass, graph, projectors)
        except Exception:
            snapshot["failed"] = True
            self.snapshots.append(snapshot)
            raise
        snapshot["projectors"] = result[0].copy()
        snapshot["diagnostics"] = result[2]
        self.snapshots.append(snapshot)
        return result


def _version() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, check=True, text=True
        ).stdout.strip()

    sources = (
        "experiments/manifold_recovery_probe.py",
        "experiments/manifold_recovery_validate.py",
        "experiments/data.py",
        "ADP/core/manifold/ADP_Manifold.py",
        "ADP/engine/manifol_engine/fit.py",
        "ADP/engine/manifol_engine/weights.py",
        "ADP/engine/manifol_engine/graphs.py",
        "ADP/engine/manifol_engine/optimisation.py",
    )
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": str(bool(git("status", "--porcelain"))),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": str(np.__config__.show(mode="dicts")),
        "source_sha256": {
            source: hashlib.sha256(Path(source).read_bytes()).hexdigest()
            for source in sources
        },
    }


def _seed_range(value: str) -> range:
    begin, end = (int(item) for item in value.split(":"))
    if begin < 0 or end <= begin:
        raise argparse.ArgumentTypeError("seeds must be START:END with END > START")
    return range(begin, end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_seed_range, default=range(10))
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "oracle_gradients",
            "oracle_init",
            "oracle_moments",
            "oracle_both",
            "knn1",
            "knn3",
            "local_mass6",
            "knn1_local_mass6",
            "knn3_local_mass6",
            "knn1_local_mass12",
            "knn3_local_mass12",
            "oracle_both_knn1",
            "oracle_both_knn3",
            "oracle_both_knn1_local_mass6",
            "oracle_both_knn3_local_mass6",
            "quad_pilot_knn1_local_mass6",
            "quad_moments_knn1_local_mass6",
            "quad_moments_knn1_local_mass12",
            "quad_moments_knn3_local_mass6",
            "quad_moments_knn2_local_mass6",
        ),
        default="baseline",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    case = make_case("manifold", "base")
    point = case.experiment.smoke
    assert point.h_min_factor is not None
    selector = "diagnostic-manifold-base"
    rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        bundle = _make_seed_bundle(
            selector,
            point,
            seed,
            common_random_fields=case.experiment.common_random_fields,
        )
        data = _generate_data(selector, point, bundle, seed)
        signal = 0.5 * point.link_scale * np.sum(np.square(data.X[:, :2]), axis=1)
        true_gradient = np.zeros_like(data.X)
        true_gradient[:, :2] = point.link_scale * data.X[:, :2] / np.std(signal)
        model = Probe(
            variant=args.variant,
            true_basis=np.swapaxes(data.beta, 1, 2),
            true_gradient=true_gradient,
            N_loc=point.N_loc,
            N_lin=point.N_lin,
            N_J=point.N_J,
            N_phi=point.N_phi,
            N_manifold=point.N_manifold,
            sync_steps=point.sync_steps,
            lambda_manifold=point.lambda_manifold,
            a=math.sqrt(2),
            h_min=point.h_min_factor * point.sigma_x / math.sqrt(point.n),
            seed=bundle.init,
            cg_tol=1e-6,
        )
        started = time.perf_counter()
        error = None
        try:
            model.fit(data.X, data.Y)
        except (RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        center_indices = model.center_indices_probe
        truth = np.swapaxes(data.beta[center_indices], 1, 2)
        radius = np.linalg.norm(data.X[center_indices, :2], axis=1)
        gradients = model.gradient_snapshot[0]
        gradient_norm = np.linalg.norm(gradients, axis=1)
        pilot_error = np.sqrt(
            np.maximum(
                0,
                1
                - np.square(
                    np.sum(gradients * truth[:, 0, :], axis=1)
                    / np.maximum(gradient_norm, np.finfo(float).tiny)
                ),
            )
        )
        for phase_index, snapshot in enumerate(model.snapshots):
            estimate = snapshot["projectors"]
            per_center = np.sqrt(
                np.maximum(
                    0,
                    1 - np.square(np.sum(estimate[:, 0] * truth[:, 0], axis=1)),
                )
            )
            graph = snapshot["graph"]
            graph_mass = np.asarray(graph.sum(axis=1)).ravel()
            for j in range(len(center_indices)):
                rows.append(  # noqa: PERF401
                    {
                        "seed": seed,
                        "phase_index": phase_index,
                        "phase": snapshot["phase"],
                        "failed": bool(snapshot.get("failed", False)),
                        "center": j,
                        "radius": radius[j],
                        "sine": per_center[j],
                        "pilot_sine": pilot_error[j],
                        "gradient_norm": gradient_norm[j],
                        "linear_mass": model.gradient_snapshot[1][j],
                        **model.linear_diagnostics[j],
                        "function_mass": snapshot.get(
                            "mass", np.full(len(radius), np.nan)
                        )[j],
                        "n_eff": snapshot.get("n_eff", np.full(len(radius), np.nan))[j],
                        "manifold_mass": graph_mass[j],
                        "manifold_neighbors": graph.indptr[j + 1] - graph.indptr[j],
                        "slope_min_singular": snapshot.get(
                            "slope_min_singular", np.full(len(radius), np.nan)
                        )[j],
                        "slope_rank_zero": snapshot.get(
                            "slope_rank_zero", np.zeros(len(radius), dtype=int)
                        )[j],
                        "h": snapshot.get("h", np.nan),
                        "alpha": snapshot.get("alpha", np.nan),
                        "linear_relative_residual_max": snapshot.get(
                            "diagnostics", {}
                        ).get("linear_relative_residual_max", np.nan),
                    }
                )
        final_quality = None
        if error is None:
            final_quality = _local_subspace_metrics(truth, model.projectors_)[0]
            overlap = np.sum(
                model.snapshots[-1]["projectors"][:, 0] * truth[:, 0], axis=1
            )
            measured = math.sqrt(np.mean(np.maximum(0, 1 - np.square(overlap))))
            assert abs(final_quality - measured) < 1e-12
        runs.append(
            {
                "seed": seed,
                "model_seed": bundle.init,
                "error": error,
                "quality": final_quality,
                "stop_reason": getattr(model, "stop_reason_", None),
                "linear_residual_max": max(
                    (
                        float(item["linear_relative_residual_max"])
                        for item in getattr(model, "trace_", [])
                    ),
                    default=None,
                ),
                "fit_seconds": time.perf_counter() - started,
            }
        )
        print(f"seed={seed} quality={final_quality} error={error}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "centers.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "runs.json").write_text(
        json.dumps(runs, indent=2, default=float), encoding="utf-8"
    )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "version": _version(),
                "selector": selector,
                "point": asdict(point),
                "seeds": list(args.seeds),
                "model": args.variant,
                "quality": "sqrt(mean_j sin^2(angle(P_j, truth_j)))",
                "threshold": 0.2,
                "kernel": "max(1-(distance2/h2)^2,0)",
                "normalization": "normalized I/U, data weight N_j omega_jl",
                "threads": {
                    name: os.environ.get(name)
                    for name in (
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OMP_NUM_THREADS",
                    )
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
