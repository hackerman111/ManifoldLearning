"""Read-only decomposition of curved manifold recovery on frozen development seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info, threadpool_limits

from ADP.engine.manifol_engine.graphs import initialize_projectors
from ADP.engine.manifol_engine.weights import local_quadratic_pilot
from experiments.manifold_generalization import DiagnosticManifold, geometry, make_data
from experiments.runner import _local_subspace_metrics


class Probe(DiagnosticManifold):
    """Retain inputs to existing hooks; never inject truth into fit."""

    def _local_gradients(
        self, X: np.ndarray, Y: np.ndarray, centers: np.ndarray, h: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.pilot_inputs = (X, Y, centers)
        return super()._local_gradients(X, Y, centers, h)

    def _initialize_projectors(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        gradients: np.ndarray,
        gradient_mass: np.ndarray,
        graph: csr_matrix,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.pilot_gradients = gradients.copy()
        self.pilot_mass = gradient_mass.copy()
        self.initial_graph = graph.copy()
        result = super()._initialize_projectors(gradients, gradient_mass, graph, m)
        self.initial_projectors = result[0].copy()
        return result


def containment(gradient: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    """Principal sine of observable gradient versus each local row space."""
    norms = np.linalg.norm(gradient, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("undefined gradient direction at a stationary center")
    direction = gradient / norms[:, None]
    coefficients = np.einsum("jmd,jd->jm", basis, direction)
    sine = np.sqrt(np.maximum(0.0, 1.0 - np.sum(coefficients**2, axis=1)))
    return {"rms": float(np.sqrt(np.mean(sine**2))), "max": float(sine.max())}


def normalized_gradients(gradient: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(gradient, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("undefined estimated gradient direction")
    return (gradient / norms[:, None])[:, None, :]


def run_case(
    seed: int,
    support: str,
    m: int,
    curvature: float,
    noise: float,
    spectrum: Literal["relative", "unit"],
) -> dict[str, Any]:
    X, Y, truth, queries, query_truth, model_seed = make_data(
        seed, 600, 8, m, curvature, noise
    )
    rotation_stream = np.random.SeedSequence(seed).spawn(5)[1]
    rotation, _ = np.linalg.qr(
        np.random.default_rng(rotation_stream).normal(size=(8, 8))
    )
    _, reconstructed_truth, exact_gradient = geometry(X, rotation, m, curvature)
    np.testing.assert_allclose(truth, reconstructed_truth, atol=1e-12)
    broad = support == "broad"
    model = Probe(
        m,
        estimator="manifold",
        localization_spectrum=spectrum,
        N_loc=300 if broad else 80,
        N_lin=300 if broad else 200,
        N_J=40,
        N_phi=40,
        N_manifold=30 if broad else 10,
        sync_steps=3,
        lambda_manifold=0.5,
        cg_tol=1e-6,
        solver="cg",
        seed=model_seed,
        scale_boundary="stop",
    )
    row: dict[str, Any] = {
        "seed": seed,
        "support": support,
        "m": m,
        "curvature": curvature,
        "noise": noise,
        "spectrum": spectrum,
        "error": None,
    }
    started = time.perf_counter()
    try:
        model.fit(X, Y)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    if hasattr(model, "pilot_gradients"):
        center_seed = np.random.SeedSequence(model_seed).spawn(2)[0]
        indices = np.random.default_rng(center_seed).choice(600, size=40, replace=False)
        center_truth = truth[indices]
        g = exact_gradient[indices]
        row["exact_gradient_min_norm"] = float(np.linalg.norm(g, axis=1).min())
        linear = normalized_gradients(model.pilot_gradients)
        row["linear_gradient_containment"] = containment(g, linear)
        if m == 1:
            row["linear_chart"] = _local_subspace_metrics(center_truth, linear)[:2]
            Xc, Yc, centers = model.pilot_inputs
            try:
                quadratic, _, _ = local_quadratic_pilot(Xc, Yc, centers)
                row["quadratic_gradient_containment"] = containment(
                    g, normalized_gradients(quadratic)
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                row["quadratic_error"] = f"{type(exc).__name__}: {exc}"
        if hasattr(model, "initial_projectors"):
            row["initial_containment"] = containment(g, model.initial_projectors)
            row["initial_chart"] = _local_subspace_metrics(
                center_truth, model.initial_projectors
            )[:2]
            oracle, _ = initialize_projectors(
                g, model.pilot_mass, model.initial_graph, m
            )
            row["exact_gradient_graph_containment"] = containment(g, oracle)
            row["exact_gradient_graph_chart"] = _local_subspace_metrics(
                center_truth, oracle
            )[:2]
    if row["error"] is None:
        center_truth = truth[model.center_indices_]
        g = exact_gradient[model.center_indices_]
        row["final_containment"] = containment(g, model.projectors_)
        row["final_chart"] = _local_subspace_metrics(center_truth, model.projectors_)[
            :2
        ]
        nearest = model._nearest_center_indices(model._prepare_queries(queries))
        row["query_oracle_chart"] = _local_subspace_metrics(
            query_truth, center_truth[nearest]
        )[:2]
        row["stop_reason"] = model.stop_reason_
        row["linear_residual_max"] = max(
            float(step["linear_relative_residual_max"]) for step in model.trace_
        )
    row["seconds"] = time.perf_counter() - started
    row["process_peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=81000)
    parser.add_argument("--seed-stop", type=int, default=81005)
    args = parser.parse_args()
    if args.seed_stop <= args.seed_start:
        parser.error("seed-stop must exceed seed-start")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = (
        "ADP/core/manifold/ADP_Manifold.py",
        "ADP/engine/manifol_engine/fit.py",
        "ADP/engine/manifol_engine/weights.py",
        "ADP/engine/manifol_engine/graphs.py",
        "ADP/engine/manifol_engine/optimisation.py",
        "experiments/manifold_generalization.py",
        "docs/experiments/manifold_curvature_2026-09-28/diagnostic.py",
    )
    with threadpool_limits(limits=1):
        manifest = {
            "seeds": [args.seed_start, args.seed_stop],
            "source_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in sources
            },
            "threads": threadpool_info(),
            "data": "f=sum(z_k**2)/2, z_k=u_k+c*u_(m+k)**2/2; n=600,d=8,J=P=40",
            "model": (
                "explicit manifold, stop boundary, relative/unit, "
                "frozen local/broad support"
            ),
            "metric": (
                "observable gradient containment at every center; "
                "generator chart descriptive"
            ),
            "rss": "cumulative process ru_maxrss MiB",
        }
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        with (args.out / "runs.jsonl").open("w") as output:
            for seed in range(args.seed_start, args.seed_stop):
                for support in ("local", "broad"):
                    for m in (1, 2, 3):
                        for curvature in (0.35, 0.8):
                            for noise in (0.0, 0.1):
                                for spectrum in ("relative", "unit"):
                                    row = run_case(
                                        seed, support, m, curvature, noise, spectrum
                                    )
                                    output.write(
                                        json.dumps(row, allow_nan=False) + "\n"
                                    )
                                    output.flush()
                                    print(
                                        seed,
                                        support,
                                        m,
                                        curvature,
                                        noise,
                                        spectrum,
                                        row["error"],
                                        row.get("final_containment"),
                                        flush=True,
                                    )


if __name__ == "__main__":
    main()
