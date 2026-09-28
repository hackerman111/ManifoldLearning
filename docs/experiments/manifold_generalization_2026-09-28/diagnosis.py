"""Small read-only counterfactuals for the frozen manifold generalization run.

Run from the repository root with PYTHONPATH=. and the existing uv environment.
This intentionally changes the estimator only inside UnitySpectrum; no model
defaults or source code used by the frozen runs are modified.
"""

import numpy as np
from threadpoolctl import threadpool_limits

from ADP import ADP_Manifold
from ADP.engine.manifol_engine.utils import bandwidth_floor
from experiments.manifold_generalization import geometry, make_data
from experiments.runner import _local_subspace_metrics


class UnitySpectrum(ADP_Manifold):
    truth_centers: np.ndarray
    error_history: list[tuple[float, float]]
    spectrum_history: list[float]

    @staticmethod
    def _initialize_projectors(gradients, gradient_mass, graph, m):
        basis, spectrum = ADP_Manifold._initialize_projectors(
            gradients, gradient_mass, graph, m
        )
        return basis, np.ones_like(spectrum)

    def _one_step(self, I, U, mass, graph, projectors):
        basis, spectrum, diagnostics = super()._one_step(I, U, mass, graph, projectors)
        self.spectrum_history.append(float(np.median(spectrum[:, -1])))
        self.error_history.append(
            _local_subspace_metrics(self.truth_centers, basis)[:2]
        )
        return basis, np.ones_like(spectrum), diagnostics


class BaselineTrace(UnitySpectrum):
    @staticmethod
    def _initialize_projectors(gradients, gradient_mass, graph, m):
        return ADP_Manifold._initialize_projectors(
            gradients, gradient_mass, graph, m
        )

    def _one_step(self, I, U, mass, graph, projectors):
        basis, spectrum, diagnostics = ADP_Manifold._one_step(
            self, I, U, mass, graph, projectors
        )
        self.spectrum_history.append(float(np.median(spectrum[:, -1])))
        self.error_history.append(
            _local_subspace_metrics(self.truth_centers, basis)[:2]
        )
        return basis, spectrum, diagnostics


def exact_gradient_initialization(seed, m, curvature):
    X, Y, truth, _, _, model_seed = make_data(seed, 600, 8, m, curvature, 0.0)
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean()
    center_seed = np.random.SeedSequence(model_seed).spawn(2)[0]
    indices = np.random.default_rng(center_seed).choice(600, size=40, replace=False)
    centers = Xc[indices]
    model = ADP_Manifold(m, estimator="manifold", N_loc=300, N_lin=300,
        N_J=40, N_phi=40, N_manifold=30, seed=model_seed)
    h_lin = model._search_bandwidth(Xc, centers, 300, lower=bandwidth_floor(Xc))
    fitted, mass, _ = model._local_gradients(Xc, Yc, centers, h_lin)
    h_graph = model._search_bandwidth(
        centers, centers, 30, lower=bandwidth_floor(centers)
    )
    graph = model._build_manifold_graph(centers, None, None, h_graph, 1.0)
    rotation_seed = np.random.SeedSequence(seed).spawn(5)[1]
    rotation, _ = np.linalg.qr(np.random.default_rng(rotation_seed).normal(size=(8, 8)))
    exact = geometry(X, rotation, m, curvature)[2][indices]
    fitted_basis, _ = model._initialize_projectors(fitted, mass, graph, m)
    exact_basis, _ = model._initialize_projectors(exact, mass, graph, m)
    return (
        _local_subspace_metrics(truth[indices], fitted_basis)[0],
        _local_subspace_metrics(truth[indices], exact_basis)[0],
    )


def spectrum_fit(seed, noise, model_type):
    X, Y, truth, queries, query_truth, model_seed = make_data(
        seed, 600, 8, 3, 0.0, noise
    )
    center_seed = np.random.SeedSequence(model_seed).spawn(2)[0]
    indices = np.random.default_rng(center_seed).choice(600, size=40, replace=False)
    model = model_type(3, estimator="manifold", N_loc=300, N_lin=300,
        N_J=40, N_phi=40, N_manifold=30, sync_steps=3,
        lambda_manifold=0.5, cg_tol=1e-6, solver="cg",
        seed=model_seed, scale_boundary="stop")
    model.truth_centers = truth[indices]
    model.error_history = []
    model.spectrum_history = []
    model.fit(X, Y)
    assert np.array_equal(indices, model.center_indices_)
    nearest = model._nearest_center_indices(model._prepare_queries(queries))
    center = _local_subspace_metrics(truth[indices], model.projectors_)
    query = _local_subspace_metrics(query_truth, model.projectors_[nearest])
    return center[0], center[1], query[1], model.n_scales_, model


def main():
    with threadpool_limits(limits=1):
        for m in (1, 2, 3):
            for curvature in (0.35, 0.8):
                errors = np.asarray([
                    exact_gradient_initialization(seed, m, curvature)
                    for seed in range(81000, 81005)
                ])
                print("exact_init", m, curvature,
                    "fitted/exact median RMS", np.median(errors, axis=0))
        for noise in (0.0, 0.1):
            for seed in range(81000, 81005):
                baseline = spectrum_fit(seed, noise, BaselineTrace)
                unity = spectrum_fit(seed, noise, UnitySpectrum)
                print("spectrum_pair", seed, noise, "baseline", baseline[:4],
                    "unity", unity[:4])
                if seed == 81000 and noise == 0.0:
                    for label, result in (("baseline", baseline), ("unity", unity)):
                        print("trajectory", label,
                            [(round(e[0], 4), round(s, 5)) for e, s in zip(
                                result[4].error_history,
                                result[4].spectrum_history,
                                strict=True,
                            )])


if __name__ == "__main__":
    main()
