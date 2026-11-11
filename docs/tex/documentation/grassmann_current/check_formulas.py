"""Check the English document against covariance formulas and full outer fits.

Run from the repository root:
    python -m tex.documentation.grassmann_current.check_formulas
Solver differential identities have independent finite-difference coverage in
tests/test_grassman.py; this check concentrates on the surrounding algorithm.
"""

from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.engine.common.calculus import generate_isotropic_proj, search_bandwidth
from ADP.engine.common.statistic import calculate_statistics
from ADP.engine.common.weights import calculate_multi_weight
from ADP.solver.grassman import solve


def moments(X, y, centers, B, spectrum, h, alpha, tensor, phi):
    delta = X[None, :, :] - centers[:, None, :]
    z = delta @ B
    squared = np.sum(delta**2, axis=-1)
    residual = np.maximum(0, squared - np.sum(z**2, axis=-1))
    principal = np.sum(spectrum * z**2, axis=-1)
    argument = (
        alpha**2 * (residual if tensor == "orthogonal" else squared) + principal
    ) / h**2
    weights = np.maximum(1 - argument**2, 0)
    mass = weights.sum(axis=1)
    a = weights / mass[:, None]
    mean, response_mean = a @ X, a @ y
    dx, dy = X[None] - mean[:, None], y[None] - response_mean[:, None]
    covariance = np.einsum("ji,jia,jib->jab", a, dx, dx)
    cross = np.einsum("ji,jia,ji->ja", a, dx, dy)
    U = phi @ covariance
    I = (phi @ cross[..., None]).squeeze(-1)
    return weights, mass, mean, response_mean, U, I


def main():
    rng = np.random.default_rng(20261023)
    X = rng.normal(size=(160, 6))
    truth = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    latent = X @ truth
    y = np.sin(latent[:, 0]) + 0.5 * latent[:, 1] ** 2
    y += 0.03 * rng.normal(size=len(X))
    centers = X[:9]
    phi = generate_isotropic_proj(rng, len(centers), 5, X.shape[1])
    batches = 0
    for tensor in ("orthogonal", "full"):
        for normalized in (True, False):
            expected = moments(
                X, y, centers, truth, np.array([1, 0.4]), 3.5, 0.6, tensor, phi
            )
            W, mass, mean, S, U, I = expected
            live_W = np.concatenate(
                [
                    block
                    for _, block in calculate_multi_weight(
                        X,
                        centers,
                        truth,
                        np.array([1, 0.4]),
                        3.5,
                        0.6,
                        ADP_Config().kernel,
                        tensor=tensor,
                    )
                ]
            )
            assert_allclose(live_W, W, atol=3e-14)
            stat = calculate_statistics(X, y, W, phi, normalized=normalized)
            assert_allclose(stat.mass, mass, atol=3e-13)
            assert_allclose(stat.mean, mean, atol=2e-14)
            assert_allclose(stat.S, S, atol=2e-14)
            assert_allclose(stat.n_eff, mass**2 / np.sum(W**2, axis=1))
            assert_allclose(
                stat.U, U if normalized else mass[:, None, None] * U, atol=2e-12
            )
            assert_allclose(stat.I, I if normalized else mass[:, None] * I, atol=2e-12)
            batches += 1

    cfg = ADP_Config(
        seed=9181, N_loc=14, N_lin=20, N_J=12, N_phi=5, outer_steps=3, h_min=0.01
    )
    updates = 0
    cases = (
        ("orthogonal", "all", True),
        ("full", "all", True),
        ("orthogonal", "exclude_centers", False),
    )
    for tensor, training, redraw in cases:
        current = replace(
            cfg, multi_tensor=tensor, training_set=training, redraw_directions=redraw
        )
        model = ADP_multi_index(2, current, ADP_solver(solve)).fit(X, y)
        center_seed, _, direction_seed = np.random.SeedSequence(current.seed).spawn(3)
        indices = np.random.default_rng(center_seed).choice(
            len(X), current.N_J, replace=False
        )
        centers = X[indices]
        keep = np.ones(len(X), dtype=bool)
        if training == "exclude_centers":
            keep[indices] = False
        Xt, yt = X[keep], y[keep]
        B, spectrum = model.result_.beta_init, np.ones(2)

        # Independent weighted affine fits verify initializer ridge and PCA.
        distance = np.sum((Xt[None] - centers[:, None]) ** 2, axis=-1)
        h_lin = search_bandwidth(
            distance, current.N_lin, current.kernel, lower=np.finfo(float).eps
        )
        init_weights = np.maximum(1 - (distance / h_lin**2) ** 2, 0)
        gradients = []
        for center, w in zip(centers, init_weights, strict=True):
            A = np.column_stack((np.ones(len(Xt)), Xt - center))
            ridge = np.column_stack(
                (np.zeros(X.shape[1]), np.sqrt(current.local_ridge) * np.eye(6))
            )
            design = np.vstack((np.sqrt(w)[:, None] * A, ridge))
            rhs = np.concatenate((np.sqrt(w) * yt, np.zeros(6)))
            gradients.append(np.linalg.lstsq(design, rhs, rcond=None)[0][1:])
        _, _, right = np.linalg.svd(
            np.sqrt(init_weights.sum(axis=1))[:, None] * gradients, full_matrices=False
        )
        assert_allclose(B @ B.T, right[:2].T @ right[:2], atol=2e-8)

        stream, fixed, errors = np.random.default_rng(direction_seed), None, []
        for row in model.trace_:
            if fixed is None or redraw:
                phi = generate_isotropic_proj(stream, len(centers), 5, 6)
                fixed = phi
            else:
                phi = fixed
            W, mass, _, S, U, I = moments(
                Xt,
                yt,
                centers,
                B,
                spectrum,
                row["h"],
                row["factor"],
                row["tensor"],
                phi,
            )
            assert np.mean(mass) >= current.N_loc - 1e-8
            score = float(np.sum((y[indices] - S) ** 2))
            assert_allclose(row["err"], score, rtol=2e-12)
            result = solve(B.T, U, I, mass=mass, lambda_prox=current.lambda_penalty)
            Y, g = result.index.T, result.coefficients
            next_B, next_spectrum = row["basis"], np.array(row["eigenvalues"])
            assert_allclose(Y @ Y.T, next_B @ next_B.T, atol=3e-7)
            singular = np.linalg.svd(np.sqrt(mass)[:, None] * g, compute_uv=False)
            assert_allclose(next_spectrum, (singular / singular[0]) ** 2, atol=2e-7)
            assert_allclose(next_B.T @ next_B, np.eye(2), atol=3e-14)
            errors.append(score)
            B, spectrum = next_B, next_spectrum
            updates += 1
        selected = int(np.argmin(errors))
        assert model.effective_parameters_["selected_iteration"] == selected
        assert_allclose(model.basis_, model.trace_[selected]["basis"], atol=0)
        assert_allclose(model.transform(X), X @ model.basis_, atol=0)
        assert len(model.trace_) == current.outer_steps
        assert model.result_.stop_reason == "outer_steps"
    print(
        f"PASS: {batches} kernel/moment batches; {len(cases)} full fits; "
        f"{updates} outer-update replays; initializer, score, spectrum, "
        "best selection and transform formulas"
    )


if __name__ == "__main__":
    main()
