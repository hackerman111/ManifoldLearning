"""Deterministic report-inspired synthetic data, with independent RNG streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import Case


@dataclass(frozen=True, slots=True)
class Problem:
    X: np.ndarray
    Y: np.ndarray
    truth: np.ndarray
    seed_streams: dict[str, int]


def generate(case: Case, seed: int) -> Problem:
    """Generate paired X/Y/truth arrays; seed streams cannot consume each other."""
    data_ss, truth_ss, noise_ss = np.random.SeedSequence(seed).spawn(3)
    data_seed = int(data_ss.generate_state(1, dtype=np.uint32)[0])
    truth_seed = int(truth_ss.generate_state(1, dtype=np.uint32)[0])
    noise_seed = int(noise_ss.generate_state(1, dtype=np.uint32)[0])

    data_rng = np.random.default_rng(data_seed)
    truth_rng = np.random.default_rng(truth_seed)
    noise_rng = np.random.default_rng(noise_seed)

    coordinates = np.arange(case.d)
    covariance = np.power(0.5, np.abs(coordinates[:, None] - coordinates[None, :]))
    chol = np.linalg.cholesky(covariance)
    X = data_rng.standard_normal((case.n, case.d)) @ chol.T

    raw_truth = truth_rng.standard_normal((case.d, case.index_dim))
    truth, _ = np.linalg.qr(raw_truth, mode="reduced")
    z = X @ truth
    z1, z2 = z[:, 0], z[:, 1]
    if case.link == "additive":
        signal = np.square(z1) + np.sin(2.5 * z2)
    elif case.link == "multiplicative":
        signal = z1 * np.sin(2.5 * z2)
    elif case.link == "aniso_curv":
        signal = np.square(z1) + np.sin(2.5 * z2) + 0.3 * np.square(z2)
    else:
        raise ValueError(f"unknown link: {case.link}")
    signal = (signal - signal.mean()) / signal.std(ddof=0)
    Y = signal + 0.2 * noise_rng.standard_normal(case.n)
    return Problem(
        X=X,
        Y=Y,
        truth=truth,
        seed_streams={"data": data_seed, "truth": truth_seed, "noise": noise_seed},
    )
