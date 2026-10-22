"""Идентифицируемые модели, сетки и метрики без плотных проекторов."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import subspace_angles

METHODS = ("ADP", "MAVE", "MAVE-package", "SIR", "dMAVE", "rMAVE")
GRASSMANN_METHOD = "ADP-Grassmann"
ACCURATE_GRASSMANN_METHOD = "ADP-Grassmann-accurate"
CONSTANT_GRASSMANN_METHOD = "ADP-Grassmann-constant"
ARMIJO_GRASSMANN_METHOD = "ADP-Grassmann-armijo"
QUADRATIC_GRASSMANN_METHOD = "ADP-Grassmann-quadratic"
INTRINSIC_GRASSMANN_METHOD = "ADP-Grassmann-intrinsic"
WARMUP_GRASSMANN_METHOD = "ADP-Grassmann-warmup"
CONJUGATE_GRASSMANN_METHOD = "ADP-Grassmann-conjugate"
RELATIVE_GRASSMANN_METHOD = "ADP-Grassmann-relative"
COMBINED10_GRASSMANN_METHOD = "ADP-Grassmann-combined10"
COMBINED_GRASSMANN_METHOD = "ADP-Grassmann-combined"
SUPPORTED_METHODS = (
    *METHODS,
    GRASSMANN_METHOD,
    ACCURATE_GRASSMANN_METHOD,
    CONSTANT_GRASSMANN_METHOD,
    ARMIJO_GRASSMANN_METHOD,
    QUADRATIC_GRASSMANN_METHOD,
    INTRINSIC_GRASSMANN_METHOD,
    WARMUP_GRASSMANN_METHOD,
    CONJUGATE_GRASSMANN_METHOD,
    RELATIVE_GRASSMANN_METHOD,
    COMBINED10_GRASSMANN_METHOD,
    COMBINED_GRASSMANN_METHOD,
)


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    n: int
    d: int
    m: int = 2
    link: str = "additive"
    rho: float = 0.0
    noise: float = 0.3
    distribution: str = "gaussian"


def catalog(profile: str) -> list[Case]:
    if profile == "smoke":
        return [Case("tiny-linear", 60, 4, 1, "linear"), Case("tiny-multi", 80, 5)]
    if profile == "multi-rank":
        return [Case(f"rank-{m}", 500, 20, m) for m in (3, 4, 5)]
    base = Case("base", 500, 10)
    cases = [base]
    cases += [replace(base, name=f"n-{n}", n=n) for n in (100, 250, 1000, 2000)]
    cases += [replace(base, name=f"d-{d}", d=d) for d in (5, 20, 50, 100)]
    cases += [
        replace(base, name="linear-m1", m=1, link="linear"),
        replace(base, name="sin-m1", m=1, link="sin"),
        replace(base, name="square-m1", m=1, link="square"),
        replace(base, name="product-m2", link="product"),
        replace(base, name="additive-m3", m=3),
    ]
    cases += [replace(base, name=f"corr-{r}", rho=r) for r in (0.5, 0.9, 0.99)]
    cases += [replace(base, name=f"noise-{s}", noise=s) for s in (0.1, 1.0)]
    cases += [
        replace(base, name=dist, distribution=dist)
        for dist in ("t5", "uniform", "mixture")
    ]
    if profile in {"full", "stress"}:
        cases += [
            Case(f"scale-n{n}-d{d}", n, d)
            for n, d in (
                (2000, 50),
                (5000, 50),
                (5000, 200),
                (10000, 100),
                (2000, 500),
                (5000, 500),
                (5000, 1000),
            )
        ]
        cases += [
            Case(f"interaction-d{d}-m{m}-r{rho}", 1000, d, m, rho=rho)
            for d in (20, 100, 200)
            for m in (1, 2, 3)
            for rho in (0.0, 0.9)
        ]
        cases += [
            replace(base, name=f"scarce-n{n}-d{d}", n=n, d=d)
            for n, d in ((120, 100), (220, 200), (520, 500))
        ]
    if profile == "stress":
        cases = [
            Case(f"stress-n{n}-d{d}", n, d)
            for n, d in (
                (20000, 100),
                (10000, 500),
                (10000, 1000),
                (5000, 2000),
                (1000, 1000),
                (500, 1000),
            )
        ]
    if profile not in {"standard", "full", "stress"}:
        raise ValueError(f"unknown profile: {profile}")
    return cases


def generate(case: Case, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entropy = int.from_bytes(hashlib.sha256(case.name.encode()).digest()[:8], "little")
    streams = np.random.SeedSequence([seed, entropy]).spawn(3)
    basis_rng, x_rng, noise_rng = (np.random.default_rng(s) for s in streams)
    truth = np.linalg.qr(basis_rng.normal(size=(case.d, case.m)))[0]
    shape = (case.n, case.d)
    if case.distribution == "gaussian":
        x = x_rng.normal(size=shape)
    elif case.distribution == "t5":
        x = x_rng.standard_t(5, size=shape) * np.sqrt(3 / 5)
    elif case.distribution == "uniform":
        x = x_rng.uniform(-np.sqrt(3), np.sqrt(3), size=shape)
    elif case.distribution == "mixture":
        x = x_rng.normal(size=shape) + x_rng.choice([-1.0, 1.0], size=shape)
        x /= np.sqrt(2)
    else:
        raise ValueError(f"unknown distribution: {case.distribution}")
    for j in range(1, case.d):
        x[:, j] = case.rho * x[:, j - 1] + np.sqrt(1 - case.rho**2) * x[:, j]
    z = x @ truth
    if case.link == "linear":
        signal = z[:, 0]
    elif case.link == "sin":
        signal = np.sin(z[:, 0])
    elif case.link == "square":
        signal = z[:, 0] ** 2
    elif case.link == "product":
        signal = z[:, 0] * z[:, 1]
    elif case.link == "additive":
        signal = np.sin(z[:, 0])
        if case.m >= 2:
            signal += 0.5 * z[:, 1] ** 2
        for j in range(2, case.m):
            signal += 0.35 * np.tanh(z[:, j])
    else:
        raise ValueError(f"unknown link: {case.link}")
    signal = (signal - signal.mean()) / signal.std()
    y = signal + case.noise * noise_rng.normal(size=case.n)
    return x, y, truth


def metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if estimate.shape != truth.shape or not np.isfinite(estimate).all():
        raise ValueError("invalid/nonfinite basis shape")
    if np.linalg.matrix_rank(estimate) != truth.shape[1]:
        raise ValueError("rank-deficient basis")
    q = np.linalg.qr(estimate)[0]
    angles = subspace_angles(q, truth)
    return {
        "projector_distance": float(np.sqrt(np.mean(np.sin(angles) ** 2))),
        "trace_score": float(np.mean(np.cos(angles) ** 2)),
        "max_sin_angle": float(np.sin(angles).max()),
        "mean_angle_deg": float(np.rad2deg(angles).mean()),
        "max_angle_deg": float(np.rad2deg(angles).max()),
        "orthogonality_error": float(
            np.linalg.norm(estimate.T @ estimate - np.eye(q.shape[1]))
        ),
    }
