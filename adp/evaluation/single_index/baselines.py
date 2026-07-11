from __future__ import annotations

import numpy as np

from ...common.utils import unit_vector
from ..baselines import fit_linear_fallback, fit_sklearn_pls, fit_statsmodels_dimred


class BaselineUnavailable(RuntimeError):
    """An optional scientific baseline is not installed or implemented."""


def fit_baseline(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if method == "random_direction":
        return unit_vector(np.random.default_rng(seed).normal(size=X.shape[1]))
    if method == "ols":
        return unit_vector(fit_linear_fallback(X, y))
    if method.startswith("statsmodels_"):
        kind = method.removeprefix("statsmodels_")
        if kind not in {"sir", "save", "phd"}:
            raise ValueError(f"unknown statsmodels baseline: {method}")
        return unit_vector(fit_statsmodels_dimred(X, y, kind))
    if method == "sklearn_pls":
        return unit_vector(fit_sklearn_pls(X, y))
    if method in {"opg", "ade", "mave", "rmave"}:
        raise BaselineUnavailable(
            f"baseline {method} requires an explicit optional adapter"
        )
    raise ValueError(f"unknown single-index baseline: {method}")
