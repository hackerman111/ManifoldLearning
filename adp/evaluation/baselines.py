from __future__ import annotations

import numpy as np


def fit_statsmodels_dimred(X: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    """Обучает готовый baseline из statsmodels."""

    from statsmodels.regression.dimred import PHD, SAVE, SIR

    cls = {"sir": SIR, "save": SAVE, "phd": PHD}[kind]
    if kind == "sir":
        result = cls(y, X).fit(slice_n=min(20, max(4, X.shape[0] // 8)))
    else:
        result = cls(y, X).fit()
    params = np.asarray(result.params, dtype=float)
    if params.ndim == 1:
        return params
    return params[:, 0]


def fit_sklearn_pls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Обучает PLS baseline из sklearn."""

    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(n_components=1, scale=True)
    model.fit(X, y.reshape(-1, 1))
    return np.asarray(model.x_weights_[:, 0], dtype=float)
