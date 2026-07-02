from __future__ import annotations

import numpy as np


def fit_statsmodels_dimred(
    X: np.ndarray,  # Матрица наблюдений n x d.
    y: np.ndarray,  # Вектор ответов длины n.
    kind: str,  # Тип готового метода: sir, save или phd.
) -> np.ndarray:
    """Обучает готовый baseline из statsmodels.

    Вход:
        X: матрица наблюдений.
        y: вектор ответов.
        kind: имя метода statsmodels.
    Выход:
        Оцененный EDR-вектор.
    """

    try:
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
    except Exception:
        return fit_linear_fallback(X, y)


def fit_sklearn_pls(
    X: np.ndarray,  # Матрица наблюдений n x d.
    y: np.ndarray,  # Вектор ответов длины n.
) -> np.ndarray:
    """Обучает PLS baseline из sklearn.

    Вход:
        X: матрица наблюдений.
        y: вектор ответов.
    Выход:
        Первый PLS-вектор весов или fallback.
    """

    try:
        from sklearn.cross_decomposition import PLSRegression

        model = PLSRegression(n_components=1, scale=True)
        model.fit(X, y.reshape(-1, 1))
        return np.asarray(model.x_weights_[:, 0], dtype=float)
    except Exception:
        return fit_linear_fallback(X, y)


def fit_linear_fallback(
    X: np.ndarray,  # Матрица наблюдений n x d.
    y: np.ndarray,  # Вектор ответов длины n.
) -> np.ndarray:
    """Строит простой линейный fallback для benchmark baseline.

    Вход:
        X: матрица наблюдений.
        y: вектор ответов.
    Выход:
        Вектор least-squares направления.
    """

    x_centered = X - X.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    try:
        beta, *_ = np.linalg.lstsq(x_centered, y_centered, rcond=None)
    except Exception:
        beta = x_centered.T @ y_centered
    return np.asarray(beta, dtype=float)
