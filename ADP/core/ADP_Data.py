# ruff: noqa: RUF002

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ADP_Data:
    X: np.ndarray
    Y: np.ndarray
    beta_true: np.ndarray
    beta_k: np.ndarray
    beta_init: np.ndarray
    a_k: np.ndarray
    rho_k: np.ndarray
    x_j: np.ndarray
    n: int
    d: int

    def __post_init__(self):
        """Синхронизировать размеры данных с фактическими массивами.

        Источник — переданные ``X`` и истинный индекс; эти размеры далее
        используются solver-ом как ``n`` и ``d``.
        """
        self.n = self.X.shape[1]
        self.d = self.beta_true.shape[0]

    def Update_beta_k(self, beta_k):
        """Добавить очередной индекс в историю внешних ADP-шагов."""
        self.beta_k += [beta_k]

    def Update_a_k(self, a_k):
        """Сохранить очередной коэффициент масштаба локализации."""
        self.a_k += [a_k]

    def Update_rho_k(self, rho_k):
        """Сохранить очередной параметр анизотропии."""
        self.rho_k += [rho_k]

    def Calculate_metric(self, cur_beta):
        """Вычислить ошибку индекса или подпространства относительно истины.

        Для multi-index используется квадрат нормы остатка после ортогональной
        проекции на истинное подпространство; для single-index — знаково-
        чувствительное скалярное сравнение, совместимое со старым API.
        """
        if self.beta_true.shape[1] != 1:
            # (SEDRqua) из multiindex.tex без матриц-проекторов размера d x d.
            residual = cur_beta - self.beta_true @ (self.beta_true.T @ cur_beta)
            return float(np.square(residual).sum())

        else:
            return 1 - np.dot(self.beta_true, cur_beta)
