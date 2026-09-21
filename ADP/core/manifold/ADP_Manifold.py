# ruff: noqa: RUF002

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
from scipy.sparse import csr_matrix

from ...engine import manifol_engine as engine
from . import ADP_Manifold_utils as model_utils
from .ADP_Manifold_result import ADP_Manifold_result


class ADP_Manifold:
    """Тонкий API-фасад structure-adaptive manifold-ADP.

    ``X`` имеет форму ``(n, d)``, ``Y`` — ``(n,)``. Численная математика,
    локальные статистики и solver находятся в ``ADP.engine``; этот класс
    хранит конфигурацию, опубликованное состояние и совместимые API-адаптеры.
    """

    def __init__(
        self,
        index_dim: int,
        *,
        N_loc: int = 15,
        N_lin: int | None = None,
        N_J: int | None = None,
        N_phi: int | None = None,
        N_manifold: int | None = None,
        lambda_manifold: float = 1.0,
        sync_steps: int = 5,
        a: float | None = None,
        h_min: float | None = None,
        batch_size: int = 32,
        seed: int = 42,
        cg_tol: float = 1e-8,
        cg_maxiter: int | None = None,
        solver: Literal["cg", "hybrid"] = "cg",
        dense_max_unknowns: int = 256,
        dense_max_bytes: int = 64 * 1024**2,
        scale_boundary: Literal["raise", "stop"] = "raise",
    ) -> None:
        """Настроить manifold-ADP с локальным размером подпространства ``m``."""
        self.index_dim = self._integer("index_dim", index_dim, minimum=1)
        self.N_loc = self._integer("N_loc", N_loc, minimum=1)
        self.N_lin = self._optional_integer("N_lin", N_lin, minimum=1)
        self.N_J = self._optional_integer("N_J", N_J, minimum=1)
        self.N_phi = self._optional_integer("N_phi", N_phi, minimum=1)
        self.N_manifold = self._optional_integer("N_manifold", N_manifold, minimum=1)
        self.lambda_manifold = self._finite_float(
            "lambda_manifold", lambda_manifold, minimum=0.0
        )
        self.sync_steps = self._integer("sync_steps", sync_steps, minimum=1)
        self.a = self._optional_float("a", a, minimum=1.0, strict=True)
        self.h_min = self._optional_float("h_min", h_min, minimum=0.0, strict=True)
        self.batch_size = self._integer("batch_size", batch_size, minimum=1)
        self.seed = self._integer("seed", seed, minimum=0)
        self.cg_tol = self._finite_float("cg_tol", cg_tol, minimum=0.0, strict=True)
        self.cg_maxiter = self._optional_integer("cg_maxiter", cg_maxiter, minimum=1)
        self.solver = model_utils.validate_solver(solver)
        self.dense_max_unknowns = self._integer(
            "dense_max_unknowns", dense_max_unknowns, minimum=0
        )
        self.dense_max_bytes = self._integer(
            "dense_max_bytes", dense_max_bytes, minimum=0
        )
        self.scale_boundary = model_utils.validate_scale_boundary(scale_boundary)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Self:
        """Оценить локальные EDR-подпространства и вернуть ``self``."""
        state = engine.fit(self, X, Y)
        self.centers_ = state.centers
        self.projectors_ = state.projectors
        self.eigenvalues_ = state.eigenvalues
        self.gradients_ = state.gradients
        self.center_values_ = state.center_values
        self.x_offset_ = state.x_offset
        self.trace_ = state.trace
        self.bandwidth_ = state.bandwidth
        self.linear_bandwidth_ = state.linear_bandwidth
        self.manifold_bandwidth_ = state.manifold_bandwidth
        self.alpha_ = state.alpha
        self.manifold_alpha_ = state.manifold_alpha
        self.n_scales_ = state.n_scales
        self.center_indices_ = state.center_indices
        self.effective_config_ = state.effective_config
        self.stop_reason_ = state.stop_reason
        self.scale_boundary_ = self.scale_boundary
        self.result_ = ADP_Manifold_result(
            centers=self.centers_,
            projectors=self.projectors_,
            eigenvalues=self.eigenvalues_,
            gradients=self.gradients_,
            center_values=self.center_values_,
            trace=self.trace_,
            bandwidth=self.bandwidth_,
            linear_bandwidth=self.linear_bandwidth_,
            manifold_bandwidth=self.manifold_bandwidth_,
            alpha=self.alpha_,
            manifold_alpha=self.manifold_alpha_,
            n_scales=self.n_scales_,
            center_indices=self.center_indices_,
            effective_config=self.effective_config_,
            stop_reason=self.stop_reason_,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Вернуть локальные координаты ближайших chart-центров."""
        return engine.transform(self, X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказать отклик ближайшей локально-линейной моделью."""
        return engine.predict(self, X)

    def _prepare_queries(self, X: np.ndarray) -> np.ndarray:
        """Проверить и центрировать точки для ``transform`` и ``predict``."""
        return model_utils.prepare_queries(X, getattr(self, "centers_", None))

    def _nearest_center_indices(self, X: np.ndarray) -> np.ndarray:
        return engine.nearest_center_indices(self, X)

    def _local_coordinates(self, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return engine.local_coordinates(self, X, indices)

    def _effective_config(self, X: np.ndarray) -> dict[str, float | int]:
        return model_utils.effective_config(
            X,
            self.index_dim,
            self.N_loc,
            self.N_lin,
            self.N_J,
            self.N_phi,
            self.N_manifold,
            self.a,
            self.h_min,
        )

    @staticmethod
    def _prepare_xy(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return model_utils.prepare_xy(X, Y)

    @staticmethod
    def _pairwise_distance2(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return engine.pairwise_distance2(X, centers)

    @staticmethod
    def _kernel(argument: np.ndarray) -> np.ndarray:
        return engine.kernel(argument)

    @staticmethod
    def _bandwidth_floor(values: np.ndarray) -> float:
        return engine.bandwidth_floor(values)

    def _weight_block(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        projectors: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        h: float,
        alpha: float,
        start: int,
    ) -> np.ndarray:
        return engine.weight_block(
            self, X, centers, projectors, eigenvalues, h, alpha, start
        )

    def _mean_mass(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        projectors: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        h: float,
        alpha: float,
    ) -> float:
        return engine.mean_mass(self, X, centers, projectors, eigenvalues, h, alpha)

    def _search_bandwidth(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        target: int,
        *,
        lower: float,
    ) -> float:
        return engine.search_bandwidth(self, X, centers, target, lower=lower)

    def _search_anisotropy(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        projectors: np.ndarray,
        eigenvalues: np.ndarray,
        h: float,
        target: int,
        *,
        kind: str,
    ) -> float:
        return engine.search_anisotropy(
            self, X, centers, projectors, eigenvalues, h, target, kind=kind
        )

    def _feasible_scale(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        projectors: np.ndarray,
        eigenvalues: np.ndarray,
        proposed: float,
        previous: float,
    ) -> tuple[float, bool]:
        return engine.feasible_scale(
            self, X, centers, projectors, eigenvalues, proposed, previous
        )

    def _local_gradients(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        centers: np.ndarray,
        h: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return engine.local_gradients(self, X, Y, centers, h)

    def _build_manifold_graph(
        self,
        centers: np.ndarray,
        projectors: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        h: float,
        alpha: float,
    ) -> csr_matrix:
        return engine.build_manifold_graph(
            self, centers, projectors, eigenvalues, h, alpha
        )

    @staticmethod
    def _initialize_projectors(
        gradients: np.ndarray,
        gradient_mass: np.ndarray,
        graph: csr_matrix,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return engine.initialize_projectors(gradients, gradient_mass, graph, m)

    @staticmethod
    def _random_directions(
        rng: np.random.Generator, J: int, P: int, d: int
    ) -> np.ndarray:
        return engine.random_directions(rng, J, P, d)

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
        return engine.calculate_statistics(
            self, X, Y, centers, directions, projectors, eigenvalues, h, alpha
        )

    def _one_step(
        self,
        I: np.ndarray,
        U: np.ndarray,
        mass: np.ndarray,
        graph: csr_matrix,
        projectors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        return engine.one_step(self, I, U, mass, graph, projectors)

    @staticmethod
    def _local_slopes(
        I: np.ndarray,
        U: np.ndarray,
        projector: np.ndarray,
        target: int,
    ) -> np.ndarray:
        return engine.local_slopes(I, U, projector, target)

    def _build_B_system(
        self,
        U: np.ndarray,
        I: np.ndarray,
        mass: np.ndarray,
        weights: np.ndarray,
        source_projectors: np.ndarray,
        slopes: np.ndarray,
    ) -> tuple[Any, Any, np.ndarray]:
        return engine.build_B_system(
            self, U, I, mass, weights, source_projectors, slopes
        )

    @staticmethod
    def _penalty_action(
        B: np.ndarray,
        source_projectors: np.ndarray,
        normalized_weights: np.ndarray,
    ) -> np.ndarray:
        return engine.penalty_action(B, source_projectors, normalized_weights)

    def _solve_B(
        self,
        operator: Any,
        preconditioner: Any,
        rhs: np.ndarray,
        initial: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        return engine.solve_B(self, operator, preconditioner, rhs, initial)

    @staticmethod
    def _recover_projector(
        B: np.ndarray,
        slopes: np.ndarray,
        gamma: np.ndarray,
        target: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return engine.recover_projector(B, slopes, gamma, target)

    def _objective(
        self,
        B: np.ndarray,
        U: np.ndarray,
        I: np.ndarray,
        gamma: np.ndarray,
        slopes: np.ndarray,
        source_projectors: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, float]:
        return engine.objective(
            self, B, U, I, gamma, slopes, source_projectors, weights
        )

    @staticmethod
    def _projector_distance(left: np.ndarray, right: np.ndarray) -> float:
        return engine.projector_distance(left, right)

    @staticmethod
    def _check_identified(
        singular_values: np.ndarray,
        shape: tuple[int, int],
        m: int,
        target: int,
    ) -> None:
        model_utils.require_identified(singular_values, shape, m, target)

    @staticmethod
    def _orient_rows(basis: np.ndarray) -> np.ndarray:
        return engine.orient_rows(basis)

    @staticmethod
    def _trace_entry(
        phase: str,
        iteration: int,
        h: float,
        h_manifold: float,
        alpha: float,
        alpha_manifold: float,
        mass: np.ndarray,
        n_eff: np.ndarray,
        function_edges: int,
        graph: csr_matrix,
        diagnostics: dict[str, float | int],
    ) -> dict[str, float | int | str]:
        return engine.trace_entry(
            phase,
            iteration,
            h,
            h_manifold,
            alpha,
            alpha_manifold,
            mass,
            n_eff,
            function_edges,
            graph,
            diagnostics,
        )

    @staticmethod
    def _integer(name: str, value: int, *, minimum: int) -> int:
        return model_utils.validate_integer(name, value, minimum)

    @classmethod
    def _optional_integer(
        cls, name: str, value: int | None, *, minimum: int
    ) -> int | None:
        return None if value is None else cls._integer(name, value, minimum=minimum)

    @staticmethod
    def _finite_float(
        name: str,
        value: float,
        *,
        minimum: float,
        strict: bool = False,
    ) -> float:
        return model_utils.validate_float(name, value, minimum, strict)

    @classmethod
    def _optional_float(
        cls,
        name: str,
        value: float | None,
        *,
        minimum: float,
        strict: bool = False,
    ) -> float | None:
        if value is None:
            return None
        return cls._finite_float(name, value, minimum=minimum, strict=strict)


ADP_manifold = ADP_Manifold

__all__ = ["ADP_Manifold", "ADP_manifold"]
