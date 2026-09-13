from __future__ import annotations

import math
from typing import Any, Literal, Self

import numpy as np
from scipy.sparse import csr_matrix, linalg as sparse_linalg


class ADP_Manifold:
    """Structure-adaptive ADP для семейства локальных EDR-подпространств.

    ``X`` имеет форму ``(n, d)``, ``Y`` — ``(n,)``. После ``fit`` строки
    ``projectors_[j]`` образуют ортонормированный базис локального
    ``index_dim``-мерного подпространства около ``centers_[j]``.
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
        scale_boundary: Literal["raise", "stop"] = "raise",
    ) -> None:
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
        if scale_boundary not in {"raise", "stop"}:
            raise ValueError("scale_boundary must be 'raise' or 'stop'")
        self.scale_boundary = scale_boundary

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Self:
        """Оценить локальные EDR-подпространства и вернуть ``self``.

        Численная ошибка, вырожденная локальная задача или недостижимая
        локальная масса приводят к ``RuntimeError`` вместо скрытого fallback.
        """
        X, Y = self._prepare_xy(X, Y)
        n, d = X.shape
        config = self._effective_config(X)
        m = self.index_dim
        N_lin = int(config["N_lin"])
        N_J = int(config["N_J"])
        N_phi = int(config["N_phi"])
        N_manifold = int(config["N_manifold"])
        a = float(config["a"])
        h_min = float(config["h_min"])

        # NUMERICAL: единое центрирование защищает Gram-формулу от offset-cancelation.
        x_offset = X.mean(axis=0)
        y_offset = float(Y.mean())
        Xc = X - x_offset
        Yc = Y - y_offset

        center_seed, direction_seed = np.random.SeedSequence(self.seed).spawn(2)
        center_rng = np.random.default_rng(center_seed)
        direction_rng = np.random.default_rng(direction_seed)
        center_indices = center_rng.choice(n, size=N_J, replace=False)
        centers = Xc[center_indices].copy()

        h_floor = self._bandwidth_floor(Xc)
        h_lin = self._search_bandwidth(Xc, centers, N_lin, lower=h_floor)
        gradients, gradient_mass, center_values = self._local_gradients(
            Xc, Yc, centers, h_lin
        )

        h_manifold = self._search_bandwidth(
            centers,
            centers,
            N_manifold,
            lower=self._bandwidth_floor(centers),
        )
        manifold_graph = self._build_manifold_graph(
            centers, None, None, h_manifold, 1.0
        )
        projectors, eigenvalues = self._initialize_projectors(
            gradients, gradient_mass, manifold_graph, m
        )

        h = self._search_bandwidth(Xc, centers, self.N_loc, lower=h_min)
        directions = self._random_directions(direction_rng, len(centers), N_phi, d)
        I, U, mass, n_eff, function_edges = self._calculate_statistics(
            Xc, Yc, centers, directions, None, None, h, 1.0
        )

        trace: list[dict[str, float | int | str]] = []
        for iteration in range(self.sync_steps):
            projectors, eigenvalues, diagnostics = self._one_step(
                I, U, mass, manifold_graph, projectors
            )
            trace.append(
                self._trace_entry(
                    "sync",
                    iteration,
                    h,
                    h_manifold,
                    1.0,
                    1.0,
                    mass,
                    n_eff,
                    function_edges,
                    manifold_graph,
                    diagnostics,
                )
            )

        alpha = 1.0
        alpha_manifold = 1.0
        scale = 0
        stop_reason = "h_min"
        while h / a >= h_min:
            proposed_h = h / a
            if self.scale_boundary == "stop":
                if (
                    self._mean_mass(
                        centers, centers, projectors, eigenvalues, h_manifold, 0.0
                    )
                    < N_manifold
                ):
                    stop_reason = "manifold_mass_boundary"
                    break
                # ESTIMATOR: отдельное правило остановки на границе массы.
                proposed_h, boundary = self._feasible_scale(
                    Xc, centers, projectors, eigenvalues, proposed_h, h
                )
                if boundary:
                    stop_reason = "function_mass_boundary"
                if proposed_h >= h:
                    break
            h = proposed_h
            scale += 1
            alpha = self._search_anisotropy(
                Xc,
                centers,
                projectors,
                eigenvalues,
                h,
                self.N_loc,
                kind="function",
            )
            directions = self._random_directions(direction_rng, len(centers), N_phi, d)
            I, U, mass, n_eff, function_edges = self._calculate_statistics(
                Xc,
                Yc,
                centers,
                directions,
                projectors,
                eigenvalues,
                h,
                alpha,
            )

            alpha_manifold = self._search_anisotropy(
                centers,
                centers,
                projectors,
                eigenvalues,
                h_manifold,
                N_manifold,
                kind="manifold",
            )
            manifold_graph = self._build_manifold_graph(
                centers,
                projectors,
                eigenvalues,
                h_manifold,
                alpha_manifold,
            )
            projectors, eigenvalues, diagnostics = self._one_step(
                I, U, mass, manifold_graph, projectors
            )
            trace.append(
                self._trace_entry(
                    "scale",
                    scale,
                    h,
                    h_manifold,
                    alpha,
                    alpha_manifold,
                    mass,
                    n_eff,
                    function_edges,
                    manifold_graph,
                    diagnostics,
                )
            )
            if stop_reason == "function_mass_boundary":
                break

        trace[-1]["stop_reason"] = stop_reason
        self.centers_ = centers + x_offset
        self.projectors_ = projectors
        self.eigenvalues_ = eigenvalues
        self.gradients_ = gradients
        self.center_values_ = center_values + y_offset
        self.x_offset_ = x_offset
        self.trace_ = trace
        self.bandwidth_ = float(h)
        self.linear_bandwidth_ = float(h_lin)
        self.manifold_bandwidth_ = float(h_manifold)
        self.alpha_ = float(alpha)
        self.manifold_alpha_ = float(alpha_manifold)
        self.n_scales_ = scale
        self.center_indices_ = center_indices
        self.effective_config_ = config
        self.stop_reason_ = stop_reason
        self.scale_boundary_ = self.scale_boundary
        return self

    def _feasible_scale(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        projectors: np.ndarray,
        eigenvalues: np.ndarray,
        proposed: float,
        previous: float,
    ) -> tuple[float, bool]:
        """Уточнить границу средней массы для фиксированной текущей геометрии.

        При alpha=0 масса максимальна. Возвращается допустимая сторона
        скобки; относительная точность sqrt(eps), без изменения N_loc.
        Это граница доступности весов, не сертификат статистической точности.
        """

        def feasible(h: float) -> bool:
            return (
                self._mean_mass(X, centers, projectors, eigenvalues, h, 0.0)
                >= self.N_loc
            )

        if feasible(proposed):
            return proposed, False
        if not feasible(previous):
            # Новая геометрия могла сделать недопустимым даже прежний масштаб.
            raise RuntimeError(
                "updated geometry has no feasible scale in the current bracket"
            )
        low, high = proposed, previous
        while high - low > math.sqrt(np.finfo(float).eps) * high:
            middle = (low + high) / 2.0
            if feasible(middle):
                high = middle
            else:
                low = middle
        return high, True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Вернуть локальные координаты ``(n, m)`` ближайших chart-центров.

        Координаты относятся к локальному базису выбранного центра и поэтому
        не задают единый глобально ориентированный chart.
        """
        queries = self._prepare_queries(X)
        indices = self._nearest_center_indices(queries)
        return self._local_coordinates(queries, indices)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказать отклик ближайшей локально-линейной моделью.

        Pilot-gradient проецируется в итоговое локальное EDR-подпространство.
        Это не меняет manifold-fit и задаёт отдельный nearest-center predictor.
        """
        queries = self._prepare_queries(X)
        indices = self._nearest_center_indices(queries)
        coordinates = self._local_coordinates(queries, indices)
        slopes = np.einsum(
            "nmd,nd->nm",
            self.projectors_[indices],
            self.gradients_[indices],
            optimize=True,
        )
        return self.center_values_[indices] + np.einsum(
            "nm,nm->n", coordinates, slopes, optimize=True
        )

    def _prepare_queries(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "centers_"):
            raise RuntimeError("fit must be called before transform or predict")
        array = np.asarray(X)
        if np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(
            array.dtype, np.number
        ):
            raise TypeError("X must contain real numeric values")
        if array.dtype.itemsize > np.dtype(np.float64).itemsize:
            raise TypeError("X has a dtype wider than float64; cast it explicitly")
        if array.ndim != 2 or array.shape[1] != self.centers_.shape[1]:
            raise ValueError(f"X must have shape (n, {self.centers_.shape[1]})")
        result = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("X must contain only finite values")
        return result

    def _nearest_center_indices(self, X: np.ndarray) -> np.ndarray:
        indices = np.empty(len(X), dtype=np.intp)
        centered_centers = self.centers_ - self.x_offset_
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            distance2 = self._pairwise_distance2(
                X[start:stop] - self.x_offset_, centered_centers
            )
            indices[start:stop] = np.argmin(distance2, axis=0)
        return indices

    def _local_coordinates(
        self,
        X: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        coordinates = np.empty((len(X), self.index_dim))
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            selected = indices[start:stop]
            coordinates[start:stop] = np.einsum(
                "bmd,bd->bm",
                self.projectors_[selected],
                (X[start:stop] - self.x_offset_)
                - (self.centers_[selected] - self.x_offset_),
                optimize=True,
            )
        return coordinates

    def _effective_config(self, X: np.ndarray) -> dict[str, float | int]:
        n, d = X.shape
        m = self.index_dim
        if n <= d + 1:
            raise ValueError("X must satisfy n > d + 1")
        if m > d:
            raise ValueError("index_dim must not exceed the number of features")
        if self.N_loc >= n:
            raise ValueError("N_loc must be smaller than n")

        N_lin = self.N_lin if self.N_lin is not None else max(2 * d, d + 2)
        N_J = (
            self.N_J if self.N_J is not None else min(n, math.ceil(2 * n / self.N_loc))
        )
        N_phi = self.N_phi if self.N_phi is not None else self.N_loc
        N_manifold = (
            self.N_manifold
            if self.N_manifold is not None
            else max(m + 1, math.ceil(N_J / 10))
        )
        a = self.a if self.a is not None else 2.0 ** (1.0 / m)
        h_min = (
            self.h_min
            if self.h_min is not None
            else 3.0 * float(np.std(X, axis=0).mean()) / math.sqrt(n)
        )

        if not d + 1 < N_lin < n:
            raise ValueError("N_lin must satisfy d + 1 < N_lin < n")
        if not m + 1 <= N_J <= n:
            raise ValueError("N_J must satisfy index_dim + 1 <= N_J <= n")
        if N_phi < m:
            raise ValueError("N_phi must be at least index_dim")
        if not m < N_manifold < N_J:
            raise ValueError("N_manifold must satisfy index_dim < N_manifold < N_J")
        if not np.isfinite(h_min) or h_min <= 0:
            raise ValueError("h_min must be finite and positive")

        return {
            "N_lin": N_lin,
            "N_J": N_J,
            "N_phi": N_phi,
            "N_manifold": N_manifold,
            "a": a,
            "h_min": h_min,
        }

    @staticmethod
    def _prepare_xy(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_array = np.asarray(X)
        Y_array = np.asarray(Y)
        for name, array in (("X", X_array), ("Y", Y_array)):
            if np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(
                array.dtype, np.number
            ):
                raise TypeError(f"{name} must contain real numeric values")
            if array.dtype.itemsize > np.dtype(np.float64).itemsize:
                raise TypeError(
                    f"{name} has a dtype wider than float64; cast it explicitly"
                )
        if X_array.ndim != 2:
            raise ValueError("X must have shape (n, d)")
        if Y_array.shape != (len(X_array),):
            raise ValueError("Y must have shape (n,)")
        X_float = np.asarray(X_array, dtype=np.float64)
        Y_float = np.asarray(Y_array, dtype=np.float64)
        if not np.all(np.isfinite(X_float)) or not np.all(np.isfinite(Y_float)):
            raise ValueError("X and Y must contain only finite values")
        return X_float, Y_float

    @staticmethod
    def _pairwise_distance2(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Вернуть ``||X_i-centers_j||^2`` без тензора ``(J,n,d)``."""
        distance2 = (
            np.einsum("bd,bd->b", centers, centers)[:, None]
            + np.einsum("nd,nd->n", X, X)[None, :]
            - 2.0 * centers @ X.T
        )
        np.maximum(distance2, 0.0, out=distance2)
        return distance2

    @staticmethod
    def _kernel(argument: np.ndarray) -> np.ndarray:
        """Epanechnikov из TeX: ``K(q)=(1-q^2)_+``."""
        np.square(argument, out=argument)
        argument *= -1.0
        argument += 1.0
        np.maximum(argument, 0.0, out=argument)
        return argument

    @staticmethod
    def _bandwidth_floor(values: np.ndarray) -> float:
        scale = float(np.max(np.ptp(values, axis=0)))
        scale = max(scale, math.sqrt(np.finfo(float).tiny))
        return math.sqrt(np.finfo(float).eps) * scale

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
        stop = min(start + self.batch_size, len(centers))
        center_block = centers[start:stop]
        distance2 = self._pairwise_distance2(X, center_block)  # (B, n)
        if projectors is None:
            distance2 /= h**2
            return self._kernel(distance2)
        assert eigenvalues is not None

        basis = projectors[start:stop]  # (B, m, d)
        coordinates = np.einsum("bmd,nd->bmn", basis, X, optimize=True)
        coordinates -= np.einsum("bmd,bd->bm", basis, center_block, optimize=True)[
            ..., None
        ]
        np.square(coordinates, out=coordinates)
        projected2 = coordinates.sum(axis=1)
        principal2 = np.einsum(
            "bm,bmn->bn", eigenvalues[start:stop], coordinates, optimize=True
        )
        residual2 = distance2 - projected2
        np.maximum(residual2, 0.0, out=residual2)
        principal2 += alpha**2 * residual2
        principal2 /= h**2
        return self._kernel(principal2)

    def _mean_mass(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        projectors: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        h: float,
        alpha: float,
    ) -> float:
        total = 0.0
        for start in range(0, len(centers), self.batch_size):
            total += float(
                self._weight_block(
                    X, centers, projectors, eigenvalues, h, alpha, start
                ).sum()
            )
        return total / len(centers)

    def _search_bandwidth(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        target: int,
        *,
        lower: float,
    ) -> float:
        """Найти минимальный bandwidth при требуемой средней локальной массе."""
        if not 0 < target < len(X):
            raise ValueError("bandwidth target must lie strictly between 0 and n")
        if self._mean_mass(X, centers, None, None, lower, 1.0) >= target:
            return float(lower)

        maximum_distance2 = 0.0
        for start in range(0, len(centers), self.batch_size):
            block = centers[start : start + self.batch_size]
            maximum_distance2 = max(
                maximum_distance2, float(self._pairwise_distance2(X, block).max())
            )
        high = max(2.0 * lower, math.sqrt(maximum_distance2), lower + 1.0e-12)
        for _ in range(100):
            if self._mean_mass(X, centers, None, None, high, 1.0) >= target:
                break
            high *= 2.0
        else:
            raise RuntimeError("could not bracket a feasible bandwidth")

        low = lower
        for _ in range(60):
            middle = 0.5 * (low + high)
            if self._mean_mass(X, centers, None, None, middle, 1.0) >= target:
                high = middle
            else:
                low = middle
        return float(high)

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
        """Найти максимальный ``alpha`` при требуемой средней массе."""
        if self._mean_mass(X, centers, projectors, eigenvalues, h, 1.0) >= target:
            return 1.0
        if self._mean_mass(X, centers, projectors, eigenvalues, h, 0.0) < target:
            raise RuntimeError(f"{kind} mass target is infeasible even with alpha=0")
        low, high = 0.0, 1.0
        tolerance = math.sqrt(np.finfo(float).eps)
        while high - low > tolerance:
            middle = 0.5 * (low + high)
            if (
                self._mean_mass(X, centers, projectors, eigenvalues, h, middle)
                >= target
            ):
                low = middle
            else:
                high = middle
        return float(low)

    def _local_gradients(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        centers: np.ndarray,
        h: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Оценить local-linear gradients без normal matrix и ridge."""
        gradients = np.empty((len(centers), X.shape[1]))
        masses = np.empty(len(centers))
        center_values = np.empty(len(centers))
        for start in range(0, len(centers), self.batch_size):
            weights = self._weight_block(X, centers, None, None, h, 1.0, start)
            for offset, weights_j in enumerate(weights):
                j = start + offset
                mass = float(weights_j.sum())
                if not np.isfinite(mass) or mass <= 0:
                    raise RuntimeError(f"empty local-linear neighborhood at center {j}")
                normalized = weights_j / mass
                mean = normalized @ X
                y_mean = float(normalized @ Y)
                root_weight = np.sqrt(weights_j)
                design = (X - mean) * root_weight[:, None]
                response = (Y - y_mean) * root_weight
                gradient, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
                if rank != X.shape[1] or not np.all(np.isfinite(gradient)):
                    raise RuntimeError(
                        f"rank-deficient local-linear fit at center {j}: "
                        f"rank={rank}, required={X.shape[1]}; increase N_lin"
                    )
                gradients[j] = gradient
                masses[j] = mass
                center_values[j] = y_mean + gradient @ (centers[j] - mean)
        return gradients, masses, center_values

    def _build_manifold_graph(
        self,
        centers: np.ndarray,
        projectors: np.ndarray | None,
        eigenvalues: np.ndarray | None,
        h: float,
        alpha: float,
    ) -> csr_matrix:
        """Построить CSR: row — target ``l``, column — source ``j``."""
        indptr = [0]
        index_chunks: list[np.ndarray] = []
        data_chunks: list[np.ndarray] = []
        count = 0
        for start in range(0, len(centers), self.batch_size):
            weights = self._weight_block(
                centers, centers, projectors, eigenvalues, h, alpha, start
            )
            for row in weights:
                indices = np.flatnonzero(row)
                if len(indices) == 0:
                    raise RuntimeError("manifold graph contains an empty row")
                values = row[indices]
                index_chunks.append(indices)
                data_chunks.append(values)
                count += len(indices)
                indptr.append(count)
        indices = np.concatenate(index_chunks).astype(np.intp, copy=False)
        data = np.concatenate(data_chunks)
        return csr_matrix(
            (data, indices, np.asarray(indptr, dtype=np.intp)),
            shape=(len(centers), len(centers)),
        )

    @staticmethod
    def _initialize_projectors(
        gradients: np.ndarray,
        gradient_mass: np.ndarray,
        graph: csr_matrix,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Инициализировать локальные bases через weighted-gradient SVD."""
        J, d = gradients.shape
        projectors = np.empty((J, m, d))
        eigenvalues = np.empty((J, m))
        for l in range(J):
            begin, end = graph.indptr[l : l + 2]
            sources = graph.indices[begin:end]
            weights = graph.data[begin:end]
            values = (
                np.sqrt(gradient_mass[sources] * weights)[:, None] * gradients[sources]
            )
            _, singular_values, right_vectors = np.linalg.svd(
                values, full_matrices=False
            )
            ADP_Manifold._check_identified(singular_values, values.shape, m, l)
            projectors[l] = ADP_Manifold._orient_rows(right_vectors[:m].copy())
            spectrum = np.square(singular_values[:m])
            eigenvalues[l] = spectrum / spectrum[0]
        return projectors, eigenvalues

    @staticmethod
    def _random_directions(
        rng: np.random.Generator,
        J: int,
        P: int,
        d: int,
    ) -> np.ndarray:
        directions = rng.standard_normal((J, P, d))
        norms = np.linalg.norm(directions, axis=2, keepdims=True)
        if np.any(norms == 0) or not np.all(np.isfinite(norms)):
            raise RuntimeError("failed to generate finite nonzero directions")
        return directions / norms

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
        """Вычислить normalized ``I/U`` при рабочей памяти ``O(B*P*n)``."""
        J, P, d = directions.shape
        I = np.empty((J, P))
        U = np.empty((J, P, d))
        mass = np.empty(J)
        n_eff = np.empty(J)
        edges = 0
        for start in range(0, J, self.batch_size):
            weights = self._weight_block(
                X, centers, projectors, eigenvalues, h, alpha, start
            )
            stop = start + len(weights)
            mass_block = weights.sum(axis=1)
            if not np.all(np.isfinite(mass_block)) or np.any(mass_block <= 0):
                raise RuntimeError("function weights contain an empty neighborhood")
            normalized = weights / mass_block[:, None]
            mean = normalized @ X
            y_mean = normalized @ Y
            phi = directions[start:stop]
            projected = phi @ X.T - (phi @ mean[..., None])
            correction = (projected @ normalized[..., None]).squeeze(-1)
            projected -= correction[..., None]
            moments = projected * normalized[:, None, :]
            summed = moments.sum(axis=2)
            I[start:stop] = moments @ Y - summed * y_mean[:, None]
            U[start:stop] = moments @ X - summed[..., None] * mean[:, None, :]
            mass[start:stop] = mass_block
            n_eff[start:stop] = 1.0 / np.square(normalized).sum(axis=1)
            edges += int(np.count_nonzero(weights))
        if not np.all(np.isfinite(I)) or not np.all(np.isfinite(U)):
            raise RuntimeError("ADP statistics contain non-finite values")
        return I, U, mass, n_eff, edges

    def _one_step(
        self,
        I: np.ndarray,
        U: np.ndarray,
        mass: np.ndarray,
        graph: csr_matrix,
        projectors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        """Выполнить один синхронный alternating step для всех targets."""
        J, m, _ = projectors.shape
        updated = np.empty_like(projectors)
        eigenvalues = np.empty((J, m))
        changes = np.empty(J)
        objectives = np.empty(J)
        penalties = np.empty(J)
        iterations_total = 0
        residual_max = 0.0

        for l in range(J):
            begin, end = graph.indptr[l : l + 2]
            sources = graph.indices[begin:end]
            weights = graph.data[begin:end]
            source_U = U[sources]
            source_I = I[sources]
            slopes = self._local_slopes(source_I, source_U, projectors[l], l)
            operator, preconditioner, rhs = self._build_B_system(
                source_U,
                source_I,
                mass[sources],
                weights,
                projectors[sources],
                slopes,
            )
            B, iterations, residual = self._solve_B(
                operator, preconditioner, rhs, projectors[l]
            )
            gamma = mass[sources] * weights
            updated[l], eigenvalues[l] = self._recover_projector(B, slopes, gamma, l)
            changes[l] = self._projector_distance(projectors[l], updated[l])
            objectives[l], penalties[l] = self._objective(
                B,
                source_U,
                source_I,
                gamma,
                slopes,
                projectors[sources],
                weights,
            )
            iterations_total += iterations
            residual_max = max(residual_max, residual)

        return (
            updated,
            eigenvalues,
            {
                "projector_change_median": float(np.median(changes)),
                "projector_change_max": float(np.max(changes)),
                "objective": float(objectives.sum()),
                "manifold_penalty": float(penalties.sum()),
                "cg_iterations": iterations_total,
                "cg_relative_residual_max": residual_max,
            },
        )

    @staticmethod
    def _local_slopes(
        I: np.ndarray,
        U: np.ndarray,
        projector: np.ndarray,
        target: int,
    ) -> np.ndarray:
        slopes = np.empty((len(I), projector.shape[0]))
        for row, (I_j, U_j) in enumerate(zip(I, U, strict=True)):
            design = U_j @ projector.T
            slope, _, rank, _ = np.linalg.lstsq(design, I_j, rcond=None)
            if rank != projector.shape[0] or not np.all(np.isfinite(slope)):
                raise RuntimeError(
                    f"rank-deficient local slope for target {target}: "
                    f"rank={rank}, required={projector.shape[0]}; "
                    "increase N_phi or N_loc"
                )
            slopes[row] = slope
        return slopes

    def _build_B_system(
        self,
        U: np.ndarray,
        I: np.ndarray,
        mass: np.ndarray,
        weights: np.ndarray,
        source_projectors: np.ndarray,
        slopes: np.ndarray,
    ) -> tuple[Any, Any, np.ndarray]:
        """Построить matrix-free normal operator для ``B`` формы ``(m,d)``.

        Оператор симметричен: действия ``rmatvec`` и ``matvec`` одинаковы.
        Penalty применяется через low-rank projectors и не материализует ``d x d``.
        """
        m = slopes.shape[1]
        d = U.shape[2]
        gamma = mass * weights
        normalized_weights = weights / weights.sum()
        weighted_slopes = gamma[:, None] * slopes
        adjoint_U = U.swapaxes(1, 2)

        def matvec(vector: np.ndarray) -> np.ndarray:
            B = vector.reshape(m, d)
            local_vectors = slopes @ B  # (K, d)
            # EXACT: batched GEMV без повторного поиска einsum-path в каждом CG.
            projected = U @ local_vectors[..., None]  # (K, P, 1)
            pulled_back = (adjoint_U @ projected).squeeze(-1)  # (K, d)
            result = weighted_slopes.T @ pulled_back
            result += self.lambda_manifold * self._penalty_action(
                B, source_projectors, normalized_weights
            )
            return result.ravel()

        data_diagonal = np.einsum(
            "j,ja,jpd->ad",
            gamma,
            np.square(slopes),
            np.square(U),
            optimize=True,
        )
        projected_diagonal = np.einsum(
            "j,jrd,jrd->d",
            normalized_weights,
            source_projectors,
            source_projectors,
            optimize=True,
        )
        penalty_diagonal = 1.0 - projected_diagonal
        np.maximum(penalty_diagonal, 0.0, out=penalty_diagonal)
        diagonal = data_diagonal + self.lambda_manifold * penalty_diagonal[None, :]
        diagonal_scale = max(float(diagonal.max()), 1.0)
        if not np.all(np.isfinite(diagonal)) or np.any(
            diagonal <= np.finfo(float).eps * diagonal_scale
        ):
            raise RuntimeError("B normal operator is rank-deficient")

        def precondition(vector: np.ndarray) -> np.ndarray:
            return vector / diagonal.ravel()

        operator_type: Any = sparse_linalg.LinearOperator
        operator = operator_type(
            (m * d, m * d), matvec=matvec, rmatvec=matvec, dtype=float
        )
        preconditioner = operator_type(
            operator.shape,
            matvec=precondition,
            rmatvec=precondition,
            dtype=float,
        )
        pulled_I = np.einsum("jpd,jp->jd", U, I, optimize=True)
        rhs = np.einsum("j,ja,jd->ad", gamma, slopes, pulled_I, optimize=True).ravel()
        return operator, preconditioner, rhs

    @staticmethod
    def _penalty_action(
        B: np.ndarray,
        source_projectors: np.ndarray,
        normalized_weights: np.ndarray,
    ) -> np.ndarray:
        coordinates = B @ source_projectors.swapaxes(1, 2)  # (K, m, m)
        coordinates *= normalized_weights[:, None, None]
        # Только (m,K*m), без промежуточных (K,m,d) или (d,d).
        projected = coordinates.transpose(1, 0, 2).reshape(len(B), -1) @ (
            source_projectors.reshape(-1, B.shape[1])
        )
        return B - projected

    def _solve_B(
        self,
        operator: Any,
        preconditioner: Any,
        rhs: np.ndarray,
        initial: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        iterations = 0

        def record(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        cg_method: Any = sparse_linalg.cg
        solution, info = cg_method(
            operator,
            rhs,
            x0=initial.ravel(),
            M=preconditioner,
            rtol=self.cg_tol,
            atol=0.0,
            maxiter=self.cg_maxiter
            or max(50, min(1000, 5 * int(np.prod(initial.shape)))),
            callback=record,
        )
        solution = np.asarray(solution, dtype=float)
        residual = operator @ solution - rhs
        relative_residual = float(np.linalg.norm(residual)) / max(
            float(np.linalg.norm(rhs)), np.finfo(float).tiny
        )
        if info != 0:
            raise RuntimeError(
                "CG did not converge for B: "
                f"info={int(info)}, iterations={iterations}, "
                f"relative_residual={relative_residual:.3e}"
            )
        limit = max(10.0 * self.cg_tol, 256.0 * np.finfo(float).eps)
        if not np.all(np.isfinite(solution)) or not np.isfinite(relative_residual):
            raise RuntimeError("CG returned a non-finite B solution")
        if relative_residual > limit:
            raise RuntimeError(
                "CG residual certificate failed for B: "
                f"{relative_residual:.3e} > {limit:.3e}"
            )
        return solution.reshape(initial.shape), iterations, relative_residual

    @staticmethod
    def _recover_projector(
        B: np.ndarray,
        slopes: np.ndarray,
        gamma: np.ndarray,
        target: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Извлечь ``P/Lambda`` из rank-``m`` фактора без ``d x d``."""
        M = np.einsum("j,ja,jb->ab", gamma, slopes, slopes, optimize=True)
        values, vectors = np.linalg.eigh(M)
        tolerance = (
            256.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(M, ord=2)))
        )
        if not np.all(np.isfinite(values)) or values[0] < -tolerance:
            raise RuntimeError(f"invalid local slope matrix at target {target}")
        np.maximum(values, 0.0, out=values)
        root = (vectors * np.sqrt(values)[None, :]) @ vectors.T
        factor = root @ B
        _, singular_values, right_vectors = np.linalg.svd(factor, full_matrices=False)
        ADP_Manifold._check_identified(
            singular_values, factor.shape, B.shape[0], target
        )
        projector = ADP_Manifold._orient_rows(right_vectors.copy())
        spectrum = np.square(singular_values)
        spectrum /= spectrum[0]
        orthogonality = np.linalg.norm(
            projector @ projector.T - np.eye(B.shape[0]), ord="fro"
        )
        if not np.isfinite(orthogonality) or orthogonality > 1.0e-10:
            raise RuntimeError(
                f"non-orthonormal projector at target {target}: {orthogonality:.3e}"
            )
        if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0):
            raise RuntimeError(f"invalid local spectrum at target {target}")
        return projector, spectrum

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
        predicted = np.einsum("jpd,jd->jp", U, slopes @ B, optimize=True)
        data_term = float(
            np.einsum("j,jp,jp->", gamma, I - predicted, I - predicted, optimize=True)
        )
        penalty_action = self._penalty_action(
            B, source_projectors, weights / weights.sum()
        )
        penalty = self.lambda_manifold * float(np.vdot(B, penalty_action).real)
        tolerance = 256.0 * np.finfo(float).eps * max(1.0, float(np.vdot(B, B)))
        if not np.isfinite(data_term) or not np.isfinite(penalty):
            raise RuntimeError("manifold objective became non-finite")
        if penalty < -tolerance:
            raise RuntimeError("manifold penalty became negative")
        penalty = max(0.0, penalty)
        return data_term + penalty, penalty

    @staticmethod
    def _projector_distance(left: np.ndarray, right: np.ndarray) -> float:
        overlap = right @ left.T
        distance2 = 2.0 * left.shape[0] - 2.0 * float(np.vdot(overlap, overlap).real)
        return math.sqrt(max(0.0, distance2))

    @staticmethod
    def _check_identified(
        singular_values: np.ndarray,
        shape: tuple[int, int],
        m: int,
        target: int,
    ) -> None:
        if len(singular_values) < m or not np.all(np.isfinite(singular_values)):
            raise RuntimeError(f"local EDR rank is smaller than {m} at target {target}")
        threshold = (
            np.finfo(float).eps
            * max(shape)
            * (float(singular_values[0]) if len(singular_values) else 0.0)
        )
        if singular_values[m - 1] <= threshold:
            raise RuntimeError(f"local EDR rank is smaller than {m} at target {target}")

    @staticmethod
    def _orient_rows(basis: np.ndarray) -> np.ndarray:
        columns = np.argmax(np.abs(basis), axis=1)
        signs = np.sign(basis[np.arange(len(basis)), columns])
        basis *= np.where(signs == 0, 1.0, signs)[:, None]
        return basis

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
        manifold_mass = np.asarray(graph.sum(axis=1)).ravel()
        entry: dict[str, float | int | str] = {
            "phase": phase,
            "iteration": iteration,
            "h": float(h),
            "h_manifold": float(h_manifold),
            "alpha": float(alpha),
            "alpha_manifold": float(alpha_manifold),
            "function_edges": function_edges,
            "manifold_edges": int(graph.nnz),
            "function_mass_q10": float(np.quantile(mass, 0.1)),
            "function_mass_min": float(np.min(mass)),
            "function_mass_mean": float(np.mean(mass)),
            "function_mass_median": float(np.median(mass)),
            "function_mass_q90": float(np.quantile(mass, 0.9)),
            "n_eff_q10": float(np.quantile(n_eff, 0.1)),
            "n_eff_min": float(np.min(n_eff)),
            "n_eff_median": float(np.median(n_eff)),
            "n_eff_q90": float(np.quantile(n_eff, 0.9)),
            "manifold_mass_q10": float(np.quantile(manifold_mass, 0.1)),
            "manifold_mass_median": float(np.median(manifold_mass)),
            "manifold_mass_q90": float(np.quantile(manifold_mass, 0.9)),
        }
        entry.update(diagnostics)
        return entry

    @staticmethod
    def _integer(name: str, value: int, *, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return int(value)

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
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a real number")
        result = float(value)
        invalid = result <= minimum if strict else result < minimum
        if not np.isfinite(result) or invalid:
            relation = "greater than" if strict else "at least"
            raise ValueError(f"{name} must be finite and {relation} {minimum}")
        return result

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
