"""Компактная поддержка и sparse neighborhood-представление весов."""

# ruff: noqa: RUF002

from dataclasses import dataclass
from functools import partial
from typing import cast

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from . import utils

_BOX_DISTANCE_CACHE_MAX_BYTES = 64 * 1024**2
_BOX_ROW_COVERAGE = 0.9


def _validated_tau(tau: float) -> float:
    """Проверить параметр плато-ядра ``tau`` и вернуть ``float``."""
    tau = float(tau)
    utils.require(
        np.isfinite(tau) and 0.0 < tau < 1.0,
        "tau must be finite and lie in (0, 1)",
    )
    return tau


def _kernel_input(value) -> np.ndarray:
    """Привести аргумент ядра к конечному массиву ``float``."""
    value = np.asarray(value, dtype=float)
    utils.require(np.all(np.isfinite(value)), "kernel input must be finite")
    return value


def box_kernel(value) -> np.ndarray:
    """Вернуть индикатор box-поддержки ``1[value < 1]``."""
    return np.asarray(_kernel_input(value) < 1.0, dtype=float)


def plateau_kernel(value, *, tau: float = 0.5) -> np.ndarray:
    """Вычислить гладкое компактное ядро с единым плато до ``tau``."""
    value = _kernel_input(value)
    tau = _validated_tau(tau)
    result = np.zeros_like(value)
    result[value <= tau] = 1.0
    boundary = (value > tau) & (value < 1.0)
    s = (value[boundary] - tau) / (1.0 - tau)
    result[boundary] = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5
    return result


def make_plateau_kernel(tau: float = 0.5):
    """Создать callable плато-ядра с зафиксированным ``tau``."""
    return partial(plateau_kernel, tau=_validated_tau(tau))


def sparse_kernel_parameters(kernel) -> tuple[str, float | None] | None:
    """Распознать sparse-ядро и вернуть его режим и параметр плато."""
    if kernel is box_kernel:
        return "box", None
    if kernel is plateau_kernel:
        return "plateau", 0.5
    if isinstance(kernel, partial) and kernel.func is plateau_kernel:
        utils.require(
            not (kernel.args or set(kernel.keywords or {}) - {"tau"}),
            "plateau kernel accepts only keyword tau",
        )
        return "plateau", _validated_tau((kernel.keywords or {}).get("tau", 0.5))
    return None


def _readonly_array(value, dtype, name, *, ndim=1):
    """Скопировать и защитить структурный массив sparse-блока от записи."""
    value = np.asarray(value, dtype=dtype)
    utils.require(value.ndim == ndim, f"{name} must be {ndim}-dimensional")
    value = value.copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True, slots=True)
class SparseNeighborhoodBlock:
    start: int
    n_observations: int
    indptr: np.ndarray
    indices: np.ndarray
    boundary_positions: np.ndarray
    boundary_weights: np.ndarray
    mode: str
    tau: float | None

    def __post_init__(self):
        """Проверить CSR-подобный sparse-блок и сделать его массивы read-only."""
        utils.require(
            not isinstance(self.start, bool)
            and isinstance(self.start, (int, np.integer))
            and self.start >= 0,
            "start must be a nonnegative integer",
        )
        utils.require(
            not isinstance(self.n_observations, bool)
            and isinstance(self.n_observations, (int, np.integer))
            and self.n_observations >= 1,
            "n_observations must be a positive integer",
        )

        indptr = _readonly_array(self.indptr, np.intp, "indptr")
        indices = _readonly_array(self.indices, np.intp, "indices")
        positions = _readonly_array(
            self.boundary_positions,
            np.intp,
            "boundary_positions",
        )
        weights = _readonly_array(
            self.boundary_weights,
            float,
            "boundary_weights",
        )
        utils.require(
            len(indptr) >= 2 and indptr[0] == 0 and indptr[-1] == len(indices),
            "indptr must delimit at least one complete row",
        )
        utils.require(
            not np.any(np.diff(indptr) < 0),
            "indptr must be nondecreasing",
        )
        utils.require(
            not (np.any(indices < 0) or np.any(indices >= self.n_observations)),
            "indices are out of range",
        )
        utils.require(
            len(positions) == len(weights),
            "boundary positions and weights must have equal length",
        )
        utils.require(
            not (
                len(positions)
                and (
                    np.any(positions < 0)
                    or np.any(positions >= len(indices))
                    or np.any(np.diff(positions) <= 0)
                )
            ),
            "boundary positions must be sorted unique edge positions",
        )
        utils.require(
            np.all(np.isfinite(weights))
            and not np.any(weights <= 0)
            and not np.any(weights >= 1),
            "boundary weights must be finite and lie in (0, 1)",
        )
        utils.require(
            self.mode in {"box", "plateau"}, "mode must be 'box' or 'plateau'"
        )
        if self.mode == "box":
            utils.require(
                not (len(positions) or self.tau is not None),
                "box blocks cannot contain boundary data",
            )
        else:
            object.__setattr__(self, "tau", _validated_tau(cast(float, self.tau)))

        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "boundary_positions", positions)
        object.__setattr__(self, "boundary_weights", weights)
        utils.require(
            not np.any(self.mass <= 0),
            "every sparse neighborhood row must have positive mass",
        )

    @property
    def rows(self) -> int:
        """Вернуть число локальных строк в блоке."""
        return len(self.indptr) - 1

    @property
    def edge_count(self) -> int:
        """Вернуть число ненулевых neighborhood-рёбер."""
        return len(self.indices)

    @property
    def boundary_count(self) -> int:
        """Вернуть число рёбер с весом plateau-boundary меньше единицы."""
        return len(self.boundary_weights)

    @property
    def mass(self) -> np.ndarray:
        """Вычислить сумму весов каждой строки из CSR и boundary-поправок."""
        result = np.diff(self.indptr).astype(float)
        if self.boundary_count:
            rows = np.searchsorted(
                self.indptr[1:],
                self.boundary_positions,
                side="right",
            )
            np.add.at(result, rows, self.boundary_weights - 1.0)
        return result

    def row(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """Материализовать индексы и веса одной локальной строки."""
        utils.require(
            not isinstance(row, bool) and isinstance(row, (int, np.integer)),
            "row must be an integer",
            TypeError,
        )
        utils.require(0 <= row < self.rows, row, IndexError)
        first, last = self.indptr[row : row + 2]
        indices = self.indices[first:last]
        weights = np.ones(last - first)
        left = np.searchsorted(self.boundary_positions, first, side="left")
        right = np.searchsorted(self.boundary_positions, last, side="left")
        if right > left:
            weights[self.boundary_positions[left:right] - first] = (
                self.boundary_weights[left:right]
            )
        return indices, weights

    def padded(self) -> tuple[np.ndarray, np.ndarray]:
        """Преобразовать variable-length строки в компактное padded-представление."""
        counts = np.diff(self.indptr)
        width = int(counts.max())
        rows = np.repeat(np.arange(self.rows, dtype=np.intp), counts)
        slots = np.arange(self.edge_count, dtype=np.intp) - np.repeat(
            self.indptr[:-1], counts
        )
        indices = np.zeros((self.rows, width), dtype=np.intp)
        weights = np.zeros((self.rows, width))
        indices[rows, slots] = self.indices
        weights[rows, slots] = 1.0
        if self.boundary_count:
            positions = self.boundary_positions
            weights[rows[positions], slots[positions]] = self.boundary_weights
        return indices, weights

    def cache_key(self, row: int):
        """Построить неизменяемый ключ поддержки для повторного использования."""
        first, last = self.indptr[row : row + 2]
        indices = self.indices[first:last]
        if self.mode == "box":
            return self.mode, indices.tobytes()
        left = np.searchsorted(self.boundary_positions, first, side="left")
        right = np.searchsorted(self.boundary_positions, last, side="left")
        relative = self.boundary_positions[left:right] - first
        return (
            self.mode,
            indices.tobytes(),
            relative.tobytes(),
            self.boundary_weights[left:right].tobytes(),
        )


class SparseStatisticsCache:
    def __init__(self):
        """Создать кэш локальных статистик между вызовами одного fit."""
        self._entries = set()
        self._active = set()
        self._call_hits = 0
        self.last_hits = 0
        self.total_hits = 0

    def begin_call(self):
        """Начать сбор ключей поддержки для нового вызова статистик."""
        self._active = set()
        self._call_hits = 0

    def finish_call(self):
        """Зафиксировать активные ключи и число cache hits текущего вызова."""
        self._entries = self._active
        self.last_hits = self._call_hits
        self.total_hits += self._call_hits

    def observe(self, block: SparseNeighborhoodBlock):
        """Обновить reuse-метрики по строкам sparse neighborhood-блока."""
        for row in range(block.rows):
            key = block.cache_key(row)
            if key in self._entries:
                self._call_hits += 1
            else:
                self._entries.add(key)
            self._active.add(key)


class NeighborhoodEngine:
    def __init__(self, X, centers, block_size=128):
        """Подготовить KD-tree и bounded-memory sparse neighborhood engine."""
        X = np.asarray(X, dtype=float)
        centers = np.asarray(centers, dtype=float)
        utils.require(
            X.ndim == 2 and 0 not in X.shape and np.all(np.isfinite(X)),
            "X must have finite non-empty shape (n, d)",
        )
        utils.require(
            centers.ndim == 2
            and centers.shape[1] == X.shape[1]
            and len(centers)
            and np.all(np.isfinite(centers)),
            "centers must have finite non-empty shape (J, d)",
        )
        utils.require(
            not isinstance(block_size, bool)
            and isinstance(block_size, (int, np.integer))
            and block_size >= 1,
            "block_size must be a positive integer",
        )
        self.X = X
        self.centers = centers
        self.block_size = int(block_size)
        self.tree = cKDTree(X)
        self._box_distance2_cache = None
        self._multi_box_support_cache = None
        self._previous_keys = None
        self.last_support_edges = 0
        self.last_boundary_edges = 0
        self.last_support_reuse_hits = 0

    def _box_distance2(self):
        """Получить кэшируемые квадраты евклидовых расстояний или ``None``."""
        required_bytes = len(self.X) * len(self.centers) * np.dtype(float).itemsize
        if required_bytes > _BOX_DISTANCE_CACHE_MAX_BYTES:
            return None
        if self._box_distance2_cache is None:
            distance2 = (
                np.square(self.centers).sum(axis=1)[:, None]
                + np.square(self.X).sum(axis=1)[None, :]
                - 2.0 * self.centers @ self.X.T
            )
            np.maximum(distance2, 0.0, out=distance2)
            if not np.all(np.isfinite(distance2)):
                return None
            distance2.setflags(write=False)
            self._box_distance2_cache = distance2
        return self._box_distance2_cache

    @staticmethod
    def _positive(value, name):
        """Проверить положительный конечный числовой параметр."""
        value = float(value)
        utils.require(
            np.isfinite(value) and value > 0, f"{name} must be finite and positive"
        )
        return value

    @staticmethod
    def _scale(value, name):
        """Проверить scale ядра в замкнутом интервале ``[0, 1]``."""
        value = float(value)
        utils.require(
            np.isfinite(value) and 0 <= value <= 1, f"{name} must lie in [0, 1]"
        )
        return value

    def _beta(self, beta):
        """Проверить zero/unit single-index и вернуть массив ``(d,)``."""
        beta = np.asarray(beta, dtype=float)
        utils.require(
            beta.shape == (self.X.shape[1],) and np.all(np.isfinite(beta)),
            "beta must have finite shape (d,)",
        )
        norm = np.linalg.norm(beta)
        utils.require(
            norm == 0 or np.isclose(norm, 1.0, rtol=1e-8, atol=1e-10),
            "beta must be zero or unit length",
        )
        return beta

    def _multi_basis(self, basis, eigenvalues):
        """Проверить ортонормированный basis и неотрицательный spectrum."""
        basis = np.asarray(basis, dtype=float)
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        utils.require(
            basis.ndim == 2
            and basis.shape[0] == self.X.shape[1]
            and bool(basis.shape[1])
            and np.all(np.isfinite(basis)),
            "basis must have finite shape (d, m)",
        )
        utils.require(
            eigenvalues.shape == (basis.shape[1],)
            and np.all(np.isfinite(eigenvalues))
            and not np.any(eigenvalues < 0),
            "eigenvalues must have finite nonnegative shape (m,)",
        )
        utils.require(
            np.allclose(
                basis.T @ basis,
                np.eye(basis.shape[1]),
                rtol=1e-8,
                atol=1e-10,
            ),
            "basis columns must be orthonormal",
        )
        return basis, eigenvalues

    def _box_requirements(self, target):
        """Перевести целевую массу в integer threshold и coverage count."""
        target = float(target)
        utils.require(
            np.isfinite(target) and target > 0,
            "target mass must be finite and positive",
        )
        return (
            int(np.ceil(target)),
            int(np.ceil(_BOX_ROW_COVERAGE * len(self.centers))),
        )

    def _single_box_row_counts(self, beta, h, rho, distance2):
        """Посчитать box-neighborhood mass для каждого single-index центра."""
        x_projection = self.X @ beta
        center_projection = self.centers @ beta
        h2 = h**2
        rho2 = rho**2
        return np.fromiter(
            (
                np.count_nonzero(
                    rho2 * distance2[row]
                    + np.square(x_projection - center_projection[row])
                    < h2
                )
                for row in range(len(self.centers))
            ),
            dtype=np.intp,
            count=len(self.centers),
        )

    def _single_box_supports(self, beta, h, rho, distance2):
        """Построить списки индексов single-index box-поддержки."""
        x_projection = self.X @ beta
        center_projection = self.centers @ beta
        h2 = h**2
        rho2 = rho**2
        return [
            np.flatnonzero(
                rho2 * distance2[row] + np.square(x_projection - center_projection[row])
                < h2
            )
            for row in range(len(self.centers))
        ]

    def _multi_box_components(self, basis, eigenvalues, distance2):
        """Разложить расстояние multi-index на orthogonal и principal части."""
        projected_X = self.X @ basis
        projected_centers = self.centers @ basis
        projected2 = np.zeros_like(distance2)
        principal2 = np.zeros_like(distance2)
        for column, eigenvalue in enumerate(eigenvalues):
            delta2 = projected_centers[:, column, None] - projected_X[None, :, column]
            np.square(delta2, out=delta2)
            projected2 += delta2
            delta2 *= eigenvalue
            principal2 += delta2
        np.subtract(distance2, projected2, out=projected2)
        np.maximum(projected2, 0.0, out=projected2)
        return projected2, principal2

    @staticmethod
    def _multi_box_key(basis, eigenvalues, h, alpha):
        """Собрать ключ кэша multi-index box-поддержки."""
        return basis.tobytes(), eigenvalues.tobytes(), float(h), float(alpha)

    def _multi_box_supports(self, orthogonal2, principal2, h, alpha):
        """Построить multi-index box-поддержки из двух компонент расстояния."""
        h2 = h**2
        alpha2 = alpha**2
        return [
            np.flatnonzero(alpha2 * orthogonal2[row] + principal2[row] < h2)
            for row in range(len(self.centers))
        ]

    def select_single_box_scale(self, beta, h, target):
        """Найти наибольший допустимый single-index box scale по mass-условию."""
        beta = self._beta(beta)
        h = self._positive(h, "h")
        required, covered = self._box_requirements(target)
        distance2 = self._box_distance2()
        if distance2 is None:
            return NotImplemented

        def enough(counts):
            """Проверить покрытие single-index центров по mass counts."""
            return np.count_nonzero(counts >= required) >= covered

        x_projection = self.X @ beta
        center_projection = self.centers @ beta
        projection2 = np.square(center_projection[:, None] - x_projection[None, :])
        h2 = h**2
        mass_one = np.fromiter(
            (
                np.count_nonzero(distance2[row] + projection2[row] < h2)
                for row in range(len(self.centers))
            ),
            dtype=np.intp,
            count=len(self.centers),
        )
        if enough(mass_one):
            return 1.0
        if not enough(np.count_nonzero(projection2 < h2, axis=1)):
            return None

        np.subtract(h2, projection2, out=projection2)
        for row in range(len(self.centers)):
            positive = distance2[row] > 0
            numerator = projection2[row]
            np.divide(
                numerator,
                distance2[row],
                out=numerator,
                where=positive,
            )
            zero = ~positive
            numerator[zero] = np.where(
                numerator[zero] > 0,
                np.inf,
                -np.inf,
            )

        kth = len(self.X) - required
        projection2.partition(kth, axis=1)
        breakpoints = projection2[:, kth].copy()
        kth = len(breakpoints) - covered
        breakpoints.partition(kth)
        breakpoint = float(breakpoints[kth])
        rho = min(
            float(np.nextafter(np.sqrt(breakpoint), 0.0)),
            1.0,
        )
        while not enough(self._single_box_row_counts(beta, h, rho, distance2)):
            next_rho = float(np.nextafter(rho, 0.0))
            utils.require(
                next_rho != rho,
                "could not represent a feasible box scale",
                RuntimeError,
            )
            rho = next_rho
        return rho

    def select_multi_box_scale(self, basis, eigenvalues, h, target):
        """Найти multi-index scale, покрывающий требуемую долю центров."""
        basis, eigenvalues = self._multi_basis(basis, eigenvalues)
        h = self._positive(h, "h")
        required, covered = self._box_requirements(target)
        distance2 = self._box_distance2()
        if distance2 is None:
            return NotImplemented

        def enough(counts):
            """Проверить покрытие multi-index центров по mass counts."""
            return np.count_nonzero(counts >= required) >= covered

        orthogonal2, principal2 = self._multi_box_components(
            basis,
            eigenvalues,
            distance2,
        )
        h2 = h**2
        mass_one = np.count_nonzero(orthogonal2 + principal2 < h2, axis=1)
        if enough(mass_one):
            alpha = 1.0
            supports = self._multi_box_supports(orthogonal2, principal2, h, alpha)
            self._multi_box_support_cache = (
                self._multi_box_key(basis, eigenvalues, h, alpha),
                supports,
            )
            return alpha
        if not enough(np.count_nonzero(principal2 < h2, axis=1)):
            return None

        np.subtract(h2, principal2, out=principal2)
        for row in range(len(self.centers)):
            positive = orthogonal2[row] > 0
            numerator = principal2[row]
            np.divide(
                numerator,
                orthogonal2[row],
                out=numerator,
                where=positive,
            )
            zero = ~positive
            numerator[zero] = np.where(
                numerator[zero] > 0,
                np.inf,
                -np.inf,
            )

        kth = len(self.X) - required
        breakpoints = np.empty(len(self.centers))
        for row in range(len(self.centers)):
            selection = principal2[row].copy()
            selection.partition(kth)
            breakpoints[row] = selection[kth]
        kth = len(breakpoints) - covered
        breakpoints.partition(kth)
        breakpoint = float(breakpoints[kth])
        alpha = min(
            float(np.nextafter(np.sqrt(breakpoint), 0.0)),
            1.0,
        )
        while not enough(np.count_nonzero(principal2 > alpha**2, axis=1)):
            next_alpha = float(np.nextafter(alpha, 0.0))
            utils.require(
                next_alpha != alpha,
                "could not represent a feasible box scale",
                RuntimeError,
            )
            alpha = next_alpha

        supports = [np.flatnonzero(row > alpha**2) for row in principal2]
        self._multi_box_support_cache = (
            self._multi_box_key(basis, eigenvalues, h, alpha),
            supports,
        )
        return alpha

    @staticmethod
    def _candidate_array(value):
        """Привести список кандидатов KD-tree к integer index array."""
        return np.asarray(value, dtype=np.intp)

    def _euclidean_candidates(self, radius):
        """Получить отсортированные KD-tree кандидаты в евклидовом радиусе."""
        return [
            self._candidate_array(row)
            for row in self.tree.query_ball_point(
                self.centers,
                radius,
                return_sorted=True,
            )
        ]

    @staticmethod
    def _intersect(left, right):
        """Пересечь projected и евклидов список кандидатов без дубликатов."""
        if right is None:
            return left
        return np.intersect1d(left, right, assume_unique=True)

    def _encode(self, supports, q_values, kernel, record):
        """Закодировать variable-length supports в bounded sparse blocks."""
        parameters = sparse_kernel_parameters(kernel)
        utils.require(
            parameters is not None,
            "sparse neighborhoods require box or plateau kernel",
        )
        parameters = utils.require_not_none(
            parameters, "sparse kernel parameters missing"
        )
        mode, tau = parameters
        blocks = []
        keys = []
        tiny = np.nextafter(0.0, 1.0)
        almost_one = np.nextafter(1.0, 0.0)
        for start in range(0, len(self.centers), self.block_size):
            stop = min(start + self.block_size, len(self.centers))
            rows = supports[start:stop]
            values = q_values[start:stop]
            counts = np.fromiter((len(row) for row in rows), dtype=np.intp)
            indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.intp)))
            indices = (
                np.concatenate(rows).astype(np.intp, copy=False)
                if indptr[-1]
                else np.empty(0, dtype=np.intp)
            )
            if mode == "box":
                positions = np.empty(0, dtype=np.intp)
                weights = np.empty(0)
            else:
                tau = cast(float, tau)
                position_rows = []
                weight_rows = []
                for offset, q in zip(indptr[:-1], values, strict=True):
                    boundary = np.flatnonzero(q > tau)
                    if len(boundary):
                        position_rows.append(offset + boundary)
                        weight_rows.append(plateau_kernel(q[boundary], tau=tau))
                positions = (
                    np.concatenate(position_rows).astype(np.intp, copy=False)
                    if position_rows
                    else np.empty(0, dtype=np.intp)
                )
                weights = (
                    np.clip(np.concatenate(weight_rows), tiny, almost_one)
                    if weight_rows
                    else np.empty(0)
                )
            block = SparseNeighborhoodBlock(
                start=start,
                n_observations=len(self.X),
                indptr=indptr,
                indices=indices,
                boundary_positions=positions,
                boundary_weights=weights,
                mode=mode,
                tau=tau,
            )
            blocks.append(block)
            if record:
                keys.extend(block.cache_key(row) for row in range(block.rows))

        if record:
            previous = self._previous_keys
            self.last_support_reuse_hits = (
                0
                if previous is None
                else sum(
                    left == right for left, right in zip(previous, keys, strict=True)
                )
            )
            self._previous_keys = keys
            self.last_support_edges = sum(block.edge_count for block in blocks)
            self.last_boundary_edges = sum(block.boundary_count for block in blocks)
        return iter(blocks)

    def isotropic_blocks(self, h, kernel, *, record=False):
        """Сформировать sparse-блоки изотропных neighborhood-весов."""
        h = self._positive(h, "h")
        parameters = sparse_kernel_parameters(kernel)
        distance2 = self._box_distance2() if parameters == ("box", None) else None
        if distance2 is not None:
            supports = [np.flatnonzero(row < h**2) for row in distance2]
            empty = np.empty(0)
            return self._encode(
                supports,
                [empty] * len(supports),
                kernel,
                record,
            )
        candidates = self._euclidean_candidates(h)
        supports = []
        q_values = []
        for center, indices in zip(self.centers, candidates, strict=True):
            difference = self.X[indices] - center
            q = np.einsum("nd,nd->n", difference, difference) / h**2
            active = q < 1.0
            supports.append(indices[active])
            q_values.append(q[active])
        return self._encode(supports, q_values, kernel, record)

    def single_blocks(self, beta, h, rho, kernel, *, record=True):
        """Сформировать sparse-блоки анизотропных single-index весов."""
        beta = self._beta(beta)
        h = self._positive(h, "h")
        rho = self._scale(rho, "rho")
        parameters = sparse_kernel_parameters(kernel)
        distance2 = self._box_distance2() if parameters == ("box", None) else None
        if distance2 is not None:
            supports = self._single_box_supports(beta, h, rho, distance2)
            empty = np.empty(0)
            return self._encode(
                supports,
                [empty] * len(supports),
                kernel,
                record,
            )

        x_projection = self.X @ beta
        center_projection = self.centers @ beta
        order = np.argsort(x_projection)
        sorted_projection = x_projection[order]
        radius = h / np.sqrt(1.0 + rho**2)
        projected = []
        for value in center_projection:
            left = np.searchsorted(sorted_projection, value - radius, side="left")
            right = np.searchsorted(sorted_projection, value + radius, side="right")
            projected.append(np.sort(order[left:right]))
        euclidean = self._euclidean_candidates(h / rho) if rho > 0 else None

        supports = []
        q_values = []
        for row, center in enumerate(self.centers):
            indices = self._intersect(
                projected[row],
                None if euclidean is None else euclidean[row],
            )
            difference = self.X[indices] - center
            distance2 = np.einsum("nd,nd->n", difference, difference)
            projection2 = np.square(difference @ beta)
            q = (rho**2 * distance2 + projection2) / h**2
            active = q < 1.0
            supports.append(indices[active])
            q_values.append(q[active])
        return self._encode(supports, q_values, kernel, record)

    def multi_blocks(
        self,
        basis,
        eigenvalues,
        h,
        alpha,
        kernel,
        *,
        record=True,
    ):
        """Сформировать sparse-блоки multi-index весов и переиспользовать support."""
        basis, eigenvalues = self._multi_basis(basis, eigenvalues)
        h = self._positive(h, "h")
        alpha = self._scale(alpha, "alpha")

        parameters = sparse_kernel_parameters(kernel)
        distance2 = self._box_distance2() if parameters == ("box", None) else None
        if distance2 is not None:
            key = self._multi_box_key(basis, eigenvalues, h, alpha)
            cached = self._multi_box_support_cache
            if cached is not None and cached[0] == key:
                supports = cached[1]
            else:
                orthogonal2, principal2 = self._multi_box_components(
                    basis,
                    eigenvalues,
                    distance2,
                )
                supports = self._multi_box_supports(
                    orthogonal2,
                    principal2,
                    h,
                    alpha,
                )
                self._multi_box_support_cache = key, supports
            empty = np.empty(0)
            return self._encode(
                supports,
                [empty] * len(supports),
                kernel,
                record,
            )

        root_eigenvalues = np.sqrt(eigenvalues)
        projected_X = (self.X @ basis) * root_eigenvalues
        projected_centers = (self.centers @ basis) * root_eigenvalues
        projected_tree = cKDTree(projected_X)
        projected = [
            self._candidate_array(row)
            for row in projected_tree.query_ball_point(
                projected_centers,
                h,
                return_sorted=True,
            )
        ]
        minimum_coefficient = min(alpha**2, float(eigenvalues.min()))
        euclidean = (
            self._euclidean_candidates(h / np.sqrt(minimum_coefficient))
            if minimum_coefficient > 0
            else None
        )

        supports = []
        q_values = []
        for row, center in enumerate(self.centers):
            indices = self._intersect(
                projected[row],
                None if euclidean is None else euclidean[row],
            )
            difference = self.X[indices] - center
            distance2 = np.einsum("nd,nd->n", difference, difference)
            coordinates = difference @ basis
            projected2 = np.einsum("nm,nm->n", coordinates, coordinates)
            principal2 = np.einsum(
                "nm,m,nm->n",
                coordinates,
                eigenvalues,
                coordinates,
            )
            orthogonal2 = np.maximum(distance2 - projected2, 0.0)
            q = (alpha**2 * orthogonal2 + principal2) / h**2
            active = q < 1.0
            supports.append(indices[active])
            q_values.append(q[active])
        return self._encode(supports, q_values, kernel, record)


def _mean_sparse_mass(blocks) -> float:
    """Вычислить среднюю массу строк sparse neighborhood-представления."""
    total = 0.0
    rows = 0
    for block in blocks:
        mass = block.mass
        total += float(mass.sum())
        rows += len(mass)
    utils.require(rows > 0, "sparse blocks must contain at least one row")
    return total / rows


def _box_sparse_mass_enough(blocks, target) -> bool:
    """Проверить box-критерий: нужная масса есть у требуемой доли строк."""
    required = int(np.ceil(target))
    qualified = 0
    rows = 0
    for block in blocks:
        mass = block.mass
        qualified += int(np.count_nonzero(mass >= required))
        rows += len(mass)
    utils.require(rows > 0, "sparse blocks must contain at least one row")
    return qualified >= int(np.ceil(_BOX_ROW_COVERAGE * rows))


def search_sparse_bandwidth(
    engine: NeighborhoodEngine,
    target: float,
    kernel,
    *,
    lower: float,
) -> float:
    """Двоичным поиском найти минимальный sparse bandwidth с нужной массой."""
    target = float(target)
    utils.require(
        target > 0 and np.isfinite(target), "target must be finite and positive"
    )
    lower = engine._positive(lower, "lower")
    parameters = sparse_kernel_parameters(kernel)
    utils.require(
        parameters is not None,
        "sparse bandwidth requires box or plateau kernel",
    )
    parameters = utils.require_not_none(parameters, "sparse kernel parameters missing")
    if parameters == ("box", None):
        distance2 = engine._box_distance2()
        if distance2 is not None:
            required, covered = engine._box_requirements(target)
            utils.require(
                required <= distance2.shape[1],
                "could not bracket a feasible bandwidth",
                RuntimeError,
            )
            if (
                np.count_nonzero(
                    np.count_nonzero(distance2 < lower**2, axis=1) >= required
                )
                >= covered
            ):
                return lower
            selection = distance2.copy()
            selection.partition(required - 1, axis=1)
            thresholds = selection[:, required - 1].copy()
            thresholds.partition(covered - 1)
            threshold = float(thresholds[covered - 1])
            h = float(np.nextafter(np.sqrt(threshold), np.inf))
            while (
                np.count_nonzero(np.count_nonzero(distance2 < h**2, axis=1) >= required)
                < covered
            ):
                h = float(np.nextafter(h, np.inf))
            return h

    def enough(h):
        """Проверить, достаточна ли sparse neighborhood-масса при ``h``."""
        blocks = engine.isotropic_blocks(h, kernel, record=False)
        if parameters == ("box", None):
            return _box_sparse_mass_enough(blocks, target)
        return _mean_sparse_mass(blocks) >= target

    low = lower
    if enough(low):
        return low
    minimum = np.minimum(engine.X.min(axis=0), engine.centers.min(axis=0))
    maximum = np.maximum(engine.X.max(axis=0), engine.centers.max(axis=0))
    diagonal = float(np.linalg.norm(maximum - minimum))
    high = max(2.0 * low, diagonal, 1.0)
    for _ in range(100):
        if enough(high):
            break
        high *= 2.0
    else:
        utils.require(False, "could not bracket a feasible bandwidth", RuntimeError)

    for _ in range(60):
        middle = (low + high) / 2.0
        if enough(middle):
            high = middle
        else:
            low = middle
    return float(high)


def search_sparse_scale(
    build_blocks,
    target_mass: float,
    *,
    rowwise: bool = False,
) -> float | None:
    """Найти максимальный scale, сохраняющий sparse mass-условие."""
    target_mass = float(target_mass)
    utils.require(
        target_mass > 0 and np.isfinite(target_mass),
        "target_mass must be finite and positive",
    )

    def enough(scale):
        """Проверить mass-критерий для пробного sparse scale."""
        blocks = build_blocks(scale)
        if rowwise:
            return _box_sparse_mass_enough(blocks, target_mass)
        return _mean_sparse_mass(blocks) >= target_mass

    if enough(1.0):
        return 1.0
    if not enough(0.0):
        return None
    low, high = 0.0, 1.0
    tolerance = np.sqrt(np.finfo(float).eps)
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if enough(middle):
            low = middle
        else:
            high = middle
    return float(low)


def _orient_basis(basis):
    """Ориентировать столбцы SVD-базиса по ведущему ненулевому компоненту."""
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis


def initialize_basis_local_sparse(
    X,
    Y,
    engine: NeighborhoodEngine,
    N_lin: int,
    kernel,
    local_ridge: float,
    index_dim: int,
) -> np.ndarray:
    """Инициализировать локальный multi-index basis по sparse weighted ridge fits."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    utils.require(
        X.shape == engine.X.shape and np.array_equal(X, engine.X),
        "X must match the neighborhood engine data",
    )
    utils.require(
        Y.shape == (len(X),) and np.all(np.isfinite(Y)),
        "Y must have finite shape (n,)",
    )
    utils.require(
        not isinstance(N_lin, bool) and isinstance(N_lin, (int, np.integer)),
        "N_lin must be an integer",
        TypeError,
    )
    utils.require(N_lin >= 1, "N_lin must be positive")
    utils.require(
        not isinstance(index_dim, bool) and isinstance(index_dim, (int, np.integer)),
        "index_dim must be an integer",
        TypeError,
    )
    utils.require(
        1 <= index_dim <= X.shape[1],
        "index_dim must lie between 1 and d",
    )
    local_ridge = float(local_ridge)
    utils.require(
        np.isfinite(local_ridge) and local_ridge > 0,
        "local_ridge must be finite and positive",
    )

    h_lin = search_sparse_bandwidth(
        engine,
        N_lin,
        kernel,
        lower=np.finfo(float).eps,
    )
    blocks = engine.isotropic_blocks(h_lin, kernel, record=False)
    d = X.shape[1]
    ridge_rows = np.zeros((d, d + 1))
    ridge_rows[:, 1:] = np.sqrt(local_ridge) * np.eye(d)
    gradients = np.empty((len(engine.centers), d))

    for block in blocks:
        for local_row in range(block.rows):
            row = block.start + local_row
            indices, weights = block.row(local_row)
            design = np.column_stack(
                (np.ones(len(indices)), X[indices] - engine.centers[row])
            )
            root_weight = np.sqrt(weights)
            augmented_design = np.vstack((design * root_weight[:, None], ridge_rows))
            augmented_Y = np.concatenate((Y[indices] * root_weight, np.zeros(d)))
            gradients[row] = np.linalg.lstsq(
                augmented_design,
                augmented_Y,
                rcond=None,
            )[0][1:]

    _, singular_values, right_vectors = np.linalg.svd(
        gradients,
        full_matrices=False,
    )
    threshold = (
        np.finfo(float).eps
        * max(gradients.shape)
        * (singular_values[0] if len(singular_values) else 0.0)
    )
    utils.require(
        len(singular_values) >= index_dim
        and singular_values[index_dim - 1] > threshold,
        "local gradients do not identify the requested index",
        RuntimeError,
    )
    return _orient_basis(right_vectors[:index_dim].T.copy())
