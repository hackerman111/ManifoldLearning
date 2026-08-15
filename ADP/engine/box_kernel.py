from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.spatial import cKDTree

_BOX_DISTANCE_CACHE_MAX_BYTES = 64 * 1024**2
_BOX_ROW_COVERAGE = 0.9


def _validated_tau(tau: float) -> float:
    tau = float(tau)
    if not np.isfinite(tau) or not 0.0 < tau < 1.0:
        raise ValueError("tau must be finite and lie in (0, 1)")
    return tau


def _kernel_input(value) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError("kernel input must be finite")
    return value


def box_kernel(value) -> np.ndarray:
    return np.asarray(_kernel_input(value) < 1.0, dtype=float)


def plateau_kernel(value, *, tau: float = 0.5) -> np.ndarray:
    value = _kernel_input(value)
    tau = _validated_tau(tau)
    result = np.zeros_like(value)
    result[value <= tau] = 1.0
    boundary = (value > tau) & (value < 1.0)
    s = (value[boundary] - tau) / (1.0 - tau)
    result[boundary] = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5
    return result


def make_plateau_kernel(tau: float = 0.5):
    return partial(plateau_kernel, tau=_validated_tau(tau))


def sparse_kernel_parameters(kernel) -> tuple[str, float | None] | None:
    if kernel is box_kernel:
        return "box", None
    if kernel is plateau_kernel:
        return "plateau", 0.5
    if isinstance(kernel, partial) and kernel.func is plateau_kernel:
        if kernel.args or set(kernel.keywords or {}) - {"tau"}:
            raise ValueError("plateau kernel accepts only keyword tau")
        return "plateau", _validated_tau((kernel.keywords or {}).get("tau", 0.5))
    return None


def _readonly_array(value, dtype, name, *, ndim=1):
    value = np.asarray(value, dtype=dtype)
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
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
        if isinstance(self.start, bool) or not isinstance(
            self.start, (int, np.integer)
        ) or self.start < 0:
            raise ValueError("start must be a nonnegative integer")
        if isinstance(self.n_observations, bool) or not isinstance(
            self.n_observations, (int, np.integer)
        ) or self.n_observations < 1:
            raise ValueError("n_observations must be a positive integer")

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
        if len(indptr) < 2 or indptr[0] != 0 or indptr[-1] != len(indices):
            raise ValueError("indptr must delimit at least one complete row")
        if np.any(np.diff(indptr) < 0):
            raise ValueError("indptr must be nondecreasing")
        if np.any(indices < 0) or np.any(indices >= self.n_observations):
            raise ValueError("indices are out of range")
        if len(positions) != len(weights):
            raise ValueError("boundary positions and weights must have equal length")
        if len(positions) and (
            np.any(positions < 0)
            or np.any(positions >= len(indices))
            or np.any(np.diff(positions) <= 0)
        ):
            raise ValueError("boundary positions must be sorted unique edge positions")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0) or np.any(
            weights >= 1
        ):
            raise ValueError("boundary weights must be finite and lie in (0, 1)")
        if self.mode not in {"box", "plateau"}:
            raise ValueError("mode must be 'box' or 'plateau'")
        if self.mode == "box":
            if len(positions) or self.tau is not None:
                raise ValueError("box blocks cannot contain boundary data")
        else:
            object.__setattr__(self, "tau", _validated_tau(self.tau))

        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "boundary_positions", positions)
        object.__setattr__(self, "boundary_weights", weights)
        if np.any(self.mass <= 0):
            raise ValueError("every sparse neighborhood row must have positive mass")

    @property
    def rows(self) -> int:
        return len(self.indptr) - 1

    @property
    def edge_count(self) -> int:
        return len(self.indices)

    @property
    def boundary_count(self) -> int:
        return len(self.boundary_weights)

    @property
    def mass(self) -> np.ndarray:
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
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be an integer")
        if not 0 <= row < self.rows:
            raise IndexError(row)
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
        self._entries = set()
        self._active = set()
        self._call_hits = 0
        self.last_hits = 0
        self.total_hits = 0

    def begin_call(self):
        self._active = set()
        self._call_hits = 0

    def finish_call(self):
        self._entries = self._active
        self.last_hits = self._call_hits
        self.total_hits += self._call_hits

    def observe(self, block: SparseNeighborhoodBlock):
        for row in range(block.rows):
            key = block.cache_key(row)
            if key in self._entries:
                self._call_hits += 1
            else:
                self._entries.add(key)
            self._active.add(key)


class NeighborhoodEngine:
    def __init__(self, X, centers, block_size=128):
        X = np.asarray(X, dtype=float)
        centers = np.asarray(centers, dtype=float)
        if X.ndim != 2 or 0 in X.shape or not np.all(np.isfinite(X)):
            raise ValueError("X must have finite non-empty shape (n, d)")
        if (
            centers.ndim != 2
            or centers.shape[1] != X.shape[1]
            or not len(centers)
            or not np.all(np.isfinite(centers))
        ):
            raise ValueError("centers must have finite non-empty shape (J, d)")
        if isinstance(block_size, bool) or not isinstance(
            block_size, (int, np.integer)
        ) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
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
        required_bytes = (
            len(self.X) * len(self.centers) * np.dtype(float).itemsize
        )
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
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    @staticmethod
    def _scale(value, name):
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
        return value

    def _beta(self, beta):
        beta = np.asarray(beta, dtype=float)
        if beta.shape != (self.X.shape[1],) or not np.all(np.isfinite(beta)):
            raise ValueError("beta must have finite shape (d,)")
        norm = np.linalg.norm(beta)
        if norm != 0 and not np.isclose(norm, 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError("beta must be zero or unit length")
        return beta

    def _multi_basis(self, basis, eigenvalues):
        basis = np.asarray(basis, dtype=float)
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        if (
            basis.ndim != 2
            or basis.shape[0] != self.X.shape[1]
            or not basis.shape[1]
            or not np.all(np.isfinite(basis))
        ):
            raise ValueError("basis must have finite shape (d, m)")
        if eigenvalues.shape != (basis.shape[1],) or not np.all(
            np.isfinite(eigenvalues)
        ) or np.any(eigenvalues < 0):
            raise ValueError("eigenvalues must have finite nonnegative shape (m,)")
        if not np.allclose(
            basis.T @ basis,
            np.eye(basis.shape[1]),
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("basis columns must be orthonormal")
        return basis, eigenvalues

    def _box_requirements(self, target):
        target = float(target)
        if not np.isfinite(target) or target <= 0:
            raise ValueError("target mass must be finite and positive")
        return (
            int(np.ceil(target)),
            int(np.ceil(_BOX_ROW_COVERAGE * len(self.centers))),
        )

    def _single_box_row_counts(self, beta, h, rho, distance2):
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
        x_projection = self.X @ beta
        center_projection = self.centers @ beta
        h2 = h**2
        rho2 = rho**2
        return [
            np.flatnonzero(
                rho2 * distance2[row]
                + np.square(x_projection - center_projection[row])
                < h2
            )
            for row in range(len(self.centers))
        ]

    def _multi_box_components(self, basis, eigenvalues, distance2):
        projected_X = self.X @ basis
        projected_centers = self.centers @ basis
        projected2 = np.zeros_like(distance2)
        principal2 = np.zeros_like(distance2)
        for column, eigenvalue in enumerate(eigenvalues):
            delta2 = (
                projected_centers[:, column, None]
                - projected_X[None, :, column]
            )
            np.square(delta2, out=delta2)
            projected2 += delta2
            delta2 *= eigenvalue
            principal2 += delta2
        np.subtract(distance2, projected2, out=projected2)
        np.maximum(projected2, 0.0, out=projected2)
        return projected2, principal2

    @staticmethod
    def _multi_box_key(basis, eigenvalues, h, alpha):
        return basis.tobytes(), eigenvalues.tobytes(), float(h), float(alpha)

    def _multi_box_supports(self, orthogonal2, principal2, h, alpha):
        h2 = h**2
        alpha2 = alpha**2
        return [
            np.flatnonzero(alpha2 * orthogonal2[row] + principal2[row] < h2)
            for row in range(len(self.centers))
        ]

    def select_single_box_scale(self, beta, h, target):
        beta = self._beta(beta)
        h = self._positive(h, "h")
        required, covered = self._box_requirements(target)
        distance2 = self._box_distance2()
        if distance2 is None:
            return NotImplemented

        def enough(counts):
            return np.count_nonzero(counts >= required) >= covered

        x_projection = self.X @ beta
        center_projection = self.centers @ beta
        projection2 = np.square(
            center_projection[:, None] - x_projection[None, :]
        )
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
            if next_rho == rho:
                raise RuntimeError("could not represent a feasible box scale")
            rho = next_rho
        return rho

    def select_multi_box_scale(self, basis, eigenvalues, h, target):
        basis, eigenvalues = self._multi_basis(basis, eigenvalues)
        h = self._positive(h, "h")
        required, covered = self._box_requirements(target)
        distance2 = self._box_distance2()
        if distance2 is None:
            return NotImplemented

        def enough(counts):
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
            supports = self._multi_box_supports(
                orthogonal2, principal2, h, alpha
            )
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
            if next_alpha == alpha:
                raise RuntimeError("could not represent a feasible box scale")
            alpha = next_alpha

        supports = [
            np.flatnonzero(row > alpha**2)
            for row in principal2
        ]
        self._multi_box_support_cache = (
            self._multi_box_key(basis, eigenvalues, h, alpha),
            supports,
        )
        return alpha

    @staticmethod
    def _candidate_array(value):
        return np.asarray(value, dtype=np.intp)

    def _euclidean_candidates(self, radius):
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
        if right is None:
            return left
        return np.intersect1d(left, right, assume_unique=True)

    def _encode(self, supports, q_values, kernel, record):
        parameters = sparse_kernel_parameters(kernel)
        if parameters is None:
            raise ValueError("sparse neighborhoods require box or plateau kernel")
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
                position_rows = []
                weight_rows = []
                for offset, q in zip(indptr[:-1], values):
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
                else sum(left == right for left, right in zip(previous, keys))
            )
            self._previous_keys = keys
            self.last_support_edges = sum(block.edge_count for block in blocks)
            self.last_boundary_edges = sum(
                block.boundary_count for block in blocks
            )
        return iter(blocks)

    def isotropic_blocks(self, h, kernel, *, record=False):
        h = self._positive(h, "h")
        parameters = sparse_kernel_parameters(kernel)
        distance2 = (
            self._box_distance2()
            if parameters == ("box", None)
            else None
        )
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
        for center, indices in zip(self.centers, candidates):
            difference = self.X[indices] - center
            q = np.einsum("nd,nd->n", difference, difference) / h**2
            active = q < 1.0
            supports.append(indices[active])
            q_values.append(q[active])
        return self._encode(supports, q_values, kernel, record)

    def single_blocks(self, beta, h, rho, kernel, *, record=True):
        beta = self._beta(beta)
        h = self._positive(h, "h")
        rho = self._scale(rho, "rho")
        parameters = sparse_kernel_parameters(kernel)
        distance2 = (
            self._box_distance2()
            if parameters == ("box", None)
            else None
        )
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
        basis, eigenvalues = self._multi_basis(basis, eigenvalues)
        h = self._positive(h, "h")
        alpha = self._scale(alpha, "alpha")

        parameters = sparse_kernel_parameters(kernel)
        distance2 = (
            self._box_distance2()
            if parameters == ("box", None)
            else None
        )
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
    total = 0.0
    rows = 0
    for block in blocks:
        mass = block.mass
        total += float(mass.sum())
        rows += len(mass)
    if rows == 0:
        raise ValueError("sparse blocks must contain at least one row")
    return total / rows


def _box_sparse_mass_enough(blocks, target) -> bool:
    required = int(np.ceil(target))
    qualified = 0
    rows = 0
    for block in blocks:
        mass = block.mass
        qualified += int(np.count_nonzero(mass >= required))
        rows += len(mass)
    if rows == 0:
        raise ValueError("sparse blocks must contain at least one row")
    return qualified >= int(np.ceil(_BOX_ROW_COVERAGE * rows))


def search_sparse_bandwidth(
    engine: NeighborhoodEngine,
    target: float,
    kernel,
    *,
    lower: float,
) -> float:
    target = float(target)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be finite and positive")
    lower = engine._positive(lower, "lower")
    parameters = sparse_kernel_parameters(kernel)
    if parameters is None:
        raise ValueError("sparse bandwidth requires box or plateau kernel")
    if parameters == ("box", None):
        distance2 = engine._box_distance2()
        if distance2 is not None:
            required, covered = engine._box_requirements(target)
            if required > distance2.shape[1]:
                raise RuntimeError("could not bracket a feasible bandwidth")
            if np.count_nonzero(
                np.count_nonzero(distance2 < lower**2, axis=1) >= required
            ) >= covered:
                return lower
            selection = distance2.copy()
            selection.partition(required - 1, axis=1)
            thresholds = selection[:, required - 1].copy()
            thresholds.partition(covered - 1)
            threshold = float(thresholds[covered - 1])
            h = float(np.nextafter(np.sqrt(threshold), np.inf))
            while np.count_nonzero(
                np.count_nonzero(distance2 < h**2, axis=1) >= required
            ) < covered:
                h = float(np.nextafter(h, np.inf))
            return h

    def enough(h):
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
        raise RuntimeError("could not bracket a feasible bandwidth")

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
    target_mass = float(target_mass)
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("target_mass must be finite and positive")

    def enough(scale):
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
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != engine.X.shape or not np.array_equal(X, engine.X):
        raise ValueError("X must match the neighborhood engine data")
    if Y.shape != (len(X),) or not np.all(np.isfinite(Y)):
        raise ValueError("Y must have finite shape (n,)")
    if isinstance(N_lin, bool) or not isinstance(N_lin, (int, np.integer)):
        raise TypeError("N_lin must be an integer")
    if N_lin < 1:
        raise ValueError("N_lin must be positive")
    if isinstance(index_dim, bool) or not isinstance(index_dim, (int, np.integer)):
        raise TypeError("index_dim must be an integer")
    if not 1 <= index_dim <= X.shape[1]:
        raise ValueError("index_dim must lie between 1 and d")
    local_ridge = float(local_ridge)
    if not np.isfinite(local_ridge) or local_ridge <= 0:
        raise ValueError("local_ridge must be finite and positive")

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
            augmented_design = np.vstack(
                (design * root_weight[:, None], ridge_rows)
            )
            augmented_Y = np.concatenate(
                (Y[indices] * root_weight, np.zeros(d))
            )
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
    if len(singular_values) < index_dim or singular_values[index_dim - 1] <= threshold:
        raise RuntimeError("local gradients do not identify the requested index")
    return _orient_basis(right_vectors[:index_dim].T.copy())
