from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from hypo.single_index.ADP_single_index import ADP_single_index as _ADP_single_index
from scipy.optimize import minimize_scalar

_CHECK_EVERY = 5
_MAX_KRYLOV = 1000
_GCV_GRID_SIZE = 21
_LAMBDA_LOG10_TOL = 0.1
_PROJECTIVE_DIRECTION_TOL = 1e-3
_RELATIVE_GCV_TOL = 1e-2


class ADP_single_index(_ADP_single_index):
    def fit(self, X, Y):
        self.lambda_ = None
        self._fixed_validation_folds = None
        super().fit(X, Y)
        timings = self.timings_
        self.timings_ = {
            name: timings["lsmr" if name == "krylov" else name]
            for name in (
                "initialization",
                "rho",
                "directions",
                "weights",
                "statistics",
                "slopes",
                "krylov",
                "total",
            )
        }
        return self

    def _calculate_statistics(self, X, Y, weights, directions):
        statistics = super()._calculate_statistics(X, Y, weights, directions)
        if self.beta_init == "random":
            statistics["cross_fitted"] = ()
            return statistics
        # Keep the strong local start on one validation target so later
        # bandwidth changes cannot accept directional drift.
        if self._fixed_validation_folds is not None:
            statistics["cross_fitted"] = self._fixed_validation_folds
            return statistics
        permutation = np.random.default_rng(self.seed).permutation(X.shape[0])
        first = np.zeros(X.shape[0], dtype=bool)
        first[permutation[::2]] = True
        second = ~first
        usable = (np.sum(weights[:, first], axis=1) > 0) & (
            np.sum(weights[:, second], axis=1) > 0
        )
        folds = []
        for training, validation in ((first, second), (second, first)):
            if not np.any(usable):
                break
            folds.append(
                (
                    super()._calculate_statistics(
                        X[training],
                        Y[training],
                        weights[usable][:, training],
                        directions[usable],
                    ),
                    super()._calculate_statistics(
                        X[validation],
                        Y[validation],
                        weights[usable][:, validation],
                        directions[usable],
                    ),
                )
            )
        self._fixed_validation_folds = folds
        statistics["cross_fitted"] = folds
        return statistics

    def _cross_fitted_moment_loss(self, folds, beta):
        if not folds:
            return None
        squared_error = 0.0
        count = 0
        for training, validation in folds:
            slopes = self._slopes(
                training["I"],
                training["U"],
                beta,
            )
            residual = validation["I"] - slopes[:, None] * (validation["U"] @ beta)
            squared_error += float(np.sum(np.square(residual)))
            count += residual.size
        loss = squared_error / count
        if not np.isfinite(loss) or loss < 0:
            raise RuntimeError("cross-fitted moment loss is invalid")
        return loss

    def _select_lambda(self, B, rho, lambda_prior, full_rows=None):
        B = np.asarray(B, dtype=float)
        if (
            B.ndim != 2
            or B.shape[0] != B.shape[1] + 1
            or B.shape[1] == 0
            or not np.all(np.isfinite(B))
            or not np.isfinite(rho)
            or rho <= 0
        ):
            raise RuntimeError("invalid projected Krylov problem")

        try:
            left, singular_values, right_transpose = np.linalg.svd(
                B, full_matrices=False
            )
        except np.linalg.LinAlgError as error:
            raise RuntimeError("projected Krylov SVD failed") from error
        if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
            raise RuntimeError("projected Krylov spectrum is invalid")
        largest = singular_values[0]
        if largest <= 0:
            raise RuntimeError("projected Krylov spectrum is invalid")
        if full_rows is None:
            full_rows = B.shape[0]
        if (
            isinstance(full_rows, (bool, np.bool_))
            or not isinstance(full_rows, (int, np.integer))
            or full_rows < B.shape[1]
        ):
            raise RuntimeError("invalid full problem row count")

        rhs = np.zeros(B.shape[0])
        rhs[0] = rho
        projected_rhs = left.T @ rhs
        right = right_transpose.T
        squared = np.square(singular_values)
        scale = squared[0]
        low, high = 1e-8 * scale, 1e2 * scale
        if lambda_prior is not None:
            try:
                prior = float(lambda_prior)
            except (TypeError, ValueError):
                prior = math.nan
            if np.isfinite(prior) and prior > 0:
                low = min(low, prior / 100)
                high = max(high, 100 * prior)
        if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
            raise RuntimeError("invalid projected GCV interval")

        def evaluate(lambda_value):
            if not np.isfinite(lambda_value) or lambda_value <= 0:
                return math.inf, None
            coordinates = right @ (
                singular_values / (squared + lambda_value) * projected_rhs
            )
            residual = B @ coordinates - rhs
            denominator = full_rows - np.sum(squared / (squared + lambda_value))
            criterion = np.linalg.norm(residual) ** 2 / denominator**2
            if (
                not np.all(np.isfinite(coordinates))
                or not np.all(np.isfinite(residual))
                or not np.isfinite(denominator)
                or denominator <= 0
                or not np.isfinite(criterion)
                or criterion < 0
            ):
                return math.inf, None
            return float(criterion), coordinates

        grid = np.geomspace(low, high, _GCV_GRID_SIZE)
        scores = np.array([evaluate(value)[0] for value in grid])
        if not np.any(np.isfinite(scores)):
            raise RuntimeError("projected GCV has no finite criterion")
        best = int(np.argmin(scores))

        def tied_boundaries():
            def tied(score):
                tolerance = (
                    64
                    * np.finfo(float).eps
                    * max(
                        np.finfo(float).tiny,
                        abs(score),
                        abs(scores[best]),
                    )
                )
                return bool(
                    np.isfinite(score) and abs(score - scores[best]) <= tolerance
                )

            return (
                tied(scores[0]),
                tied(scores[-1]),
            )

        lower_boundary, upper_boundary = tied_boundaries()
        if lower_boundary or upper_boundary:
            if lower_boundary and not upper_boundary:
                low /= 10
            else:
                high *= 10
            if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
                raise RuntimeError("could not expand projected GCV interval")
            grid = np.geomspace(low, high, _GCV_GRID_SIZE)
            scores = np.array([evaluate(value)[0] for value in grid])
            if not np.any(np.isfinite(scores)):
                raise RuntimeError("projected GCV has no finite criterion")
            best = int(np.argmin(scores))
            lower_boundary, upper_boundary = tied_boundaries()

        selected_lambda = float(grid[best])
        selected_gcv = float(scores[best])
        boundary = lower_boundary or upper_boundary
        if not boundary:
            refined = minimize_scalar(
                lambda log_lambda: evaluate(math.exp(log_lambda))[0],
                bounds=(math.log(grid[best - 1]), math.log(grid[best + 1])),
                method="bounded",
                options={"xatol": 1e-4},
            )
            refined_lambda = math.exp(refined.x)
            refined_gcv, _ = evaluate(refined_lambda)
            if (
                refined.success
                and np.isfinite(refined_lambda)
                and refined_lambda > 0
                and np.isfinite(refined_gcv)
                and refined_gcv < selected_gcv
            ):
                selected_lambda = refined_lambda
                selected_gcv = refined_gcv

        selected_gcv, coordinates = evaluate(selected_lambda)
        if (
            not np.isfinite(selected_lambda)
            or selected_lambda <= 0
            or not np.isfinite(selected_gcv)
            or selected_gcv < 0
            or coordinates is None
            or not np.all(np.isfinite(coordinates))
        ):
            raise RuntimeError("projected GCV selection failed")
        return selected_lambda, selected_gcv, coordinates, bool(boundary)

    @staticmethod
    def _reorthogonalize(vector, basis):
        vector = vector.copy()
        for _ in range(2):
            for basis_vector in basis:
                vector -= np.dot(basis_vector, vector) * basis_vector
        return vector

    def _hybrid_krylov(self, I, U, slopes, beta_prior):
        d = U.shape[2]
        previous_lambda = getattr(self, "lambda_", None)

        def matvec(vector):
            return (slopes[:, None] * (U @ vector)).ravel()

        def rmatvec(vector):
            return np.einsum(
                "j,jpd,jp->d",
                slopes,
                U,
                vector.reshape(I.shape),
                optimize=True,
            )

        beta_prior = np.asarray(beta_prior, dtype=float)
        if beta_prior.shape != (d,) or not np.all(np.isfinite(beta_prior)):
            raise RuntimeError("Krylov solver received an invalid beta")
        fitted = matvec(beta_prior)
        residual = I.ravel() - fitted
        right_hand_side_norm = np.linalg.norm(I.ravel())
        fitted_norm = np.linalg.norm(fitted)
        rho = np.linalg.norm(residual)
        residual_tolerance = np.finfo(float).eps * max(
            right_hand_side_norm, fitted_norm
        )
        # Frobenius norm of A without materializing its rows.
        operator_scale_squared = np.einsum(
            "j,jpd,jpd->",
            np.square(slopes),
            U,
            U,
            optimize=True,
        )
        if not (
            np.all(np.isfinite(fitted))
            and np.all(np.isfinite(residual))
            and np.isfinite(right_hand_side_norm)
            and np.isfinite(fitted_norm)
            and np.isfinite(rho)
            and np.isfinite(residual_tolerance)
            and np.isfinite(operator_scale_squared)
            and operator_scale_squared >= 0
        ):
            raise RuntimeError("Krylov solver received nonfinite data")
        if rho <= residual_tolerance:
            return beta_prior.copy(), {
                "solver_status": "zero_residual",
                "krylov_iterations": 0,
                "selected_lambda": previous_lambda,
                "projected_gcv": 0.0,
                "lambda_at_boundary": False,
            }

        unit_roundoff = np.finfo(float).eps
        recurrence_tolerance = unit_roundoff * math.sqrt(operator_scale_squared)
        u = residual / rho
        transpose_product = rmatvec(u)
        alpha = np.linalg.norm(transpose_product)
        if not (
            np.all(np.isfinite(u))
            and np.all(np.isfinite(transpose_product))
            and np.isfinite(alpha)
        ):
            raise RuntimeError("Krylov recurrence returned nonfinite values")
        if alpha <= recurrence_tolerance:
            return beta_prior.copy(), {
                "solver_status": "breakdown",
                "krylov_iterations": 0,
                "selected_lambda": previous_lambda,
                "projected_gcv": float(rho**2),
                "lambda_at_boundary": False,
            }
        v = transpose_product / alpha

        basis = []
        left_basis = [u.copy()]
        diagonal = []
        subdiagonal = []
        previous_checkpoint = None
        stable_checkpoints = 0
        valid_candidate = None
        kmax = min(d, _MAX_KRYLOV)

        def finish(candidate, record, status=None, iterations=None):
            record = dict(record)
            if status is not None:
                record["solver_status"] = status
            if iterations is not None:
                record["krylov_iterations"] = iterations
            self.lambda_ = record["selected_lambda"]
            return candidate, record

        for iteration in range(1, kmax + 1):
            if not (
                np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.isfinite(alpha)
            ):
                raise RuntimeError("Krylov recurrence returned nonfinite values")
            basis.append(v.copy())
            diagonal.append(float(alpha))

            product = matvec(v)
            next_u_raw = self._reorthogonalize(product - alpha * u, left_basis)
            beta_coefficient = np.linalg.norm(next_u_raw)
            if not (
                np.all(np.isfinite(product))
                and np.all(np.isfinite(next_u_raw))
                and np.isfinite(beta_coefficient)
            ):
                raise RuntimeError("Krylov recurrence returned nonfinite values")
            beta_breakdown = beta_coefficient <= recurrence_tolerance
            subdiagonal.append(float(beta_coefficient))

            next_u = next_v = None
            next_alpha = math.nan
            alpha_breakdown = False
            if not beta_breakdown:
                next_u = next_u_raw / beta_coefficient
                left_basis.append(next_u.copy())
                next_transpose_product = rmatvec(next_u)
                next_v_raw = self._reorthogonalize(
                    next_transpose_product - beta_coefficient * v,
                    basis,
                )
                next_alpha = np.linalg.norm(next_v_raw)
                if not (
                    np.all(np.isfinite(next_u))
                    and np.all(np.isfinite(next_transpose_product))
                    and np.all(np.isfinite(next_v_raw))
                    and np.isfinite(next_alpha)
                ):
                    raise RuntimeError("Krylov recurrence returned nonfinite values")
                alpha_breakdown = next_alpha <= recurrence_tolerance
                if not alpha_breakdown:
                    next_v = next_v_raw / next_alpha

            breakdown = beta_breakdown or alpha_breakdown
            at_limit = iteration == kmax
            if iteration % _CHECK_EVERY == 0 or at_limit or breakdown:
                B = np.zeros((iteration + 1, iteration))
                indices = np.arange(iteration)
                B[indices, indices] = diagonal
                B[indices + 1, indices] = subdiagonal
                V = np.column_stack(basis)
                try:
                    (
                        selected_lambda,
                        projected_gcv,
                        coordinates,
                        boundary,
                    ) = self._select_lambda(
                        B,
                        rho,
                        previous_lambda,
                        I.size,
                    )
                    candidate = beta_prior + V @ coordinates
                    candidate_norm = np.linalg.norm(candidate)
                    if (
                        not np.all(np.isfinite(candidate))
                        or not np.isfinite(candidate_norm)
                        or candidate_norm == 0
                    ):
                        raise RuntimeError("Krylov solver returned an invalid beta")
                    candidate /= candidate_norm
                    if np.dot(candidate, beta_prior) < 0:
                        candidate = -candidate
                except RuntimeError:
                    if breakdown and valid_candidate is not None:
                        candidate, record = valid_candidate
                        return finish(
                            candidate,
                            record,
                            "breakdown",
                            iteration,
                        )
                    raise

                record = {
                    "solver_status": "max_iterations",
                    "krylov_iterations": iteration,
                    "selected_lambda": selected_lambda,
                    "projected_gcv": projected_gcv,
                    "lambda_at_boundary": boundary,
                }
                valid_candidate = candidate, record

                if previous_checkpoint is not None:
                    previous_beta, checkpoint_lambda, previous_gcv = previous_checkpoint
                    dot = np.clip(abs(np.dot(candidate, previous_beta)), 0.0, 1.0)
                    stable = (
                        not boundary
                        and abs(
                            math.log10(selected_lambda) - math.log10(checkpoint_lambda)
                        )
                        < _LAMBDA_LOG10_TOL
                        and math.sqrt(2 * (1 - dot)) < _PROJECTIVE_DIRECTION_TOL
                        and abs(projected_gcv - previous_gcv)
                        / max(1.0, abs(projected_gcv))
                        < _RELATIVE_GCV_TOL
                    )
                    stable_checkpoints = stable_checkpoints + 1 if stable else 0
                previous_checkpoint = (
                    candidate.copy(),
                    selected_lambda,
                    projected_gcv,
                )

                if breakdown:
                    return finish(candidate, record, "breakdown")
                if stable_checkpoints >= 2:
                    return finish(candidate, record, "stabilized")
                if at_limit:
                    return finish(candidate, record)

            if breakdown:
                if valid_candidate is None:
                    raise RuntimeError("Krylov breakdown before a valid projection")
                candidate, record = valid_candidate
                return finish(
                    candidate,
                    record,
                    "breakdown",
                    iteration,
                )
            u, v, alpha = next_u, next_v, next_alpha

        raise RuntimeError("Krylov solver did not produce a valid candidate")

    def _alternating(self, statistics, beta):
        I, U = statistics["I"], statistics["U"]
        folds = statistics.get("cross_fitted", ())
        delta = math.inf
        record = None
        for inner in range(self.inner_steps):
            prior = beta
            prior_loss = self._cross_fitted_moment_loss(folds, prior)
            started = perf_counter()
            slopes = self._slopes(I, U, prior)
            self.timings_["slopes"] += perf_counter() - started
            started = perf_counter()
            candidate, record = self._hybrid_krylov(I, U, slopes, prior)
            self.timings_["lsmr"] += perf_counter() - started
            candidate_loss = self._cross_fitted_moment_loss(folds, candidate)
            accepted = prior_loss is None or candidate_loss <= prior_loss
            beta = candidate if accepted else prior
            delta = min(
                np.linalg.norm(beta - prior),
                np.linalg.norm(beta + prior),
            )
            if delta < self.tol:
                break
        started = perf_counter()
        slopes = self._slopes(I, U, beta)
        self.timings_["slopes"] += perf_counter() - started
        return (
            beta,
            slopes,
            {
                **record,
                "inner_iterations": inner + 1,
                "beta_delta": float(delta),
                "candidate_accepted": bool(accepted),
                "validation_loss": (candidate_loss if accepted else prior_loss),
            },
        )
