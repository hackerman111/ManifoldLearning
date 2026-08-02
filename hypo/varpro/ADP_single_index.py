from time import perf_counter

import numpy as np

from hypo.single_index.ADP_single_index import ADP_single_index as _ADP_single_index


class ADP_single_index(_ADP_single_index):
    def __init__(self, *, lbfgs_memory=10, **kwargs):
        if (
            isinstance(lbfgs_memory, bool)
            or not isinstance(lbfgs_memory, (int, np.integer))
            or lbfgs_memory < 1
        ):
            raise ValueError("lbfgs_memory must be a positive integer")
        kwargs.setdefault("inner_steps", 50)
        super().__init__(**kwargs)
        self.lbfgs_memory = lbfgs_memory

    def fit(self, X, Y):
        super().fit(X, Y)
        timings = self.timings_
        self.timings_ = {
            "initialization": timings["initialization"],
            "rho": timings["rho"],
            "directions": timings["directions"],
            "weights": timings["weights"],
            "statistics": timings["statistics"],
            "rlbfgs": timings["lsmr"],
            "total": timings["total"],
        }
        return self

    def _profiled_value_gradient(self, I, U, beta):
        projected = U @ beta
        numerator = np.sum(I * projected, axis=1)
        denominator = np.sum(projected * projected, axis=1) + self.local_ridge
        slopes = numerator / denominator
        residual = I - slopes[:, None] * projected
        value = 0.5 * (
            np.sum(residual * residual)
            + self.local_ridge * np.dot(slopes, slopes)
        )
        euclidean = -np.einsum(
            "j,jpd,jp->d", slopes, U, residual, optimize=True
        )
        gradient = euclidean - beta * np.dot(beta, euclidean)
        if not (
            np.isfinite(value)
            and np.all(np.isfinite(slopes))
            and np.all(np.isfinite(gradient))
        ):
            raise RuntimeError("VarPro objective returned nonfinite values")
        return float(value), gradient, slopes

    @staticmethod
    def _lbfgs_direction(gradient, history):
        vector = gradient.copy()
        coefficients = []
        for step, change in reversed(history):
            inverse_curvature = 1.0 / np.dot(step, change)
            coefficient = inverse_curvature * np.dot(step, vector)
            coefficients.append(coefficient)
            vector -= coefficient * change
        if history:
            step, change = history[-1]
            vector *= np.dot(step, change) / np.dot(change, change)
        for (step, change), coefficient in zip(
            history, reversed(coefficients), strict=True
        ):
            inverse_curvature = 1.0 / np.dot(step, change)
            vector += step * (
                coefficient - inverse_curvature * np.dot(change, vector)
            )
        return -vector

    @staticmethod
    def _project(beta, vector):
        return vector - beta * np.dot(beta, vector)

    @staticmethod
    def _has_positive_curvature(step, change):
        if not (
            np.all(np.isfinite(step)) and np.all(np.isfinite(change))
        ):
            return False
        step_norm = np.linalg.norm(step)
        change_norm = np.linalg.norm(change)
        curvature = np.dot(step, change)
        return bool(
            np.isfinite(step_norm)
            and step_norm > 0
            and np.isfinite(change_norm)
            and change_norm > 0
            and np.isfinite(curvature)
            and curvature > 1e-10 * step_norm * change_norm
        )

    def _riemannian_lbfgs(self, I, U, beta):
        beta = np.asarray(beta, dtype=float).copy()
        norm = np.linalg.norm(beta)
        if not np.all(np.isfinite(beta)) or not np.isfinite(norm) or norm == 0:
            raise RuntimeError("VarPro received an invalid beta")
        beta /= norm
        initial_beta = beta.copy()
        value, gradient, slopes = self._profiled_value_gradient(I, U, beta)
        gradient_norm = np.linalg.norm(gradient)
        if not np.isfinite(gradient_norm):
            raise RuntimeError("VarPro returned an invalid gradient norm")

        history = []
        iterations = line_search_steps = 0
        status = "gradient" if gradient_norm <= self.tol else "max_iterations"
        while gradient_norm > self.tol and iterations < self.inner_steps:
            direction = self._project(
                beta, self._lbfgs_direction(gradient, history)
            )
            directional_derivative = np.dot(gradient, direction)
            if (
                not np.all(np.isfinite(direction))
                or not np.isfinite(directional_derivative)
                or directional_derivative >= 0
            ):
                direction = -gradient
                directional_derivative = -np.dot(gradient, gradient)
            if (
                not np.all(np.isfinite(direction))
                or not np.isfinite(directional_derivative)
            ):
                raise RuntimeError("VarPro returned an invalid direction")

            alpha = 1.0
            accepted = None
            for _ in range(30):
                candidate = beta + alpha * direction
                candidate_norm = np.linalg.norm(candidate)
                if (
                    not np.all(np.isfinite(candidate))
                    or not np.isfinite(candidate_norm)
                    or candidate_norm == 0
                ):
                    raise RuntimeError("VarPro retraction returned an invalid beta")
                candidate /= candidate_norm
                candidate_result = self._profiled_value_gradient(I, U, candidate)
                line_search_steps += 1
                if candidate_result[0] <= (
                    value + 1e-4 * alpha * directional_derivative
                ):
                    accepted = candidate, candidate_result
                    break
                alpha *= 0.5

            if accepted is None:
                status = "line_search_failed"
                break

            candidate, (new_value, new_gradient, new_slopes) = accepted
            transported_history = []
            for old_step, old_change in history:
                transported_step = self._project(candidate, old_step)
                transported_change = self._project(candidate, old_change)
                if self._has_positive_curvature(
                    transported_step, transported_change
                ):
                    transported_history.append(
                        (transported_step, transported_change)
                    )
            history = transported_history
            step = self._project(candidate, alpha * direction)
            change = new_gradient - self._project(candidate, gradient)
            if self._has_positive_curvature(step, change):
                history.append((step, change))
                history = history[-self.lbfgs_memory :]

            relative_change = abs(value - new_value) / max(1.0, abs(value))
            beta, value, gradient, slopes = (
                candidate,
                new_value,
                new_gradient,
                new_slopes,
            )
            iterations += 1
            gradient_norm = np.linalg.norm(gradient)
            if not np.isfinite(gradient_norm):
                raise RuntimeError("VarPro returned an invalid gradient norm")
            if gradient_norm <= self.tol:
                status = "gradient"
                break
            if relative_change <= self.tol:
                status = "objective"
                break
        else:
            status = "gradient" if gradient_norm <= self.tol else "max_iterations"

        if np.dot(beta, initial_beta) < 0:
            beta = -beta
            slopes = -slopes
        delta = min(
            np.linalg.norm(beta - initial_beta),
            np.linalg.norm(beta + initial_beta),
        )
        return (
            beta,
            slopes,
            {
                "inner_iterations": iterations,
                "solver_status": status,
                "solver_iterations": iterations,
                "objective": float(value),
                "gradient_norm": float(gradient_norm),
                "line_search_steps": line_search_steps,
                "beta_delta": float(delta),
            },
        )

    def _alternating(self, statistics, beta):
        started = perf_counter()
        result = self._riemannian_lbfgs(statistics["I"], statistics["U"], beta)
        self.timings_["lsmr"] += perf_counter() - started
        return result
