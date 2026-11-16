"""Минибатчи только внутри frozen Grassmann solve; public ADP не меняется."""

from __future__ import annotations

import numpy as np

from ADP.solver import grassman as g
from ADP.solver.LSMR import HPAOResult

EPOCHS = 2
BATCHES = 4
FINAL_STEPS = 4


def _frame(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.number):
        raise ValueError("basis must be real numeric")
    B = np.asarray(value, dtype=float)
    if (
        B.ndim != 2
        or not 1 <= B.shape[0] < B.shape[1]
        or not np.all(np.isfinite(B))
        or not np.allclose(B @ B.T, np.eye(len(B)), rtol=1e-9, atol=1e-11)
    ):
        raise ValueError("basis must be finite orthonormal rows with 1 <= m < d")
    return B


def cosine_step(t: int, updates: int) -> float:
    """Явный cosine EMA от .5 до .05, без переноса scaling law статьи."""
    if updates < 1 or not 0 <= t < updates:
        raise ValueError("invalid cosine schedule index")
    if updates == 1:
        return 0.5
    return float(0.5 * (0.1 + 0.45 * (1 + np.cos(np.pi * t / (updates - 1)))))


class EmaSubspace:
    """Непрерывный PSD EMA через d-by-r фактор; история не усекается."""

    def __init__(self, B: np.ndarray) -> None:
        B = _frame(B)
        self.shape = B.shape
        self.mass = 1e-10
        self.factor = np.sqrt(self.mass) * B.T

    def update(self, B: np.ndarray, alpha: float) -> None:
        B = _frame(B)
        if B.shape != self.shape:
            raise ValueError("EMA basis shape changed")
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be finite in (0,1]")
        self.factor = np.concatenate(
            (np.sqrt(1 - alpha) * self.factor, np.sqrt(alpha) * B.T), axis=1
        )
        self.mass = (1 - alpha) * self.mass + alpha

    def proposal(self) -> tuple[np.ndarray | None, float]:
        left, singular, _ = np.linalg.svd(self.factor, full_matrices=False)
        m = self.shape[0]
        values = singular**2 / self.mass
        gap = float(values[m - 1] - (values[m] if len(values) > m else 0))
        if gap <= 64 * np.finfo(float).eps * max(1.0, float(values[0])):
            return None, gap
        return left[:, :m].T.copy(), gap


class MiniBatchSolver:
    """APPROXIMATE optimizer; каждое возвращённое refit использует все центры.

    Новый объект на каждый fit. RNG минибатчей отделён от RNG public fit_index.
    Численный guard сравнивает float64 values, не является interval proof.
    """

    def __init__(self, seed: int, *, ema: bool = False) -> None:
        if (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(ema, bool):
            raise TypeError("ema must be boolean")
        self.seed, self.ema, self.calls = int(seed), ema, 0

    def __call__(
        self,
        index_init: np.ndarray,
        U: np.ndarray,
        I: np.ndarray,
        *,
        mass: np.ndarray | None = None,
        lambda_prox: float = 0.05,
        local_ridge: float = 0.0,
        tol: float = 1e-6,
    ) -> HPAOResult:
        B, U, I, mass = g._validate_inputs(index_init, U, I, mass)
        assert mass is not None
        B = _frame(B)
        for name, value in (("lambda_prox", lambda_prox), ("tol", tol)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(local_ridge) or local_ridge < 0:
            raise ValueError("local_ridge must be finite and nonnegative")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.calls]))
        self.calls += 1
        J, m = len(U), len(B)
        batches = min(J, BATCHES)
        updates = EPOCHS * batches
        average = EmaSubspace(B) if self.ema else None
        current = B.copy()
        mini_diagnostics = []
        for _ in range(EPOCHS):
            for idx in np.array_split(rng.permutation(J), batches):
                result = g.solve(
                    current,
                    U[idx],
                    I[idx],
                    mass=mass[idx] * (J / len(idx)),
                    max_steps=1,
                    lambda_prox=lambda_prox,
                    local_ridge=local_ridge,
                    tol=tol,
                )
                current = _frame(result.index)
                mini_diagnostics.append(result.diagnostics)
                if average is not None:
                    average.update(
                        current, cosine_step(len(mini_diagnostics) - 1, updates)
                    )

        proposals = [("input", B), ("last", current)]
        ema_gap = None
        if average is not None:
            averaged, ema_gap = average.proposal()
            if averaged is not None:
                proposals.append(("ema", averaged))
        values: dict[str, float] = {}
        best, best_value, selected = B, float("inf"), "input"
        for name, proposed in proposals:
            state = g._profile(g._project(U, proposed.T), I, mass, local_ridge)
            values[name] = state.value
            if state.value < best_value:
                best, best_value, selected = proposed, state.value, name

        polished = g.solve(
            best,
            U,
            I,
            mass=mass,
            max_steps=FINAL_STEPS,
            lambda_prox=lambda_prox,
            local_ridge=local_ridge,
            tol=tol,
        )
        output = _frame(polished.index)
        state = g._profile(g._project(U, output.T), I, mass, local_ridge)
        rejected = state.value > best_value
        if rejected:
            output = best
            state = g._profile(g._project(U, output.T), I, mass, local_ridge)
        gradient = g._gradient(output.T, U, state, mass)
        gradient_norm = float(np.linalg.norm(gradient))
        if not np.isfinite(gradient_norm):
            raise RuntimeError("non-finite full-profile gradient")
        scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
        converged = bool(state.smooth and gradient_norm <= tol * scale)
        diagnostics: dict[str, object] = {
            "linear_solver": "grassman_minibatch",
            "method": "mini_ema" if self.ema else "mini",
            "loss": 0.5 * float(np.sum(mass[:, None] * state.residual**2)),
            "profile_objective": state.value,
            "profile_history": (values["input"], best_value, state.value),
            "converged": converged,
            "stop_reason": "stationary" if converged else "bounded_full_finalization",
            "certificate_scope": "full_frozen_profile",
            "stationarity_applicable": state.smooth,
            "gradient_norm": gradient_norm,
            "riemannian_gradient": gradient_norm / scale,
            "local_rank_loss": int(np.count_nonzero(state.ranks < m)),
            "orthogonality_error": float(np.linalg.norm(output @ output.T - np.eye(m))),
            "full_candidate_values": values,
            "selected_start": selected,
            "ema_gap": ema_gap,
            "finalization_rejected": rejected,
            "full_finalization": polished.diagnostics,
            "mini_batch_calls": len(mini_diagnostics),
            "mini_batch_accepted_steps": sum(
                int(d["accepted_steps"]) for d in mini_diagnostics
            ),
            "mini_batch_profile_evaluations": sum(
                int(d["profile_evaluations"]) for d in mini_diagnostics
            ),
            "mini_batch_rank_guard_fallbacks": sum(
                int(d["rank_guard_fallbacks"]) for d in mini_diagnostics
            ),
            "nominal_full_center_steps": EPOCHS + FINAL_STEPS,
            "full_guard_profile_evaluations": len(proposals) + 1 + int(rejected),
            "local_ridge": local_ridge,
        }
        return HPAOResult(output.copy(), state.coefficients.copy(), diagnostics)
