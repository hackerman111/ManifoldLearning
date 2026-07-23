from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adp import ADP, ADPConfig, StageRegistry


class DirectBetaSolver:
    """Плотное решение beta для небольших контрольных экспериментов."""

    def __init__(self, config: ADPConfig) -> None:
        self.config = config

    def solve(
        self,
        statistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        del x0
        if statistics.S is None or statistics.U is None:
            raise ValueError("direct solver требует статистики S и U")
        d = statistics.U.shape[2]
        residual = statistics.imav - intercepts[:, None] * statistics.S
        u_flat = np.asarray(statistics.U).reshape(-1, d)
        slope_flat = np.repeat(
            np.asarray(slopes, dtype=float), statistics.U.shape[1]
        )
        weighted_u = slope_flat[:, None] * u_flat
        regularization = float(lambda_penalty) + float(self.config.ridge)
        matrix = weighted_u.T @ weighted_u
        matrix += regularization * np.eye(d)
        rhs = weighted_u.T @ residual.reshape(-1)
        rhs += float(lambda_penalty) * prior
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def run_comparison(
    *,
    n: int = 120,
    d: int = 8,
    n_centers: int = 24,
    n_directions: int = 6,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Сравнивает два beta solver на полностью одинаковых входах."""

    config = ADPConfig(
        n_centers=n_centers,
        n_directions=n_directions,
        min_neighbors=min(8, max(3, n // 10)),
        outer_steps=1,
        inner_steps=4,
        show_progress=False,
        random_state=seed,
    )
    generator = ADP.create("new", config)
    data = generator.generate_data(
        n=n,
        d=d,
        n_centers=n_centers,
        n_directions=n_directions,
        noise=0.03,
        link="linear",
    )

    registry = StageRegistry.with_defaults()
    registry.register(
        "beta_solver",
        "direct",
        lambda context: DirectBetaSolver(context.config),
    )
    beta0 = np.random.default_rng(seed + 1).normal(size=d)
    beta0 /= np.linalg.norm(beta0)

    rows: list[dict[str, Any]] = []
    for solver_name in ("cg", "direct"):
        model = ADP.create(
            "new",
            config,
            stages={"beta_solver": solver_name},
            registry=registry,
        )
        result = model.fit(
            data.X,
            data.y,
            centers=data.centers,
            beta0=beta0,
            directions=data.directions,
        )
        rows.append(
            {
                "solver": solver_name,
                "cosine_abs": model.score(data.beta)["cosine_abs"],
                "objective": float(result.objective),
                "beta_solver_time_sec": result.stage_timings["beta_solver"],
                "beta_solver_calls": result.stage_calls["beta_solver"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение beta solver ADP на одинаковых входных данных"
    )
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--n-centers", type=int, default=24)
    parser.add_argument("--n-directions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = run_comparison(
        n=args.n,
        d=args.d,
        n_centers=args.n_centers,
        n_directions=args.n_directions,
        seed=args.seed,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
