"""Independent scalar-response identifiability reference for curved charts."""

import numpy as np
import pytest


@pytest.mark.parametrize("m", (2, 3))
def test_scalar_response_does_not_identify_curved_rank_m_chart(m: int) -> None:
    """An input-dependent rotation preserves every Y but changes row(Dz)."""
    point = np.zeros(m + 1)
    point[0] = 1.0

    def maps(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        original = x[:m].copy()
        angle = x[-1]
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated = original.copy()
        rotated[0] = cosine * original[0] - sine * original[1]
        rotated[1] = sine * original[0] + cosine * original[1]
        return original, rotated

    rng = np.random.default_rng(92028)
    for x in rng.normal(size=(20, m + 1)):
        original, rotated = maps(x)
        np.testing.assert_allclose(
            original @ original, rotated @ rotated, rtol=1e-14, atol=1e-14
        )

    step = 1e-6
    directions = np.eye(m + 1) * step
    jacobians = []
    for component in (0, 1):
        columns = [
            (maps(point + direction)[component] - maps(point - direction)[component])
            / (2 * step)
            for direction in directions
        ]
        jacobians.append(np.column_stack(columns))
    first, second = jacobians
    first_basis, _ = np.linalg.qr(first.T)
    second_basis, _ = np.linalg.qr(second.T)
    smallest_cosine = np.linalg.svd(first_basis.T @ second_basis)[1][-1]
    assert np.sqrt(1.0 - smallest_cosine**2) == pytest.approx(2**-0.5, abs=1e-9)


def test_curved_rank_one_chart_is_gradient_direction_off_stationary_set() -> None:
    x = np.array([0.7, -0.3])
    curvature = 0.8
    z = x[0] + 0.5 * curvature * x[1] ** 2
    chart = np.array([1.0, curvature * x[1]])
    gradient = z * chart
    cosine = abs(np.dot(gradient, chart)) / (
        np.linalg.norm(gradient) * np.linalg.norm(chart)
    )
    assert cosine == pytest.approx(1.0)
