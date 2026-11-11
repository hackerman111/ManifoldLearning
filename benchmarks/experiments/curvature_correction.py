"""Opt-in frozen-curvature moment correction; no default/API changes."""

from dataclasses import replace

import numpy as np

from experiments import local_polynomial as poly


def correct(basis, statistics, data, directions, *, ridge=1e-4):
    """Subtract the fitted local quadratic contribution from normalized I.

    This is a lagged correction at the incoming basis, not minimization of
    the full quadratic profile. Exact quadratic consistency needs ridge=0
    and identifiable local quadratic coefficients.
    """
    state = poly.profile(basis, data, degree=2, ridge=ridge)
    corrected_I = statistics.I.copy()
    for j, section in enumerate(data.local):
        z = section.delta @ basis
        quadratic = 0.5 * np.sum((z @ state.hessians[j]) * z, axis=1)
        quadratic -= section.weights @ quadratic
        covariance = section.delta.T @ (section.weights * quadratic)
        corrected_I[j] -= directions[j] @ covariance
    return replace(statistics, I=corrected_I)
