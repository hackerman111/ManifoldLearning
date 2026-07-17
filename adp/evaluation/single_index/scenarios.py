from __future__ import annotations

from itertools import product

from .types import EXPERIMENT_SELECTORS, ExperimentParameters


EXPERIMENT_COUNTS = {
    "1": 800,
    "2": 2_000,
    "3": 4_200,
    "4": 3_600,
    "5": 3_000,
    "6": 3_600,
    "7.1": 1_200,
    "7.2": 1_200,
    "8.1": 800,
    "8.2": 2_000,
    "8.3": 1_600,
}
PROFILE_IDS = {
    "smoke": EXPERIMENT_SELECTORS,
    "full": EXPERIMENT_SELECTORS,
}

_STANDARD_DIMENSIONS = (25, 100)
_STANDARD_RATIOS = (2.0, 5.0, 10.0)
_DISTRIBUTION_RATIOS = (2.0, 5.0)


def parse_experiment_selectors(value: str) -> tuple[str, ...]:
    """Parse selectors, returning unique values in canonical specification order."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("experiment selectors must not be empty")
    requested = tuple(part.strip() for part in value.split(","))
    if any(not selector for selector in requested):
        raise ValueError("experiment selectors must not be empty")
    if requested == ("all",):
        return EXPERIMENT_SELECTORS
    unknown = sorted(set(requested) - set(EXPERIMENT_SELECTORS))
    if unknown:
        raise ValueError(f"unknown experiment selector: {', '.join(unknown)}")
    selected = set(requested)
    return tuple(
        selector for selector in EXPERIMENT_SELECTORS if selector in selected
    )


def parse_seed_selection(value: str) -> tuple[int, ...]:
    """Parse an inclusive range or comma list into sorted unique seeds."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("seed selection must not be empty")
    selection = value.strip()
    if ":" in selection:
        if "," in selection or selection.count(":") != 1:
            raise ValueError("seed range must have the form START:STOP")
        start_text, stop_text = (part.strip() for part in selection.split(":"))
        try:
            start = int(start_text)
            stop = int(stop_text)
        except ValueError as exc:
            raise ValueError("seed range must contain integers") from exc
        _validate_seed(start)
        _validate_seed(stop)
        if start > stop:
            raise ValueError("seed range start must not exceed stop")
        return tuple(range(start, stop + 1))

    parts = tuple(part.strip() for part in selection.split(","))
    if any(not part for part in parts):
        raise ValueError("seed selection must contain integers")
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("seed selection must contain integers") from exc
    for seed in seeds:
        _validate_seed(seed)
    return tuple(sorted(set(seeds)))


def full_parameter_grid(selector: str) -> tuple[ExperimentParameters, ...]:
    """Return the literal independent full-profile grid for one experiment."""

    _validate_selector(selector)
    if selector == "1":
        return tuple(
            ExperimentParameters(
                d=d,
                n_over_d=n_over_d,
                link=link,
                sigma_eps=0.0,
            )
            for d, n_over_d, link in product(
                (5, 25),
                (5.0, 10.0),
                ("linear", "quadratic"),
            )
        )
    if selector == "2":
        return tuple(
            ExperimentParameters(d=d, n_over_d=n_over_d)
            for d, n_over_d in product(
                (5, 25, 50, 100),
                (1.0, 1.15, 2.0, 5.0, 10.0),
            )
        )
    if selector == "3":
        return tuple(
            ExperimentParameters(
                d=d,
                n_over_d=n_over_d,
                sigma_eps=sigma_eps,
            )
            for d, n_over_d, sigma_eps in product(
                _STANDARD_DIMENSIONS,
                _STANDARD_RATIOS,
                (0.0, 0.316, 0.5, 0.707, 1.0, 1.414, 2.0),
            )
        )
    if selector == "4":
        return tuple(
            ExperimentParameters(
                d=d,
                n_over_d=n_over_d,
                rho_corr=rho_corr,
            )
            for d, n_over_d, rho_corr in product(
                _STANDARD_DIMENSIONS,
                _STANDARD_RATIOS,
                (0.0, 0.25, 0.5, 0.75, 0.9, 0.95),
            )
        )
    if selector == "5":
        return tuple(
            ExperimentParameters(
                d=d,
                n_over_d=n_over_d,
                sigma_x=sigma_x,
            )
            for d, n_over_d, sigma_x in product(
                _STANDARD_DIMENSIONS,
                _STANDARD_RATIOS,
                (0.25, 0.5, 1.0, 2.0, 4.0),
            )
        )
    if selector == "6":
        return tuple(
            ExperimentParameters(d=d, n_over_d=n_over_d, link=link)
            for d, n_over_d, link in product(
                _STANDARD_DIMENSIONS,
                _STANDARD_RATIOS,
                (
                    "linear",
                    "quadratic",
                    "square",
                    "sin",
                    "tanh",
                    "oscillating",
                ),
            )
        )
    if selector == "7.1":
        return tuple(
            ExperimentParameters(
                d=d,
                n_over_d=n_over_d,
                x_distribution=distribution,
            )
            for d, n_over_d, distribution in product(
                _STANDARD_DIMENSIONS,
                _DISTRIBUTION_RATIOS,
                ("gaussian", "uniform", "student_t5"),
            )
        )
    if selector == "7.2":
        return tuple(
            ExperimentParameters(
                d=d,
                n_over_d=n_over_d,
                noise_distribution=distribution,
            )
            for d, n_over_d, distribution in product(
                _STANDARD_DIMENSIONS,
                _DISTRIBUTION_RATIOS,
                ("gaussian", "student_t5", "student_t3"),
            )
        )
    if selector == "8.1":
        return tuple(
            ExperimentParameters(
                d=d,
                n_over_d=n_over_d,
                heteroscedastic=heteroscedastic,
            )
            for d, n_over_d, heteroscedastic in product(
                _STANDARD_DIMENSIONS,
                _DISTRIBUTION_RATIOS,
                (False, True),
            )
        )
    if selector == "8.2":
        return tuple(
            ExperimentParameters(
                d=d,
                n_over_d=n_over_d,
                outlier_fraction=outlier_fraction,
                outlier_scale=outlier_scale,
            )
            for d, n_over_d, (outlier_fraction, outlier_scale) in product(
                _STANDARD_DIMENSIONS,
                _DISTRIBUTION_RATIOS,
                (
                    (0.0, 1.0),
                    (0.01, 5.0),
                    (0.01, 10.0),
                    (0.05, 5.0),
                    (0.05, 10.0),
                ),
            )
        )
    return tuple(
        ExperimentParameters(d=d, n_over_d=n_over_d, delta=delta)
        for d, n_over_d, delta in product(
            _STANDARD_DIMENSIONS,
            _DISTRIBUTION_RATIOS,
            (0.0, 0.1, 0.25, 0.5),
        )
    )


def smoke_parameter_grid(selector: str) -> tuple[ExperimentParameters, ...]:
    """Return one tiny representative parameter set for a smoke experiment."""

    _validate_selector(selector)
    representative = {
        "1": ExperimentParameters(
            d=4,
            n_over_d=5.0,
            link="linear",
            sigma_eps=0.0,
        ),
        "2": ExperimentParameters(d=4, n_over_d=2.0),
        "3": ExperimentParameters(d=4, n_over_d=5.0, sigma_eps=1.0),
        "4": ExperimentParameters(d=4, n_over_d=5.0, rho_corr=0.5),
        "5": ExperimentParameters(d=4, n_over_d=5.0, sigma_x=2.0),
        "6": ExperimentParameters(d=4, n_over_d=5.0, link="sin"),
        "7.1": ExperimentParameters(
            d=4,
            n_over_d=5.0,
            x_distribution="uniform",
        ),
        "7.2": ExperimentParameters(
            d=4,
            n_over_d=5.0,
            noise_distribution="student_t5",
        ),
        "8.1": ExperimentParameters(
            d=4,
            n_over_d=5.0,
            heteroscedastic=True,
        ),
        "8.2": ExperimentParameters(
            d=4,
            n_over_d=5.0,
            outlier_fraction=0.01,
            outlier_scale=5.0,
        ),
        "8.3": ExperimentParameters(d=4, n_over_d=5.0, delta=0.1),
    }
    return (representative[selector],)


def _validate_selector(selector: str) -> None:
    if selector not in EXPERIMENT_SELECTORS:
        raise ValueError(f"unknown experiment selector: {selector}")


def _validate_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("seeds must be nonnegative")


__all__ = [
    "EXPERIMENT_COUNTS",
    "EXPERIMENT_SELECTORS",
    "PROFILE_IDS",
    "full_parameter_grid",
    "parse_experiment_selectors",
    "parse_seed_selection",
    "smoke_parameter_grid",
]
