from __future__ import annotations

import math

from ADP.cli.experiment import CATALOG, _make_seed_bundle, _selected_experiments


def test_parameter_scaling_catalog_is_paired_and_valid() -> None:
    selectors = tuple(
        experiment.selector
        for experiment in _selected_experiments(
            "parameter-scaling",
            custom=CATALOG["custom"],
        )
    )
    assert selectors == (
        "scale-si-nphi",
        "scale-si-nloc",
        "scale-si-nj",
        "scale-mi-nphi",
        "scale-mi-nloc",
        "scale-mi-nj",
    )

    expected_sizes = {"nphi": 54, "nloc": 54, "nj": 45}
    for selector in selectors:
        experiment = CATALOG[selector]
        parameter = experiment.condition_field
        assert parameter is not None
        assert len(experiment.full) == expected_sizes[selector.rsplit("-", 1)[-1]]
        assert experiment.full_runs == 5
        assert experiment.common_random_fields == (parameter,)
        assert {(point.n, point.d) for point in experiment.full} == {
            (n, d) for n in (500, 1000, 2000) for d in (10, 25, 50)
        }
        assert all(
            point.N_J is not None
            and point.N_loc is not None
            and point.N_phi is not None
            and math.ceil(point.n / point.N_loc) <= point.N_J <= point.n
            and (point.mode == "single" or point.N_phi > point.index_dim)
            for point in experiment.full
        )

        grouped: dict[tuple[int, int], list[object]] = {}
        for point in experiment.full:
            grouped.setdefault((point.n, point.d), []).append(
                _make_seed_bundle(
                    selector,
                    point,
                    7,
                    common_random_fields=experiment.common_random_fields,
                )
            )
        assert all(len(set(seeds)) == 1 for seeds in grouped.values())


def test_detailed_nd_catalog_refines_mi_boundary_and_pairs_si_nloc() -> None:
    experiments = _selected_experiments("nd-detail", custom=CATALOG["custom"])
    assert tuple(experiment.selector for experiment in experiments) == (
        "mi-boundary-nd",
        "si-nloc-nd",
    )

    multi, single = experiments
    assert len(multi.full) == 63
    assert multi.full_runs == 10
    assert multi.quality_threshold == 0.95
    assert {(point.n, point.d) for point in multi.full} == {
        (n, d)
        for n in (400, 600, 800, 1000, 1400, 1800, 2400)
        for d in (10, 15, 20, 25, 30, 35, 40, 50, 60)
    }
    assert all(point.mode == "multi" and point.index_dim == 2 for point in multi.full)

    assert len(single.full) == 210
    assert single.full_runs == 10
    assert single.quality_threshold == 0.9
    assert single.condition_field == "N_loc"
    assert single.common_random_fields == ("N_loc",)
    assert {point.N_loc for point in single.full} == {10, 15, 20, 25, 30, 40, 50}
    assert {(point.n, point.d) for point in single.full} == {
        (n, d) for n in (400, 700, 1000, 1500, 2200) for d in (10, 20, 30, 40, 50, 60)
    }

    grouped: dict[tuple[int, int], list[object]] = {}
    for point in single.full:
        grouped.setdefault((point.n, point.d), []).append(
            _make_seed_bundle(
                single.selector,
                point,
                7,
                common_random_fields=single.common_random_fields,
            )
        )
    assert all(len(set(seeds)) == 1 for seeds in grouped.values())
