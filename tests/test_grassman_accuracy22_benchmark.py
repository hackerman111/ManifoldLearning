from __future__ import annotations

from types import FunctionType
from typing import cast

from experiments.grassman_accuracy_2026_10_22.benchmark import (
    METHODS,
    _catalog_cases,
    _summaries,
)
from experiments.grassman_accuracy_2026_10_22.candidates import solver_for
from experiments.r_sdr.design import Case


def test_gn_quadratic_is_an_explicit_benchmark_method() -> None:
    assert "gn_quadratic" in METHODS


def test_positive_local_ridge_variants_are_explicit_methods() -> None:
    assert {"ridge6", "ridge4", "ridge2"}.issubset(METHODS)


def test_krylov_budgets_are_explicit_benchmark_methods() -> None:
    assert {"krylov10", "krylov20"}.issubset(METHODS)


def test_lora_svd_and_lead_candidates_are_registered() -> None:
    assert {
        "lora1_10",
        "lora_k10",
        "lora_k20",
        "svd1_10",
        "svd_k10",
        "svd_k20",
        "enriched10",
        "enriched20",
        "svd_greedy10",
    }.issubset(METHODS)


def test_projected_candidates_are_explicit_benchmark_methods() -> None:
    assert {"projected10", "projected20"}.issubset(METHODS)


def test_budget10_changes_only_the_iteration_budget() -> None:
    baseline = solver_for("baseline")
    budget10 = solver_for("budget10")
    assert "budget10" in METHODS
    baseline_fn = cast(FunctionType, baseline)
    budget_fn = cast(FunctionType, budget10.func)
    assert budget_fn.__code__.co_code == baseline_fn.__code__.co_code
    assert budget_fn.__code__.co_consts == baseline_fn.__code__.co_consts
    assert budget_fn.__defaults__ == baseline_fn.__defaults__
    assert budget10.keywords == {"max_steps": 10}


def test_multi_rank_cases_join_full_catalog_without_changing_existing_cases() -> None:
    cases = _catalog_cases()
    assert cases["d-20"] == Case("d-20", 500, 20, 2)
    assert [
        (cases[f"rank-{m}"].n, cases[f"rank-{m}"].d, cases[f"rank-{m}"].m)
        for m in (3, 4, 5)
    ] == [(500, 20, 3), (500, 20, 4), (500, 20, 5)]


def test_analysis_keeps_failed_candidate_in_planned_pair_denominator() -> None:
    case = Case("tiny", 80, 5, 1)
    rows = [
        {"case": "tiny", "seed": 10, "method": "baseline", "status": "ok"},
        {
            "case": "tiny",
            "seed": 10,
            "method": "MEANMAVE",
            "status": "ok",
            "projector_distance": 0.2,
        },
        {
            "case": "tiny",
            "seed": 10,
            "method": "diagonal",
            "status": "error",
            "error": "failed",
        },
    ]
    result = _summaries(rows, [case], ["diagonal", "MEANMAVE"], [10, 11])
    pair = result["methods"]["diagonal"]["tiny"]
    assert result["planned_rows"] == 4
    assert result["rows_recorded"] == 3
    assert pair["planned_pairs"] == 2
    assert pair["valid_pairs"] == 0
    assert pair["candidate_failures"] == 2
    assert pair["edr_ratio_geomean"] is None


def test_analysis_pairs_same_seed_and_reports_quality_and_time_ratios() -> None:
    case = Case("tiny", 80, 5, 1)
    rows = [
        {
            "case": "tiny",
            "seed": 10,
            "method": "stable_jacobian",
            "status": "ok",
            "projector_distance": 0.1,
            "fit_seconds": 2.0,
        },
        {
            "case": "tiny",
            "seed": 10,
            "method": "MEANMAVE",
            "status": "ok",
            "projector_distance": 0.2,
            "fit_seconds": 4.0,
        },
        {
            "case": "tiny",
            "seed": 11,
            "method": "stable_jacobian",
            "status": "ok",
            "projector_distance": 0.9,
            "fit_seconds": 8.0,
        },
    ]
    pair = _summaries(rows, [case], ["stable_jacobian", "MEANMAVE"], [10, 11])[
        "methods"
    ]["stable_jacobian"]["tiny"]
    assert pair["valid_pairs"] == 1
    assert pair["edr_ratio_geomean"] == 0.5
    assert pair["fit_time_ratio_median"] == 0.5
    assert pair["fit_time_ratio_p90"] == 0.5
    assert pair["meanmave_failures"] == 1
