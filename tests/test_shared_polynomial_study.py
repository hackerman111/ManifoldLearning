"""Research gates must fail closed on missing, failed or mismatched fits."""

import importlib.util


def module():
    assert importlib.util.find_spec("experiments.shared_polynomial_study") is not None
    from experiments import shared_polynomial_study

    return shared_polynomial_study


def fixture(lib):
    jobs, rows = [], []
    for link, m in lib.CASES:
        for seed in range(10):
            for arm in lib.ARMS:
                jobs.append([link, m, seed, arm])
                rows.append(
                    {
                        "link": link,
                        "m": m,
                        "seed": seed,
                        "arm": arm,
                        "status": "ok",
                        "edr": 0.05 if arm == "cubic" else 0.4,
                        "fit_seconds": 1.5 if arm == "cubic" else 1.0,
                        "process_seconds": 2.0,
                        "peak_rss_mib": 100.0,
                        "checked": True,
                        "data_sha256": f"data-{link}-{m}-{seed}",
                        "initial_sha256": "local",
                        "basis_sha256": "refined" if arm == "cubic" else "baseline",
                        "refinement_start_sha256": "baseline",
                        "method_seed": seed,
                        "diagnostics": {
                            "converged": True,
                            "normal_residual_max": 1e-14,
                            "gradient_rank_ok": True,
                            "tangent_rank_ok": True,
                            "history": [1.0, 0.8, 0.4],
                        },
                    }
                )
    return jobs, rows


def test_complete_improvement_passes_all_frozen_gates():
    lib = module()
    jobs, rows = fixture(lib)
    summary = lib.summarize(rows, jobs)
    assert summary["passed"]
    assert summary["comparisons"]["two_sided"]["mean_delta_edr"] < -0.3


def test_missing_failed_or_duplicate_outcome_cannot_pass():
    lib = module()
    for mutation in ("missing", "failed", "duplicate"):
        jobs, rows = fixture(lib)
        if mutation == "missing":
            rows.pop()
        elif mutation == "failed":
            rows[-1]["status"] = "timeout"
        else:
            rows.append(rows[-1])
        assert not lib.summarize(rows, jobs)["passed"]


def test_initializer_mismatch_or_increasing_profile_cannot_pass():
    lib = module()
    for mutation in ("initialization", "history", "stationarity", "rank"):
        jobs, rows = fixture(lib)
        candidates = [row for row in rows if row["arm"] == "cubic"]
        if mutation == "initialization":
            candidates[0]["refinement_start_sha256"] = "unpaired"
        elif mutation == "history":
            candidates[0]["diagnostics"]["history"] = [1.0, 1.1]
        elif mutation == "stationarity":
            for row in candidates[:7]:
                row["diagnostics"]["converged"] = False
        else:
            candidates[0]["diagnostics"]["tangent_rank_ok"] = False
        assert not lib.summarize(rows, jobs)["passed"]


def test_single_large_regression_is_not_hidden_by_average():
    lib = module()
    jobs, rows = fixture(lib)
    next(row for row in rows if row["arm"] == "cubic")["edr"] = 0.5
    summary = lib.summarize(rows, jobs)
    assert summary["comparisons"]["two_sided"]["mean_delta_edr"] < -0.3
    assert not summary["passed"]


def test_descriptive_controls_must_also_share_their_initialization():
    lib = module()
    for arm in ("projected", "cubic_euclidean"):
        jobs, rows = fixture(lib)
        target = next(row for row in rows if row["arm"] == arm)
        if arm == "projected":
            target["initial_sha256"] = "wrong"
        else:
            target["refinement_start_sha256"] = "wrong"
        assert not lib.summarize(rows, jobs)["passed"]
