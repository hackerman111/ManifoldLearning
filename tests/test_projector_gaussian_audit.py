import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1] / "benchmarks/results/verification/weak-localization-20261028"
    ),
)
import audit_results as audit


def _rows(candidate, *, weak_delta=-0.02, maxsin_delta=-0.02):
    rows = []
    for seed in range(2):
        for case_id, link, m in (("weak", "additive", 5), ("ordinary", "sine", 3)):
            seeded_case_id = f"{case_id}-{seed}"
            base = 0.25 if case_id == "weak" else 0.2
            delta = weak_delta if case_id == "weak" else -0.06
            rows.extend(
                [
                    {
                        "id": f"{case_id}-{seed}-control",
                        "case_id": seeded_case_id,
                        "arm": "control",
                        "seed": seed,
                        "case": {"name": case_id, "link": link, "m": m},
                        "status": "ok",
                        "edr_recalc": base,
                        "max_sin_recalc": 0.5,
                        "fit_seconds": 1.0,
                        "process_seconds": 2.0,
                        "process_peak_rss_mib": 100.0,
                    },
                    {
                        "id": f"{case_id}-{seed}-candidate",
                        "case_id": seeded_case_id,
                        "arm": candidate,
                        "seed": seed,
                        "case": {"name": case_id, "link": link, "m": m},
                        "status": "ok",
                        "edr_recalc": base + delta,
                        "max_sin_recalc": 0.5 + maxsin_delta,
                        "fit_seconds": 1.0,
                        "process_seconds": 2.0,
                        "process_peak_rss_mib": 100.0,
                    },
                ]
            )
    return rows


def test_missing_or_duplicate_fit_rows_block_statistical_gates():
    rows = _rows("candidate")
    jobs = [{"id": r["id"], "case_id": r["case_id"], "arm": r["arm"]} for r in rows]
    missing = audit.compare_arm(
        rows[:-1], jobs, "candidate", "control", bootstrap_draws=200
    )
    duplicate = audit.compare_arm(
        rows + [rows[0]], jobs + [jobs[0]], "candidate", "control", bootstrap_draws=200
    )
    assert not missing["gates_pass"]
    assert not duplicate["gates_pass"]
    assert not missing["gates"]["complete_pairs"]
    assert not duplicate["gates"]["unique_rows"]


def test_weak_case_and_max_angle_are_independent_required_gates():
    rows = _rows("candidate", weak_delta=0.0, maxsin_delta=0.01)
    jobs = [{"id": r["id"], "case_id": r["case_id"], "arm": r["arm"]} for r in rows]
    result = audit.compare_arm(rows, jobs, "candidate", "control", bootstrap_draws=200)
    assert not result["gates"]["weak_additive_m5_ratio"]
    assert not result["gates"]["max_sin_mean_and_upper"]
    assert not result["gates_pass"]


def test_cluster_bootstrap_resamples_whole_seed_blocks_deterministically():
    deltas = {
        (seed, cell): float(seed + cell) for seed in range(3) for cell in range(2)
    }
    first = audit.cluster_bootstrap(
        deltas, (0, 1, 2), (0, 1), draws=500, rng_seed=2841028
    )
    second = audit.cluster_bootstrap(
        deltas, (0, 1, 2), (0, 1), draws=500, rng_seed=2841028
    )
    assert first == second
    assert first["mean"] == 1.5
    assert first["upper_familywise"] >= first["mean"]


def test_global_gates_require_every_arm_and_every_full_moment_certificate():
    arms = ("projected", "full", "two_sided", "compact_projector", "gaussian_projector")
    jobs, rows = [], []
    for seed in range(2):
        for cell in ("additive-m5", "sine-m3"):
            case_id = f"{cell}-{seed}"
            for arm in arms:
                job_id = f"{case_id}-{arm}"
                jobs.append({"id": job_id, "case_id": case_id, "arm": arm})
                rows.append(
                    {"id": job_id, "status": "ok", "numerical_certificates_ok": True}
                )
    assert all(audit.global_quality_gates(rows, jobs).values())
    rows[-1]["status"] = "error"
    assert not audit.global_quality_gates(rows, jobs)["all_planned_fits_successful"]
    rows[-1]["status"] = "ok"
    rows[-1]["numerical_certificates_ok"] = False
    assert not audit.global_quality_gates(rows, jobs)[
        "all_full_moment_solver_certificates"
    ]


def test_frozen_selection_schedule_requires_exact_six_cells_and_five_arms():
    seeds = list(range(2_841_000, 2_841_020))
    inputs, jobs = {}, []
    for seed in seeds:
        for link in ("additive", "quadratic", "sine"):
            for m in (3, 5):
                case_id = f"{link}-m{m}-{seed}"
                inputs[case_id] = {
                    "seed": seed,
                    "case": {
                        "name": case_id,
                        "link": link,
                        "d": 25,
                        "m": m,
                        "n": 5000,
                        "noise": 0.3,
                        "scenario": "standard",
                    },
                }
                for arm in audit.ARMS:
                    jobs.append(
                        {"id": f"{case_id}-{arm}", "case_id": case_id, "arm": arm}
                    )
    manifest = {"phase": "selection", "seeds": seeds, "inputs": inputs, "jobs": jobs}
    errors = []
    audit._validate_schedule(manifest, errors)
    assert not errors
    assert len(jobs) == 600
    audit._validate_schedule({**manifest, "jobs": jobs[:-1]}, errors := [])
    assert any("full product" in error for error in errors)
