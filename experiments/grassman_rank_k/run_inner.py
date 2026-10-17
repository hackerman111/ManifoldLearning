"""Serial frozen-input stage; numerical worker implementations remain separate."""

from __future__ import annotations

from .common import (
    OUT,
    _assert_frozen,
    _fresh_worker,
    _read_row,
    _run_child,
    jsonl_append,
)


def inner_runs() -> None:
    _assert_frozen()
    output = OUT / "results" / "inner.jsonl"
    if output.exists():
        raise FileExistsError(output)
    for m in (2, 3, 5, 8):
        for d in (20, 100, 300):
            r = min(m, d - m)
            for seed in (81600, 81700):
                snapshot = OUT / "captures" / f"inner_m{m}_d{d}_seed{seed}_J64.npz"
                arms: list[tuple[str, int | None, str, int, float]] = [
                    ("core", None, "optimized", 1, 1e-7),
                    ("rank_one", None, "optimized", 1, 1e-7),
                    ("spectral", None, "optimized", 1, 1e-7),
                    ("flat", None, "optimized", 1, 1e-7),
                ]
                arms.extend(
                    ("block_fixed", k, variant, 1, 1e-7)
                    for k in range(1, r + 1)
                    for variant in ("optimized", "spectral")
                )
                arms.extend(
                    (arm, None, variant, 1, 1e-7)
                    for arm, variant in (
                        ("block_energy", "optimized"),
                        ("block_cost", "optimized"),
                        ("block_spectral", "spectral"),
                    )
                )
                arms.extend(
                    (arm, rank_k, variant, 100, 1e-7)
                    for arm, rank_k, variant in (
                        ("core", None, "optimized"),
                        ("rank_one", None, "optimized"),
                        ("spectral", None, "optimized"),
                        ("flat", None, "optimized"),
                        ("block1", 1, "optimized"),
                        ("block_full", m, "optimized"),
                        ("block_energy", None, "optimized"),
                        ("block_cost", None, "optimized"),
                        ("block_spectral", None, "spectral"),
                    )
                )
                arms.extend(
                    ("rank_one", None, "optimized", k, 1e-7) for k in range(2, r + 1)
                )
                for arm, k, variant, budget, tol in arms:
                    key = f"{m}_{d}_{seed}_{arm}_{k}_{variant}_{budget}"
                    worker = OUT / "results" / "workers" / f"inner_{key}.json"
                    worker.parent.mkdir(parents=True, exist_ok=True)
                    _fresh_worker(worker)
                    status = _run_child(
                        "experiments.grassman_rank_k.inner",
                        [
                            "--snapshot",
                            str(snapshot),
                            "--arm",
                            arm,
                            "--budget",
                            str(budget),
                            "--tol",
                            str(tol),
                            "--variant",
                            variant,
                            "--output",
                            str(worker),
                        ]
                        + (["--rank-k", str(k)] if k is not None else []),
                        60,
                    )
                    row = (
                        _read_row(worker)
                        if worker.exists()
                        else {"status": status["status"]}
                    )
                    row.update(
                        {
                            "scenario": "inner",
                            "m": m,
                            "d": d,
                            "seed": seed,
                            "arm": arm,
                            "rank_k": k,
                            "variant": variant,
                            "budget": budget,
                            "comparison_kind": (
                                "sequential_rank_one_budget"
                                if arm == "rank_one" and 1 < budget < 100
                                else "fixed_rank_one_step"
                                if arm == "block_fixed"
                                else "convergence_path"
                                if budget > 1
                                else "adaptive_one_step"
                            ),
                            "worker_status": status,
                        }
                    )
                    jsonl_append(output, row)
                refkey = f"{m}_{d}_{seed}_reference"
                worker = OUT / "results" / "workers" / f"reference_{refkey}.json"
                worker.parent.mkdir(parents=True, exist_ok=True)
                _fresh_worker(worker)
                status = _run_child(
                    "experiments.grassman_rank_k.inner",
                    [
                        "--snapshot",
                        str(snapshot),
                        "--arm",
                        "reference",
                        "--budget",
                        "1000",
                        "--tol",
                        "1e-9",
                        "--reference",
                        "--output",
                        str(worker),
                    ],
                    180,
                )
                row = (
                    _read_row(worker)
                    if worker.exists()
                    else {"status": status["status"]}
                )
                row.update(
                    {
                        "scenario": "inner",
                        "m": m,
                        "d": d,
                        "seed": seed,
                        "arm": "reference",
                        "comparison_kind": "reference",
                        "worker_status": status,
                    }
                )
                row.setdefault("status", "ok")
                jsonl_append(output, row)
                if m == 5 and d == 100:
                    for J in (16, 256):
                        cost_snapshot = (
                            OUT / "captures" / f"inner_m5_d100_seed{seed}_J{J}.npz"
                        )
                        for arm, variant in (
                            ("block_energy", "optimized"),
                            ("block_cost", "optimized"),
                            ("block_spectral", "spectral"),
                        ):
                            for budget in (1, 100):
                                key = f"m5_d100_J{J}_{seed}_{arm}_{variant}_{budget}"
                                worker = (
                                    OUT / "results" / "workers" / f"inner_{key}.json"
                                )
                                _fresh_worker(worker)
                                status = _run_child(
                                    "experiments.grassman_rank_k.inner",
                                    [
                                        "--snapshot",
                                        str(cost_snapshot),
                                        "--arm",
                                        arm,
                                        "--budget",
                                        str(budget),
                                        "--tol",
                                        "1e-7",
                                        "--variant",
                                        variant,
                                        "--output",
                                        str(worker),
                                    ],
                                    60,
                                )
                                row = (
                                    _read_row(worker)
                                    if worker.exists()
                                    else {"status": status["status"]}
                                )
                                row.update(
                                    {
                                        "scenario": "inner_cost_grid",
                                        "m": 5,
                                        "d": 100,
                                        "J": J,
                                        "seed": seed,
                                        "arm": arm,
                                        "variant": variant,
                                        "budget": budget,
                                        "comparison_kind": "cost_grid_adaptive_budget",
                                        "worker_status": status,
                                    }
                                )
                                jsonl_append(output, row)
                        for k in range(1, 6):
                            for variant in ("optimized", "spectral"):
                                key = f"m5_d100_J{J}_{seed}_fixed{k}_{variant}"
                                worker = (
                                    OUT / "results" / "workers" / f"inner_{key}.json"
                                )
                                _fresh_worker(worker)
                                status = _run_child(
                                    "experiments.grassman_rank_k.inner",
                                    [
                                        "--snapshot",
                                        str(cost_snapshot),
                                        "--arm",
                                        "block_fixed",
                                        "--rank-k",
                                        str(k),
                                        "--budget",
                                        "1",
                                        "--tol",
                                        "1e-7",
                                        "--variant",
                                        variant,
                                        "--output",
                                        str(worker),
                                    ],
                                    60,
                                )
                                row = (
                                    _read_row(worker)
                                    if worker.exists()
                                    else {"status": status["status"]}
                                )
                                row.update(
                                    {
                                        "scenario": "inner_cost_grid",
                                        "m": 5,
                                        "d": 100,
                                        "J": J,
                                        "seed": seed,
                                        "arm": "block_fixed",
                                        "rank_k": k,
                                        "variant": variant,
                                        "budget": 1,
                                        "comparison_kind": "fixed_rank_one_step",
                                        "worker_status": status,
                                    }
                                )
                                jsonl_append(output, row)
                            key = f"m5_d100_J{J}_{seed}_rankone{k}"
                            worker = OUT / "results" / "workers" / f"inner_{key}.json"
                            _fresh_worker(worker)
                            status = _run_child(
                                "experiments.grassman_rank_k.inner",
                                [
                                    "--snapshot",
                                    str(cost_snapshot),
                                    "--arm",
                                    "rank_one",
                                    "--budget",
                                    str(k),
                                    "--tol",
                                    "1e-7",
                                    "--output",
                                    str(worker),
                                ],
                                60,
                            )
                            row = (
                                _read_row(worker)
                                if worker.exists()
                                else {"status": status["status"]}
                            )
                            row.update(
                                {
                                    "scenario": "inner_cost_grid",
                                    "m": 5,
                                    "d": 100,
                                    "J": J,
                                    "seed": seed,
                                    "arm": "rank_one",
                                    "rank_k": k,
                                    "variant": "optimized",
                                    "budget": k,
                                    "comparison_kind": "sequential_rank_one_budget",
                                    "worker_status": status,
                                }
                            )
                            jsonl_append(output, row)
