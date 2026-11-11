from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from experiments.grassman_accuracy_2026_10_22.lowrank_analyze import geo, main


def test_geometric_mean_handles_exact_zero_error() -> None:
    assert geo([0.0, 0.5]) == 0.0


def _pilot(tmp_path: Path, *, bad_orthogonality=False, failed=False, extra_seed=False):
    source = b"frozen test source"
    digest = hashlib.sha256(source).hexdigest()
    cases = ["case-a", "case-b"]
    methods = ["baseline", "budget10", "projected10", "MEANMAVE"]
    seeds = [111800, 111801]
    protocol = {
        "cases": [{"name": case} for case in cases],
        "methods": methods,
        "seed": seeds[0],
        "runs": len(seeds),
        "sources": {
            "fake.py": digest,
            "ADP/solver/grassman.py": digest,
            "benchmarks/experiments/grassman_accuracy_2026_10_22/reference.py": digest,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"protocol": protocol}))
    with zipfile.ZipFile(tmp_path / "sources.zip", "w") as archive:
        archive.writestr("fake.py", source)
        archive.writestr("ADP/solver/grassman.py", source)
        archive.writestr(
            "benchmarks/experiments/grassman_accuracy_2026_10_22/reference.py", source
        )
    rows = []
    for case_i, case in enumerate(cases):
        for seed in seeds:
            for method_i, method in enumerate(methods):
                value = 0.08 + 0.01 * case_i + 0.001 * (seed - seeds[0])
                if method == "projected10":
                    value *= 0.90
                if method == "budget10":
                    value *= 0.95
                row = {
                    "case": case,
                    "seed": seed,
                    "method": method,
                    "method_seed": seed + 1000,
                    "input_hashes": {
                        k: f"{case}-{seed}-{k}" for k in ("x", "y", "truth")
                    },
                    "status": "error"
                    if failed and method == "projected10" and seed == seeds[0]
                    else "ok",
                    "projector_distance": value,
                    "fit_seconds": 2.0 + method_i * 0.1,
                    "peak_rss_mib": 100.0 + method_i,
                    "orthogonality_error": float("nan")
                    if bad_orthogonality and method == "projected10"
                    else 1e-14,
                    "stop_reason": "h_min",
                }
                rows.append(row)
    if extra_seed:
        rows.append({**rows[0], "case": "case-a", "seed": 123456, "method": "baseline"})
    (tmp_path / "runs.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return tmp_path


def test_analysis_checks_full_pairs_gates_and_budget_ablation(tmp_path: Path) -> None:
    result = main(_pilot(tmp_path))
    projected = result["methods"]["projected10"]
    comparison = result["comparisons"]["projected10_vs_budget10"]
    assert result["rows"] == result["planned_rows"] == 16
    assert projected["gate_complete"] and projected["all_gates_pass"]
    assert projected["complete_seed_blocks"] == 2
    assert len(projected["EDR_geomean_base_bootstrap95"]) == 2
    assert comparison["gate_complete"]
    assert comparison["valid_pairs"] == 4
    assert len(comparison["pairs"]) == 4


def test_nonfinite_orthogonality_fails_gate_without_crashing(tmp_path: Path) -> None:
    result = main(_pilot(tmp_path, bad_orthogonality=True))
    projected = result["methods"]["projected10"]
    assert projected["valid_pairs"] == 0
    assert not projected["gate_complete"]
    assert not projected["gates"]["orthogonality_le_1e_10"]


def test_failed_candidate_row_is_retained_and_fails_gate(tmp_path: Path) -> None:
    result = main(_pilot(tmp_path, failed=True))
    projected = result["methods"]["projected10"]
    assert result["failures"][0]["method"] == "projected10"
    assert projected["valid_pairs"] < projected["planned_pairs"]
    assert not projected["gate_complete"]


def test_foreign_seed_row_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate or foreign result"):
        main(_pilot(tmp_path, extra_seed=True))
