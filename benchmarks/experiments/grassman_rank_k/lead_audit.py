"""Independent structural audit of saved experiment evidence (no solver runs)."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import OUT, ROOT, array_hash, source_hashes


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(final: bool = False) -> dict[str, Any]:
    manifest = json.loads((OUT / "freeze_manifest.json").read_text())
    assert source_hashes() == manifest["source_sha256"], "numerical source drift"
    counter_path = OUT / "results" / "objective_counter_manifest.json"
    counter_hash = None
    if counter_path.exists():
        counter_manifest = json.loads(counter_path.read_text())
        counter_hash = hashlib.sha256(
            (OUT / counter_manifest["source"]).read_bytes()
        ).hexdigest()
        assert counter_hash == counter_manifest["source_sha256"]
        assert (
            counter_manifest["base_manifest_sha256"]
            == hashlib.sha256((OUT / "freeze_manifest.json").read_bytes()).hexdigest()
        )
    head = subprocess.check_output(
        ["git", "show", "HEAD:ADP/solver/grassman.py"], cwd=ROOT, text=True
    )
    current = (ROOT / "ADP/solver/grassman.py").read_text()

    def solve_ast(text: str) -> str:
        module = ast.parse(text)
        solve = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "solve"
        )
        return ast.dump(solve)

    assert solve_ast(head) == solve_ast(current), "production solve changed"
    for workload in manifest["inner_workloads"]:
        with np.load(OUT / workload["path"], allow_pickle=False) as data:
            actual = {key: array_hash(data[key]) for key in data.files}
        assert actual == workload["hashes"], "snapshot drift"
    captured = {}
    for name in ("capture", "capture_heldout"):
        for row in _read(OUT / "results" / f"{name}.jsonl"):
            if row.get("status") != "captured":
                continue
            with np.load(row["snapshot"], allow_pickle=False) as data:
                for key in ("index_init", "U", "I", "mass", "truth"):
                    if key in data:
                        assert array_hash(data[key]) == row["input_hashes"][key]
            captured[(row["scenario"], row["seed"])] = row["input_hashes"]
    stages = {}
    max_orth = max_profile_drift = 0.0
    for name, expected in (
        ("fullfit_dev", 54),
        ("fullfit_heldout", 30),
        ("inner", 792),
    ):
        rows = _read(OUT / "results" / f"{name}.jsonl")
        if final:
            assert len(rows) == expected, (name, len(rows), expected)
        keys = []
        nonfinite = []
        missing_environment = 0
        statuses = Counter(str(row.get("status")) for row in rows)
        for row in rows:
            key = tuple(
                row.get(k)
                for k in (
                    "scenario",
                    "seed",
                    "m",
                    "d",
                    "J",
                    "arm",
                    "rank_k",
                    "variant",
                    "budget",
                )
            )
            keys.append(key)
            if row.get("status") != "ok":
                continue
            if name != "inner":
                assert row["pair_hash_match"] and row["data_hash_match"], key
                capture = captured[(row["scenario"], row["seed"])]
                for field in ("index_init", "U", "I", "mass"):
                    assert row["first_statistics_hashes"][field] == capture[field]
                if name == "fullfit_heldout":
                    assert (
                        row["selection_sha256"]
                        == hashlib.sha256(
                            (OUT / "results" / "selection.json").read_bytes()
                        ).hexdigest()
                    )
                assert row["completed"] and row["finite"], key
                summary = row["solver_summary"]
                drift = summary.get("max_profile_history_positive_difference")
                if drift is not None:
                    max_profile_drift = max(max_profile_drift, drift)
                    assert drift <= 1e-7, (key, drift)
            else:
                diagnostic = row.get("diagnostics", {})
                if counter_hash:
                    assert row["objective_counter_source_sha256"] == counter_hash
                    if row["arm"] != "reference":
                        count = row["objective_evaluation_count"]
                        assert count == diagnostic["profile_evaluations"] + 2
                        counts = [p["objective_evaluations"] for p in row["trace"]]
                        assert counts == sorted(counts)
                        assert counts[-1] == count
                history = diagnostic.get("profile_history", [])
                if history:
                    drift = float(np.max(np.diff(history))) if len(history) > 1 else 0.0
                    max_profile_drift = max(max_profile_drift, drift)
                    assert drift <= 1e-9 * max(1.0, history[0]), (key, drift)
                    assert np.isclose(
                        row["objective"],
                        diagnostic["profile_objective"],
                        rtol=1e-9,
                        atol=1e-9,
                    ), key
                if row["arm"] == "block_spectral":
                    baseline = (
                        next(
                            r
                            for r in rows
                            if r.get("scenario") == row.get("scenario")
                            and all(
                                r.get(k) == row.get(k)
                                for k in ("seed", "m", "d", "J", "budget")
                            )
                            and r.get("arm") == "spectral"
                        )
                        if row.get("scenario") == "inner"
                        else None
                    )
                    if baseline and baseline.get("status") == "ok":
                        assert np.isclose(
                            row["objective"],
                            baseline["objective"],
                            rtol=1e-12,
                            atol=1e-10,
                        ), key
            orth = row.get("orthogonality_error", 0.0)
            max_orth = max(max_orth, orth)
            assert orth < 1e-9, (key, orth)
            if not np.isfinite(row.get("objective", row.get("projector_error", 0.0))):
                nonfinite.append(key)
            env = row.get("environment")
            if env:
                assert all(int(v) == 1 for v in env["thread_env"].values()), key
                assert all(pool["num_threads"] == 1 for pool in env["threadpools"]), key
            else:
                missing_environment += 1
        assert len(keys) == len(set(keys)), (name, "duplicate jobs")
        assert not nonfinite, nonfinite
        stages[name] = {
            "count": len(rows),
            "statuses": dict(statuses),
            "expected": expected,
            "rows_without_environment": missing_environment,
            "sha256": hashlib.sha256(
                (OUT / "results" / f"{name}.jsonl").read_bytes()
            ).hexdigest()
            if rows
            else None,
        }
    polish_path = OUT / "results" / "reference_polish_manifest.json"
    if polish_path.exists():
        polish_manifest = json.loads(polish_path.read_text())
        assert polish_manifest["inner_jsonl_sha256"] == stages["inner"]["sha256"]
        for name, expected_hash in polish_manifest["source_hashes"].items():
            path = ROOT / "tests" / name if name.startswith("test_") else OUT / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        polished = _read(OUT / "results" / "reference_polish.jsonl")
        if final:
            assert len(polished) == polish_manifest["selected_case_count"] == 5
        for row in polished:
            assert row["worker_status"]["status"] == "ok"
            result = row["polish_result"]
            assert result["status"] == "ok" and np.isfinite(result["objective"])
            assert result["objective"] <= row["original_reference"]["objective"]
            assert result["orthogonality_error"] < 1e-9
            assert all(
                pool["num_threads"] == 1
                for pool in result["environment"]["threadpools"]
            )
            if result["converged"]:
                assert result["normalized_gradient"] <= 1e-9
        stages["reference_polish"] = {
            "count": len(polished),
            "converged": sum(row["polish_result"]["converged"] for row in polished),
            "all_objectives_improved": True,
            "sha256": hashlib.sha256(
                (OUT / "results" / "reference_polish.jsonl").read_bytes()
            ).hexdigest(),
        }
    return {
        "source_hashes_verified": len(manifest["source_sha256"]),
        "snapshots_verified": len(manifest["inner_workloads"]),
        "core_solve_AST_identical_HEAD": True,
        "stages": stages,
        "max_orthogonality_error": max_orth,
        "max_profile_history_positive_difference": max_profile_drift,
        "audit_scope": (
            "source/input hashes, pairing, geometry, monotonicity, threads, "
            "independent refit and spectral identity; no scientific inference"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    result = audit(args.final)
    target = (
        OUT
        / "results"
        / ("lead_audit.json" if args.final else "lead_progress_audit.json")
    )
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
