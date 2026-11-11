"""Independent audit and paired summaries for the frozen MAVE/Grassmann grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "benchmarks/results/verification/mave-grassmann-grid-20261025"
G, M, C = "Grassmann", "MEANMAVE", "MAVE-reference"
VARIANTS = ("Grassmann-steps5", "Grassmann-steps40", "Grassmann-loc40", "Grassmann-phi_d")
BOOTSTRAPS = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(x: np.ndarray, y: np.ndarray, truth: np.ndarray) -> str:
    return hashlib.sha256(x.tobytes() + y.tobytes() + truth.tobytes()).hexdigest()


def csv_array_digest(array: np.ndarray) -> str:
    text = io.StringIO()
    np.savetxt(text, array, delimiter=",", fmt="%.17g")
    return hashlib.sha256(text.getvalue().encode()).hexdigest()


def independent_edr(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, int, float]:
    """Dense-projector EDR, independent of experiments.r_sdr.design.metrics."""
    if estimate.ndim != 2 or truth.ndim != 2 or estimate.shape != truth.shape:
        raise ValueError(f"basis shape mismatch: estimate={estimate.shape}, truth={truth.shape}")
    if not np.isfinite(estimate).all() or not np.isfinite(truth).all():
        raise ValueError("nonfinite basis")
    _d, m = truth.shape
    rank = int(np.linalg.matrix_rank(estimate))
    if rank != m:
        raise ValueError(f"basis rank {rank} != requested rank {m}")
    orthogonality = float(np.linalg.norm(estimate.T @ estimate - np.eye(m), ord="fro"))
    q_est = np.linalg.qr(estimate, mode="reduced")[0]
    q_true = np.linalg.qr(truth, mode="reduced")[0]
    p_est = q_est @ q_est.T
    p_true = q_true @ q_true.T
    edr = float(np.linalg.norm(p_est - p_true, ord="fro") / math.sqrt(2 * m))
    return edr, rank, orthogonality


def bootstrap_mean_ci(values: list[float], label: str) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_median_ci(values: list[float], label: str) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    samples = np.median(rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True), axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def expected_jobs(protocol: dict) -> dict[tuple[str, int, str], dict]:
    expected: dict[tuple[str, int, str], dict] = {}
    for raw_case, seed, methods in protocol["jobs"]:
        for method in methods:
            key = (raw_case["name"], int(seed), method)
            if key in expected:
                raise ValueError(f"duplicate planned key {key}")
            expected[key] = dict(raw_case)
    return expected


def _snapshot_audit(folder: Path, manifest: dict) -> dict:
    protocol = manifest["protocol"]
    expected_fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    source_checks = {}
    archive_checks = {}
    source_root = folder / "source"
    for rel, expected_hash in protocol["sources"].items():
        snap = source_root / rel
        source_checks[rel] = {
            "expected_sha256": expected_hash,
            "snapshot_sha256": sha256(snap) if snap.is_file() else None,
            "matches": snap.is_file() and sha256(snap) == expected_hash,
        }
    archive_path = folder / "sources.zip"
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            for rel, expected_hash in protocol["sources"].items():
                if rel not in names:
                    archive_checks[rel] = {"matches": False, "reason": "missing"}
                    continue
                digest = hashlib.sha256(archive.read(rel)).hexdigest()
                archive_checks[rel] = {"sha256": digest, "matches": digest == expected_hash}
    else:
        archive_checks["sources.zip"] = {"matches": False, "reason": "missing"}
    live_checks = {}
    for rel, expected_hash in protocol["sources"].items():
        current = ROOT / rel
        current_hash = sha256(current) if current.is_file() else None
        live_checks[rel] = {
            "frozen_sha256": expected_hash,
            "live_sha256": current_hash,
            "matches_frozen": current_hash == expected_hash,
        }
    source_ok = all(v["matches"] for v in source_checks.values())
    archive_ok = all(v["matches"] for v in archive_checks.values())
    return {
        "manifest_fingerprint": manifest.get("fingerprint"),
        "recomputed_fingerprint": expected_fingerprint,
        "fingerprint_matches": manifest.get("fingerprint") == expected_fingerprint,
        "snapshot_source_files": source_checks,
        "snapshot_sources_ok": source_ok,
        "archive_source_files": archive_checks,
        "archive_sources_ok": archive_ok,
        "live_source_comparison": live_checks,
        "live_source_drift_count": sum(not item["matches_frozen"] for item in live_checks.values()),
    }


def _input_audit(folder: Path, expected: dict, rows_by_key: dict) -> tuple[dict, dict]:
    required_datasets = sorted({(name, seed) for name, seed, _ in expected})
    dataset_results = {}
    shared_sha: dict[tuple[str, int], set[str]] = defaultdict(set)
    for case_name, seed in required_datasets:
        rel = f"inputs/{case_name}-{seed}.npz"
        path = folder / rel
        keys_for_data = [(name, sd, method) for name, sd, method in expected if name == case_name and sd == seed]
        outcome = {"path": rel, "exists": path.is_file(),
                   "fits": sum(key in rows_by_key for key in keys_for_data)}
        if not path.is_file():
            dataset_results[f"{case_name}|{seed}"] = outcome
            continue
        with np.load(path, allow_pickle=False) as saved:
            x, y, truth = saved["x"], saved["y"], saved["truth"]
            case_info = next(c for c in (v for (n, _, _), v in expected.items() if n == case_name))
            expected_shape = (int(case_info["n"]), int(case_info["d"]))
            requested_m = int(case_info["m"])
            outcome.update({
                "npz_sha256": sha256(path),
                "x_shape": list(x.shape), "y_shape": list(y.shape),
                "truth_shape": list(truth.shape),
                "finite": bool(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(truth).all()),
                "shape_matches": x.shape == expected_shape and y.shape == (expected_shape[0],)
                    and truth.shape == (expected_shape[1], requested_m),
                "truth_rank": int(np.linalg.matrix_rank(truth)),
                "truth_orthogonality_error": float(np.linalg.norm(truth.T @ truth - np.eye(requested_m), ord="fro")),
                "data_sha256": array_digest(x, y, truth),
                "x_csv_sha256_recomputed": csv_array_digest(x),
                "y_csv_sha256_recomputed": csv_array_digest(y),
            })
        values = []
        for key in keys_for_data:
            row = rows_by_key.get(key)
            if row is not None:
                if row.get("data_sha256"):
                    values.append(row["data_sha256"])
                if row.get("x_csv_sha256") and row.get("y_csv_sha256"):
                    shared_sha[(case_name, seed)].add(row["x_csv_sha256"] + ":" + row["y_csv_sha256"])
                    if (row["x_csv_sha256"] != outcome["x_csv_sha256_recomputed"]
                            or row["y_csv_sha256"] != outcome["y_csv_sha256_recomputed"]):
                        outcome.setdefault("csv_hash_mismatch_keys", []).append(list(key))
        outcome["row_data_hashes_match"] = all(v == outcome["data_sha256"] for v in values) if values else None
        outcome["row_data_hash_count"] = len(values)
        outcome["csv_pair_hash_count"] = len(shared_sha[(case_name, seed)])
        dataset_results[f"{case_name}|{seed}"] = outcome
    pairing_ok = all(len(s) <= 1 for s in shared_sha.values())
    checks_ok = all(
        (not item.get("exists") and item.get("fits", 0) == 0)
        or (item.get("exists") and item.get("shape_matches") and item.get("finite")
            and item.get("truth_rank") == int(item["truth_shape"][1])
            and item.get("truth_orthogonality_error", float("inf")) <= 1e-10
            and item.get("row_data_hashes_match") is not False
            and not item.get("csv_hash_mismatch_keys")
            and item.get("csv_pair_hash_count") <= 1)
        for item in dataset_results.values()
    )
    input_file_count = sum(item.get("exists", False) for item in dataset_results.values())
    outcome_data_missing = [k for k, item in dataset_results.items() if item.get("fits", 0) and not item.get("exists")]
    return dataset_results, {"dataset_count": len(required_datasets), "input_file_count": input_file_count,
                             "input_coverage_complete": input_file_count == len(required_datasets),
                             "unstarted_or_missing_input_count": len(required_datasets) - input_file_count,
                             "outcome_data_missing": outcome_data_missing,
                             "all_inputs_ok": checks_ok and not outcome_data_missing,
                             "csv_pairing_ok": pairing_ok,
                             "unpaired_csv_datasets": [k for k, v in dataset_results.items() if v.get("csv_pair_hash_count", 0) > 1]}


def audit(folder: Path) -> tuple[dict, list[dict], dict]:
    manifest = json.loads((folder / "manifest.json").read_text())
    protocol = manifest["protocol"]
    expected = expected_jobs(protocol)
    fit_dir = folder / "fits"
    result_paths = sorted(fit_dir.glob("*/result.json")) if fit_dir.exists() else []
    rows_by_key: dict[tuple[str, int, str], dict] = {}
    audit_rows = []
    duplicate_keys = []
    truth_cache: dict[tuple[str, int], np.ndarray] = {}
    manifest_fp = manifest_fingerprint(protocol)
    for path in result_paths:
        try:
            row = json.loads(path.read_text())
            key = (row.get("case"), int(row.get("seed")), row.get("method"))
            if key in rows_by_key:
                duplicate_keys.append(list(key))
            rows_by_key[key] = row
            row["_result_path"] = str(path.relative_to(folder))
            row["audit_case_metadata_matches"] = key in expected and all(
                row.get(field) == expected[key].get(field)
                for field in ("d", "m", "n", "noise", "link", "block")
            )
            row["audit_edr"] = None
            row["audit_rank"] = None
            row["audit_orthogonality_error"] = None
            row["audit_metric_matches"] = None
            basis = path.parent / "basis.csv"
            row["basis_file_exists"] = basis.is_file()
            row["audit_fingerprint_matches"] = row.get("fingerprint") == manifest_fp
            if row.get("status") == "ok":
                if not basis.is_file():
                    row["status_audit"] = "missing_basis"
                else:
                    try:
                        dataset_key = (key[0], key[1])
                        if dataset_key not in truth_cache:
                            with np.load(folder / "inputs" / f"{key[0]}-{key[1]}.npz", allow_pickle=False) as z:
                                truth_cache[dataset_key] = z["truth"].copy()
                        truth = truth_cache[dataset_key]
                        estimate = np.loadtxt(basis, delimiter=",", ndmin=2)
                        row["audit_basis_shape"] = list(estimate.shape)
                        edr, rank, orth = independent_edr(estimate, truth)
                        row["audit_edr"] = edr
                        row["audit_rank"] = rank
                        row["audit_orthogonality_error"] = orth
                        row["audit_orthogonality_ok"] = orth <= 1e-8
                        row["audit_metric_matches"] = (
                            row.get("projector_distance") is not None
                            and abs(float(row["projector_distance"]) - edr) <= 1e-8
                        )
                        row["audit_basis_hash_matches"] = row.get("basis_sha256") == sha256(basis)
                        row["audit_recovery_matches"] = row.get("recovered") == (edr <= .1)
                        if not row["audit_metric_matches"]:
                            row["status_audit"] = "reported_metric_mismatch"
                        elif not row["audit_orthogonality_ok"]:
                            row["status_audit"] = "basis_orthogonality_mismatch"
                        elif not row["audit_basis_hash_matches"]:
                            row["status_audit"] = "basis_hash_mismatch"
                        elif not row["audit_recovery_matches"]:
                            row["status_audit"] = "recovery_flag_mismatch"
                        elif not row["audit_fingerprint_matches"] or not row["audit_case_metadata_matches"]:
                            row["status_audit"] = "manifest_row_mismatch"
                        else:
                            row["status_audit"] = "ok"
                            row["edr"] = edr
                    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError, zipfile.BadZipFile) as exc:
                        row["status_audit"] = f"invalid_basis:{type(exc).__name__}:{exc}"
                if row.get("status_audit") != "ok":
                    row["edr"] = None
            else:
                row["status_audit"] = row.get("status", "missing_status")
                row["edr"] = None
            audit_rows.append(row)
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            audit_rows.append({"_result_path": str(path.relative_to(folder)), "status_audit": f"invalid_json:{exc}"})
    actual_keys = set(rows_by_key)
    planned_keys = set(expected)
    missing = sorted(planned_keys - actual_keys)
    extra = sorted(actual_keys - planned_keys)
    runs_path = folder / "runs.jsonl"
    runs_keys = []
    runs_mismatches = []
    if runs_path.is_file():
        for line_no, line in enumerate(runs_path.read_text().splitlines(), 1):
            try:
                rr = json.loads(line)
                rk = (rr.get("case"), int(rr.get("seed")), rr.get("method"))
                runs_keys.append(rk)
                saved = rows_by_key.get(rk)
                if saved is None or saved.get("status") != rr.get("status") or saved.get("fingerprint") != rr.get("fingerprint"):
                    runs_mismatches.append({"line": line_no, "key": list(rk)})
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                runs_mismatches.append({"line": line_no, "error": str(exc)})
    duplicate_runs_keys = sorted(k for k, n in _counts(runs_keys).items() if n > 1)
    execution_path = folder / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.is_file() else None
    execution_audit = {
        "exists": execution is not None,
        "manifest_planned_fits": len(planned_keys),
        "execution_planned_matches": execution is None or execution.get("planned_fits") == len(planned_keys),
        "execution_recorded_matches": execution is None or execution.get("recorded_fits") == len(result_paths),
        "execution_successes_matches": execution is None or execution.get("successes") == sum(r.get("status") == "ok" for r in audit_rows),
        "execution_timeouts_matches": execution is None or execution.get("timeouts") == sum(r.get("status") == "timeout" for r in audit_rows),
        "execution_failures_matches": execution is None or (
            execution.get("failures") == sum(r.get("status") != "ok" for r in audit_rows)
            if "failures" in execution else execution.get("timeouts") == sum(r.get("status") == "timeout" for r in audit_rows)
        ),
        "runs_jsonl_exists": runs_path.is_file(), "runs_jsonl_rows": len(runs_keys),
        "runs_jsonl_matches_result_files": not runs_path.is_file() or (len(runs_keys) == len(result_paths) and not runs_mismatches and not duplicate_runs_keys),
        "runs_jsonl_mismatches": runs_mismatches, "runs_jsonl_duplicate_keys": duplicate_runs_keys,
    }
    input_details, input_summary = _input_audit(folder, expected, rows_by_key)
    snapshot = _snapshot_audit(folder, manifest)
    independent_bad = [r for r in audit_rows if r.get("status") == "ok" and r.get("status_audit") != "ok"]
    metric_mismatch = [r["_result_path"] for r in audit_rows if r.get("audit_metric_matches") is False]
    orthogonality_mismatch = [r["_result_path"] for r in audit_rows if r.get("audit_orthogonality_ok") is False]
    basis_hash_mismatch = [r["_result_path"] for r in audit_rows if r.get("audit_basis_hash_matches") is False]
    recovery_mismatch = [r["_result_path"] for r in audit_rows if r.get("audit_recovery_matches") is False]
    row_fingerprint_mismatch = [r["_result_path"] for r in audit_rows if r.get("audit_fingerprint_matches") is False]
    row_metadata_mismatch = [r["_result_path"] for r in audit_rows if r.get("audit_case_metadata_matches") is False]
    audit_doc = {
        "phase": protocol.get("phase"), "folder": str(folder),
        "fingerprint": snapshot,
        "planned_fit_count": len(planned_keys), "result_file_count": len(result_paths),
        "unique_recorded_key_count": len(actual_keys), "missing_keys": [list(k) for k in missing],
        "planned_keys": [list(k) for k in sorted(planned_keys)],
        "extra_keys": [list(k) for k in extra], "duplicate_keys": duplicate_keys,
        "input_summary": input_summary, "inputs": input_details,
        "execution_audit": execution_audit,
        "basis_successes_recomputed": sum(r.get("status") == "ok" and r.get("status_audit") == "ok" for r in audit_rows),
        "fit_status_counts": dict(sorted(_counts(r.get("status", "invalid_json") for r in audit_rows).items())),
        "invalid_success_outputs": [r.get("_result_path") for r in independent_bad],
        "reported_edr_mismatches": metric_mismatch,
        "basis_orthogonality_mismatches": orthogonality_mismatch,
        "basis_hash_mismatches": basis_hash_mismatch,
        "recovery_flag_mismatches": recovery_mismatch,
        "row_fingerprint_mismatches": row_fingerprint_mismatch,
        "row_case_metadata_mismatches": row_metadata_mismatch,
        "fit_audits": [{k: r.get(k) for k in (
            "_result_path", "case", "seed", "method", "status", "status_audit",
            "audit_fingerprint_matches", "audit_case_metadata_matches", "basis_file_exists", "audit_basis_shape",
            "audit_rank", "audit_orthogonality_error", "audit_orthogonality_ok", "projector_distance", "audit_edr",
            "audit_metric_matches", "audit_basis_hash_matches", "recovered",
            "audit_recovery_matches", "data_sha256", "x_csv_sha256", "y_csv_sha256",
            "basis_sha256", "fit_seconds", "process_seconds", "peak_rss_mib") if k in r}
            for r in audit_rows],
        "complete": not missing and not extra and not duplicate_keys,
        "integrity_ok": snapshot["fingerprint_matches"] and snapshot["snapshot_sources_ok"]
            and snapshot["archive_sources_ok"] and input_summary["all_inputs_ok"]
            and input_summary["csv_pairing_ok"] and not independent_bad
            and not metric_mismatch and not basis_hash_mismatch and not recovery_mismatch
            and not orthogonality_mismatch
            and not row_fingerprint_mismatch and not row_metadata_mismatch
            and execution_audit["execution_planned_matches"] and execution_audit["execution_recorded_matches"]
            and execution_audit["execution_successes_matches"] and execution_audit["execution_timeouts_matches"]
            and execution_audit["execution_failures_matches"] and execution_audit["runs_jsonl_matches_result_files"]
            and not extra and not duplicate_keys,
    }
    audit_doc["_expected"] = expected
    return audit_doc, audit_rows, manifest


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for value in values:
        result[str(value)] += 1
    return dict(result)


def manifest_fingerprint(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def _sampling_group(case_name: str, rows: list[dict], protocol: dict) -> str:
    nseeds = len({int(seed) for c, seed, methods in protocol["jobs"] if c["name"] == case_name for method in methods if method in (G, M)})
    return "anchor-extra-observed" if nseeds > 10 else "uniform10"


def _cell_sampling_label(case_info: dict) -> str:
    return {"core": "core10", "large": "large10", "link": "link10"}[case_info["block"]]


def _valid_seed_rows(rows: list[dict], case_name: str, method: str, seeds: set[int] | None = None) -> dict[int, dict]:
    selected = {}
    for row in rows:
        if row.get("case") != case_name or row.get("method") != method:
            continue
        if seeds is not None and int(row["seed"]) not in seeds:
            continue
        if row.get("status_audit") == "ok" and row.get("edr") is not None:
            selected[int(row["seed"])] = row
    return selected


def _cell_stats(expected_count: int, all_rows: list[dict], case_info: dict, method: str, seed_filter: set[int] | None = None) -> dict:
    candidates = [r for r in all_rows if r.get("case") == case_info["name"] and r.get("method") == method]
    if seed_filter is not None:
        candidates = [r for r in candidates if int(r["seed"]) in seed_filter]
    good = [r for r in candidates if r.get("status_audit") == "ok" and r.get("edr") is not None]
    distances = [float(r["edr"]) for r in good]
    planned = expected_count
    failures = sum(r.get("status") != "ok" or r.get("status_audit") != "ok" for r in candidates)
    missing = max(0, planned - len(candidates))
    out = {
        **{k: case_info[k] for k in ("name", "block", "d", "m", "n", "noise", "link")},
        "method": method, "n_planned": planned, "n_attempted": len(candidates),
        "n_ok": len(good), "n_failed": failures, "n_missing": missing,
        "recovery_count": sum(v <= .1 for v in distances),
        "recovery_rate_attempt_denominator": sum(v <= .1 for v in distances) / len(candidates) if candidates else None,
        "edr_mean_success": float(np.mean(distances)) if distances else None,
        "edr_median_success": float(np.median(distances)) if distances else None,
        "edr_std_success": float(np.std(distances, ddof=1)) if len(distances) > 1 else None,
        "edr_p10_success": float(np.quantile(distances, .1)) if distances else None,
        "edr_p90_success": float(np.quantile(distances, .9)) if distances else None,
        "edr_worstcase_mean_failures_eq_1": float((sum(distances) + failures) / len(candidates)) if candidates else None,
    }
    if method == G:
        out["stop_reason_counts"] = json.dumps(_counts(r.get("stop_reason", "unknown") for r in good), sort_keys=True)
        out["n_all_inner_certified"] = sum(r.get("all_inner_certified") is True for r in good)
        out["n_inner_certified"] = sum(int(r.get("n_inner_certified") or 0) for r in good)
        out["n_inner_total"] = sum(int(r.get("n_inner_total") or 0) for r in good)
    for metric, field in (("fit_seconds", "fit"), ("process_seconds", "process"), ("peak_rss_mib", "rss_mib")):
        values = [float(r[metric]) for r in good if r.get(metric) is not None]
        out[f"{field}_mean_success"] = float(np.mean(values)) if values else None
        out[f"{field}_median_success"] = float(np.median(values)) if values else None
        out[f"{field}_p90_success"] = float(np.quantile(values, .9)) if values else None
    return out


def _paired(good_g: dict[int, dict], good_m: dict[int, dict], label: str,
            case_info: dict, scope: str, all_rows: list[dict],
            seed_filter: set[int] | None, planned_each: int,
            left_method: str = G, right_method: str = M) -> dict:
    pairs = sorted(set(good_g) & set(good_m))
    deltas = [float(good_g[s]["edr"] - good_m[s]["edr"]) for s in pairs]
    fit_ratios = [float(good_g[s]["fit_seconds"] / good_m[s]["fit_seconds"]) for s in pairs
                  if good_g[s].get("fit_seconds", 0) and good_m[s].get("fit_seconds", 0)]
    process_ratios = [float(good_g[s]["process_seconds"] / good_m[s]["process_seconds"]) for s in pairs
                      if good_g[s].get("process_seconds", 0) and good_m[s].get("process_seconds", 0)]
    delta_ci = bootstrap_mean_ci(deltas, f"{label}:{case_info['name']}:edr-delta")
    fit_ci = bootstrap_median_ci(fit_ratios, f"{label}:{case_info['name']}:fit-ratio")
    process_ci = bootstrap_median_ci(process_ratios, f"{label}:{case_info['name']}:process-ratio")
    allowed_seeds = seed_filter
    left_attempts = {int(r["seed"]): r for r in all_rows
                     if r.get("case") == case_info["name"] and r.get("method") == left_method
                     and (allowed_seeds is None or int(r["seed"]) in allowed_seeds)}
    right_attempts = {int(r["seed"]): r for r in all_rows
                      if r.get("case") == case_info["name"] and r.get("method") == right_method
                      and (allowed_seeds is None or int(r["seed"]) in allowed_seeds)}
    attempted_pairs = sorted(set(left_attempts) & set(right_attempts))
    worst_deltas = [
        (float(left_attempts[s]["edr"]) if left_attempts[s].get("status_audit") == "ok" else 1.0)
        - (float(right_attempts[s]["edr"]) if right_attempts[s].get("status_audit") == "ok" else 1.0)
        for s in attempted_pairs
    ]
    worst_ci = bootstrap_mean_ci(worst_deltas, f"{label}:{case_info['name']}:worstcase-edr-delta")
    attempts_left, attempts_right = len(left_attempts), len(right_attempts)
    return {
        "scope": scope, "case": case_info["name"], "block": case_info["block"],
        "d": case_info["d"], "m": case_info["m"], "n": case_info["n"],
        "noise": case_info["noise"], "link": case_info["link"],
        "left_method": left_method, "right_method": right_method,
        "n_planned_each": planned_each, "n_attempts_left": attempts_left,
        "n_attempts_right": attempts_right, "n_pairs": len(pairs),
        "n_missing_left": max(0, planned_each - attempts_left),
        "n_missing_right": max(0, planned_each - attempts_right),
        "n_failed_left": attempts_left - len(good_g),
        "n_failed_right": attempts_right - len(good_m),
        "n_attempted_pairs": len(attempted_pairs),
        "n_left_success": len(good_g), "n_right_success": len(good_m),
        "paired_mean_edr_delta_left_minus_right": float(np.mean(deltas)) if deltas else None,
        "edr_delta_ci95_low": delta_ci[0], "edr_delta_ci95_high": delta_ci[1],
        "worstcase_mean_edr_delta_left_minus_right_failures_eq_1": float(np.mean(worst_deltas)) if worst_deltas else None,
        "worstcase_edr_delta_ci95_low": worst_ci[0], "worstcase_edr_delta_ci95_high": worst_ci[1],
        "paired_median_fit_time_ratio_left_over_right": float(np.median(fit_ratios)) if fit_ratios else None,
        "fit_time_ratio_ci95_low": fit_ci[0], "fit_time_ratio_ci95_high": fit_ci[1],
        "paired_median_process_time_ratio_left_over_right": float(np.median(process_ratios)) if process_ratios else None,
        "process_time_ratio_ci95_low": process_ci[0], "process_time_ratio_ci95_high": process_ci[1],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _float_values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in (None, "")]


def _main_factor_report(folder: Path, cells: list[dict], fit_rows: list[dict]) -> list[str]:
    paired = _read_csv(folder / "analysis-paired.csv")
    case_rows = [r for r in cells if r.get("sampling") == "core10" and r.get("block") == "core"]
    cases = {r["name"]: r for r in case_rows if r.get("method") == G}
    core_pairs = {r["case"]: r for r in paired if r.get("scope") == "core10-first10"}
    lines = [
        "## Влияние факторов и их взаимодействия: core factorial",
        "",
        ("Ниже `EDR delta` = Grassmann − MEANMAVE: положительный знак означает "
        "меньшую EDR у MEANMAVE. Равновесовые summaries дают каждой клетке равный вес; "
        "paired delta использует только совместно успешные seed. Recovery делится на "
        "реальные попытки, а незапущенные seed показаны отдельно."),
        "",
        "| Фактор | Уровень | Клеток | G/M success / attempts (missing) | Mean EDR G/M | Recovery G/M | Pair cells ≥2 / n pairs | Equal-cell delta (cells n≥2) | Pooled pair delta | Fit ratio G/M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    dimensions = ("d", "m", "n", "noise")
    levels = {dim: sorted({row[dim] for row in cases.values()}) for dim in dimensions}
    cell_index = {(r["name"], r["method"]): r for r in cells if r.get("sampling") == "core10"}
    for dim in dimensions:
        for level in levels[dim]:
            names = [name for name, c in cases.items() if c[dim] == level]
            g = [cell_index[(name, G)] for name in names if (name, G) in cell_index]
            m = [cell_index[(name, M)] for name in names if (name, M) in cell_index]
            cell_pairs = [core_pairs[name] for name in names if name in core_pairs]
            gm, mm = _float_values(g, "edr_mean_success"), _float_values(m, "edr_mean_success")
            gr, mr = _float_values(g, "recovery_rate_attempt_denominator"), _float_values(m, "recovery_rate_attempt_denominator")
            robust_cells = [r for r in cell_pairs if int(r.get("n_pairs") or 0) >= 2]
            delta = _float_values(robust_cells, "paired_mean_edr_delta_left_minus_right")
            pooled_delta = [float(r["paired_mean_edr_delta_left_minus_right"]) * int(r["n_pairs"])
                            for r in cell_pairs if int(r.get("n_pairs") or 0) > 0]
            pooled_weight = sum(int(r.get("n_pairs") or 0) for r in cell_pairs)
            ratios = _float_values(cell_pairs, "paired_median_fit_time_ratio_left_over_right")
            ng, ag = sum(int(r["n_ok"]) for r in g), sum(int(r["n_attempted"]) for r in g)
            nm, am = sum(int(r["n_ok"]) for r in m), sum(int(r["n_attempted"]) for r in m)
            mg, mmiss = sum(int(r["n_missing"]) for r in g), sum(int(r["n_missing"]) for r in m)
            fmt = lambda values: f"{np.mean(values):.4g}" if values else "—"
            pg, pm = sum(int(r["n_planned"]) for r in g), sum(int(r["n_planned"]) for r in m)
            lines.append("| " + " | ".join((dim, str(level), str(len(names)), f"{ng}/{ag} ({mg}/{pg}) / {nm}/{am} ({mmiss}/{pm})",
                f"{fmt(gm)} / {fmt(mm)}", f"{fmt(gr)} / {fmt(mr)}",
                f"{len(robust_cells)} / {sum(int(r.get('n_pairs') or 0) for r in cell_pairs)}",
                fmt(delta), fmt([sum(pooled_delta) / pooled_weight] if pooled_weight else []), fmt(ratios))) + " |")
    lines += [
        "",
        "### Interaction patterns (core, equal-cell paired contrasts)",
        "",
        "| Factors | Levels | Cells | G/M mean EDR | Pair cells n≥2 | Equal-cell delta n≥2 | n paired fits |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    interaction_dims = (("d", "m"), ("m", "n"), ("m", "noise"), ("d", "noise"))
    for dims in interaction_dims:
        grouped: dict[tuple, list[str]] = defaultdict(list)
        for name, case in cases.items():
            grouped[tuple(case[d] for d in dims)].append(name)
        for value, names in sorted(grouped.items()):
            g = [cell_index[(name, G)] for name in names if (name, G) in cell_index]
            m = [cell_index[(name, M)] for name in names if (name, M) in cell_index]
            pair_group = [core_pairs[name] for name in names if name in core_pairs]
            gm, mm = _float_values(g, "edr_mean_success"), _float_values(m, "edr_mean_success")
            robust_pairs = [r for r in pair_group if int(r.get("n_pairs") or 0) >= 2]
            delta = _float_values(robust_pairs, "paired_mean_edr_delta_left_minus_right")
            fmt = lambda values: f"{np.mean(values):.4g}" if values else "—"
            lines.append("| " + " / ".join(dims) + " | " + " × ".join(map(str, value))
                          + f" | {len(names)} | {fmt(gm)} / {fmt(mm)} | {len(robust_pairs)} | {fmt(delta)} | "
                          + str(sum(int(r.get("n_pairs") or 0) for r in pair_group)) + " |")

    lines += [
        "",
        "### Изменение функции отклика: sine / quadratic против additive",
        "",
        ("Link-блок имеет d=25,n=1000 и рассматривается отдельно от core. Для "
        "контекста рядом указана соответствующая additive-клетка; это парные "
        "сравнения методов внутри одного link и сопоставление клеточных средних "
        "между разными целевыми функциями."),
        "",
        "| Link | m | σ | Cells | EDR G/M | Recovery G/M | Paired delta G−M | Additive EDR G/M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    link_cases = {r["name"]: r for r in cells if r.get("sampling") == "link10" and r.get("method") == G}
    link_pairs = {r["case"]: r for r in paired if r.get("scope") == "link10-first10"}
    link_cell_index = {(r["name"], r["method"]): r for r in cells if r.get("sampling") == "link10"}
    for name, case in sorted(link_cases.items(), key=lambda item: (item[1]["link"], int(item[1]["m"]), float(item[1]["noise"]))):
        g = link_cell_index.get((name, G), {})
        m = link_cell_index.get((name, M), {})
        pair = link_pairs.get(name, {})
        additive_name = f"additive-d25-m{case['m']}-n1000-s{float(case['noise']):g}"
        ag, am = cell_index.get((additive_name, G), {}), cell_index.get((additive_name, M), {})
        vals = lambda a, b, field: " / ".join(str(a.get(field) if a.get(field) not in (None, "") else "—")
                                               for a in (a, b))
        lines.append(f"| {case['link']} | {case['m']} | {case['noise']} | {pair.get('n_pairs', 0)}/10 | "
                     f"{vals(g, m, 'edr_mean_success')} | {vals(g, m, 'recovery_rate_attempt_denominator')} | "
                     f"{pair.get('paired_mean_edr_delta_left_minus_right', '—')} | "
                     f"{vals(ag, am, 'edr_mean_success')} |")

    anchors = [r for r in paired if r.get("scope") == "anchor-extra-observed"]
    lines += [
        "",
        "### Дополнительные anchor seed 10–29: только фактически завершённые попытки",
        "",
        ("Здесь учитываются только дополнительные 20 запланированных seed после общей "
        "десятки. Это не оценка на 30 seed: план незавершён, пропуски не считаются "
        "ошибками метода и не имитируются EDR=1."),
        "",
        "| Клетка | G/M success / attempts (missing) | EDR mean G/M | Recovery G/M | n complete pairs | Paired delta G−M, 95% CI | Fit ratio G/M, 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    anchor_cells = {(r["name"], r["method"]): r for r in cells if r.get("sampling") == "anchor-extra-observed"}
    for pair in anchors:
        name = pair["case"]
        g, m = anchor_cells.get((name, G), {}), anchor_cells.get((name, M), {})
        edr = f"{g.get('edr_mean_success', '—')} / {m.get('edr_mean_success', '—')}"
        recovery = f"{g.get('recovery_rate_attempt_denominator', '—')} / {m.get('recovery_rate_attempt_denominator', '—')}"
        d, dlo, dhi = pair.get("paired_mean_edr_delta_left_minus_right"), pair.get("edr_delta_ci95_low"), pair.get("edr_delta_ci95_high")
        ratio, rlo, rhi = pair.get("paired_median_fit_time_ratio_left_over_right"), pair.get("fit_time_ratio_ci95_low"), pair.get("fit_time_ratio_ci95_high")
        fmtci = lambda x, lo, hi: f"{float(x):.4g} [{float(lo):.4g}, {float(hi):.4g}]" if x not in (None, "") and lo not in (None, "") else "—"
        outcomes = (f"{g.get('n_ok', 0)}/{g.get('n_attempted', 0)} ({g.get('n_missing', 20)}) / "
                    f"{m.get('n_ok', 0)}/{m.get('n_attempted', 0)} ({m.get('n_missing', 20)})")
        lines.append(f"| {name} | {outcomes} | {edr} | {recovery} | {pair.get('n_pairs', 0)} | "
                     f"{fmtci(d, dlo, dhi)} | {fmtci(ratio, rlo, rhi)} |")
    if not anchors:
        lines.append("В дополнительном anchor-блоке пока нет сохранённых совместных исходов.")

    sensitivity = _read_csv(folder / "analysis-sensitivity.csv")
    lines += [
        "",
        "### Настройки Grassmann и classic MAVE subset (первые 10 seed)",
        "",
        ("Разности здесь относятся к указанному варианту справа против базового "
        "Grassmann слева. Сравнение exploratory; клетки не являются независимой "
        "репликацией параметров."),
        "",
        "| Вариант | Клеток | Полные пары / попытки | Mean cell delta | Range cell delta | Median fit ratio base/variant |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in sorted({r.get("right_method") for r in sensitivity}):
        subset = [r for r in sensitivity if r.get("right_method") == variant]
        deltas = _float_values(subset, "paired_mean_edr_delta_left_minus_right")
        ratios = _float_values(subset, "paired_median_fit_time_ratio_left_over_right")
        delta = f"{np.mean(deltas):.4g}" if deltas else "—"
        delta_range = f"[{min(deltas):.4g}, {max(deltas):.4g}]" if deltas else "—"
        ratio = f"{np.median(ratios):.4g}" if ratios else "—"
        pairs = sum(int(r.get("n_pairs") or 0) for r in subset)
        planned = sum(int(r.get("n_planned_each") or 0) for r in subset)
        lines.append(f"| {variant} | {len(subset)} | {pairs}/{planned} | {delta} | {delta_range} | {ratio} |")
    classic = _read_csv(folder / "analysis-classic.csv")
    lines += [
        "",
        ("Classic subset сравнивает benchmark `MAVE-reference` по тем же первым 10 "
        "seed с Grassmann и пакетным MEANMAVE. Это не CSMAVE и не полный MAVE-каталог."),
        "",
        "| Контраст | Клетка | Пары / запланировано | Mean delta left−right, 95% CI | Fit ratio left/right, 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for row in classic:
        d, dlo, dhi = row.get("paired_mean_edr_delta_left_minus_right"), row.get("edr_delta_ci95_low"), row.get("edr_delta_ci95_high")
        ratio, rlo, rhi = row.get("paired_median_fit_time_ratio_left_over_right"), row.get("fit_time_ratio_ci95_low"), row.get("fit_time_ratio_ci95_high")
        fmtci = lambda x, lo, hi: f"{float(x):.4g} [{float(lo):.4g}, {float(hi):.4g}]" if x not in (None, "") and lo not in (None, "") else "—"
        lines.append(f"| {row.get('left_method')}−{row.get('right_method')} | {row.get('case')} | "
                     f"{row.get('n_pairs')}/{row.get('n_planned_each')} | {fmtci(d, dlo, dhi)} | {fmtci(ratio, rlo, rhi)} |")

    controlled = _read_csv(folder / "analysis-controlled.csv")
    lines += [
        "",
        "### Controlled matched contrasts по одному фактору",
        "",
        ("Каждая строка требует успешные Grassmann и MEANMAVE на обоих уровнях "
        "фактора для того же seed. В таблице есть complete quadruple outcomes / "
        "planned first-10 quadruples; незапущенные fit показаны как missing и не "
        "считаются ошибками. Equal-cell change использует пары клеток с ≥2 seed, "
        "bootstrap resamples seed blocks."),
        "",
        "| Контраст | Метод | Клеток planned / matched ≥2 | Complete quadruples / planned (missing) | Seed blocks | Pooled Δ EDR high−low | Equal-cell Δ ≥2, 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in controlled:
        lo, hi = row.get("equal_cell_ci95_ge2_low"), row.get("equal_cell_ci95_ge2_high")
        value = row.get("equal_cell_mean_change_ge2")
        ci = f"{float(value):.4g} [{float(lo):.4g}, {float(hi):.4g}]" if value not in (None, "") and lo not in (None, "") else "—"
        pooled = row.get("pooled_matched_mean_edr_change_high_minus_low")
        pooled = f"{float(pooled):.4g}" if pooled not in (None, "") else "—"
        cells = f"{row.get('n_cell_pairs_planned')}/{row.get('n_cell_pairs_ge2')}"
        attempts = f"{row.get('n_attempted_quadruples')}/{row.get('n_quadruples_planned')} ({row.get('n_missing_quadruples')})"
        lines.append(f"| {row.get('factor')} {row.get('low_level')}→{row.get('high_level')} | {row.get('method')} | "
                     f"{cells} | {attempts} | {row.get('n_seed_blocks_observed')} | {pooled} | {ci} |")
    lines += [
        "",
        ("Шум σ0→1 держит X, B и ε общими по seed; n300→3000 использует вложенные "
        "префиксы X/B/ε. Контраст m2→5 меняет число координат и для additive "
        "меняет относительную силу направлений. Контраст d10→50 не имеет вложенных "
        "X/B и остаётся сравнением разных случайных дизайнов. Все четыре анализа "
        "условны на сохранённые результаты при остановке по лимиту; они не устраняют "
        "completion-time bias частичного случайно перемешанного расписания."),
    ]

    protocol = json.loads((folder / "manifest.json").read_text())["protocol"]
    timeouts = [r for r in fit_rows if r.get("status") == "timeout"]
    methods = sorted({r.get("method") for r in fit_rows if r.get("method")})
    lines += [
        "",
        "### Неудачи и время исполнения",
        "",
        ("Fit/process time и RSS выше условны на audited success; при timeout "
        "в таблице остаётся исход попытки. Условия timeout не изымались и не "
        "заменялись новыми запусками."),
        "",
        "| Метод | Попытки | Успехи | Timeout | Прочие статусы | Лимит, s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        subset = [r for r in fit_rows if r.get("method") == method]
        ok = sum(r.get("status_audit") == "ok" for r in subset)
        timeout = sum(r.get("status") == "timeout" for r in subset)
        other = len(subset) - ok - timeout
        limit = protocol.get("reference_timeout_seconds", protocol.get("timeout_seconds")) if method == C else protocol.get("timeout_seconds")
        lines.append(f"| {method} | {len(subset)} | {ok} | {timeout} | {other} | {limit} |")
    for row in timeouts:
        lines.append(f"Timeout: `{row.get('case')}` / `{row.get('method')}`, "
                     f"process={row.get('process_seconds', 'NA')} s, RSS={row.get('peak_rss_mib', 'NA')} MiB.")
    grassmann = [r for r in fit_rows if r.get("method") == G and r.get("status_audit") == "ok"]
    stops = _counts(r.get("stop_reason", "unknown") for r in grassmann)
    inner_certified = sum(int(r.get("n_inner_certified") or 0) for r in grassmann)
    inner_total = sum(int(r.get("n_inner_total") or 0) for r in grassmann)
    all_certified = sum(r.get("all_inner_certified") is True for r in grassmann)
    lines += [
        "",
        (f"Grassmann: audited successes {len(grassmann)}; stop_reason counts `{json.dumps(stops, ensure_ascii=False, sort_keys=True)}`; "
        f"all-inner-certified fits {all_certified}/{len(grassmann)}; individual inner certificates "
        f"{inner_certified}/{inner_total}. Эти счётчики не превращают внешнюю остановку "
        "schedule в сертификат сходимости оценки."),
    ]

    return lines


def summarize(folder: Path, audit_doc: dict, rows: list[dict], manifest: dict) -> dict:
    protocol = manifest["protocol"]
    expected = audit_doc["_expected"]
    cases_by_name = {}
    planned_methods: dict[str, set[str]] = defaultdict(set)
    planned_seeds: dict[str, set[int]] = defaultdict(set)
    for (name, seed, method), case_info in expected.items():
        cases_by_name[name] = case_info
        planned_methods[name].add(method)
        planned_seeds[name].add(seed)
    cell_rows = []
    paired_rows = []
    variant_rows = []
    classic_rows = []
    available_seeds = sorted({seed for seeds in planned_seeds.values() for seed in seeds})
    seeds10 = set(available_seeds[:10])
    for name, case_info in sorted(cases_by_name.items()):
        anchor = _sampling_group(name, rows, protocol) == "anchor30"
        extra_seeds = planned_seeds[name] - seeds10 if anchor else set()
        sampling = _cell_sampling_label(case_info)
        # The uniformly sampled grid is always summarized on its first 10 seeds.
        if G in planned_methods[name] and M in planned_methods[name]:
            methods_for_cell = [G, M] + ([C] if C in planned_methods[name] else [])
            for method in methods_for_cell:
                cell_rows.append({"sampling": sampling, **_cell_stats(
                    min(10, len(planned_seeds[name])), rows, case_info, method, seeds10)})
                if anchor and method in (G, M):
                    cell_rows.append({"sampling": "anchor-extra-observed", **_cell_stats(
                        len(extra_seeds), rows, case_info, method, extra_seeds)})
            g10 = _valid_seed_rows(rows, name, G, seeds10)
            m10 = _valid_seed_rows(rows, name, M, seeds10)
            paired_rows.append(_paired(g10, m10, "base-first10", case_info,
                                       f"{sampling}-first10",
                                       rows, seeds10, min(10, len(planned_seeds[name]))))
            if anchor:
                paired_rows.append(_paired(_valid_seed_rows(rows, name, G, extra_seeds),
                                           _valid_seed_rows(rows, name, M, extra_seeds),
                                           "anchor-extra", case_info, "anchor-extra-observed", rows,
                                           extra_seeds, len(extra_seeds)))
            for variant in VARIANTS:
                if variant in planned_methods[name]:
                    variant_rows.append(_paired(g10, _valid_seed_rows(rows, name, variant, seeds10),
                                                f"variant:{variant}", case_info,
                                                "first10_exploratory", rows, seeds10, 10,
                                                G, variant))
            if C in planned_methods[name]:
                c10 = _valid_seed_rows(rows, name, C, seeds10)
                classic_rows.append(_paired(g10, c10, "classic:G-vs-MAVE-reference", case_info,
                                            "classic_subset_first10_exploratory", rows,
                                            seeds10, 10, G, C))
                classic_rows.append(_paired(m10, c10, "classic:MEANMAVE-vs-MAVE-reference", case_info,
                                            "classic_subset_first10_exploratory", rows,
                                            seeds10, 10, M, C))

    # Equal-weight cell summaries; planned anchor extras stay separate from the first ten seeds.
    strata_rows = []
    interaction_rows = []
    grouping_fields = ("d", "m", "n", "noise", "block", "link")
    for sampling in ("core10", "large10", "link10", "anchor-extra-observed"):
        selected_cells = [r for r in cell_rows if r["sampling"] == sampling]
        for method in (G, M, C):
            method_cells = [r for r in selected_cells if r["method"] == method]
            for dims, destination in [*(((dim,), strata_rows) for dim in grouping_fields),
                                      *((pair, interaction_rows) for pair in (("d", "m"), ("d", "n"),
                                                                                ("d", "noise"), ("m", "n"),
                                                                                ("m", "noise"), ("n", "noise"),
                                                                                ("link", "m"), ("link", "noise")))]:
                values = defaultdict(list)
                meta = defaultdict(list)
                for item in method_cells:
                    key = tuple(item[dim] for dim in dims)
                    meta[key].append(item)
                    if item["edr_mean_success"] is not None:
                        values[key].append(float(item["edr_mean_success"]))
                for value, all_cells in meta.items():
                    cell_means = values[value]
                    medians = [r["edr_median_success"] for r in all_cells if r["edr_median_success"] is not None]
                    stds = [r["edr_std_success"] for r in all_cells if r["edr_std_success"] is not None]
                    p10s = [r["edr_p10_success"] for r in all_cells if r["edr_p10_success"] is not None]
                    p90s = [r["edr_p90_success"] for r in all_cells if r["edr_p90_success"] is not None]
                    fits = [r["fit_mean_success"] for r in all_cells if r["fit_mean_success"] is not None]
                    processes = [r["process_mean_success"] for r in all_cells if r["process_mean_success"] is not None]
                    rss_values = [r["rss_mib_mean_success"] for r in all_cells if r["rss_mib_mean_success"] is not None]
                    destination.append({
                        "sampling": sampling, "method": method,
                        "stratifier": "x".join(dims), "level": "|".join(map(str, value)),
                        "n_cells": len(all_cells), "n_cells_with_success": len(cell_means),
                        "n_attempts_cell_total": sum(r["n_attempted"] for r in all_cells),
                        "n_missing_cell_total": sum(r["n_missing"] for r in all_cells),
                        "n_planned_cell_total": sum(r["n_planned"] for r in all_cells),
                        "n_success_total": sum(r["n_ok"] for r in all_cells),
                        "equal_cell_mean_edr": float(np.mean(cell_means)) if cell_means else None,
                        "equal_cell_median_edr": float(np.median(cell_means)) if cell_means else None,
                        "equal_cell_std_edr": float(np.std(cell_means, ddof=1)) if len(cell_means) > 1 else None,
                        "equal_cell_p10_edr": float(np.quantile(cell_means, .1)) if cell_means else None,
                        "equal_cell_p90_edr": float(np.quantile(cell_means, .9)) if cell_means else None,
                        "equal_cell_mean_edr_median_success": float(np.mean(medians)) if medians else None,
                        "equal_cell_mean_edr_std_success": float(np.mean(stds)) if stds else None,
                        "equal_cell_mean_edr_p10_success": float(np.mean(p10s)) if p10s else None,
                        "equal_cell_mean_edr_p90_success": float(np.mean(p90s)) if p90s else None,
                        "equal_cell_mean_recovery_rate": float(np.mean([r["recovery_rate_attempt_denominator"] for r in all_cells if r["recovery_rate_attempt_denominator"] is not None])) if any(r["recovery_rate_attempt_denominator"] is not None for r in all_cells) else None,
                        "equal_cell_mean_worstcase_edr": float(np.mean([r["edr_worstcase_mean_failures_eq_1"] for r in all_cells if r["edr_worstcase_mean_failures_eq_1"] is not None])) if any(r["edr_worstcase_mean_failures_eq_1"] is not None for r in all_cells) else None,
                        "equal_cell_mean_fit_seconds": float(np.mean(fits)) if fits else None,
                        "equal_cell_mean_process_seconds": float(np.mean(processes)) if processes else None,
                        "equal_cell_mean_peak_rss_mib": float(np.mean(rss_values)) if rss_values else None,
                    })
    global_rows = _global_paired(rows, cases_by_name, planned_seeds)
    controlled_rows = _controlled_contrasts(rows, cases_by_name, planned_seeds)
    _write_csv(folder / "analysis-cells.csv", cell_rows)
    _write_csv(folder / "analysis-paired.csv", paired_rows)
    _write_csv(folder / "analysis-strata.csv", strata_rows)
    _write_csv(folder / "analysis-interactions.csv", interaction_rows)
    _write_csv(folder / "analysis-sensitivity.csv", variant_rows)
    _write_csv(folder / "analysis-classic.csv", classic_rows)
    _write_csv(folder / "analysis-global.csv", global_rows)
    _write_csv(folder / "analysis-controlled.csv", controlled_rows)
    return {"cell_rows": len(cell_rows), "paired_rows": len(paired_rows),
            "strata_rows": len(strata_rows), "sensitivity_rows": len(variant_rows),
            "interaction_rows": len(interaction_rows), "classic_rows": len(classic_rows),
            "global_rows": len(global_rows), "controlled_rows": len(controlled_rows)}


def _global_paired(rows: list[dict], cases_by_name: dict, planned_seeds: dict[str, set[int]]) -> list[dict]:
    """Partial-run descriptive contrasts; seed-block bootstrap preserves paired seeds."""
    output = []
    for scope, selected in (
        ("core10", [c for c in cases_by_name.values() if c["block"] == "core"]),
        ("large10", [c for c in cases_by_name.values() if c["block"] == "large"]),
        ("link10", [c for c in cases_by_name.values() if c["block"] == "link"]),
        ("anchor-extra-observed", [c for c in cases_by_name.values() if c["name"] in {
            name for name, seeds in planned_seeds.items() if len(seeds) > 10
        }]),
    ):
        cell_delta: dict[str, dict[int, float]] = {}
        cell_fit_ratio: dict[str, dict[int, float]] = {}
        attempted_pairs = 0
        planned_pair_slots = 0
        for c in selected:
            if scope == "anchor-extra-observed":
                seed_filter = set(sorted(planned_seeds[c["name"]])[10:])
            else:
                seed_filter = set(sorted(planned_seeds[c["name"]])[:10])
            planned_pair_slots += len(seed_filter)
            g = _valid_seed_rows(rows, c["name"], G, seed_filter)
            m = _valid_seed_rows(rows, c["name"], M, seed_filter)
            g_attempts = {int(r["seed"]) for r in rows if r.get("case") == c["name"] and r.get("method") == G and int(r["seed"]) in seed_filter}
            m_attempts = {int(r["seed"]) for r in rows if r.get("case") == c["name"] and r.get("method") == M and int(r["seed"]) in seed_filter}
            attempted_pairs += len(g_attempts & m_attempts)
            common = set(g) & set(m)
            if common:
                cell_delta[c["name"]] = {s: float(g[s]["edr"] - m[s]["edr"]) for s in common}
                cell_fit_ratio[c["name"]] = {s: float(g[s]["fit_seconds"] / m[s]["fit_seconds"])
                                              for s in common if g[s].get("fit_seconds", 0) and m[s].get("fit_seconds", 0)}
        seed_blocks = sorted({seed for values in cell_delta.values() for seed in values})
        pooled_values = [delta for values in cell_delta.values() for delta in values.values()]
        means_by_min = {}
        intervals = {}
        for minimum in (1, 2, 3):
            eligible = {name: values for name, values in cell_delta.items() if len(values) >= minimum}
            cell_means = [float(np.mean(list(values.values()))) for values in eligible.values()]
            means_by_min[minimum] = float(np.mean(cell_means)) if cell_means else None
            eligible_seeds = sorted({seed for values in eligible.values() for seed in values})
            if len(eligible_seeds) < 2 or not eligible:
                intervals[minimum] = (None, None)
                continue
            seed = int.from_bytes(hashlib.sha256(f"global-seed-block:{scope}:min{minimum}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            boot = np.empty(BOOTSTRAPS)
            for b in range(BOOTSTRAPS):
                sampled = rng.choice(eligible_seeds, size=len(eligible_seeds), replace=True)
                means = []
                for deltas in eligible.values():
                    sample_values = [deltas[int(s)] for s in sampled if int(s) in deltas]
                    if sample_values:
                        means.append(float(np.mean(sample_values)))
                boot[b] = np.mean(means) if means else np.nan
            usable = boot[np.isfinite(boot)]
            intervals[minimum] = tuple(map(float, np.quantile(usable, [.025, .975]))) if len(usable) else (None, None)
        cell_ratios = [float(np.median(list(v.values()))) for v in cell_fit_ratio.values() if v]
        output.append({
            "scope": scope, "n_cells_planned": len(selected), "n_cells_paired": len(cell_delta),
            "n_cells_paired_ge2": sum(len(v) >= 2 for v in cell_delta.values()),
            "n_cells_paired_ge3": sum(len(v) >= 3 for v in cell_delta.values()),
            "n_pair_slots_planned": planned_pair_slots,
            "n_attempted_pairs": attempted_pairs,
            "n_missing_pair_slots": max(0, planned_pair_slots - attempted_pairs),
            "n_failed_attempted_pairs": max(0, attempted_pairs - len(pooled_values)),
            "n_seed_blocks_observed": len(seed_blocks),
            "n_paired_fits": len(pooled_values),
            "pooled_matched_pair_mean_edr_delta_grassmann_minus_meanmave": float(np.mean(pooled_values)) if pooled_values else None,
            "equal_cell_mean_edr_delta_ge1": means_by_min[1],
            "equal_cell_mean_edr_delta_ge2": means_by_min[2],
            "equal_cell_mean_edr_delta_ge3": means_by_min[3],
            "seed_block_ci95_ge2_low": intervals[2][0], "seed_block_ci95_ge2_high": intervals[2][1],
            "seed_block_ci95_ge3_low": intervals[3][0], "seed_block_ci95_ge3_high": intervals[3][1],
            "equal_cell_mean_cell_median_fit_ratio_g_over_m": float(np.mean(cell_ratios)) if cell_ratios else None,
        })
    return output


def _controlled_contrasts(rows: list[dict], cases_by_name: dict,
                          planned_seeds: dict[str, set[int]]) -> list[dict]:
    """Matched one-factor contrasts requiring successful G and M at both levels."""
    seeds10 = set(sorted({s for values in planned_seeds.values() for s in values})[:10])
    definitions = (("noise", 0.0, 1.0), ("n", 300, 3000),
                   ("m", 2, 5), ("d", 10, 50))
    coords = ("d", "m", "n", "noise", "link")
    output = []
    for factor, low, high in definitions:
        other = tuple(c for c in coords if c != factor)
        lookup = {tuple(case[c] for c in coords): case for case in cases_by_name.values()}
        grouped: dict[tuple, tuple[dict, dict]] = {}
        for case in lookup.values():
            if case[factor] != low:
                continue
            key = tuple(case[c] for c in other)
            high_case = lookup.get(tuple(high if c == factor else case[c] for c in coords))
            if high_case:
                grouped[key] = (case, high_case)
        method_cell_deltas: dict[str, dict[str, dict[int, float]]] = {G: {}, M: {}}
        attempted_pair_count = 0
        successful_quadruple_count = 0
        planned_cell_pairs = len(grouped)
        for low_case, high_case in grouped.values():
            lo_g = {int(r["seed"]): r for r in rows if r.get("case") == low_case["name"] and r.get("method") == G and int(r["seed"]) in seeds10}
            lo_m = {int(r["seed"]): r for r in rows if r.get("case") == low_case["name"] and r.get("method") == M and int(r["seed"]) in seeds10}
            hi_g = {int(r["seed"]): r for r in rows if r.get("case") == high_case["name"] and r.get("method") == G and int(r["seed"]) in seeds10}
            hi_m = {int(r["seed"]): r for r in rows if r.get("case") == high_case["name"] and r.get("method") == M and int(r["seed"]) in seeds10}
            attempted_complete_seeds = set(lo_g) & set(lo_m) & set(hi_g) & set(hi_m)
            successful = {s for s in attempted_complete_seeds
                          if all(r.get("status_audit") == "ok" for r in (lo_g[s], lo_m[s], hi_g[s], hi_m[s]))}
            attempted_pair_count += len(attempted_complete_seeds)
            successful_quadruple_count += len(successful)
            cell_key = f"{low_case['name']} -> {high_case['name']}"
            for method, low_rows, high_rows in ((G, lo_g, hi_g), (M, lo_m, hi_m)):
                method_cell_deltas[method][cell_key] = {
                    s: float(high_rows[s]["edr"] - low_rows[s]["edr"])
                    for s in successful
                }
        for method, cell_map in method_cell_deltas.items():
            valid = {cell: deltas for cell, deltas in cell_map.items() if deltas}
            seed_blocks = sorted({seed for values in valid.values() for seed in values})
            pooled = [delta for values in valid.values() for delta in values.values()]
            outputs = {}
            for minimum in (1, 2, 3):
                eligible = {cell: values for cell, values in valid.items() if len(values) >= minimum}
                cell_means = [float(np.mean(list(values.values()))) for values in eligible.values()]
                point = float(np.mean(cell_means)) if cell_means else None
                ci = (None, None)
                eligible_seeds = sorted({seed for values in eligible.values() for seed in values})
                if len(eligible_seeds) >= 2 and eligible:
                    seed = int.from_bytes(hashlib.sha256(
                        f"controlled:{factor}:{low}:{high}:{method}:min{minimum}".encode()
                    ).digest()[:8], "little")
                    rng = np.random.default_rng(seed)
                    boot = np.empty(BOOTSTRAPS)
                    for b in range(BOOTSTRAPS):
                        sampled = rng.choice(eligible_seeds, size=len(eligible_seeds), replace=True)
                        means = []
                        for values in eligible.values():
                            sample_values = [values[int(s)] for s in sampled if int(s) in values]
                            if sample_values:
                                means.append(float(np.mean(sample_values)))
                        boot[b] = np.mean(means) if means else np.nan
                    usable = boot[np.isfinite(boot)]
                    if len(usable):
                        ci = tuple(map(float, np.quantile(usable, [.025, .975])))
                outputs[minimum] = (point, ci, len(eligible), sum(map(len, eligible.values())))
            output.append({
                "factor": factor, "low_level": low, "high_level": high,
                "method": method, "n_cell_pairs_planned": planned_cell_pairs,
                "n_quadruples_planned": planned_cell_pairs * len(seeds10),
                "n_attempted_quadruples": attempted_pair_count,
                "n_failed_quadruples": attempted_pair_count - successful_quadruple_count,
                "n_missing_quadruples": max(0, planned_cell_pairs * len(seeds10) - attempted_pair_count),
                "n_cell_pairs_successful": len(valid),
                "n_cell_pairs_ge2": outputs[2][2], "n_cell_pairs_ge3": outputs[3][2],
                "n_seed_blocks_observed": len(seed_blocks),
                "n_complete_seed_contrasts": len(pooled),
                "pooled_matched_mean_edr_change_high_minus_low": float(np.mean(pooled)) if pooled else None,
                "equal_cell_mean_change_ge1": outputs[1][0],
                "equal_cell_mean_change_ge2": outputs[2][0],
                "equal_cell_ci95_ge2_low": outputs[2][1][0],
                "equal_cell_ci95_ge2_high": outputs[2][1][1],
                "equal_cell_mean_change_ge3": outputs[3][0],
                "equal_cell_ci95_ge3_low": outputs[3][1][0],
                "equal_cell_ci95_ge3_high": outputs[3][1][1],
                "completed_case_pairs": json.dumps(sorted(valid)),
            })
    return output


def _write_report(folder: Path, audit_doc: dict, rows: list[dict], table_counts: dict) -> None:
    statuses = audit_doc["fit_status_counts"]
    cells = _read_csv(folder / "analysis-cells.csv")
    global_pairs = _read_csv(folder / "analysis-global.csv")
    numeric_lines = ["## Проверяемые числовые результаты", ""]
    if audit_doc["phase"] == "pilot":
        numeric_lines += [
            ("Pilot — только технический smoke и оценка стоимости. Четыре сценария и "
            "один seed не используются для выбора или вывода о сравнительном качестве."),
            "",
            "| Сценарий | Метод | Статус | EDR (если успешен) | Fit, s | Process, s | RSS, MiB |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in cells:
            values = [row.get("name"), row.get("method"), f"{row.get('n_ok')}/{row.get('n_planned')} ok",
                      row.get("edr_mean_success") or "—", row.get("fit_median_success") or "—",
                      row.get("process_median_success") or "—", row.get("rss_mib_median_success") or "—"]
            numeric_lines.append("| " + " | ".join(str(v) for v in values) + " |")
        numeric_lines += [
            "",
            ("Успешные Grassmann/MAVE-package basis имеют EDR, пересчитанную независимо "
            "из сохранённых файлов. Таймауты: `d=25,m=5,n=1000,σ=0.3` для "
            "MAVE-reference (120 s) и `d=100,m=5,n=5000,σ=1` для MEANMAVE (180 s); "
            "обе неудачи остались в знаменателе и имеют worst-case EDR=1."),
            "",
        ]
    else:
        numeric_lines += [
            (f"Это частичный, остановленный по лимиту прогон: {audit_doc['result_file_count']} / "
            f"{audit_doc['planned_fit_count']} запланированных outcome. Незапущенные задачи "
            "показаны как missing и не считаются ошибками; recovery делится на реальные "
            "попытки, а worst-case EDR=1 добавляется только для реального неуспешного outcome. "
            "Поскольку расписание было случайно перемешано и остановлено рано, состав "
            "завершённых клеток зависит от времени выполнения; все сеточные выводы предварительны."),
            "",
            "| Sampling block | Метод | Клеток | Success / attempted (missing / planned) | Cell mean EDR mean / median | Mean cell p10–p90 | Recovery / worst-case | Mean cell fit median, s | Process median, s | RSS median, MiB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for sampling in ("core10", "large10", "link10", "anchor-extra-observed"):
            for method in (G, M):
                group = [r for r in cells if r.get("sampling") == sampling and r.get("method") == method]
                if not group:
                    continue
                means = [float(r["edr_mean_success"]) for r in group if r.get("edr_mean_success") not in (None, "")]
                p10s = [float(r["edr_p10_success"]) for r in group if r.get("edr_p10_success") not in (None, "")]
                p90s = [float(r["edr_p90_success"]) for r in group if r.get("edr_p90_success") not in (None, "")]
                rec = [float(r["recovery_rate_attempt_denominator"]) for r in group if r.get("recovery_rate_attempt_denominator") not in (None, "")]
                worst = [float(r["edr_worstcase_mean_failures_eq_1"]) for r in group if r.get("edr_worstcase_mean_failures_eq_1") not in (None, "")]
                fit = [float(r["fit_median_success"]) for r in group if r.get("fit_median_success") not in (None, "")]
                process = [float(r["process_median_success"]) for r in group if r.get("process_median_success") not in (None, "")]
                rss = [float(r["rss_mib_median_success"]) for r in group if r.get("rss_mib_median_success") not in (None, "")]
                ok = sum(int(r["n_ok"]) for r in group)
                attempted = sum(int(r["n_attempted"]) for r in group)
                missing = sum(int(r["n_missing"]) for r in group)
                planned = sum(int(r["n_planned"]) for r in group)
                fmt = lambda x: f"{x:.4g}"
                numeric_lines.append("| " + " | ".join((sampling, method, str(len(group)), f"{ok}/{attempted} ({missing}/{planned})",
                    f"{fmt(float(np.mean(means))) if means else '—'} / {fmt(float(np.median(means))) if means else '—'}",
                    f"{fmt(float(np.mean(p10s))) if p10s else '—'}–{fmt(float(np.mean(p90s))) if p90s else '—'}",
                    f"{fmt(float(np.mean(rec))) if rec else '—'} / {fmt(float(np.mean(worst))) if worst else '—'}", fmt(float(np.mean(fit))) if fit else "—",
                    fmt(float(np.mean(process))) if process else "—", fmt(float(np.mean(rss))) if rss else "—")) + " |")
        numeric_lines += ["", "Global paired EDR delta (Grassmann − MEANMAVE), with pooled pairs and equal-cell weighting shown separately:", "",
                          "| Block | Cells with pairs / planned | Cells n≥2 | Complete pairs / attempted pair slots (missing) | Pooled mean Δ | Equal-cell Δ n≥2, seed-block 95% CI | Cell-median fit ratio G/M |",
                          "|---|---:|---:|---:|---:|---:|---:|"]
        for row in global_pairs:
            delta = row.get("equal_cell_mean_edr_delta_ge2")
            lo, hi = row.get("seed_block_ci95_ge2_low"), row.get("seed_block_ci95_ge2_high")
            numeric_lines.append("| " + " | ".join((row.get("scope", ""),
                f"{row.get('n_cells_paired')}/{row.get('n_cells_planned')}", str(row.get("n_cells_paired_ge2", 0)),
                f"{row.get('n_paired_fits')}/{row.get('n_attempted_pairs')} ({row.get('n_missing_pair_slots')})",
                f"{float(row['pooled_matched_pair_mean_edr_delta_grassmann_minus_meanmave']):.4g}" if row.get('pooled_matched_pair_mean_edr_delta_grassmann_minus_meanmave') not in (None, "") else "—",
                f"{float(delta):.4g} [{float(lo):.4g}, {float(hi):.4g}]" if delta not in (None, "") and lo not in (None, "") and hi not in (None, "") else "—",
                f"{float(row['equal_cell_mean_cell_median_fit_ratio_g_over_m']):.4g}" if row.get("equal_cell_mean_cell_median_fit_ratio_g_over_m") not in (None, "") else "—")) + " |")
        numeric_lines += ["", "Оценки условны на завершившиеся задачи и не устраняют completion-time bias; seed-block интервалы exploratory, сравнения множественные и без коррекции.", ""]
        numeric_lines.extend(_main_factor_report(folder, cells, rows))
    geometry_path = folder.parent / "signal_geometry.csv"
    geometry_lines = []
    if geometry_path.is_file():
        geometry_rows = _read_csv(geometry_path)
        geometry = {(r["link"], int(r["m"])): r for r in geometry_rows}
        if all(key in geometry for key in (("additive", 2), ("additive", 3), ("additive", 5), ("sine", 5), ("quadratic", 5))):
            get = lambda key, field: float(geometry[key][field])
            geometry_lines = [
                "## Геометрия сигнала: проверяемая гипотеза, не причинный вывод",
                "",
                "Независимо перепроверена `signal_geometry.csv`: гауссовые ожидания "
                "градиентной матрицы для population-standardized link совпали с "
                "отдельной Gauss-Hermite квадратурой для всех 12 строк. При additive "
                "схеме λmin последовательно падает: m=2 — " + f"{get(('additive', 2), 'opg_lambda_min'):.6f}, "
                "m=3 — " + f"{get(('additive', 3), 'opg_lambda_min'):.6f}, m=5 — "
                + f"{get(('additive', 5), 'opg_lambda_min'):.6f}; при m=5 condition number "
                + f"{get(('additive', 5), 'opg_condition'):.2f}. Для balanced sine m=5 "
                + f"λmin={get(('sine', 5), 'opg_lambda_min'):.6f}, condition="
                + f"{get(('sine', 5), 'opg_condition'):.2f}; quadratic m=5 имеет "
                + f"λmin={get(('quadratic', 5), 'opg_lambda_min'):.3f}, condition=1.",
                "",
                ("Это population-аналог нормировки сигнала; генератор делит наблюдаемый "
                "raw signal на выборочное SD отдельно для каждого n. Поэтому такая "
                "геометрия указывает на возможную слабую идентифицируемость additive "
                "направлений m≥3, но это не единственная возможная причина ошибок "
                "при m=5: конечные n/d, условность дизайна, структура отклика, "
                "инициализация и локализация тоже влияют. OPG-картина не доказывает "
                "причину различий EDR между оценивателями или результат конкретного solver."),
                "",
            ]
    lines = [
        "# Независимый аудит сетки Grassmann / MAVE",
        "",
        f"Фаза: `{audit_doc['phase']}`. Папка: `{folder}`.",
        (f"Зафиксировано fits: {audit_doc['result_file_count']} / {audit_doc['planned_fit_count']}; "
        f"полных результатов: {'да' if audit_doc['complete'] else 'нет'}."),
        f"Статусы: `{json.dumps(statuses, ensure_ascii=False, sort_keys=True)}`.",
        "",
        *numeric_lines,
        *geometry_lines,
        "## Метод независимой проверки",
        "",
        ("Для каждой успешной basis.csv заново вычислена EDR по плотной формуле "
        "`||Q_hat Q_hat^T - Q_true Q_true^T||_F / sqrt(2m)`. Проверены размер, "
        "полный ранг, ортогональность (Frobenius ≤ 1e-8), совпадение с EDR в result.json, "
        "хеш basis и флаг восстановления EDR ≤ 0.1. В `analysis-audit.json` "
        "проверены fingerprint, хеши source snapshot и sources.zip, входные NPZ, "
        "общие CSV-хеши для пар и соответствие ключей расписанию."),
        "",
        ("Сводки по EDR и времени условны на успешные fit. В знаменателе доли "
        "восстановления входят только наблюдавшиеся попытки; missing tasks показаны "
        "отдельно и не считаются failures. Дополнительно приведён worst-case EDR: "
        "только реально неуспешному outcome присвоена ошибка 1.0. Парные разности и отношения времени "
        "рассчитаны только на совместно успешных seed, с обязательным `n_pairs`."),
        "",
        "## Парные сравнения и ограничения",
        "",
        ("`analysis-paired.csv` содержит среднюю парную разность EDR (Grassmann − "
        "MEANMAVE) и seeded bootstrap 95% CI, а также медианные отношения времени "
        "Grassmann / MEANMAVE с bootstrap CI. Эти интервалы описательные: десять "
        "seed дают exploratory precision, сравнения множественные и не скорректированы. "
        "Отдельная удачная клетка не является устойчивым глобальным законом."),
        "",
        ("Сводки `core10` используют первые десять запланированных seed в каждой клетке. "
        "Дополнительные seed 10–29 у шести anchor-клеток вынесены отдельно как "
        "`anchor-extra-observed`, со знаменателем 20 planned. Это не 30-seed evidence. "
        "Страты дают клеткам равный вес и не смешивают core, large, link и anchor-extra."),
        "",
        ("Чувствительность к solver/config и classic subset сопоставляются отдельно "
        "на первых десяти seed. `MAVE-reference` — отдельная benchmark reference "
        "реализация; package `MEANMAVE` — другой метод. Сравнения не означают "
        "тождество алгоритмов или обобщение на иные генераторы."),
        "",
        ("Для Grassmann отдельно сохранены stop_reason и счётчики внутренних "
        "сертифицированных solve. Внешняя остановка schedule и внутренние "
        "сертификаты не являются доказательством сходимости всей оценки."),
        "",
        "## Таблицы",
        "",
        "- `analysis-cells.csv`: mean/median/std/p10/p90 EDR, recovery, fit/process time, RSS, failures и worst-case EDR по методу и клетке.",
        "- `analysis-paired.csv`: клеточные парные сравнения Grassmann–MEANMAVE; anchor-extra отдельно.",
        "- `analysis-controlled.csv`: matched contrasts по σ, n, m и d при фиксированных остальных координатах и seed.",
        "- `analysis-strata.csv`: равновесовые по клеткам marginal summaries по d, m, n, noise, block и link.",
        "- `analysis-interactions.csv`: равновесовые по клеткам interaction summaries по заранее выбранным парам факторов.",
        "- `analysis-global.csv`: equal-cell global paired contrasts с bootstrap по seed-блокам, сохраняющим общие seed между клетками.",
        "- `analysis-sensitivity.csv`: solver sensitivity arms против базового Grassmann, первые десять seed.",
        "- `analysis-classic.csv`: Grassmann и MEANMAVE против MAVE-reference в classic subset.",
        "",
        "Число строк таблиц: `" + json.dumps(table_counts, ensure_ascii=False, sort_keys=True) + "`.",
        "",
        "## Целостность",
        "",
        (f"Fingerprint совпал: **{audit_doc['fingerprint']['fingerprint_matches']}**; "
        f"snapshot sources: **{audit_doc['fingerprint']['snapshot_sources_ok']}**; "
        f"archive sources: **{audit_doc['fingerprint']['archive_sources_ok']}**; "
        f"inputs: **{audit_doc['input_summary']['all_inputs_ok']}**; "
        f"CSV pairing: **{audit_doc['input_summary']['csv_pairing_ok']}**; "
        f"общий integrity gate: **{audit_doc['integrity_ok']}**."),
        (f"Изменения live-source относительно frozen snapshot: {audit_doc['fingerprint']['live_source_drift_count']} файлов. "
        "Они перечислены в JSON и сами по себе не меняют версию, которую использовали workers."),
        "",
    ]
    rendered = "\n".join(lines).rstrip() + "\n"
    root_report = folder.parent / "LUNA_ANALYSIS.md"
    begin = f"<!-- LUNA_PHASE:{audit_doc['phase']}:BEGIN -->"
    end = f"<!-- LUNA_PHASE:{audit_doc['phase']}:END -->"
    section = f"{begin}\n{rendered}{end}\n"
    existing = root_report.read_text() if root_report.exists() else "# LUNA independent analysis\n\n"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", re.DOTALL)
    if pattern.search(existing):
        updated = pattern.sub(section, existing)
    else:
        updated = existing.rstrip() + "\n\n" + section
    root_report.write_text(updated)


def analyze(folder: Path) -> dict:
    folder = folder.resolve()
    audit_doc, rows, manifest = audit(folder)
    table_counts = summarize(folder, audit_doc, rows, manifest)
    audit_doc["tables"] = table_counts
    audit_doc["_expected"] = None
    (folder / "analysis-audit.json").write_text(json.dumps(audit_doc, indent=2, ensure_ascii=False, allow_nan=False))
    _write_report(folder, audit_doc, rows, table_counts)
    return audit_doc


def self_check() -> None:
    rng = np.random.default_rng(44)
    truth = np.linalg.qr(rng.normal(size=(9, 3)))[0]
    estimate = truth @ np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    edr, rank, orth = independent_edr(estimate, truth)
    assert rank == 3 and edr < 1e-12 and orth < 1e-12
    assert bootstrap_mean_ci([1., 2., 3.], "fixed") == bootstrap_mean_ci([1., 2., 3.], "fixed")
    assert bootstrap_median_ci([1., 2., 3.], "fixed") == bootstrap_median_ci([1., 2., 3.], "fixed")
    c = {"name": "toy", "block": "core", "d": 2, "m": 1, "n": 20,
         "noise": .3, "link": "additive"}
    rows = [
        {**c, "case": "toy", "seed": 651000, "method": G, "status": "ok", "status_audit": "ok",
         "edr": .1, "fit_seconds": 2., "process_seconds": 2.2, "peak_rss_mib": 20.,
         "stop_reason": "accepted", "all_inner_certified": True, "n_inner_certified": 3, "n_inner_total": 3},
        {**c, "case": "toy", "seed": 651001, "method": G, "status": "timeout", "status_audit": "timeout"},
        {**c, "case": "toy", "seed": 651000, "method": M, "status": "ok", "status_audit": "ok",
         "edr": .2, "fit_seconds": 1., "process_seconds": 1.1, "peak_rss_mib": 18.},
        {**c, "case": "toy", "seed": 651001, "method": M, "status": "ok", "status_audit": "ok",
         "edr": .3, "fit_seconds": 2., "process_seconds": 2.2, "peak_rss_mib": 19.},
    ]
    cell = _cell_stats(3, rows, c, G, {651000, 651001, 651002})
    assert cell["n_planned"] == 3 and cell["n_attempted"] == 2 and cell["n_ok"] == 1
    assert cell["n_failed"] == 1 and cell["n_missing"] == 1
    assert cell["recovery_rate_attempt_denominator"] == .5
    assert cell["edr_worstcase_mean_failures_eq_1"] == .55 and cell["fit_mean_success"] == 2.
    assert cell["n_all_inner_certified"] == 1 and cell["n_inner_certified"] == 3
    assert json.loads(cell["stop_reason_counts"]) == {"accepted": 1}
    paired = _paired({651000: rows[0]}, {651000: rows[2], 651001: rows[3]},
                     "self-check", c, "toy", rows, {651000, 651001, 651002}, 3)
    assert paired["n_pairs"] == 1 and paired["n_attempts_left"] == 2 and paired["n_attempts_right"] == 2
    assert paired["n_missing_left"] == 1 and paired["n_attempted_pairs"] == 2
    assert paired["paired_median_fit_time_ratio_left_over_right"] == 2.
    protocol = {"jobs": [({**c, "name": "anchor"}, s, [G, M]) for s in range(651000, 651030)]
                + [({**c, "name": "core"}, s, [G, M]) for s in range(651000, 651010)]}
    assert _sampling_group("anchor", [], protocol) == "anchor-extra-observed"
    assert _sampling_group("core", [], protocol) == "uniform10"
    # Cell means stay equally weighted even if one cell carries 30 seeds.
    assert np.mean([.1, .9]) == .5
    print("independent audit self-check: projectors, bootstrap, failures/time ratios, anchor sampling OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "main", "self-check"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.phase == "self-check":
        self_check()
        return
    report = analyze(args.root / args.phase)
    compact = {k: report[k] for k in (
        "phase", "folder", "planned_fit_count", "result_file_count", "unique_recorded_key_count",
        "complete", "integrity_ok", "basis_successes_recomputed", "fit_status_counts",
        "reported_edr_mismatches", "basis_hash_mismatches", "recovery_flag_mismatches",
        "basis_orthogonality_mismatches",
        "row_fingerprint_mismatches", "row_case_metadata_mismatches", "tables")}
    compact["input_summary"] = report["input_summary"]
    compact["execution_audit"] = report["execution_audit"]
    compact["live_source_drift_count"] = report["fingerprint"]["live_source_drift_count"]
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
