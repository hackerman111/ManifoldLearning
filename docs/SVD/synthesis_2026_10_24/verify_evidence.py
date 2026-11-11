"""Audit saved evidence for the synthesis; never rerun fits or timings.

Run from EDR_code/: .venv/bin/python docs/SVD/synthesis_2026_10_24/verify_evidence.py
Only this directory receives outputs. Original research artifacts are read-only.
"""

import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = set()
RESULT = {"scope": "saved-row aggregation and small formula checks; no fit or timing rerun"}


def read(relative):
    SOURCES.add(relative)
    return (ROOT / relative).read_text()


def load(relative):
    return json.loads(read(relative))


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-14), (actual, expected)


def rows_at(relative, method_key, count):
    rows = [json.loads(line) for line in read(relative).splitlines() if line.strip()]
    assert len(rows) == count
    assert all(row["status"] == "ok" for row in rows)
    by = {(r["case"], r["seed"], r[method_key]): r for r in rows}
    assert len(by) == count
    blocks = {(r["case"], r["seed"]) for r in rows}
    methods = {r[method_key] for r in rows}
    assert len(blocks) * len(methods) == count
    for case, seed in blocks:
        group = [by[case, seed, method] for method in methods]
        if "input_hashes" in group[0]:
            keys = ("X", "Y", "truth", "initial_basis") if method_key == "arm" else ("x", "y", "truth")
            for key in keys:
                assert len({r["input_hashes"][key] for r in group}) == 1
        else:
            for key in ("data_sha256", "x_csv_sha256", "y_csv_sha256"):
                assert len({r[key] for r in group}) == 1
        if method_key == "method":
            assert len({r["method_seed"] for r in group}) == 1
    return by, sorted(blocks)


def compare(by, blocks, candidate, base, metric="projector_distance", timer="fit_seconds"):
    ratios = [by[*key, candidate][metric] / by[*key, base][metric] for key in blocks]
    times = [by[*key, candidate][timer] / by[*key, base][timer] for key in blocks]
    return {"pairs": len(blocks), "edr_geomean": math.exp(statistics.mean(map(math.log, ratios))),
            "time_median": statistics.median(times), "time_p90": float(np.quantile(times, .9))}


def check_overall(actual, expected):
    close(actual["edr_geomean"], expected["edr_ratio_geomean"])
    close(actual["time_median"], expected["time_ratio_median_p90"][0])
    close(actual["time_p90"], expected["time_ratio_median_p90"][1])
    actual["reported_edr_bootstrap95"] = expected["edr_ratio_bootstrap95"]


tex = read("docs/SVD/AI_research_synthesis.tex")
links = re.findall(r"\\repo\{([^{}]+)\}", tex)
assert links and all((ROOT / path).is_file() for path in links)
SOURCES.update(links)
RESULT["document_links"] = {"total": len(links), "unique": len(set(links)), "all_exist": True}

relative_root = "benchmarks/results/outputs/r_sdr/grassman-relative-heldout-2026-10-22/"
by, blocks = rows_at(relative_root + "runs.jsonl", "method", 90)
summary = load(relative_root + "analysis_summary.json")
relative = compare(by, blocks, summary["candidate"], summary["baseline"])
check_overall(relative, summary["overall"])
assert all(summary["gates"].values())
RESULT["relative_heldout"] = relative

combined_root = "benchmarks/results/outputs/r_sdr/grassman-combined-heldout-2026-10-22/"
by, blocks = rows_at(combined_root + "runs.jsonl", "method", 150)
summary = load(combined_root + "combined_analysis.json")
RESULT["combined_heldout"] = {}
for name, item in summary.items():
    actual = compare(by, blocks, item["candidate"], item["baseline"])
    check_overall(actual, item["overall"])
    RESULT["combined_heldout"][name] = actual

lowrank_root = "benchmarks/results/outputs/r_sdr/grassman-lowrank22-development/"
by, blocks = rows_at(lowrank_root + "runs.jsonl", "method", 280)
summary = load(lowrank_root + "paired_analysis.json")
assert len(blocks) == 70 and not summary["failures"] and not summary["missing_rows"]
RESULT["lowrank_development"] = {}
for method in ("budget10", "projected10"):
    actual = compare(by, blocks, method, "baseline")
    item = summary["methods"][method]
    close(actual["edr_geomean"], item["EDR_geomean_base"])
    close(actual["time_median"], item["fit_time_median_base"])
    close(actual["time_p90"], item["fit_time_p90_base"])
    assert not item["all_gates_pass"]
    actual["reported_edr_bootstrap95"] = item["EDR_geomean_base_bootstrap95"]
    RESULT["lowrank_development"][method] = actual
actual = compare(by, blocks, "projected10", "budget10")
item = summary["comparisons"]["projected10_vs_budget10"]
close(actual["edr_geomean"], item["EDR_geomean"])
actual["reported_edr_bootstrap95"] = item["EDR_geomean_bootstrap95"]
RESULT["lowrank_development"]["projected10_vs_budget10"] = actual

by, blocks = rows_at("benchmarks/experiments/adaptive_centers/results/fullfit_dev.jsonl", "arm", 120)
assert len(blocks) == 12
assert not any(r["inner_certificates"]["all_converged"] for r in by.values())
RESULT["adaptive_centers"] = {}
for candidate, base, delta4, time3, recovered in [
    ("floor25", "baseline", .0, .994, 1),
    ("floor50", "baseline", .0015, .989, 2),
    ("floor25_search", "floor25", .0, 1.682, 1),
    ("logdet_equal_j", "baseline", .0464, 2.481, 0),
    ("logdet_coverage_equal_j", "baseline", .0397, 2.487, 0),
    ("logdet_coverage_quad_equal_j", "baseline", .0242, 2.814, 0),
]:
    deltas = [by[*key, candidate]["edr_error"] - by[*key, base]["edr_error"] for key in blocks]
    times = [by[*key, candidate]["fit_wall_sec"] / by[*key, base]["fit_wall_sec"] for key in blocks]
    errors = [by[*key, candidate]["edr_error"] for key in blocks]
    median_delta, median_time = statistics.median(deltas), statistics.median(times)
    assert abs(median_delta - delta4) <= .000051
    assert abs(median_time - time3) <= .00051
    assert sum(e <= .1 for e in errors) == recovered
    RESULT["adaptive_centers"][candidate + "_vs_" + base] = {
        "pairs": len(blocks), "edr_delta_median": median_delta,
        "time_ratio_median": median_time, "recovered": recovered}

svd = load("benchmarks/experiments/svd_improvements_2026-09-30/validation_summary.json")
assert svd["passed"] and svd["rows"] == 144 and not svd["failures"]
RESULT["svd_saved_validation"] = {key: svd[key] for key in (
    "rows", "passed", "scaled_median_time_ratio", "scaled_median_pass_ratio",
    "max_quality_change", "default_isotropic_time_ratio")}

angle = load("docs/SVD/theory_2026_10_01/angle_results.json")
ratios = []
for row in angle["rows"]:
    ratio = (row["common_projection_s"] + row["preparation_s"] + row["scalar_curve_s"]) / (
        row["common_projection_s"] + row["batched_svd_curve_s"])
    close(ratio, row["total_time_ratio"])
    ratios.append(ratio)
RESULT["saved_angle_time_ratio_range"] = [min(ratios), max(ratios)]

RESULT["fresh_formula_checks"] = {}
with tempfile.TemporaryDirectory(prefix="svd-synthesis-formulas-") as directory:
    for script, output, count in [
        ("docs/SVD/documentation/check_math.py", "math_checks.json", 24),
        ("docs/SVD/theory_2026_10_01/check_formulas.py", "formula_results.json", 203),
    ]:
        SOURCES.add(script)
        target = Path(directory) / Path(script).name
        shutil.copy2(ROOT / script, target)
        result = subprocess.run([sys.executable, str(target)], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        data = json.loads((Path(directory) / output).read_text())
        assert len(data["checks"]) == count
        assert data.get("passed", data.get("status") == "passed")
        (OUT / ("fresh_" + output)).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        RESULT["fresh_formula_checks"][script] = {"count": count, "passed": True}

RESULT["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                           for path in sorted(SOURCES)}
RESULT["interval_scope"] = "CIs retained from saved analyses; point aggregates recomputed independently"
RESULT["status"] = "passed"
(OUT / "evidence_audit.json").write_text(json.dumps(RESULT, indent=2, ensure_ascii=False) + "\n")
print(f"Evidence audit passed: {len(RESULT['source_sha256'])} source hashes, "
      f"{len(links)} source links, 640 saved full-fit rows, 24 + 203 fresh formula checks.")
