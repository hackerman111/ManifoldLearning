"""Audit and summarize the four frozen exploratory accuracy pilots."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmarks/results/outputs/r_sdr"
PILOTS = {
    "accuracy": ("grassman-accuracy22-pilot", 24),
    "gn": ("grassman-accuracy22-gn-pilot", 16),
    "ridge": ("grassman-accuracy22-ridge-pilot", 20),
    "krylov": ("grassman-accuracy22-krylov-pilot", 20),
}
METHODS = (
    "relative10",
    "smooth_relative",
    "stable_jacobian",
    "diagonal",
    "gn_quadratic",
    "ridge6",
    "ridge4",
    "ridge2",
    "krylov10",
    "krylov20",
)
CASES = ("d-5", "d-20", "d-50", "corr-0.9")


def geo(xs: list[float]) -> float | None:
    return math.exp(statistics.mean(map(math.log, xs))) if xs else None


def q90(xs: list[float]) -> float | None:
    if not xs:
        return None
    a = sorted(xs)
    i = 0.9 * (len(a) - 1)
    return a[int(i)] + (i - int(i)) * (a[min(int(i) + 1, len(a) - 1)] - a[int(i)])


def load(name: str) -> tuple[list[dict], dict]:
    path = OUT / name
    rows = [json.loads(s) for s in (path / "runs.jsonl").read_text().splitlines()]
    protocol = json.loads((path / "manifest.json").read_text())["protocol"]
    with zipfile.ZipFile(path / "sources.zip") as zf:
        for rel, sha in protocol["sources"].items():
            if hashlib.sha256(zf.read(rel)).hexdigest() != sha:
                raise ValueError(f"frozen source hash mismatch: {name}:{rel}")
    return rows, protocol


def summarize() -> dict:
    pilots, manifests, inputs = {}, {}, {}
    failures, stops, lsmr = [], Counter(), Counter()
    for key, (name, count) in PILOTS.items():
        rows, manifest = load(name)
        if len(rows) != count or manifest["seed"] != 90800 or manifest["runs"] != 1:
            raise ValueError(f"unexpected row/seed protocol: {key}")
        pilots[key], manifests[key] = rows, manifest
        for r in rows:
            if r.get("status") != "ok":
                failures.append((key, r.get("case"), r.get("method"), r.get("status")))
            stops[(r.get("method"), r.get("stop_reason"))] += 1
            ident = (r["case"], r["seed"])
            hashes = {k: r["input_hashes"][k] for k in ("x", "y", "truth")}
            paired = (hashes, r["method_seed"])
            old = inputs.setdefault(ident, paired)
            if old != paired:
                raise ValueError(f"paired data/seed differs: {ident}")
            for d in r.get("inner_diagnostics", []):
                for code, iterations, *_ in d.get("krylov_lsmr", []):
                    lsmr[(r["method"], "calls")] += 1
                    lsmr[(r["method"], f"istop_{code}")] += 1
                    lsmr[(r["method"], "iterations")] += iterations

    source_equal = {}
    for k, m in manifests.items():
        src = m["sources"]
        source_equal[k] = src.get("ADP/solver/grassman.py") is not None and src.get(
            "ADP/solver/grassman.py"
        ) == src.get("benchmarks/experiments/grassman_accuracy_2026_10_22/reference.py")
    live = hashlib.sha256((ROOT / "ADP/solver/grassman.py").read_bytes()).hexdigest()
    ref = hashlib.sha256(
        (ROOT / "benchmarks/experiments/grassman_accuracy_2026_10_22/reference.py").read_bytes()
    ).hexdigest()
    head_clean = (
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "ADP/solver/grassman.py"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    out = {
        "rows": sum(map(len, pilots.values())),
        "failures": failures,
        "stops": {f"{m}:{s}": n for (m, s), n in sorted(stops.items())},
        "lsmr": {f"{m}:{k}": v for (m, k), v in sorted(lsmr.items())},
        "integrity": {
            "input_pairs": len(inputs),
            "matched_method_seeds": True,
            "snapshots_verified": True,
            "source_reference_equal": source_equal,
            "live_hash_equal": live == ref,
            "live_grassman_sha256": live,
            "HEAD_clean": head_clean,
        },
    }
    for method in METHODS:
        pairs = []
        for rows in pilots.values():
            by = {(r["case"], r["method"]): r for r in rows}
            for case in CASES:
                if (case, method) not in by:
                    continue
                c, b, r = by[case, method], by[case, "baseline"], by[case, "MEANMAVE"]
                pairs.append(
                    (
                        case,
                        c["projector_distance"] / b["projector_distance"],
                        c["fit_seconds"] / b["fit_seconds"],
                        c["peak_rss_mib"] / b["peak_rss_mib"],
                        c["projector_distance"] / r["projector_distance"],
                        b["projector_distance"] / r["projector_distance"],
                        c["projector_distance"] - b["projector_distance"],
                    )
                )
        d50 = next((p for p in pairs if pairs and p[0] == "d-50"), None)
        out[method] = {
            "n": len(pairs),
            "EDR_geomean_vs_base": geo([p[1] for p in pairs]),
            "time_median_vs_base": statistics.median(p[2] for p in pairs)
            if pairs
            else None,
            "time_p90_vs_base": q90([p[2] for p in pairs]),
            "RSS_median_vs_base": statistics.median(p[3] for p in pairs)
            if pairs
            else None,
            "EDR_geomean_vs_R": geo([p[4] for p in pairs]),
            "baseline_geomean_vs_R": geo([p[5] for p in pairs]),
            "d50_EDR_ratio": d50[1] if d50 else None,
            "d50_delta": d50[6] if d50 else None,
            "d50_time_ratio": d50[2] if d50 else None,
            "cases": pairs,
        }
    return out


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, sort_keys=True))
