"""Create an auditable summary from the paired profiling artifacts."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


baseline = {r["seed"]: r for r in rows(ROOT / "results/runs.jsonl") if r["mode"] == "baseline"}
profiled = {r["seed"]: r for r in rows(ROOT / "results/runs.jsonl") if r["mode"] == "profiled"}
timed = {r["seed"]: r for r in rows(ROOT / "lightweight_results/runs.jsonl")}
seeds = sorted(baseline)

for seed in seeds:
    ref, prof, light = baseline[seed], profiled[seed], timed[seed]
    assert ref["status"] == prof["status"] == light["status"] == "ok"
    assert ref["basis_hash"] == prof["basis_hash"] == light["basis_hash"]
    assert ref["diagnostics_hash"] == prof["diagnostics_hash"] == light["diagnostics_hash"]
    assert ref["input_hashes"] == prof["input_hashes"] == light["input_hashes"]
    assert not any(key.startswith("other_") for key in light["stages"])

    stages = light["stages"]
    stage_sum = sum(v["exclusive_sec"] for k, v in stages.items() if k != "solve")
    assert abs(stage_sum - light["solver_wall_sec"]) < 1e-8


def local_profile(row: dict) -> float:
    s = row["stages"]
    return sum(s.get(k, {}).get("exclusive_sec", 0.0) for k in ("_profile", "profile_svd", "profile_lstsq"))


def stage(row: dict, *keys: str) -> float:
    return sum(row["stages"].get(k, {}).get("exclusive_sec", 0.0) for k in keys)


categories = {
    "local_profile": lambda r: local_profile(r),
    "project_gradient": lambda r: stage(r, "_project", "_gradient"),
    "core_jacobian": lambda r: stage(r, "_core_jacobian"),
    "gn_lstsq": lambda r: stage(r, "gn_lstsq"),
    "polar": lambda r: stage(r, "_polar"),
    "geometry_qr": lambda r: stage(r, "geometry_qr"),
    "geometry_svd": lambda r: stage(r, "geometry_svd"),
    "remainder": lambda r: stage(r, "solve_remainder"),
}
stage_totals = {name: sum(fn(timed[s]) for s in seeds) for name, fn in categories.items()}
solver_total = sum(timed[s]["solver_wall_sec"] for s in seeds)
timing_rows = []
for seed in seeds:
    b, p, t = baseline[seed], profiled[seed], timed[seed]
    timing_rows.append({
        "seed": seed,
        "fit_sec": b["fit_time_sec"],
        "solver_sec_baseline": b["solver_wall_sec"],
        "solver_sec_profiled": p["solver_wall_sec"],
        "solver_sec_timed": t["solver_wall_sec"],
        "cprofile_overhead_pct": 100 * (p["solver_wall_sec"] / b["solver_wall_sec"] - 1),
        "light_timer_overhead_pct": 100 * (t["solver_wall_sec"] / b["solver_wall_sec"] - 1),
        "solver_fraction_of_full_fit_pct": 100 * b["solver_wall_sec"] / b["fit_time_sec"],
        "quality": b["projector_distance_to_truth"],
        "stop_reason": b["fit_stop_reason"],
        "outer_iterations": b["outer_iterations"],
        "solver_calls": b["solver_calls"],
        "rank_guard_fallbacks": b["aggregate_diagnostics"]["rank_guard_fallbacks"],
        "profile_evaluations": b["aggregate_diagnostics"]["profile_evaluations"],
        "gn_solves": b["aggregate_diagnostics"]["gn_solves"],
        "peak_rss_kib": b["peak_rss_kib"],
    })

summary = {
    "seeds": seeds,
    "all_pairs_exact_input_basis_diagnostics_match": True,
    "protocol": {
        "cprofile": json.loads((ROOT / "results/protocol.json").read_text()),
        "lightweight": json.loads((ROOT / "lightweight_results/protocol.json").read_text()),
    },
    "per_seed": timing_rows,
    "lightweight_stage_seconds_pooled": stage_totals,
    "lightweight_stage_share_of_solver_pooled_pct": {k: 100 * v / solver_total for k, v in stage_totals.items()},
    "local_profile_share_per_seed_pct": {s: 100 * local_profile(timed[s]) / timed[s]["solver_wall_sec"] for s in seeds},
    "solver_seconds_median": {
        "baseline": statistics.median(baseline[s]["solver_wall_sec"] for s in seeds),
        "cprofiled": statistics.median(profiled[s]["solver_wall_sec"] for s in seeds),
        "light_timed": statistics.median(timed[s]["solver_wall_sec"] for s in seeds),
    },
    "median_cprofile_overhead_pct": statistics.median(r["cprofile_overhead_pct"] for r in timing_rows),
    "median_light_timer_overhead_pct": statistics.median(r["light_timer_overhead_pct"] for r in timing_rows),
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

lines = [
    "# Grassmann solve stage profile (2026-10-06)",
    "",
    "## Protocol",
    "",
    "Existing Spokoiny full-fit generator; n=800, d=50, m=2, J=800, P=10, three seeds (0,1,2), float64, max_steps=5. Each subprocess uses one BLAS/OMP/MKL thread. Baseline and cProfiled rows are paired sequentially; a separate lightweight timer-only pass reused the exact baselines. Production solver source was not changed.",
    "",
    "Raw JSONL, protocol metadata, and per-seed `.prof` files are in `results/` and `lightweight_results/`. The initial failed smoke serialization rows are retained in `failures/`; the first timer-only pass with out-of-solve linalg counted is retained in `failures/lightweight_scope_miss/` and excluded from analysis.",
    "",
    "## Paired results",
    "",
    "| Seed | Full fit s (baseline) | Solver s (baseline) | Solver s (timer only) | Timer delta | Solver / fit | Quality | Stop | RSS MiB | Rank fallbacks | Profile evals | GN solves |",
    "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
]
for r in timing_rows:
    lines.append(f"| {r['seed']} | {r['fit_sec']:.2f} | {r['solver_sec_baseline']:.3f} | {r['solver_sec_timed']:.3f} | {r['light_timer_overhead_pct']:+.1f}% | {r['solver_fraction_of_full_fit_pct']:.1f}% | {r['quality']:.5f} | {r['stop_reason']} | {r['peak_rss_kib']/1024:.1f} | {r['rank_guard_fallbacks']} | {r['profile_evaluations']} | {r['gn_solves']} |")

lines += [
    "",
    f"The basis and complete solver-diagnostics hashes match exactly for baseline, cProfiled, and timer-only runs at every seed. Timer-only solver-time deltas have median {summary['median_light_timer_overhead_pct']:+.1f}% (per seed shown above); cProfile overhead has median {summary['median_cprofile_overhead_pct']:+.1f}%, so stage ranking uses timer-only rows.",
    "",
    "## Lightweight disjoint stage accounting",
    "",
    "Pooled shares use the sum of stage seconds divided by total timer-only solver seconds across all three seeds. The per-seed local-profile share is also shown because fallback-heavy seeds vary substantially.",
    "",
    "| Disjoint bucket | Pooled seconds | Pooled solver share |",
    "|---|---:|---:|",
]
for name, seconds in stage_totals.items():
    lines.append(f"| {name} | {seconds:.3f} | {100*seconds/solver_total:.1f}% |")
lines += [
    "",
    f"Local-profile share by seed: {', '.join(f'{s}: {summary['local_profile_share_per_seed_pct'][s]:.1f}%' for s in seeds)}; pooled share {100*stage_totals['local_profile']/solver_total:.1f}%. `project_gradient` combines `_project` and `_gradient`; the geometry QR/SVD buckets are separate. `remainder` covers Python/controller, line-search, and instrumentation work not assigned to named child stages; it is not a pure line-search measurement.",
    "",
    "## Interpretation and limits",
    "",
    "Local profiling/refitting is the largest pooled stage at 47.3%, but its per-seed share ranges from about one third to 57.8%. Seed 2 has 605 rank-guard fallbacks, 8,373 profile evaluations, and only 205 GN solves; its cProfile artifact shows repeated local profile SVD and bounded scalar minimization as the dominant fallback path. Seeds 0/1 have 190/0 rank fallbacks. This variation is solver-path dependent, so the pooled share is descriptive of these three seeds rather than universal.",
    "",
    "Across these full fits, solver time is 10–23% of fit time; most runtime remains outside `solve`. All fits stop at `h_min` after 162 outer iterations; the final inner call reports `max_steps` with `converged=false`. Quality and stop state were unchanged across paired instrumentation modes. RSS is process peak RSS and includes Python/runtime overhead.",
    "",
    "CProfile timings are retained only as a diagnostic artifact because instrumentation increased solver wall time materially (median 26.7%). No optimization or estimator change was made.",
    "",
    "## Provenance",
    "",
    f"Solver source SHA-256: `{summary['protocol']['lightweight']['source_sha256']}`. Timer-only harness SHA-256: `{summary['protocol']['lightweight']['harness_sha256']}`. cProfile harness SHA-256: `{summary['protocol']['cprofile']['harness_sha256']}`. Runtime, command, thread settings, commit, and dirty paths are recorded in both protocol JSON files.",
    "",
]
(ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
