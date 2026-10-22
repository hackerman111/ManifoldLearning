# Grassmann versus MAVE on the main R-SDR benchmark

Task `grassman-full-rsdr-2026-10-18`; status **active**.

Goal: run the main `full` R-SDR catalog for `ADP-Grassmann`, ordinary MAVE, and package `MAVE::MEANMAVE`, using the same 10 seeds per case. The main catalog has 50 cases; expected total is 1,500 fits.

Non-goals: rerun SIR/dMAVE/rMAVE/refined MAVE or LSMR; tune from oracle truth; change estimators or production defaults; pool results with the prior `multi-rank` experiment.

Known evidence: the runner already exposes `ADP-Grassmann` as an opt-in variant. The prior multi-rank experiment used the documented five-step budget and had complete paired inputs, but Grassmann calls reached their cap and mostly took the rank-one fallback. The full catalog ranges to `n=10000,d=1000,m<=3`; runtime/memory may exceed the compact run.

Hypothesis: runtime and recovery differences can vary substantially with `n,d` and rank; report results per case and aggregate only where scientifically comparable, keeping failures in denominators.

Invariants: identical X/Y/truth for every method and seed; known target rank; 10 repetitions per case; frozen `ADP-Grassmann` five-step budget, `local_ridge=0`; keep failure rows and solver diagnostics; all output goes to a fresh directory.

Change class: `APPROXIMATE` solver variant; this task changes experiment coverage only, not the estimator or production default.

Exact read set: `agent-notes/contracts/research.md`, `agent-notes/ADP/r-sdr-comparison.md`, `experiments/r_sdr/{README.md,__main__.py,design.py,runner.py,report.py}`, and the existing multi-rank artifacts for protocol context.

## Work units

- [x] Run a dry-run and verify exactly 50 cases × 10 seeds × 3 methods = 1,500 fits; confirm the output directory is fresh and inspect resource limits. The fresh target was `benchmark_outputs/r_sdr/full-grassman-vs-mave-2026-10-18/`; per-fit limits are 300 s and 4096 MiB.
- [ ] Run the full benchmark with `ADP-Grassmann,MAVE,MAVE-package`, 10 paired runs, and `--grassman-steps 5`; preserve journal, failures, and logs.
- [ ] Analyze completed output; verify per-seed input hashes, fit counts, status/failure denominators, paired comparisons, memory and stopping diagnostics.
- [ ] Summarize results per case and across the main catalog with limits; update experiment guide, `STATE.md`, and this plan.

Verification: focused R-SDR tests and CLI dry-run before the long run; completion manifest and `runs.jsonl`/`paired.csv` audit after it; `--analyze`, scoped Ruff/format where code changes, and `git diff --check`.

Stop conditions: changed source/environment fingerprint, mismatched paired data, invalid output, repeated systematic Grassmann failures, or resource consumption beyond the runner's configured limits. Do not tune the five-step budget after seeing quality results.

Checkpoint 2026-10-18: superseded by solver-accuracy research; prior catalog is incomplete. Preserve its source snapshot and journal; do not resume against changed sources.
