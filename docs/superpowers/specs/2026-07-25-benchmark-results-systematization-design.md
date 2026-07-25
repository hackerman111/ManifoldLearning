# Benchmark Results Systematization Design

## Goal

Systematize the results in:

- `benchmark_outputs/common_edr/exp_2`;
- `benchmark_outputs/small_ridge_comparison`;
- `benchmark_outputs/ridge_eta_comparison`;
- `benchmark_outputs/ridge_main`.

Experiments 3–6 under `benchmark_outputs/common_edr` are explicitly excluded
from every table, aggregate, and conclusion.

## Selected approach

Create one concise Markdown report backed by normalized CSV tables. Recompute
the selected aggregates from the authoritative row-level files rather than
copying the existing summary files. Cross-check the recomputed values against
the saved summaries where equivalent fields exist.

This keeps the report readable while retaining machine-readable tables for
later plotting, LaTeX export, or additional statistical analysis.

Two narrower alternatives were rejected:

1. using only the existing `model_summary.csv` and `comparison_summary.csv`,
   because they omit initialization quality and several useful paired
   comparisons;
2. writing four independent reports, because the ridge directories overlap and
   their common cases have identical deterministic numerical results.

## Inputs and units of analysis

For `common_edr`, use `exp_2/run_summary.csv` and `exp_2/series.csv`.
The unit of analysis is one `(parameter case, seed)` run.

For ridge comparisons, use `runs.csv`, `comparisons.csv`, and `manifest.json`
from each directory. The unit of model-level analysis is one
`(case_id, seed, model)` run. The unit of baseline-relative analysis is one
paired `(case_id, seed, candidate_model)` comparison.

Overlapping ridge runs remain attributed to their source series in the
per-series tables. They are not pooled into a larger pseudo-independent sample.
Numerical equality of overlapping cases is reported as a reproducibility
check.

## Output tables

The report and CSV package contain four main tables:

1. **Series passport**
   - source series;
   - experiment;
   - models;
   - dimensions `d`;
   - requested `n/d`;
   - seed range and count;
   - number of cases and runs;
   - baseline model;
   - completion status.

2. **`common_edr/exp_2` summary**
   - total, successful, nonconverged, and numerical-failure run counts;
   - convergence rate;
   - median and IQR of `cosine_abs`, `fit_wall_time_sec`,
     `algorithm_rss_peak_delta_mib`, outer iterations, and total inner
     iterations;
   - fixed ADP parameters: outer and inner step limits, tolerance, minimum
     neighbors, ridge, initial beta mode, local-mass mode, and direction count.

3. **Ridge model summary**
   - source series and model;
   - run and success counts;
   - median and IQR of initial direction quality
     `beta_ref_cosine_abs`, final `cosine_abs`, fit time, peak RSS increase,
     and objective;
   - median final-minus-initial cosine.

4. **Ridge cell and baseline-relative summary**
   - source series, `d`, requested `n/d`, and model;
   - run count and median/IQR model metrics;
   - for candidates, paired median/IQR time speedup, memory ratio, cosine gap,
     final-direction agreement, and objective gap relative to `e1_control`;
   - paired quality win rate;
   - numerical-equivalence rate.

## Statistical conventions

- Report medians with the 25th and 75th percentiles.
- Do not interpret technical `nonconverged` as poor directional quality;
  present status and `cosine_abs` separately.
- Use `rss_peak_delta_mib` for ridge memory and
  `algorithm_rss_peak_delta_mib` for `common_edr`; label both as MiB.
- Treat `time_speedup > 1` as the candidate being faster than `e1_control`.
- Treat `peak_delta_memory_ratio < 1` as the candidate using less incremental
  peak memory.
- Compute quality win rate as the share of valid pairs with
  `candidate_cosine_abs > baseline_cosine_abs`.
- Preserve requested `n/d` in grouping labels and retain actual `n/d` as a
  diagnostic field where rounding changes it.
- Do not pool medians across source series when machine load and run size differ.

## Validation

Validation must confirm:

- only `common_edr/exp_2` is read;
- output row counts match the expected grouping keys;
- all ridge runs have unique `(case_id, seed, model)` keys;
- all paired rows have matching baseline and candidate cases/seeds;
- recomputed common fields agree with existing saved summaries within floating
  point tolerance;
- the three missing `common_edr/exp_6` runs and all experiment 3–6 results are
  absent from outputs and narrative conclusions;
- overlapping ridge cases have identical input fingerprints and deterministic
  numerical results across source directories.

## Deliverables

- one Markdown report under `benchmark_outputs/results_summary/`;
- normalized CSV tables under the same directory;
- one small reproducible analysis script that reads the authoritative source
  files and regenerates every table and validation check.

