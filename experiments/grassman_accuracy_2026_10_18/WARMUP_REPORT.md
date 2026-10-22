# Warmup development result

**Rejected at the frozen development gate. Validation was not run.** The
candidate is faster, but misses the required paired EDR geometric mean and
has one severe individual EDR regression. No tuning or seed reuse is justified.

## Audit and frozen settings

- 90/90 fits completed; all statuses `ok`; 30/30 case-seed blocks pair across
  `ADP-Grassmann`, `ADP-Grassmann-warmup`, and `MAVE-package`.
- All 30 pairs pass data, X, y, and method-seed hash checks. No bad blocks.
  The manifest fingerprint is
  `bffd31ca0efaa122ce2599b5d41b969a861eb1188bd14580b1a4d77660dfd83d`;
  93 source hashes are retained in the analysis JSON.
- Seeds 101800–101809; n=500, m=2, dimensions 5/20/50; one thread;
  max steps 10, smooth cap 5, tolerance 1e-6, local ridge 0.
- Environment: Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1, R 4.6.1,
  MAVE 1.3.12. Thread variables were all 1.

## Gates

| Gate | Result | Decision |
|---|---:|---|
| Paired EDR geometric mean ratio, limit ≤0.95 | 1.01325 (95% paired bootstrap CI 0.83220–1.34436) | Fail |
| Worst individual paired EDR ratio, limit ≤2 | 28.75183 | Fail |
| Per-case median EDR delta, limit ≤0.005 | d-5 +0.000037; d-20 +0.000313; d-50 −0.050693 | Pass |
| Fit-time ratio median / p90, limits ≤1.10 / 1.25 | 0.7551 / 0.9099 | Pass |
| Median RSS ratio, limit ≤1.10 | 1.0078 (88.78 / 88.09 MiB) | Pass |
| Maximum candidate orthogonality error, limit ≤1e-10 | 3.40e-15 | Pass |
| Complete successful pairs | 30/30; no failures or hash mismatches | Pass |

The fit-time ratio bootstrap 95% interval is 0.6915–0.7804. Across the 30
pairs, candidate EDR is lower in 19 and higher in 11.

## Per-case medians

| Case | Paired EDR ratio geomean [95% CI] | Median EDR delta | EDR: Grassmann / warmup / R MEANMAVE | Median fit seconds: Grassmann / warmup / R MEANMAVE | Time ratio median / p90 |
|---|---:|---:|---:|---:|---:|
| d-5 | 1.4151 [1.0000, 2.7766] | +0.000037 | 0.03529 / 0.03904 / 0.02678 | 0.3149 / 0.2043 / 0.1485 | 0.6520 / 0.7283 |
| d-20 | 1.0164 [0.9919, 1.0468] | +0.000313 | 0.10257 / 0.10374 / 0.08117 | 0.5376 / 0.4093 / 1.0750 | 0.7526 / 0.8719 |
| d-50 | 0.7233 [0.5567, 0.8585] | −0.050693 | 0.21688 / 0.17326 / 0.13503 | 0.8941 / 0.7412 / 4.4795 | 0.8335 / 0.9646 |

## Individual outlier and solver diagnostics

The worst pair is d-5, seed 101809: warmup EDR 0.559407 versus baseline
0.019456 (ratio 28.7518); R MEANMAVE EDR is 0.020327. The paired method
seed is 1353470310, and data/X/y hashes match across all three methods. Warmup
fit time is 0.2345 s versus baseline 0.3654 s.

Both ADP fits stop at outer `h_min`; neither has an inner certificate. Warmup
inner calls accept 10 steps at each scale: the first stops `max_steps`
(profile objective 74.8964, gradient 0.15181), the second
`rank_boundary_stationary` (objective 8.90723, gradient 8.96e-7), with
stationarity inapplicable for both. Baseline calls stop `max_steps` after 5
steps each. These diagnostics describe the failure but do not establish its
cause.

Across all candidate fits: 1,329 accepted steps, 130 rank-guard fallbacks,
1,199 GN solves, 509 local rank-loss calls, 10/140 stationarity-applicable
calls, and zero certified inner calls. Baseline: 700 accepted steps, 650
rank-guard fallbacks, 50 GN solves, 512 local rank-loss calls, and zero
certified inner calls. All 30 candidate outer fits stop at `h_min`.

Full per-seed pairs, hashes, failures, diagnostics, and bootstrap values are in
`benchmark_outputs/r_sdr/grassman-warmup-dev-2026-10-18/analysis_summary.json`.
