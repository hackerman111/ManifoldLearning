# Intrinsic development gate

**Decision: PASS all frozen selection gates.** This licenses the untouched
validation run under the frozen protocol; it does not promote a production
default or establish a global recovery claim.

- Frozen output: `benchmarks/results/outputs/r_sdr/grassman-intrinsic-dev-2026-10-18`;
  90/90 successful fits, 30/30 paired blocks, no failures or bad hash blocks.
- Pairing audit: 30 unique data, X-CSV, and Y-CSV hashes; method seeds match
  within each pair. All 91 source hashes match the frozen snapshot.
- Manifest fingerprint `f88c4be2…2c54ce`; `intrinsic.py` SHA-256
  `33824ff1…de31e588`. Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1,
  R 4.6.1 / MAVE 1.3.12; BLAS thread limits are 1.
- Aggregate paired EDR geometric ratio **0.89504** (bootstrap 95% CI
  0.84340–0.94238; limit <=0.95); 24 wins / 6 losses. Worst regression:
  d-20 seed 99806, ratio 1.11187, EDR delta +0.00987.
- Aggregate fit-time ratio median/p90 **0.66773/1.00834** (limits
  1.10/1.25); median RSS ratio **1.00485** (88.590/88.162 MiB); maximum
  candidate orthogonality **3.70e-15**. No paired time ratio exceeds 1.25.

| Case | EDR ratio geo (95% CI) | Median EDR delta | Time ratio median/p90 | Baseline → intrinsic EDR median | Baseline → intrinsic fit median/p90 (s) | R MEANMAVE EDR; fit median/p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| d-5 | 1.00004 (0.99748–1.00313) | -0.000032 | 0.57167/0.69207 | 0.04881 → 0.04876 | 0.2994/0.3322 → 0.1775/0.1889 | 0.03141; 0.148/0.150 |
| d-20 | 0.93835 (0.85146–1.00320) | -0.003518 | 0.63248/1.14148 | 0.10944 → 0.10051 | 0.5977/0.6774 → 0.3816/0.3878 | 0.08304; 1.077/1.0909 |
| d-50 | 0.76408 (0.70593–0.80997) | -0.048982 | 0.85111/0.95910 | 0.23647 → 0.18441 | 0.8329/0.9352 → 0.7021/0.7201 | 0.14321; 4.513/4.5692 |

- Both Grassmann methods stop outer scheduling at `h_min`; candidate inner
  certified calls: 0. Candidate accepted steps / rank-guard fallbacks /
  GN solves / local-rank-loss totals: d-5 175/2/173/173; d-20 500/0/500/195;
  d-50 660/1/659/128. Stationarity was applicable for 8 d-50 calls only.
- R reports `package_not_exposed` for all ten runs per case. Full per-seed
  results, hashes, stops, and bootstrap summaries are in `analysis_summary.json`.

## Untouched validation

**Decision: FAIL the EDR gate.** Validation `100800–100809` completed 90/90
successful fits and 30/30 paired blocks. The aggregate EDR ratio is **0.98546**
(bootstrap 95% CI 0.87116–1.18394; required <=0.95). Case median-delta,
runtime, RSS, and orthogonality gates pass: time median/p90 0.65760/0.91733,
RSS ratio 1.00310 (88.488/88.215 MiB), max orthogonality 2.79e-15.

- Pairing: 30 unique data/X/Y hashes, consistent method seeds, no bad blocks;
  all 91 frozen source hashes match. Fingerprint `702a4a7d…6841bdf`;
  `intrinsic.py` SHA-256 is unchanged (`33824ff1…de31e588`). Environment:
  Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1, R 4.6.1/MAVE 1.3.12, one thread.
- There are 21 EDR wins and 9 losses. The worst is d-5 seed 100803: candidate
  EDR 0.58715 vs baseline 0.05831 (10.069x; delta +0.52884). Both rows are
  status `ok`, share the same input hashes, and have orthonormal bases.

| Case | EDR ratio geo (95% CI) | Median EDR delta | Time ratio median/p90 | Baseline → intrinsic EDR median | R MEANMAVE EDR; fit median/p90 (s) |
|---|---:|---:|---:|---:|---:|
| d-5 | 1.26071 (0.99969–2.00140) | +0.000027 | 0.55885/0.59176 | 0.04742 → 0.04741 | 0.03117; 0.150/0.1512 |
| d-20 | 0.90807 (0.79027–0.98793) | -0.003529 | 0.65938/0.76928 | 0.10736 → 0.10003 | 0.08462; 1.0675/1.0764 |
| d-50 | 0.83596 (0.78958–0.87690) | -0.031507 | 0.88042/0.99070 | 0.19427 → 0.16541 | 0.14354; 4.4785/4.5061 |

- Outlier diagnostics: both fits have 2 outer iterations and stop at `h_min`;
  candidate selected iteration 0/error 52.1345, baseline iteration 1/error
  36.5748. First inner final profile objective is 50.6433 candidate vs 8.1617
  baseline: 10 vs 5 accepted steps, 7 local rank-loss centers each, 10
  intrinsic proposals/0 fallbacks vs 0 GN solves/5 rank-guard fallbacks, and
  gradients 0.15199 vs 0.02485. Second inner objectives nearly match (8.0197
  vs 7.9684). All calls stop at `max_steps`; rank-boundary stationarity is
  inapplicable and neither fit has inner certification. No worker failure was
  recorded. Full outer trace was not persisted, so the precise cause is unknown.
- Validation outlier and EDR gate failure stand despite passing aggregate time,
  memory, orthogonality, and case-median-delta gates. Keep this regression
  visible; the saved JSON includes all paired outliers and raw per-seed data.
