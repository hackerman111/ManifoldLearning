# Conjugate rank-guard development result

Frozen development stage only: 30 paired blocks, d=5/20/50, seeds 103800–103809; 90/90 fits succeeded. All 30 blocks have exactly the three planned methods and matching data, X, Y, and method-seed hashes; no duplicate/malformed blocks or failures. Validation remains untouched.

## Frozen gates

| Gate | Result | Observed |
|---|---:|---:|
| Complete successful pairs | PASS | 30/30 |
| EDR ratio geometric mean <= 0.95 | PASS | 0.9271; bootstrap 95% CI [0.8890, 0.9628] |
| Every case median EDR delta <= 0.005 | PASS | d-5 -0.000132; d-20 -0.003071; d-50 -0.032283 |
| Paired time ratio median <= 1.10 | FAIL | 1.1322; bootstrap median CI [1.0283, 1.1856] |
| Paired time ratio p90 <= 1.25 | FAIL | 1.3975 |
| Median RSS ratio <= 1.10 | PASS | 0.9981 (candidate 87.93 MiB; baseline 88.10 MiB) |
| Finite and orthogonal <= 1e-10 | PASS | max orthogonality 1.32e-15 |
| Maximum individual EDR ratio <= 2 | PASS | 1.1245 |

Overall candidate/baseline paired time ratios across cases are median 1.1322 and p90 1.3975. Candidate EDR was lower in 26/30 pairs and higher in 4/30; no ties. MAVE-package EDR was lower than baseline in 30/30 pairs.

## Per-case medians

EDR is projector distance; fit time is seconds. Stops show outer stop reason counts; inner-certified is count of certified inner solver calls.

| Case | Method | EDR median | Fit median (s) | Stop counts | Inner certified | Rank guard fallbacks |
|---|---|---:|---:|---|---:|---:|
| d-5 | ADP-Grassmann | 0.042052 | 0.301405 | h_min 10 | 0 | 100 |
| d-5 | ADP-Grassmann-conjugate | 0.041929 | 0.342653 | h_min 10 | 0 | 200 |
| d-5 | MAVE-package | 0.031920 | 0.150000 | package_not_exposed 10 | n/a | 0 |
| d-20 | ADP-Grassmann | 0.107285 | 0.544484 | h_min 10 | 0 | 240 |
| d-20 | ADP-Grassmann-conjugate | 0.104309 | 0.644584 | h_min 10 | 0 | 480 |
| d-20 | MAVE-package | 0.079911 | 1.086000 | package_not_exposed 10 | n/a | 0 |
| d-50 | ADP-Grassmann | 0.222222 | 0.818355 | h_min 10 | 0 | 265 |
| d-50 | ADP-Grassmann-conjugate | 0.186496 | 0.896942 | h_min 10 | 0 | 530 |
| d-50 | MAVE-package | 0.143953 | 4.530000 | package_not_exposed 10 | n/a | 0 |

## Profile-history diagnostic

Relative final profile drop is `(first - last) / max(1, abs(first))`, summarized over calls with a profile history. Budget-5 group: 19 calls; quantiles [min, p25, median, p75, p90, max] = [0.04705, 0.11016, 0.13809, 0.19629, 0.22485, 0.28363], rank-guard fallbacks 0. Budget-10 group: 121 calls; quantiles [0.00786, 0.06461, 0.12767, 0.19331, 0.33129, 0.95720], rank-guard fallbacks 1210. These are descriptive call-level summaries, not a paired convergence comparison.

## Provenance

Manifest fingerprint: `d9eb70bfb025afbcf80cebdb92873ee7417fb235b1be726d9ee82fbd6b39be08`.

| Source | SHA-256 |
|---|---|
| `ADP/solver/grassman.py` | `6d5aa753da9d9c8532545fa7519e63ba479c0f4082a8a6d52358cd4c66af72a0` |
| `ADP/solver/grassman_exp.py` | `5c96b145af109102ba705a6962d86dab2421b6a87a3106807e34fcc812db8071` |
| `experiments/grassman_accuracy_2026_10_18/backtracking.py` | `816c5d739e2cd914eaffae034dca892d9d74224e6045caa40d760d4e0486061e` |
| `experiments/grassman_accuracy_2026_10_18/conjugate_guard.py` | `4f7f21afe34e4b06b1a4a2190dc460e5f9d4165219800b90f7cfc0a15cec88db` |

Full per-seed outcomes, paired hashes, failures, and source/environment fingerprints are in `benchmark_outputs/r_sdr/grassman-conjugate-dev-2026-10-22/analysis_summary.json` and `manifest.json`. The failed timing gates reject this candidate under the frozen development protocol; validation is therefore not run.
