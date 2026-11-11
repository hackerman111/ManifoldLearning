# Relative intrinsic GN: frozen development

Raw: `benchmarks/results/outputs/r_sdr/grassman-relative-dev-repaired-2026-10-22/`.
Seeds105800..105809, d5/d20/d50,n500,m2,original5/relative10/actualR MEANMAVE.
One BLAS thread, isolated process fit time/RSS. Algorithm/gates frozen before data.

90/90 successful fits;30 paired blocks; no duplicate/missing/hash mismatches. 22 quality wins.
EDR geometric ratio 0.898376, bootstrap95 [0.844245030839548, 0.9489031209291838].
Time paired median/p90 [0.6505802577317027, 0.7903783357046622]; median95CI [0.6061489065447365, 0.6944706813940965].
RSS ratio 1.005236; orthogonality max 4.22e-15.
Worst EDR ratio 1.077314, d-5/105807.

| Case | Original median EDR/time(s) | Relative median EDR/time(s) | R MEANMAVE median EDR/time(s) |
|---|---:|---:|---:|
| d-5 | 0.045575 / 0.3415 | 0.047308 / 0.1916 | 0.031389 / 0.1515 |
| d-20 | 0.107713 / 0.5967 | 0.105483 / 0.3826 | 0.078715 / 1.0770 |
| d-50 | 0.208145 / 0.9166 | 0.163961 / 0.6975 | 0.137017 / 4.5290 |

Frozen gates:
- complete_successful_pairs: PASS
- edr_geomean: PASS
- case_median_edr_delta: PASS
- time_ratio_median: PASS
- time_ratio_p90: PASS
- median_rss_ratio: PASS
- finite_and_orthogonal: PASS
- max_individual_edr_ratio: PASS

Decision: development passes all gates; unchanged106800 validation required.
All ADP outer schedules stop at h_min; zero certified inner calls in both
methods. Finite orthogonal output and reduced loss do not prove convergence.
R remains more accurate in the displayed case medians; speed/quality separate.

Original invalid launch retained separately under grassman-relative-dev-2026-10-22:
63 rows,21candidate errors before optimization (duplicate lambda_prox). Repair
removes only duplicate runner forwarding, inherited damping still.05. Prototype
and production hashes identical; no algorithm/budget/gate retuning.
Settings, math and seed boundaries: RELATIVE_GN_PROTOCOL.md / RELATIVE_GN_MATH.md.
Full source/config/environment fingerprints and raw failures are preserved.
