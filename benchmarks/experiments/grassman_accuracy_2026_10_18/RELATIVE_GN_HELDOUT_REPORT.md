# Relative intrinsic GN: independent validation

Raw: `benchmarks/results/outputs/r_sdr/grassman-relative-heldout-2026-10-22/`.
Unchanged frozen settings;106800..106809,d5/d20/d50,n500,m2,one thread.
Development and validation source fingerprint dictionaries are identical.

90/90 successful fits;30 matched blocks; no duplicate/missing/hash failures; 20 quality wins.
EDRgeo 0.923729, paired bootstrap95 [0.8866219314118116, 0.9564945036219509].
Time median/p90 [0.6535720918613088, 0.9932971623766564]; median95CI [0.6156167347201783, 0.7893223984762936].
RSS ratio 1.002106; orthogonality max 3.59e-15.
Worst EDR ratio 1.036418 at d-5/106807.

| Case | Original EDR/time(s) | Relative EDR/time(s) | R MEANMAVE EDR/time(s) |
|---|---:|---:|---:|
| d-5 | 0.049694 / 0.3191 | 0.049903 / 0.1859 | 0.031150 / 0.1505 |
| d-20 | 0.110213 / 0.6154 | 0.100482 / 0.3865 | 0.075923 / 1.0825 |
| d-50 | 0.200713 / 0.7229 | 0.171312 / 0.6620 | 0.141289 / 4.4765 |

Frozen gates:
- complete_successful_pairs: PASS
- edr_geomean: PASS
- case_median_edr_delta: PASS
- time_ratio_median: PASS
- time_ratio_p90: PASS
- median_rss_ratio: PASS
- finite_and_orthogonal: PASS
- max_individual_edr_ratio: PASS

Decision: all prespecified validation gates pass, as did development.
Accept this configuration as an explicit option for the measured workload.
The 95% interval includes improvements smaller than5%; the gate was defined
on the point estimate. No universal5% recovery guarantee follows.
d5 quality is almost unchanged; most benefit is at d20/d50. R remains more
accurate in case medians. Outer h_min stops and zero certified inner calls
do not establish convergence. Source snapshots, all raw outputs and errors
remain available; the invalid first development launch is excluded explicitly.

Next: opt-in production port with exact baseline/prototype replay, no default change.
