# Quadratic development gate

**Decision: FAIL.** The frozen development run misses the aggregate EDR gate;
validation seed range 98800–98809 remains untouched. Do not tune on these seeds.

- Frozen run: `benchmark_outputs/r_sdr/grassman-quadratic-dev-2026-10-18`;
  90/90 successful fits, 30/30 paired blocks, no failures or hash mismatches.
- Pairing audit: 30 unique data, X-CSV, and Y-CSV hashes; method seeds agree
  within each block. All 89 frozen source hashes match the snapshot.
- Manifest fingerprint `ac238b64…ed97d1`; candidate source hash
  `quadratic.py` = `bf85af3b…d262abff`. Python 3.13.12, NumPy 2.5.2,
  SciPy 1.18.1, R 4.6.1 / MAVE 1.3.12; all worker thread limits are 1.
- Aggregate paired EDR ratio geometric mean **0.96196** (bootstrap 95% CI
  0.92384–0.99003; required <=0.95): **fail**. 20 pair wins / 10 losses;
  worst regression is d-20 seed 97804, ratio 1.06501 (EDR delta +0.00490).
- Other aggregate gates pass: fit-time ratio median/p90 **0.82290/1.09843**
  (limits 1.10/1.25); median RSS ratio **1.00193** (88.295/88.125 MiB);
  max candidate orthogonality **9.53e-16**. d-20 has time-ratio p90 1.29828;
  seed 97803 is 1.31403x and 97808 is 1.29653x.

| Case | EDR ratio geo (95% CI) | Median EDR delta | Time ratio median/p90 | Baseline → quadratic EDR median | Baseline → quadratic fit median/p90 (s) | R MEANMAVE EDR; fit median/p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| d-5 | 1.00386 (0.99746–1.01050) | +0.000032 | 0.68709/0.79894 | 0.03658 → 0.03653 | 0.3275/0.3800 → 0.2162/0.2277 | 0.02402; 0.1515/0.1533 |
| d-20 | 0.98948 (0.96187–1.01774) | -0.000678 | 0.93479/1.29828 | 0.09376 → 0.09492 | 0.4604/0.6453 → 0.4258/0.4468 | 0.07273; 1.0820/1.1074 |
| d-50 | 0.89617 (0.81407–0.95915) | -0.009838 | 0.91292/1.02081 | 0.20072 → 0.17518 | 0.8286/0.9346 → 0.7481/0.7807 | 0.14249; 4.4790/4.4897 |

- All Grassmann outer stops are `h_min`; candidate inner certified calls: 0.
  Candidate accepted steps / rank-guard fallbacks / GN solves / local rank-loss
  counts across the ten fits: d-5 196/196/0/206; d-20 470/440/30/139;
  d-50 630/560/70/121. R reports `package_not_exposed` for all cases.
- Full per-seed hashes, metrics, stops, and diagnostics: `analysis_summary.json`.
  This is one frozen development set; it supports no global recovery claim.
