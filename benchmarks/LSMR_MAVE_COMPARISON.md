# LSMR against MAVE and MEANMAVE

The comparison reuses the saved `highdim-dm` and `d25-n5000` benchmark data and
baseline rows. It runs only three new LSMR fits per rank (`m=3,5`) in each
workload; MAVE and MEANMAVE are not rerun. The new per-seed records, exact data
hashes, and solver diagnostics are saved under `benchmark_outputs/r_sdr/`.

To reproduce the three-method comparison on the existing `d=10` benchmark:

```bash
rtk proxy python -m benchmarks.lsmr_mave_existing
```

To reproduce the separate `d=25,n=5000` comparison against the saved MEANMAVE
fits:

```bash
rtk proxy python -m benchmarks.lsmr_meanmave_recheck --resume
```

Both protocols select the first three seeds with successful saved baselines per
rank. For `d=10`, the seeds are `312000..312002`; for `d=25`, they are
`412000..412002`. The existing benchmark configurations and 80-step LSMR
budget are retained. EDR is lower when recovery is better; the benchmark's
recovery threshold is `EDR <= 0.1`.

## `d=10`: same data for all three methods

Median EDR and fit time over three paired seeds:

| `(n,d,m)` | Method | EDR | Fit time (s) | Recovered |
|---|---|---:|---:|---:|
| `(300,10,3)` | LSMR | 0.5781 | 16.27 | 0/3 |
| `(300,10,3)` | MAVE | 0.5549 | 5.66 | 0/3 |
| `(300,10,3)` | MEANMAVE | 0.5667 | 0.38 | 0/3 |
| `(500,10,5)` | LSMR | 0.5077 | 31.34 | 0/3 |
| `(500,10,5)` | MAVE | 0.5740 | 20.70 | 0/3 |
| `(500,10,5)` | MEANMAVE | 0.5287 | 0.43 | 0/3 |

For `m=3`, LSMR's median EDR is slightly worse than both baselines; for `m=5`,
it is better than both on these three seeds. LSMR takes 2.71x and 1.49x as long
as MAVE, respectively, and 42.5x and 63.3x as long as MEANMAVE. No LSMR fit
received an inner convergence certificate.

## `d=25,n=5000`: saved MEANMAVE comparison

| Rank | Method | EDR | Fit time (s) | Recovered |
|---:|---|---:|---:|---:|
| 3 | LSMR | 0.5775 | 33.23 | 0/3 |
| 3 | MEANMAVE | 0.0958 | 12.51 | 2/3 |
| 5 | LSMR | 0.7287 | 70.34 | 0/3 |
| 5 | MEANMAVE | 0.4747 | 12.67 | 0/3 |

At this larger dimension LSMR is slower by 2.66x for `m=3` and 5.55x for
`m=5`, with higher EDR in both cases. Ordinary MAVE has no saved results in
this `d=25,n=5000` benchmark, so the direct three-method comparison above uses
the existing `d=10` cases where both MAVE variants completed.

These are six-seed-count pilot comparisons per workload, not broad recovery
estimates. LSMR did not pass inner convergence certification in the saved runs;
the results do not justify replacing the Grassmann default.
