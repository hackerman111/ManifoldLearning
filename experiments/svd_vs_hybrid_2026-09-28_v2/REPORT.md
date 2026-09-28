# SVD vs HYBRID: paired full-fit comparison

## Protocol

This is an exploratory comparison of the opt-in `rank=2` SVD method and the
current CPU HYBRID solver for `m=3`. It ran the complete public
`ADP_multi_index.fit` loop on the same generated data and fit seed for each
pair: three seeds at `(n,d,N_J,N_phi,N_lin)=(500,20,40,10,60)` and three at
`(900,60,64,12,100)`. All fits used float64, one BLAS/OpenMP thread, and three
outer iterations. Per-run details and raw traces are in `runs.jsonl`; exact
configuration and environment are in `protocol.json`.

Quality is normalized projector distance
`||B_est B_est.T - B_true B_true.T||_F / sqrt(2m)`; smaller is better. Time is
`perf_counter` around `fit()` only. Peak RSS is the worker process high-water
mark and includes interpreter/import memory. `tracemalloc` records Python
allocations and misses many native allocations.

The methods optimize different inner objectives. SVD uses the fixed-g,
rank-constrained objective in `SVD.tex`; HYBRID uses the current HPAO
correction-penalty objective. Both were passed `lambda_penalty=0.05`, but their
objectives are not equivalent. The SVD public wrapper also completes the
remaining `m-r` basis direction from the incoming prior. Treat quality
differences as descriptive for these two methods, not an apples-to-apples
optimizer comparison.

## Results

Medians over three runs per cell:

| Case | Solver | Fit time (s) | Process peak RSS (MiB) | `tracemalloc` peak (MiB) | Projector distance |
|---|---|---:|---:|---:|---:|
| small | SVD | 0.241 | 74.25 | 0.401 | 0.474 |
| small | HYBRID | 0.076 | 77.16 | 0.767 | 0.637 |
| medium | SVD | 0.750 | 82.87 | 1.409 | 0.666 |
| medium | HYBRID | 0.273 | 83.53 | 3.698 | 0.723 |

Across the paired seeds, SVD took 3.25x HYBRID time on the small case and
2.75x on the medium case (median paired ratio). It used about 2.9 MiB and
0.7 MiB less process peak RSS, respectively; the peaks are close and include
the same interpreter/import baseline. Python-tracked peak allocations were
about 48% lower on the small case and 62% lower on the medium case. This
tracemalloc reduction did not translate into a comparable process RSS change.

SVD's median projector distance was lower by 0.056 on the small case and
0.026 on the medium case. The paired SVD-minus-HYBRID quality differences were
`[-0.056, -0.043, -0.175]` and `[+0.035, -0.096, -0.026]`, respectively, so
there is one medium seed where HYBRID had the lower distance. Both methods
started from identical bases within each pair. With only three seeds, this is
not stable recovery evidence.

All 12 valid fits completed three outer iterations and stopped at the explicit
`outer_steps` cap; neither method was shown to have converged in the outer
loop. SVD's inner alternating solves converged in all three steps for 2/3 runs
in each case (one small-case run hit its 20-iteration cap). HYBRID's HPAO
convergence flag was false in all six runs. HYBRID selected its `cached-svd`
linear backend in these shapes, so its LSMR iteration count was zero; this is
not zero solver work. SVD used a median 35 inner iterations (small) and 54
(medium) across its three outer steps.

The first pilot folder `../svd_vs_hybrid_2026-09-28/` preserves 12 pre-fit
attempts rejected by the input validator because `N_J < ceil(n/N_loc)`. They
are protocol failures and are excluded from every result above; the corrected
v2 run completed all 12 fits.

## Reading the result

For these settings, HYBRID is materially faster, while SVD has lower median
projector error and lower Python-tracked temporary allocation. Process peak
RSS is effectively similar at this scale. The quality difference is
confounded by the distinct objectives and SVD's prior-dependent completion,
and the three-step outer cap prevents convergence claims. Larger paired runs
and an objective-aware evaluation are needed before choosing either method for
production workloads.
