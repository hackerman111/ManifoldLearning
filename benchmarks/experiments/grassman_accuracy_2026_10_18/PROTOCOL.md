# Frozen accuracy comparison

Frozen after exploratory seed 90800, before development seed 91800 is run.

Candidate: `ADP-Grassmann-accurate`, original core-GN directions/rank guard,
localized scalar profile with full-design numerical guard, max_steps=10.
Baseline: unmodified `ADP-Grassmann`, max_steps=5. R comparator:
`MAVE-package` invokes real `MAVE::mave.compute(method="MEANMAVE")`.
No weights/statistics/ridge/initialization/schedule changes.

Cases: existing full-catalog d-5, d-20, d-50 (n=500,m=2), selected from the
previous incomplete catalog's quality gaps. Ten paired datasets per case.
All methods use identical X/y/truth hashes, method seed and ADP config.
Development seeds 91800..91809; validation 92800..92809 untouched until pass.
Source, input, environment fingerprints and raw failures are saved by R-SDR.
One BLAS/OMP thread, float64, 300 s / 4096 MiB per process, randomized method
order in every paired block. Fit clock is model.fit; process time is separate.
Peak RSS is sampled Linux /proc VmHWM/VmRSS (polling may miss short peaks).

R3 and R4 gates, fixed before looking at development results:

- No additional failed/invalid fits; complete 30 candidate/baseline pairs.
- Geometric mean of paired EDR ratios <=0.95 (>=5% relative improvement).
- No case median of paired EDR differences >0.005.
- Median paired full-fit time ratio <=1.10 and p90 <=1.25.
- Candidate median RSS <=1.10 baseline; no forbidden allocation.
- Finite result, orthogonality <=1e-10; no false convergence claim. Rank
  guards, accepted steps, inner certificates and outer schedule stops retained.

Report paired bootstrap intervals without requiring significance from this
small suite. Use fail/invalid rows in denominators. No tuning after R3; a
failed gate leaves validation untouched and closes this candidate. Any new
hypothesis needs a separate frozen protocol and new development seeds.
Results apply only to this bounded n/d/rank/link suite, not the 50-case catalog.

Command (replace output and seed for validation only after R3 passes):

    OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.r_sdr --profile full --cases 'd-5,d-20,d-50' --runs 10 --seed 91800 --methods ADP-Grassmann,ADP-Grassmann-accurate,MAVE-package --grassman-steps 5 --threads 1 --keep-inputs --output benchmarks/results/outputs/r_sdr/grassman-accuracy-dev-2026-10-18
