# Intrinsic rank-one warmup experiment

Explicit candidate: `ADP-Grassmann-warmup` calls the frozen `warmup.solve`
wrapper. Production defaults and all earlier labels remain unchanged.

- Each solver call begins with one original bounded-scalar rank-one step when
  its initial profile is nonsmooth, then uses the intrinsic regular-center GN
  direction. Full all-center profile acceptance, live gradient, and original
  stationarity-certificate semantics remain in force.
- Budget: 10 total steps for initially nonsmooth profiles, 5 for initially
  smooth profiles; `tol=1e-6`, `local_ridge=0`, one warmup step. The candidate
  is an approximate optimizer direction, not a new estimator.
- Paired workload: cases d-5/d-20/d-50, n=500, m=2; methods
  `ADP-Grassmann` (5 steps), `ADP-Grassmann-warmup`, and R
  `MAVE-package` (MEANMAVE). Ten paired seeds per case, same data and method
  seed within each pair, one BLAS thread, 300 s and 4096 MiB process limits.
- Development seeds: 101800–101809. Untouched validation seeds:
  102800–102809, used only after development passes.
- Reject on incomplete/failed pairs; candidate/baseline EDR geometric mean
  ratio >0.95; any case median EDR delta >0.005; paired fit-time median/p90
  ratio >1.10/1.25; median RSS ratio >1.10; non-finite basis or
  orthogonality error >1e-10; or any individual paired EDR ratio >2.
- Freeze source, input, environment, and method-seed hashes; report paired
  bootstrap uncertainty, individual extremes, actual R medians, stops, and
  certificates. Failure closes the candidate; do not tune on consumed seeds.

The protocol evaluates one experimental direction. It does not justify a
production-default change or a universal recovery claim.
