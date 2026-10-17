# Supplementary localization microbenchmark

This supplement isolates P1 locality cost and is not a recovery benchmark. It
does not alter the frozen full-fit manifest. Its own manifest records the
parent manifest SHA-256 and hashes these supplementary scripts.

Cases are `(n=500,d=30,J=500,m=2,h=1)` and `(n=10000,d=100,J=500,m=2,h=.4)`.
Each uses a fixed-seed AR(1) Gaussian `X` (`rho=.5`), centers sampled from `X`,
an orthonormal random basis, eigenvalues `[1,.25]`, and a common cached squared
distance matrix `D`. `alpha=.02`; alpha-search target is 20. Seeds are
`93000..93002`. Each arm/mode/seed is a separate fresh process, serially
executed with one verified BLAS/OMP thread.

For weight-only mode, compare `calculate_multi_weight(..., distance2=D)` with
`projected_components(X,C,B,eigen,D).weights(alpha)`. Timed stages include the
respective localization representation construction and final CSR weight
assembly. Construct `D` before the timed interval and report its separate
construction time and memory. In alpha mode, compare
`calculate_alpha_k(...,distance2=D)` followed by weights against
`projected_components` preparation, its alpha search, and weights. No prepared
geometry is reused across timing stages. For each arm, compute the opposing
arm's reference after timing, then report max absolute weight error, support
agreement, and alpha error outside the timed interval. Record dense-equivalent
and CSR storage plus process peak RSS.

Report each of the three paired seed repeats; do not infer full-fit recovery or
large-scale speedups beyond these two workloads. This is supplemental to, not a
replacement for, the full-fit results in `../REPORT.md`.
