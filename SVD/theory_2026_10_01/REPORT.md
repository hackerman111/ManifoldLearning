# Unified EDR/SVD theory: verification and research record

Date: 2026-10-01. Deliverable: [EDR_unified_theory.tex](../EDR_unified_theory.tex)
and [compiled PDF](EDR_unified_theory.pdf), a standalone Russian synthesis.
Production ADP code and user source notes were not changed.

## Mathematical synthesis

The common formulation is separable operator least squares with optional
variable projection, a reduced linear dictionary of updates, and a chart or
retraction that constructs the actual matrix/subspace/field. These are different
special cases, not assertions that their objective functions coincide.

- Fixed-g SVD: ambient matrix or low-rank correction, exact curvature-normalized
  rank-one gain, diagonal scales as a subcase of a full core.
- Exact closed SVD: complete quadratic operator must be separable `G X H`,
  including the penalty; two-sided rank-preserving whitening then applies.
- Grassmann: local coefficients profiled, SVD of a horizontal tangent gives
  commuting plane rotations, and the resulting basis correction has rank at
  most the tangent rank. Arbitrary fixed-g ambient correction is not equivalent.
- Spatial field: CP/Tucker first acts on linear coefficients or on a common
  horizontal tangent field; the latter preserves full local basis rank after
  retraction. It need not produce a low-rank tensor of final bases or an
  integrable tangent distribution of a predictor manifold.
- Anisotropy: separate localization, residual, penalty and search metrics;
  spectral tensor actions avoid dense d-by-d allocations.

Corrections to draft claims: negative Tucker ridge sign; w-squared in kernel
moment noise covariance; normalized Gaussian direction law; finite polar chart
boundary; conditional GLS/MAVE/Fisher interpretations; closed-form curve value
versus global minimizer; rank-one descent versus convergence of the entire
sequence. The independent Luna mathematical review found that the joint GN
residual/Jacobian needed explicit square-root mass factors. The document now
states those factors, and the numerical reference already included them. The
lead also added the tau-zero deficient-rank pseudoinverse derivative guard.

## Independent formula references

Run from the repository root:

```sh
python SVD/theory_2026_10_01/check_formulas.py
```

Result: **203 checks across 35 named groups passed**, five geometry/profile
configurations and additional special cases. Python 3.14.7, NumPy 2.5.2,
SciPy 1.18.1, float64; exact recorded environment and dirty paths are in
[formula_results.json](formula_results.json). No production solver imports.
Seeds 601001--601005 include m=1, m=3, p<m with positive local ridge,
tau=0 with full local rank, d<2m, and p>d+1.

| Claim | Independent comparison | Largest relative error |
|---|---|---:|
| Grassmann exponential | Ambient matrix exponential | 2.72e-16 |
| Commuting rotations | Product of ambient exponentials vs tangent SVD exponential | 2.95e-16 |
| Scalar Schur profile | Direct augmented least squares on each rotated basis | 2.96e-15 |
| Profile gradient | Central differences along geodesics, h=1e-5 | 1.38e-9 |
| Full profile residual Jacobian | Central differences of augmented weighted residual | 1.10e-10 |
| Two-sided SVD tail | Direct separable quadratic energy difference | 5.42e-16 |

Other checks cover forward/adjoint, rank-one gain, all core cross terms, full
core versus diagonal refit, untruncated joint QR, polar angle mapping and
subspace equivalence, profile basis invariance, second-order expansion,
MAVE centered covariance identity, squared-weight covariance, spectral tensor
actions, and distinct correction/basis ranks. Final checks also cover both blocks of
penalty whitening, tangent Tucker horizontality/full basis rank, and a flat
null-direction atom with zero global ridge. A weighted 2-by-2 counterexample
gives ordinary truncated-SVD loss 100 versus a feasible rank-one loss 0.04.
A tau-zero rank-loss example shows profile values 0 arbitrarily near the
boundary and 1 at the exact degenerate direction.

These numerical checks corroborate identities and guards. They do not prove
novelty, a global optimizer, convergence of the external adaptive procedure,
or statistical recovery. No broad ADP regression suite was run because no
production implementation changed.

## Narrow angle-evaluation microbenchmark

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python SVD/theory_2026_10_01/angle_microbenchmark.py
```

Both methods use the same synthetic frozen statistics and same cached
`U_j[Y,v]`: J=128, p=40, d=300, m=3, tau=0.2, 61 angles. The reference
repeats **batched augmented SVD**, avoiding J individual Python solves and
normal equations. The proposal eliminates the m-1 unchanged columns once
using augmented QR, then evaluates a scalar trigonometric rational profile.
Both paths are warmed; timings are medians of seven repetitions with
alternating evaluation order. The shared dense projection is warmed and
measured seven times as well. It is included in the total ratio below.

| Seed | Shared projection ms | Scalar prep ms | Scalar curve ms | Batched SVD curve ms | Total time ratio |
|---:|---:|---:|---:|---:|---:|
| 602001 | 4.053 | 0.553 | 0.569 | 26.940 | 0.1670 |
| 602002 | 4.076 | 0.327 | 0.545 | 26.867 | 0.1599 |
| 602003 | 4.050 | 0.317 | 0.540 | 25.764 | 0.1646 |

Relative curve errors: 1.38e-16--1.66e-16. For the curve including scalar
preparation, time ratios are 0.0324--0.0417; including common projection they
are 0.1599--0.1670 (about 6.0--6.3 times less elapsed time).
The reference and scalar formulation evaluate the same ridge data profile;
the optional chordal penalty was not timed. Raw samples, BLAS build,
thread variables, code revision, dirty paths, and allocation measurements are
in [angle_results.json](angle_results.json).

Memory measured separately using tracemalloc, not RSS/native LAPACK peak:
scalar curve 7,776 traced bytes, preparation 559,584, repeated-SVD curve
564,352; preexisting inputs are excluded. The proposal therefore has a much
smaller **curve-evaluation** scratch allocation, but this experiment does
not establish a lower total peak memory once preparation and common inputs
are counted. Speed ratios concern a preselected plane's curve only: no
direction search, angle minimization, optimizer iteration, bandwidth update,
full fit or projector recovery was measured. Hardware/load and short timing
variability limit extrapolation.

## Corpus and primary literature

Read-only GPT-6 Luna scouts routed source material and literature. The lead
reopened critical formulas, checked the live fixed-g objective/metric/refit,
and derived the mathematical statements independently. No material inside
SVD/documentatio, documentation, archive, pdf or PDF was read.

The current allowed seven-note corpus is inventoried in Appendix A of the
document. `SVD_solver.tex` retains a historical source map to older filenames;
the new map uses files actually present. Live verification used
`ADP/solver/SVD.py:164-301,690-728,895-1025`,
`ADP/solver/LSMR.py:429-489`, `weights.py:88-184`, `calculus.py:390-422`.
The preexisting unfinished joint optimizer plan/state were preserved verbatim
as PREVIOUS_PLAN.md and PREVIOUS_STATE.md; its implementation was not resumed.

Primary sources accessed on 2026-10-01:

| Source | Verified scope and original route |
|---|---|
| Hristache--Juditsky--Polzehl--Spokoiny (2001) | Structural adaptation of ADE; [author institution preprint](https://www.wias-berlin.de/preprint/569/wias_preprints_569.pdf). Rates are not transferred to this model. |
| Xia--Tong--Li--Zhu (2002) | MAVE observation-level objective (2.7), centered local LS and adaptive weights; [original paper hosted by university](https://www.math.uestc.edu.cn/__local/C/90/85/E60A4514D06FE5FC67CDED2804A_4B9ABA5D_14768C.pdf?e=.pdf). |
| Golub--Pereyra (1973) | VARPRO, constant-rank derivative condition; [publisher record](https://epubs.siam.org/doi/abs/10.1137/0710036?journalCode=sjnaam), retrieved via indexed publisher text. Direct DOI open failed. |
| Edelman--Arias--Smith (1998) | Sections 2.5.1--2.5.4, Grassmann exponential (2.65), gradient (2.70); [author PDF](https://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf). |
| Balzano--Nowak--Recht (2010) | GROUSE profiled objective and rank-one geodesic update, section 3; [original preprint](https://arxiv.org/pdf/1006.4046). |
| Srebro--Jaakkola (2003) | Weighted low-rank approximation does not generally admit ordinary SVD solution; [author PDF](https://people.csail.mit.edu/tommi/papers/SreJaa-icml03.pdf). |
| Mishra et al. (2012/2014) | Smooth fixed-rank quotient optimization, gradient/trust-region algorithms; [original preprint](https://arxiv.org/abs/1209.0430). |
| Absil--Malick (2012) | Conditions on projection-like first/second-order retractions; [author page](https://sites.uclouvain.be/absil/2010.038). Our spectral angle examples are derived separately. |
| Wang et al. (2014) | Singular-atom matrix pursuit and joint weight update; [original proceedings](https://proceedings.mlr.press/v32/wanga14.html). Matrix-completion guarantees not transferred. |
| Van Loan--Pitsianis (1992 report/1993 chapter) | Rearrangement/SVD Kronecker approximation; [original institutional report record](https://ecommons.cornell.edu/entities/publication/ad6a5c71-2632-4652-b33a-72016d7abc4d). |
| Kolda--Bader (2009) | CP/Tucker decompositions, distinct ranks; [author manuscript page](https://www.kolda.net/publication/koba09/). |
| Cheng--Wu (2013) | Tangent regression for predictors on a manifold, different from varying regression EDR; [original preprint](https://arxiv.org/abs/1201.0327). |

Eckart--Young (2036) is cited for the classical rank-preserving Frobenius
truncation theorem; the sufficient separable extension is proved in the note.
The publisher DOI was inaccessible in this browsing session. No inaccessible
full text is treated as evidence for additional claims. The bounded search
is not an exhaustive novelty investigation.

## Build and layout

Build twice from root (another pass if labels change):

```sh
xelatex -halt-on-error -interaction=nonstopmode \
  -output-directory=SVD/theory_2026_10_01 SVD/EDR_unified_theory.tex
```

XeLaTeX/TeX Live 2026, Noto Serif, A4. LuaLaTeX was initially tried and failed
because its fontloader had no writable default cache; XeLaTeX builds within
the existing workspace permissions. Final PDF, log and stdout are retained.
Layout was checked by rendering the title/contents, dense math, and source-map
pages and by scanning final build logs for overflow, missing glyphs and
unresolved references. Final page count and build audit are recorded in
`build_audit.json`.
