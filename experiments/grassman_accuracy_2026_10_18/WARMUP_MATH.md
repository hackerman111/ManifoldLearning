# One original direction before intrinsic GN

The numerical intrinsic-rank model remains APPROXIMATE; see INTRINSIC_RANK.md.
It preserves the full profiled objective, every true trial and the original
rank-boundary certificate semantics. No discarded-tail bound is claimed.

A profiled GN model linearizes a residual at its current point. Positive
damping and a negative full-gradient slope ensure only a local descent
proposal; a small fixed budget need not leave a poor region of the profile.
The failed100800 candidate supplies an example, not a general explanation:
its first inner objective50.6433 remained above the baseline8.16165 after
ten versus five accepted steps. Both runs were finite and orthogonal.

The new proposal retains the ORIGINAL rank-one bounded scalar search for
the first step of each initially nonsmooth frozen solve. Its angle, polar
geometry, full true-profile Armijo test and refit are exactly the original
one-step solver. Subsequent steps use the same intrinsic GN proposal, with
full-gradient slope, all-center acceptance and all original guards. At an
initially smooth point retain five GN steps. Total nonsmooth budget is ten,
including that original first step. No additional outer fit or estimator
selection is introduced. Positive ridge bypasses intrinsic classification.

This is a safeguarded initialization heuristic, not a basin/global optimum
guarantee. One-step equality to the original solver is independently tested
on intrinsic and projected rank-loss inputs. Independent least squares
checks full coefficients/objective/ranks, monotone history and orthogonality;
second-step tests exercise GN and bounded-workspace fallback.

Use only90800 for the pilot. Freeze before101800 development;102800 remains
untouched until pass. The former99800/100800 data are consumed and cannot
serve as independent validation. Add a predeclared paired EDR outlier gate
of2x to the previous aggregate quality, case-median, time and memory gates.
