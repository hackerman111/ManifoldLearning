# Exactly constant scalar curve contributions

For the rank-one geodesic use an orthogonal index rotation [A,a]. The local
design is [C,x(t)] with C=M A and x(t)=Ma cos(t)+w sin(t). If M and w have
support in a single row r, C and x(t) are multiples of e_r. If C is nonzero,
the column space of [C,x(t)] is exactly span(e_r) for every t. At tau=0 its
least-squares minimum is sum_{k != r} I_k², independent of t. This argument
does not apply to positive ridge, multiple supported rows, or m=1 without
a constant column. When M=w=0 the loss is ||I||² for any ridge and m.

The implementation detects support by exact zero comparison, without a
tolerance that could silently discard a nonzero design entry. It requires
||C||_F² > sqrt(eps)*(||C||_F²+(||Ma||+||w||)²) before caching a single-row
center. This bound retains a substantial constant component throughout the
curve and avoids cancellation at the full numerical rank cutoff. C has rank
one here, so its Frobenius norm is its only nonzero singular value. Empty or
underflowed C fails the guard. Nonfinite data is rejected by the full profile.
Loss is summed outside row support, without subtracting two nearly equal
target energies. Cached zeros/one-row losses include mass and remain in the
objective; centers are not removed from gradient, coefficients or GN guards.

Other centers retain candidate.py's guarded scalar/direct-SVD path. Every
accepted step still refits the original full design through the original
profile, and the final certificate recomputes live UY. Mathematical scalar
elimination is EXACT; floating-point evaluation is NUMERICAL. Extra steps
are APPROXIMATE, so full paired fits against baseline and actual R MEANMAVE
are required before promotion. No global convergence or recovery theorem.

Cost: one O(J*p*m²) support/complement setup per curve and O(J) masks/losses;
constant centers incur no SVD/lstsq per scalar evaluation. All remaining
storage and full acceptance paths are unchanged; no ambient d*d matrix.
