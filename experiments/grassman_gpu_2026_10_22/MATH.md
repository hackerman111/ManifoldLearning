# GPU port: objective and numerical boundary

The reference is the exact live combined20 source at task start, SHA256
6610d885741dac4d604fe49e3e73120cc08c71b55cd75afea7ae185771cebfd7.
`cpu_reference.py` retains it. Production NumPy comparisons are bitwise.

With Y=(d,m), Y.T Y=I, each center uses M_j=U_j Y and the minimum-norm
coefficient g_j minimizing ||I_j-M_j g||²+tau||g||². The profile is
F(Y)=sum_j mass_j (||r_j||²+tau||g_j||²), r=I-Mg.
The horizontal envelope gradient is (Id-YY.T) times
-2 sum_j mass_j (U_j.T r_j) g_j.T. Batched SVD/QR and contraction order
implement these same formulas on CuPy. No float32 is introduced.

For horizontal Z=VK, W_j=U_j V, the smooth derivative solves
(M.T M+tau Id) dg = (WK).T r - M.T(WK)g using SVD/QR factors;
dr=-(WK)g-M dg. Ridge residual derivative is -sqrt(tau)dg. Full q*m
cross terms, root masses, relative GN damping and live-profile Armijo
are shared by both backends. Polar retraction is unchanged:
(Y+VK)(Id+K.T K)^(-1/2), with cached projected operator action.
Intrinsic-rank exclusion only proposes steps; the full objective accepts them.

An exact-zero U_j contributes mass_j||I_j||² independent of Y. Existing
constant restoration and original gradient stopping normalization are retained.
The recursive result restores host coefficients using only an active-row mask;
the large I/U arrays remain on GPU. Final index/coefficient arrays return to
NumPy for the existing HPAOResult/public model contract.

Classification: EXACT backend arithmetic; NUMERICAL differences in reductions
and SVD implementations. CPU-sensitive lstsq repeats the factorization with
the same eps*max(p,m) rank cutoff. GPU uses its batched minimum-norm SVD
with that explicit cutoff, avoiding a separate device solver/host sync per
sensitive center. This preserves the mathematical rank policy, not bitwise
singular values at the threshold. GN smooth/rank guards stay unchanged.

Near-collinear systems with condition about 1e12 amplify float64 rounding:
on the deliberately stressed unregularized fixture, objective differences
were about 6.5e-6 and 1.6e-5 relative, while the solver stayed at the same
uncertified rank-boundary stationary point. Such problems cannot justify
uniform 1e-9 CPU/GPU objective parity. The parity stress instead uses explicit
tau=1e-4; exact rank-deficient unregularized inputs remain separately tested.
No ridge/default/cutoff is changed to conceal this sensitivity.

Heavy pilot full fits also amplify tiny backend differences through repeated
outer localization: early objectives match, later fitted subspaces diverge.
Therefore frozen-input parity and independent paired recovery are separate
requirements; no all-trajectory equivalence is claimed.

Large-array costs: U is O(JPd), profiles O(JPm), gradient scratch <=4MiB
apart from one center, GN estimate is 8*(4*rows*q*m+(q*m)^2) and retains
the configured 16MiB limit/fallback. Library CUDA workspaces and memory-pool
reservation are additional. No (J,n,d) or dense (d,d) array is introduced.

GPU QR specialization: for local matrices with min dimension<=32<max,
Householder thin QR A=QR followed by R=L S V.T gives A=(QL)S V.T.
For wide matrices apply this to A.T and transpose factors back. Singular
values and min-norm coefficients are preserved in exact arithmetic; this
is NUMERICAL in float64. The caller retains the original p/m/d cutoff.
Rank-only calls compute QR(mode=r), avoiding left-factor reconstruction.
No normal matrix or squared conditioning is introduced. Installed CuPy14.1.1
otherwise loops cuSOLVER SVD across these local matrices; see official
https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html .
Independent tall/wide/rank-loss reconstruction/spectrum tests and profile
lstsq comparisons pass; all nine CPU-reference cases remain bitwise.

Augmented GN compression: for A=[J;sqrt(mu)Id]=QR, minimizing ||Ax-b||²
is equivalent to minimizing ||Rx-Q.T b||² because the orthogonal complement
of b is independent of x. Solve the latter by small SVD, preserving the
original eps*max(A.shape) cutoff rather than recomputing it from R.shape.
The GPU path uses this NUMERICAL reformulation; CPU lstsq is untouched.
QR uses a tall thin copy already covered by the four-array workspace estimate;
CUDA library workspace remains additional. No J.T J is formed.
