# M2: formula, direct reference and oracle diagnosis

The selected **implemented** local objective for target `l`, after fixing slopes,
is

```text
F_l(B) = sum_j N_j omega_jl ||I_j - U_j B.T a_jl||^2
       + lambda tr[B.T B (Id - E_l)],
E_l = sum_j omega_jl P_j.T P_j / sum_j omega_jl.
```

`I_j` has shape `(P,)`, `U_j` `(P,d)`, `a_jl` `(m,)`, `B_l` and
`P_l` `(m,d)`. The CSR row `l` contains source columns `j`, so data weights
are `N_j omega_jl`. `tests/test_manifold.py` compares weight/moment arrays to
an explicit center-by-observation construction, a direct augmented LS solution
to the B normal operator, a finite-difference objective derivative, an adjoint
identity, and low-rank projector recovery. The focused tests pass (13/13).

| Formula / question | Code / reference result |
|---|---|
| Normalized directional `I,U` and graph orientation | `weights.calculate_statistics`, `graphs.build_manifold_graph`; explicit tests agree to `2e-14`. |
| Slopes at fixed `P_l` | `optimisation.local_slopes`; direct `lstsq(U_j P_l.T, I_j)` agrees to `1e-13`. Rank-zero remains an explicit failure. |
| B objective, gradient and minimizer | `build_B_system`/`objective`; augmented direct LS and finite difference agree in the deterministic test, with CG residual `<1e-10`. |
| Projector from B | `recover_projector`; explicit `B.T M B` or rank-one projector agrees in focused tests. |
| Penalty mass | `tex/manifold-ade.tex:214-225` includes `W_l`, while `:334-353` omits it. Live code and `tests/test_manifold.py:397-420` implement the latter. It is an unresolved **ESTIMATOR** choice, not an algebraic defect. |
| `h_M` schedule | `tex/manifold-ade.tex:752-770` updates `h_k` explicitly but writes `h_M,k` without an update instruction. Live `fit.py` holds `h_M` fixed. A decreasing `h_M` is an **ESTIMATOR** variant; manuscript text does not prove the current implementation wrong. |

The probe's `oracle_*_selection` directories use the same selection data,
center seeds and final metric as baseline. True gradients are the exact
derivatives of the standardized quadratic radial signal; oracle moments set
`I_j=U_j gradient_true(x_j)`. No oracle uses validation seed.

| Selection variant | Successful fits / 10 | Initial RMS over centers | Median final RMS over successful fits | Recovered |
|---|---:|---:|---:|---:|
| baseline | 4 | 0.479 | 0.496 | 0 |
| true pilot gradients | 4 | 0.428 | 0.446 | 0 |
| true initial projectors | 6 | 0 | 0.315 | 0 |
| true directional moments | 7 | 0.479 | 0.454 | 0 |
| true initial projectors and moments | 7 | 0 | 0.243 | 0 |

With perfect initialization and perfect first-order moments, the first
manifold update already raises aggregate RMS sine to 0.223, and the later
updates leave it around 0.24–0.26. This isolates a structural bias in the
localized objective/graph on the radial scenario; pilot error and sketch noise
alone cannot explain the observed quality. Rank failures persist when a source
has one effective observation. Next: test explicit localizer and mass variants,
one factor at a time, against the unchanged baseline/reference.
