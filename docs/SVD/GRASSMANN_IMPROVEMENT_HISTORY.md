# Attempts to improve the Grassmann solver using the SVD research notes

Snapshot: 23 October 2026. This is an English account of the recent research
sessions, reconstructed from repository reports, proofs and decisions. The
main focus is the 18–22 October accuracy work and its latest low-rank follow-up;
the earlier October studies explain where those attempts came from. Experiment
directory dates mark the beginning of a study; several reports in the
`2026_10_18` directory were completed on 22 October.

The current algorithm is described separately in
[ADP_Grassmann_current.tex](../tex/ADP_Grassmann_current.tex) and its
[PDF](../tex/ADP_Grassmann_current.pdf). This history reports saved experiments;
the large benchmarks were not rerun while preparing it. Results from different
studies have different inputs and budgets and must not be pooled.

The main outcome is mixed: the reduced profiled core and its implementation
optimizations are in production, and relative intrinsic GN passed independent
quality/time validation. Several mathematically valid extensions reduced the
inner objective but failed full-fit quality or runtime gates. The latest
projected-Jacobian candidate failed its development quality gate. The LoRA-like
correction has only exploratory evidence.

## 1. What we took from the SVD folder

| Source | Relevant idea | How it entered the investigation |
|---|---|---|
| [EDR_unified_theory.tex](EDR_unified_theory.tex), `sec:grassmann`, `sec:angles`, `sec:reduced`, `eq:dg` | Eliminate local coefficients; choose gradient singular planes; differentiate the complete profiled residual; cache operator actions | Initial reduced core GN, rank-one profiles and later geometric variants |
| [glm.tex](glm.tex), `sec:cartan`, `sec:angles`, `sec:fullstep`, `sec:improve` | Joint plane rotations, larger tangent steps, alternative search geometry and compression | Full-tangent GN, joint-angle `flat_gn`, later accuracy/rank-domain diagnosis |
| [edr_research_note.tex](edr_research_note.tex), `eq:schur`, `eq:dg`, `eq:metricstep`, `eq:CX` | Schur elimination, exact chart derivatives/Hessian actions, coefficient metrics, factorized moment operators | Structured Schur/VARPRO Newton, metric descent, recycling and CX experiments |
| [SVD_solver.tex](SVD_solver.tex), `sec:second`, `sec:coarse`, `sec:block`, `sec:algorithm` | Use SVD to select a space, solve all coefficients there, refine additive low-rank corrections through an augmented operator | Latest SVD-block, LoRA-like, and residual-enriched proposals |

An essential distinction survived every study. The production Grassmann
objective is the **profiled finite-sketch** objective

\[
F_\tau(Y)=\sum_j N_j\min_{g_j}
\{\|I_j-U_jYg_j\|^2+\tau\|g_j\|^2\},\qquad Y^\top Y=I_m.
\]

The fixed-coefficient SVD correction problem freezes the current `g_j` and
optimizes an ambient additive matrix. It can generate a proposal for `Y`, but
does not solve the same profiled constrained problem. Likewise, restricting
the rank of a matrix, restricting the rank of its correction, and restricting
the number of Grassmann rotation planes are different operations.

Positive local ridge changes the statistical objective. Damping in an
augmented numerical step does not. Lower frozen `F`, fewer iterations and
better EDR recovery are separate outcomes.

## 2. Initial reduced core and the October 1–3 implementation bridge

The initial theory-derived core uses the full coefficient differential

\[
(M_j^\top M_j+\tau I)\,dg_j
 =T_j^\top r_j-M_j^\top T_jg_j,
\qquad T_j=U_jVK.
\]

It retains the `T_j.T @ r_j` term and optimizes all `q*m` entries of `K`
jointly, with polar retraction and true-profile Armijo acceptance. Its core
is complete **inside the chosen left plane**; it is not the full
`m*(d-m)`-dimensional tangent solve. The original motivation and guards are
recorded in the [decision log](../agent-notes/DECISIONS.md), entries
“Unify EDR/SVD” and “Explicit profiled Grassmann solver,” dated 1 October.

The subsequent CPU optimization was an implementation bridge, rather than a
new SVD-folder algorithm: safe `m<=2` QR profiles, bounded gradient scratch,
transpose GEMM, batched Jacobian assembly and cache reuse preserved the
objective, rank guards, tolerances and optimizer. Saved frozen calls were
about **2.52 times faster**; three paired complete fits had median time ratio
**0.941939**, with close final projectors. This narrow evidence did not prove
convergence or a universal fastest method. The optimized core was promoted
on 3 October; `grassman_optim.py` was removed and its pre-optimization
reference archived. Sources: [implementation](../../benchmarks/experiments/grassman_optim_2026_10_01/IMPLEMENTATION.md),
[measurements](../../benchmarks/experiments/grassman_optim_2026_10_01/REPORT.md),
[promotion](../../benchmarks/experiments/grassman_promotion_2026_10_03/REPORT.md).

## 3. Full tangent GN and joint-angle flat optimization: correct formulas, failed gates

**Full tangent GN.** The `glm.tex` audit led to a matrix-free full horizontal
Jacobian/adjoint, augmented LSMR and polar/Armijo solver. Finite differences,
adjoint tests and dense least-squares references passed. However, all **45/45**
measured frozen GN linear solves exhausted the Krylov cap and failed the
normal-residual certificate. They fell back to core GN, at **11.63–20.21 times**
its frozen time. In adaptive fits, all 27 inner calls encountered rank loss,
giving 135 guarded fallbacks: the proposed full GN update was never applied.
Median full-fit time ratio was **1.2143** without a meaningful recovery gain.
The failed stage left seeds 73200–73204 unused. This rejects the bounded,
unpreconditioned implementation tested, not the mathematics of full GN.
Source: [full-GN report and theory audit](../../benchmarks/experiments/glm_grassmann_2026_10_02/REPORT.md).

**`flat_gn`.** A separate implementation cached `U[Y,V]` and jointly refined
the `q` rotation angles in a fixed gradient-selected flat. Frozen final
objective/gradient ratios were **0.586/0.606**, but time was **2.557 times**
core GN. Six full-fit pairs gave median time ratio **1.855** and worse
projector error in **4/6** pairs. The largest regression was `+0.071155`.
All inner calls remained budget-limited; raw full-fit rows also lacked a final
orthogonality field. Quality and time gates already failed independently of
that missing field. Held-out seeds 74200–74202 were not run. The method remains
an explicit experimental option in `grassman_exp.py`.
Source: [flat report](../../benchmarks/experiments/grassman_flat_2026_10_02/REPORT.md).

The theory audit also rejected shortcuts such as treating constant rank as
permission to invert a singular normal matrix, treating multi-plane polar and
exponential scalar paths as identical, or inferring global convergence from
sufficient decrease alone. Those limits matter when interpreting all later
variants; see the [audit proof](../../benchmarks/experiments/glm_grassmann_2026_10_02/PROOF.md).

## 4. Structured Schur, Newton, metric and CX work around October 15

The structured research branch implemented consequences of
`edr_research_note.tex`: inactive-coefficient Schur elimination, reduced-chart
GN and exact Newton actions, coefficient-based metric steps, recycling and
factorized `U=CX` operators. “Exact Newton” applies to the selected chart,
not the entire Grassmann problem. Primary runs all exhausted a 30-step cap;
there was no common-certificate speed comparison. Median primary solver times
were about **22.1 ms** core, **45.7 ms** metric, **36.4 ms** Schur GN and
**116–123 ms** Newton. Near-truth tests certified, but candidates were slower.

There was one positive weak-signal calibration: metric descent certified on
3/3 seeds versus core's 2/3, and was **1.6621 times faster** on the two shared
certified pairs, with median projector errors `9.00e-6` versus `9.22e-4`.
This was a small supplemental pilot, not a promotion result. The separate
Schur evaluate/HVP kernel was about **4.5 times faster per distinct trial**
after setup; that was not end-to-end solver time. CX saved storage only for
sparse support; full-support factorized storage was **6.505 times** dense and
operator actions were slower. Complete-fit smoke tests were slower for all
candidates. Sources: [proof/provenance](../../benchmarks/experiments/edr_structured_2026_10_15/PROOF.md),
[structured report](../../benchmarks/experiments/edr_structured_2026_10_15/REPORT.md).

The subsequent hard-Spokoiny metric test failed its frozen development gate:
median five-step fit-time ratio **0.982** missed the required `<=0.909`, and
one seed's projector regression **5.666e-4** exceeded `1e-4`. With a 100-step
budget, the five jointly certified metric/core pairs had time ratio **2.040**
and essentially equal recovery. The generator's population OPG was isotropic,
so the presumed intrinsically weak population direction was unsupported in
that benchmark. No held-out run or promotion followed.
Source: [metric report](../../benchmarks/experiments/metric_spokoiny_2026_10_15/REPORT.md).

## 5. Joint rank-k rotations

The rank-k study compared fixed, full, energy-selected and cost-selected
blocks of gradient SVD planes. Optimized blocks jointly refined angles rather
than greedily rotating one plane. None of the four candidates passed the
selection-stage 20% speedup threshold. The protocol nevertheless selected
`block1` for its specified held-out comparison. Across ten held-out pairs,
its median full-fit time ratio versus core was **0.99963**, with descriptive
95% interval **[0.98914,1.01554]** and zero median EDR difference. One pair
regressed by `0.000120`. No inner call attained stationarity.

Some multi-index cases always fell through the same rank-one guard, so they
did not distinguish the proposed joint-angle models. Larger blocks could
lower frozen profile values while worsening EDR; a budget rank is not a
statistical model-selection rule. `block_rank_k` remains an explicit
experimental entry point. Source: [rank-k report](../../benchmarks/experiments/grassman_rank_k/REPORT.md).

## 6. October 18–22: rank-domain diagnosis and the sequence before relative GN

Captured actual ADP calls showed that rank guards reflected genuine local
rank loss. Simply weakening the full-rank derivative guard was invalid.
The diagnosis started from `glm.tex` and the live core. It also exposed a
scalar-profile rank-cutoff mismatch: an apparently useful tiny local direction
could be dropped by the full design's least-squares cutoff. True-profile
acceptance protected the returned objective; a guarded scalar prototype
repaired the proposal evaluation.

The study then tried several bounded alternatives. Ratios below are
candidate/original5, using each variant's own fresh development data unless
the row explicitly states validation or pilot. Lower is better.

| Attempt | EDR geometric ratio | Full-fit time median / p90 | Outcome |
|---|---:|---:|---|
| Guarded local scalar search, budget10 | 0.9093 | 1.368 / 1.729 | Accuracy improved; time gate failed |
| Exact zero-operator centers, budget5 | 0.99973 | 0.8749 / 1.1504 | Time gain; original quality gate failed |
| Direct rank-one Armijo10, initial-smooth cap5 | 0.95550 | 0.95545 / 1.12086 | Original quality gate failed |
| Up to three quadratic angle trials | 0.96196 | 0.82290 / 1.09843 | Original quality gate failed |
| Initial intrinsic-rank GN, development | 0.89504 | 0.66773 / 1.00834 | Development passed |
| Same initial intrinsic GN, independent validation | 0.98546 | 0.65760 / 0.91733 | Quality failed; one 10.069-fold error regression |
| One scalar warmup before GN | 1.01325 | 0.7551 / 0.9099 | Quality and 28.75-fold outlier failed |
| Conjugate rank-guard search, development | 0.9271 | 1.1322 / 1.3975 | Both time thresholds failed |

Additional reused-seed pilots explored per-center QR/SVD reuse, exact constant
scalar curves, extra intrinsic polish after the original five steps, and
joint spectral fallback at rank guards. None gave a clear joint recovery/time
winner. In particular, a **1.23–1.31 times faster isolated profile** did not
offset full-fit overhead. QR reuse after the conjugate direction remained
slower than that direction on all three pilot cases. Reserved validation
seeds were left unused after the relevant failed gates.
Sources: [study report](../../benchmarks/experiments/grassman_accuracy_2026_10_18/REPORT.md),
[conjugate development](../../benchmarks/experiments/grassman_accuracy_2026_10_18/CONJUGATE_DEV_REPORT.md),
[QR-reuse pilot](../../benchmarks/experiments/grassman_accuracy_2026_10_18/CONJUGATE_QR_REUSE_PILOT.md).

## 7. Relative intrinsic GN: the accepted measured improvement

This revision forms a GN proposal from regular centers only when deficient
centers satisfy the numerical intrinsic-rank test `rank(U_j)=rank(U_j Y)<m`.
In exact arithmetic those deficient-center values are locally constant while
the rank equality persists. Numerical rank agreement is only a proposal
criterion: the full gradient, all-center profile acceptance and final live
certificate remain mandatory.

The key damping revision was `mu=lambda_prox*||J_reg||_F^2/(q*m)`. It scales
the regular-center GN model consistently with response and mass units;
the earlier, unnormalized intrinsic candidate's failed validation was not
discarded or relabeled as success.

| Stage | Candidate/original5 EDR ratio [95% interval] | Time median / p90 |
|---|---:|---:|
| Development, seeds105800–105809 | 0.898376 [0.844245,0.948903] | 0.650580 / 0.790378 |
| Untouched validation, seeds106800–106809 | 0.923729 [0.886622,0.956495] | 0.653572 / 0.993297 |

Each stage used 30 paired datasets, 90 successful fits across
`n=500,m=2,d=5/20/50`, including actual R `MAVE::MEANMAVE`. All eight frozen
gates passed. A 120-fit exact public-port replay and a later timing confirmation
used consumed data; they were not another independent validation. The accepted
settings were max10, initial-smooth cap5, relative GN on, quadratic trials off
and constant-center elimination off. All outer fits stopped at `h_min`, with
no certified inner calls in that comparison. R was more accurate in each
reported case median. Source: [relative-GN report](../../benchmarks/experiments/grassman_accuracy_2026_10_18/RELATIVE_GN_REPORT.md).

## 8. Combined20: requested integration and larger budget

The later explicit request combined relative GN with direct Armijo fallback,
up to three quadratic proposals and exact zero-operator elimination, and
increased the budget to 20 without an initial-smooth cap. These earlier effects
came from different datasets and are not additive.

Independent validation used seeds108800–108809, 30 paired datasets and
150 successful fits across five methods. The ablations were decisive:

| Validation comparison | EDR geometric ratio [95% interval] | Full-fit time median / p90 |
|---|---:|---:|
| Combined20 / original5 | 0.925311 [0.861556,0.983259] | 0.860129 / 1.189895 |
| Combined10 / accepted relative10 | 1.006697 [0.998577,1.016538] | 1.006664 / 1.038341 |
| Combined20 / combined10 | 0.996743 [0.974926,1.023747] | 1.306993 / 1.367969 |
| Combined20 / accepted relative10 | 1.003418 [0.986369,1.028379] | 1.275697 / 1.421531 |

Combined20 passed the historical gates against original5 and became the
requested public default. However, integration and extra compute did **not**
establish an added recovery gain over accepted relative10. Validation RSS
ratio versus original5 was 1.004027 and maximum orthogonality error `4.32e-15`;
only **1/140** final inner calls certified stationarity in validation
(**3/140** in development), with all outer stops at `h_min`. More accepted
steps did not mean monotone EDR improvement. R retained better case-median
recovery. Source: [combined report](../../benchmarks/experiments/grassman_accuracy_2026_10_18/COMBINED_REPORT.md).

## 9. Latest broad screen: nine candidates, no new winner

The next study froze current combined20 as the baseline. On the reused seed
90800, four unique cases generated 80 total fit rows across four pilot runs;
repeated baseline/R rows were not independent observations. The nine new
variants were an orthogonal-form full profile Jacobian, relative damping also
on smooth profiles, diagonal curvature planes, GN length interpolation, three
positive local ridges, and two short full-Jacobian Krylov proposals.

No candidate produced a joint accuracy/time improvement. Representative
candidate/baseline EDR and median fit-time ratios were `stable_jacobian`
**1.0000/1.032**, `diagonal` **1.1713/1.049**, `ridge6`
**1.0282/0.838**, and `krylov20` **1.0304/1.444**. Positive ridge was an
explicit estimator change and lost recovery. Removing inverse-squared
notation from the derivative did not itself improve precision: the saved
high-precision reference gave relative errors approximately `3.81e-9`
and `3.84e-9` for the old and proposed formulas.

All 80 fits completed, but only seven of 256 ADP inner calls certified,
all in the ridge pilot. No fresh development or held-out seeds were consumed
by this broad screen. Sources: [mathematical variants](../../benchmarks/experiments/grassman_accuracy_2026_10_22/MATH.md),
[screening report](../../benchmarks/experiments/grassman_accuracy_2026_10_22/REPORT.md).

## 10. Latest SVD correction, LoRA-like and enriched-space attempts

This follow-up used `SVD_solver.tex` more directly. Freeze `g_j` and solve a
horizontal additive surrogate

\[
q(\Delta)=\sum_j N_j\|r_j-U_j\Delta g_j\|^2+\mu\|\Delta\|_F^2,
\qquad Y^\top\Delta=0,\quad\operatorname{rank}(\Delta)\le k.
\]

The implementations tried: a literal greedy SVD correction control; all
`k*m` coefficients jointly optimized in the gradient SVD plane; two-factor
LoRA-like alternating refinement with capped matrix-free LSMR; and enrichment
by the full frozen-coefficient normal residual. The new full-k arm permits
`k=min(m,d-m)`; the literal old SVD control only permits rank `<m`, so it is
not an equal-rank full-k ablation.

The additive matrix is followed by polar normalization and full reprofiling.
At zero local ridge, exact arithmetic gives
`F(polar(Y+Delta)) <= sum N_j||r_j-U_j Delta g_j||^2 <= q(Delta)`
by a corresponding coefficient rescaling. This supports a descent proposal,
not an optimal rank-k solution or recovery guarantee. True-profile Armijo and
live final checks remain necessary, especially near numerical rank cutoffs.
The correction-rank constraint applies **before** retraction.

The first seven-case pilot had 77 successful fits on seed90800. The
`lora_k10` EDR ratio was **0.990**, median/p90 time **1.017/1.164**;
its d50 EDR was **4.7% worse**. `svd_k10` was faster, time **0.660**, but
EDR **1.014** and correlated-case EDR **15.1% worse**. LSMR repeatedly hit
its three-iteration cap. No independently validated LoRA improvement exists.
Sources: [correction mathematics](../../benchmarks/experiments/grassman_accuracy_2026_10_22/LOWRANK_MATH.md),
[pilot and later results](../../benchmarks/experiments/grassman_accuracy_2026_10_22/LOWRANK_REPORT.md).

## 11. Projected-Jacobian follow-up and the failed development gate

For a smooth profiled problem the residual derivative splits into two
orthogonal components, `J_full=J_perp+J_parallel`. At local least-squares
normality, `J_perp.T @ r=J_full.T @ r`, while
`J_full.T @ J_full-J_perp.T @ J_perp` is positive semidefinite. Keeping
`J_perp` preserves the gradient but drops curvature. This is an approximate
model, not an equivalent numerical formula for the full derivative.

An additional 28-fit pilot on the same seven seed90800 datasets gave
`projected10` EDR ratio **0.986** and time **0.723**. The frozen development
comparison then used 70 new paired blocks, seeds111800–111809, seven cases,
and four methods: baseline20, its budget10 ablation, projected10 and actual
R MEANMAVE. All **280/280** fits completed; pairing and source snapshots were
audited in the original study.

| Development comparison | EDR geometric ratio [seed-block 95% interval] | Full-fit time median / p90 |
|---|---:|---:|
| Budget10 / baseline20 | 0.9990 [0.9768,1.0241] | 0.742 / 0.941 |
| Projected10 / baseline20 | 1.0006 [0.9813,1.0223] | 0.737 / 0.885 |
| Projected10 / budget10 | 1.0017 [0.9886,1.0168] | 0.990 / 1.029 |

The prespecified accuracy gate required geometric EDR ratio `<=0.98` and
an upper interval endpoint below one, plus case/individual regression limits.
Both ten-step arms failed quality. Projected10's maximum case-median increase
was `0.0110`, and its largest individual increase was `0.1238`. Time, RSS
and orthogonality passed, but the budget control showed no pooled recovery
benefit attributable to the projected curvature change. Validation seeds
112800–112809 were deliberately left unused. No promotion followed.

This does not validate or refute LoRA: **there was no LoRA arm in the frozen
development comparison**. The follow-up contains 385 fits in total, but only
77 unique case/seed blocks: seven pilot blocks reused across 105 fits and
70 new development blocks across 280 fits. Source: [complete low-rank report](../../benchmarks/experiments/grassman_accuracy_2026_10_22/LOWRANK_REPORT.md),
sections “Projected-plane pilot” and “Frozen development gate.”

## 12. What remains in the code, and what remains unproved

- **Current production:** reduced core GN, guarded QR/SVD profiles, cached
  operator actions, relative intrinsic GN, value-based Armijo/quadratic
  rank-one fallback, spectral workspace fallback, exact zero-operator
  reduction and the requested 20-step default in `ADP/solver/grassman.py`.
- **Explicit experiments:** flat/full tangent GN in `grassman_exp.py`,
  separate block-rank-k entry point, structured/metric work in
  `grassman_research.py`, and latest correction/LoRA/projected candidates in
  `experiments/grassman_accuracy_2026_10_22/`. Their presence is not promotion.
- **Theory or separate statistical work:** spatial-field/Tucker models,
  covariance-power or GLS metrics, and broader homotopies are not established
  upgrades of the current global solver. A spatial Tucker proposal must not
  be described as an implemented low-Tucker-rank basis after retraction.
- **Backend work:** NumPy/CuPy support is distinct from the mathematical
  proposals above. These reports do not establish a universal GPU speed or
  recovery claim.

The repeated practical obstacles were real local rank loss, the cost of
additional operator passes, the gap between frozen-profile decrease and EDR
recovery, and gains explained by a smaller iteration budget. Failed gates and
unused validation seeds are part of the result. Historical gains quoted here
apply to their recorded data, estimator settings, budgets and machine; the
documentation task adds no new performance or recovery evidence.
