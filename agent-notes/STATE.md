# Current agent state

Completed bounded task: `manifold-curvature-recovery-2026-09-28` in `PLAN.md`.
The completed grid-repair plan/state were archived under
`agent-notes/history/`; its code and artifacts are still dirty and must be
preserved. The earlier generalization-repair task remains archived as pending,
but its 360/360 development selection and 120/120 flat validation are now
audited. All 11 source hashes in the validation manifest match live files;
the outcome is in `docs/experiments/manifold_generalization_2026-09-28/repair.md`.

Established mathematical limit: with scalar `f=||z||²/2`, a smooth rotation
`z -> R(x)z` preserves every response but can alter `row(Dz)` for `m>=2`.
Thus curved generator-chart distances are diagnostic only. For `m=1`, off
`z=0`, the chart row is the identifiable gradient direction. For flat profiles
the fixed global EDR space is the identifiable target under that model class.

Measured prior mechanisms: relative spectral weighting loses the weak third
direction on flat `m=3`; explicit unit weighting passed the paired development
gate 10/10 but increased support failures elsewhere. Broad graph pooling of
distinct tangent spaces biases curved initialization even with exact source
gradients. Nearest-center query error has a discretization floor. Grid work
found 21 rank failures in 82 attempts and only 2/40 identifiable recoveries.

Flat held-out result: broad/unit recovers 30/30 (m=1,2,3 at both noise
levels), zero failures; broad/relative recovers 14/30. Local/unit recovers
only 19/30 with five numerical failures. Unit spectrum is validated for the
frozen flat broad profiles, not as a general default or curved repair. Here
`c=0` is a global m-index model passed to the manifold estimator as a
constant-EDR control. With `c>0`, the scalar response is still a global
multi-index function of the first 2m rotated coordinates; the local
generator chart varies but is not thereby identified.

Complete 240/240 curved development diagnostic under
`docs/experiments/manifold_curvature_2026-09-28/`: broad/unit completed
60/60 but 0/60 passed observable-gradient containment, despite certified
CG residuals <=1e-6. Exact-gradient graph counterfactual also passed 0/60;
local/unit had 21/60 rank failures. Ordinary local-linear and 60-neighbor
quadratic gradients were poor at worst centers. This rejects a simple
spectrum/pilot/graph-size repair. For m=2/3, full chart remains unidentifiable.

One isolated structural gradient candidate, Gaussian-Stein top-2m followed by
a CV-selected quartic scalar surrogate, failed 12/12 all-center gates on the
first frozen development seed; global active-subspace max sine .321–.583.
It remains a negative experiment artifact, not production code. No untouched
curved validation was run after this stop gate. Final curvature manifests
match all seven diagnostic and two prototype source hashes; 240/240
diagnostic semantic rows agree exactly after the final type-only edit.
See `report.md` in the curvature directory.

Relevant route: `agent-notes/ADP/manifold.md`; experiment protocols and
diagnosis under `docs/experiments/manifold_generalization_2026-09-28/`;
`experiments/manifold_generalization*.py`; live `manifol_engine` fit/weights,
graphs/optimization and `ADP_Manifold` API. Production defaults and prior
results remain untouched.

Verification: `tests/test_manifold.py`, `tests/test_manifold_grid.py`, and the
new independent gauge-reference test passed 20/20; Ruff check and format
passed on the new scripts/test, Pyright reported 0 errors, and
`git diff --check` passed. The route `agent-notes/ADP/manifold.md`,
`experiments/README.md`, and durable decisions now point to the negative
curved result and validated flat broad/unit result. Prior dirty work remains
intact. A future curved-recovery study needs a new identifiable target model
or quantitative smoothness/signal assumptions and a new frozen protocol.
