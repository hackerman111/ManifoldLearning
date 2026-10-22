# Plan: experimental adaptive centers P0-P3

Task adaptive-centers-2026-10-17; status **done**.
Goal: audit SVD/adaptive_centers_anisotropic_adp_edr_report.tex; implement
opt-in P0 spectral floor, P1 exact projected neighbor search, P2 cheap
projected statistics/prefilter, P3 LogDet plus mass coverage; paired recovery,
full-fit time and RSS evidence with Luna benchmarking. Preserve defaults.
Previous completed plan: agent-notes/history/plans/grassman-rank-k-2026-10-16-complete.md.

Classes: P0/P2-selection/P3 ESTIMATOR; P1 EXACT modulo roundoff; P2 moments
EXACT for current weights. No solver/default/ridge/density changes. No claim
that surrogate selection or scalar quadrature guarantees EDR recovery.
Known: shared fit_index uses raw normalized spectrum for weights/alpha;
Epanechnikov K(q)=(1-q*q)+, q is squared localized distance. Existing alpha
compacts after projection but weight path computes all pairs. Existing
box engine is kernel-specific and cannot encode Epanechnikov weights.
Invariants: float64, common pool/init/sketch streams, floor only localization,
alpha uses full pool, canonical spectrum raw; sparse O(E) weights, bounded
row scratch, no J*n*d or d*d. Coverage nonnegative fixed observation masses;
normalized logdet(I+sum uuT/eps), fixed u during greedy; no inverse; optional
PoU masses unnormalized. Full-pool S prediction selects best on common pool.

Exact read set: AGENTS/RTK, contracts/numerics,research,engineering; WORKFLOW;
ADP/multi-index,index-pipeline routes; report lines180-270,330-410,510-580,
659-680; ADP/core/{ADP_Config,ADP_Statistic}.py, ADP/__init__.py;
engine/common/{index_fit,weights,calculus,statistic,box_kernel}.py;
relevant tests/config; Luna-cited harness/generators.

## Bounded units and evidence
1. [done] Formula audit, small independent reference tests, freeze protocol.
2. [done] Config plus separate experimental geometry/moments/selection
   modules; minimal shared fit hooks. Exact weight/statistics/alpha comparisons,
   boundary/zero-eigen/offset/degenerate/validation/PoU/greedy tests.
   Evidence:24 new tests pass,83 shared non-manifold tests pass, scoped
   Ruff/Pyright clean; MATH.md records formula corrections and boundaries.
3. [done] Luna owns experiments/adaptive_centers only: frozen protocol,
   isolated one-thread workers, fresh development/heldout seeds, raw pairing
   hashes, failures, EDR/orthogonality/stops/inner certificates/time/RSS.
   Smoke10/10 ok (92799), common input/init hashes, threads1, coefficient
   mapping valid; development120/120 ok under frozen source snapshot.
   Controls isolate floor, search, projected-statistics overhead, score,
   equal-J random, pure LogDet, LogDet+coverage, optional PoU. Lead audits.
   Lead raw audit:74 source hashes/common inputs/threads/mapping pass;
   no global gate passes, selection=null, heldout untouched. All inner
   certificates unconverged. P1 divergent-pair replay preserves same-state
   weights/alpha/I/U; full adaptive trajectories differ under roundoff.
   Supplement:24workers under separate manifest,exact weights/support/alpha;
   n10000 localization time.493weights/.414alpha+weights baseline,n500 slower.
   Does not establish full-fit improvement. Reports/raw/plots preserved.
4. [done] Final27 feature tests included in86 shared passes; scoped Ruff/
   Pyright clean. Full suite blocked by existing catalog/manifold issues.
   Post-timing seed=None regression fixed; original integer streams and two
   fullfit hashes preserved. Immutable source snapshot plus explicit patch
   overlay audited by verification/lead_audit.py. Routes/STATE/DECISIONS current.
   No default promotion; all findings conditional on measured solver schedule.

Stop/revise on false-negative search, wrong formula/generator, pairing/thread
mismatch, rank/nonfinite hidden by regularization. Retain failures; no tuning
on heldout. Benchmark limits and feasibility stops reported explicitly.
