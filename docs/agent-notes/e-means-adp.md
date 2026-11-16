# e-means-inspired optimizer of original anisotropic ADP

Research task2026-11-13; isolated APPROXIMATE solver prototype, public ADP
source/defaults/API unchanged. This is not an e-means clustering replacement.
Completed NEGATIVE selection:320/320 successful fits, mean EDR.4080->.4374
for mini+EMA at median paired time ratio.8864. Quality gates fail; validation
untouched, no promotion. Full evidence: experiment REPORT.md.

Completed followup N_J=n (mini/EMA quality selection failed): opt-in
--all-centers, original4 arms plus bounded-J
original20 control on same NEW selection1134400..409, conditional validation
1135400..409. Candidate/settings/gates unchanged. ALL_CENTERS_MATH.md contains
fixed-geometry variance proof/costs; ALL_CENTERS_REVIEW.md records pre-freeze
review. Evidence under verification/e-means-all-centers-20261113/campaign.
ALL_CENTERS_REPORT.md is the measured followup report:400/400ok,8396audit
checks/2000hashes. All20 meanEDR.332018 vs bounded.395326 (time3.169533),
but worst all-J harm+.431130. Mini/EMA means.359664/.358669 at paired times
.831401/.843886; worst both+.350098, quality gates fail, no validation.
No universal all-J advantage, no candidate promotion or adaptive EDR theorem.

- [README](../../benchmarks/experiments/e_means_adp/README.md): usage/replay,
  source/data snapshots, measurement semantics and lifecycle.
- [MATH](../../benchmarks/experiments/e_means_adp/MATH.md): four families,
  input-relative exact-arithmetic safeguard, gauge-correct projector EMA,
  operation/memory/conditioning bounds, rank-cutoff domain and EDR counterexample.
- [REVIEW](../../benchmarks/experiments/e_means_adp/REVIEW.md): independent
  adversarial proof/code/runner audit and resolved findings.
- [REPORT](../../benchmarks/experiments/e_means_adp/REPORT.md): final evidence,
  gates and exact remaining gap.
- Frozen evidence: benchmarks/results/verification/e-means-adp-20261113/
  campaign/{PROTOCOL.json,environment.json,source,selection,completion.json}.
  Eight cases x10seeds x4arms=320selection fits. Validation is conditional on
  passed selection gates AND independent read-back audit; no default promotion.

Sources (2026-11-13):

| Path | Range / responsibility |
| --- | --- |
| benchmarks/experiments/e_means_adp/candidate.py |39-68 low-rank EMA;69-end frozen-profile wrapper |
| benchmarks/experiments/e_means_adp/study.py |64-115 fresh worker;116-212 gates;213-295 freeze;296-430 run/validation;431-end CLI |
| benchmarks/experiments/e_means_adp/audit.py |17-67 independent gates;68-end source/data/basis/manifest/config/actual-J audit |
| tests/test_e_means_adp.py | reference/gauge/rank/corrupt proposal/RNG/worker/tamper tests |

Two epochs of4 one-step batch calls; optional EMA is an extra starting point.
Original solver then runs4 full-center steps. Input/last/EMA/full final proposal
are evaluated with the full original profile. Returned coefficients and
top-level gradient/certificates are recomputed on all centers. Nested
full_finalization describes the ATTEMPT and may have been rejected.

Create a NEW MiniBatchSolver per fit; its RNG advances between outer calls.
J/b mass scaling changes smooth-GN effective damping, explicitly pinned in
protocol. Kernel/local moments/alpha/h/best-step formulas stay original, but
new basis estimates change realized adaptive weights at later scales.
No theorem of superiority to original20, global convergence or EDR recovery.
No interval-arithmetic certification across numerical rank boundaries.
