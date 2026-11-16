# e-means ADP with all observations as centers

Task e-means-all-centers-20261113; status **done**. User asks whether N_J=n
changes the preceding negative result. This is a new frozen research regime,
not retuning the failed campaign. Preserve every previous dirty file/output.

**Goal:** Compare original20/full6/mini/mini_ema with N_J=n on identical data;
include original20_bounded on those same fresh data to separate the practical
effect of using all centers from optimizer changes. No public source/default
changes or commit. No theorem of EDR improvement from having more centers.
**Evidence:** Prior320-fit selection with J200/300 failed quality gates;
report and immutable campaign retained. Using all centers changes initializer,
bandwidth/alpha geometry, random-direction layout and the finite-center target.
**Class:** ESTIMATOR/configuration change between bounded-J and all-J regimes;
APPROXIMATE optimizer comparison within the common all-J regime.
**Hypotheses:** Larger mini-batches (n/4 instead of Jbounded/4) may improve
proposal quality. Full-center coverage removes center-subset sampling for a
FIXED geometry, but cannot prove adaptive EDR improvement.
**Invariants:** Original public ADP, kernel/moments, other settings and exact
candidate code; four batches,two epochs,four full final steps, cosine EMA.
Same8cases,n800/d20/m3 and n2000/d50/m3, N_phi12,N_loc20,N_lin2d.
One BLAS thread;120s/2048MiB polled per-fit cap. Raw failures stay visible.
Selection1134400..409; new validation1135400..409 conditional on unchanged gates.
Old heldout1132400..409 stays untouched. No candidate/budget/threshold retuning.

**Read set:** AGENTS, prior PLAN/STATE, e-means route/README/MATH/REVIEW;
contracts/research and WORKFLOW; e_means_adp/{study,audit,candidate}.py;
tests/test_e_means_adp.py; index_fit.py center selection/initialization;
previous frozen protocol and source hashes. No broad scan.

**Bounded units / evidence:**
- [x] Implement only an opt-in all-center experiment profile with fresh seeds,
  explicit bounded-J control and frozen job/config audit; test actual N_J and
  reject job tampering. Old bounded profile remains compatible.
  Evidence:79 focused passes, Ruff clean, Pyright0 errors/warnings;
  live old-campaign audit7348 checks/320fits passes without rewriting it.
- [x] State fixed-geometry finite-population variance identity and its proof,
  costs O(n^2) distance storage/O(n p d) U, exact adaptive/statistical gap;
  independent adversarial review of changed harness before freeze.
  Evidence:ALL_CENTERS_MATH.md and ALL_CENTERS_REVIEW.md; reviewer found no
  blocking defect and independently confirmed32 tests/old-campaign audit.
- [x] Freeze400 selection fits (8x10x5), source/environment/data/config/gates;
  run serial isolated workers, audit completeness, actual configurations and
  independent metrics/gates. Validation only if mini/mini_ema pass selection.
  Evidence:400/400ok,8396audit checks/2000artifacts; both candidates fail
  frozen quality gates. No validation generated, completion validated=false.
- [x] Report all-center vs all-center quality/time/RSS/certificates, separate
  bounded-J control effects; independently review claims, final checks,
  update routing/STATE/DECISIONS and mark done.
  Evidence:ALL_CENTERS_REPORT/REVIEW,analysis.json/by-case.csv/final-audit.json;
  independent final reviewer reproduced all8396checks/2000hashes and main
  contrasts, no blocking defect.79final tests/Ruff/format/diff pass; Pyright0
  errors/warnings on unchanged frozen Python.68public source hashes unchanged.

**Outcome:** All-J original20 meanEDR.332018 vs bounded.395326 at paired
time3.169533, but14/80 blocks worsened (worst+.431130). All-J mini/EMA means
.359664/.358669 versus original.332018, times.831401/.843886, worst+.350098;
both quality gates fail. No validation, no retuning or promotion. Scoped
finite-population identity proved; adaptive EDR/convergence gap remains.

**Stop conditions:** Keep past failed outputs immutable. No validation after
failed selection; no using old heldout for this new regime. No recovery claim
from a profile/variance identity, inner stationarity or a positive case alone.
If all-J exceeds resources, record failure rather than silently subsample.
Previous plan/state archived in history/{plans,state}/
e-means-all-centers-20261113-preexisting-*.md.
