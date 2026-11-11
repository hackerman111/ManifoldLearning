# ADP moment integration routing

Completed study artifacts: benchmarks/results/verification/adp-moment-integration-20261030/.
Read REPORT.md, THEORY.md (exact scope/proofs/costs/gaps), PROTOCOL.json,
R1_AUDIT.md and AUDIT.json. Separate from global weak-moment/profile study.

Source: benchmarks/experiments/adp_moment_integration.py:35-110 evaluator/proposal;
:110-end fit adapter (initializer and solve/statistics hooks). Fixed m=2;
CPU/new/all only, no experimental_centers; private patch process-local and
restored. New experiment only; ADP production/defaults unchanged.
Runner benchmarks/experiments/integration_study.py records original config, data/code
hashes, sequential fresh worker timing/RSS and seed-cluster gates. All source
is frozen and archived in SOURCE_SNAPSHOT.zip; do not edit and rerun under
same task id. Tests: tests/test_adp_moment_integration.py (four contracts), plus
weak_moments/local_polynomial/shared_polynomial:38 passed.

R3 selection240/240 successful: residual_start/adp ratio.9425 upper1.0305FAIL;
proposal/(20+20 restart) ratio1.0034 upper1.0646FAIL. No validation for these
arms. Only6/176 proposals accepted; strict objective decrease not EDR evidence.
R4 WLS2/WLS1 validation80/80 successful, ratio.44798 percentile[.40309,.49591],
time1.3207,RSS.9998,worstpair delta-.09369PASS. Same residual start/ridge1e-4,
local degree change. On Gaussian quadratic cases meanEDR.5623→.2607,recovery
0/16→12/16. No held-out comparison with ordinary ADP. Anisotropic stress poor;
most inner solves budget-limited. No global convergence/universal theorem.

First default window is Euclidean and independent of start mathematically;
later windows adaptive. Moment proposals only preserve frozen inner objective.
Prototype wrapper retains O(Jn) weights, WLS O(Ed) neighborhoods and O(d^2)
chart complement; no production streaming/memory improvement claim.
Audits recompute data/projector metrics/moment initializers/gates/hashes, not
iterative fits or per-stage U/I objective replay (operators not saved).
