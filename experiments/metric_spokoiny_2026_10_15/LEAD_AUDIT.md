# Independent audit of the hard Spokoiny study

The original numerical library and generator remained unchanged throughout
this study. The frozen protocol specifies development seeds0/1/2, budgets5/20
for full fits and5/100 for frozen objectives, and a validation gate. These
are development measurements without timing repetitions or confidence
intervals, not an estimate of performance over all datasets.

## Mathematics and independent checks

MATH.md derives the coefficient-metric rank-constrained linear descent and
its limits as a curvature model. The q=m=2 proposal preserves the left
gradient span at full rank; q=1 introduces a further restriction. A rational
moment derivation and independent Luna quadrature establish population OPG
=(5557/1155)I2 for this exact rotated-square polynomial generator. This does
not equate population OPG to fitted coefficient R or prove that a metric
cannot help on finite data.

The post-timing audit_proposals.py checks all nine original core inputs:
negative directional derivative, actual q2, coefficient spectrum, cap/floor
activity, and mass-rescaling identity Z(100G,100mass)=Z(G,mass)/10. Its source
hash is saved in proposal_audit.json. Both angle caps and default floors are
inactive on all sampled inputs; R condition numbers range1.057..4.801. The
maximum occurs at a local rank-boundary input. Initial proposals cannot
establish the cap behavior throughout the optimizer trajectory.

## Raw results and provenance

The independent stdlib audit is saved as results_repaired/lead_audit.json.
It checks18 current source hashes, unchanged protocol hash, old runner bytes,
and that run.py is the only changed manifest source between original and
repaired launches. Fourteen imported files (fullfit/capture JSONL plus12 NPZ)
are byte-identical. Original90 pre-solver CLI errors are retained separately;
they are launcher failures, not numerical failures.

Successful repaired evidence is9 budget5 fits,4 diagnostic captures,90 frozen
solves and6 budget20 fits. All frozen (seed,call,arm,budget) keys are unique,
covering the complete predeclared grid. Paired U/I/mass/index hashes match;
full-fit data/truth/init/first-statistic hashes match across arms for every
seed and budget. All observed BLAS pools and configured thread counts equal1.
All frozen objectives are nonincreasing; maximum orthogonality error across
fits and frozen solves is6.22e-15. The saved plot was inspected: full-fit
recovery and frozen full-gradient certificates agree with raw rows.

At budget5 metric2/core paired median whole-fit time is.981649, solver time
.857432. Seed1 projector regression is5.666e-4, exceeding1e-4; independent
gate recomputation is false. Validation is explicitly skipped and seeds3..7
are absent. At budget20 median whole-fit ratio is1.015490, solver1.050700;
recovery is nearly identical. The five jointly smooth-certified frozen pairs
have median time ratio2.039975; no speed-to-convergence claim uses the other
endpoints. Zero budget5 frozen endpoints are certified. Two middle inputs
have local rank loss; small gradients there do not certify smooth stationarity.

## Supplementary profile and conclusion

Two untimed profiles use the same smooth-certified seed1/call160 input. Core
accepts48 steps, metric2 accepts36. They reach essentially equal objectives
and recovery. Core has50 _profile and48 _polar calls; metric2 has249 and247.
The existing research line search performs full profile/retraction on each
trial, explaining substantial extra trial work on this specific input.
Profiler elapsed times are excluded from benchmark timing claims, and this
single profile does not establish the bottleneck at every phase.

Final focused checks: Luna reports10 tests passed in0.31s and scoped Pyright
zero diagnostics across five experiment modules and the test. Ruff format
passes all six. Lead split one long audit string after E501; scoped Ruff
now passes that module as well as the five already-clean files. The exact
pre-format executed audit source is archived in
results_repaired/proposal_audit_source.py.txt and matches the JSON source
hash; only equivalent adjacent string literals changed afterward. No raw
numerical artifact was rewritten. Profile-driver CLI help also passes.

The current implementation has no demonstrated robust recovery or wall-time
advantage on this hard benchmark. Rank1 is not a reliable replacement. Keep
the metric method experimental. Calibrating proposal scale and reducing
profile work per rejected trial are plausible separate future hypotheses,
requiring a new protocol and untouched validation; neither is implemented
or measured here. Focused tests and scoped lint/type evidence are recorded
in verification.md and harness_verification.md.
