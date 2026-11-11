# Lead audit

Reopened live solver, all17 new numerical tests, original18 tests and benchmark
worker/aggregation; independently recomputed raw final paired results. No
estimator/budget/tolerance change in optimized module. Original solver hash
070d0bb91962f23610b3e0a647e8f10fb417bac75e049305c3ef38223966c1a2 remains unchanged.

Final frozen rows (`frozen_final_hash`): independent pure-Python projector
calculation uses B.T B for row-bases and confirms differences
1.3809e-15/1.9361e-15/1.8959e-15; objectives -2.8422e-14/0/0. Median paired
speedup2.520431. Every solver before/after source hash matches. Lead found and
requested correction of initial harness frozen B B.T comparison, which only
compared identity matrices; earlier artifacts remain and final corrected
protocol uses actual projectors. Fullfit orientation was correct throughout.

Final fullfit rows (`fullfit_final_hash`): all paired input hashes and effective
configs match; before/after source hashes match. Recomputed median paired
ratio0.9419390633 (5.8061% runtime reduction). RSS original/optimized KiB:
114268/114360,117308/111436,107352/114392; median delta+92KiB, range
-5872 to+7040KiB. This is fresh-process Linux high-water RSS, not isolated
solver allocation; no stable memory reduction claim. Bounded scratch is
an implementation property, distinct from noisy process RSS.

All810 accepted inner steps/seed and162 capped/unconverged calls match.
Profile evaluations differ on seeds0/2 (3390/3427 and8916/8794); GN/rank
fallback totals match. Original fullfit cProfile seed0 has6.724s in solver
of41.144s total (~16.3%), calculate_statistics25.912s. Instrumented timing
is diagnostic and is not compared with uninstrumented runtime.

Important numerical limitation: seed2 call106 profile objectives
32.9295351816/29.4937765231, a10.4337% transient difference; both are
rank-one fallback calls. Final projector difference<=6.90e-10 and final
profile objective delta<=3.49e-8 do not imply identical outer trajectories.
Call106 capture/cross-solve diagnostic resolves the concern: on exact shared
inputs objective differences<=1.66e-8, projector differences<=2.30e-8, same
5 steps/status. Captured incoming basis difference3.60e-8 accompanies U8.60%,
I12.99%, mass2.19% differences. Both localrank2 but conditioning guard false.
Outer input sensitivity is measured; the exact upstream amplification
mechanism is not isolated. Accept numerical implementation with this limit;
see call106_diagnostic/summary.json and final REPORT.

Configured Ruff format/check and targeted Pyright0errors0warnings pass for
final solver via /tmp/grassman-tools. Initial Pyright invocation used invalid
absolute include and scanned/tmp; corrected relative include+explicit file
completed cleanly. Project config unchanged; uv/.venv absent, live system
interpreter used. See VERIFICATION for Luna test/benchmark tooling logs.
