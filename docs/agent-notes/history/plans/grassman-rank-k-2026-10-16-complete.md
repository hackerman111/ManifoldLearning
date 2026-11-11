# Plan: simultaneous Grassmann rank-k recovery

Task grassman-rank-k-2026-10-16; status **done**.
Goal: explicit block_rank_k; rank/error law and paired whole-fit EDR/time
versus core_gn and spectral. User authorizes Luna for routine work.
Non-goals: replace core defaults, change statistics/estimator, claim global F*.
Previous plan archived: agent-notes/history/plans/metric-spokoiny-2026-10-15-complete.md.
Existing dirty work is preserved.

Known evidence: spectral already uses simultaneous truncated-gradient geodesic;
flat_gn jointly optimizes angles with cached projections. Prior flat selection
failed; fresh seeds and disclosure of this overlap are required.
Class APPROXIMATE optimizer of identical finite-sketch profile. No ESTIMATOR
change. k limits selected tangent space, not EDR dimension m.
Invariants: float64, true local refit on trials/final, full-gradient stop,
orthonormality, angle bound, Armijo, rank/workspace guards, no d*d arrays.
Workspace bounds extra block/GN scratch, not input U or baseline profile state.

Exact read set: AGENTS/RTK, WORKFLOW, numerics/research/engineering contracts,
solvers.md; grassman.py profile/gradient/solve; grassman_exp flat/spectral/loop;
tests/test_grassman*.py; pyproject; Luna-cited ADP adapter/generators/harnesses.

## Bounded units and acceptance

1. [done] Audit overlap/math, freeze experiments/grassman_rank_k/PROTOCOL.md.
   Independent numerical F_ref is reference, never global minimum. Fresh seeds:
   development81600..81602; heldout81700..81704. Inner grid m2/3/5/8,
   d20/100/300, same frozen U/I/mass/Y. Freeze task train/validation split.
2. [done] Tests first; preserve grassman.solve API/default; separate
   grassman.block_rank_k forwards to exp method. Reuse spectral/flat with
   fixed/energy/cost rank, bounded joint work. Cost probes <=2 evaluations
   once per solve; cache winner as fallback. Validate rank/geometry, direct
   refit/finite differences, stress, full-gradient stop, fallback/workspace.
   Evidence:71 final solver/reference/counter/polish tests pass; scoped lint/type pass. Independent
   AST comparison proves core solve identical HEAD. Luna read-only diff audit
   reports no new defect; reference profile/FD/multistart independently checked.
3. [done] Luna bounded harness: serial fresh workers,
   pinned threads, hashes/environment/failures/RSS. Inner convergence and
   one-step k sweep plus k sequential rank-one; independent multi-start ref;
   fixed-time/evaluation/outer-budget comparisons.
   Source manifest frozen32 files,28 input snapshots. Lead split launcher into
   run/run_inner/common; scoped lint/type and nine-arm smoke pass. Full-suite
   collection blocked known catalog failures. No numerical source edits after
   freeze. SELECTION_RULES.md clarifies four optimized candidates before
   development numerical analysis; spectral block is identity control.
   Separate observational run_counted_inner wrapper (outside numerical freeze)
   counts all exact profile trials, including rank-one Schur values;5 counter
   tests and worker smoke pass. Separate source manifest before inner stage.
4. [done] Paired ADP multi-index and original hard Spokoiny full fits,
   all baselines and fixed/adaptive blocks. Development then frozen choice
   and heldout5 seeds. Retain regressions; normalized projector EDR,
   fit/solver wall, success/iterations/orthogonality. Success median fit<=.8
   core, no worse median EDR/failure rate; compare spectral too.
   Timeout180s fit/60s frozen. Keep all failures.
   Development54/54 ok, hashes/threads/geometry pass. Frozen candidate block1:
   paired median fit/core .99914; no candidate passes .8 gate. Original MI
   case core/optimized all30 updates/fit use rank guard, same recovery;
   spectral does not use that guard and has different/worse EDR. Retain and
   disclose limitation. selection.json pins dev SHA before heldout.
   Heldout30/30 completed ok; pairing/selection hashes pass, max final
   orthogonality2.28e-15. Inner792/792 completed, no retuning.
5. [done] Lead raw pairing/reference audit; fit c on training only,
   heldout prediction violations; plots/tables/raw REPORT.md. Focused/shared
   solver tests, scoped lint/type, full suite once documenting unrelated
   failures. Update routes/STATE/DECISIONS; mark done with evidence.
   Reference quality gate: keep original1000-step CG evidence. For endpoints
   missing tol1e-9, add separate independent local-chart L-BFGS polish,
   max5000/start, three same starts, then independent CG certificate polish.
   Tau0 basis-invariant profile; B=Y0+(I-Y0Y0.T)Z has full column rank;
   chart gradient is horizontal-to-Y0 Euclidean envelope gradient. No d*d.
   Separate source/inputs manifest and raw JSONL; only after main timed stage.
   Test direct chart gradient by FD before runs. Preserve original results,
   report reference sensitivity/certification, no solver/rank retuning.
   Main792/792 ok incl24refs/768solverrows. Lead finalaudit32sources28inputs,
   coreAST unchanged, allpairing/counters/threads pass; maxorth4.87e-14,
   no positive history drift. Original19/24refs meettol; separate5/5 supplement
   improves all5, certifies3 more, total22/24. Two chartFD/gauge/reference tests
   pass before supplement. Strict heldout law c=.9779/.9190 (optimized/spectral),
   logRMSE=.1837/.2269; finite trend, no universal guarantee. All200 adjacent-k
   comparisons improve F; preserve340 lower-F/worse-EDR pairs. Final heldout
   fit/core median .99963,95% bootstrap[.98914,1.01554], no20% gain. Report,
   raw JSONL/CSV/plots, exact source snapshot and hash manifest saved. Default
   core remains unchanged; experimental alternative only. Full-suite collection
   blocked3 existing multiv2 catalog errors, documented in verification.md.

Stop/revise on wrong generator, uncontrolled threads, refit/geometry mismatch,
nonfinite or objective increase. No retuning after validation; no default
promotion. No performance theorem from finite empirical samples.
