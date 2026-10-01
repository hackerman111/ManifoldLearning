# Current state
Active grassman-code-optimization-2026-10-01; PLAN active.
User explicitly assigned testing/benchmarks to GPT-6 Luna (luna_testing).
Deliverable ADP/solver/grassman_optim.py now permanently frozen. Original
hash070d0bb91962f23610b3e0a647e8f10fb417bac75e049305c3ef38223966c1a2 unchanged.
Prior completed-task state archived under experiments/grassman_optim_2026_10_01/PREVIOUS_*.

EXACT/NUMERICAL: batched full q*m Jacobian; solve sqrt(mass)/tan cache;
frozen-step damping design/RHS, AngleProfile u/guard, spectral projections.
Safe m<=2 reorthogonalized local QR with triangular normal_solve; original
SVD for m>2/p<m/cond^2*eps>=sqrt(eps)/tiny-scale. Same rank/smooth semantics.
Gradient U_j.T@residual followed by adjoint.T@weightedcoefficients; center
chunks bound scratch4MiB. _project transpose GEMM + contiguous small-result
copy; no additional U-sized array or dense d*d. Final certificate live U.
Last numerical-inert correction frees core mtw/wr/rhs before later scratch
allocations, preserving original workspace estimate. Final module574 lines.

Luna17 new tests pass preprojection; lead reviewed all independent/FD/rank/
ridge/layout/forcedchunk tests and micro raw data. Final Ruff format/check
and Pyright0errors0warnings pass (/tmp/grassman-tools, uv/.venv unavailable).
Initial invalid Pyright absolute include scanned/tmp; corrected relative
include plus explicit file fixed invocation; no project config changed.

Preliminary heavy six fullfits under consistent pre-memory-cleanup hash:
fullfit_final/{fullfit_runs.jsonl,fullfit_protocol.json}. Original/opt seconds:
seed0 30.194/28.125; seed1 31.839/28.900; seed2 32.728/31.023; median ratio.9315.
Same162 outer/h_min; projector diffs8.92e-15/1.09e-15/6.90e-10, quality matches;
all finalcore5 max_steps/unconverged. RSS delta+1656/-4940/-688KiB.
Earlier frozen QR/gradient speedup~1.7x, projector<2e-15; projection GEMM
~.33 vs1.08ms before contiguous copy. Fullfit includes shared outer costs.

Lead found harness frozen-projector orientation bug (row-basis needs B.T B;
fullfit correct), missing trace/config provenance, and end-buffered rows.
Luna corrects harness; preliminary artifacts retained. Final rich-protocol six
fullfits and correctly aggregated frozen repeats pending after numerical-inert
memory correction. Agent also runs original18 against optimized, focused
shared tests, then separate original fullfit cProfile to measure solver share.
No competing lead CPU work during timing. Next: audit final raw rows/hashes,
counts/numerics/protocol, update report/routes/PLAN/STATE/DECISIONS and finish.
ADP/{README,solvers,multi-index} include SRC-GRASSMAN-OPTIM. Dirty prior/user
SVD files and other changes preserved.

Final6 rich-protocol runs complete (fullfit_final_hash): ratio.94194 (5.8%
faster); times30.285/28.526,31.488/29.247,32.144/31.740. Final frozen ratio
.39676 (2.5204x), objective<=2.85e-14, projector<2e-15. Lead independently
recomputed frozen aggregation, correct orientation and source stability.
Luna35 optimized tests and127 focused shared pass; exact logs pendingreport.
Fullfit source/input hashes/config match paired. All810 accepted steps/seed
and162capped/unconverged calls match. Seed0 GN620/rank190 with profileeval
3390/3427; seed1 all810 nofallback; seed2 GN205/rank605, eval8916/8794.
Final objectives differ<=3.49e-8, finalprojector<=6.90e-10, but transient seed2
call106 objectives32.929535/29.493777 (10.43%); bothrankfallback5 steps.
This invalidates any claim of identical full trajectory; optimizer/objective
unchanged but numerical branch sensitivity must be diagnosed/reported.
Luna captures both call106inputs and cross-solves exact same snapshots,
untimed diagnostic. No solver edits. Next inspect diagnostic and decide
acceptance/correction, then complete docs. Fullfit cProfile originalseed0
solver6.724s/41.14s (~16.3% profiled runtime), diagnostic only; paths
fullfit_final_hash/original_fullfit_seed0_cprofile.{txt,json}.

Call106 diagnostic resolves transient concern: both solvers cross-solved exact
original and optimized captured snapshots; common-input objectives differ
<=1.66e-8, projectors<=2.30e-8, same cap/steps. Incoming basis3.60e-8 but
U8.60%/I12.99%/mass2.19% differ. All localrank2 yet conditioning guard false.
Accept NUMERICAL optimization; no identical outer trajectory claim or causal
claim about precise upstream amplification. Captures/summary/scripts under
call106_diagnostic/. Agent completing REPORT/VERIFICATION; no more code edits.
Next: lead audit diagnostic/report/static logs, final route/state/plan updates.
