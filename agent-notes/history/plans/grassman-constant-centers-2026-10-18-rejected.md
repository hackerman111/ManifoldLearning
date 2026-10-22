# Objective-constant centers in Grassmann

Task `grassman-constant-centers-2026-10-18`; status **done (candidate rejected at R3)**.
Goal: improve solver recovery at original five-step budget by preventing exactly zero U centers from triggering global GN rank fallback; retain stable localized scalar fallback from the earlier experiment.
Non-goals: omit nonzero low-rank centers, relax true rank-boundary guards, change estimator/statistics/ridge, rerun or tune the rejected ten-step candidate.

Evidence: R0 diagnostics in experiments/grassman_accuracy_2026_10_18/ show rank-zero U centers alone block GN on some real R-SDR calls (e.g. frozen008/009). Profile/coefficient roundoff is tiny on ordinary inputs. Ten-step localized candidate reduced EDR9.1% but cost36.8% in frozen development and was rejected. Its heldout92800..92809 remains untouched; archived plan under agent-notes/history/plans/grassman-accuracy-local10-2026-10-18-rejected.md.

Hypothesis: for U_j=0 exactly, F_j(Y)=mass_j||I_j||² is constant and g_j=0 is minimum-norm at any Y. Removing these centers from the solve, and restoring their coefficients/constant loss, preserves the objective and gradient while allowing the original GN test on informative centers. This is EXACT reduction/NUMERICAL evaluation plus APPROXIMATE optimizer path change; keep explicit opt-in until paired validation.
Invariants: same finite-sketch profile, tau, data, outer settings, direction law, five-step cap, final live gradient; original scale=max(1,sum mass I²) preserved in tolerance and diagnostics; all centers returned in original order; no ambient d*d matrix. Nonzero deficient centers retain rank-one fallback.
Read set: prior MATH.md/PROTOCOL.md and diagnostics; experiments/grassman_accuracy_2026_10_18/{candidate,constant_centers}.py; ADP/solver/{grassman,LSMR}.py; experiments/r_sdr/{worker,runner,design,protocol}.py; focused accuracy/R-SDR tests.

- [x] R0: identify exact constant-center cause from existing diagnostics; archive failed previous hypothesis.
- [x] R1/R2: derive constant/tolerance restoration, independent audit/reference tests; exploratory pilot90800 only; freeze candidate/config before fresh development.
- [ ] R3: 30 paired datasets, cases d-5,d-20,d-50, development93800..93809, baseline/candidate/real R MEANMAVE; same gates as previous protocol: EDR geometric ratio<=.95; per-case median paired delta<=.005; median/p90 fit ratios<=1.10/1.25; median RSS ratio<=1.10; no extra failures; orthogonality<=1e-10. Persist all rows, sources, input hashes.
- [ ] R4: only R3 pass permits heldout94800..94809 with frozen algorithm, settings and same gates. No tuning on R3 failures.
- [ ] R5: only validated result permits explicit opt-in public integration; verify replay, affected notes/STATE/DECISIONS, report and mark plan done. Failed R3 closes this candidate and leaves validation untouched.

Verification: independent constant objective and gradient identity, tolerance scaling with nonzero constant responses, coefficients/mass row order; unchanged no-zero input path; existing rank-boundary tests; small direct reference plus paired full-fit quality/time/RSS and failures; scoped Ruff/Pyright, git diff --check.
Stop: derivative assumption changes, objective/certificate-scale mismatch, nonzero omitted center, source/input drift, forbidden allocation, frozen failed gate. Do not reuse consumed development seeds91800..91809 to select this candidate.

Frozen details: experiments/grassman_accuracy_2026_10_18/CONSTANT_PROTOCOL.md. Independent constant/reference tests pass.

R3 complete90/90, hashes matched; EDR ratio.9997287 fails.95 (CI[.98048,1.02375]); median/p90 time.87490/1.15042, RSS1.00301 pass. Heldout94800..94809 untouched. Evidence benchmark_outputs/r_sdr/grassman-constant-dev-2026-10-18/analysis_summary.json.
