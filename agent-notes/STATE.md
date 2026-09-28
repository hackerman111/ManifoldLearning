# Current agent state

No active task. The completed paired heavy-point comparison is recorded in
`PLAN.md` (`svd-vs-hybrid-heavy-30s-2026-09-28`). The prior small/medium
comparison remains in `experiments/svd_vs_hybrid_2026-09-28_v2/` with its
archived plan at `agent-notes/history/plan-svd-vs-hybrid-small-2026-09-28.md`.

The heavy workload exactly reproduces saved point `mi-spokoini-m2-dhigh`
(Spokoini, m=2, d=50, n=800, N_loc=10, N_lin=100, N_J=800, N_phi=10,
a=exp(1/100), h_min=1, local initialization, last step, no outer cap,
solver max_steps=5). Paired seeds 0-2 completed for SVD rank 1 and HYBRID; no
errors or timeouts occurred. None of the six fits met the 30-second target.
Median SVD/HYBRID runtime was 47.56/58.99 s, normalized projector distance
0.0951/0.0481, and process peak RSS 98,172/125,848 KiB. SVD used less time
and memory in all pairs; HYBRID had lower projector distance in all pairs.
The objectives differ, so recovery is descriptive and not an equivalent-inner-
objective comparison.

All fits reached `h_min` after 162 outer iterations. HYBRID's convergence
diagnostic was false in all runs. SVD's rank-one inner loop hit its 20-step cap
on 22, 49, and 43 outer updates. See
`experiments/svd_vs_hybrid_spokoini_dhigh_30s_untraced_2026-09-28/REPORT.md`
for interpretation and per-seed results; `runs.jsonl` and `protocol.json`
contain raw metrics and the frozen protocol. The runner is
`benchmarks/svd_vs_hybrid.py`. Process high-water RSS was collected; traced
Python peak was disabled to avoid distorting this timing-target run.

The older saved runs at this point used `solver=lsmr`, not HYBRID; their times
(37.20, 34.53, 35.39 s) are context only and are excluded from the paired
comparison.
