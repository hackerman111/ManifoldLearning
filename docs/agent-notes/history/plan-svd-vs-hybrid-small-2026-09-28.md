# Compare SVD multi-index training with HYBRID on small and medium points

task_id: svd-vs-hybrid-2026-09-28
status: done
change_class: APPROXIMATE comparison of explicit rank-r solver variant

Completed paired full-fit comparison: 2 cases, 3 seeds per case, isolated
workers, float64 and one BLAS thread. Report and raw evidence are preserved at
`experiments/svd_vs_hybrid_2026-09-28_v2/`. SVD was 3.18x slower on the small
case and 2.75x slower on medium; process peak RSS was close; median projector
distance favored SVD on the sampled pairs. Objectives differ and all runs hit
the three-step outer cap, so the result is exploratory. The invalid initial
pilot is preserved at `experiments/svd_vs_hybrid_2026-09-28/`.
