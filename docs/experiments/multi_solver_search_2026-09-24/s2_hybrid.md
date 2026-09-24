# S2 — существующий HYBRID на frozen задачах

Бюджет зафиксирован до запуска в `hypotheses.md`: 12 задач × LSMR/HYBRID = 24 solver trials, cap 80 и лимит 60 s на trial. Команда:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.multi_solver_compare \
  --frozen-dir benchmark_outputs/diagnostic/multi_solver_search_s1_20260924 \
  --output benchmark_outputs/diagnostic/multi_solver_search_s2_20260924.json
```

Результат и все failures — в [JSON](../../../benchmark_outputs/diagnostic/multi_solver_search_s2_20260924.json). Порядок методов чередовался между задачами, перед серией оба были прогреты на отдельной малой задаче. Ни один frozen hash не изменился. Все 24 trials завершились, timeout и numerical failure не было. Общий reference пересчитал конечный objective и stationarity независимо от solver. У всех конечных линейных поправок normal-residual ratio ≤0.1.

| Точка | Пар | Медиана парного HYBRID/LSMR времени | Сумма времени LSMR/HYBRID | Сертификаты LSMR/HYBRID | AO steps LSMR/HYBRID | Krylov iterations LSMR/HYBRID |
|---|---:|---:|---:|---:|---:|---:|
| d10 | 6 | 1.919 | 1.043 / 1.925 s | 6/6 / 6/6 | 113 / 113 | 2,855 / 0 (cached SVD) |
| n1000,d100 | 6 | 1.037 | 47.238 / 46.642 s | 3/6 / 3/6 | 427 / 427 | 23,042 / 15,233 |

На d100 HYBRID сделал на 7,809 итераций меньше, но `RidgeWorkspace` и scaled augmented solves съели выигрыш. На d10 bounded dense-SVD заново строится на каждом AO шаге; несмотря на ноль Krylov iterations, общее время выросло. Максимальная абсолютная разница конечного objective между методами: `2.44e-11` (d10) и `3.88e-8` (d100), значительно ниже критерия `1e-6 max(1,|F_baseline|)`. Сертификация не улучшилась: трудные frozen задачи остались несертифицированными у обоих методов.

H1 опровергнута для этих точек и настроек; H2 поддержана. Этот selection-результат не является универсальной оценкой HYBRID и не доказывает статистического восстановления. Полные fits для HYBRID по плану не нужны. Следующий кандидат — reduced-objective first-order method; он допускается к реализации только после proof note и отдельного аудита предпосылок.
