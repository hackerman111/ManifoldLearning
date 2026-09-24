# S4 — первые 24 frozen trials: reduced L-BFGS vs HPAO-LSMR

Предварительный [H4](hypotheses.md) и dry-run зафиксировали 12 frozen задач × два метода, LSMR cap80, reduced L-BFGS cap320, timeout60 s/trial, CPU/1 BLAS thread, чередование порядка. Команда:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.multi_solver_frozen \
  --frozen-dir benchmark_outputs/diagnostic/multi_solver_search_s1_20260924 \
  --output benchmark_outputs/diagnostic/multi_solver_search_s4_20260924
```

Все 24/24 trials завершились с независимым objective/stationarity certificate, без timeout и numerical failure. [Raw results](../../../benchmark_outputs/diagnostic/multi_solver_search_s4_20260924/results.json) включают per-step objective/gradient-vs-wall curves, failure accounting, source/frozen hashes и peak RSS из отдельных процессов.

| Точка | Frozen пар | Сертификаты LSMR / reduced | Медиана парного reduced/LSMR времени | Медиана парного peak RSS | Макс. ухудшение final objective |
|---|---:|---:|---:|---:|---:|
| d10 | 6 | 6/6 / 6/6 | 4.244× | 0.997× | 2.4e-10 |
| n1000,d100 | 6 | 3/6 / 6/6 | 0.883× | 0.999× | +1.2143 |

На d100 три пары завершились с заметно худшим reduced objective: seed1001 outer0 `+0.13265`, seed1001 outer2 `+1.21427`, seed1002 outer0 `+0.30251` к HPAO cap80. Допуск плана здесь порядка `3e-4`; даже сертифицированная и быстрее найденная стационарная точка не заменяет сравнение objective. На seed1001 outer0/2 reduced сделал 958/664 отклонённых Armijo trials и работал 30.97/28.38 s, дольше HPAO. На остальных d100 парах reduced работал за 0.536–0.967 времени HPAO; это не даёт медиану ≤0.70 на всех шести. Все значения RSS близки; прирост памяти не объясняет неудачу.

H4 опровергнута по скорости на d10/d100 и по objective на d100, несмотря на улучшение доли формально сертифицированных frozen задач. Полные fits для чистого reduced solver запрещены до нового selection evidence. Три baseline d100 вызова без сертификата требуют длинного cap160→320 reference в рамках исходного бюджета (не глобальный минимум). Кривые проблемных задач показывают большое снижение objective уже после первого HPAO AO-шагa: seed1001 outer0 `347.41→304.80`, outer2 `304.18→285.59`, seed1002 outer0 `256.28→230.84`. Это мотивирует один заранее заданный staged кандидат: ровно один HPAO шаг, затем guarded reduced L-BFGS, без sweep числа AO шагов и без скрытого fallback. Его proof gate и bounded six-task d100 проверка записываются до реализации; на d10 возможна явная стратегия использовать прежний HPAO.
