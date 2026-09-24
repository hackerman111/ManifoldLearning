# S4 — последний frozen кандидат H5 и решение

H5 был [зафиксирован заранее](hypotheses.md) после провала чистого reduced L-BFGS: один принятый HPAO AO-шаг, затем guarded reduced L-BFGS, без подбора длины warm start. [Доказательство композиции](proofs/hpao_warm_reduced.md) и [независимый audit](proofs/audit_hpao_warm_reduced.md) закрыты до реализации. Изолированный прототип прошёл 10 focused tests и Ruff; dry-run ограничил эксперимент шестью frozen задачами `n=1000,d=100,m=2`, cap320, 60 s/trial, суммарный предел 360 s. Те же файлы U/I/mass/B проверены по SHA-256. Validation seed 3000–3019 не использовались.

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.multi_solver_frozen \
  --frozen-dir benchmark_outputs/diagnostic/multi_solver_search_s1_20260924 \
  --output benchmark_outputs/diagnostic/multi_solver_search_s4_warm_20260924 \
  --methods warm-reduced --point n1000
```

[Raw H5 results](../../../benchmark_outputs/diagnostic/multi_solver_search_s4_warm_20260924/results.json) содержат версии/хеши исходников, frozen hashes, однониточные настройки, время и peak RSS отдельных процессов, полный trace объективной функции и градиента, независимый конечный certificate, принятую warm-поправку и причины остановки. Все 6 trials завершились без timeout/exception/certificate mismatch. Независимый evaluator совпал с solver diagnostic на каждом результате. `success` в raw означает, что процесс закончил работу; признак математической сходимости — только `diagnostics.converged`.

Проверка raw также подтвердила 6/6 frozen hashes и точное совпадение objective до/после warm шага с первыми двумя значениями прежней cap80 HPAO-траектории; trace содержит все принятые reduced шаги.

После прогона окружение проверено отдельно: Python 3.14.7, NumPy 2.5.2, SciPy 1.18.1, NumPy сообщает CBLAS/LAPACK 3.12.0; переменные `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS` в raw равны 1. Checkout был dirty, поэтому для воспроизводимости важны записанные source hashes: все шесть проверенных файлов по-прежнему точно совпадают с raw. Длинный reference сохранён только с trial rows без собственного environment/source manifest; его objective используется как дополнительный локальный контроль, а время не входит в парный speed ratio.

| Seed / outer | HPAO cap80 objective | Длинный HPAO reference | H5 objective | H5 certificate | H5/HPAO wall |
|---|---:|---:|---:|---|---:|
| 1000 / 0 | 323.397389 | — | 323.397389 | да | 0.875 |
| 1000 / 2 | 313.908423 | 313.908420, cap160, да | 313.908420 | да | 0.595 |
| 1001 / 0 | 291.273673 | 291.273672, cap160, да | 291.273672 | да | 1.479 |
| 1001 / 2 | 272.611297 | 272.532961, cap320, **нет** | **273.825571** | **нет** | **2.906** |
| 1002 / 0 | 226.330806 | — | 226.330806 | да | 1.066 |
| 1002 / 2 | 203.476611 | — | 203.476611 | да | 0.860 |

Длинный контрольный прогон был разрешён до H5: три исходно несертифицированные d100 задачи заново решались из того же frozen B с cap160; только seed1001/outer2 повторён с cap320, потому что cap160 не дал certificate. [Raw reference](../../../benchmark_outputs/diagnostic/multi_solver_search_s4_long_20260924/results.json): две задачи сертифицированы при cap160 за 11.07/11.99 s. Seed1001/outer2 осталась несертифицированной при cap320 за 46.29 s, с objective 272.532961 и riemannian gradient `2.67e-4`; это локальный, не глобальный reference. Все четыре прогона завершились без ошибки.

H5 сертифицировал 5/6 задач, тогда как предсказание требовало 6/6. На seed1001/outer2 он достиг cap320, сделал 817 Armijo rejects, сохранил `aligned_step=1.17e-5` и не получил два последовательных certificate; его objective хуже cap80 на **1.214274** и длинного reference на **1.292610**, при допускаемом отклонении около `2.7e-4`. Первый HPAO шаг на этой задаче уменьшил objective `304.178011→285.592739`, но reduced фаза пришла к тому же худшему basin, что и чистый reduced solver (разница конечных objective `1.6e-8`). Поэтому basin-гипотеза H5 опровергнута именно на различающей задаче.

Медианное отношение завершённого wall time H5/HPAO cap80 по всем шести парам — **0.971**, среди пяти сертифицированных — **0.875**, оба выше цели `≤0.70`; на шестой задаче время до certificate цензурировано, а не считается успешными 33.58 s. Суммарно H5 потратил 65.15 s на шесть вызовов против 43.04 s HPAO cap80. Медианное отношение peak RSS — **1.0003**, память укладывается в цель, но это не компенсирует objective/certificate/time failure. H5 отвергнут до full-fit selection.

Итог ограниченного поиска: существующий HYBRID не ускорил трудную точку, чистый reduced дал 6/6 frozen сертификатов ценой худшего objective и недостаточной скорости, staged H5 не исправил худший basin и сертифицировал лишь 5/6. Ни один метод не проходит заранее заданный frozen gate. По stop condition PLAN.md S5/S6 не запускались; held-out seed не расходовали. Публичный solver/default и статистический estimator не менялись. Этот отрицательный результат относится к проверенным точкам, настройкам и бюджетам, а не доказывает невозможность другого метода.
