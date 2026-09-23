# CPU bottleneck полного `fit`: multi-index и manifold ADP

Измерено 23 сентября 2026 года на текущем dirty checkout, HEAD
`c58c5cf4eb269ea3c7476e775342c771c9419158`. Это профиль двух разных
estimator-ов на их собственных синтетических задачах; абсолютные времена
между моделями не являются сравнением качества или скорости одного метода.
Исходные данные каждой серии, полный Git status, SHA-256 всех `ADP/**/*.py`
и benchmark-скрипта, версии Python/NumPy/SciPy, BLAS и параметры находятся
в [JSON-артефактах](experiments/bottlenecks_2026-09-23/). Внутри каждой
финальной серии SHA-256 до и после прогонов совпали.
Машина: Intel Core i5-12400F (6 ядер, 12 потоков), Linux
7.1.11-arch1-1, Python 3.14.7, NumPy 2.5.2, SciPy 1.18.1.
Конфигурация NumPy указывает `cblas/lapack 3.12.0`; `threadpoolctl` в этой
среде видит `libgomp` с одним потоком при указанных ниже переменных.

## Протокол и границы измерения

- `X` равномерно на `[-1,1]^d`; `SeedSequence(20260923)` независимо задает
  данные и шум `0.02`; seed модели — 17. Multi: `Y=sin(2X₀)+0.7X₁²+noise`,
  `m=2`. Manifold: `Y=X₀+0.3X₁²+noise`, `m=1`. Везде `float64`, Epanechnikov
  `K(q)=max(1-q²,0)` при `q=distance²/h²`, batch size 32, CPU,
  `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.
- Multi: `n=600,d=8,P=16,N_loc=30,N_lin=40`, `J=48` или `96`, два outer
  шага, local initialization, normalized estimator `new`, orthogonal
  tensor, `h_min=0.3`, LSMR/HPAO `max_steps=50,tol=1e-6`.
  Manifold: `n=600,d=6,P=12,N_loc=30,N_lin=80,N_manifold=6`, `J=40` или
  `80`, два sync и один scale, `h_min=0.6,a=2`, CG `rtol=1e-8`,
  `scale_boundary=stop`.
- В каждой серии один отдельный процесс-прогрев и десять отдельных
  процессов-измерений. `fit_seconds` охватывает публичный `model.fit`;
  импорт и генерация данных завершены до таймера. Отдельный процесс нужен,
  чтобы `resource.ru_maxrss` относился к одному fit. Время фазы меряется
  `perf_counter`; sampling `/proc/self/statm` раз в 2 ms дает приблизительный
  момент пика RSS. `tracemalloc` и `cProfile` запускались отдельно.
- В таблицах указаны медиана десяти повторов и разброс 10–90% для полного
  fit. Доли — медиана доли фазы в каждом запуске. Вложенные hooks не
  складываются с родительскими: например, `_mean_mass` включен в поиски
  bandwidth/anisotropy, а `_solve_B` — в `_one_step`.

| Модель / `J` | Полный fit, ms; 10–90% | Peak RSS, MiB | RSS перед fit, MiB | Solver evidence |
| --- | ---: | ---: | ---: | --- |
| Multi / 48 | 70.07; 68.78–71.03 | 78.35 | 71.06 | 2 outer; LSMR iterations 256+568; оба `converged=True` |
| Multi / 96 | 97.83; 96.30–99.11 | 79.38 | 71.17 | 2 outer; LSMR iterations 218+578; оба `converged=True` |
| Manifold / 40 | 180.42; 177.91–182.21 | 75.49 | 71.01 | 3 projector updates; max CG relative residual `6.53e-9` |
| Manifold / 80 | 343.62; 342.07–347.12 | 76.14 | 71.14 | 3 projector updates; max CG relative residual `9.86e-9` |

Пик `ru_maxrss` — абсолютный пик процесса, включая Python, NumPy/SciPy,
импорты и аллокатор; вычитать RSS перед fit можно только как наблюдаемое
увеличение, а не как точную стоимость массивов. `statm` и `ru_maxrss`
используют разные счетчики; их значения могут расходиться на доли MiB.
Максимум sampled RSS multi наблюдался во время статистик, manifold — во
время статистик либо projector update. RSS может удерживать память после
завершения аллокации, поэтому это указание на интервал, а не доказательство,
какая операция выделила байты.

## Подтвержденные ограничения

Обозначения: `n` — число наблюдений, `d` — признаков, `J` — центров, `P` —
random directions, `m` — размер индекса, `E` — число ненулевых ребер CSR
manifold-графа, `B=32` — размер блока центров. `L=2` — outer steps multi,
`S=3` — projector updates manifold. `Q` — суммарные LSMR итерации,
`q̄` — среднее число CG итераций на target, `T` — число проверок массы в
поиске bandwidth/anisotropy. Оценки описывают выполняемые операции, а не
подогнаны по двум точкам времени.

| Модель и участок кода | Операция и оценка времени / рабочей памяти | J base → J2, ms и доля полного fit | Когда доминирует; уверенность |
| --- | --- | ---: | --- |
| Multi [`ADP/solver/LSMR.py:515`](../ADP/solver/LSMR.py#L515), [`ADP/solver/_multi_operator.py:8`](../ADP/solver/_multi_operator.py#L8), [`:21`](../ADP/solver/_multi_operator.py#L21) | HPAO вызывает matrix-free LSMR; каждый forward/adjoint действует на `U=(J,P,d)`. Время порядка `O(Q·J·P·d + Q·J·m·d)` плюс local refits/stationarity; память `O(JPd+JP+Jm+md)` без плотной матрицы оператора. | **51.29 / 73.1% → 63.24 / 64.8%** | В этом сходящемся режиме главный wall-clock bottleneck. `Q=824→796`, поэтому время выросло главным образом от большего J при почти том же числе итераций. Уверенность высокая для этих форм; рост при других `P,d` не измерялся. |
| Multi [`ADP/engine/common/index_fit.py:343`](../ADP/engine/common/index_fit.py#L343), [`ADP/engine/common/statistic.py:29`](../ADP/engine/common/statistic.py#L29), [`:133`](../ADP/engine/common/statistic.py#L133), [`ADP/engine/common/weights.py:88`](../ADP/engine/common/weights.py#L88) | Потоковые kernel weights и `I/U`; dense upper bound по времени `O(L·J·P·n·d + L·J·n·m)`, при sparse-support статистике меньше. Постоянная память `O(JPd)` плюс блок `O(BPn+Bn)`; вместе с сохраненным `distance2=(J,n)` — `O(Jn+JPd+BPn)`. | **11.26 / 16.1% → 19.86 / 20.3%** | Подтвержденный второй расход времени; ~1.76× при удвоении J. `calculate_multi_weight` — generator, его тело исполняется при потреблении в `calculate_statistics`, поэтому отдельный hook вызова generator почти нулевой. Уверенность высокая для фазового времени, средняя для разделения weights/moments. |
| Multi [`ADP/engine/common/index_fit.py:260`](../ADP/engine/common/index_fit.py#L260), [`ADP/engine/common/calculus.py:16`](../ADP/engine/common/calculus.py#L16), [`ADP/engine/common/initialize.py:70`](../ADP/engine/common/initialize.py#L70) | Полные расстояния `O(Jnd)`, массив `O(Jn)`; затем локальные weighted ridge fits примерно `O(J·k_lin·d²)` и SVD градиентов. | **4.47 / 6.4% → 8.83 / 9.1%** | Не главный bottleneck здесь, но ~1.97× при удвоении J. Локальный support `k_lin` зависит от выбранного bandwidth; рост по d здесь не изолирован. Уверенность высокая для времени и форм, экстраполяция по d не подтверждена. |
| Manifold [`ADP/engine/manifol_engine/optimisation.py:38`](../ADP/engine/manifol_engine/optimisation.py#L38), [`:153`](../ADP/engine/manifol_engine/optimisation.py#L153), [`:217`](../ADP/engine/manifol_engine/optimisation.py#L217) | Каждый из `S·J` target обновлений собирает соседей `K_l`, решает slopes, строит matrix-free B-system, выполняет CG и восстанавливает projector. Время порядка `O(S·E·P·m·d·(1+q̄))` плюс малые SVD; память `O(JPd+E+K_max·P·d+Jmd)`, без `(md)²`. | **101.54 / 56.4% → 199.44 / 58.0%** | Главный wall-clock bottleneck при 3 обновлениях. `E=480→932`, target-solves `120→240`, суммарные CG итерации `719→1431`: около 2× по работе и времени. Уверенность высокая для этого ограниченного числа соседей. |
| Manifold [`ADP/engine/manifol_engine/weights.py:52`](../ADP/engine/manifol_engine/weights.py#L52), [`:72`](../ADP/engine/manifol_engine/weights.py#L72), [`:109`](../ADP/engine/manifol_engine/weights.py#L109) | Повторные проверки mass пересчитывают блочные расстояния/веса. Для центров против `N∈{n,J}` точек одна проверка `O(J·N·d+J·N·m·d)` в анизотропном случае, рабочая память `O(BN+BmN)`. Всего 217 вызовов `_mean_mass` в каждом fit; bandwidth и anisotropy — неперекрывающиеся внешние фазы. | **bandwidth 46.29 / 25.6% + anisotropy 18.28 / 10.1% → 85.48 / 24.9% + 33.30 / 9.8%** | Второй крупный источник времени; ~1.85× и ~1.82× при удвоении J и одинаковом числе проверок. Вложенное `_mean_mass` (~64.97→119.63 ms) уже включено в эти и boundary фазы. Уверенность высокая для этих настроек. |
| Manifold [`ADP/engine/manifol_engine/weights.py:183`](../ADP/engine/manifol_engine/weights.py#L183), [`ADP/engine/common/statistic.py:133`](../ADP/engine/common/statistic.py#L133) | Дважды формируются normalized `I=(J,P)`, `U=(J,P,d)` через плотный GEMM путь `O(JPnd)`; блок `Q=(B,P,n)` занимает `O(BPn)`. Фактический function support ~2.5k из `J·n=24k` при J=40, но этот manifold-путь не сокращает `X` по support. | **6.78 / 3.8% → 12.86 / 3.7%** | Сейчас не bottleneck времени, но кандидат при существенно большем `n` или `P`; это гипотеза, не результат такого масштабирования. Для формы массивов уверенность высокая. |

Внутри manifold `_one_step` при J=40 из 101.54 ms: `_local_slopes`
16.96 ms, `_build_B_system` 25.69 ms, `_solve_B` 25.76 ms,
`_recover_projector` 17.86 ms, `_objective` 11.01 ms, остальное ~4.26 ms.
Таким образом, один CG solve занимает около 25% фазы, а построение системы
вместе с solve — около 51%; гипотеза, что только повторные B-решения
поглощают всю фазу, для этих форм не подтвердилась. Полные локальные
градиенты в [`weights.py:136`](../ADP/engine/manifol_engine/weights.py#L136)
стоят 2.75→5.34 ms (около 1.5% fit); они решают `J` задач `lstsq` на
`(n,d)` design с оценкой `O(Jnd²)` и рабочей памятью `O(nd+Bn)`.
Они могут стать важны при другом `d`; в этом протоколе менялся только J.

Память согласуется с формами массивов, но не разложена на точные аллокации.
При multi base `distance2=(48,600)` — 0.22 MiB, `U=(48,16,8)` — 0.047 MiB,
крупный временный `Q=(32,16,600)` — 2.34 MiB; при удвоении J размер Q
не меняется, а `distance2` и U удваиваются. При manifold base
`Q=(32,12,600)` — 1.76 MiB, `U=(40,12,6)` — 0.022 MiB, CSR хранит только
480 ребер. Режим `traced` показал Python peak 5.98 MiB для multi и
2.62 MiB для manifold, но `tracemalloc` не покрывает всю нативную память
и сам меняет тайминги.

## Проверка влияния измерения и результата

| Модель base | Без hooks, ms | Time hooks, ms | `tracemalloc`, ms | Неучтенное время в time профиле |
| --- | ---: | ---: | ---: | ---: |
| Multi | 68.85 | 70.07 (+1.8%) | 298.24 (4.26× к time) | 0.35 ms / 0.49% |
| Manifold | 171.86 | 180.42 (+5.0%) | 548.60 (3.04× к time) | 1.41 ms / 0.79% |

Все десять base-повторов `off/time/traced` и отдельный `cProfile` run
вернули один и тот же projector для каждой модели (максимальная Frobenius
разность проекторов 0), тот же stop reason и те же solver iterations.
Multi завершил `outer_steps` с двумя сходящимися HPAO-шагами;
manifold завершил `h_min`, имел 480 ребер на каждом из трех обновлений и
максимальный относительный CG residual `6.53e-9`. Разброс фаз при time
профилировании мал относительно разницы между крупными фазами. `cProfile`
использован только для поиска функций: его одинарные прогоны заметно
медленнее, поэтому его собственные времена не вошли в таблицы.

Предварительные `multi_base_{time,off,traced,cprofile}.json` и
`multi_J2_time.json` без суффикса `_converged` сохранены: там default HPAO
`max_steps=5` дал `converged=False` на обоих шагах. Их время не используется
для вывода о сходящемся fit. Увеличение лимита до 50 — изменение настроек
эксперимента, а не правка алгоритма; в пределах каждой финальной серии
настройки и estimator одинаковы.

## Гипотезы за пределами измеренного режима

- При росте `d` полные локальные `lstsq` manifold могут стать крупной фазой;
  здесь `d` между paired base/J2 не менялся.
- При росте `n` или `P` плотный manifold `Q=(B,P,n)` и повторные проверки
  mass могут ограничить время и память; проверено только удвоение `J`.
- Где именно выделен каждый пик RSS, sampling не устанавливает: аллокатор
  может удерживать память между фазами. Для такого вывода нужен отдельный
  allocator-level профиль.

## Повторение

Из корня репозитория, с установленными зависимостями текущего проекта:

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.fit_bottlenecks --case multi_base --profile time --runs 10 --output /tmp/multi_base_time.json
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.fit_bottlenecks --case multi_J2 --profile time --runs 10 --output /tmp/multi_J2_time.json
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.fit_bottlenecks --case manifold_base --profile time --runs 10 --output /tmp/manifold_base_time.json
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.fit_bottlenecks --case manifold_J2 --profile time --runs 10 --output /tmp/manifold_J2_time.json
```

Для проверки overhead повторите нужный `base` case с `--profile off` и
`--profile traced`; `--profile cprofile --runs 1` дает приблизительный
список вызываемых функций. Имена output должны быть новыми: benchmark
не перезаписывает существующие данные. Контрольные малые формы:
`multi_control=(300,6,24,8,2)` и
`manifold_control=(300,4,20,8,1)` в порядке `(n,d,J,P,m)`.
Проверять выводы при другом `n,d,P,m`, kernel, solver tolerance, числе
потоков или GPU нужно отдельным опытом.
