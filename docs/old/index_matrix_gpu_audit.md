Аудит матричных операций и GPU: single- и multi-index ADP
=======================================================

Дата: 2026-09-21. Исследован commit `31b7ae783adc882b586ba62a82aa89e8dedb8229`.
Численный код не изменён. `prompt.txt` существовал как untracked-файл до аудита.
Manifold исключён. Выводы относятся к фактической реализации, а не к согласованию
её с рукописью. Замеры — CPU; работоспособность и скорость CUDA не проверены:
CuPy отсутствует, `nvidia-smi` сообщает о недоступном драйвере.

**Главный вывод.** Перенос возможен без изменения статистического метода.
Начинать следует с сохранения данных на GPU и пакетных локальных вычислений,
а не с замены LSMR на нормальные уравнения. На CPU есть уже проверяемые точные
оптимизации, но часть из них реализована только в одном из двух путей исполнения.
Инициализация и последовательные итерации решателя могут ограничить общий выигрыш.

**1. Два разных вычислительных пути.**

| Вход | Статистики | Решатель | GPU сейчас |
|---|---|---|---|
| `core/single/ADP_single_index.py`, `core/multi/ADP_multi_index.py` | `engine/common/ADP_Statistic_engine.py` | по умолчанию `solver/legacy_lsmr.py` | CuPy-ветка статистик и legacy LSMR |
| `cli/main.py`, через него `experiments/runner.py` | `engine/common/statistic.py` | `solver/LSMR.py`, `CG.py`, `HYBRID.py` | NumPy/SciPy, нет CLI-флага GPU |
| `core/ADP_Solver.py::ADP_Solver` | `engine/common/statistic.py` | HPAO LSMR по умолчанию | CPU |

В legacy basis имеет форму `(d,m)`, в HPAO — `(m,d)`.
HPAO использует minimum-norm local refit, correction/proximal шаги, gauge fix
и сертификаты; legacy — локальный ridge и alternating global ridge.
Перенос одного решателя вместо другого не является EXACT-оптимизацией.
Настройка `ADP_Config.gpu` не делает CLI/HPAO автоматически GPU-совместимым.

**2. Реализуемые формулы и стоимость.**

Обозначения: `X=(n,d)`, `Phi=(J,P,d)`, `I=(J,P)`, `U=(J,P,d)`,
`B=(m,d)`, `L=(J,m)`, размер блока центров `b`, соседство `k_j`,
`E=sum(k_j)`. Все оценки памяти ниже для float64.

Для центра j: `s_j=sum(w_ji)`, `a_ji=w_ji/s_j`,
`mu_j=sum(a_ji X_i)`, `ybar_j=sum(a_ji Y_i)`.
Оба движка статистик вычисляют:

```text
Q_j = Phi_j (X - mu_j)^T                         (P,n)
r_j = Q_j a_j                                    (P,)
H_j = (Q_j - r_j[:,None]) * a_j[None,:]          (P,n)
I_j = s_j H_j (Y - ybar_j)                       (P,)
U_j = s_j H_j (X - mu_j)                         (P,d)
```

Код дополнительно использует глобальное центрирование X и коррекции сумм H.
В `statistic.py(normalized=True)` множитель `s_j` убран из I/U и передаётся
внешним весом в solver. Нельзя потерять эту нормировку: она меняет объектив.
Плотное построение стоит `O(J P n d)`, работа по поддержке — `O(E P d)`;
хранение готового U остаётся `O(J P d)` в обоих случаях.

Single prediction: `l_j U_j beta`. Legacy local slope:
`l_j = <I_j,U_j beta> / (||U_j beta||² + local_ridge)`.
Глобальная legacy-задача при фиксированных slopes:
`min_beta sum_j ||I_j-l_j U_j beta||² + lambda ||beta-beta_prior||²`;
после решения код нормирует beta. HPAO refit не добавляет этот local ridge.

Multi prediction: `U_j B.T L_j`, эквивалентно `U_j (L_j B).T`.
Глобальный оператор действует на `m*d` неизвестных, размер выхода `J*P`.
Одна пара forward/adjoint при правильном порядке стоит
`O(J P d + J m d)`, не требует явной design-матрицы `(JP,md)`.
Все перекрёстные взаимодействия компонент m должны сохраниться.

Локализация уже использует низкий ранг:

```text
single legacy: q = (rho² ||delta||² + (beta.T delta)²) / h²
single new:    q = (rho² (||delta||²-(beta.T delta)²)
                   + (beta.T delta)²) / h²
multi orthogonal: q = (alpha² (||delta||²-||B delta||²)
                      + sum_a eigenvalue_a (B delta)_a²) / h²
```

Отрицательные ортогональные квадраты из округления обрезаются до нуля.
Веса `kernel(q)` получают **квадратичный аргумент**. Текущий `epanechnikov`
считает `max(1-q²,0)`, а не `max(1-q,0)`; менять это при GPU-переносе нельзя.
Основной pairwise helper уже использует Gram identity, общий сдвиг и прямой
пересчёт подозрительно малых расстояний. Это следует сохранить при переносе.

**3. Кандидаты на оптимизацию.**

| Место | Предложение | Класс и граница |
|---|---|---|
| `ADP_Statistic_engine._dense_block` | `Phi.reshape(b*P,d) @ Xc.T`, затем `H.reshape(b*P,n) @ Xc`; использовать Q как буфер H | EXACT; уже есть в `statistic._dense_moments` |
| `ADP_Statistic_engine._local_values` | CPU: обрабатывать фактическое k_j без padded `(b,k_max,d)`; GPU: группировать соседства близкой длины и batched matmul | EXACT, если ни один ненулевой сосед не теряется; CPU/GPU crossover различается |
| `legacy_lsmr._solve_basis` | forward: `local=L @ B`, затем batched `U_j @ local_j`; adjoint: `pulled=U_j.T @ residual_j`, затем `L.T @ pulled` | EXACT; готовые функции `_multi_operator` уже применяются в HPAO |
| Single forward/adjoint | view `U2=U.reshape(J*P,d)`, `U2 @ beta`, `U2.T @ (l[:,None]*R).ravel()` | EXACT; кандидат GEMV, без обещания ускорения до измерения; следить за contiguous/view |
| `legacy_lsmr._multi_coefficients` | Gram: `projected.swapaxes(1,2) @ projected`, RHS: `projected.swapaxes(1,2) @ I[...,None]` | EXACT замена einsum; малая `(J,m,m)` система, выигрыш не измерен |
| `legacy_lsmr._solve_multi`, `ADP_multi_index_engine.subspace_distance` | после ортонормирования считать `||Q-V(V.T Q)||_F`, вместо `||QQ.T-VV.T||_F/sqrt(2)` | NUMERICAL; `O(dm²)` вместо `O(d²m)`, память `O(dm)` вместо `O(d²)` |
| `weights.calculate_weight`, без distance cache | вынести `sum(X²)` из цикла по блокам | EXACT; для классов с готовым distance2 выигрыша здесь не будет |
| Повторные statistics-вызовы | подготовить глобально центрированные X один раз на fit и использовать на CPU/GPU | EXACT при тех же данных/центрах; новый cache invalidation не нужен между неизменными X |
| Local initialization | пакетный QR/SVD по центрам с ограничением памяти; сохранить intercept, ridge и rank cutoff | NUMERICAL; не заменять на массовое построение `(J,d,d)` |
| `LSMR._stationarity` | измерить GEMM-формы оставшихся contractions; повторно использовать проекции, когда index/U неизменны | EXACT; полезность зависит от профиля |

У расстояния subspace в engine есть дополнительная нормировка `1/sqrt(m)`;
в legacy stopping — её нет. Для неортонормированных входов сначала нужен
rank-revealing QR/SVD, как сейчас. Формула `sqrt(m-||Q.T V||²)` теряет точность
для почти совпадающих подпространств; предпочтителен остаток проекции.

У локальных Gram-систем legacy уже есть риск ухудшения обусловленности.
Пакетный SVD/QR augmented ridge — отдельный NUMERICAL-кандидат, а не повод
ввести большой global `A.T @ A`. CG уже существует как отдельный вариант;
его matrix-free normal action не отменяет возведения числа обусловленности
в квадрат. Для переноса HPAO нужно сохранить проверки residual/status,
trust acceptance и minimum-norm ранг локальных задач.
Legacy сейчас сохраняет stop code/iterations, но проверяет главным образом
конечность/ненулевую норму результата. Перед новым backend нужен явный тест
обработки предела итераций и независимый residual certificate.

**4. Замеры CPU.**

Скрипт: `benchmarks/index_matrix_audit.py`. Все исходные числа и окружение:
`docs/index_matrix_audit_results.json`. NumPy 2.5.2, SciPy 1.18.1,
OpenBLAS 0.3.34, один BLAS-поток, float64, seed 1/2/3; прогрев и пять повторов.
В таблице диапазоны медиан между seed, а не доверительные интервалы.
Память — отдельный `tracemalloc` peak вызова без входных массивов;
это не RSS и не полное потребление native BLAS.

| Проба | Время до → кандидат, ms | Traced peak до → кандидат, MiB |
|---|---|---|
| Dense statistics `(n,d,J,P)=(4000,150,64,20)` | 134.2–135.4 → 90.3–93.5 | 66.6 → 47.0 |
| Те же размеры, 5% ненулевых весов | 15.9–18.7 → 11.7–12.6 | 25.1–25.5 → 8.3 |
| Statistics `(10000,1000,32,20)`, 1% весов | 53.9–66.1 → 40.6–49.2 | 137.6–148.5 → 88.4–88.7 |
| Multi forward `(J,P,d,m)=(1000,20,1000,10)` | 24.0–24.8 → 7.1–7.6 | 1.68 → 7.78 |
| Multi adjoint, те же размеры | 7.47–7.55 → 7.46–7.90 | 7.71 → 7.71 |
| Subspace distance `(d,m)=(1000,10)` | 2.56–2.87 → 0.023–0.026 | 15.26 → 0.15 |

В statistics сопоставлены два уже существующих движка при `normalized=False`.
Dense-результаты совпали, максимальная абсолютная ошибка sparse-моментов
`1.71e-13`. Forward relative error ≤ `7.69e-16`; adjoint совпал.
Проверено adjoint identity. У forward выигрыш оплачен буфером `(J,d)`:
нельзя назвать эту замену одновременно ускорением и экономией памяти.
Adjoint в NumPy уже выбирает удачный contraction path: ускорение не подтверждено.
Это синтетические статистики, не доказательство качества восстановления EDR.

Дополнительно выполнены два диагностических CLI-запуска:
`n=2000,d=100,J=200,P=20,N_lin=220`, multi m=3, два outer-шага,
три solver-шага, local initialization. Во встроенном профиле:

| Стадия | Single, s | Multi, s |
|---|---:|---:|
| Инициализация | 0.790 | 0.781 |
| Статистики | 0.086 | 0.090 |
| Решатель | 0.127 | 1.126 |
| Всего | 1.060 | 2.047 |

Это один запуск на модель с включённым profiling/tracemalloc, не сравнительный
benchmark. Оба остановлены по outer_steps; сходимость и восстановление не
установлены. Первоначальная попытка J=64 была отклонена валидацией
`N_J >= ceil(n/N_loc)`; повторено с J=200. Эта ошибка не исключалась из оценки
качества: оценка качества по этим пробам вообще не проводится.
Главное наблюдение: при таком режиме single ограничен инициализацией,
multi — решателем и инициализацией. Изолированное ускорение statistics не
может дать такой же множитель ускорения всему fit.

Команды воспроизведения:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache \
  uv run --no-sync python -m benchmarks.index_matrix_audit /tmp/index_matrix_audit.json
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache \
  uv run --no-sync python -m ADP.cli --mode single --n 2000 --d 100 --N_J 200 \
  --N_phi 20 --N_lin 220 --outer-steps 2 --solver-max-steps 3
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache \
  uv run --no-sync python -m ADP.cli --mode multi --index-dim 3 --n 2000 --d 100 \
  --N_J 200 --N_phi 20 --N_lin 220 --outer-steps 2 --solver-max-steps 3
```

**5. Память при переносе.**

При `n=10000,d=1000,J=10000,P=20,m=10`:

| Объект | Размер float64 |
|---|---:|
| X `(n,d)` | 80 MB |
| Один U или Phi `(J,P,d)` | 1.6 GB |
| Полный distance2 или W `(J,n)` | 800 MB |
| Local gradients `(J,d)` | 80 MB |
| Один Q/H block `(32,20,n)` | 51.2 MB |
| Padded local X `(32,100,d)` | 25.6 MB |
| Явная multi design `(JP,md)` | 16 GB |
| Запрещённый differences `(J,n,d)` | 800 GB |

Размеры десятичные, не MiB/GiB. U+Phi уже занимают 3.2 GB, без X, distance2,
буферов и CUDA workspace. Общий путь по-прежнему хранит distance2 `(J,n)`.
Compact поддержка не гарантирует малый VRAM: готовый U остаётся плотным.
Для конкретного GPU нужно измерить peak и резервировать память под solver.

**6. Практический GPU-путь без изменения метода.**

Текущий `calculate_statistics_gpu` каждый outer-шаг копирует X/Y/Phi,
принимает CPU-блоки весов, копирует padded neighbors, возвращает mean/n_eff/eta
на CPU. Legacy solver возвращает basis/coefficients на CPU. Генерация направлений,
инициализация, geometry, bandwidth/scale search и sparse KD-tree остаются CPU.
Флаг GPU означает частичную разгрузку, а не полностью GPU-resident fit.

Предпочтительная граница — числовое состояние одного fit:

```text
X/Y на GPU один раз → centered X → initialization
  → tiled geometry/weights → statistics → solver → следующий outer-шаг
CPU: входная валидация, управление, небольшие summaries, конечный результат
```

Для первого переноса достаточно CuPy и существующего LinearOperator.
CuPy документирует LSMR с поддержкой LinearOperator и damp:
https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.lsmr.html
Версию нужно закрепить и проверить в реальном окружении; документация latest
сама по себе не является проверкой установленного пакета.

Приоритеты переноса:

1. Плотные моменты и pairwise geometry — большие GEMM по блокам. Центрировать
   перед Gram; пограничные расстояния пересчитывать, а не только clip.
2. Компактная поддержка — CSR либо exact neighbor lists. Группировка по k_j
   ограничивает padding; слишком длинные строки обрабатывать отдельно.
   CPU-порог 25% ненулевых весов нельзя автоматически считать GPU-порогом.
   KD-tree/Python lists заменить tiled GPU screening при подтверждённом crossover.
   Для Epanechnikov текущий sparse engine не включается автоматически: он
   распознаёт только box/plateau. Можно строить exact support для Epanechnikov,
   сохраняя ненулевые веса, а не менять ядро ради существующей ветки.
3. Multi local refit — batched SVD/QR малых `(P,m)` задач с теми же cutoff и
   degeneracy checks; initialization — отдельные memory-bounded батчи.
4. LSMR forward/adjoint на GPU, скаляры и проверки синхронизировать осознанно.
   Single global action преимущественно GEMV и чтение U, multi — batched GEMV
   плюс GEMM. Большие U могут выиграть от пропускной способности памяти,
   но нельзя ожидать пиковой GEMM-производительности от всего Krylov solver.
5. Поиск h/rho/alpha — reductions на GPU; одна адаптивная проверка вызывает
   зависимость от предыдущего результата. Не копировать массив масс ради bool.
   Перенос box order statistics обязан сохранить строгую границу и coverage;
   средняя масса и rowwise coverage — разные условия.

Итерации outer и Krylov зависимы. Распараллеливать следует центры, направления,
локальные разложения, а при мелких задачах — независимые fit одного размера.
GPU-генератор с тем же числовым seed не обязан дать NumPy-направления; для
reference-проверки нужны сохранённые одинаковые directions, затем отдельный
протокол backend RNG с сохранением распределения и независимых потоков.

GPU timing должен включать синхронизацию и отдельно учитывать upload, warmup,
kernel/JIT и конечный download. CPU perf_counter без synchronize недостаточен;
официальная рекомендация — CUDA events / `cupyx.profiler.benchmark`:
https://docs.cupy.dev/en/stable/user_guide/performance.html
Для VRAM учитывать и живые массивы, и memory pool; tracemalloc не измеряет VRAM.

**7. Более глубокие преобразования и их статус.**

Факторизация статистик может убрать U, но полезна не при любых размерах.
На поддержке центра положим `F=diag(sqrt(a)) (X_neighbors-mu)` формы `(k,d)`.
В точной арифметике `U=s Phi F.T F`. Из rank-revealing QR/SVD F получается
`U=Q R` ранга `r<=min(P,k-1,d)`. При сохранении всех ненулевых компонент
и исходного конечного Phi это EXACT-представление, а корректировки округления
нужно сверить с текущим H. Хранение факторов стоит `J*r*(P+d)` вместо `J*P*d`;
когда `r≈P`, выигрыша нет. SVD/QR имеет собственную стоимость подготовки.

Альтернатива — хранить H и соседей и считать
`U_j v = s_j H_j ((X_neighbors-mu_j)v)`, с точным adjoint.
Это EXACT в вещественной арифметике, но добавляет gathers и проходы по соседям
на каждой Krylov-итерации. Если `k>P`, может стать существенно медленнее
готового U. Выбирать по `k,P,d,iterations`, а не только по VRAM.

| Исследовательский вариант | Класс | Что необходимо проверить |
|---|---|---|
| FP32/mixed precision, уточнение residual в FP64 | APPROXIMATE относительно FP64 reference | FP64-качество, обусловленность, независимый residual, реальный GPU |
| Меньше Krylov iterations/реже сертификаты/ослабленный tolerance | APPROXIMATE | ошибка коррекции и качество при прежней цели |
| Усечение факторов U по singular values | APPROXIMATE, если лишь способ решить прежнюю цель с сертификатом; иначе ESTIMATOR | ошибка относительно исходного U, а не только усечённого |
| Top-k cap соседей, approximate NN с пропусками | ESTIMATOR при смене поддержки | парное восстановление EDR и local mass; opt-in |
| Общие направления для всех центров, изменение P или refresh | ESTIMATOR | меняется конечный sketch и зависимости между центрами |
| Полное локальное пространство вместо finite sketch | ESTIMATOR | это другая целевая функция, даже если быстрее |
| Другой kernel или фиксированное расписание h/rho/alpha | ESTIMATOR | отдельный вариант с полным протоколом |

Для первого GPU-прототипа эти изменения не нужны. Потенциальный следующий
алгоритмический кандидат — блочные/recycling Krylov-методы для повторных
correction-задач, но сохранять сертификат и отдельно мерить память; HYBRID
уже содержит reusable ridge workspace. При малых `md` его bounded dense
SVD-путь может быть уместнее GPU-итераций; при больших `md` материализация
design становится главным ограничением.

**8. Рекомендуемый порядок дальнейшей работы.**

Сначала зафиксировать целевой интерфейс: современные CLI/HPAO и классы legacy
неэквивалентны. Затем перенести проверенные локальные EXACT/NUMERICAL-приёмы
в нужный путь, не заменяя solver и нормировку. Следующий этап — GPU-resident
statistics/operator с фиксированными входами и CPU reference; после него
перенос geometry, initialization и scale search по профилю полного fit.

Приёмка: dense reference и adjoint; offset/collinearity/unequal mass;
zero/degenerate neighborhoods; сохранение rank cutoff; solver status и
сертификат; совпадение поддержки на границе; несколько paired seed;
single sign-invariant quality, multi trace score/principal angles;
синхронизированные wall-clock и VRAM/RSS. Измерять малый, средний и стрессовый
режим: GPU crossover не выводится из одной большой GEMM-пробы.

В этом аудите прошли 25 focused tests из `test_multi_operator.py` и
`test_statistics_optimization.py` (manifold исключён), встроенные проверки
benchmark и Ruff для добавленного скрипта. Полный suite и GPU-тесты не запускались.
