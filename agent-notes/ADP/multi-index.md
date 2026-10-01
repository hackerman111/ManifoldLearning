# Multi-index: алгоритм и API

## Представление индекса

Публичный estimator принимает `index_dim=m`. В shared fit-loop basis хранится строками `(m,d)` для solver HPAO; публичный `basis_`/`beta_` и result хранят столбцы `(d,m)`. При возврате из solver индекс приводится к ортонормированному basis, строки канонизируются знаками, а спектр локальных коэффициентов масштабируется относительно старшего eigenvalue. Сравнивать нужно подпространства (projectors/principal angles), а не поэлементно basis. Источники: **SRC-MI-FIT**, **SRC-MI-CANONICAL**, **SRC-MI-ENGINE**, **SRC-MI-RESULT**.

Текущий `transform(X)` — обычная проекция через оцененный общий базис `(d,m)`. Это global multi-index, не локальные проекторы manifold-модели.

## Веса и anisotropy

Пусть `z_q` — q-я координата смещения вдоль ортонормированного EDR basis, `lambda_q` — сохраненный спектр, а `r²` — квадрат ортогонального остатка. Текущая функция `calculate_multi_weight` использует

    principal2 = sum_q lambda_q * z_q**2
    residual2 = ||x-center||**2 - sum_q z_q**2
    argument = (alpha**2 * residual2 + principal2) / h**2

Когда `tensor="orthogonal"`, именно ортогональный остаток масштабируется `alpha`; когда `tensor="full"`, `alpha²` умножает полное расстояние. Поэтому варианты меняют estimator, не считать их alias-ами. Источники: **SRC-MI-WEIGHTS**, **SRC-MI-ALPHA**.

Есть деталь outer-loop: для multi-index первая итерация принудительно задает `effective_tensor="orthogonal"`; после нее используется `config.multi_tensor`. Это стоит проверить, если сравнивается первый trace row с последующими.

`alpha` выбирается в `[0,1]` как наибольший масштаб, при котором средняя локальная mass остается не ниже `N_loc`. Для Epanechnikov есть exact compaction: после каждого принятого lower endpoint удаляются элементы, которые уже не попадут в compact support при оставшемся интервале alpha. Источник: **SRC-MI-ALPHA**.

Случайные направления раскладываются на часть ортогональную EDR-базису и principal-часть. Principal noise умножается на `sqrt(eigenvalues)`, orthogonal noise — на `alpha`, итог нормируется.

## Инициализация и outer loop

`local`/`local-cv` получают local ridge-gradient в каждом центре и делают SVD матрицы градиентов; при `estimator="new"` начальная gradient-PCA взвешивается `sqrt(local_mass)`. `pilot` строит неглубокую MLP и ортонормирует матрицу первого слоя; `random` делает QR Gaussian matrix. Недостаточный rank отклоняется как неидентифицируемый multi-index.

Каждый шаг цикла:

1. генерирует или переиспользует `(J,P,d)` directions;
2. вычисляет principal/residual weights и local `I/U`;
3. передает в HPAO basis-строки `(m,d)`; для normalized statistics mass — внешний множитель objective;
4. canonicalizes basis и переоценивает normalized eigenvalues;
5. после уменьшения h подбирает следующий alpha; если локальную массу обеспечить нельзя, останавливается.

Для outer-step selection используется SSE на Y в выбранных центрах, даже когда solver diagnostic loss другой. `trace_indices=True` включает basis, eigenvalues, alpha, mass summaries и solver diagnostics.

## Профильный Grassmann solver (opt-in)

`ADP/solver/grassman.py::solve` возвращает `HPAOResult`, принимает CPU
`index_init=(m,d), U=(J,p,d), I=(J,p), mass=(J,)`. При `local_ridge=0`
минимизирует тот же unpenalized minimum-norm finite-sketch профиль,
что local-refit HPAO; не fixed-g rank-r цель SVD. `local_ridge>0` — явный
ESTIMATOR вариант. `lambda_prox` здесь только damping малого GN solve,
не постоянный statistical/chordal penalty. Production default не изменён.

`method="rank_one"` использует rank-one геодезику и scalar Schur curve
(улучшение I; `angle_backend="refit"` — независимая абляция стоимости).
`method="spectral"` выбирает q по энергии горизонтального градиента и
проверяет cached geodesic через Armijo. Default `"core_gn"` (улучшение II)
решает augmented least squares во всех q*m координатах Z=V K, сохраняя
полный `T.T*r` член и связи коэффициентов; polar retraction/Armijo.
Операторные действия U[Y,V] кэшируются только внутри текущего шага.
Нет dense d*d проектора/Hessian. При превышении `workspace_bytes`
полный core заменяется spectral; при rank/conditioning guard — rank-one
по значениям, без применения неверной full-rank производной.

`converged` требует normalized horizontal gradient <=tol; это inner frozen
stationarity, не global optimum. На rank boundary сертификат неприменим;
`rank_boundary_stationary`, `max_steps`, `line_search_failed` сохраняют
незавершённость явно. Итоговый residual/gradient пересчитывается через live U.
Диагностики: profile/loss history, step_ranks, GN/evaluation/fallback counts,
local rank loss, orthogonality, stationarity_applicable.

Пример подключения без конфликта `method` конструктора адаптера:

```python
from functools import partial
from ADP import ADP_solver
from ADP.solver.grassman import solve
solver = ADP_solver(partial(solve, method="core_gn", max_steps=5))
```

Для CPU-оптимизации того же optimizer можно импортировать `solve` из
`ADP.solver.grassman_optim`: public signature/defaults сохранены. Safe малые
local QR, bounded reassociated gradient, transpose GEMM и frozen-step caches
описаны в [solvers.md](solvers.md#профильный-grassmann-opt-in).
Это EXACT/NUMERICAL относительно исходного grassman, без изменения цели,
допусков или iteration budget. Исходный модуль остаётся проверяемым эталоном.

Проверки: `tests/test_grassman.py` (independent lstsq, Schur, FD full Jacobian,
dense ridge, gauge/rank/stress/noiseless recovery). Paired heavy Spokoiny
и frozen ablations: `benchmarks/grassman_benchmark.py`, результаты/ограничения
в `experiments/grassman_2026_10_01/REPORT.md`. Три full-fit seeds:
core5 paired time ratio .874, ниже projector error на всех3, ~13MiB больше
RSS; core20 ratio1.260. Все inner calls capped/unconverged. Schur/refit
rank-one frozen paired gain5.72x; GN .0949s vsspectral .2157s при близком loss.
Adaptive-rank advantage не наблюдалось; QR compression при p10<d+1 бесполезен.
Это bounded outer-budget evidence, не heldout/global/convergence guarantee.

## Явный SVD-решатель: ранг матрицы и ранг поправки

Новая общая теория: [EDR_unified_theory.tex](../../SVD/EDR_unified_theory.tex),
метки `sec:framework`, `sec:svd`, `sec:grassmann`, `sec:angles`, `sec:field`,
`sec:metrics`. Она объединяет fixed-g matrix/correction, профилированные
Grassmann-повороты и tangent Tucker как разные частные случаи; не заменяет
их цели друг другом. Карта текущих семи разрешённых заметок — `sec:map`;
литература — `sec:literature`. Проверки/сборка/узкий angle benchmark:
[REPORT.md](../../SVD/theory_2026_10_01/REPORT.md) (203 формульных проверки,
3 frozen-curve seeds, total time ratio .160–.167; не full-fit recovery).
Теоретическая работа не меняла production solver. Профильный angle-search
и полный малый GN core теперь реализованы opt-in в `grassman.py` ниже;
пространственный tangent Tucker остаётся предложением.
Отдельно различаются localization/residual/penalty/search metrics;
GLS covariance требует squared kernel weights, не только Sigma/N.

Ранее указанные `SVD/SVD.tex`, `SVD/SVD_form.tex`, `chat_*.md` и
`problem.md` сейчас отсутствуют в корне `SVD/`; пользовательские перемещения
не отменялись, исключённые подкаталоги не читались. Поведение correction
сверять по живому коду ниже. Подробное прежнее изложение:
[SVD_solver.tex](../../SVD/SVD_solver.tex), разделы
`sec:variants`, `sec:limits`, `sec:second`, `sec:gpu`, `sec:algorithm`.
Приложение `sec:sources` учитывает историческую структуру исходной папки;
текущая карта содержится в новом документе. Второй решатель, совместные блоки, преобразованные и
случайные кандидаты остаются предложениями. Отбор обычных SVD-пар Q по
точному выигрышу реализован экспериментальной опцией ниже. Проверка формул:
`rtk proxy python SVD/documentation/check_math.py`; запись проверки и сборки —
`SVD/documentation/VERIFICATION.md`.

`ADP/solver/SVD.py` реализует два явных `rank=r<m` варианта для фиксированных
локальных коэффициентов `g_j`. Функционал по умолчанию
`sum_j mass_j ||I_j-U_j B.T g_j||² + lambda||B-P||²`; это не текущий
HPAO correction penalty. По умолчанию `low_rank_target="matrix"` сохраняет
старое ограничение `rank(B)<=r` и возвращает низкоранговую `(m,d)` матрицу
`B`. Режим `low_rank_target="correction"` ограничивает `rank(B-P)<=r`,
начинает с `Delta=0`, один раз считает `I-forward(U,P,g)` и возвращает
сырой полный `P+Delta`; `rank=0` возвращает `P`. Жадные
rank-1 компоненты находятся попеременными шагами: малая `(m,m)` задача для
`a`, для `v` — прямой Cholesky при `lambda>0, d<=128` либо плоский LSMR с
явной проверкой normal residual и fallback, compact QR/SVD с кэшем `U@V`
и совместное решение `(k,k)` задачи масштабов. Correction переиспользует эти
численные primitives с нулевой ridge-целью и штрафом `lambda||Delta||²`.
Временный буфер прямого
решения ограничен 16 MiB; `direct_max_dimension=None` оставляет только LSMR.
Из `SVD_solver.tex` (`eq:Ha`, `eq:diagprecond`) добавлена явная опция
`precondition_v=True` (default `False`): диагональ нормального оператора
v-подзадачи масштабирует справа **оба** блока расширенной системы, включая
ridge. Центр регуляризации и ненулевой warm start сохранены; при `lambda=0`
масштабирование отключено, чтобы сохранить решение минимальной нормы.
Суммы квадратов столбцов `U` кэшируются до 16 MiB; выше лимита диагональ
считается потоково без `U**2` или `d*d` матрицы. Исходная нормальная невязка
остаётся критерием приёмки, при необходимости применяется строгий
немасштабированный LSMR. Оценка невязки самого LSMR может относиться к
масштабированным координатам и не заменяет исходный сертификат. В diagnostics
добавлены `precondition_v` и `preconditioner_cache_bytes`.
В расширенном операторе устранены вложенные `LinearOperator` вызовы;
эта точная оптимизация действует и при выключенном предобусловливании.
Неограниченные/нечисловые нормы и нечисловой сертификат явно отклоняются.
Адаптивный допуск LSMR доступен через `adaptive_krylov=True`, но по умолчанию
выключен после потери качества на одном тяжёлом seed. Глобальный
rank-r optimum не гарантируется.

Экспериментальный `rank_one_search="gradient"` (default `"alternating"`)
берёт все ненулевые сингулярные пары непроецированной
`Q=sum_j mass_j g_j e_j.T U_j + lambda(prior-low_rank)=-grad(F)/2`.
Для единичных a,v считает `R=a.T Q v`,
`D=sum_j mass_j (g_j.T a)² ||U_j v||²+lambda`, выбирает максимум
`gamma=R²/D` и шаг `sigma=R/D` (SVD_solver.tex, eq:Q/eq:gain и раздел
«Выбор направления по уменьшению, а не только по градиенту»).
Попеременные a/v solves пропускаются; сжатие QR/SVD, совместный refit
масштабов, проверка исходной цели и итоговый basis/refit общие.
Внутренняя метрика применяется через тот же whitening. Это APPROXIMATE,
цель не меняется. Поиск по конечному набору пар не даёт глобального
rank-one/rank-r optimum. Полезная добавка без роста ранга отклоняется с
`no_rank_growth`; это эвристическая остановка, не сертификат стационарности.
У `gradient` tuples `inner_iterations`, `inner_converged`, `lsmr_*` пусты,
число linear solves равно 0, `v_normal_residual_applicable=False`;
нулевой `v_normal_residual_max` не является сертификатом.
`inner_tol`, `inner_maxiter`, direct/LSMR/warm-start/preconditioning настройки
направлений относятся только к `alternating`; `rank_tol` активен в обоих.
Дополнительный scratch для кандидатов O(md+Jp), до m действий Uv на добавку,
плотной Hessian/d*d нет. Reference/edge/public-fit проверки:
`tests/test_svd_gradient.py`; метрика: `tests/test_svd_metric.py`.
Парный фиксированный-g пилот d100/300, 3 seeds, оба rank targets:
`experiments/svd_gradient_2026-09-30/{pilot.py,REPORT.md,runs.csv,metadata.json}`.

Новая экспериментальная опция `metric_power=0/.5/1` и `metric_floor=rho`
заменяет proximal penalty на `lambda tr((B-P) A (B-P).T)`;
`A=rho I+(1-rho)(K/||K||op)^p`. Default p=0 сохраняет Frobenius.
Full: `K=alpha² I+P.T Lambda P`; orthogonal:
`K=alpha²(I-P.T P)+P.T Lambda P`. h не входит в метрику.
Multi-index model передаёт текущие alpha, eigenvalues и effective_tensor через
явный opt-in `solver_metric_context` fit-loop; старые двухаргументные callbacks
сохраняют контракт. Адаптер отклоняет повторные fixed/dynamic настройки.
В прямом `solve_fixed_coefficients` нужны `metric_alpha`, `metric_eigenvalues`,
`metric_tensor`; incoming P ортонормирован и Lambda имеет max=1.
Whitening допускает неортонормированный внутренний prior; выход и directions
completion возвращаются в исходные координаты. Дополнительная постоянная
память — один U-sized буфер, работа O(J*p*d*m); dense d*d metric нет.
Сертификаты v-подзадач и rank_scales относятся к whitened координатам;
correction Frobenius norm/singular values — к исходным. Orthogonal при
нулевых eigenvalues требует положительного floor; нечисловая SPD отклоняется.
Опция — ESTIMATOR, не гарантия восстановления или глобального rank optimum.
Проверки: `tests/test_svd_metric.py`; парный пилот и ограничения:
`experiments/svd_a_metric_2026-09-30/{AUDIT,REPORT}.md` (120 строк, без exceptions;
held-out correction не подтвердил улучшение, default не меняется).

Для публичного обучения используется существующий custom-solver hook:

```python
from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.solver.SVD import solve as solve_svd

model = ADP_multi_index(
    3, ADP_Config(), ADP_solver(solve_svd, rank=2, low_rank_target="correction", precondition_v=True)
).fit(X, y)
```

`solve` сначала вычисляет `g_j` локальным refit для входящего полного `P`.
Correction mode ортонормирует строки `P+Delta` через thin QR, проверяет полный
row rank и заново оценивает локальные коэффициенты. Matrix mode по-прежнему
дополняет найденные `r` правых направлений проекцией прежнего `P`; оставшиеся
`m-r` направления опираются на prior, а не определяются матрицей `B`.
Внешний fit-loop затем применяет обычную канонизацию по спектру
коэффициентов. Вариант CPU-only; текущий LSMR остаётся default.

Доказательство, независимый аудит и новые парные замеры:
`experiments/svd_improvements_2026-09-30/{AUDIT,REPORT}.md`.
Selection и untouched validation: по 144 успешных строки; размеры d100/300/600,
оба rank-режима, обычные/масштабированные столбцы, full fit d50/150.
Validation: time ratio `.2255` на масштабированных fixed-g данных,
`.9430` для default на обычных; full-fit качество отличается <=2.27e-8.
Это численное совпадение и локальные замеры, не новое утверждение восстановления.

Источники: **SRC-SVD-SOLVER**, **SRC-SOLVER-API**, **SRC-MI-CANONICAL**;
эталонные проверки: `tests/test_svd_solver.py`; fixed-g замеры:
`benchmarks/svd_correction.py`, `experiments/svd_correction_2026-09-29/`
(d=20) и `experiments/svd_correction_high_d_2026-09-29/` (d=100/300/600;
небольшой и изменчивый парный выигрыш только на d=600). Full-fit LSMR/SVD
comparison: `benchmarks/svd_vs_hybrid.py --comparison lsmr-svd` and
`experiments/lsmr_vs_svd_fullfit_2026-09-29/REPORT.md` (100/50/25 paired
seeds; SVD faster, large-case projector recovery favored LSMR; objectives
differ).

## Локаторы

| ID | Фрагмент | Команда sed |
|---|---|---|
| SRC-MI-FIT | публичный fit и `(d,m)` publication/transform | `rtk proxy sed -n '20,108p' ADP/core/multi/ADP_multi_index.py` |
| SRC-MI-RESULT | projector distance result API | `rtk proxy sed -n '12,51p' ADP/core/multi/ADP_multi_index_result.py` |
| SRC-MI-UTIL | orthogonality/rank/result validation | `rtk proxy sed -n '9,95p' ADP/core/multi/ADP_multi_index_utils.py` |
| SRC-MI-CANONICAL | weighted coefficient rotation and canonical spectrum | `rtk proxy sed -n '108,185p' ADP/engine/common/index_fit.py` |
| SRC-MI-WEIGHTS | projected principal/residual kernel geometry | `rtk proxy sed -n '88,184p' ADP/engine/common/weights.py` |
| SRC-MI-ALPHA | alpha search and exact compact-support reduction | `rtk proxy sed -n '259,389p' ADP/engine/common/calculus.py` |
| SRC-MI-DIRECTIONS | principal plus orthogonal random sketch | `rtk proxy sed -n '390,422p' ADP/engine/common/calculus.py` |
| SRC-MI-INIT | local basis/spectrum initialization paths | `rtk proxy sed -n '70,184p' ADP/engine/common/initialize.py` |
| SRC-MI-DRIVER | effective tensor, fit/trace/stop/selection | `rtk proxy sed -n '305,552p' ADP/engine/common/index_fit.py` |
| SRC-MI-ENGINE | basis QR, orientation, subspace/projector distance | `rtk proxy sed -n '12,69p' ADP/engine/multi_index/ADP_multi_index_engine.py` |
| SRC-GRASSMAN | profile/Schur/Jacobian/polar/solve | `rtk proxy sed -n '1,460p' ADP/solver/grassman.py` |
| SRC-GRASSMAN-OPTIM | safe local QR/bounded gradient/transpose GEMM/batched full GN; same optimizer | `rtk proxy sed -n '1,574p' ADP/solver/grassman_optim.py` |
| SRC-SVD-SOLVER | fixed-g rank-r objectives, default certified Cholesky/LSMR or experimental gradient-gain pairs, optional diagonal/metric scaling, prior completion or QR | `rtk proxy sed -n '1,1021p' ADP/solver/SVD.py` |

Все коды извлекаются из `ADP/`; полный каталог файлов — [README.md](README.md#каталог-исходников). Общая статистика и mass contract — [index-pipeline.md](index-pipeline.md).
