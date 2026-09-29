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

## Явный SVD-решатель: ранг матрицы и ранг поправки

Материалы исходной постановки теперь находятся в `SVD/SVD.tex` и
`SVD/SVD_form.tex`; прежний корневой `SVD_corr.tex` отсутствует в текущем
рабочем дереве. Поведение режима поправки сверять по живому коду ниже.
Единое изложение: [SVD_solver.tex](../../SVD/SVD_solver.tex), разделы
`sec:variants`, `sec:limits`, `sec:second`, `sec:gpu`, `sec:algorithm`.
Приложение `sec:sources` учитывает все шесть файлов исходной папки, включая
пустой `chat_2.md`. Предложенные второй решатель, совместные блоки и новые
правила поиска не являются реализованными возможностями. Проверка формул:
`rtk proxy python SVD/documentation/check_math.py`; запись проверки и сборки —
`SVD/documentation/VERIFICATION.md`.

`ADP/solver/SVD.py` реализует два явных `rank=r<m` варианта для фиксированных
локальных коэффициентов `g_j`. Общий функционал ровно
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
| SRC-MI-FIT | публичный fit и `(d,m)` publication/transform | `rtk proxy sed -n '20,104p' ADP/core/multi/ADP_multi_index.py` |
| SRC-MI-RESULT | projector distance result API | `rtk proxy sed -n '12,51p' ADP/core/multi/ADP_multi_index_result.py` |
| SRC-MI-UTIL | orthogonality/rank/result validation | `rtk proxy sed -n '9,95p' ADP/core/multi/ADP_multi_index_utils.py` |
| SRC-MI-CANONICAL | weighted coefficient rotation and canonical spectrum | `rtk proxy sed -n '108,185p' ADP/engine/common/index_fit.py` |
| SRC-MI-WEIGHTS | projected principal/residual kernel geometry | `rtk proxy sed -n '88,184p' ADP/engine/common/weights.py` |
| SRC-MI-ALPHA | alpha search and exact compact-support reduction | `rtk proxy sed -n '259,389p' ADP/engine/common/calculus.py` |
| SRC-MI-DIRECTIONS | principal plus orthogonal random sketch | `rtk proxy sed -n '390,422p' ADP/engine/common/calculus.py` |
| SRC-MI-INIT | local basis/spectrum initialization paths | `rtk proxy sed -n '70,184p' ADP/engine/common/initialize.py` |
| SRC-MI-DRIVER | effective tensor, fit/trace/stop/selection | `rtk proxy sed -n '304,540p' ADP/engine/common/index_fit.py` |
| SRC-MI-ENGINE | basis QR, orientation, subspace/projector distance | `rtk proxy sed -n '12,69p' ADP/engine/multi_index/ADP_multi_index_engine.py` |
| SRC-SVD-SOLVER | fixed-g rank-r objectives, certified Cholesky/LSMR, optional diagonal scaling, bounded factor/energy caches, prior completion or QR | `rtk proxy sed -n '1,782p' ADP/solver/SVD.py` |

Все коды извлекаются из `ADP/`; полный каталог файлов — [README.md](README.md#каталог-исходников). Общая статистика и mass contract — [index-pipeline.md](index-pipeline.md).
