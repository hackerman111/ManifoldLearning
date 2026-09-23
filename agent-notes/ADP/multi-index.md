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

Все коды извлекаются из `ADP/`; полный каталог файлов — [README.md](README.md#каталог-исходников). Общая статистика и mass contract — [index-pipeline.md](index-pipeline.md).
