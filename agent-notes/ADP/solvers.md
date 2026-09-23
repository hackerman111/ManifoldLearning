# Solver layer: цели, обновления и сертификаты

## Общий контракт current index solver

Модели передают solver-у `U=(J,P,d)`, `I=(J,P)` и индекс: single `(d,)`, multi row-basis `(m,d)`. При normalized statistics внешний вектор `mass=(J,)` умножает вклад каждой точки; если I/U уже не нормированы, передается `mass=None`. `ADP_solver.fit_current` проверяет тип `HPAOResult`; адаптер старого пользовательского solver с `ADP_SolverResult` — иной контракт. Источники: **SRC-SOLVER-ADAPTER**, **SRC-STAT-CPU**.

Актуальные single/multi wrappers по умолчанию используют `ADP.solver.LSMR.solve`. В CLI выбор CG/LSMR/HYBRID явный. `CG.solve` — самостоятельная current реализация; `legacy_lsmr.py` — отдельный прежний путь.

## HPAO-LSMR: current default

При фиксированных статистиках solver минимизирует weighted least-squares по локальным коэффициентам и индексу. Для заданного индекса локальные коэффициенты refit-ятся отдельно по каждому j: у single используется скалярное решение, у multi — minimum-norm SVD решения малой задачи. Затем global correction строится LinearOperator-ом со строго парным forward/adjoint и решается LSMR как ridge least squares с `damp=sqrt(lambda_prox)`. Не материализуется матрица размера `(J*P) x (m*d)`.

Каждая correction ограничивается trust radius и проверяется нормальным residual certificate в исходных координатах. Кандидат QR/SVD gauge-fix-ится до unit single vector или ортонормированного multi basis с согласованной ротацией коэффициентов, так что prediction сохраняется. После refit candidate принимается только если objective не вырос сверх roundoff. Проксимальный lambda удваивается для слишком большого/неуспешного шага и постепенно уменьшается после последовательности достаточно малых принятых шагов. Convergence требует одновременно малого relative loss change, шага, riemannian gradient, local gradient и orthogonality; сертификат должен выполниться два шага подряд.

В diagnostics попадают stop/iterations, normal residual, correction norm, gauge error, rank loss, objective history, lambdas и stationarity. Для multi basis расстояние вычисляется sign/basis-invariant через projector residual. Источники: **SRC-HPAO-ENTRY**, **SRC-HPAO-GAUGE**.

## HYBRID для single/multi

`HYBRID.solve` не меняет внешний HPAO алгоритм: он передает `linear_solver="hybrid"` в current `LSMR.solve`. Для маленькой joint least-squares системы (с учетом лимитов числа unknowns и bytes) строится bounded design и делается cached dense SVD. В большой системе применяются forward/adjoint matrix-free действия и scaled augmented LSMR. Повторные проверки trust radius могут использовать проецированное Krylov-пространство, но отбраковка проверяет lower bound в исходных A/A*; она не заменяет финальную correction.

По умолчанию `hybrid_inner_rtol=None`: остается строгая LSMR tolerance. Положительный `hybrid_inner_rtol` — отдельный opt-in APPROXIMATE режим multi-index: допускается меньшая точность inner correction, но проверяется сертификат `||normal residual||/(lambda*||correction||)` и запускается refinement при необходимости. CLI запрещает эту опцию вне `mode=multi, solver=hybrid`. Не описывать ее как точный ускоренный режим. Источники: **SRC-HYBRID-HPAO**, **SRC-HYBRID-INDEX**, **SRC-CLI-CHECKS**, **SRC-CLI-INDEX**.

## Отдельный CG для single/multi

`CG.solve` выполняет alternating update: локальные коэффициенты refit, затем глобальная регуляризованная quadratic subproblem решается preconditioned CG. Оператор matrix-free, но реализует normal operator `A.T @ A`; это точное решение proximal subproblem, не augmented least-squares formulation. Код проверяет CG status и residual, нормализует/gauge-fix-ит кандидат, заново refit-ит коэффициенты, проверяет отсутствие роста objective и считает stationarity diagnostics. С учетом repository numerical rules предпочитайте LSMR как baseline при чувствительности к обусловленности; CG выбирается явно и требует сравнения на задаче. Источники: **SRC-CG-AO**, **SRC-CG-OP**.

## Manifold B-solvers

Manifold `one_step` на каждом target строит local slopes и решает задачу для `B=(m,d)`. `solver="cg"` строит matrix-free SPD normal operator и Jacobi preconditioner, вызывает SciPy CG, затем отдельно вычисляет относительный residual. Несмотря на отсутствие dense `(m*d)^2`, это normal-equation путь. Источники: **SRC-MAN-STEP**, **SRC-MAN-B-SYSTEM**.

`solver="hybrid"` (реализация в HYBRID.py, не смешивать с HPAO HYBRID выше) сначала выбирает bounded dense-SVD, если подходят лимиты. Иначе делает block-PCG с небольшим m-by-m block preconditioner; при проблеме PCG формирует augmented operator с факторизованным manifold penalty и пробует LSMR. Результат повторно сертифицируется градиентом исходной manifold-задачи. Диагностики называют backend (например `block-pcg->scaled-lsmr`), итерации и residual. Источники: **SRC-HYBRID-MANIFOLD**, **SRC-MAN-STEP**.

## Совместимость и риск путаницы

- `ADP/core/ADP_Solver.py::ADP_solver.fit_current` — current путь, требует `HPAOResult`; `ADP_solver.fit` предназначен для legacy adapter-контракта.
- `ADP/core/ADP_Solver.py::ADP_Solver.fit` — прежняя orchestration, вызывает legacy-shaped solver и возвращает индекс без `HPAOResult`.
- `ADP/solver/legacy_lsmr.py` не является алиасом текущего HPAO `LSMR.py`.
- GPU solver поддерживает current LSMR. GPU статистика может либо держать I/U на GPU для него, либо передавать I/U блоками на CPU; HYBRID/CG с device U не поддержаны (source: **SRC-GPU-STAT**, **SRC-HPAO-ENTRY**, **SRC-SOLVER-ADAPTER**).

## Локаторы

| ID | Фрагмент | Команда sed |
|---|---|---|
| SRC-HPAO-ENTRY | HPAO input contract, checks, outer AO loop, line search | `rtk proxy sed -n '40,311p' ADP/solver/LSMR.py` |
| SRC-HPAO-LOCAL | local refit, shapes и rank-sensitive fallback | `rtk proxy sed -n '400,460p' ADP/solver/LSMR.py` |
| SRC-HPAO-OP | matrix-free global operator, adjoint, damped LSMR/certificate | `rtk proxy sed -n '468,564p' ADP/solver/LSMR.py` |
| SRC-HPAO-GAUGE | gauge fix, invariant distance и stationarity | `rtk proxy sed -n '565,652p' ADP/solver/LSMR.py` |
| SRC-CG-AO | alternating CG solve and objective checks | `rtk proxy sed -n '34,130p' ADP/solver/CG.py` |
| SRC-CG-OP | normal operator, Jacobi preconditioner, CG residual | `rtk proxy sed -n '155,276p' ADP/solver/CG.py` |
| SRC-HYBRID-INDEX | dense/augmented linear solvers and RidgeWorkspace | `rtk proxy sed -n '29,420p' ADP/solver/HYBRID.py` |
| SRC-HYBRID-HPAO | HPAO delegation and opt-in inner tolerance | `rtk proxy sed -n '421,434p' ADP/solver/HYBRID.py` |
| SRC-HYBRID-MANIFOLD | manifold PenaltyRoot, block PCG, augmented fallback | `rtk proxy sed -n '435,703p' ADP/solver/HYBRID.py` |
| SRC-MULTI-OP | multi forward/adjoint algebra | `rtk proxy sed -n '1,29p' ADP/solver/_multi_operator.py` |
| SRC-SOLVER-ADAPTER | current vs legacy adapter contracts | `rtk proxy sed -n '23,179p' ADP/core/ADP_Solver.py` |
| SRC-LEGACY-SOLVER | older LSMR implementation | `rtk proxy sed -n '1,350p' ADP/solver/legacy_lsmr.py` |

Перед изменением математики прочитайте тесты конкретной задачи и запишите, сохраняется ли цель (`EXACT`/`NUMERICAL`) или меняются solver tolerance/estimator (`APPROXIMATE`/`ESTIMATOR`). Полный каталог — [README.md](README.md#каталог-исходников).
