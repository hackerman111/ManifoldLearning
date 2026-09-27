# Solver layer: цели, обновления и сертификаты

## Общий контракт current index solver

Модели передают solver-у `U=(J,P,d)`, `I=(J,P)` и индекс: single `(d,)`, multi row-basis `(m,d)`. При normalized statistics внешний вектор `mass=(J,)` умножает вклад каждой точки; если I/U уже не нормированы, передается `mass=None`. `ADP_solver.fit_current` проверяет тип `HPAOResult`; адаптер старого пользовательского solver с `ADP_SolverResult` — иной контракт. Источники: **SRC-SOLVER-ADAPTER**, **SRC-STAT-CPU**.

Актуальные single/multi wrappers по умолчанию используют `ADP.solver.LSMR.solve`. В CLI выбор CG/LSMR/HYBRID явный. `CG.solve` — самостоятельная current реализация.

`LSMR.py` — current HPAO solver: принимает `index_init, U, I` и настройки текущей задачи (`mass`, `lambda_prox`, `max_steps`), возвращает `HPAOResult` с индексом, локальными коэффициентами и diagnostics. Он делает local refit и outer alternating updates, а SciPy/CuPy LSMR решает inner correction. Совместимый `LSMR.lsmr()` возвращает только индекс. `legacy_lsmr.py` — отдельная прежняя реализация: принимает `ADP_Statistics` и `beta`, использует `lambda_penalty`/`local_ridge`, возвращает `ADP_SolverResult` и реализует старые single/multi alternating steps. Это разные API, objective и backend boundaries. Внутренних вызовов `legacy_lsmr.solve` сейчас нет; сохранять его нужно только для внешних пользователей старого контракта.

## HPAO-LSMR: current default

При фиксированных статистиках solver минимизирует weighted least-squares по локальным коэффициентам и индексу. Для заданного индекса локальные коэффициенты refit-ятся отдельно по каждому j: у single используется скалярное решение, у multi — minimum-norm SVD решения малой задачи. Затем global correction строится LinearOperator-ом со строго парным forward/adjoint и решается LSMR как ridge least squares с `damp=sqrt(lambda_prox)`. Не материализуется матрица размера `(J*P) x (m*d)`.

Каждая correction ограничивается trust radius и проверяется нормальным residual certificate в исходных координатах. CPU/GPU operator предварительно умножает локальные coefficients на `sqrt(mass)` и использует их и в forward, и в adjoint; это точная перестановка скалярного множителя, без `(J,P)` массива масштабированных residual на каждом действии. В пределах correction `Aᵀr` вычисляется один раз и переиспользуется при certificate. Кандидат QR/SVD gauge-fix-ится до unit single vector или ортонормированного multi basis с согласованной ротацией коэффициентов, так что prediction сохраняется. После refit candidate принимается только если objective не вырос сверх roundoff. Проксимальный lambda удваивается для слишком большого/неуспешного шага и постепенно уменьшается после последовательности достаточно малых принятых шагов. Convergence требует одновременно малого relative loss change, шага, riemannian gradient, local gradient и orthogonality; сертификат должен выполниться два шага подряд.

В diagnostics попадают stop/iterations, normal residual, correction norm, gauge error, rank loss, objective history, lambdas и stationarity. Для multi basis расстояние вычисляется sign/basis-invariant через projector residual. Источники: **SRC-HPAO-ENTRY**, **SRC-HPAO-GAUGE**.

Для разбора достижения лимита HPAO также сохраняет `relative_loss_change`, `aligned_step`, `consecutive_certified_steps`, `certificate_failures` и счётчики `rejected_trials` по screening/normal residual/trust radius/objective. Это только diagnostics: на S1 итоговые loss, число принятых шагов и convergence точно совпали с прежним selection для 18/18 вызовов.

## HYBRID для multi-index

`HYBRID_multi.solve` не меняет внешний HPAO алгоритм: он передает `linear_solver="hybrid"` в current `LSMR.solve`. Для маленькой joint least-squares системы (с учетом лимитов числа unknowns и bytes) строится bounded design и делается cached dense SVD. В большой системе применяются forward/adjoint matrix-free действия и scaled augmented LSMR. Повторные проверки trust radius могут использовать проецированное Krylov-пространство, но отбраковка проверяет lower bound в исходных A/A*; она не заменяет финальную correction.

Точный live boundary: `LSMR.solve:87–88` отклоняет HYBRID при
`index.ndim != 2`; wrapper не поддерживает single vector. Общий HPAO
код поддерживает single только с LSMR. `HYBRID_single.solve` оставлен как
явный route, который делегирует current HPAO-LSMR; workspace HYBRID относится
только к multi-index.

По умолчанию `hybrid_inner_rtol=None`: остается строгая LSMR tolerance. Положительный `hybrid_inner_rtol` — отдельный opt-in APPROXIMATE режим multi-index: допускается меньшая точность inner correction, но проверяется сертификат `||normal residual||/(lambda*||correction||)` и запускается refinement при необходимости. CLI разрешает эту опцию только в `mode=multi` с `hybrid`. Не описывать ее как точный ускоренный режим. Источники: **SRC-HYBRID-HPAO**, **SRC-HYBRID-INDEX**, **SRC-CLI-CHECKS**, **SRC-CLI-INDEX**.

## Функционал β и аудит 2026-09-27

Полный вывод, категории математических сокращений и bounded HYBRID-профиль:
`docs/experiments/beta_functional_2026-09-27/report.md`; raw
`audit_final.json`; isolated reference `benchmarks/beta_functional_audit.py`.
Статистическая цель `F=1/2 sum mass ||I-U B.T c||²`, `BB.T=I`;
β-подзадача при фиксированном C добавляет `lambda/2 ||B-B0||²`,
снимая ортонормальность до gauge. Joint Hessian имеет cross-blocks
`sum mass*c_a*c_b*U.T U`; независимые solves по строкам неверны.

Thin QR `[U_j,I_j]` без rank truncation точно сохраняет все residual norms
и сокращает P до d+1 при `P>d+1`. На frozen d10/P40 ridge reference
прошёл; d100/P40 не сокращается. Для интеграции обязательны исходные
P-dependent local/dense cutoffs и прежние нормировки, поэтому прямой
`HYBRID.solve(B,V,y)` не является проверенным эквивалентным solver API.
Common-metric spectral solve требует proportional `U_j.T U_j`; frozen
inputs это опровергают. Gradient-PCA без разных local metrics меняет цель.

Профиль показывает cached SVD на d10 и уже BLAS-backed A/A* на d100.
Stationarity reassociation ускоряет отдельный kernel, но большие mass
меняют округление local certificate; design `multiply(out=...)` уменьшает
allocation, но медленнее. Все кандидаты остались isolated, production
не менялся. Тесты/границы измерений — в отчёте.

## Изолированные эксперименты повторного поиска 2026-09-24

`experiments/reduced_gauss_newton.py` содержит ограниченный CPU-прототип
для прежнего reduced objective. Он использует полный residual Jacobian
усечённого локального refit, матрично-свободный augmented LSMR, исходные
rank/cutoff guards и общий итоговый certificate. Это APPROXIMATE-метод с
условным математическим выводом; он не зарегистрирован в моделях/CLI.
Источник и проверка: `docs/experiments/multi_solver_retry_2026-09-24/`
`proof_reduced_gn.md`, `h7.md`; `tests/test_reduced_gauss_newton.py`.

Вторая попытка уточнить точность существующего опционального HYBRID
(`hybrid_inner_rtol=0.01`) уменьшила отказы linear certificate, но на
трудной frozen точке замедлила выполнение и не изменила AO-сходимость.
Код отклонённой поправки архивирован в `h6_rejected.patch`, не применяется
к live `ADP/solver/HYBRID/HYBRID_multi.py`. На трёх худших reduced-L-BFGS endpoints
ограниченный диагностический Hessian имеет положительный спектр; это
свидетельство иных локальных областей, а не общей гарантии для всех входов.
Итог, условия измерений и границы интерпретации: `completion.md` той же папки.

## Отдельный CG для single/multi

`CG.solve` выполняет alternating update: локальные коэффициенты refit, затем глобальная регуляризованная quadratic subproblem решается preconditioned CG. Оператор matrix-free, но реализует normal operator `A.T @ A`; это точное решение proximal subproblem, не augmented least-squares formulation. Код проверяет CG status и residual, нормализует/gauge-fix-ит кандидат, заново refit-ит коэффициенты, проверяет отсутствие роста objective и считает stationarity diagnostics. С учетом repository numerical rules предпочитайте LSMR как baseline при чувствительности к обусловленности; CG выбирается явно и требует сравнения на задаче. Источники: **SRC-CG-AO**, **SRC-CG-OP**.

## Manifold B-solvers

Manifold `one_step` на каждом target строит local slopes и решает задачу для `B=(m,d)`. `solver="cg"` строит matrix-free SPD normal operator и Jacobi preconditioner, вызывает SciPy CG, затем отдельно вычисляет относительный residual. Несмотря на отсутствие dense `(m*d)^2`, это normal-equation путь. Источники: **SRC-MAN-STEP**, **SRC-MAN-B-SYSTEM**.

`solver="hybrid"` (в `HYBRID_manifold.py`, отдельно от HPAO multi solver) сначала выбирает bounded dense-SVD, если подходят лимиты. Иначе делает block-PCG с небольшим m-by-m block preconditioner; при проблеме PCG формирует augmented operator с факторизованным manifold penalty и пробует LSMR. Результат повторно сертифицируется градиентом исходной manifold-задачи. Диагностики называют backend (например `block-pcg->scaled-lsmr`), итерации и residual. Источники: **SRC-HYBRID-MANIFOLD**, **SRC-MAN-STEP**.

## Совместимость и риск путаницы

- `ADP/core/ADP_Solver.py::ADP_solver.fit_current` — current путь, требует `HPAOResult`; `ADP_solver.fit` предназначен для legacy adapter-контракта.
- `ADP/core/ADP_Solver.py::ADP_Solver.fit` — прежняя orchestration, вызывает legacy-shaped solver и возвращает индекс без `HPAOResult`.
- `ADP/solver/legacy_lsmr.py` не является алиасом текущего HPAO `LSMR.py`.
- GPU solver поддерживает current LSMR. GPU статистика может либо держать I/U на GPU для него, либо передавать I/U блоками на CPU; HYBRID/CG с device U не поддержаны (source: **SRC-GPU-STAT**, **SRC-HPAO-ENTRY**, **SRC-SOLVER-ADAPTER**).

## Локаторы

| ID | Фрагмент | Команда sed |
|---|---|---|
| SRC-HPAO-ENTRY | HPAO input contract, checks, outer AO loop, line search and convergence diagnostics | `rtk proxy sed -n '40,340p' ADP/solver/LSMR.py` |
| SRC-HPAO-LOCAL | local refit, shapes и rank-sensitive fallback | `rtk proxy sed -n '429,489p' ADP/solver/LSMR.py` |
| SRC-HPAO-OP | matrix-free global operator, adjoint and damped LSMR correction/certificate | `rtk proxy sed -n '504,596p' ADP/solver/LSMR.py` |
| SRC-HPAO-GAUGE | gauge fix, invariant distance и stationarity | `rtk proxy sed -n '597,684p' ADP/solver/LSMR.py` |
| SRC-CG-AO | alternating CG solve and objective checks | `rtk proxy sed -n '34,130p' ADP/solver/CG.py` |
| SRC-CG-OP | normal operator, Jacobi preconditioner, CG residual | `rtk proxy sed -n '155,276p' ADP/solver/CG.py` |
| SRC-HYBRID-INDEX | dense/augmented linear solvers and RidgeWorkspace | `rtk proxy sed -n '1,338p' ADP/solver/HYBRID/HYBRID_multi.py` |
| SRC-HYBRID-HPAO | HPAO delegation and opt-in inner tolerance | `rtk proxy sed -n '327,338p' ADP/solver/HYBRID/HYBRID_multi.py` |
| SRC-HYBRID-MANIFOLD | manifold PenaltyRoot, block PCG, augmented fallback | `rtk proxy sed -n '1,291p' ADP/solver/HYBRID/HYBRID_manifold.py` |
| SRC-MULTI-OP | multi forward/adjoint algebra | `rtk proxy sed -n '1,29p' ADP/solver/_multi_operator.py` |
| SRC-SOLVER-ADAPTER | current vs legacy adapter contracts | `rtk proxy sed -n '23,179p' ADP/core/ADP_Solver.py` |
| SRC-LEGACY-SOLVER | older LSMR implementation | `rtk proxy sed -n '1,350p' ADP/solver/legacy_lsmr.py` |

Перед изменением математики прочитайте тесты конкретной задачи и запишите, сохраняется ли цель (`EXACT`/`NUMERICAL`) или меняются solver tolerance/estimator (`APPROXIMATE`/`ESTIMATOR`). Полный каталог — [README.md](README.md#каталог-исходников).
