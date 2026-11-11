# Solver layer: цели, обновления и сертификаты

## Общий контракт current index solver

Модели передают solver-у `U=(J,P,d)`, `I=(J,P)` и индекс: single `(d,)`, multi row-basis `(m,d)`. При normalized statistics внешний вектор `mass=(J,)` умножает вклад каждой точки; если I/U уже не нормированы, передается `mass=None`. `ADP_solver.fit_current` также принимает opt-in dynamic metric kwargs для SVD (см. multi-index.md), отклоняя дубликаты fixed settings; проверяет тип `HPAOResult`; адаптер старого пользовательского solver с `ADP_SolverResult` — иной контракт. Источники: **SRC-SOLVER-ADAPTER**, **SRC-STAT-CPU**.

Актуальные single/multi wrappers по умолчанию используют `ADP.solver.LSMR.solve`. В CLI выбор CG/LSMR/HYBRID явный. `CG.solve` — самостоятельная current реализация.

`LSMR.py` — current HPAO solver: принимает `index_init, U, I` и настройки текущей задачи (`mass`, `lambda_prox`, `max_steps`), возвращает `HPAOResult` с индексом, локальными коэффициентами и diagnostics. Он делает local refit и outer alternating updates, а SciPy/CuPy LSMR решает inner correction. Совместимый `LSMR.lsmr()` возвращает только индекс. `legacy_lsmr.py` — отдельная прежняя реализация: принимает `ADP_Statistics` и `beta`, использует `lambda_penalty`/`local_ridge`, возвращает `ADP_SolverResult` и реализует старые single/multi alternating steps. Это разные API, objective и backend boundaries. Внутренних вызовов `legacy_lsmr.solve` сейчас нет; сохранять его нужно только для внешних пользователей старого контракта.

## Экспериментальный SVD multi-index

`ADP.solver.SVD.solve` / `solve_fixed_coefficients` поддерживают
`rank_one_search="gradient"`: отбор сингулярных пар Q=-grad(F)/2 по
точному выигрышу R²/D, шаг R/D и общий refit масштабов.
Default `"alternating"` сохраняет условные a/v solves и сертификаты.
Оба rank targets и metric whitening поддержаны; остановки gradient
эвристические. У него `v_normal_residual_applicable=False` и пустые
inner/LSMR tuples. Контракт, ограничения и актуальный SRC-SVD-SOLVER:
[multi-index.md](multi-index.md#явный-svd-решатель-ранг-матрицы-и-ранг-поправки).

## Профильный Grassmann (opt-in)

Английская история SVD-derived попыток и решений:
[GRASSMANN_IMPROVEMENT_HISTORY.md](../../../docs/SVD/GRASSMANN_IMPROVEMENT_HISTORY.md).
Основной фокус18–22октября, с источниками ранних core/flat/full-GN/Schur/
metric/rank-k идей; различаются пилоты, failed gates и accepted relative GN.
Live алгоритм целиком: [ADP_Grassmann_current.tex](../../../docs/tex/ADP_Grassmann_current.tex)
(PDF рядом), verification и hashes рядом с TeX.

Замер production core_gn (2026-10-06):
`benchmarks/experiments/grassman_stage_profile_2026_10_06/report.md`. На трёх тяжёлых
Spokoiny fit n800/d50/m2/J800/P10, один BLAS thread, local profile/refit
занимает pooled47.3% solver wall (33.3–57.8% по сидам); solve10.3–22.2% fit.
Timer-only профили и baseline имеют точное совпадение basis/diagnostics;
цифры зависят от fallback-пути и не являются общей оценкой всех задач.

`ADP.solver.grassman.solve`: default20, relative intrinsic GN + quadratic3
rank-one Armijo + exact zero-U exclusion; smooth_max_steps=None. Full q*m-core
profiled GN + polar Armijo,
NumPy/CuPy multi-index HPAOResult. Параметры method/angle_backend удалены; альтернативы
rank_one/spectral и Schur/refit абляции доступны только в grassman_exp.
Внутренние rank/workspace/no-descent fallback-шаги core_gn сохранены.
Default tau=0 сохраняет unpenalized finite-sketch профиль; positive tau
отдельный estimator. GN damping не permanent penalty. No global guarantee;
rank/workspace guards и uncompleted status диагностируются. API/границы и
локаторы: [multi-index.md](multi-index.md#профильный-grassmann-solver-opt-in).

`relative_gn=True` добавляет regular-center GN proposal при
численном intrinsic rank loss и damping=lambda_prox*||J||F²/(q*m). APPROXIMATE
optimizer; полные gradient/trial/final profile и прежняя applicability сохранены.
Smooth GN прежний; недопустимая модель ->
direct rank-one Armijo, workspace/плохой damping -> прежний spectral fallback.
Dev105800 и unchanged heldout106800 проходят все quality/time/RSS/outlier gates;
отчёты `benchmarks/experiments/grassman_accuracy_2026_10_18/RELATIVE_GN_*REPORT.md`.
Это n500,m2,d5/20/50 evidence, без гарантии сходимости. Public port120/120
full-fit bases/diagnostics точно совпадает с сохранёнными prototype/baseline.

С2026-10-22 пользователь явно запросил объединение прежних joint time/EDR
gains ниже старого порога и больший бюджет. Текущий combined20 прошёл150+150
paired fits с реальным R MEANMAVE, validation EDRgeo.925311/time.860129 vs old5.
Отдельная польза состава/бюджета vs relative10 не доказана: final EDR1.003418,
time1.275697. COMBINED_{MATH,PROTOCOL,REPORT}.md. Exact zero-U reduction
восстанавливает полный loss/coefs/rank count и прежний gradient scale;
stationarity относится к active objective плюс известные константы.
Прежний accepted10: relativeTrue,quadratic0,constantFalse,max10,smooth5.
Исходный контроль: relativeFalse,quadratic0,constantFalse,max5.

Новый bounded accuracy поиск22октября: `benchmarks/experiments/grassman_accuracy_2026_10_22/`
(`MATH.md`, `REPORT.md`, immutable `reference.py`, `candidates.py`, `benchmark.py`).
80 полных paired ADP/R MEANMAVE fits на reused90800, n500,m2,d5/20/50/corr.9.
Девять новых вариантов: orthogonal Jacobian, smooth-relative damping,
диагональная плоскость, GN length interpolation, три local_ridge,
intrinsic full-J/J* Krylov plane10/20. Ни один не дал joint EDR/time выигрыша;
в этом первоначальном поиске111800/112800 не использовались, production не менялся. Это pilot rejection,
не доказательство невозможности улучшения. Не повторять как новый эксперимент.

Продолжение по запросу LoRA/rank-k: в той же папке `lowrank.py`,
`LOWRANK_MATH.md`, `LOWRANK_REPORT.md`, `lowrank_analyze.py`.
APPROXIMATE proposals: joint k*m coefficients в gradient SVD space,
двухфакторное ALS/LSMR уточнение rank-k additive Delta, расширение плоскости
по frozen-g normal residual. Fixed-g цель — только surrogate; polar/reprofile,
полные Armijo и final certificate обязательны. Не смешивать с угловым
`block_rank_k` или SVD.py matrix target; rank(Delta) задан до retraction.
Standalone helpers требуют contiguous U, чтобы не копировать весь тензор.
Дополнительный projected-Jacobian proposal сохраняет J.T r, удаляя PSD
слагаемое кривизны. Pilot105fits/семь paired cases90800 завершён; выбран
projected10 для development111800..809 с baseline20/budget10/реальным R.
Development280fits/70datasets не прошёл gate: EDR/base1.00065,
CI[.98131,1.02227], time median.737; P10/budget10 EDR1.00169.
LoRA_k10 остаётся только pilot evidence, не независимым улучшением.
Validation112800..809 не запускался; production не изменён. Статус — PLAN/STATE.
Независимые проверки: tests/test_grassman_{lowrank22,projected22}.py.

`grassman.py` теперь содержит проверенную CPU-оптимизацию core_gn:
EXACT/NUMERICAL при прежних настройках, цели и HPAOResult. Safe m<=2
profiles используют reorthogonalized QR/triangular solves; unsafe, p<m или
m>2 — исходный SVD. Градиент с bounded scratch 4 MiB, transpose GEMM,
батчевый полный q*m Jacobian и live финальный сертификат сохранены.
Paired evidence: `benchmarks/experiments/grassman_optim_2026_10_01/{REPORT,LEAD_AUDIT}.md`:
frozen speedup2.52x; fullfit median ratio0.942, final projectors близки.
Старый SVD эталон: `benchmarks/experiments/grassman_optim_2026_10_01/reference.py`.
`grassman_optim.py` удалён; экспериментальные режимы — `grassman_exp.py`,
общие numerical kernels импортируются из `grassman.py`.
`grassman_exp.solve(method="flat_gn")` (default experimental method) — APPROXIMATE opt-in из glm.tex: выбирает
q парных SVD-плоскостей и делает до `flat_max_steps=8` совместных GN по
q углам с точным cached Exp/profile/Jacobian/Armijo. Ambient U отсутствует
в inner loop; unknowns q вместо q*m. Углы ограничены `max_angle`; глобальный
минимум тора не обещан. Rank guard -> rank_one, workspace/no descent -> spectral.
Диагностики добавляют flat_angles, flat_inner_iterations и flat_fallbacks.
Математический маршрут: `benchmarks/experiments/grassman_flat_2026_10_02/PROOF.md`;
Luna проверки: `tests/test_grassman_flat.py`, benchmark/report рядом в
`benchmarks/grassman_flat_benchmark.py`, `benchmarks/experiments/grassman_flat_2026_10_02/`.
45 tests/120 rows passed execution, but adaptive quality/time selection fails;
joint flat stays experimental, heldout skipped. REPORT distinguishes frozen
objective gains from EDR recovery and records missing fullfit orth metric.
Источники **SRC-GRASSMAN**: `sed -n '1,857p' ADP/solver/grassman.py`;
**SRC-GRASSMAN-EXP**: `sed -n '1,819p' ADP/solver/grassman_exp.py`;
flat kernels:32-150, cached spectral:153-180, solve:183-571;
full tangent J/J*:574-661, `solve_full_gn`:664-819.
Full tangent GN тоже не прошёл selection; сохранён
явным экспериментальным entry point. Исторический
`benchmarks/experiments/glm_grassmann_2026_10_02/full_gn.py` реэкспортирует этот solver.
Перенос: `benchmarks/experiments/grassman_promotion_2026_10_03/relocation_parity.json`.

`grassman.block_rank_k` (813-857) — отдельный явный экспериментальный вход,
`grassman.solve` и его сигнатура сохранены. Делегирует
`grassman_exp.solve(method="block_rank_k")`: rank_k фиксирован или None;
rank_strategy energy/cost; block_variant spectral/optimized. Spectral точно
переиспользует существующий общий шаг; optimized тот же flat GN, по умолчанию
два совместных шага. Cost-пилот проверяет максимум два prefix ranks один раз
за solve, использует true-profile decrease / measured trial+projection time,
затем ограничивает energy rank выбранным cap. Это эвристика с timer noise.
Rank guard -> rank_one; workspace/no descent -> spectral, live full-gradient
сертификат. Workspace относится к extra GN scratch, не input/profile cache.
APPROXIMATE, цель прежняя; новая теория глобальной сходимости не заявляется.
Протокол/условная математика/результаты: `benchmarks/experiments/grassman_rank_k/`;
тесты `tests/test_grassman_rank_k.py`, независимый reference
`tests/test_grassman_rank_reference.py`, counter/polish tests. Исследование
2026-10-16 завершено: `REPORT.md`, 71 focused tests pass, 881 workers ok.
Heldout block1/core fit ratio .99963; порог .8 не пройден. Одношаговый F
улучшается с k, но EDR/time advantage не подтверждён. Core оставлен основным.
Сырые результаты и точные исходники сохранены, reference22/24 certified;
глобальная экспоненциальная гарантия не доказана, см. MATH.md.

## Структурированный Grassmann: явный research entry point

`ADP.solver.grassman_research.solve` — opt-in APPROXIMATE optimizer той же
finite-sketch профильной цели. Методы `schur_gn`, `schur_newton`, `metric`;
параметры `precondition`, `recycle_steps`, `core_steps`, `metric_floor`.
`lambda_prox` — damping, `local_ridge` — tau (положительный меняет estimator).
`precondition` применяется только к Newton CG; Schur GN решает augmented
least squares напрямую. В Newton lambda_prox задаёт положительный floor
предобусловливателя, без добавления штрафа к целевой функции.
Production `grassman.solve` и текущие defaults не менялись.

`statistic_operator.py`: `DenseOperator(U)` и `FactorizedOperator(C,X,...)`
предоставляют `project(V)`, `adjoint(P)`, `shape`, `nbytes`. Функция
`factorized_statistics(X,y,weights,directions,normalized=...)` возвращает
`(operator,I,mass)` без dense U; ordered weight blocks поддержаны. C хранится
в CSR, X/y центрируются для устойчивости. При normalized=True передать mass
solver-у; при False передать mass=None (масса уже в C/I). API existing
`calculate_statistics` не менялся; CX выбирается явным frozen вызовом.

`_grassman_schur.SchurCore` поддерживает полный rectangular q*a core и A=(m,a)
с исключением m-a коэффициентов через augmented SVD. Производные учитывают
ridge-метрику полярного чарта. `hvp` точный внутри выбранного чарта;
Steihaug сохраняет отрицательную кривизну. Положительные block-diagonal
предобусловливатели изменяют CG, не цель. Shared solver использует a=m
(q*m, как baseline); kernel API дополнительно поддерживает a<m.

Recycling использует cached U[Y,V] лишь внутри одного frozen solve; ограниченный
градиент не является сертификатом. Финальный UY и полный gradient всегда
пересчитываются; счётчики operator/full-gradient/recycled/Newton действий
есть в diagnostics. Tau=0 rank guard и workspace guard дают metric fallback.

Доказательство, протокол и Luna evidence:
`benchmarks/experiments/edr_structured_2026_10_15/{PROOF,REPORT}.md`, `protocol.json`,
`results/`; tests `test_edr_{operators,schur,research}.py`.
Локаторы: `rg -n '^class |^def ' ADP/solver/statistic_operator.py
ADP/solver/_grassman_schur.py ADP/solver/grassman_research.py`.

## Метрический спуск: hard Spokoiny, 2026-10-15

Маршрут углублённой проверки с GPT-6 Luna:
`benchmarks/experiments/metric_spokoiny_2026_10_15/{PROTOCOL,MATH,REPORT,LEAD_AUDIT}.md`,
raw/PNG/manifest/audits в `results_repaired/`, independent tests:
`tests/test_metric_spokoiny.py`. Численный код и defaults в этой задаче не
менялись. Original benchmark n800/d50/m2/J800/P10, tau0, seeds0/1/2.

При пяти inner шагах median metric2/core fit ratio.981649, solver.857432;
seed1 recovery regression5.666e-4 нарушает frozen selection gate, heldout3..7
не запускались. При20 шагах fit ratio1.015490, recovery почти совпадает.
На пяти jointly smooth-certified frozen парах (budget100,tol1e-6) metric2
медленнее2.040x; budget5 не даёт ни одного сертификата. Rank1 не имеет
устойчивого преимущества; local rank-boundary endpoints не сертифицированы.

R=sum mass*g*gT — coefficient metric, не Hessian. При m=q2/full rank
не меняет col(G); rank1 добавляет ограничение. Для точного генератора
population OPG=(5557/1155)I2, поэтому intrinsically weak-index premise здесь
не подтверждается; empirical R отличается от OPG. На9 input snapshots
condition R1.057..4.801, floor/caps неактивны. Common mass scaling c меняет
предложение на1/sqrt(c), не минимизатор. MATH и proposal_audit описывают
scale confound, isotropic и matched controls; нет общего geometry benefit.

Untimed profile только seed1/call160: metric36 accepted/core48, но249/50
profile calls и247/48 polar. Это evidence extra line-search work на данном
входе, не общий профиль всех фаз. Current metric остаётся experimental.
Original90 launcher failures (missing --snapshot) сохранены; repaired90/90
успешны в отдельном manifest с byte-identical imported fits/captures.

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

`benchmarks/experiments/reduced_gauss_newton.py` содержит ограниченный CPU-прототип
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
- GPU solver поддерживает current LSMR и Grassmann. GPU статистика может либо держать I/U на GPU для них, либо передавать I/U блоками на CPU; HYBRID/CG с device U не поддержаны (source: **SRC-GPU-STAT**, **SRC-HPAO-ENTRY**, **SRC-SOLVER-ADAPTER**). Новые accuracy prototypes22октября проверялись только на CPU.

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
| SRC-SOLVER-ADAPTER | current vs legacy adapter contracts | `rtk proxy sed -n '23,186p' ADP/core/ADP_Solver.py` |
| SRC-LEGACY-SOLVER | older LSMR implementation | `rtk proxy sed -n '1,350p' ADP/solver/legacy_lsmr.py` |

Перед изменением математики прочитайте тесты конкретной задачи и запишите, сохраняется ли цель (`EXACT`/`NUMERICAL`) или меняются solver tolerance/estimator (`APPROXIMATE`/`ESTIMATOR`). Полный каталог — [README.md](README.md#каталог-исходников).
