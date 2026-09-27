# Функционал β в AO: точные сокращения и аудит HYBRID

Дата: 2026-09-27. Объект — текущий CPU single/multi-index ADP. Основной
результат: функционал и β-подзадача формализованы; найдено точное сжатие
уравнений при `P>d+1`, но универсальный новый solver для `P<d` не установлен.
Проверены NumPy-бутылочные горлышки HYBRID. Production ADP не менялся.

## 1. Обозначения и источник формулы

- `n` — число наблюдений; `J` — число центров; `P` — число направлений;
  `d` — размер признаков; `m` — размер EDR-подпространства.
- `U_j ∈ R^(P×d)`, `I_j ∈ R^P` — фиксированные локальные моменты.
- `ω_j=mass_j≥0`; при legacy-ненормированных моментах solver использует
  `ω_j=1`. `mass` не равна effective sample size `n_eff`.
- `B∈R^(m×d)` хранит β по строкам, `C∈R^(J×m)` хранит `c_j^T` по строкам.
  Публичная multi `beta_` хранит транспонированный базис `(d,m)`.

Веса, направления, U/I/mass фиксированы внутри рассматриваемого AO solve.
Outer обновление локализации/моментов меняет конечную задачу; производные
ниже не дифференцируют outer rule выбора весов по текущему B.

При нормированных весах `a_ji=w_ji/Σ_i w_ji`, локальных средних `μ_j,ȳ_j`
и `z_ji=X_i−μ_j` имеем в точной арифметике

\[
\Sigma_j=\sum_i a_{ji}z_{ji}z_{ji}^{T},\qquad
t_j=\sum_i a_{ji}z_{ji}(Y_i-\bar y_j),\qquad
U_j=\Phi_j\Sigma_j,\quad I_j=\Phi_jt_j.
\]

Код дополнительно вычитает остаточный weighted first moment, накопленный
при округлении. Эти поправки сохраняются; приведённая формула объясняет
алгебру, а не разрешает заменить стабилизированную реализацию.
Локальный intercept исключается благодаря `Σ_i a_ji z_ji=0`.
Источник: `ADP/engine/common/statistic.py:29–229`.

**Исходная статистическая задача live-кода:**

\[
\boxed{F(B,C)=\frac12\sum_{j=1}^{J}\omega_j
       \|I_j-U_jB^{T}c_j\|_2^2,
       \qquad BB^{T}=I_m.}
\tag{1}
\]

Для single `m=1`, `B=β^T`, `c_j` скаляр, `||β||=1`.
При фиксированном B положительная масса не меняет локальный minimizer:

\[
Z_j=U_jB^T,\qquad c_j=Z_j^\dagger I_j.
\tag{2}
\]

При нулевой массе локальный коэффициент для (1) произволен; live-код
выбирает тот же refit. Численная реализация (2) — SVD с cutoff
`eps*max(P,m)*s_max`, а в чувствительных центрах — исходный `lstsq`.
Строго математическая `†` и cutoff-псевдообратная различаются на очень
малых ненулевых singular values. Нельзя менять cutoff при оптимизации.

При исключении C аналитическая reduced-цель равна

\[
F_{\rm red}(B)=\frac12\sum_j\omega_j
 \|(I_P-Z_j Z_j^\dagger)I_j\|^2.
\tag{3}
\]

Она имеет меньше неизвестных, но сохраняет nonconvex subspace search.
Дифференцирование гладко на областях постоянного ранга с отделёнными
singular values; смена ранга/численного cutoff требует отдельной защиты.
Это известный принцип [variable projection](https://www.cs.umd.edu/users/oleary/software/varpro.pdf),
не новая идея solver-а. Прошлые reduced L-BFGS/GN здесь не прошли joint gate;
см. `../multi_solver_search_2026-09-24/s4_final.md` и
`../multi_solver_retry_2026-09-24/completion.md`.

**Различие с TeX:** `tex/multiindex.tex:1010–1236` сначала задаёт
`1/2 Σω(||I−i||²+||i−prediction||²)`. Исключение auxiliary `i` даёт
`1/4 Σω||I−prediction||²`, тогда как live loss имеет множитель `1/2`.
Этот постоянный множитель не меняет unregularized minimizers, но влияет
на масштабы objective/gradient/tolerances. В TeX AO отдельно записана
неполовинная сумма SSE плюс `λ||B−B0||²`; она совпадает с (4) после
умножения всей (4) на 2. Не переносим λ между формулами с разной
относительной нормировкой штрафа автоматически.

## 2. Именно β-подзадача и её производные

После (2) коэффициенты C **фиксируются**. HPAO временно снимает
ортонормальность и ищет коррекцию `Δ=B−B0`:

\[
\boxed{Q_\lambda(B\mid C,B_0)=\frac12\sum_j\omega_j
  \|I_j-U_jB^Tc_j\|^2+\frac\lambda2\|B-B_0\|_F^2.}
\tag{4}
\]

`lambda_prox` — штраф именно этого correction-шагa; его **нет** в (1).
Пусть `r_j=I_j−U_j B0^T c_j`, `r=(sqrt(ω_j)r_j)_j`. Для row-major
`vec_r(B)` определим

\[
A\,\operatorname{vec}_r(D)
 = (\sqrt{\omega_j}U_jD^Tc_j)_j,\qquad
\operatorname{unvec}_r(A^*v)=\sum_j\sqrt{\omega_j}c_jv_j^TU_j.
\tag{5}
\]

Тогда решается `min_δ 1/2||Aδ−r||²+λ/2||δ||²` и

\[
(A^*A+\lambda I)\delta=A^*r.
\tag{6}
\]

Это вывод уравнения, не предложение формировать dense normal matrix.
Для `λ>0` решение единственно; для `λ=0` current correction выбирает
minimum norm. Правая Jacobi scaling при `λ=0` в HYBRID отключена именно
для сохранения minimum-norm semantics.

Градиент, Hessian-action и линейное уравнение непосредственно по B:

\[
\nabla_B Q=\sum_j\omega_j c_j
 (U_jB^Tc_j-I_j)^TU_j+\lambda(B-B_0),
\]
\[
\mathcal H_\lambda[D]=\sum_j\omega_j(c_jc_j^T)D(U_j^TU_j)+\lambda D,
\]
\[
\sum_j\omega_j(c_jc_j^T)B(U_j^TU_j)+\lambda B
 =\sum_j\omega_j c_j I_j^TU_j+\lambda B_0.
\tag{7}
\]

Блок `(a,b)` Hessian равен `Σ_j ω_j c_ja c_jb U_j^TU_j + λδ_ab I_d`.
Поэтому независимые solves для строк β **не эквивалентны**: cross-terms
обычно ненулевые. Например, `c=(1,1)` и `U=I_2` дают ненулевой off-diagonal
block `I_2`. Формула для одного индекса:

\[
\left(\sum_j\omega_j c_j^2U_j^TU_j+\lambda I_d\right)\beta
 =\sum_j\omega_j c_jU_j^TI_j+\lambda\beta_0.
\tag{8}
\]

После relaxed solve код делает QR gauge с согласованным преобразованием C,
сохраняя prediction, затем новый local refit и проверку уменьшения (1).
Это отличается от прямой минимизации (4) на Stiefel manifold.
Final stationarity включает tangent gradient
`G−sym(G B^T)B`, local gradients и ортонормальность; два последовательных
сертификата обязательны. Linear solver stop не заменяет этот AO certificate.

Для `e=A^*(r−Aδ)−λδ`, `λ>0`:

\[
\|\delta-\delta_*\|\le\|e\|/\lambda.
\]

Проверка `||e||/(λ||δ||)≤θ` даёт ошибку `≤θ||δ||`; относительно
`||δ_*||` граница `θ/(1−θ)` при `θ<1`. При `λ=0` нужен отдельный
spectral-gap bound для ошибки решения; нормальный residual сам по себе
такой границы не даёт.

## 3. Явные категории методов сокращения сложности

| Категория | Что исключает/сокращает | Статус для этой задачи |
|---|---|---|
| Статистические тождества и достаточные моменты | ADP-центрирование исключает intercept, `n` исходных уравнений заменяются направленными I/U; sparse support исключает нулевые веса | Уже применено; сохранять конечный sketch и стабилизированную нормировку. Замена sketch полным covariance loss — ESTIMATOR. |
| Аналитическое исключение nuisance variables | Auxiliary i, intercept, локальные c; variable projection (3) | Точно в математической области постоянного rank; меньшая размерность не гарантирует более дешёвый nonlinear solve или тот же basin. |
| Точные представления и геометрия | Row-space QR, факторизованные forward/adjoint, gauge вместо d×d projector; common feature subspace | EXACT при доказанной факторизации. Конкретное QR-сжатие доказано ниже. Quotient/Grassmann убирает gauge redundancy, но не нелинейность. |
| Специальная алгебра Hessian | Общая metric, commuting metrics, Kronecker/Sylvester или block diagonal structure | Даёт spectral solve только при дополнительных проверенных предпосылках; на frozen I/U общая metric отвергнута. |
| Численные алгоритмы с прежней целью | Matrix-free LSMR/LSQR, preconditioning, SVD малой задачи, Krylov recycling, trust bounds | Меняют цену solve/число итераций. Не исключают статистические элементы. Учитывать conditioning, setup и исходный certificate. |
| Приближённое решение прежней конечной цели | Stochastic center gradients, control variates, inexact inner solves, low-rank Hessian preconditioner | APPROXIMATE, либо preconditioner с полным certificate; требуются variance/rank/convergence proof и собственный bounded gate. Не реализованы здесь. |
| Другая статистическая процедура | ADE, gradient outer products, Stein/score moments, inverse-regression moments | Могут исключить AO полностью, но это ESTIMATOR с identifiability/density/eigengap условиями, не точное ускорение (1). |
| Реализация той же алгебры | BLAS, перестановка contractions, fusion/out, caching, chunking, sparse actions | Проверять wall и allocation отдельно. Выигрыш kernel не доказывает выигрыш fit. Проверенные кандидаты и ограничения ниже. |

Сам `ADP` — статистический приём исключения элементов задачи, а `LSMR` —
алгоритм решения оставшейся линейной подзадачи. Эти уровни не смешиваются.
Grassmann имеет размер `m(d−m)` вместо `md`; устраняется лишь gauge, а не
основная стоимость доступа к U. Common feature row-space размером `r<d`
также может сократить correction до `mr`, но union local row-spaces обычно
полного ранга, а строить его dense SVD при `d=1000` без бюджeта нельзя.

## 4. Точное сокращение №1: QR расширенных локальных моментов

**Теорема.** Пусть `W_j=[U_j I_j]∈R^(P×(d+1))`, его thin QR без
rank truncation есть `W_j=Q_j R_j`, `Q_j^TQ_j=I_q`,
`q=min(P,d+1)`. Разделим `R_j=[V_j y_j]`. Для любого `v∈R^d`:

\[
\boxed{\|I_j-U_jv\|^2=\|y_j-V_jv\|^2.}
\tag{9}
\]

**Доказательство:** `I_j−U_jv=W_j(−v,1)^T=Q_j R_j(−v,1)^T`,
а умножение на Q сохраняет норму. Следовательно, (1), (3), (4) и множества
их exact minimizers сохраняются при замене `(U,I)` на `(V,y)`.
Никакие eigenvalues не обнуляются. Теорема действует и при нулевом U,
ненулевом I, rank deficiency, любых неотрицательных mass.

Для joint design `A=diag(Q_j) A_small`, residual `r=diag(Q_j) r_small`.
Поэтому `A^*A`, `A^*r`, singular values, `||r||`, `||Aδ−r||` и исходный
normal residual неизменны; это также доказывает (9) независимо через
достаточные LS-моменты. Оставляем Q только на время QR, постоянного хранения
Q не требуется: `np.linalg.qr(..., mode="r")` возвращает R.

Важная деталь: если QR сделать лишь для U и отбросить `I_perp`, objective
меняется на константу `1/2 Σω||I_perp||²`. Minimizer может остаться тем же,
но relative loss change, gradient scaling и LS stopping уже другие.
Именно поэтому сжимаем `[U,I]`, включая весь I, а не отбрасываем константу.

**Цена:** setup `O(J P (d+1)^2)` только в выгодном режиме `P>d+1`;
постоянный массив моментов `8J(d+1)^2` bytes вместо `8JP(d+1)`.
Это маленький factor R, а не normal matrix для generic solve.
Forward/adjoint U-action после setup уменьшается `P/(d+1)` раз по числу
операций. При `P≤d+1` число строк остаётся P: метод не даёт сокращения,
и QR здесь запускать не следует. В stress `d=1000,P=40` не применим.

**Численная граница и API:** Householder QR backward stable, но не bitwise
identical. Для интеграции нужно пронести первоначальный P в
`eps*max(P,m)` local cutoff и `eps*max(JP,md)` dense cutoff, сохранить
нормировки certificate, iteration semantics и исходный finite/rank handling.
Прямой вызов `HYBRID.solve(B,V,y)` этого не гарантирует; он не рекомендуется
и здесь не подключён. Прототип проверяет representation и один
regularized fixed-C solve, не весь adaptive fit.

На трёх frozen d10 inputs `J=500,P=40,d=10,m=2` число строк стало 11,
моменты `1,760,000→484,000` bytes (−72.5%). Ошибка ridge correction
при `λ=1` составила `5.29e-17–7.63e-17`; normal-residual ratios в обеих
формах `1.42e-12–6.03e-12`.

| Seed | QR setup, ms (один sample) | Исходная workspace + correction, ms (median) | Сжатая workspace + correction, ms (median) |
|---|---:|---:|---:|
| 1000 | 5.39 | 8.34 | 2.30 |
| 1001 | 7.85 | 10.09 | 2.48 |
| 1002 | 5.38 | 8.61 | 2.18 |

Сумма setup и одного solve иногда дороже исходного solve; повторные AO
шаги могут амортизировать setup, пока U/I неизменны. Это **гипотеза цены
полного fit**, не проверенный full-fit speedup. QR нужно заново выполнять
после outer refresh I/U. Сжатие — известная LS-факторизация, новой
универсальной методикой не называется.

## 5. Условное сокращение №2: общая локальная metric

Пусть `U_j^TU_j=α_j H`, `α_j≥0`, один общий PSD H. Тогда (7) превращается
в generalized Sylvester equation:

\[
M B H+\lambda B=D,\quad
M=\sum_j\omega_j\alpha_j c_jc_j^T,\quad
D=\sum_j\omega_j c_jI_j^TU_j+\lambda B_0.
\tag{10}
\]

При `M=S diag(μ)S^T`, `H=T diag(η)T^T`:

\[
B=S Y T^T,\qquad Y_{ab}=\frac{(S^TDT)_{ab}}{\mu_a\eta_b+\lambda}.
\tag{11}
\]

Подстановка в (10) доказывает формулу; при `λ>0` знаменатели положительны.
Это может исключить Krylov solve, если H diagonal/factored/имеет уже
известный дешёвый eigenbasis. Формировать dense `d×d` H ради (11) в общем
случае нельзя: setup `O(d³)` и `O(d²)` память могут превосходить исходный solve.
Тест сравнивает (11) с независимым augmented LS лишь при `d=4`.

Если H дополнительно SPD, то `g_j=H^−1 U_j^T I_j/α_j` при `α_j>0`
и исходная joint-задача переписывается как weighted rank-m approximation
градиентов в **общей H-metric**, плюс остаточные константы. Через
`H^(1/2)` её можно свести к truncated SVD; любой rank-m factor затем
ортонормируется с компенсацией C. Этот вывод требует общей SPD metric,
а не просто smoothness f. При `P<d` каждый `U_j^TU_j` имеет rank≤P<d:
SPD H невозможна. Общая singular metric оставляет её nullspace
неидентифицируемым и не даёт уникального Euclidean EDR projector.

Проверка proportional metrics без `d×d`:

\[
\langle U_a^TU_a,U_b^TU_b\rangle_F=\|U_aU_b^T\|_F^2.
\]

Frobenius cosine первой пары активных центров на шести frozen inputs:
`0.671,0.591,0.341` (d10) и `0.190,0.370,0.317` (d100).
Для proportional positive metrics cosine обязан быть 1; разница намного
больше roundoff. Следовательно, (10) не является exact solver текущих inputs.
Замена разных H_j их средним — approximation/estimator, не доказанное тождество.
Commuting метрики дали бы отдельные малые системы по feature-eigenbasis,
но общего такого basis здесь не установлено; его dense поиск не предлагается.

## 6. Сокращение №3: average derivatives / gradient moments

Для smooth multi-index `f(x)=h(B_*x)` chain rule даёт
`∇f(x)=B_*^T∇h(B_*x)`. Тогда

\[
E[\nabla f(X)]=B_*^T E[\nabla h(B_*X)],\qquad
\Gamma=E[\nabla f\nabla f^T]=B_*^T K B_*,
\quad K=E[\nabla h\nabla h^T].
\tag{12}
\]

Один mean gradient имеет rank≤1, поэтому не восстанавливает общий m>1.
Он может быть нулём даже при m=1: `f(x)=x_1²`, симметричный X.
Gradient outer products восстанавливают EDR при `K≻0` и контролируемой
ошибке оценки Γ. Дополнительные smoothness/density/eigengap условия —
часть estimator, не следствие малого Y-MSE. Близкий отдельный подход:
[smoothed gradient outer products](https://arxiv.org/abs/2312.15469).

При известной differentiable density p и исчезающем boundary term
интегрирование по частям даёт `E[f(X)(−∇log p(X))]=E[∇f(X)]`.
Для Gaussian доступны также higher-order Stein identities, которые могут
избежать direct derivative estimation. Но первый score-moment сохраняет
проблему cancellation/rank≤1; higher-order moments требуют дополнительных
условий на link и могут иметь цену d². Не переносим Gaussian identity на
произвольное неизвестное распределение X.

Формальная замена ADP-цели градиентной PCA не exact. Пусть
`g_j=U_j^† I_j`, `e_j=I_j−U_jg_j`, `U_j^T e_j=0`. Для любого v:

\[
\|I_j-U_jv\|²=\|e_j\|²+(g_j-v)^T(U_j^TU_j)(g_j-v).
\tag{13}
\]

Из (13) видно: локальная metric остаётся. Обычная Euclidean PCA/усреднение
её выбрасывает; equivalence есть лишь при специальных условиях раздела 5.
При `P<d` minimum-norm g_j дополнительно произвольно фиксирует
неизмеренные направления. Пример `U=[1,0],I=1`: разные линии с ненулевой
первой координатой допускают нулевой ADP loss после local refit, тогда как
`g=U^†I=e_1` искусственно выбирает одну из них.

Численный counterexample уже для relaxed single с фиксированными c=1:
`g1=e1`, `g2=e2`, `H1=diag(100,1)`, `H2=I`. Правильный minimizer
`β=(100/101,1/2)`, средний gradient `(1/2,1/2)` отличается по первой
компоненте почти на 0.5. Поэтому gradient replacement закрыт как exact
solver; отдельный estimator-поиск сюда не включён.

## 7. Реальная сложность и скрытые расходы HYBRID

Current factorized forward сначала вычисляет `C B` размера `(J,d)`,
затем `U_j(CB)_j`. Adjoint делает `U_j^T v_j`, затем `C^T pulled`.
Цена пары действий `O(JPd+Jmd)`, рабочая память `O(Jd+JP+md)` сверх U.
Наивная materialization `(J,P,m)` для каждого Krylov action уже устранена.
Local refit отдельно стоит `O(JPdm+JPm²+Jm³)` при m≤P; U занимает `8JPd`.
При k итерациях linear solve главная цена `O(k(JPd+Jmd))`, дополненная
trust screening, local refits и gauges. Dense `(JP,md)` допустим в HYBRID
только под существующим bounded gate; общий dense Hessian `(md)²` не строится.

HYBRID для индексов делегирует HPAO в `LSMR.py`. Несмотря на обобщённое
описание старой routing note, текущий `HYBRID.solve` принимает **только
multi-index matrix**; single отвергается в LSMR validation.
Manifold HYBRID в том же файле решает иную задачу с graph penalty,
block-PCG и fallback LSMR. Выводы данного profile к нему не переносятся.

Измерения: Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1, CPU float64,
OpenBLAS 1 thread; две фактические BLAS-библиотеки и source hashes в raw.
Шесть hash-verified прежних frozen snapshots, три seed на d10 и d100,
`J=500,P=40,m=2`; дополнительно synthetic `J=100,P=20,d=1000,m=2/10`.
Это диагностика реализации, не новые independent selection datasets.
Каждая пара kernels/трёхшаговых trajectories имеет warmup, 9 повторов с
чередованием порядка; tracemalloc отдельно от wall. `cProfile` используется
для маршрутизации времени; его числа не входят в wall ratios.

**Профиль:** на d10 cached dense SVD — основной расход, `0.021/0.041 s`
в representative profile; сборка design `0.005 s`, stationarity `0.003 s`.
На d100 основное время — matrix-free A/A* и trust screening. В pilot
representative profile A/A* own time `0.087/0.126 s`, около 69%; эти actions
уже используют `matmul`, не `einsum`. Final profiles каждой задачи в raw.
Повторный SVD при смене C необходим: joint design меняется. SVD уже
кешируется между λ-пробами при фиксированном C; переносить его через AO
без обновления нельзя.

| Проверенная замена | d10 candidate/base median по трём seed | d100 candidate/base median по трём seed | Verdict |
|---|---:|---:|---|
| A/A* matmul → прямой einsum без planner | 1.719 | 1.360 | На измеренных inputs medians хуже; текущие BLAS actions сохранить. |
| Stationarity через matmul и перестановку local-gradient action | 0.739 | 0.868 | Kernel быстрее; полная трёхшаговая HPAO с двумя substitutions даёт 0.961 и 0.998. Нет убедительного общего speedup. |
| Design broadcast temporary → multiply(out=design_view) | 1.129 | не применяется, backend matrix-free | −31.5% traced allocation, но kernel медленнее. |

Полные d100 trajectory ratios `0.998,1.092,0.977`: сохранены также
замедления; не выбираем только выигрышный seed. Synthetic stationarity
ratios для d1000,m2/10: `0.735/0.887`. Это лишь kernel.
На всех шести обычных snapshots совпали index/coefficients, число
accepted/rejected trials и linear iterations в проверенной трёхшаговой
trajectory. Эти comparisons не устанавливают долгосрочную recovery.

Почему design out может быть медленнее: запись идет в strided view
столбцовых блоков dense design, тогда как прежний broadcast product сначала
пишет contiguous temporary. Это объяснение по layout, а не отдельное
измерение hardware counters. Allocation `4,869,664→3,333,664` bytes на d10;
выигрыш памяти не следует выдавать за выигрыш времени.

**Численный counterexample stationarity:** точное тождество
`(U_j B^T)^T r_j=B U_j^T r_j` меняет association сумм. После local LS
градиент почти нулевой, mass усиливает rounding. Стресс с mass до `1e8`
дал normalized local scores `1.7100e-8` и `1.8186e-8` (разница `1.09e-9`).
Порог между ними изменит локальную certificate-проверку. Error укладывается
в scale-dependent floating-point bound, но одинаковая trajectory при
любом tol не доказана. Кандидат остаётся изолированным; production
tolerances не ослаблялись. Первые строгие comparison tests выявили эту
разницу; текущий стресс-тест явно проверяет её как контрпример.

Другие конкретные расходы, проверенные по live-коду и профилю:

1. `RidgeWorkspace.__init__` заново вычисляет `Σ_p U_jpd²` на каждом AO
   шаге, хотя U неизменен внутри solve. Его можно кешировать один раз;
   нужно `8Jd` bytes (4 MB при J=500,d=1000) и invalidation по U.
   Это точное исключение повторного `O(JPd)` прохода. Constructor включает
   и прочую работу, поэтому время constructor не равно возможной экономии
   этого кеша; end-to-end gain отдельно не измерен. В код не внесён.
2. `rejects_trust` оплачивает до 32 A/A* actions для построения basis и
   дополнительные actions для проверки bounds. Это не бесплатный reuse,
   но удалять исходную проверку lower bound нельзя. В representative d100
   profile screening cumtime порядка 0.041/0.126 s (перекрывается с A/A*).
3. `_local_refit` имеет batched SVD и дополнительный `lstsq` только на
   rank-sensitive центрах. Вырожденные центры реальны; удалить fallback
   для ускорения значит изменить minimum-norm/rank contract.
4. Global/local stationarity повторно проецирует U; projected norm нужна
   существующему certificate scale. Одной замены local numerator на
   `B U_j^T r_j` недостаточно для исключения projection во всей routine.
   Можно было бы передавать norms из local refit, но нужно отдельное
   измерение общей цены и сохранение rounding/cutoff.
5. `einsum(optimize=True)` внутри stationarity повторно вызывает planner.
   Он виден в cProfile, но не является главным расходом всего solver.
   В NumPy 2.5 многие contractions выполняются `bmm_einsum`; старые общие
   утверждения «einsum всегда медленнее matmul» к этим shapes не применимы.
   [NumPy einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
   допускает reuse contraction path;
   [matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
   использует BLAS, когда применимо. Здесь решение определяется измерением.
6. `_local` уже reused в RidgeWorkspace и не возвращается напрямую из
   actions: результат не должен alias следующий вызов. Добавление output
   reuse без проверки ownership сломает Krylov recurrence.

Итак, скрытые расходы есть, но они различны: repeated reductions и
certificate projections; bounded-design temporary; trust-probe matvecs;
малые Python/planner costs. Простая глобальная замена einsum не решает
основную цену. [LSMR](https://web.stanford.edu/group/SOL/software/lsmr/)
поддерживает operator actions; скорость зависит и от их числа, и от цены.

## 8. Воспроизведение и завершение

Final raw: [`audit_final.json`](audit_final.json), содержит paired samples,
allocations, profiles, hashes, old/new correction certificates и environment.
Pilot сохранён отдельно в `audit.json`; после изменения audit-script
его hash закономерно отличается, final hash соответствует текущему скрипту.

```bash
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --no-sync python -m benchmarks.beta_functional_audit \
  --output /tmp/beta-audit-new.json --repeats 9
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --no-sync python -m pytest tests/test_beta_functional_audit.py \
  tests/test_hybrid.py tests/test_hybrid_optimization.py \
  tests/test_hybrid_recycling.py tests/test_lsmr.py -q
```

Новые reference tests: 8 passed; совокупная проверка: **70 passed, 1 failed**.
Сбой `test_manifold_hybrid_fit_matches_cg` происходит при создании
`ADP_Manifold(2)` с нынешним default `local_quadratic`, который разрешает
лишь m=1. Он уже описан в предыдущем solver completion; новых правок
manifold/production в этой задаче нет. Сбой сохраняется в учёте.
Ruff и Pyright новых Python-файлов прошли; formatting/diff checks
зафиксированы в завершённом PLAN/STATE.

Доказанные результаты (4)–(13) не являются теоремой о глобальной
минимальности или восстановлении EDR полного adaptive fit. Полное время fit,
trace_score/recovery, новые held-out seeds и GPU здесь не измерялись.
Подходящий математический следующий кандидат — QR-сжатие при P>d+1
с сохранением исходных cutoff/normalizations; для d100/P40 нужно иное
основание. В данном bounded цикле нового универсального solver нет,
поэтому запрос закрыт формализацией, категориями, конкретным точным
сокращением и воспроизводимым HYBRID-аудитом.
