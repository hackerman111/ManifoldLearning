Если обобщить ADP, то нужный тебе шаблон выглядит так:

\[
\text{cheap importance statistic } s_i
\quad\longrightarrow\quad
\text{оставить top-}k\text{ / значения выше }\tau
\quad\longrightarrow\quad
\text{дорогой алгоритм только на них}.
\]

Для твоей задачи я бы особенно смотрел на **OPG/DGSM, leverage scores, sensitivity/coresets и query-aware sparse selection**. Они ближе всего к идее «сначала вычислить полезность, потом выбросить большую часть».

## 1. Методы, наиболее близкие к Average Derivative

Пусть

\[
m(x)=\mathbb E[Y\mid X=x].
\]

Для single-index

\[
m(x)=g(\beta^\top x)
\]

ADP использует

\[
\mathbb E[\nabla m(X)]
=

\beta\,\mathbb E[g'(\beta^\top X)].
\]

Проблема очевидна: если \(g'\) меняет знак, всё может сократиться. Например \(g(z)=z^2\) и симметричный \(X\):

\[
\mathbb E[g'(\beta^\top X)]
=

2\mathbb E[\beta^\top X]=0.
\]

### OPG: Outer Product of Gradients

Вместо среднего градиента берём

\[
C
=

\mathbb E[
\nabla m(X)\nabla m(X)^\top
].
\]

Для single-index:

\[
C
=

\beta\beta^\top
\mathbb E[g'(\beta^\top X)^2].
\]

Для multi-index \(m(x)=g(B^\top x)\):

\[
C
=

B\,
\mathbb E[
\nabla g(B^\top X)\nabla g(B^\top X)^\top
]B^\top.
\]

Поэтому можно сделать eigendecomposition

\[
C=U\Lambda U^\top
\]

и **выкинуть направления с маленькими \(\lambda_i\)**.

Это почти буквальная версия нужной тебе идеи: посчитать один оператор, посмотреть его спектр, дальше работать только в активном подпространстве. Современные SDR-работы всё ещё используют OPG как базовый объект. :chatgpt-content-reference{index="0"}

Особенно интересно посмотреть свежую работу **Pautrel & Portier, “Riemannian Stochastic Optimization for Sufficient Dimension Reduction”, 2026**. Они связывают MAVE и OPG и используют sparse nearest-neighbour localization плюс оптимизацию на Stiefel/Grassmann manifold, чтобы избежать дорогого полного локального расчёта. :chatgpt-content-reference{index="1"}

### DGSM: Derivative-Based Global Sensitivity Measures

Если тебе надо выкидывать именно **координаты**, а не направления, бери диагональ OPG:

\[
\nu_j
=

\mathbb E
\left[
\left(
\frac{\partial f(X)}{\partial x_j}
\right)^2
\right].
\]

То есть

\[
\nu_j=C_{jj}.
\]

Если \(\nu_j\approx0\), функция почти не меняется вдоль \(x_j\).

У этого подхода есть приятная теория. Например для независимых uniform variables Sobol и Kucherenko получают оценку вида

\[
S_j^{\mathrm{tot}}
\le
\frac{\nu_j}{\pi^2\operatorname{Var}(f)},
\]

с поправкой масштаба области. Поэтому маленький derivative score даёт основание отбрасывать переменную как имеющую малый total effect. :chatgpt-content-reference{index="2"}

Для твоей идеи это один из самых чистых вариантов:

\[
s_j
=

\frac1n\sum_i
\left(\partial_j \hat f(x_i)\right)^2,
\qquad
S=\{j:s_j>\tau\}.
\]

После первого прохода алгоритм работает только на \(X_S\).

### Principal Hessian Directions

Можно перейти на второй порядок:

\[
H(x)=\nabla^2m(x).
\]

Для

\[
m(x)=g(B^\top x)
\]

имеем

\[
H(x)
=

B\nabla^2g(B^\top x)B^\top.
\]

Поэтому собственные направления усреднённого Hessian-подобного оператора снова лежат в нужном subspace.

Это полезно как раз там, где ADP умирает из-за симметрии. Principal Hessian Directions использует эту идею вместе с Stein identity. :chatgpt-content-reference{index="3"}

Я бы рассматривал гибрид:

\[
C
=

\alpha
\mathbb E[\nabla f\nabla f^\top] +
(1-\alpha)
\mathbb E[H_f^2].
\]

Это уже не стандартный ADP, но как research-направление естественно: первый член ловит slope, второй curvature.

## 2. Другие методы вида «посчитать статистику и удалить лишнее»

| Метод                   | Что считаем                                | Что потом выбрасываем                |
| ----------------------- | ------------------------------------------ | ------------------------------------ |
| **SIR**                 | \(\operatorname{Var}(\mathbb E[X\mid Y])\) | направления с маленькими eigenvalues |
| **SAVE**                | изменения \(\operatorname{Var}(X\mid Y)\)  | слабые направления                   |
| **PHD**                 | second-order moment / Hessian              | направления с малой curvature        |
| **OPG**                 | \(E[\nabla m\nabla m^\top]\)               | слабые eigendirections               |
| **DGSM**                | \(E[(\partial_jf)^2]\)                     | слабые coordinates                   |
| **SIS**                 | marginal correlation/utility               | признаки с маленьким score           |
| **Sparse MAVE**         | MAVE + \(L_1\)                             | целые predictors                     |
| **Ridge leverage**      | statistical leverage                       | datapoints/kernel columns            |
| **Coreset sensitivity** | worst-case contribution                    | datapoints                           |

SIR получает подпространство из inverse regression \(X\mid Y\), не оценивая непосредственно \(m(x)\). :chatgpt-content-reference{index="4"} SAVE/PHD нужны, в частности, потому что first-order inverse moments могут не увидеть симметричные зависимости.

**Sparse MAVE** особенно близок к твоей постановке multi-index: MAVE оценивает центральное mean-subspace, а \(L_1\)-штраф одновременно зануляет неинформативные predictors. :chatgpt-content-reference{index="5"}

**Sure Independence Screening** ещё проще:

\[
s_j
=

|\operatorname{Corr}(X_j,Y)|.
\]

Оставляем, например, \(O(n/\log n)\) лучших признаков, а уже потом запускаем дорогой метод. Fan и Lv доказывают sure-screening property при соответствующих предположениях. Главный недостаток: marginally useless variable может быть jointly essential, поэтому есть iterative SIS. :chatgpt-content-reference{index="6"}

## 3. Очень интересное направление для твоего kernel/ADP случая: leverage scores

Если дорогая часть возникает из-за большого количества observations или kernel centers, можно выбирать не признаки, а **столбцы kernel matrix**.

Для матрицы \(A\) leverage score строки:

\[
\ell_i
=

a_i^\top(A^\top A)^\dagger a_i.
\]

В ridge-варианте примерно:

\[
\ell_i^\lambda
=

a_i^\top
(A^\top A+\lambda I)^{-1}
a_i.
\]

Большой score означает, что точку плохо представляют остальные точки.

В kernel methods это приводит к **ridge leverage score Nyström sampling**: оставить только наиболее информативные landmarks и строить kernel approximation по ним. El Alaoui & Mahoney связывают необходимое число выбранных столбцов с effective dimension. :chatgpt-content-reference{index="7"}

Musco & Musco дают recursive algorithm с

\[
O(ns)
\]

kernel evaluations для \(s\) выбранных landmarks вместо полного \(n^2\). :chatgpt-content-reference{index="8"}

Если у тебя в ADP/MAVE-подобном методе локальный kernel сейчас использует почти все центры, это потенциально очень сильная идея:

\[
\text{approx leverage}
\rightarrow
J_{\text{active}}\ll n
\rightarrow
\text{только эти centers участвуют в следующих шагах}.
\]

Причём score можно пересчитывать после изменения projection \(B\), то есть сделать **adaptive landmark screening**. Для твоего варианта с большим числом kernel centers это, вероятно, интереснее обычного feature selection.

## 4. Coresets: ещё более общий принцип

Sensitivity точки определяется примерно как

\[
s_i
=

\sup_\theta
\frac{L_i(\theta)}
{\sum_j L_j(\theta)}.
\]

Точки с маленьким \(s_i\) почти никогда существенно не влияют на objective, поэтому sampling делают пропорционально \(s_i\).

Это очень близко философски к sparse attention: **оценить максимальный возможный вклад элемента и не вычислять остальные элементы полностью**. :chatgpt-content-reference{index="10"}

CRAIG использует ещё более подходящую для optimization версию. Он выбирает небольшой subset, чей weighted gradient аппроксимирует полный gradient:

\[
\sum_{i\in S}w_i\nabla L_i(\theta)
\approx
\sum_{i=1}^n\nabla L_i(\theta).
\]

То есть можно выбрасывать observations непосредственно по их полезности для текущего optimization step. :chatgpt-content-reference{index="11"}

Это наталкивает на вариант для ADP:

\[
s_i
=

\|\nabla_\beta L_i\|
\]

или лучше

\[
s_i
\approx
\left\|
H^{-1/2}\nabla_\beta L_i
\right\|^2.
\]

После этого считать дорогую local regression только для high-score observations.

---

## 5. Sparse attention, где реально выбирается «маленькая полезная часть»

Здесь я бы не особо смотрел на старые Longformer/BigBird. Там sparsity в основном задаётся структурно. Для твоей идеи интереснее **content-dependent selection**.

### H2O: Heavy-Hitter Oracle

Идея:

\[
s_i^{(t)}
=

s_i^{(t-1)} +
A_{t,i},
\]

где \(A_{t,i}\) это attention, полученный token \(i\).

Храним recent tokens плюс tokens с большим накопленным score. Авторы формулируют KV eviction как dynamic submodular problem. :chatgpt-content-reference{index="12"}

Это почти идеальный пример:

> дешёвая накопительная статистика → permanently discard low-score items.

### Scissorhands

Используется гипотеза **persistence of importance**: если token был полезен раньше, у него выше вероятность оставаться полезным позже.

Поэтому прошлый attention используется как estimator будущей importance. KV cache удалось уменьшать до \(5\times\) в экспериментах авторов без заметного ухудшения их метрик. :chatgpt-content-reference{index="13"}

Это интересная идея и вне transformers:

\[
s_i^{t+1}
=

\rho s_i^t+(1-\rho)\hat s_i^t.
\]

Не пересчитывать importance с нуля на каждой итерации.

### Quest

На мой взгляд, это одна из самых интересных тебе работ.

Attention:

\[
q^\top k_i.
\]

Но считать его для всех \(i\) дорого. Quest группирует keys в pages и хранит coordinate-wise min/max. По query можно **дёшево оценить upper bound потенциального attention** страницы.

Только top-\(k\) перспективных pages загружаются и обсчитываются точно. :chatgpt-content-reference{index="14"}

То есть:

\[
\boxed{
\text{cheap upper bound}
\rightarrow
\text{prune}
\rightarrow
\text{exact expensive calculation}
}
\]

Это, пожалуй, самый прямой ответ на твоё «посчитать что-то, чтобы выкинуть ненужную часть».

Для твоего kernel/local-regression случая аналог:

\[
\widehat U_j(x)
\ge
\text{possible contribution of center }j
\]

и если

\[
\widehat U_j(x)<\tau,
\]

вообще не считать точный kernel/weight для \(j\).

### SnapKV

Берётся маленькое observation window в конце prompt. Attention из него используется как **pilot sample**, предсказывающий, какие старые tokens потребуются во время дальнейшей генерации. После этого KV cache заранее сокращается. :chatgpt-content-reference{index="15"}

Общая идея:

\[
\text{маленький pilot computation}
\rightarrow
\hat s_i
\rightarrow
\text{полный computation только на top-k}.
\]

Для статистического метода можно делать буквально то же самое: оценить importance на subsample \(m\ll n\), затем полный solver только на выбранном subset.

### SparseK

Очень подходящая paper:

**Lou et al., 2024, “Sparser is Faster and Less is More”**.

Отдельная scoring network вычисляет importance, затем differentiable top-\(k\) operator оставляет фиксированное число KV. :chatgpt-content-reference{index="16"}

Схема:

\[
u_i=h_\phi(x_i),
\]

\[
m=\operatorname{SparseK}(u,k),
\]

\[
\text{дорогая функция только для }\operatorname{supp}(m).
\]

Это уже общий differentiable-selection layer, который можно перенести далеко за transformers.

### Native Sparse Attention

DeepSeek NSA, ACL 2025, сейчас один из наиболее интересных вариантов архитектурно. Он сочетает:

\[
\text{compression} +
\text{dynamic selection} +
\text{local sliding window}.
\]

Сначала coarse representation даёт глобальное представление, затем selection выбирает важные blocks, плюс сохраняется локальная область. :chatgpt-content-reference{index="17"}

Для твоей задачи эту архитектуру можно перевести буквально как

\[
\boxed{
\text{coarse global approximation} +
\text{exact top-k interactions} +
\text{always-keep local neighbours}
}
\]

и это, на мой взгляд, очень плодотворный шаблон.

## Что я бы пробовал именно как research ideas

Самая интересная линия выглядит так:

\[
\boxed{
\text{ADP/OPG score}
\rightarrow
\text{screen directions/features}
\rightarrow
\text{leverage/Quest-like score}
\rightarrow
\text{screen observations/centers}
\rightarrow
\text{exact local optimization}
}
\]

То есть sparsity на **двух осях**.

Сначала:

\[
C=
\frac1n\sum_i
\nabla \hat m(x_i)\nabla\hat m(x_i)^\top
\]

и оставляем только eigenspace \(r\ll d\).

Потом внутри этого пространства для каждого query \(x\) не рассматриваем все \(n\) observations, а строим дешёвый proxy

\[
s_j(x)
\approx
\text{potential contribution of }x_j
\]

и считаем точные kernel weights/local regressions только для

\[
j\in\operatorname{TopK}(s(x)).
\]

Ещё сильнее вариант с безопасным pruning:

\[
s_j^{\rm upper}(x)
<
\text{current }k\text{-th best lower bound}
\quad\Rightarrow\quad
j\text{ можно отбросить}.
\]

Это уже аналог **branch-and-bound / Quest**, где при корректном bound отбрасывание не эвристическое.

Из статей я бы начал с: **OPG/MAVE → DGSM → Ridge Leverage Nyström → Quest → SparseK → Native Sparse Attention**. Они дают шесть разных математических механизмов одной и той же идеи: gradient energy, conditional variance, statistical leverage, upper bounds, learned top-\(k\), hierarchical coarse-to-fine selection. :chatgpt-content-reference{index="18"}

Единственного «лучшего» солвера для СЛУ \(Ax=b\) нет. Если выбирать по типу матрицы, то на 2026 год ориентир такой:

| Матрица                                    | Что обычно брать                    |
| ------------------------------------------ | ----------------------------------- |
| Плотная, общего вида                       | **LAPACK `dgesv` / LU**             |
| Плотная SPD                                | **Cholesky `dposv`**                |
| Разреженная, общего вида, нужна надёжность | **PARDISO**                         |
| Разреженная SPD                            | **PARDISO** или CHOLMOD             |
| Очень большая разреженная                  | **Krylov + хороший preconditioner** |
| PDE, эллиптические задачи                  | **CG + AMG**                        |
| Большая несимметричная                     | **GMRES + AMG/ILU**                 |
| GPU, плотная                               | **cuSOLVER**                        |
| GPU, огромная sparse                       | **PETSc + AMGx / hypre**            |

Если под «лучшим» понимать **универсальный быстрый direct solver для sparse-матриц на одном многопоточном CPU**, я бы сейчас в первую очередь смотрел на **Intel oneMKL PARDISO**. Он поддерживает \(LU\), \(LDL^T\), \(LL^T\), несколько правых частей и параллельную факторизацию. Intel продолжала заметно оптимизировать его в oneMKL 2025–2026. :chatgpt-content-reference{index="0"}

Но для действительно больших задач direct solver часто проигрывает из-за fill-in. Например, исходная матрица может иметь \(10^7\) ненулевых элементов, а факторы \(L,U\) после исключения могут потребовать на порядок больше памяти. Тогда обычно используют

\[
M^{-1}Ax=M^{-1}b
\]

и Krylov-метод:

- SPD: **CG + AMG**
- symmetric indefinite: **MINRES + подходящий preconditioner**
- nonsymmetric: **GMRES / FGMRES + AMG или ILU**

Для PDE это часто быстрее асимптотически: хороший multigrid может приближаться к \(O(\operatorname{nnz}(A))\). PETSc прямо рекомендует multigrid для многих эллиптических задач и предоставляет `GAMG`, `hypre` и GPU-ориентированный `AMGx`. :chatgpt-content-reference{index="1"}

Если вопрос именно про **алгоритм**, а не библиотеку, то правило ещё проще:

\[
\boxed{\text{dense} \rightarrow LU/Cholesky,\qquad
\text{huge sparse} \rightarrow Krylov + preconditioner}
\]

Причём для SPD:

\[
\boxed{\text{CG + AMG}}
\]

можно считать одним из наиболее сильных общих вариантов.

Если ты спрашиваешь в контексте твоего **ADP / ridge шага с большими \(n,d,J\)**, там ответ будет другим: я могу отдельно сравнить **LSMR, LSQR, CG/CGNR, Cholesky, QR и matrix-free методы** именно для твоей матрицы и сказать, что, вероятно, будет быстрее LSMR.
