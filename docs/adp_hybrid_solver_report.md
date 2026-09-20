# HPAO-LSMR: эффективный солвер для ADP single-index и multi-index

## Результат

Для альтернированной оптимизации из `multiindex(3).tex` рекомендуется солвер **HPAO-LSMR**:

> **Hybrid Proximal Alternating Optimization + matrix-free LSMR**, с точными локальными least-squares шагами, глобальным proximal-шагом в виде линейного оператора, компенсацией калибровки после QR/SVD и автоматическим выбором между двумя представлениями статистик: разреженным факторизованным `QH` и плотным `U`.

Солвер не строит:

- глобальную design matrix размера `(Jp) x (dm)`;
- нормальную матрицу размера `d x d` или `(dm) x (dm)`;
- тензор разностей размера `n x J x d`;
- набор локальных ковариаций размера `J x d x d`.

При `n=J=10000`, `d=1000`, `m=5`, `p=10`, FP64:

- разреженный режим с `k=10` занимает около 1 GB вместе с кэшем всех попарных расстояний;
- точный плотный fallback занимает менее 2 GB плюс рабочий tile;
- явная multi-index design matrix потребовала бы 4 GB и потому не создаётся;
- тензор разностей потребовал бы 800 GB и категорически запрещён;
- один Krylov-проход использует либо `Theta(ndm + E(p+m))` операций в режиме `QH`, либо `Theta(Jpd + Jdm)` в режиме `U`, где `E=sum_j k_j`.

Для основной практической цели `n=J=10000`, `d=100` тот же алгоритм укладывается примерно в 0.9 GB при кэшировании полной матрицы расстояний. Если 800 MB под расстояния нежелательны, они вычисляются tile-by-tile, и пик памяти резко уменьшается ценой повторных GEMM.

Ниже дано точное определение задачи, алгоритм, доказательство корректности и монотонности, критерии численной устойчивости, ресурсные оценки и stress-test.

## 1. Фиксация математической спецификации

В приложенных версиях есть несовместимые нормировки и две разные формулы анизотропного тензора. Поэтому сначала фиксируется ветка из наиболее нового файла `multiindex(3).tex`.

Пусть:

- `X in R^(n x d)` и `Y in R^n`;
- `j=1,...,J` нумерует центры;
- `p` есть число направлений в локальной ADP-статистике;
- `I_j in R^p`, `U_j in R^(p x d)`;
- `c_j=n_eff,j=sum_i w_ij > 0`;
- `m` есть размерность multi-index пространства;
- `P in R^(m x d)`, `P P^T=I_m`;
- `ell_j in R^m` есть локальный multi-index slope.

### 1.1. Единая задача

Обе задачи являются частными случаями

\[
F(P,\ell)
=\frac12\sum_{j=1}^J c_j
\left\|I_j-U_jP^T\ell_j\right\|_2^2,
\qquad PP^T=I_m.
\tag{1}
\]

Для single-index берётся `m=1`, `P=beta^T`, `ell_j=f_j` и `||beta||=1`:

\[
F(\beta,f)
=\frac12\sum_{j=1}^J c_j
\left\|I_j-f_jU_j\beta\right\|_2^2.
\tag{2}
\]

В `manifold_v2.tex` встречается версия без внешнего множителя `c_j`, а также версия с nuisance-intercept. Один код может поддерживать их через параметр `gamma_j` и опциональный nuisance-столбец, но внутри одного запуска нормировки смешивать нельзя. В этом отчёте `gamma_j=c_j`.

Для свободного глобального шага определим продолжение того же data-term на все `B in R^(m x d)`:

\[
\widetilde F(B,\ell)
=\frac12\sum_jc_j\|I_j-U_jB^T\ell_j\|^2,
\qquad
F(P,\ell)=\widetilde F(P,\ell)\quad\text{при }PP^T=I_m.
\tag{2a}
\]

### 1.2. Точная ветка анизотропии

Для single-index в `multiindex(3).tex`

\[
T^2=h^{-2}\{\alpha^2(I-\beta\beta^T)+\beta\beta^T\}.
\]

Если `Delta=x_i-x_j`, `D=||Delta||^2`, `s=beta^T Delta`, то

\[
q=\|T\Delta\|^2
=\frac{\alpha^2(D-s^2)+s^2}{h^2}
=\frac{\alpha^2D+(1-\alpha^2)s^2}{h^2}.
\tag{3}
\]

Формула из `manifold_v2.tex`, `T^2=h^{-2}(alpha^2 I+beta beta^T)`, даёт другое значение `q=(alpha^2D+s^2)/h^2`. Это другой estimator, не только другая реализация.

Для multi-index:

\[
T^2=h^{-2}\{\alpha^2(I-P^TP)+P^T\Lambda P\},
\]

\[
q=\frac{\alpha^2(D-\|u\|^2)+u^T\Lambda u}{h^2},
\qquad u=P\Delta.
\tag{4}
\]

В файле ядро записано как `K(t)=(1-t^2)_+`, а аргументом служит `t=q=||T Delta||^2`. Следовательно, буквальная формула веса

\[
w_{ij}=(1-q_{ij}^2)_+,
\tag{5}
\]

а не `(1-q)_+`. Обе функции имеют одну support, но разные значения весов и дают разные статистики.

## 2. Реестр рассмотренных семейств солверов

| Семейство | Сильная сторона | Блокирующий недостаток | Роль в итоговом солвере |
|---|---|---|---|
| Явные нормальные уравнения из TeX | Простая закрытая формула | `O(d^2)` или `O(d^2m^2)` память, квадрат числа обусловленности | Не использовать |
| Cholesky / PCG для нормальных уравнений | Быстро при отличном preconditioner | Теряется устойчивость, preconditioner меняется вместе с slopes | Только контролируемый fallback |
| Прямой QR / TSQR | Численно устойчив | Для multi-index слишком дорог и требует явной матрицы | Fallback для single-index при `d<=100..256` |
| LSQR / LSMR | Matrix-free, только `A` и `A*` | Нужны точный adjoint и контроль inner tolerance | **Основной глобальный шаг: LSMR** |
| Kaczmarz / coordinate descent | Дешёвые отдельные строки | Слаб для несовместной ridge-задачи и плотного coupling | Не основной |
| Sketch solve | Меньше строк | Меняет исходную задачу | Только sketch-preconditioner с проверкой |
| Nyström / randomized SVD | Может резко сократить Krylov-итерации | Setup окупается не всегда, оператор меняется на AO | Условный preconditioner после pilot-run |
| Variable projection + Riemannian L-BFGS | Убирает локальные slopes из внешних переменных | Плох при смене численного ранга локальных задач, нет универсального выигрыша | Опциональный polish после 1-2 ALS sweeps |
| Свободная факторизация и gauge-fix | Точно сохраняет predictions | Требует правильного преобразования slopes | **Основа монотонного AO** |
| Sparse `QH` statistics | Очень малая память, fused matvec | При плотных окрестностях становится дороже `U` | **Основной sparse backend** |
| Явные `U_j` | Очень быстрый matvec | Память `O(Jpd)` | **Dense / many-Krylov backend** |

LSMR выбран вместо формирования `A^T A`: Golub-Kahan bidiagonalization работает с линейным оператором, а норма нормального residual `||A^T r||` у LSMR монотонна. Это соответствует исходной работе [Fong and Saunders](https://epubs.siam.org/doi/10.1137/10079687X). Параметр `damp` в SciPy входит как квадрат в штраф, что прямо указано в [официальной документации SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html).

## 3. Точная факторизация ADP-статистик

Пусть `N_j` есть support весов центра `j`, `k_j=|N_j|`,

\[
a_{ji}=\frac{w_{ij}}{c_j},\qquad
\bar x_j=\sum_{i\in N_j}a_{ji}x_i,\qquad
\bar y_j=\sum_{i\in N_j}a_{ji}y_i.
\]

Определим

\[
H_j=\operatorname{diag}(\sqrt{a_j})
\left(X_{N_j}-{\bf1}\bar x_j^T\right)
\in\mathbb R^{k_j\times d},
\tag{6}
\]

\[
g_j=\sqrt{a_j}\odot(Y_{N_j}-\bar y_j{\bf1})
\in\mathbb R^{k_j}.
\tag{7}
\]

Если строки `Phi_j in R^(p x d)` содержат направления ADP, то

\[
Q_j=\Phi_jH_j^T\in\mathbb R^{p\times k_j},
\qquad
U_j=Q_jH_j,
\qquad
I_j=Q_jg_j.
\tag{8}
\]

Это тождество, а не аппроксимация. Оно даёт два взаимозаменяемых backend:

1. **QH backend:** хранить `Q_j`, weights и индексы, применять `H_j` неявно.
2. **U backend:** один раз материализовать `U_j` и `I_j`, затем удалить `Q_j` и локальные workspaces.

Операторы `H_j` не требуют хранить `H_j`:

\[
H_jv=\sqrt{a_j}\odot
\left((Xv)_{N_j}-{\bf1}\,a_j^T(Xv)_{N_j}\right),
\tag{9}
\]

\[
H_j^Tt=X_{N_j}^T
\left[\sqrt{a_j}\odot t-a_j{\bf1}^T(\sqrt{a_j}\odot t)\right].
\tag{10}
\]

Центрирование даёт `rank(H_j)<=k_j-1`. Поэтому необходимое условие идентификации `m` slopes: `k_j>=m+1` и фактический численный ранг `U_jP^T` не меньше `m`.

## 4. Один внутренний HPAO-шаг

### 4.1. Локальный exact solve

При фиксированном `P_t` определим

\[
C_j=U_jP_t^T\in\mathbb R^{p\times m}.
\]

Точное minimum-norm решение:

\[
\ell_j^{+}=C_j^\dagger I_j.
\tag{11}
\]

Поскольку `m` мало, используется pivoted QR при полном ранге и SVD при сомнительном ранге. Матрица `C_j^TC_j` не обращается.

Теорема ниже использует точный algebraic rank. В floating point effective rank определяется backward-error порогом, например

\[
\tau_j=\max(p,m)\,u_{64}\,\sigma_{\max}(C_j)
+\tau_{data,j},
\]

где `tau_data,j` отражает известную погрешность statistics. Отсечение ненулевого singular value означает решение ближайшей rank-reduced локальной задачи; порог и perturbation certificate сохраняются в diagnostics.

Single-index имеет специальную формулу:

\[
u_j=U_j\beta_t,
\qquad
f_j^+=
\begin{cases}
\dfrac{I_j^Tu_j}{u_j^Tu_j},&u_j\ne0,\\
0,&u_j=0.
\end{cases}
\tag{12}
\]

Численно малое, но ненулевое singular value не следует молча заменять ridge. Солвер сохраняет центр, применяет minimum-norm pseudoinverse в объявленном численном ранге и помечает rank deficiency. Исключение центра меняет `c_j`, а уменьшение `m` меняет модель; оба действия допустимы только как отдельный model-selection restart, не внутри того же монотонного run. Локальный ridge также допустим только как явно объявленное изменение objective.

### 4.2. Глобальный proximal solve

При фиксированных `ell_j^+` свободная матрица `B in R^(m x d)` находится около текущего `P_t`:

\[
\min_B\;
\frac12\sum_j c_j
\|I_j-U_jB^T\ell_j^+\|^2
+\frac{\lambda_t}{2}\|B-P_t\|_F^2.
\tag{13}
\]

Решается correction `Delta=B-P_t`. Зададим

\[
r_j=\sqrt{c_j}\left(I_j-U_jP_t^T\ell_j^+\right),
\tag{14}
\]

\[
(\mathcal A_t\Delta)_j
=\sqrt{c_j}\,U_j\Delta^T\ell_j^+.
\tag{15}
\]

Тогда

\[
\Delta_t=\arg\min_\Delta
\frac12\|\mathcal A_t\Delta-r\|^2
+\frac{\lambda_t}{2}\|\Delta\|_F^2.
\tag{16}
\]

Сопряжённый оператор для `z=(z_1,...,z_J)`:

\[
\mathcal A_t^*z
=\sum_j\sqrt{c_j}\,
\ell_j^+\left(U_j^Tz_j\right)^T
\in\mathbb R^{m\times d}.
\tag{17}
\]

Решается augmented least-squares система

\[
\begin{bmatrix}\mathcal A_t\\\sqrt{\lambda_t}I\end{bmatrix}
\Delta
\simeq
\begin{bmatrix}r\\0\end{bmatrix}.
\tag{18}
\]

Для SciPy `lsmr` нужно передавать `damp=sqrt(lambda_t)`, не `lambda_t`. Нельзя решать исходную систему для `B` с правой частью `I`: штраф центрирован в `P_t`, поэтому correction formulation обязательна.

Если используется right scaling `Delta=D_s y`, стандартный `damp` уже задаёт неправильный штраф. Тогда оператор должен быть явно augmented:

\[
y\mapsto
\begin{bmatrix}\mathcal A_tD_s y\\\sqrt{\lambda_t}D_s y\end{bmatrix}.
\tag{19}
\]

После такого solve normal residual обязательно пересчитывается в исходных координатах `Delta`:

\[
q=A^*r-(A^*A+\lambda I)\Delta.
\tag{19a}
\]

Малость residual в `y`-координатах контролирует `D_s^Tq` и при плохо обусловленном `D_s` не сертифицирует descent исходной задачи.

### 4.3. Gauge-fix без изменения fitted values

После решения `B_hat=P_t+Delta_t` нельзя просто нормировать `B_hat` и оставить slopes прежними.

Для single-index, если `r_beta=||beta_hat||`, выполняется

\[
\beta_{t+1}=\widehat\beta/r_\beta,
\qquad
f_{j,t+1}=r_\beta f_j^+.
\tag{20}
\]

Тогда `f_new U beta_new=f_old U beta_hat` точно.

Если `beta_hat=0`, деление не выполняется: берётся `beta_(t+1)=beta_t`, все gauge-компенсированные slopes полагаются нулевыми, затем выполняется обычный локальный refit. Это точно представляет нулевые predictions свободного шага и сохраняет descent.

Для multi-index используется thin QR

\[
\widehat B^T=QR,
\qquad
P_{t+1}=Q^T,
\qquad
\ell_{j,t+1}=R\ell_j^+.
\tag{21}
\]

Тогда

\[
P_{t+1}^T\ell_{j,t+1}
=Q R\ell_j^+
=\widehat B^T\ell_j^+.
\]

SVD-вариант `B_hat=L Sigma V^T` даёт `P=V^T`, `ell_j=Sigma L^T ell_j^+`.

Базис затем выравнивается с предыдущим `P_t` ортогональным Procrustes-преобразованием. Если

\[
P_tP_{t+1}^T=U_o\Sigma_oV_o^T,
\qquad O=U_oV_o^T,
\]

то одновременно выполняются

\[
P_{t+1}\leftarrow OP_{t+1},
\qquad
\ell_{j,t+1}\leftarrow O\ell_{j,t+1}.
\tag{22}
\]

Predictions снова сохраняются точно. Это устраняет случайные повороты базиса между итерациями.

Если `B_hat` численно имеет ранг меньше `m`, QR/SVD может ортонормированно дополнить базис, а singular `R` всё равно точно сохранит fitted values. Часть дополненного пространства не идентифицирована, поэтому rank loss обязательно сообщается. Если он сохраняется на выходе, run останавливается как неидентифицированный; уменьшение `m` запускает новую задачу. Произвольные дополненные направления нельзя объявлять найденными EDR-направлениями.

## 5. Fused QH-оператор

Матрицу `U_j=Q_jH_j` не обязательно формировать. Для forward multi-index шага:

1. Вычислить один GEMM `Z=X Delta^T in R^(n x m)`.
2. Для каждого центра вычислить `zbar_j=sum_i a_ji Z_i`.
3. На ребре `(j,i)` вычислить

\[
v_{ji}=\sqrt{a_{ji}}\,\ell_j^T(Z_i-\bar z_j).
\]

4. Вернуть блок `sqrt(c_j) Q_j v_j`.

Для adjoint:

1. `t_j=Q_j^T z_j`.
2. `rho_j=sum_i sqrt(a_ji)t_ji`.
3. Scatter в `G in R^(n x m)`:

\[
G_i\mathrel{+}=
\sqrt{c_j}\left(\sqrt{a_{ji}}t_{ji}-a_{ji}\rho_j\right)\ell_j.
\tag{23}
\]

4. Вернуть `G^T X in R^(m x d)`.

Forward+adjoint имеют сложность

\[
\Theta(ndm+E(p+m)),
\qquad E=\sum_jk_j,
\tag{24}
\]

а не `Theta(Edm)`. Ключевой GEMM `X Delta^T` выполняется один раз на весь граф соседств.

Для single-index берётся `m=1`; стоимость пары операторов `Theta(nd+pE)`.

Явный `U` backend имеет стоимость пары

\[
\Theta(Jpd+Jdm)
\tag{25}
\]

и память `8Jpd` bytes в FP64. Он выгоден при плотных окрестностях или большом числе Krylov-проходов.

### 5.1. Автоматический выбор backend

Первичная граница по памяти:

\[
\operatorname{mem}(Q)\approx 8pE,
\qquad
\operatorname{mem}(U)\approx 8Jpd.
\]

Следовательно, среднее `kbar=E/J<d` обычно благоприятствует `QH`. Но окончательный выбор делается по измеренному времени:

Более точная per-center граница в FP64, если factor-center хранит `Q`, веса и `int32` индексы:

\[
8pk_j+8k_j+4k_j<8pd
\quad\Longleftrightarrow\quad
k_j<\frac{8pd}{8p+12}.
\tag{26a}
\]

Она близка к `k_j<d`, но учитывает реальный edge overhead.

\[
T_{QH}=T_{Q\text{-build}}+N_{pass}T_{QH\text{-pass}},
\]

\[
T_U=T_{Q\text{-build}}+T_{U\text{-materialize}}
+N_{pass}T_{U\text{-pass}}.
\tag{26}
\]

Перед первым полным solve запускаются 2-3 pilot-прохода. Допустим и per-center hybrid: sparse центры остаются `QH`, dense центры сразу аккумулируют `U_j,I_j`.

### 5.2. Layout и параллельное исполнение

Для высокой фактической пропускной способности, а не только хорошего big-O:

- `Q_j` хранятся одним packed contiguous массивом, support indices как `int32`, row offsets как `int64`, веса FP64;
- workspaces `Z in R^(n x m)`, `G in R^(n x m)` и edge buffers выделяются один раз и переиспользуются всеми LSMR-вызовами;
- `X Delta^T` и `G^T X` выполняются BLAS-3 GEMM; циклы по отдельным координатам `d` отсутствуют;
- малые `Q_j` contractions группируются по `(p,k_j)` и исполняются batched GEMV/GEMM;
- CPU adjoint использует thread-local `G` размером всего `n x m`, затем reduction; GPU использует edges, сгруппированные по target `i`, либо controlled atomics при малом `m`;
- центры обрабатываются блоками, чтобы packed `Q`, indices и edge scalars текущего блока помещались в last-level cache;
- каждый новый тип или layout оператора проходит один random adjoint test до первого LSMR solve;
- внутри Krylov-итераций запрещены динамические allocations и повторная генерация направлений.

При `centers=X` попарные расстояния симметричны. Их можно вычислять и хранить по треугольникам, почти вдвое сокращая cache-D, но downstream tile API должен уметь транспонированно читать второй треугольник без материализации полной копии.

## 6. Доказательство корректности внутреннего AO

### Теорема

Рассмотрим фиксированный внешний ADP-шаг, то есть `I_j`, `U_j`, `c_j` не меняются во время внутреннего AO. Пусть:

1. `c_j>0`;
2. локальные задачи (11) решаются minimum-norm pseudoinverse в постоянном фактическом ранге, а все half-step slopes ограничены; достаточное условие:

\[
\inf_{j,t}\sigma_{\min}^+(U_jP_t^T)\ge\sigma_{loc}>0;
\]

3. `0<lambda_min<=lambda_t<=lambda_max<infinity`;
4. после каждого gauge-fix выполняется точный локальный refit, поэтому сохраняемая пара `(P_t,ell_t)` локально block-optimal;
5. inner global solve либо является точным minimizer (16), либо удовлетворяет (33) с единой константой `theta_t<=theta_bar<1`.

Тогда HPAO-LSMR:

- порождает невозрастающую последовательность значений исходной data-loss `F`;
- имеет `sum_t ||Delta_t||_F^2<infinity` и потому `||Delta_t||_F -> 0`;
- каждая предельная точка является first-order stationary точкой constrained-задачи (1);
- QR/SVD и Procrustes шаги не изменяют ни один fitted value;
- `QH` и `U` backend дают один и тот же математический оператор.

### Доказательство

**Шаг 1. Эквивалентность статистик.** Из (8) непосредственно

\[
Q_jH_jv=U_jv,
\qquad Q_jg_j=I_j.
\]

Формулы (9) и (10) следуют раскрытием центрирования. Их скалярное произведение удовлетворяет

\[
\langle H_jv,t\rangle=\langle v,H_j^Tt\rangle.
\]

Подстановка (10) в (17) даёт scatter-формулу (23). Поэтому fused `QH` реализует тот же `A` и `A*`, что явные `U_j`.

**Шаг 2. Локальное убывание.** Для каждого `j` формула (11) есть точный minimizer least-squares блока. Поэтому

\[
F(P_t,\ell^+)\le F(P_t,\ell_t).
\tag{27}
\]

**Шаг 3. Глобальное убывание, exact-вариант.** Нулевой correction `Delta=0`, то есть `B=P_t`, допустим в (16). Оптимальность точного minimizer `Delta_t` даёт

\[
\widetilde F(\widehat B,\ell^+)
+\frac{\lambda_t}{2}\|\Delta_t\|_F^2
\le F(P_t,\ell^+).
\tag{28}
\]

Пусть `tilde ell_t` обозначает slopes сразу после QR/SVD и Procrustes. Gauge-fix удовлетворяет

\[
P_{t+1}^T\widetilde\ell_{j,t}=\widehat B^T\ell_j^+,
\]

а ортогональный Procrustes шаг сохраняет тот же product. Финальный локальный refit определяет `ell_(t+1)` и может только дополнительно уменьшить loss. Поэтому

\[
F(P_{t+1},\ell_{t+1})
\le F(P_{t+1},\widetilde\ell_t)
=\widetilde F(\widehat B,\ell^+)
\le F(P_t,\ell^+)-\frac{\lambda_t}{2}\|\Delta_t\|_F^2
\le F(P_t,\ell_t)-\frac{\lambda_t}{2}\|\Delta_t\|_F^2.
\tag{29}
\]

Это место исправляет ошибку простой нормировки из TeX: нормировка без преобразования slopes не сохраняет objective.

**Шаг 4. Квадратичная суммируемость proximal-шагов для exact solve.** Суммируем (29), учитывая `F>=0` и `lambda_t>=lambda_min`:

\[
\sum_{t=0}^\infty\|\Delta_t\|_F^2
\le\frac{2F(P_0,\ell_0)}{\lambda_{\min}}<\infty.
\tag{30}
\]

Значит `Delta_t -> 0`. Это не утверждает `sum_t||Delta_t||<infinity` и не доказывает сходимость всей последовательности. Поскольку `sigma_min(P_t)=1`, неравенство Вейля даёт

\[
\sigma_{\min}(\widehat B_t)\ge1-\|\Delta_t\|_2,
\]

поэтому `B_hat_t` автоматически имеет полный row rank для всех достаточно больших `t`. Polar map

\[
P_{pol}(\widehat B)
=(\widehat B\widehat B^T)^{-1/2}\widehat B
\]

непрерывен около row-Stiefel manifold, а Procrustes выбирает ближайшую к `P_t` ориентацию того же row space. Поэтому `P_(t+1)-P_t->0`.

**Шаг 5. Стационарность.** Нормальное уравнение точного глобального шага для свободного продолжения:

\[
\nabla_B\widetilde F(\widehat B,\ell^+)
+\lambda_t(\widehat B-P_t)=0.
\tag{31}
\]

Берём сходящуюся подпоследовательность сохраняемых `P_t`. Из постоянства локального ранга и нижней границы на ненулевые singular values minimum-norm map

\[
P\mapsto(U_jP^T)^\dagger I_j
\]

непрерывен. Поэтому `ell_t^+` и сохраняемые после refit `ell_t` имеют одни предельные значения. Кроме того, для gauge half-step существует `T_t` такое, что

\[
\widetilde\ell_{j,t}=T_t\ell_{j,t}^+,
\qquad
T_t=P_{t+1}\widehat B_t^T\longrightarrow I_m.
\tag{30a}
\]

Из ограниченности `lambda_t`, (30), непрерывности и точности локальных solves следует

\[
\nabla_B\widetilde F(P_*,\ell_*)=0,
\qquad
(U_jP_*^T)^T(I_j-U_jP_*^T\ell_{j,*})=0.
\tag{32}
\]

Первое равенство (32) уже даёт нулевой полный евклидов градиент `G=nabla_P F`, а значит и его Riemannian projection

\[
\operatorname{grad}F
=G-\operatorname{sym}(GP^T)P
=0.
\tag{32a}
\]

Локальные normal equations отдельно дают `nabla_(ell_j)F=0`. Они также влекут `GP^T=0`, но одного последнего равенства без `G=0` было бы недостаточно для stationarity. Предельная точка first-order stationary для исходной constrained `F`, а `lambda_t` остаётся proximal-параметром шага, не частью конечного statistical objective.

**Шаг 6. Inexact solve.** Пусть

\[
q_t=\mathcal A_t^*r-
(\mathcal A_t^*\mathcal A_t+\lambda_tI)\Delta_t.
\]

Если

\[
\|q_t\|\le\theta_t\lambda_t\|\Delta_t\|,
\qquad 0\le\theta_t<1,
\tag{33}
\]

то раскрытие квадрата даёт

\[
\widetilde F(\widehat B,\ell^+)-F(P_t,\ell^+)
\le-\frac12\|\mathcal A_t\Delta_t\|^2
-(1-\theta_t)\lambda_t\|\Delta_t\|^2.
\tag{34}
\]

Gauge-fix сохраняет левую data-loss, а финальный локальный refit её уменьшает, поэтому (34) переносится на полностью сохранённую пару `(P_(t+1),ell_(t+1))`. Следовательно, монотонность сохраняется. Если `theta_t<=theta_bar<1`, суммирование (34) даёт отдельную inexact-оценку

\[
\sum_t\|\Delta_t\|_F^2
\le
\frac{F(P_0,\ell_0)}{(1-\bar\theta)\lambda_{\min}},
\tag{34a}
\]

и потому `Delta_t->0`. Из (33) и верхней границы на `lambda_t` следует `q_t->0`. Data-gradient в новой свободной точке равен

\[
g_t=\mathcal A_t^*(\mathcal A_t\Delta_t-r)
=-(\lambda_t\Delta_t+q_t)\longrightarrow0.
\tag{34b}
\]

Поэтому доказательство предельной stationarity проходит и для сертифицированного inexact solve. Суммируемая абсолютная ошибка может заменить (33) в доказательстве предельной stationarity, но сама по себе не гарантирует поитерационную монотонность; accepted step всё равно должен удовлетворять (34) или прямой проверке `F_new<=F_old`. Обычных `atol/btol` недостаточно для сертификата: после LSMR явно вычисляется (33), при необходимости выполняется correction restart.

Теорема доказана.

### Область действия доказательства

Доказательство относится к внутреннему AO при фиксированных `I,U,c`. Внешний structural adaptation меняет веса, направления и сам objective. Его нельзя выдавать за descent одного фиксированного функционала. Каждый внешний шаг решается этим солвером до заданного inner certificate.

## 7. Построение весов и statistics без взрыва памяти

### 7.1. Расстояния

Никогда не строится `Delta_(i,j,:)`. Попарные квадраты расстояний считаются tiled GEMM:

\[
D_{ij}=\|x_i\|^2+\|x_j\|^2-2x_i^Tx_j.
\tag{35}
\]

Для `n=J=10000` полная `D` содержит `10^8` чисел и занимает 800 MB в FP64. Есть два режима:

- **cache-D:** один раз сохранить `D`; оптимально, если внешних ADP-шагов несколько и RAM позволяет;
- **stream-D:** хранить только tile и пересчитывать GEMM на каждом weight scan.

Перед GEMM из данных и центров вычитается один общий reference vector, например глобальное среднее `X`; расстояния от этого не меняются, а cancellation при больших абсолютных offsets уменьшается. Из-за остаточного cancellation около границы `q=1` подозрительные малые отрицательные `D` не используются для решения о support, а пары в узкой error-полосе пересчитываются прямой FP64-нормой.

### 7.2. Точные screens

Из (3) все пары с

\[
|\beta^T\Delta|\ge h
\]

точно имеют нулевой вес независимо от `D` и `alpha`.

Из (4) все пары с

\[
u^T\Lambda u\ge h^2
\]

точно имеют нулевой вес.

Screen применяется до полной оценки `q`, если projected coordinates уже доступны, но только с FP64 error certificate. Пара отбрасывается непосредственно лишь когда нижняя граница с учётом rounding error всё ещё не меньше `h^2`. В пограничной полосе всегда вычисляется полный `q`, иначе projected-coordinate rounding способен создать false negative.

Для multi-index перед screen проверяются `||PP^T-I||` и `Lambda>=0` с заданным численным допуском. В полном тесте величина `D-||u||^2` не обнуляется безусловно. Если вычитание попало в полосу cancellation, она пересчитывается напрямую как

\[
\|(I-P^TP)\Delta\|^2.
\]

### 7.3. Выбор alpha

В complement tensor масса support невозрастает по `alpha`. Поэтому для заданного требования к средней массе выбирается **наибольшее допустимое** `alpha`: оно даёт наиболее узкий orthogonal support и самую разреженную допустимую задачу.

Фраза `smallest value` в multi-index разделе `multiindex(3).tex` приводит к `alpha=0`, если ноль допустим, то есть к максимально широкому support. Для вычислительно осмысленной и согласованной с single-index частью процедуры это спецификационная опечатка; solver использует `largest feasible alpha`.

### 7.4. Двухпроходная сборка

Значение `n_eff=sum_i w_ij` не определяет `k_j=|supp(w_j)|`. Например, `10000` малых положительных весов могут иметь массу `20`. Поэтому нельзя заранее выделять sparse graph из условия `n_eff≈20`.

Точная схема:

1. Первый tiled pass считает `k_j`, `c_j`, `ybar_j` и backend score. Полный `xbar_j in R^d` сохраняется только для dense-`U` центров; для `QH` достаточно projected mean либо неявного `a_j^T(Xv)_Nj`.
2. Для sparse центров второй pass создаёт CSR и `Q_j`.
3. Для dense центров второй pass напрямую аккумулирует `U_j,I_j`, не сохраняя dense weights.
4. `W`, `D`, `Q` и `U` никогда не хранятся все одновременно без необходимости.

Это даёт worst-case memory bound, не зависящий от `E`: плотный случай автоматически переходит к streaming `U`.

Для dense центра, когда weighted means уже известны из первого прохода, второй проход непосредственно суммирует

\[
U_j=\sum_{i\in N_j}a_{ji}
\bigl[\Phi_j(x_i-\bar x_j)\bigr](x_i-\bar x_j)^T,
\]

\[
I_j=\sum_{i\in N_j}a_{ji}
\bigl[\Phi_j(x_i-\bar x_j)\bigr](y_i-\bar y_j).
\tag{35a}
\]

Суммирование выполняется FP64 block-pairwise или weighted-merge формулами. Выражение `X^T W X/c - xbar xbar^T` как разность близких больших матриц не используется.

Матрицы направлений `Phi_j` тоже не нужно хранить для всех центров. Они воспроизводятся blockwise counter-based PRNG по `(outer_seed,j,s)` и удаляются сразу после построения `Q_j` или `U_j`. Это одновременно уменьшает память и сохраняет побитовую воспроизводимость независимо от числа потоков.

### 7.5. Поиск h и alpha без повторных полных scan

Для первого изотропного шага положим `e_ij=D_ij^2`, `y=h_0^4` и `L=J n_min`. Тогда

\[
M(y)=\sum_{ij}\left(1-\frac{e_{ij}}{y}\right)_+.
\]

Между соседними breakpoints `e_ij` active set фиксирован и

\[
M(y)=N_A-\frac{S_A}{y},
\qquad
y_*=\frac{S_A}{N_A-L}.
\tag{35b}
\]

Нужный breakpoint-интервал находится radix-selection с одновременным подсчётом `N_A,S_A` или selection по рабочей копии. Это заменяет десятки полных bisection scans одной selection-процедурой.

Для следующих шагов положим `z=alpha^2` и

\[
q_{ij}(z)=\frac{a_{ij}z+b_{ij}}{h^2},
\]

где `(a,b)=(D-s^2,s^2)` в single-index и `(D-||u||^2,u^T Lambda u)` в multi-index. Breakpoint пары:

\[
\tau_{ij}=\frac{h^2-b_{ij}}{a_{ij}},
\tag{35c}
\]

если `a_ij>0`. На фиксированном active set условие массы для ядра (5) является квадратным уравнением по `z`; его корень проверяется в найденном breakpoint-интервале. Пары с `a_ij=0` обрабатываются отдельно. Tiled monotone bisection остаётся memory-minimal fallback.

Для single-index projected coordinates `X beta` сортируются один раз, и exact screen реализуется двумя указателями по интервалам `|x_i^T beta-x_j^T beta|<h`. Для multi-index используется exact radius search в координатах `Lambda^(1/2) P X`; при `m<=5` подходят grid/kd-tree/ball-tree. Если candidate ratio становится большим, solver автоматически переключается на последовательный tiled scan `D`.

## 8. Параметр lambda, scaling и stopping

### 8.1. Что означает lambda

`lambda_t` в (13) есть proximal step-control, не statistical ridge. GCV не определяет его. Уже для `A=I` GCV может быть константой по `lambda`, тогда как длина шага меняется как `1/(1+lambda)`.

Если нужен statistical ridge, вводится отдельный параметр `lambda_stat` в неизменяемый objective. Его нельзя смешивать с `lambda_prox`.

### 8.2. Безразмерная шкала

Точная Frobenius-норма глобального оператора равна

\[
\|\mathcal A_t\|_F^2
=\sum_jc_j\|\ell_j\|^2\|U_j\|_F^2.
\tag{35d}
\]

Если нормы `U_j` доступны, вычислить

\[
s_A^2=\frac{\|\mathcal A_t\|_F^2}{dm}
\]

точно и работать с

\[
\bar\lambda_t=\lambda_t/s_A^2.
\]

Практический старт:

- single-index: `lambda_bar=0.05`;
- multi-index: `lambda_bar=0.10`.

В чистом `QH` режиме несколько Hutchinson probes дают дешёвую scale estimate, но не детерминированный bound. Значения `0.05/0.10` являются defaults, не частью теоремы.

### 8.3. Trust radius

Основной trust diagnostic есть размер **свободного proximal correction**

\[
\delta_t=\|\Delta_t\|_F/\sqrt m.
\]

Principal angles логируются дополнительно, но не заменяют `delta_t`: чисто радиальный single-index шаг имеет нулевой угол после нормировки и ненулевую `||Delta||`. Практические пределы:

- single, `d=100`: `delta_max=0.35`;
- single, `d=1000`: `delta_max=0.20`;
- multi, `d=100`: `delta_max=0.25`;
- multi, `d=1000`: `delta_max=0.15`.

Политика:

- шаг больше `0.9 delta_max`: удвоить `lambda` и повторить;
- `0.25..0.9 delta_max`: оставить `lambda`;
- меньше `0.25 delta_max` два раза подряд при хорошем descent: разделить `lambda` на 2 до floor.

Теорема требует конечный `lambda_max`. Практический cap задаётся в безразмерной шкале, например `lambda_bar_max=10^6`. При достижении cap и нарушенном trust radius run возвращает diagnostic, а малый шаг не принимается за сходимость без проверки data-gradient.

### 8.4. Condition floor

Если `L>=||A||_2`, то для **нескалированного** augmented operator

\[
\lambda_{cond}=\frac{L^2}{\kappa_*^2-1}
\tag{36}
\]

гарантирует condition number augmented operator не больше `kappa_*`. Разумный default `kappa_*=300`. Для детерминированной гарантии `L` должен быть сертифицированной верхней границей, например точной `||A||F`; необрамлённая power/Hutchinson estimate такой гарантии не даёт. При right scaling

\[
\kappa(C_\lambda D_s)\le\kappa(C_\lambda)\kappa(D_s),
\]

поэтому исходная граница может резко ухудшиться. Condition transformed augmented operator оценивается заново.

Для mixed precision можно использовать защитный floor

\[
\lambda_{mp}=10(2u_{eff}+u_{eff}^2)\|A\|_F^2.
\tag{37}
\]

(37) защищает только от perturbation Hessian при наличии bound `||A_hat-A||<=u_eff||A||F`, включающего GEMM, центрирование, scatter-add и редукции. Сам этот floor не сертифицирует ошибку правой части и descent. Если используются аппроксимированные `A_hat,r_hat`, полный a-priori критерий имеет вид

\[
\|e_g\|+\|E_H\|\|\widehat\Delta\|+\|\widehat q\|
\le\theta\lambda\|\widehat\Delta\|,
\]

\[
E_H=\widehat A^*\widehat A-A^*A,
\qquad e_g=\widehat A^*\widehat r-A^*r.
\tag{37a}
\]

Предпочтительный полный сертификат требует истинный `q` из (19a), вычисленный reference FP64 реализацией исходных statistics и `A/A*`, и accepted `F`, также вычисленную FP64. Простое преобразование уже округлённого сохранённого FP32-оператора в FP64 reference-пересчётом не является. Если reference operator недоступен, mixed solve считается эвристикой. Если FP64 certificate не проходит, весь solve выполняется FP64, а не молча увеличивается `lambda`.

### 8.5. LSMR forcing

Проверять (33) с:

- `theta=0.10` на первых грубых шагах;
- `theta=0.02..0.03` в основном режиме;
- `theta=0.005..0.01` перед финальным stop.

Normal residual `q_t` измеряет точность решения текущего proximal subproblem, но не близость ADP-задачи к стационарности. Например, при огромном `lambda_t` exact LSMR имеет `q_t=0`, хотя data-gradient может оставаться большим. Поэтому после gauge-fix выполняется **новый точный локальный refit** и рассчитываются

\[
G_P=-\sum_jc_j\ell_j
\left[U_j^T(I_j-U_jP^T\ell_j)\right]^T,
\]

\[
G_R=G_P-\operatorname{sym}(G_PP^T)P,
\qquad
g_{\ell,j}=-c_j(U_jP^T)^T(I_j-U_jP^T\ell_j).
\tag{38a}
\]

Для exact локального refit второй gradient равен нулю с точностью QR/SVD. Пусть `A_ell` обозначает оператор (15), построенный с текущими slopes. Сертифицированный outer stop внутреннего AO:

\[
s_G=\max\{1,\|\mathcal A_{\ell}\|_F\sqrt{2F(P,\ell)}\},
\]

\[
\frac{|F_{t+1}-F_t|}{\max(1,F_t)}<10^{-7},
\qquad
\max\left\{
\frac{\|G_R\|_F}{s_G},
\max_j\frac{\|(U_jP^T)^T(I_j-U_jP^T\ell_j)\|}
{\max\{1,\|U_jP^T\|_2\|I_j\|\}},
\|P_{t+1}P_{t+1}^T-I_m\|_F
\right\}<10^{-6},
\]

два раза подряд, вместе с малым gauge-aligned шагом. Порог можно ослабить для noisy data, но proximal normal residual `q_t`, Riemannian data-gradient и orthogonality всегда логируются **раздельно**.

### 8.6. Preconditioner и fallback

Сначала запуск без сложного preconditioner. Randomized Nyström строится только если pilot показывает:

- более 40 LSMR-итераций при single `d=100`;
- более 60 при single `d=1000`;
- более 80 при multi `d=1000`.

Если используется **точное** инвариантное пространство первых `r` singular vectors и `sigma_(r+1)^2<=lambda`, idealized deflation даёт для предобусловленной normal matrix

\[
\kappa\le1+\frac{\sigma_{r+1}^2}{\lambda}\le2,
\]

а для соответствующего augmented operator bound равен `sqrt(2)`. Randomized subspace сам по себе это условие не сертифицирует. Проверяемый вариант использует `H=A^*A`,

\[
\widetilde H=HV(V^THV)^\dagger V^TH,
\quad R=H-\widetilde H\succeq0,
\quad M=\lambda I+\widetilde H,
\]

и certificate

\[
\kappa\bigl(M^{-1/2}(H+\lambda I)M^{-1/2}\bigr)
\le1+\frac{\operatorname{tr}R}{\lambda}.
\tag{38b}
\]

Без upper bound на trace residual Nyström остаётся heuristic acceleration.

Практическая основа такого подхода описана в работе о [randomized Nyström preconditioning](https://epubs.siam.org/doi/10.1137/21M1466244). Preconditioner перестраивается, если slopes или внешний ADP-оператор существенно изменились.

Для single-index при `d<=100..256` и медленном Krylov разрешён прямой pivoted QR. Для multi-index при `d=1000` глобальная матрица не материализуется даже как fallback.

Variable projection с Riemannian L-BFGS может использоваться только как accepted polish после стабильных ALS sweeps: локальные slopes профилируются, line search должен уменьшать исходную `F`. Основа variable projection дана у [Golub and Pereyra](https://epubs.siam.org/doi/10.1137/0710036), геометрия Stiefel/Grassmann оптимизации у [Edelman, Arias and Smith](https://epubs.siam.org/doi/10.1137/S0895479895290954). При смене локального ранга этот режим отключается.

Все числовые defaults этого раздела, включая initial `lambda`, trust radii, `kappa_*=300`, forcing values, iteration triggers и stopping thresholds, являются engineering policy. Теорема использует только явно перечисленные bounds и residual certificates. Хорошая обусловленность inner solve не гарантирует быстрое outer AO: чрезмерно большой `lambda` делает proximal iteration сколь угодно медленной, что и требует отдельного Riemannian-gradient stop.

## 9. Stress-test памяти

Пусть `n=J=10000`, `m=5`, FP64. В таблице `p=10`, кроме отдельно отмеченной строки.

| Объект | `d=100` | `d=1000` | Решение |
|---|---:|---:|---|
| `X` | 8 MB | 80 MB | Хранить |
| Отдельный массив центров | 8 MB | 80 MB | Alias при `J=n`, иначе хранить |
| Полная `D`, `n x J` | 800 MB | 800 MB | Cache или tiles |
| Явные `U_j`, `J x p x d` | 80 MB | 800 MB | Только dense backend |
| `Q_j`, sparse `k=10` | 8 MB | 8 MB | Основной sparse backend |
| CSR graph, `E=100000` | около 1.28 MB | около 1.28 MB | Хранить |
| Multi design `(Jp) x (dm)` | 0.4 GB | 4 GB | Не строить |
| Все локальные `d x d` covariance | 0.8 GB | 80 GB | Не строить |
| Все разности `n x J x d` | 80 GB | 800 GB | Запрещено |

Итоговые пики:

| Режим | `d=100` | `d=1000` |
|---|---:|---:|
| Sparse `QH`, `k=p=10`, cache-D | около 0.83-0.90 GB | около 0.90-1.0 GB |
| Dense exact `U`, cache-D | около 0.90 GB | около 1.8 GB + tile |
| Dense exact `U`, `p=40`, cache-D | около 1.15 GB | около 4.2 GB + tile |
| Stream-D | минус 800 MB | минус 800 MB |

Эти bounds включают основные persistent arrays, но не дублирование данных библиотекой и не GPU allocator overhead. Для GPU реализация должна заранее резервировать workspaces и не держать одновременно `D`, dense `W`, `Q` и `U`.

## 10. Stress-test времени и арифметики

### 10.1. Детерминированные оценки

Для полной матрицы расстояний leading GEMM требует примерно

\[
2nJd
\]

FLOP:

- `d=100`: 20 GFLOP;
- `d=1000`: 200 GFLOP.

При sparse `k=p=10` построение всех `Q_j` стоит порядка `2pEd=2` GFLOP для `d=1000`; materialization всех `U_j` добавляет ещё примерно 2 GFLOP.

В худшем плотном случае `E=nJ=10^8`, `p=10`, `d=1000` exact statistics требуют порядка нескольких TFLOP. Это всё ещё memory-feasible в streaming режиме, но не существует distribution-free способа одновременно сохранить точную compact-support ADP-статистику и гарантировать subquadratic scan всех `nJ` потенциальных пар. Ускорение получается из exact screens, разреженности и низкоразмерного projected neighbor search, когда они присутствуют.

### 10.2. Воспроизводимая синтетическая проверка оператора

Скрипт `adp_operator_benchmark.py` строит fused single-index `QH` LinearOperator без design matrix и запускает задачу точно на `n=J=10000`, `d=1000`, `p=k=10`. Зафиксированный запуск использовал 9-vCPU AMD EPYC, NumPy 2.3.5, SciPy 1.17.0, FP64 и `damp=0.1`:

| Проверка | Результат |
|---|---:|
| Размер оператора | `100000 x 1000` |
| Persistent arrays benchmark | 85.14 MiB |
| Относительная ошибка adjoint identity | `1.19e-18` |
| Fused forward+adjoint | 0.0144 s |
| LSMR iterations | 27 |
| LSMR solve | 0.186 s |
| Оценка condition | 1.237 |
| Cosine с известным направлением | 1.00000000 |

Команда:

```bash
python3 adp_operator_benchmark.py
```

Это throughput sanity-check, не оценка полного ADP runtime. В него не входят distance scan, построение weights/statistics, multi-index contractions и реальная обусловленность конкретных данных. Измеренные времена и число итераций не являются гарантией. Сертифицированы adjoint identity, отсутствие запрещённых матриц и выполнимость operator solve на полном stress-размере.

## 11. Псевдокод

```text
input: X, Y, centers, p, m, outer schedule (h, alpha), P0

precompute/cache or stream squared distances D by tiled GEMM

for each outer ADP step:
    choose largest feasible alpha by monotone search
    pass 1 over distance tiles:
        compute exact q and compact-support weights
        accumulate c_j, k_j, ybar_j; keep xbar_j only for dense-U
        classify each center as QH or U backend

    pass 2 over tiles:
        QH centers: build CSR supports and Q_j, then I_j=Q_j g_j
        U centers: stream-accumulate U_j and I_j; discard dense weights

    P = Procrustes_align(P_from_previous_outer_step)
    initialize ell

    for t = 0,1,...,max_inner:
        # exact local block
        for j:
            C_j = U_j P^T              # or Q_j(H_j P^T)
            ell_j = pinv_rank_revealing(C_j) I_j
            record rank and condition

        # matrix-free proximal global block
        define A(Delta), A*(z) using hybrid QH/U backend
        estimate operator scale and choose lambda_t
        solve [A; sqrt(lambda_t) I] Delta = [r; 0] by LSMR
        verify adjoint once per operator version
        verify normal-residual certificate (33)

        B = P + Delta
        [Q,R] = thin_qr(B^T)
        P_new = Q^T
        ell_j = R ell_j for all j
        Procrustes-align P_new and ell to P

        recompute exact local slopes at P_new
        evaluate original F, Riemannian gradient, rank,
                 orthogonality PP^T-I_m, ||Delta||F/sqrt(m), angles
        reject/retry with larger lambda if correction trust radius is exceeded
        adapt lambda
        stop only after two certified small-change iterations
        P = P_new

    M = sum_j c_j ell_j ell_j^T          # m x m only
    M = O Lambda O^T
    P = O^T P; ell_j = O^T ell_j
    if lambda_max(Lambda)>0:
        output P, Lambda/lambda_max(Lambda), diagnostics
    else:
        retain isotropic tensor and output zero-spectrum diagnostic
```

Финальный EDR tensor не требует `d x d` eigendecomposition. Сначала считается маленькая матрица

\[
M=\sum_jc_j\ell_j\ell_j^T\in\mathbb R^{m\times m},
\]

затем `M=O Lambda O^T`, `P_can=O^T P`. Представление

\[
J_{EDR}=P_{can}^T\Lambda P_{can}
\]

остаётся low-rank. Для repeated eigenvalues базис внутри соответствующего eigenspace выравнивается с предыдущим шагом.

## 12. Обязательные diagnostics и критерий приёмки

Реализация считается корректной только если проходят все проверки:

1. Random adjoint test:

\[
\frac{|\langle A\Delta,z\rangle-\langle\Delta,A^*z\rangle|}
{\|A\Delta\|\|z\|+\|\Delta\|\|A^*z\|}
<10^{-11}
\]

в FP64.

2. `QH` и explicit `U` дают совпадающие `I`, forward и adjoint на малой общей задаче.
3. После gauge-fix fitted vectors совпадают до `1e-12` relative.
4. Data-loss не возрастает на accepted AO iterations.
5. Проверяется normal-residual certificate (33), а не только внутренний флаг LSMR; он не используется вместо Riemannian-gradient stop.
6. `||PP^T-I_m||_F` логируется после каждого QR/SVD.
7. Rank loss локальных `C_j` и глобального `B` не скрывается.
8. Ни один allocation не имеет формы `n x J x d`, `J x d x d` или `(Jp) x (dm)`.
9. Weight branch, kernel interpretation и внешняя нормировка `c_j` записываются в metadata запуска.
10. Результаты sparse и dense backend совпадают на одинаковых weights.

## 13. Исправления, необходимые в приложенных формулах

Перед переносом алгоритма в основной текст следует исправить:

1. Размер single slopes: `R^J`, не `R^p`.
2. Размер multi slopes: `R^(Jm)`, не `R^(mp)`.
3. `Id_dimp` в single global step: должно быть `Id_d`.
4. Последний multi-index шаг в `manifold_v2.tex` ошибочно возвращается к scalar `f_j` и `beta`; нужен residual `I_j-U_jP^T ell_j`.
5. Нормировка `beta` требует `f_j <- ||beta_hat|| f_j`.
6. QR/SVD нормировка `B` требует соответствующего преобразования всех `ell_j`.
7. Нельзя смешивать normalized statistics с `c_j` и unnormalized statistics без `c_j`.
8. Нельзя смешивать complement tensor из (3) с tensor `alpha^2 I+beta beta^T`.
9. В multi-index выборе `alpha` требуется largest feasible, не smallest feasible.
10. Локальный multi-index inverse нужно заменить на QR/SVD pseudoinverse.
11. Если после исключения nuisance-параметра data-term получает множитель `1/4`, шкала `lambda` должна измениться тем же образом.
12. `P in R^(m x d)` и projector `P^TP in R^(d x d)` должны иметь разные имена в коде.
13. Начальная запись `T_0=h_0^(-2)I` несовместима с последующим `q=D/h_0^2`; согласованная форма есть `T_0=h_0^(-1)I`.

## 14. Итоговая гарантия эффективности

При фиксированном числе ADP-направлений `p` и малом index dimension `m` HPAO-LSMR имеет:

\[
\text{память}
=O(nd+nJ_{cache}+\min\{pE,Jpd\}+dm),
\tag{38}
\]

где `nJ_cache` отсутствует в stream-D режиме, а dense fallback не хранит `E` edges.

Один глобальный Krylov forward+adjoint стоит

\[
O\left(\min\{ndm+E(p+m),\;Jpd+Jdm\}\right)
\tag{39}
\]

с поправкой на per-center hybrid. Локальные solves стоят

\[
O(Jpm^2+Jm^3)
\]

после формирования `C_j`; при малом `m` это вторично.

В stress-case `n=J=10000`, `d=1000` доказанные memory bounds оставляют задачу выполнимой на обычной workstation. В целевом `d=100` прямой QR уже возможен как страховка, но matrix-free HPAO-LSMR остаётся единым основным алгоритмом. Его преимущество не основано на предположении `n_eff≈k`: он сохраняет ограниченную память и в adversarial dense-support случае.
