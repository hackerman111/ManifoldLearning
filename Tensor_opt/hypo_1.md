## 1. Что здесь можно усилить

После предыдущих оптимизаций есть три разных объекта, которые нельзя смешивать:

[
w_{ij}
\quad\longrightarrow\quad
\mathcal N_j={i:w_{ij}>0}
\quad\longrightarrow\quad
(I_j,U_j).
]

Для центра (j) обозначим

[
k_j=|\mathcal N_j|.
]

После локального центрирования ADP имеет точную факторизацию

[
U_j=\Phi_jC_j=Q_jR_j,
]

где

[
C_j=R_j^\top R_j,
\qquad
\operatorname{rank}C_j
======================

\operatorname{rank}R_j
\le k_j-1.
]

Следовательно,

[
\boxed{
\operatorname{rank}U_j
\le
\min{n_\Phi,k_j-1,d}.
}
\tag{1}
]

Это уже установлено в твоём предыдущем разборе.

Отсюда первый сильный вывод:

[
\boxed{
\text{единственный прямой способ заставить }U_j
\text{ иметь меньший гарантированный ранг через kernel}
=======================================================

\text{уменьшить }k_j.
}
]

Изменение значений ненулевых весов при неизменном support не улучшает верхнюю оценку (1).

Я рассматривал четыре действительно разных механизма:

| Семейство                       | Что меняем                               | Статус                                    |
| ------------------------------- | ---------------------------------------- | ----------------------------------------- |
| Support-aware выбор центров     | множество (j)                            | есть точные правила и эвристики           |
| Information-aware выбор центров | вклад центра в глобальную LS-задачу      | строгая теория после построения статистик |
| Изменение kernel                | (k_j), значения весов, возможность reuse | есть точные вычислительные результаты     |
| Динамический reuse support      | пересчёт между outer steps               | exact при сертифицированных границах      |

---

# 2. Первый новый результат: центр с (k_j\le1) бесполезен точно

Это простое, но полезное следствие нулевого support.

Если (k_j=1), существует единственная точка (X_i) с ненулевым весом. Тогда локальное среднее

[
M_j=X_i.
]

Поэтому

[
X_i-M_j=0
]

и

[
R_j=0.
]

Следовательно,

[
C_j=0,\qquad
U_j=0.
]

Также центрированный локальный первый момент равен нулю, поэтому

[
I_j=0.
]

Получаем:

[
\boxed{
k_j\le1
\quad\Longrightarrow\quad
(I_j,U_j)=(0,0).
}
\tag{2}
]

Такой центр можно удалить **до генерации направлений и подсчёта статистик**, не меняя objective.

Это уже не эвристика.

Более общий вариант:

[
\operatorname{rank}R_j=0
\quad\Longrightarrow\quad
U_j=I_j=0.
]

Например, это происходит, если все точки ненулевого support совпадают.

---

# 3. Осознанный выбор центров через граф ненулевых весов

Здесь твоя идея работает особенно хорошо.

После дешёвого screening мы уже приблизительно или точно знаем двудольный граф

[
G_k=(\mathcal J,\mathcal X,E_k),
]

где

[
(j,i)\in E_k
\iff
w_{ij}^{(k)}>0.
]

Для single-index возможные ненулевые веса можно найти через одномерную проекцию. Более сильное exact условие имеет вид

[
|\beta^\top(X_i-x_j)|
\ge
\frac{h}{\sqrt{1+\alpha^2}}
\Longrightarrow w_{ij}=0.
]

Такой screening является оптимальным среди тестов, использующих только эту проекцию.

Именно (G_k) можно использовать для выбора центров **до дорогостоящего (U_j)**.

## 3.1. Maximum-coverage centers

Пусть разрешено использовать только (s<J) центров.

Определим

[
F(S)
====

\left|
\bigcup_{j\in S}\mathcal N_j
\right|,
\qquad |S|\le s.
\tag{3}
]

Мы ищем

[
\max_{|S|\le s}F(S).
]

То есть центр выбирается не случайно, а если его support покрывает точки, плохо покрытые уже выбранными центрами.

Функция (F) монотонна и субмодулярна.

Действительно, если (A\subseteq B), то для любого центра (j)

[
F(A\cup{j})-F(A)
================

|\mathcal N_j\setminus\cup_{\ell\in A}\mathcal N_\ell|
]

и

[
F(B\cup{j})-F(B)
================

|\mathcal N_j\setminus\cup_{\ell\in B}\mathcal N_\ell|.
]

Так как

[
\bigcup_{\ell\in A}\mathcal N_\ell
\subseteq
\bigcup_{\ell\in B}\mathcal N_\ell,
]

получаем diminishing returns:

[
\Delta(j\mid A)\ge\Delta(j\mid B).
]

Поэтому стандартный greedy:

[
j_t
===

\arg\max_j
|\mathcal N_j\setminus C_{t-1}|
]

даёт

[
\boxed{
F(S_{\rm greedy})
\ge
(1-e^{-1})F(S_{\rm opt}).
}
\tag{4}
]

То есть мы можем **строго оптимально с точностью (1-1/e)** решать конкретную задачу выбора representative centers.

### Что это доказывает

Это гарантирует качество покрытия weight graph.

### Что это не доказывает

Это не доказывает

[
\widehat\beta_S\approx\widehat\beta_{\rm full}.
]

Два центра с одинаковым support могут содержать разную derivative information.

Поэтому coverage полезен как первая стадия, но одного его недостаточно.

---

# 4. Лучше: weighted coverage

Можно штрафовать плохие локальные окрестности.

Например,

[
n_{\mathrm{eff},j}
==================

\frac{(\sum_iw_{ij})^2}
{\sum_iw_{ij}^2}.
]

Тогда определить качество центра

[
q_j
===

\mathbf 1{n_{\mathrm{eff},j}\ge n_{\min}}.
]

И использовать

[
F(S)
====

\sum_{i=1}^n
a_i
\mathbf1
\left{
i\in\bigcup_{j\in S:q_j=1}\mathcal N_j
\right}.
\tag{5}
]

Это всё ещё weighted coverage, поэтому остаётся монотонной субмодулярной функцией и сохраняет гарантию (1-1/e).

Эта версия автоматически отбрасывает:

- центры с почти пустыми окрестностями;
- центры, где один вес почти полностью доминирует;
- сильно дублирующиеся центры.

В предыдущем разборе (n_{\rm eff}) уже возникал как более содержательный критерий устойчивости локальной статистики, чем просто число ненулевых весов.

---

# 5. Ещё сильнее: учитывать стоимость центра

Цена обработки центра примерно пропорциональна (k_j).

При явном (U_j):

[
T_j=O(k_jn_\Phi d).
]

При factorized варианте основная (d)-мерная работа имеет порядок

[
T_j=O(k_jd+k_jn_\Phi).
]

Поэтому естественная эвристика:

[
\boxed{
\text{score}_j
==============

\frac{
|\mathcal N_j\setminus C|
}{
k_j
}.
}
\tag{6}
]

Она выбирает максимум новой информации на одно обработанное sparse edge.

Более содержательный вариант:

[
\text{score}_j
==============

\frac{
n_{\rm eff,j}
\cdot
|\mathcal N_j\setminus C|
}{
k_j
}.
\tag{7}
]

Я бы использовал именно это как **дешёвую pre-statistics эвристику**.

Она использует только веса и support.

---

# 6. Information-aware selection: центр как блок глобальной ridge-задачи

После построения статистик можно сделать существенно более математически правильный отбор.

В single-index на фиксированном ALS-шаге положим

[
A_j=f_jU_j.
]

Глобальная ridge-задача имеет Hessian

[
H
=

\lambda I+
\sum_{j=1}^J A_j^\top A_j.
\tag{8}
]

То есть центр (j) не просто «точка». Он вносит PSD-блок

[
H_j=A_j^\top A_j.
]

Следовательно, хорошие центры должны сохранять сумму этих матриц.

## Ridge leverage центра

Определим

[
\tau_j
======

\operatorname{tr}
\left(
A_jH^{-1}A_j^\top
\right).
\tag{9}
]

Это block ridge leverage score.

Причём

[
\sum_j\tau_j
============

\operatorname{tr}
\left[
(H-\lambda I)H^{-1}
\right]
=======

d_{\rm eff},
]

где

[
\boxed{
d_{\rm eff}
===========

\operatorname{tr}(I-\lambda H^{-1})
\le d.
}
\tag{10}
]

Центры с большим (\tau_j) содержат информацию в направлениях, которые плохо представлены остальными центрами.

Это намного лучше случайного выбора.

---

# 7. Почему leverage sampling действительно сохраняет задачу

Берём

[
p_j=\frac{\tau_j}{d_{\rm eff}}
]

и независимо выбираем (s) центров.

Для выбранного центра используем масштаб

[
\frac{1}{sp_j}.
]

Получаем случайный Hessian

[
\widehat H
==========

\lambda I+
\frac1s
\sum_{\ell=1}^s
\frac{A_{j_\ell}^\top A_{j_\ell}}
{p_{j_\ell}}.
\tag{11}
]

Сразу:

[
\mathbb E\widehat H=H.
]

Теперь определим

[
B_j
===

H^{-1/2}A_j^\top A_jH^{-1/2}.
]

Имеем

[
B_j\succeq0
]

и

[
|B_j|_2
\le
\operatorname{tr}B_j
====================

\tau_j.
]

Поэтому при (p_j=\tau_j/d_{\rm eff})

[
\left|
\frac{B_j}{p_j}
\right|_2
\le d_{\rm eff}.
]

Сумма математических ожиданий равна

[
\sum_jB_j
=========

I-\lambda H^{-1}
\preceq I.
]

Matrix concentration поэтому даёт при

[
s
=

O\left(
\frac{
d_{\rm eff}
\log(d/\delta)
}{
\varepsilon^2
}
\right)
\tag{12}
]

с вероятностью не менее (1-\delta)

[
\left|
H^{-1/2}
(\widehat H-H)
H^{-1/2}
\right|_2
\le\varepsilon.
]

Следовательно,

[
\boxed{
(1-\varepsilon)H
\preceq
\widehat H
\preceq
(1+\varepsilon)H.
}
\tag{13}
]

Это уже настоящая сильная оптимизация центров: вместо (J) центров можно сохранить число порядка effective dimension, при этом ridge geometry глобальной задачи сохраняется спектрально.

---

# 8. Но leverage scores нельзя использовать как первую screening-стадию

Здесь есть принципиальный недостаток:

[
\tau_j
]

требует знания (U_j).

Если сначала вычислить все (U_j), а потом удалить центры, основная стоимость statistics stage уже понесена.

Поэтому я предлагаю двухступенчатую схему.

### Outer iteration (k=0)

Использовать:

[
\text{support coverage} +
n_{\rm eff}
]

для первого отбора.

### После первого решённого шага

У нас уже имеются (U_j), (f_j) или их факторизации.

Считаем approximate leverage scores и запоминаем.

### На шаге (k+1)

Используем leverage scores с шага (k) как prior:

[
p_j^{(k+1)}
\propto
\tau_j^{(k)} +
\gamma,c_j^{(k+1)},
\tag{14}
]

где (c_j) представляет support-novelty score.

Это уже **эвристика**, потому что (U_j^{(k+1)}\ne U_j^{(k)}).

Но она имеет понятный механизм: часть score отвечает за глобальную linear-algebra information, часть за изменившуюся локальную геометрию.

---

# 9. Особенно хороший центр можно определить ещё дешевле

В single-index текущая (\beta_k) уже известна.

Для каждого support можно дешёво посчитать variance вдоль текущего индекса:

[
v_j
===

\sum_{i\in\mathcal N_j}
w_{ij}
\left[
\beta_k^\top(X_i-M_j)
\right]^2.
\tag{15}
]

Все

[
z_i=\beta_k^\top X_i
]

уже вычисляются для exact screening. Поэтому после этого (v_j) считается за

[
O(k_j),
]

а не (O(k_jd)).

Центр с

[
v_j\approx0
]

почти не содержит локальной изменчивости вдоль текущего EDR-направления.

Можно использовать

[
\boxed{
\text{score}_j
==============

\frac{
n_{\rm eff,j}
,v_j
,\operatorname{novelty}_j
}{
k_j
}.
}
\tag{16}
]

Это очень дешёвая ADP-specific эвристика.

Она значительно ближе к derivative estimation, чем просто coverage.

**Пробел:** (v_j) измеряет информацию только вдоль текущей (\beta_k). Если (\beta_k) плоха, метод способен недооценить центры, необходимые для её исправления.

Поэтому стоит сохранять, например, (10%-20%) exploration centers со случайным или coverage-based выбором.

---

# 10. Что происходит при замене ядра

Теперь к второй части вопроса.

Пусть

[
w_{ij}=K(q_{ij}),
\qquad
q_{ij}=|T(X_i-x_j)|^2.
]

Предположим

[
K(q)>0\quad (q<R),
\qquad
K(q)=0\quad(q\ge R).
]

Тогда

[
\boxed{
w_{ij}=0
\iff q_{ij}\ge R.
}
\tag{17}
]

Все exact-screening рассуждения просто заменяют порог (1) на (R).

---

# 11. Отрицательный результат: форма compact kernel не меняет sparsity

Рассмотрим два ядра (K_1,K_2), для которых

[
K_1(q)>0
\iff q<1
]

и

[
K_2(q)>0
\iff q<1.
]

Тогда

[
\operatorname{supp}W^{(1)}
==========================

\operatorname{supp}W^{(2)}.
]

Следовательно,

[
k_j^{(1)}=k_j^{(2)}
]

для каждого центра.

По (1):

[
\operatorname{rank}U_j
\le
\min(n_\Phi,k_j-1,d)
]

имеет одну и ту же верхнюю границу.

Поэтому переход, например, между

[
(1-q)_+,
\qquad
(1-q)^2_+,
\qquad
(1-q^2)_+
]

**сам по себе не даёт ни одного дополнительного нуля**.

Это относится и к памяти sparse representation:

[
E=\operatorname{nnz}W
]

не меняется.

То есть поиск «более разреженного Epanechnikov-подобного kernel с тем же support» математически бесперспективен.

---

# 12. Kernel не может в общем случае сделать (U_j) разреженной по координатам

Строка (U_j) имеет вид

[
U_{jr}
======

\sum_{i\in\mathcal N_j}
c_{jri}(X_i-M_j).
\tag{18}
]

Даже если только (k_j=2) коэффициента ненулевые, каждый

[
X_i-M_j\in\mathbb R^d
]

обычно плотный.

Скалярный kernel меняет только коэффициенты (c_{jri}).

Он не создаёт структурных нулей в координатах (X_i-M_j).

Поэтому для generic dense data

[
\boxed{
\text{никакой scalar radial kernel не гарантирует coordinate-sparse }U_j.
}
\tag{19}
]

Чтобы получить column sparsity в (U_j), нужно уже другое изменение модели:

- sparse features (X);
- coordinate selection;
- sparse projection;
- отдельная регуляризация (U).

Kernel здесь не решает задачу.

---

# 13. Но kernel может гарантировать малый ранг через bounded support

Вот более перспективная конструкция.

Для каждого центра выбрать локальный bandwidth (h_j) как расстояние до (K)-го соседа в текущей метрике:

[
h_j^2=q_{j,(K)}.
]

Использовать любой compact kernel

[
w_{ij}
======

\psi(q_{ij}/h_j^2)
\mathbf1{q_{ij}<h_j^2}.
\tag{20}
]

При отсутствии ties:

[
k_j\le K.
]

Следовательно,

[
\boxed{
\operatorname{rank}U_j
\le
\min(n_\Phi,K-1).
}
\tag{21}
]

Кроме того,

[
\boxed{
\operatorname{nnz}W\le JK.
}
\tag{22}
]

Это уже **детерминированная память**, а не ожидаемая sparsity.

Если (K) не зависит от (n),

[
M(W)=O(JK)
]

вместо

[
O(Jn).
]

Для sparse statistics

[
T_{\rm stat}
============

O(JK n_\Phi d)
]

вместо

[
O(Jn n_\Phi d).
]

А при factorized (U_j=Q_jR_j) хранение становится порядка

[
O(JKn_\Phi+JK)
]

вместо

[
O(Jn_\Phi d).
]

Разреженное представление статистик уже снижает основную стоимость с (O(JnPd)) до (O(EPd)).

---

# 14. Это даёт очень интересный выбор (K)

Если

[
K\le n_\Phi,
]

то

[
\operatorname{rank}U_j\le K-1<n_\Phi.
]

То есть (U_j) становится **гарантированно row-rank deficient**.

Например:

[
n_\Phi=32,\qquad K=16
]

даёт

[
\operatorname{rank}U_j\le15.
]

Но здесь возникает статистический конфликт.

Для устойчивой локальной derivative estimation нужен достаточный effective sample size.

Поэтому нельзя просто ставить (K=2) ради rank (1).

Практически ограничение должно иметь вид

[
K
\ge
K_{\min}
]

и одновременно

[
n_{\rm eff,j}\ge n_{\min}.
]

---

# 15. Какое ядро я бы исследовал первым

Не новое гладкое ядро, а **adaptive (K)-support kernel**:

[
\boxed{
K_j(q)
======

\left(1-\frac{q^2}{h_j^4}\right)_+,
\qquad
h_j^2=q_{j,(K)}.
}
\tag{23}
]

Почему:

1. сохраняется привычная форма текущего kernel;
2. support ограничен (K);
3. (\operatorname{nnz}W\le JK);
4. (\operatorname{rank}U_j\le K-1);
5. память заранее известна;
6. подходит neighbor-list representation без CSR padding uncertainty;
7. density адаптируется через (h_j).

Пункт 7 статистически правдоподобен, но я здесь **не утверждаю**, что этот estimator лучше текущего global-(h) ADP. Это требует отдельного statistical analysis.

---

# 16. Ещё более вычислительно интересное ядро: box kernel

Возьмём

[
\boxed{
K_{\rm box}(q)=\mathbf1{q<1}.
}
\tag{24}
]

Support такой же, как у текущего compact kernel.

Следовательно, sparsity и rank не улучшаются.

Но появляется другое свойство:

[
w_{ij}\in{0,1}.
]

Вес зависит **только от support**.

Это особенно важно между outer iterations.

Из предыдущего exact reuse результата:

[
q_{ij}^{(k-1)}-\eta_kD_{ij}\ge1
\Longrightarrow
w_{ij}^{(k)}=0,
]

и

[
q_{ij}^{(k-1)}+\eta_kD_{ij}<1
\Longrightarrow
w_{ij}^{(k)}>0.
]

Для smooth kernel этого недостаточно: даже если точка осталась внутри support,

[
K(q_k)\ne K(q_{k-1}).
]

Для box kernel, если мы доказали, что threshold не пересечён, то

[
\boxed{
w_{ij}^{(k)}
============

w_{ij}^{(k-1)}
}
]

точно.

То есть пересчитывать нужно только boundary set

[
B_k
===

\left{
(i,j):
|q_{ij}^{(k-1)}-1|
\le
\eta_kD_{ij}
\right}.
\tag{25}
]

Если

[
|B_k|\ll Jn,
]

weight-update становится существенно дешевле.

Это сильное вычислительное преимущество box kernel, которого нет у Epanechnikov.

Статистически box kernel менее гладкий. Я не утверждаю, что качество ADP сохранится.

---

# 17. Компромисс: ступенчатое compact kernel

Можно обобщить box kernel.

Выбрать

[
0=t_0<t_1<\cdots<t_L=1
]

и

[
K_L(q)=c_\ell,
\qquad
q\in[t_{\ell-1},t_\ell),
]

а при

[
q\ge1
]

положить (K_L(q)=0).

Например (L=4) или (8).

Теперь вес меняется только при пересечении одного из (L) thresholds.

Если известен интервал

[
q_k\in[q_{k-1}-\delta_{ij},
q_{k-1}+\delta_{ij}]
]

и весь этот интервал лежит внутри одной ступени, то

[
\boxed{
w_{ij}^{(k)}=w_{ij}^{(k-1)}
}
\tag{26}
]

точно.

Получается семейство:

[
L=1
\quad\text{box},
]

[
L\to\infty
\quad\text{приближение smooth kernel}.
]

Это новый параметр time/statistical-smoothness tradeoff.

Он не меняет sparsity, зато способен резко уменьшить число обновляемых weight values.

---

# 18. Можно объединить center selection и kernel design

Теперь получается интересная схема.

На outer step (k):

### Стадия 1. Cheap projected screening

Single-index:

[
z_i=X_i^\top\beta_{k-1}.
]

Exact candidate interval:

[
|z_i-z_j|
<
\frac{h_j}{\sqrt{1+\alpha^2}}.
]

Не рассматриваем заведомо нулевые пары. Exact range search для этого уже имеет стоимость

[
O((n+J)d+n\log n+J\log n+C_kd)
]

вместо (O(Jnd)), где (C_k) число surviving candidates.

### Стадия 2. Adaptive (K)-support

Оставляем не более (K) реальных соседей на центр.

Получаем sparse graph

[
E_k\le JK.
]

### Стадия 3. Exact center pruning

Удаляем

[
k_j\le1.
]

### Стадия 4. Stability pruning

Удаляем из candidate centers те, для которых

[
n_{\rm eff,j}<n_{\min}.
]

Это уже policy, а не алгебраическая эквивалентность.

### Стадия 5. Greedy coverage

Из оставшихся выбираем (s) центров, максимизируя новый support.

Гарантия:

[
F(S_{\rm greedy})
\ge(1-1/e)F(S_{\rm opt}).
]

### Стадия 6. Information correction

Score:

[
\text{score}_j
==============

\frac{
n_{\rm eff,j}
v_j
\operatorname{novelty}_j
}{
k_j
}.
]

На следующих outer steps добавляем прошлый leverage score.

### Стадия 7. Factorized statistics

Для выбранных центров:

[
U_j=Q_jR_j
]

не материализуется.

---

# 19. Сложность такой схемы

Пусть:

[
s=\text{число выбранных центров},
\qquad
K=\max_j k_j,
\qquad
p=n_\Phi.
]

Вместо исходного

[
O(Jnpd)
]

на statistics stage получаем после построения sparse neighborhoods

[
\boxed{
O(sKpd)
}
\tag{27}
]

для явного вычисления статистик.

Отношение:

[
\frac{T_{\rm old}}{T_{\rm new}}
\approx
\frac{Jn}{sK}.
]

Например,

[
J=n=2000,\qquad
s=400,\qquad
K=32.
]

Тогда арифметическое отношение:

[
\frac{2000\cdot2000}
{400\cdot32}
============

312.5.
]

Это не означает (312.5\times) wall-clock speedup: остаются screening, memory access, solver и служебные расходы.

Но число пар, на которых реально считаются ADP statistics, уменьшается именно в (312.5) раза.

---

# 20. Память

Для явного

[
U\in\mathbb R^{J\times p\times d}
]

память:

[
M_{\rm explicit}
================

\Theta(Jpd).
]

После выбора центров:

[
\Theta(spd).
]

При факторизации:

[
U_j=Q_jR_j,
]

можно хранить только neighbor list, веса и (Q_j):

[
\boxed{
M_{\rm fact}
============

O(sKp+sK).
}
\tag{28}
]

При

[
d=1000,\quad
J=2000,\quad
p=32,\quad
s=400,\quad
K=32
]

явный полный (U):

[
2000\cdot32\cdot1000
====================

64\cdot10^6
]

чисел.

В `float32`:

[
256\text{ MB}.
]

Только (Q_j):

[
400\cdot32\cdot32
=================

409,600
]

чисел:

[
1.64\text{ MB}.
]

Плюс neighbor indices и weights.

В предыдущем анализе factorized ADP уже сокращал локальную память с (O(Jpd)) до (O(Ep)). Здесь center selection дополнительно заменяет (E) примерно на (sK).

---

# 21. Самая сильная версия, которую я сейчас считаю перспективной

Я бы не искал «лучший polynomial kernel».

Я бы изменил саму вычислительную постановку:

[
\boxed{
\begin{array}{c}
\text{exact projected zero screening}
[1mm]
\downarrow
\
\text{adaptive bounded support }k_j\le K
[1mm]
\downarrow
\
\text{support graph }G_k
[1mm]
\downarrow
\
\text{удаление }k_j\le1
[1mm]
\downarrow
\
\text{coverage/information-aware center selection}
[1mm]
\downarrow
\
s\ll J
[1mm]
\downarrow
\
U_j=Q_jR_j,\quad
\operatorname{rank}U_j\le K-1
[1mm]
\downarrow
\
\text{matrix-free solver}
\end{array}}
]

Причём есть два режима kernel.

**Режим A, ближе к исходному estimator:** adaptive (K)-support Epanechnikov. Он даёт гарантии

[
\operatorname{nnz}W\le JK,
\qquad
\operatorname{rank}U_j\le K-1.
]

**Режим B, максимально вычислительный:** box или небольшое ступенчатое kernel. Оно не уменьшает support сильнее, зато позволяет между outer steps **точно не обновлять веса**, пока сертифицировано отсутствие пересечения thresholds.

---

# 22. Что выдерживает враждебную проверку

Строго доказано:

- (k_j\le1\Rightarrow U_j=I_j=0).
- (\operatorname{rank}U_j\le\min(n_\Phi,k_j-1,d)).
- Два compact kernels с одинаковым support имеют одинаковый pattern нулевых весов.
- Scalar kernel не гарантирует coordinate sparsity (U_j) для плотных данных.
- (K)-bounded neighborhoods дают (\operatorname{nnz}W\le JK) и rank (\le K-1).
- Coverage objective является субмодулярным, greedy даёт (1-1/e).
- Block leverage sampling на фиксированной ridge-задаче позволяет спектрально аппроксимировать Hessian числом центров порядка (d_{\rm eff}\log d/\varepsilon^2).
- Box/step kernel допускает exact weight reuse для пар, сертифицированно не пересекающих thresholds.
- Exact zero-screening и factorized (U_j) не требуют (d\times d)-матриц.

Не доказано без дополнительных предположений:

- что coverage-selected centers сохраняют (\widehat\beta);
- что adaptive (K)-support kernel статистически лучше исходного global-bandwidth kernel;
- что прошлые leverage scores точно оптимальны на следующем outer step;
- что (C_k=o(Jn)) или (s\ll J) для произвольных данных;
- что box kernel сохраняет статистическую скорость исходного ADP.

Поэтому наиболее сильная **точная** оптимизация остаётся

[
\boxed{
\text{exact zero screening} +
\text{bounded sparse neighborhoods} +
\text{factorized }U_j
}
]

а **осознанный выбор центров** я бы делал гибридом

[
\boxed{
\text{support coverage} +
n_{\rm eff} +
\text{projected local variance} +
\text{previous-step leverage}.
}
]

Именно последняя конструкция, на мой взгляд, заслуживает отдельного алгоритма: она использует информацию, которую weight engine уже получил почти бесплатно, чтобы не тратить (O(k_jn_\Phi d)) на заведомо дублирующие или малоинформативные центры.

---

Да. Для нового multi-index тензора логика оптимизации весов переносится почти полностью, причём часть результатов становится естественнее из-за малой размерности (m).

Новый тензор можно записать как

[
\mathbb T_k^2
=============

h_k^{-2}
\left[
\alpha_k^2 I_d +
P_k^\top(\Lambda_k-\alpha_k^2I_m)P_k
\right],
\qquad
P_kP_k^\top=I_m.
]

Это та же матрица, что

[
\boxed{
\mathbb T_k^2
=============

h_k^{-2}
\left[
\alpha_k^2(I_d-P_k^\top P_k) +
P_k^\top\Lambda_kP_k
\right].
}
\tag{1}
]

Именно такая форма используется в multi-index structural adaptation.

## 1. Точный критерий нулевого веса переносится

Положим

[
\Delta_{ij}=X_i-x_j,
\qquad
u_{ij}=P_k\Delta_{ij}\in\mathbb R^m,
\qquad
D_{ij}=|\Delta_{ij}|^2.
]

Тогда

[
q_{ij}
======

|\mathbb T_k\Delta_{ij}|^2
]

и из (1)

[
\boxed{
q_{ij}
======

\frac{
\alpha_k^2
\left(D_{ij}-|u_{ij}|^2\right) +
u_{ij}^\top\Lambda_k u_{ij}
}{h_k^2}.
}
\tag{2}
]

Поскольку (P_kP_k^\top=I_m),

[
D_{ij}-|u_{ij}|^2
=================

|(I-P_k^\top P_k)\Delta_{ij}|^2
\ge0.
]

Если (\Lambda_k\succeq0), оба члена (2) неотрицательны. Поэтому

[
q_{ij}
\ge
\frac{u_{ij}^\top\Lambda_k u_{ij}}{h_k^2}.
]

Для текущего compact kernel

[
K(t)=(1-t^2)_+,
\qquad
w_{ij}=0\iff q_{ij}\ge1,
]

получаем

[
\boxed{
u_{ij}^\top\Lambda_k u_{ij}\ge h_k^2
\quad\Longrightarrow\quad
w_{ij}=0.
}
\tag{3}
]

Это **exact screening**: ни один ненулевой вес не теряется.

---

## 2. Более того, условие (3) оптимально, если известна только (P\Delta)

Это важный результат.

Зафиксируем

[
u=P\Delta.
]

Среди всех (\Delta), имеющих эту проекцию,

[
|\Delta|^2-|u|^2\ge0.
]

Минимум достигается при

[
\Delta=P^\top u.
]

Следовательно,

[
\inf_{\Delta:P\Delta=u}
q(\Delta)
=========

\frac{u^\top\Lambda u}{h^2}.
\tag{4}
]

Поэтому если

[
u^\top\Lambda u<h^2,
]

существует допустимая (\Delta), для которой (w>0).

То есть

[
\boxed{
\text{никакой более сильный exact test, использующий только }P\Delta,
\text{ невозможен}.
}
]

Это полезно: нет смысла искать хитрую формулу только из (m)-мерной EDR-проекции. Для усиления screening нужна дополнительная информация об ортогональной компоненте.

---

# 3. Самое полезное: screening превращается в radius search в (\mathbb R^m)

Положим

[
Z_i=\Lambda_k^{1/2}P_kX_i,
\qquad
z_j=\Lambda_k^{1/2}P_kx_j.
]

Тогда

[
u_{ij}^\top\Lambda_ku_{ij}
==========================

|Z_i-z_j|^2.
]

Из необходимого условия ненулевого веса следует

[
w_{ij}>0
\quad\Longrightarrow\quad
\boxed{
|Z_i-z_j|<h_k.
}
\tag{5}
]

То есть вместо перебора всех точек в (d)-мерном пространстве сначала ищем кандидатов в пространстве размерности

[
m\ll d.
]

Например,

[
d=1000,\qquad m=2,
]

и первая стадия проверки пары использует двумерное расстояние.

Схема:

[
X
\xrightarrow{P_k}
\mathbb R^m
\xrightarrow{\Lambda_k^{1/2}}
Z
\xrightarrow{\text{radius }h_k}
\text{candidate pairs}
\xrightarrow{\text{exact (2)}}
w_{ij}.
]

---

# 4. Кеш (D_{ij}) переносится без изменений

Это, вероятно, лучший вариант при твоих размерах (n,J\sim2000).

Один раз считаем

[
D_{ij}=|X_i-x_j|^2.
]

Если

[
n=J=2000,
]

это (4\cdot10^6) чисел:

- `float32`: (16) MB;
- `float64`: (32) MB.

После этого на каждом outer step полный (d)-мерный (\Delta_{ij}) больше для весов вообще не нужен.

Сначала

[
Z=XP_k^\top
]

за

[
O(ndm).
]

Для каждой пары

[
u_{ij}=Z_i-Z_j
]

и

[
b_{ij}=u_{ij}^\top\Lambda_ku_{ij}
]

стоят (O(m)).

Затем

[
\boxed{
q_{ij}
======

h_k^{-2}
\left[
\alpha_k^2(D_{ij}-|u_{ij}|^2)
+b_{ij}
\right].
}
\tag{6}
]

Итого weight stage после первоначального построения (D):

[
\boxed{
O(ndm+Jnm)
}
]

вместо работы порядка

[
O(Jnd)
]

с исходными координатами на каждой итерации.

При (m=2,d=1000) разница большая.

---

# 5. Если (D) хранить не хочется, previous screening тоже переносится

Сначала применяем (3).

Пусть после projected radius search осталось (C_k) пар.

Полный квадрат расстояния

[
D_{ij}
]

считается только для них.

Стоимость:

[
O((n+J)dm) +
T_{\rm range} +
O(C_kd).
\tag{7}
]

Если

[
C_k\ll Jn,
]

получаем сильное ускорение.

Но здесь есть тот же строгий предел, что раньше:

[
C_k=o(Jn)
]

нельзя гарантировать для произвольных данных. Может существовать выборка, у которой все точки попадают внутрь projected radius.

---

# 6. Иерархический screening тоже сохраняется

Можно добавить несколько направлений из (P^\perp).

Пусть

[
Q_r\in\mathbb R^{r\times d},
\qquad
\operatorname{row}(P)\subseteq\operatorname{row}(Q_r),
]

и

[
P=RQ_r.
]

Определим

[
z=Q_r\Delta.
]

Тогда

[
|\Delta|^2\ge|z|^2
]

и получаем более сильную lower bound

[
\boxed{
q(\Delta)
\ge
\frac{
\alpha^2
\left(
|z|^2-|Rz|^2
\right) +
(Rz)^\top\Lambda(Rz)
}{h^2}.
}
\tag{8}
]

Значит,

[
\boxed{
\alpha^2
(|z|^2-|Rz|^2) +
(Rz)^\top\Lambda(Rz)
\ge h^2
\Rightarrow
w=0.
}
\tag{9}
]

Можно сделать каскад:

[
m
\rightarrow
m+4
\rightarrow
m+16
\rightarrow
d.
]

То есть сначала EDR screening, потом несколько дополнительных координат, и только самые трудные пары доходят до полного расстояния.

---

# 7. Оптимизация (\alpha_k) переносится буквально

Из (2) положим

[
z=\alpha_k^2,
]

[
a_{ij}
======

D_{ij}-|u_{ij}|^2
\ge0,
]

[
b_{ij}
======

u_{ij}^\top\Lambda_ku_{ij}
\ge0.
]

Тогда

[
\boxed{
q_{ij}(z)
=========

\frac{a_{ij}z+b_{ij}}{h_k^2}.
}
\tag{10}
]

Это именно структура, которая нам нужна.

Так как (a_{ij}\ge0),

[
\alpha_1\le\alpha_2
\quad\Longrightarrow\quad
q_{ij}(\alpha_1)
\le
q_{ij}(\alpha_2).
]

Для невозрастающего compact kernel

[
w_{ij}(\alpha_2)
\le w_{ij}(\alpha_1).
]

Следовательно,

[
\boxed{
\operatorname{supp}W(\alpha_2)
\subseteq
\operatorname{supp}W(\alpha_1).
}
\tag{11}
]

Поэтому при ограничении на массу

[
M(\alpha)\ge L
]

выбирать надо

[
\boxed{
\alpha_*
========

\max{\alpha:M(\alpha)\ge L}.
}
\tag{12}
]

Именно максимальное допустимое (\alpha) минимизирует число ненулевых весов.

---

# 8. Есть ещё бесплатное отсечение при подборе (\alpha)

Из (10):

[
b_{ij}\ge h^2
]

означает

[
q_{ij}(\alpha)\ge1
]

**для любого (\alpha\ge0)**.

То есть

[
\boxed{
b_{ij}\ge h^2
\Longrightarrow
w_{ij}(\alpha)=0
\quad\forall\alpha.
}
\tag{13}
]

Такие пары можно полностью удалить ещё **до подбора (\alpha_k)**.

Для остальных, если (a_{ij}>0), support исчезает при

[
a_{ij}\alpha^2+b_{ij}\ge h^2.
]

Breakpoint:

[
\boxed{
\alpha_{ij}^2
=============

\frac{h^2-b_{ij}}{a_{ij}}.
}
\tag{14}
]

То есть весь support как функция (\alpha) меняется только в известных breakpoint-ах.

Это означает, что старую идею exact selection вместо многократного binary search можно перенести полностью.

---

# 9. Подбор (h_k) также не меняется

При фиксированных (\alpha,P,\Lambda) обозначим

[
c_{ij}
======

\alpha^2
(D_{ij}-|u_{ij}|^2) +
u_{ij}^\top\Lambda u_{ij}.
]

Тогда

[
q_{ij}=\frac{c_{ij}}{h^2}.
]

Для текущего

[
K(q)=(1-q^2)_+
]

имеем

[
w_{ij}
======

\left(
1-\frac{c_{ij}^2}{h^4}
\right)_+.
]

Поэтому прежняя breakpoint-логика для exact выбора минимального допустимого (h) остаётся без изменений.

---

# 10. Reuse между итерациями становится особенно интересным

Положим

[
A_k=\mathbb T_k^2.
]

Для одной фиксированной пары

[
q_k=\Delta^\top A_k\Delta.
]

Тогда

[
|q_k-q_{k-1}|
\le
|A_k-A_{k-1}|_2D_{ij}.
]

Пусть

[
\eta_k=|A_k-A_{k-1}|_2.
]

Тогда

[
\boxed{
q_{k-1}-\eta_kD_{ij}\ge1
\Rightarrow
w_{ij}^{(k)}=0.
}
\tag{15}
]

То есть старый zero может быть сертифицирован без вычисления нового (q_{ij}).

И аналогично

[
q_{k-1}+\eta_kD_{ij}<1
]

гарантирует, что пара осталась внутри support.

---

# 11. Для нового тензора (\eta_k) тоже можно считать без (d\times d)

Здесь появляется новая полезная оптимизация.

Запишем

[
A_k
===

a_kI+
P_k^\top C_kP_k,
]

где

[
a_k=\frac{\alpha_k^2}{h_k^2},
]

[
C_k
===

\frac{\Lambda_k-\alpha_k^2I_m}{h_k^2}.
]

Тогда

[
A_k-A_{k-1}
===========

(a_k-a_{k-1})I +
P_k^\top C_kP_k
---------------

P_{k-1}^\top C_{k-1}P_{k-1}.
\tag{16}
]

Низкоранговая часть живёт в пространстве

[
\mathcal S
==========

\operatorname{span}
{\operatorname{row}P_k,
\operatorname{row}P_{k-1}}
]

размерности не более

[
2m.
]

На (\mathcal S^\perp)

[
A_k-A_{k-1}
===========

(a_k-a_{k-1})I.
]

Поэтому для вычисления **точного**

[
\eta_k=|A_k-A_{k-1}|_2
]

не нужна матрица (d\times d).

Достаточно:

1. построить ортонормированный базис (Q) пространства (\mathcal S), (r\le2m);
2. собрать матрицу размера (r\times r);
3. найти её крайние собственные значения.

Получаем

[
\boxed{
\eta_k
======

\max
\left{
|a_k-a_{k-1}|,
,
\left|
Q(A_k-A_{k-1})Q^\top
\right|_2
\right}.
}
\tag{17}
]

При (m=2) это eigendecomposition максимум (4\times4).

Это делает reuse screening практически дешёвым.

---

# 12. Осознанный выбор центров тоже переносится

Теперь дешёвый projected screening даёт нам support graph

[
\mathcal N_j
============

{i:w_{ij}>0}
]

ещё до подсчёта (U_j).

Поэтому все предыдущие идеи сохраняются:

[
k_j=|\mathcal N_j|,
]

[
n_{\rm eff,j}
=============

\frac{(\sum_iw_{ij})^2}{\sum_iw_{ij}^2},
]

coverage выбранных центров и т. д.

Причём для multi-index support можно предварительно искать именно в (m)-мерном пространстве (5), поэтому **center selection теперь потенциально дешевле**, чем в исходной (d)-мерной геометрии.

После окончательного определения support остаётся также точный результат

[
k_j\le1
\quad\Longrightarrow\quad
U_j=I_j=0.
]

Такие центры можно удалить.

---

## Что я бы поменял в weight engine

Для нового тензора схема получается такой:

[
\boxed{
X
\overset{P_k}{\longrightarrow}
Z=XP_k^\top
}
]

[
\boxed{
u_{ij}=Z_i-z_j
}
]

[
\boxed{
b_{ij}=u_{ij}^\top\Lambda_ku_{ij}
}
]

Первый exact screening:

[
\boxed{
b_{ij}\ge h_k^2
\Rightarrow
w_{ij}=0.
}
]

Только surviving pairs идут дальше.

Если (D_{ij}) кеширован:

[
\boxed{
q_{ij}
======

\frac{
\alpha_k^2(D_{ij}-|u_{ij}|^2)+b_{ij}
}{
h_k^2
}.
}
]

После этого:

[
q_{ij}\ge1
\Rightarrow w_{ij}=0,
]

а для оставшихся вычисляется сам (K(q_{ij})).

То есть я бы использовал именно

[
\boxed{
\text{projected radius search}
\rightarrow
\text{exact zero screening}
\rightarrow
\text{cached }D
\rightarrow
\text{точный }q
\rightarrow
\text{sparse neighbor list}.
}
]

Новая форма тензора этому не мешает. Напротив, она явно выделяет (m)-мерный член (P\Delta), который и даёт наиболее сильный возможный cheap exact screening.

Единственные необходимые условия для доказательства: (P_kP_k^\top=I_m), (\Lambda_k\succeq0) и compact-support kernel. Если в новой версии ADE тензор отличается от формулы выше, сам `ADEmain.tex` содержит только `\input sourceB/multiindex` и `\input sourceB/manifold-ade`, а этих двух подключаемых файлов в приложении нет, поэтому для иной формулы нужно проверять разложение заново.
