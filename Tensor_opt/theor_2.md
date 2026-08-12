Восстановленный ответ: exact screening для manifold-ADP

Да. Для manifold-версии точное предварительное отсечение переносится на локальные анизотропные тензоры с локальным (P_j) вместо одного глобального (P). Более того, в manifold возникают два независимых exact-screening слоя:

веса (w_{ij}), связывающие центр (x_j) с наблюдениями (X_i);

веса (\delta_{j\alpha}), связывающие локальные EDR-пространства соседних центров.

1. Первый уровень: веса (w_{ij})

Для каждого центра (j) используем

h_k^{-2}\alpha_k^2(I-P_j^\top P_j)+h_k^{-2}J_j,]

где

[J_j=P_j^\top\Lambda_jP_j,\qquadP_jP_j^\top=I_m,\qquad\Lambda_j\succeq0.]

Пусть

[\Delta_{ij}=X_i-x_j.]

Тогда

|T_j^{(k)}\Delta_{ij}|^2

\Delta_{ij}^\top(T_j^{(k)})^2\Delta_{ij}.]

Так как (P_j^\top P_j) есть ортогональный проектор,

|\Delta|^2-|P_j\Delta|^2\ge0.]

Поэтому

h_k^{-2}\alpha_k^2|(I-P_j^\top P_j)\Delta_{ij}|^2+h_k^{-2}(P_j\Delta_{ij})^\top\Lambda_j(P_j\Delta_{ij}),]

и, следовательно,

[q_{ij}\geh_k^{-2}(P_j\Delta_{ij})^\top\Lambda_j(P_j\Delta_{ij}).]

Если ядро имеет компактный носитель

[K(q)=0,\qquad q\ge1,]

то

[\boxed{(P_j(X_i-x_j))^\top\Lambda_j(P_j(X_i-x_j))\ge h_k^2\Longrightarroww_{ij}=0.}]

Это exact screening: ни один ненулевой вес не удаляется.

Для доказательства не нужны ни гладкость manifold, ни приближение (P_i\approx P_j).

2. Почему локальные (P_j) не ломают доказательство

В global multi-index используется одна матрица (P). В manifold она заменяется семейством (P_j).

Доказательство выше поточечное и использует только

[P_jP_j^\top=I_m,\qquad\Lambda_j\succeq0,\qquad\alpha_k^2\ge0.]

Равенство (P_j=P_\alpha) не требуется. Поэтому global screening превращается в семейство независимых local screenings.

3. Второй exact-screening слой

Manifold-алгоритм использует веса между центрами

K!\left(|T_{M,\alpha}(x_j-x_\alpha)|^2\right).]

Пусть

h_{M,k}^{-2}\alpha_{M,k}^2(I-P_\alpha^\top P_\alpha)+h_{M,k}^{-2}P_\alpha^\top\Lambda_\alpha P_\alpha.]

Для

[\Delta_{j\alpha}=x_j-x_\alpha]

получаем

h_{M,k}^{-2}\alpha_{M,k}^2|(I-P_\alpha^\top P_\alpha)\Delta_{j\alpha}|^2+h_{M,k}^{-2}(P_\alpha\Delta_{j\alpha})^\top\Lambda_\alpha(P_\alpha\Delta_{j\alpha}).]

Следовательно,

[\boxed{(P_\alpha(x_j-x_\alpha))^\top\Lambda_\alpha(P_\alpha(x_j-x_\alpha))\ge h_{M,k}^2\Longrightarrow\delta_{j\alpha}=0.}]

Это важно, поскольку без разреживания manifold penalty потенциально содержит (J^2) связей.

4. Двухуровневая разреженность

Определим два графа:

[G_X={(j,i):w_{ij}>0},]

[G_M={(\alpha,j):\delta_{j\alpha}>0}.]

Первый граф связывает центры с наблюдениями. Второй связывает локальные EDR-пространства.

После exact screening:

ADP-статистики считаются только по рёбрам (G_X);

manifold penalty считается только по рёбрам (G_M).

То есть manifold-ADP естественно сводится к двум разреженным графам.

5. Почему одной корректности screening недостаточно для ускорения

В global multi-index можно один раз вычислить

[Z=XP^\top\in\mathbb R^{n\times m}]

за

[O(ndm),]

а затем использовать

[P(X_i-x_j)=Z_i-Z_j.]

В manifold матрица (P_j) зависит от центра. Наивный расчёт всех (P_jX_i) требует

[O(Jndm),]

то есть сам screening может стать настолько же дорогим, как исходный расчёт расстояний.

Поэтому строго корректно утверждать:

exact screening переносится на manifold без дополнительных предположений, но ускорение требует дополнительной низкоранговой структуры семейства ({P_j}).

6. Общий manifold-superspace

Предположим, что существует пространство размерности (m_s\ll d)

[P_\in\mathbb R^{m_s\times d},\qquadP_P_*^\top=I_{m_s},]

такое, что

[\operatorname{row}(P_j)\subseteq\operatorname{row}(P_*)\qquad\forall j.]

Тогда существует

[R_j\in\mathbb R^{m\times m_s}]

с

[P_j=R_jP_*.]

Один раз вычисляем

[Z_i=P_*X_i\in\mathbb R^{m_s}]

за

[O(ndm_s).]

Для центров аналогично

[z_j=P_*x_j.]

Тогда

R_j(Z_i-z_j).]

Проверка одной пары выполняется в размерности (m_s), а не (d).

7. Теорема о точном низкоразмерном manifold screening

Предположим:

(P_jP_j^\top=I_m);

(\Lambda_j\succeq0);

(K(q)=0) при (q\ge1);

существует (P_\in\mathbb R^{m_s\times d}), (m_s\ll d), такое что (P_j=R_jP_).

Определим

[Z_i=P_X_i,\qquadz_j=P_x_j.]

Тогда

[\boxed{[R_j(Z_i-z_j)]^\top\Lambda_j[R_j(Z_i-z_j)]\ge h_k^2\Longrightarroww_{ij}=0.}]

Доказательство:

R_jP_*(X_i-x_j)

R_j(Z_i-z_j).]

А из разложения (q_{ij}) следует

[q_{ij}\geh_k^{-2}[R_j(Z_i-z_j)]^\top\Lambda_j[R_j(Z_i-z_j)].]

Если числитель не меньше (h_k^2), то (q_{ij}\ge1), следовательно (K(q_{ij})=0).

8. Сложность

Без screening:

O(Jnd)+O(J^2d).]

Пусть после screening остаётся

[C_X=|G_X|,\qquadC_M=|G_M|.]

При общем пространстве размерности (m_s):

O((n+J)dm_s),]

O(Jnm,m_s+J^2m,m_s),]

а полные (d)-мерные вычисления нужны только для surviving pairs:

O((C_X+C_M)d).]

Итого

O!\left((n+J)dm_s+(Jn+J^2)m,m_s+(C_X+C_M)d\right).}]

Сравнение идёт с

[O((Jn+J^2)d).]

Строгое ускорение получается в режиме

[m,m_s\ll d,\qquadC_X\ll Jn,\qquadC_M\ll J^2.]

Например,

[d=1000,\qquad m=2,\qquad m_s=5.]

Projected screening использует порядка

[m,m_s=10]

координатных операций на пару вместо порядка (1000).

9. Блочный superspace вместо глобального

Одно общее (P_*) не обязательно.

Разобьём центры на группы

[G_1,\ldots,G_s]

и для каждой группы определим точное объединённое пространство

\operatorname{span}{\operatorname{row}(P_j)\in G_g}.]

Пусть

[r_g=\dim\mathcal S_g.]

Выбираем ортонормированную матрицу

[Q_g\in\mathbb R^{r_g\times d}]

с

[\operatorname{row}(P_j)\subseteq\operatorname{row}(Q_g)\qquad(j\in G_g).]

Тогда

[P_j=R_jQ_g]

точно и

R_j[Q_gX_i-Q_gx_j].]

Это даёт exact blockwise low-rank screening без приближения (P_j\approx P_{j'}).

Всегда

[m\le r_g\le\min(d,m|G_g|).]

Если (r_g\to d), вычислительное преимущество исчезает.

10. Роль гладкости manifold

Гладкость локальных пространств не нужна для корректности screening.

Она может быть полезна для объяснения, почему для близких центров пространство

[\operatorname{span}{P_j\in G_g}]

имеет небольшую размерность (r_g).

Поэтому надо разделять два утверждения:

[P_jP_j^\top=I,\quad \Lambda_j\succeq0\Longrightarrow\text{exact screening},]

а

[\text{малый общий или локальный span}\Longrightarrow\text{дешёвый screening}.]

11. Самый сильный восстановленный результат

Для manifold-ADP exact screening переносится на оба слоя:

[\boxed{(P_j\Delta_{ij})^\top\Lambda_j(P_j\Delta_{ij})\ge h_k^2\Longrightarroww_{ij}=0,}]

[\boxed{(P_\alpha\Delta_{j\alpha})^\top\Lambda_\alpha(P_\alpha\Delta_{j\alpha})\ge h_{M,k}^2\Longrightarrow\delta_{j\alpha}=0.}]

Эти условия не меняют ни одного ненулевого веса.

Без дополнительной low-rank структуры семейства ({P_j}) нельзя доказать ускорение относительно (O(Jnd)). Если же локальные пространства лежат в общем или блочном пространстве малой размерности, projected screening можно выполнять в этой малой размерности, а дорогие (d)-мерные вычисления делать только для surviving pairs.

Практическая схема:

[\boxed{\text{двухуровневый sparse graph}+\text{exact screening}+\text{общий или блочный low-rank span локальных }P_j.}]
