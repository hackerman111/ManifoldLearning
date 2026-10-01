Сейчас теория выглядит **сильной как теория surrogate-солвера для фиксированных \(g_j\)**, но ещё не как законченная теория замены HPAO в multi-index ADP. Самые серьёзные проблемы не там, где мы раньше думали: главный вопрос не «как лучше искать SVD», а **что именно означает \(\operatorname{rank}(B)<m\) для \(m\)-мерного EDR и когда такая аппроксимация вообще может быть хорошей**.

### Что у нас уже строго есть

Для

\[
F(B)=
\sum_jN_j\|I_j-U_jB^\top g_j\|^2+
\lambda\|B-P\|_F^2
\]

при \(\lambda>0\) есть SPD-оператор

\[
\mathcal L(X)
=

\sum_jN_j(g_jg_j^\top)XH_j+\lambda X,
\qquad H_j=U_j^\top U_j,
\]

и

\[
F(B)-F(B_\star)
=

\langle B-B_\star,\mathcal L(B-B_\star)\rangle.
\]

Для rank-1 направления \(Z=av^\top\) точный лучший scale известен:

\[
\sigma^\star=
\frac{\langle Q,Z\rangle}
{\langle Z,\mathcal L Z\rangle},
\]

а лучший возможный rank-1 gain равен введённой нами conditional spectral norm squared. При точной изотропии \(H_j=\eta_jI\) rank-\(r\) задача глобально решается whitening + truncated SVD. Всё это математически нормально.

Но дальше начинаются существенные пробелы.

1. **Главное внутреннее противоречие pure-\(B\) подхода: \(r<m\) конфликтует с proximal term.** Поскольку \(P P^\top=I_m\), у \(P\) ровно \(m\) singular values, равных единице. Поэтому по Eckart–Young

\[
\boxed{
\min_{\operatorname{rank}(B)\le r}
\|B-P\|_F^2=m-r.
}
\]

Следовательно, для любого rank-\(r\) кандидата

\[
\boxed{
F(B)\ge \lambda(m-r).
}
\]

Это не loose bound, а неизбежный penalty floor. Более того, если текущий basis уже хорош и

\[
F(P)
=

\sum_jN_j\|I_j-U_jP^\top g_j\|^2
<
\lambda(m-r),
\]

то **ни одна** матрица rank \(\le r\) не может даже улучшить objective относительно \(P\). Значит pure truncated-SVD solver потенциально полезен на ранних шагах, но около сходимости структурно начинает конфликтовать с HPAO proximal step.

Есть ещё более сильная формулировка. Для полного minimizer \(B_\star\)

\[
F(B)-F(B_\star)
\ge
\lambda\|B-B_\star\|_F^2.
\]

Поэтому оптимальный rank-\(r\) solver удовлетворяет нижней оценке

\[
\boxed{
F(B_r^\star)-F(B_\star)
\ge
\lambda
\sum_{k>r}\sigma_k(B_\star)^2.
}
\]

Если полный HPAO шаг близок к \(P\), то его singular spectrum как раз **не должен быстро спадать**. Если

\[
\|B_\star-P\|_2\le\delta<1,
\]

то по perturbation singular values

\[
\sigma_k(B_\star)\ge1-\delta,
\]

и потому

\[
\boxed{
F(B_r^\star)-F(B_\star)
\ge
\lambda(m-r)(1-\delta)^2.
}
\]

Это, пожалуй, самая серьёзная теоретическая проблема pure-\(B\) версии.

2. **Rank-\(r\) \(B\) не содержит \(m\)-мерный EDR, если \(r<m\).** Если downstream structural matrix имеет обычный вид

\[
J_B=B^\top S_gB,
\qquad S_g\succeq0,
\]

то

\[
\boxed{
\operatorname{rank}(J_B)
\le
\operatorname{rank}(B)
\le r<m.
}
\]

То есть из самого \(B_r\) невозможно извлечь \(m\) ненулевых EDR-направлений. Текущий код это фактически признаёт: `solve_fixed_coefficients` возвращает rank-deficient \(B\), «not an m-dimensional EDR basis», а затем `solve()` искусственно достраивает найденные правые singular directions направлениями из старого \(P\). :chatgpt-content-reference{index="0"} :chatgpt-content-reference{index="1"}

Это означает, что фактический алгоритм сейчас:

\[
\text{optimize rank-}r\ B
\quad\longrightarrow\quad
\text{throw away its left factor/scales as a basis}
\]

\[
\longrightarrow
\text{keep }r\text{ right directions} +
(m-r)\text{ old directions from }P.
\]

После этого коэффициенты \(g_j\) вообще пересчитываются заново. :chatgpt-content-reference{index="2"}

**Нет теоремы**, связывающей минимум \(F(B_r)\) с качеством этого достроенного \(m\)-мерного basis. Это сейчас крупнейший разрыв между optimizer theory и EDR theory.

3. **Точная изотропная модель почти несовместима с \(p\ll d\).** У тебя

\[
U_j\in\mathbb R^{p\times d},
\qquad
H_j=U_j^\top U_j,
\]

поэтому

\[
\operatorname{rank}(H_j)\le p.
\]

Если \(p<d\), точное равенство

\[
H_j=\eta_jI_d,\qquad \eta_j>0
\]

вообще невозможно.

Более того, если

\[
\eta_j=\frac{\operatorname{tr}H_j}{d},
\]

то

\[
\lambda_{\max}(H_j)
\ge
\frac{\operatorname{tr}H_j}{p}
=

\frac{d}{p}\eta_j.
\]

Отсюда

\[
\boxed{
\|H_j-\eta_jI\|_2
\ge
\eta_j\left(\frac dp-1\right)
}
\]

и одновременно, из наличия нулевых eigenvalues,

\[
\|H_j-\eta_jI\|_2\ge\eta_j.
\]

При \(d/p\gg1\) наша spectral perturbation model может быть **провально плохой именно в operator norm**, даже если \(\eta_j=\operatorname{tr}(H_j)/d\) является лучшей scalar approximation во Frobenius norm.

Это означает, что условие

\[
\rho
=

\sum_jN_j\|g_j\|^2
\|H_j-\eta_jI\|_2
<
\lambda
\]

для нашего multiplicative certificate, вероятно, слишком сильное в высокоразмерном режиме.

Изотропный SVD всё ещё может быть хорошей **эвристической инициализацией**, но пока нельзя строить всю теорию метода вокруг предположения, что \(\theta=\rho/\lambda\ll1\).

4. **Для общего anisotropic случая нет аналога Eckart–Young.** Векторизованный Hessian имеет структуру

\[
\boxed{
K=
\sum_j
N_j
H_j\otimes(g_jg_j^\top) +
\lambda I.
}
\]

Это не обычная left/right separable metric. Поэтому задача

\[
\min_{\operatorname{rank}(B)\le r}
\|B-B_\star\|_{\mathcal L}
\]

является generalized weighted low-rank approximation.

Общая weighted low-rank approximation известна как вычислительно сложная задача: Gillis и Glineur доказали NP-hardness даже для rank-1 в достаточно общей weighted постановке. :chatgpt-content-reference{index="3"} Но отсюда **нельзя автоматически заключать**, что именно наш специальный оператор с левыми rank-1 факторами \(g_jg_j^\top\) NP-hard. Это отдельный открытый вопрос.

Это даже потенциально интересный теоретический результат: либо найти tractable subclass благодаря

\[
\sum_j H_j\otimes g_jg_j^\top,
\]

либо доказать hardness именно для этой структуры. Общая литература по tensor spectral problems показывает, что переход от матричного SVD к bilinear/multilinear spectral optimization часто делает задачу трудной, вплоть до NP-hardness, но прямого reduction для нашего functional у нас пока нет. :chatgpt-content-reference{index="4"}

5. **Наш alternating \(a\leftrightarrow v\) решает conditional задачи точно, но не глобальную spectral задачу.** При фиксированном \(v\)

\[
a\propto G(v)^{-1}Qv,
\]

и при фиксированном \(a\)

\[
v\propto H(a)^{-1}Q^\top a.
\]

Это даёт монотонное улучшение conditional gain, но общий quotient

\[
\frac{(a^\top Qv)^2}
{\sum_jN_j(a^\top g_j)^2(v^\top H_jv)+\lambda}
\]

неконвексен по паре \((a,v)\). Поэтому у нас нет доказательства, что найденная rank-1 component действительно глобально лучшая.

Это согласуется с broader low-rank optimization literature: greedy методы через ведущие singular directions имеют approximation guarantees для определённых convex low-rank задач, но требуют специальных smoothness/approximation условий; они не дают автоматически exact best-rank-\(r\) solution для произвольной weighted metric. :chatgpt-content-reference{index="5"}

6. **Block ALS решает проблему greedy accumulation только частично.** Факторизация

\[
B=XY^\top
\]

даёт две квадратичные subproblem и гарантирует

\[
F_{t+1}\le F_t
\]

при точном решении блоков. Но этого недостаточно для global convergence.

Работы по matrix completion/sensing получают сильные гарантии alternating minimization именно при дополнительных структурных условиях, например sampling/incoherence/RIP и хорошей инициализации. :chatgpt-content-reference{index="6"} Обзор Chi, Lu, Chen подчёркивает ту же картину: хорошие nonconvex low-rank guarantees обычно возникают за счёт специфической статистической/операторной структуры, а не просто из-за factorization. :chatgpt-content-reference{index="7"}

У нас пока отсутствует аналог такой теоремы для

\[
\mathcal L(X)
=

\sum_jN_j(g_jg_j^\top)XH_j+\lambda X.
\]

Нужно исследовать restricted conditioning на low-rank secants, например

\[
\alpha_r\|X\|_F^2
\le
\langle X,\mathcal L(X)\rangle
\le
\beta_r\|X\|_F^2,
\qquad
\operatorname{rank}(X)\le2r,
\]

и понять, достаточно ли отношения \(\beta_r/\alpha_r\) для benign landscape или локальной линейной сходимости.

7. **Во всей нашей оптимизационной теории \(g_j,U_j,I_j\) считаются фиксированными. В ADP они случайные и адаптивные.** Это очень большой statistical gap. \(g_j\) сами оцениваются локально, \(U_j,I_j\) зависят от kernel weights, bandwidth и текущего EDR basis. Поэтому даже идеально решённая fixed-\(g\) low-rank задача не означает consistency всего ADP.

Для сравнения, оригинальная structure-adaptive работа Hristache et al. доказывала \(n^{-1/2}\)-rate для EDR при \(m\le3\) под своими условиями. :chatgpt-content-reference{index="8"} SAMM Dalalyan–Juditsky–Spokoiny получил \(\sqrt n\)-consistency с логарифмическим фактором для \(m^\ast\le4\). :chatgpt-content-reference{index="9"}

Для нашего solver пока нет аналога цепочки

\[
\text{sample errors in }I,U,g
\rightarrow
\text{error of }B_r
\rightarrow
\text{error of returned EDR projector}.
\]

Особенно нужно показать, что low-rank optimization bias меньше статистической ошибки исходного estimator.

8. **Rank \(r\) solver-а не равен structural dimension \(m\), но теория выбора \(r\) пока их опасно сближает.** В SDR structural dimension является параметром модели. В successive direction extraction Yin–Li–Cook, например, отдельно разрабатывается процедура определения dimension, причём их reduction к последовательным single-index задачам требует elliptical predictors. :chatgpt-content-reference{index="10"}

У нас же

\[
m=\dim(\text{EDR})
\]

и

\[
r=\operatorname{rank}(B_{\rm approximation})
\]

это принципиально разные величины.

Критерии

\[
\varepsilon_{\rm sv},
\qquad
\varepsilon_{\rm energy}
\]

выбирают **численный rank аппроксимации**, а не оценивают structural dimension. При anisotropic \(H_j\) даже их точная objective-интерпретация пропадает.

9. **Нет ещё связи objective error с projector error.** Даже если мы докажем

\[
F(B_r)-F(B_\star)\le\varepsilon,
\]

из strong convexity следует только

\[
\|B_r-B_\star\|_F
\le
\sqrt{\varepsilon/\lambda}.
\]

Для EDR нужен результат вида

\[
\|\Pi_r-\Pi_\star\|
\lesssim
\frac{\|J_r-J_\star\|}
{\operatorname{eigengap}(J_\star)}.
\]

То есть потребуется отдельный Davis--Kahan/Wedin слой и нижняя оценка eigengap structural matrix. При \(r<m\) это особенно плохо, потому что \(J(B_r)\) сам rank-deficient, поэтому такой переход нельзя сделать без анализа heuristic completion.

10. **\(\lambda>0\) сейчас является существенным условием теории, а не технической деталью.** В коде разрешён \(\lambda=0\), но тогда \(\mathcal L\) может быть singular: \(H_j\) low rank, а \(g_j\) могут не возбуждать все \(m\) направления. Исчезают uniqueness \(B_\star\), whitening \(G^{-1/2}\), lower curvature bounds и часть certificate. Теория случая \(\lambda=0\) потребует pseudoinverse и явного identifiability condition.

### Что говорит литература именно в пользу нашего направления

Мотивация вычислительно разумна. Современный SMAVE 2026 прямо указывает, что existing localized SDR methods могут иметь как минимум квадратичную по \(n\) стоимость outer iteration, и использует sparse localization + Riemannian stochastic optimization для масштабирования. :chatgpt-content-reference{index="11"} То есть попытка избежать полного дорогого solve актуальна.

Также сама идея «хорошая spectral initialization + nonconvex refinement» очень типична для современной low-rank theory: именно такой двухфазный шаблон подробно обсуждается в обзорах по nonconvex matrix factorization. :chatgpt-content-reference{index="12"} Поэтому наша isotropic SVD initialization + exact anisotropic refinement выглядит методологически здраво, даже если exact-isotropy theorem редко применим буквально.

### Где сейчас самый перспективный теоретический пробел

Я бы не пытался первым делом доказывать ещё более тонкие свойства \(\varepsilon_{\rm sv}\). В чистом `rank(B)` solver сначала нужно ответить на **один фундаментальный вопрос**:

\[
\boxed{
\text{Когда full HPAO minimizer }B_\star
\text{ действительно хорошо аппроксимируется rank }r<m?
}
\]

У нас уже есть необходимое условие через spectral tail:

\[
F(B_r^\star)-F(B_\star)
\ge
\lambda\sum_{k>r}\sigma_k(B_\star)^2.
\]

Если эксперименты покажут, что из-за proximal pull к \(P\)

\[
\sigma_1(B_\star),\ldots,\sigma_m(B_\star)\approx1,
\]

то pure truncated-\(B\) направление имеет фундаментальный потолок и никакой более умный SVD/ALS его не исправит.

Если же окажется, что на ранних outer steps \(B_\star\) действительно имеет резко спадающий spectrum, тогда возникает очень интересная теория **adaptive low-rank early HPAO**: low-rank solver используется только пока spectral tail мал, а затем автоматически переключается на full solver. Это уже согласует нашу математику с главным ограничением rank-\(B\) постановки.
