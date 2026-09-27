# SIR+SAVE как начальное подпространство

Класс изменения: ESTIMATOR. Это условное обоснование initializer, а не
теорема о конечном ADP fit. Дата фиксации: 2026-09-27, до selection.

## Population target

Пусть X имеет невырожденное Gaussian распределение, Z — центрированный
whitened X, Cov(Z)=I, и Y независимо от Z условно на V.T Z, где V.T V=I_m.
В частности, это выполнено для Y=g(B.T X)+epsilon с epsilon, независимым от X.
Обозначим T=V.T Z, P=VV.T, Q=I-P. Gaussian разложение Z=VT+QZ имеет
QZ, независимый от (T,Y), с нулевым средним и ковариацией Q.

Для любой фиксированной нарезки отклика H=s(Y):

    mu_h = E[Z|H=h] = V E[T|H=h],
    C_h = Cov(Z|H=h) = Q + V Cov(T|H=h) V.T,
    I-C_h = V (I_m-Cov(T|H=h)) V.T.

Следовательно, для положительных вероятностей p_h оператор

    M = sum_h p_h [mu_h mu_h.T + (I-C_h)^2]
      = V K V.T,  K >= 0,

имеет range(M) в span(V). Равенство подпространств требует rank(K)=m.
Этот coverage assumption обязателен: ни SIR, ни SAVE, ни их сумма не
гарантируют обнаружение всех nonlinear links. При even single link
mu_h может быть нулём, но SAVE видит изменение условной дисперсии.
Коэффициенты обоих слагаемых фиксированы равными 1, число slices — 10.

## Выборочная формула и whitening

Для Xc=(n,d), n>d, 1<=m<=d-2, n>=20, считаем pivoted thin QR:
Xc[:,piv]=Qx R. Требование m+1<d позволяет eigsh вычислить дополнительную
eigenpair для gap без dense fallback; в каждом из 10 slices хотя бы 2 строки.
Проверяем
полный численный ранг R через singular values с cutoff
eps*max(n,d)*s_max. Z=sqrt(n)*Qx удовлетворяет Z.T Z/n=I.
Сортируем Y stable-sort и режем на 10 равных по числу наблюдений частей;
sample covariance делим на n_h (не n_h-1). При связках Y это соглашение
зависит от исходного порядка: непрерывные ответы в frozen study ties не
создают. Константный Y отклоняется до вычислений.

Пусть Vhat — top m eigenvectors sample M. В исходных координатах

    Braw[piv,:] = solve_triangular(R, Vhat),
    Bhat = thin_QR(Braw).

Действительно Xc Braw=Qx Vhat=Z Vhat/sqrt(n), поэтому это то же
предикторное подпространство. Возврат Vhat без R solve ошибочен для
коррелированных X. Никакой explicit inverse не строится. Финальный QR
меняет только координаты внутри subspace, знак столбцов канонизируется.

## Matrix-free operator, сертификаты и perturbation scope

Внутри slice храним D_h=Z_h-mu_h. Для произвольного v:

    C_h v = D_h.T (D_h v) / n_h,
    Mv = sum_h p_h [mu_h (mu_h.T v) + (v-C_h v)-C_h(v-C_h v)].

Это буквально symmetric PSD dense reference выше: adjoint совпадает с
forward. eigsh вычисляет m+1 старших eigenpairs, затем проверяются
finite values, eigen residual и separation lambda_m-lambda_(m+1).
Почти нулевой eigengap отклоняется, не скрывается случайным дополнением.
Сертификат относится к sample operator, не к population recovery.

Для двух операторов, выраженных в одной системе координат, если
population eigengap delta=lambda_m(M)>0, sample perturbation
||Mhat-M||_2=e<delta/2, стандартное invariant-subspace perturbation
рассуждение даёт ||sin Theta||_2 <= e/(delta-e) <= 2e/delta.
Это абстрактная perturbation lemma: empirical QR coordinates нельзя
отождествлять с population whitening. Их согласование и ошибка whitening
должны быть включены в e. Здесь нет доказанной конечновыборочной оценки e:
sample whitening,
empirical quantile slices и conditioning X требуют дополнительного анализа.
Back-transform дополнительно чувствителен к conditioning whitening.
Population формулы выше применяются к фиксированным slices; перенос на
empirical quantile slicing не объявляется конечновыборочной теоремой.

## Ресурсы и ограничения

Thin QR требует O(n*d^2) времени и O(n*d+d^2) памяти при n>d. Один Mv:
O(n*d+H*d); iterative eigen workspace O(d*max(20,2*(m+1)+1)). Нет H*d*d
covariances, dense normal matrix, full inverse или (J,n,d).
При n=10000,d=1000 один n*d float64 буфер =80 MB; d*d R=8 MB.
QR/copies дают несколько таких буферов; фактический process RSS измеряется
отдельно. Это не обещание дешёвого старта при n≈d или d=1000.

Отклоняем n<=d, rank deficiency, non-finite/complex data, constant Y,
невозможные slices/index_dim, незавершённый eigen solve и малый eigengap.
Не переносим Gaussian theorem на uniform/student inputs. Никакого
утверждения об улучшении адаптивной ADP траектории: для этого R3/R4.

## Проверяемые источники

- [Glaws, Constantine, Cook, Inverse regression for ridge recovery](https://arxiv.org/html/1702.02227):
  SIR/SAVE moment ranges, coverage limitations и влияние distribution assumptions.
- [Pautrel, Portier, SMAVE](https://arxiv.org/html/2606.00413v1):
  отдельный MAVE estimator с projected localization; его stationary convergence
  не является теоремой о recovery current ADP.

Формулы QR, matrix-free action и conditional Gaussian decomposition выше
выведены для этого прототипа. Они будут проверены independent dense
reference и invariance checks до full fits.

## Независимый аудит R1

Read-only GPT-6 Luna проверил Gaussian decomposition, QR back-transform,
PSD/action и scope. Алгебраических ошибок не найдено; уточнены независимость
шума от X, domain eigsh, размеры slices и common-coordinate смысл
perturbation bound. Аудит не подтверждает качество полного восстановления.
