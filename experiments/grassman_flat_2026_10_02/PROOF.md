# Совместные углы одного Grassmann flat

Источник: `SVD/glm.tex`, prop:flatfit и sec:alg. Это отдельная схема,
не полный tangent GN из предыдущего исследования.

## Геометрия и точный кэш

Y=(d,m), YᵀY=I, V=(d,q), VᵀV=I, YᵀV=0; строки Aᵀ=(q,m)
ортонормированы. После горизонтализации/QR левых SVD-векторов небольшой
повторный SVD VᵀG восстанавливает парные плоскости и их знак.

```
Y(theta) = Y + [YA(cos(theta)-1) + V sin(theta)] Aᵀ
M(theta) = M + [MA(cos(theta)-1) + W sin(theta)] Aᵀ
M=UY, W=UV, D_l=-MA_l sin(theta_l)+W_l cos(theta_l)
T_l=partial_l M(theta)=D_l a_lᵀ
```

Действия U требуются для M/W и внешнего градиента. Inner loop использует
только M, MA, W и малые матрицы. Y(theta) ортонормирован математически;
его сборка в ambient пространстве нужна только после принятия flat.
В подпространстве каждый угол периодичен с периодом pi. Реализация ищет
локально в [-max_angle,max_angle]^q, не обещает глобальный минимум тора.

## Полная производная профиля

Для каждого центра: H=M(theta)ᵀM(theta)+tau I,
Hg=M(theta)ᵀI, r=I-M(theta)g. В безопасной full-rank области:

```
H dg_l = a_l (D_lᵀr) - (M(theta)ᵀD_l)(a_lᵀg)
dr_l = -D_l(a_lᵀg) - M(theta)dg_l
packed residual = sqrt(mass)*[r; -sqrt(tau)g]
Jac_l = sqrt(mass)*[dr_l; -sqrt(tau)dg_l]
grad_theta Phi = 2 Jacᵀ packed_residual
```

Формула получается дифференцированием Hg=MᵀI. Solve H выполняется
через существующие QR/SVD-факторы `_Profile.normal_solve`; normal matrix
и inverse не строятся. Все q столбцов сохраняют cross terms в GN solve;
theta обновляется одним вектором, последовательно углы не оптимизируются.
При theta=0 производная равна `<G, v_l a_lᵀ>`.

GN решает augmented LS `[Jac; sqrt(lambda)I] step = [-residual; 0]`,
lambda — numerical damping, профиль не меняется. Step ограничивается
так, чтобы весь вектор theta оставался в box. Истинный профиль refit-ится
при каждом trial; Armijo с полным angular gradient обеспечивает убывание.
Граница box может иметь ненулевой полный угловой градиент; inner stop
не является stationarity certificate исходной Grassmann-задачи.

При rank/conditioning boundary производная не применяется: value-only
rank_one fallback. При превышении workspace — spectral. При невозможности
принять первый GN step — spectral. Итоговый общий градиент и residual
пересчитываются через live U, сохраняя внешний сертификат.

## Стоимость и проверяемые границы

Кэш O(J*p*(m+q)); Jacobian O(J*(p+m)*q), dg O(J*m*q),
малый augmented solve имеет q unknowns вместо q*m у core_gn.
Каждая inner evaluation — сборка p*m локального design и local QR/SVD;
стоимость не зависит от d после выбора flat. Workspace проверяется до
Jacobian. Нет d*d projector/Hessian и полного tangent Jacobian.

Классы: identities EXACT/NUMERICAL; ограниченная joint angle оптимизация
APPROXIMATE, явный `method="flat_gn"`. Objective и tau сохраняются.
Численные проверки должны отдельно подтвердить ambient Exp/refit,
ненулевой q>1 Jacobian (включая ridge/mass), gauge/pi invariance,
rank/workspace fallback и отсутствие ambient actions внутри inner loop.
Преимущество по времени/восстановлению — гипотеза для Luna benchmark.
