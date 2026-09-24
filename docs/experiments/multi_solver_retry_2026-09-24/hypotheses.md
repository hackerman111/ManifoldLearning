# Повторный поиск: гипотезы, зафиксированные до solver trials

Selection повторно используется для разработки; held-out не читали.

|ID/status|Наблюдение и механизм|Альтернатива|Прогноз/опровержение|Бюджет и proof|
|---|---|---|---|---|
|H6/refuted by frozen gate|R1: normal reject устраняется точностью при той же lambda. Certificate-directed atol и refinement уберут ложные trust updates. Existing opt-in HYBRID с relative_tol=0.01, без dense/SVD и screening workspace.|Точность дороже экономии; AO всё равно медленно/другой basin.|На6d100 normal rejects уменьшатся>=80%, certificate>=5/6 приcap80, objectives не хуже reference; общий gate solver ratio<=0.70 и d10<=1.10.|Proof/audit до patch; 12frozen×baseline/H6,60s/trial. Настройки фиксированы, без tolerance sweep.|
|H7/proof admitted, testing|R1: reduced худшие точки имеют положительную кривизну; простая first-order оптимизация выбирает иные basins, freezing C замедляет AO. Полный reduced residual Jacobian учитывает изменение local refit и может дать лучшее направление.|GN также попадает в худший basin или Jacobian solves слишком дороги.|Заранее требуется >=11/12 certificates, все objectives в gate и d100ratio<=0.70; иначе reject.|Только при неудаче H6 и после отдельного proof/audit; до12frozen trials×60s, baseline R3 уже измерен. Не более двух кандидатов.|

H6 result: h6.md. H7 proof/audit completed in proof_reduced_gn.md before implementation; cap160, no warm start, lambda floor=lambda_prox, eta0.1, trust radius0.25sqrt(m), Armijo1e-4. Frozen budget12×60s.
