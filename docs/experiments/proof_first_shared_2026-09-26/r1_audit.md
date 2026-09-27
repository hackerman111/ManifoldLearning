# R1: независимый аудит H1

Read-only аудит `h1_proof.md` и live source locators из `r0.md` выполнен
отдельным агентом до прототипа. Проверены QR/Haar marginal law, четвёртые
моменты, дисперсия при `P<d` и `P=kd+r`, агрегирование одного `Φ_j` по
всем manifold targets, normal-system perturbation и projector eigengap.
В этих формулах ошибки не найдены.

Исправления после аудита:

1. Multi local refit допускает minimum-norm решение при потере rank;
   manifold требует полный rank `m`. `h1_proof.md` теперь разделяет случаи.
2. Perturbation/projector bound относится к фиксированной quadratic
   correction, не ко всему adaptive HPAO endpoint с trust-radius,
   `λ`, gauge-fix, refit и acceptance. Эти границы явно разделены.
3. В manifold `sync_steps` повторно используют один `Φ`; условная на
   текущих проекторах variance formula применяется к первому sync
   step и первому шагу каждого нового масштаба, а не к дальнейшим
   sync steps на том же скетче.
4. `P≥d` может требовать `d×d` temporaries и `U_j` на каждом центре;
   `P=d=1000`, batch32,n10000 даёт 2.56 GB statistics scratch сверх
   3.2 GB `U/Φ`. Большой полный блок не проходит memory gate.
5. Сжатие полных блоков в `sqrt(k)I_d` точно сохраняет Gram и
   квадратичную цель, но меняет row-wise `eta`, rank cutoffs и
   потенциальный HYBRID backend. Поэтому оно допускается лишь как
   изолированная NUMERICAL-репрезентация H1 с reference/rank проверками,
   а не как автоматически эквивалентный float64 fit.

**Gate:** R2 допускается для H1 на изотропных направлениях с независимым
draw и bounded `P,d,J`. Теорема гарантирует pointwise variance reduction
и условную устойчивость одного шага при перечисленных предпосылках.
Качество полного fit и фактическая стоимость остаются экспериментальным
вопросом. Анизотропный multi, fixed-sketch reuse и высокоразмерный
`P≥d` исключены из текущего прототипа.
