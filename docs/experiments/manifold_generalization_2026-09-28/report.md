# Manifold в более общем случае: m=1,2,3

Эксперимент написан и выполнен. При текущих фиксированных конфигурациях
устойчивое восстановление меняющейся геометрии при m>1 не подтверждено.
На плоских контролях успехи есть, но доля падает с ростом m. Это вывод
для заданного генератора и параметров, не универсальная невозможность.

## Что проверялось

[Протокол](protocol.md): n=600,d=8, случайное ортогональное вращение,
m=1/2/3, кривизна c=0/0.35/0.8, шум 0/0.1, два режима local/broad,
5 seeds. Truth=row(Dz) нелинейного rank-m отображения.
Используется существующий estimator=manifold, solver=cg.
Recovery: RMS и максимум principal sine <=0.2 на всех 40 центрах
и 512 независимых queries, без фильтрации плохих точек.
Residual/stop reason — только debug, не критерий восстановления.

## Основная серия с scale_boundary=raise

Все 180 fits выполнены за суммарные 74.70 s fit/evaluation wall time.
0 recovery; 180 exceptions: 149 недостижимых function mass targets,
24 rank failures initial EDR (12 для m=2 и 12 для m=3), 7 local slope rank
failures. Начальная геометрия сохранена там, где инициализация состоялась.
Начальное подпространство не считается результатом прерванного fit.
Данные: [manifest](main/manifest.json), [runs](main/runs.jsonl),
[summary](main/summary.json).

## Диагностическая серия с scale_boundary=stop

Это отдельное exploratory сравнение на тех же 180 профилях после
наблюдения mass failures. Опция существовала до эксперимента; defaults
и алгоритм не изменены. Серия не является untouched validation.
Все 180 fits выполнены за суммарные 94.83 s.
148 fits завершились на function_mass_boundary; 31 rank failure
и одна ошибка feasible scale bracket.
28/180 полного восстановления (и по центрам, и по queries).
Максимальный linear relative residual среди завершённых fits
9.99984e-7: сертификация inner solve не означает правильную геометрию.

В таблице каждый профиль объединяет два уровня шума и пять seeds (10 fits).
Ошибки остаются в знаменателе recovery; медианы геометрии вычислены
только для завершённых fits. Центров/углов внутри метрик не исключали.
Полная разбивка по шуму находится в summary.json.

| support | m | c | recovery / 10 | errors | center RMS | center max | query max | oracle query max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local | 1 | 0 | 7/10 | 3 | 0.0061 | 0.0251 | 0.0061 | 0.0000 |
| local | 1 | 0.35 | 0/10 | 2 | 0.2980 | 0.7466 | 0.7792 | 0.6730 |
| local | 1 | 0.8 | 0/10 | 2 | 0.4116 | 0.9107 | 0.9976 | 0.9834 |
| local | 2 | 0 | 4/10 | 4 | 0.0429 | 0.1073 | 0.0625 | 0.0000 |
| local | 2 | 0.35 | 0/10 | 4 | 0.3162 | 0.8073 | 0.8237 | 0.7232 |
| local | 2 | 0.8 | 0/10 | 4 | 0.6036 | 0.9998 | 1.0000 | 0.9956 |
| local | 3 | 0 | 2/10 | 4 | 0.0668 | 0.3614 | 0.1250 | 0.0000 |
| local | 3 | 0.35 | 0/10 | 4 | 0.3324 | 0.7520 | 0.8471 | 0.7600 |
| local | 3 | 0.8 | 0/10 | 4 | 0.6402 | 0.9999 | 1.0000 | 0.9931 |
| broad | 1 | 0 | 10/10 | 0 | 0.0163 | 0.0190 | 0.0187 | 0.0000 |
| broad | 1 | 0.35 | 0/10 | 0 | 0.2967 | 0.6507 | 0.7627 | 0.6636 |
| broad | 1 | 0.8 | 0/10 | 0 | 0.5254 | 0.8856 | 0.9509 | 0.9819 |
| broad | 2 | 0 | 5/10 | 1 | 0.0776 | 0.1087 | 0.1087 | 0.0000 |
| broad | 2 | 0.35 | 0/10 | 0 | 0.5206 | 0.8910 | 0.9478 | 0.7019 |
| broad | 2 | 0.8 | 0/10 | 0 | 0.6880 | 0.9993 | 1.0000 | 0.9950 |
| broad | 3 | 0 | 0/10 | 0 | 0.5822 | 0.9646 | 0.9646 | 0.0000 |
| broad | 3 | 0.35 | 0/10 | 0 | 0.5416 | 0.9033 | 0.9844 | 0.7600 |
| broad | 3 | 0.8 | 0/10 | 0 | 0.7153 | 1.0000 | 1.0000 | 0.9884 |

Данные: [manifest](main_stop/manifest.json), [runs](main_stop/runs.jsonl),
[summary](main_stop/summary.json). Trace и source hashes сохранены.
RSS измерен cumulative Linux process high-water mark; не per-fit peak.

## Диагноз и границы

- Плоский контроль: m=1 — 17/20, m=2 — 9/20, m=3 — 2/20.
  Это уже исключает уверенное утверждение о стабильной работе при росте m.
- Меняющаяся геометрия m>1: 0/80; по всем m при c>0 — 0/120.
  Center errors тоже выше порога: отрицательный исход не обусловлен
  исключительно переносом nearest-chart на queries.
- Local support для части seeds вырождается: minimal graph degree и
  mass/n_eff дают конкретную диагностику, все исключения записаны.
  Broad support убирает большинство rank failures, но ухудшает локальность;
  увеличение средней массы не является гарантией восстановления.
- Oracle nearest-chart max при c>0 также не проходит 0.2 ни в одном
  завершённом fit: даже точные bases в 40 центрах не обеспечивают выбранную
  строгую query-проверку. Нужна отдельно проверяемая плотность покрытия
  или интерполяция charts, а не ослабление порога после исхода.
- Truth — заданное local EDR-распределение. Единственный scalar Y не
  определяет произвольное меняющееся rank-m распределение однозначно.
  Gaussian queries — конечная независимая проверка, не доказательство
  равномерного восстановления на всей непрерывной области.

## Воспроизведение

Из корня репозитория (output directory должен быть новым):

```bash
UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.manifold_generalization --self-check
UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.manifold_generalization --profile main --out /tmp/manifold-generalization-raise
UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.manifold_generalization --profile main --scale-boundary stop --out /tmp/manifold-generalization-stop
```

Проверки: Ruff и Pyright прошли; final smoke — 36/36 записанных fits.
Self-check прошёл. Полный pytest не запускался: production-код не менялся.

self-check проверяет gradient через независимые finite differences,
принадлежность gradient row(Dz), basis rotation, principal sine против
малого dense projector reference, deterministic data и независимость
noise от X/truth/queries/model seed.

Артефакты main/ и main_stop/ содержат hashes именно исходников на момент
запуска. Финальные правки типов/расположения center_indices и записи summary
пустого incomplete-run не меняют
численного пути; финальный smoke сохранён отдельно в smoke_final/.
Два ранних smoke preserved: [smoke](smoke/manifest.json) и
[smoke_support](smoke_support/manifest.json); причины изменения поддержки
до main описаны в protocol.md. Ни один прошлый результат не перезаписан.

Следующий исследовательский шаг — отдельный протокол поддержки/плотности
charts и анализа локальной идентифицируемости при m>1. Этот эксперимент
не меняет defaults, не исправляет estimator и не объявляет stop-вариант
универсальным решением.
