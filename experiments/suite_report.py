"""Общий индекс результатов без объединения разных метрик качества."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def summarize_series(path: Path) -> dict[str, object]:
    """Сохранить неудачи в знаменателе; агрегировать только конечные метрики."""
    counts: Counter[str] = Counter()
    stops: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    values: dict[str, list[float]] = {
        key: [] for key in ("quality", "fit_time_sec", "max_stage_traced_peak_mib")
    }
    with (path / "runs.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            counts["n_total"] += 1
            counts[row["status"]] += 1
            for field in ("convergence_pass", "quality_pass", "recovered"):
                counts[field] += row.get(field) == "True"
            stops[row.get("stop_reason") or row["status"]] += 1
            if row.get("error"):
                errors[row["error"]] += 1
            for key, numbers in values.items():
                value = row.get(key)
                if value and math.isfinite(number := float(value)):
                    numbers.append(number)
    manifest = json.loads((path / "series.json").read_text(encoding="utf-8"))
    recovery = manifest["recovery"]
    result: dict[str, object] = {
        "experiment": manifest["experiment"],
        "title": manifest["title"],
        "directory": path.name,
        "quality_metric": recovery["metric"],
        "quality_direction": recovery["direction"],
        "quality_threshold": recovery["threshold"],
        "n_total": counts["n_total"],
        "n_converged": counts["convergence_pass"],
        "n_quality_pass": counts["quality_pass"],
        "n_recovered": counts["recovered"],
        "n_numerical_failure": counts["numerical_failure"],
        "n_nonconverged": counts["nonconverged"],
        "recovery_rate": counts["recovered"] / counts["n_total"],
    }
    for key, numbers in values.items():
        for suffix, quantile in (("q05", 0.05), ("median", 0.5), ("q95", 0.95)):
            result[f"{key}_{suffix}"] = (
                float(np.quantile(numbers, quantile)) if numbers else None
            )
    result["fit_time_total_sec"] = sum(values["fit_time_sec"])
    result["stop_reasons"] = json.dumps(stops, ensure_ascii=False, sort_keys=True)
    result["errors"] = json.dumps(errors, ensure_ascii=False, sort_keys=True)
    return result


def write_suite_report(
    root: Path, rows: list[dict[str, object]], *, mode: str, profile: str
) -> None:
    """Обновлять таблицу и читаемый индекс после каждой завершённой серии."""
    if rows:
        with (root / "overview.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        f"# ADP: {mode} ({profile})",
        "",
        "Восстановление = сходимость и прохождение порога качества. "
        "Численные ошибки остаются в знаменателе долей.",
        "",
        "Квантили в overview.csv описывают распределение результатов сетки. "
        "Сравнение отдельных уровней и интервалы Уилсона "
        "находятся в summary.csv и phase_summary.csv каждой серии.",
        "",
        "Память: tracemalloc внутри fit; это не полный RSS процесса. "
        "Время fit не включает генерацию данных и построение отчётов.",
        "",
        "Метрики: single — abs(cos), больше лучше; multi — trace_score "
        "(среднее cos² главных углов), больше лучше; manifold — средняя ошибка "
        "локального проектора, меньше лучше. Дополнительная multi-метрика "
        "sum(sin²) сохранена в runs.csv. Разные метрики не объединяются.",
        "",
    ]
    if profile == "smoke":
        lines.extend(
            [
                "Smoke проверяет работоспособность; уменьшенные точки не позволяют "
                "оценивать устойчивость восстановления или влияние факторов.",
                "",
            ]
        )
    lines.extend(
        [
            "| Серия | Fits | Сошлось | Качество прошло | Восстановлено | Ошибки |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| [{row['experiment']}]({row['directory']}/summary.md) "
        f"| {row['n_total']} | {row['n_converged']} | {row['n_quality_pass']} "
        f"| {row['n_recovered']} | {row['n_numerical_failure']} |"
        for row in rows
    )
    lines.extend(["", "## Причины остановок и ошибок", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['experiment']}: {row['title']}",
                "",
                f"Порог: {row['quality_metric']} {row['quality_direction']} "
                f"{row['quality_threshold']}.",
                "",
                f"Остановки: `{row['stop_reasons']}`.",
                "",
            ]
        )
        if row["errors"] != "{}":
            lines.extend([f"Ошибки: `{row['errors']}`.", ""])
    (root / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
