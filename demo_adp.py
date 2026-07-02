from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from adp import ADP, ADPConfig, ADPData, ADPResult


VariantName = Literal["new", "old"]


@dataclass(slots=True)
class DemoSettings:
    """Настройки строгой демонстрационной проверки.

    Вход:
        Значения полей объекта.
    Выход:
        Объект с размерами данных, числом повторов и порогами приемки.
    """

    n: int = 520
    d: int = 10
    n_centers: int = 110
    n_directions: int = 14
    min_neighbors: float = 14.0
    outer_steps: int = 4
    inner_steps: int = 10
    repeats: int = 3
    noise: float = 0.04
    corr: float = 0.30
    link: str = "sin"
    min_cosine_abs: float = 0.80
    min_mean_cosine_abs: float = 0.88
    max_mean_angle_deg: float = 32.0
    data_seed: int = 1_000
    model_seed: int = 10_000
    output_dir: Path = field(default_factory=lambda: Path("demo_outputs"))


@dataclass(slots=True)
class RunRecord:
    """Итог одного запуска одного варианта ADP.

    Вход:
        Поля объекта после обучения.
    Выход:
        Запись с метриками и локальным вердиктом.
    """

    variant: str
    repeat: int
    cosine_abs: float
    angle_deg: float
    objective: float
    elapsed_sec: float
    history_steps: int
    weights_mean: float
    passed: bool


@dataclass(slots=True)
class VariantSummary:
    """Сводный итог по одному варианту ADP.

    Вход:
        Поля объекта после объединения повторов.
    Выход:
        Сводка с минимальным качеством, средним качеством и вердиктом.
    """

    variant: str
    repeat_count: int
    passed_count: int
    min_cosine_abs: float
    mean_cosine_abs: float
    max_angle_deg: float
    mean_angle_deg: float
    passed: bool


def make_demo_config(
    settings: DemoSettings,  # Настройки демонстрационной проверки.
    random_state: int,  # Начальное число для воспроизводимости.
) -> ADPConfig:
    """Создает конфигурацию ADP для строгой проверки.

    Вход:
        settings: размеры данных и параметры обучения.
        random_state: начальное число генератора модели.
    Выход:
        ADPConfig для new или old варианта.
    """

    return ADPConfig(
        n_centers=settings.n_centers,
        n_directions=settings.n_directions,
        min_neighbors=settings.min_neighbors,
        outer_steps=settings.outer_steps,
        inner_steps=settings.inner_steps,
        show_progress=False,
        random_state=random_state,
    )


def generate_demo_data(
    settings: DemoSettings,  # Настройки демонстрационной проверки.
    repeat: int,  # Номер повтора.
) -> ADPData:
    """Генерирует один большой набор данных для проверки.

    Вход:
        settings: размеры данных и параметры шума.
        repeat: номер повтора.
    Выход:
        ADPData с X, y, истинным beta и центрами.
    """

    generator = ADP.create("new", make_demo_config(settings, settings.data_seed + repeat))
    return generator.generate_data(
        n=settings.n,
        d=settings.d,
        n_centers=settings.n_centers,
        n_directions=settings.n_directions,
        noise=settings.noise,
        corr=settings.corr,
        link=settings.link,
    )


def fit_variant(
    variant: VariantName,  # Вариант ADP: new или old.
    data: ADPData,  # Данные текущего повтора.
    settings: DemoSettings,  # Настройки демонстрационной проверки.
    repeat: int,  # Номер повтора.
    random_state: int,  # Начальное число модели.
    save_plots: bool,  # Сохранять ли графики для этого запуска.
) -> RunRecord:
    """Обучает один вариант ADP и строит запись проверки.

    Вход:
        variant: имя варианта ADP.
        data: данные с известным истинным beta.
        settings: пороги и параметры обучения.
        repeat: номер повтора.
        random_state: начальное число модели.
        save_plots: признак сохранения графиков.
    Выход:
        RunRecord с метриками и вердиктом по одному запуску.
    """

    model = ADP.create(variant, make_demo_config(settings, random_state))
    started = time.perf_counter()
    result = model.fit(data.X, data.y, centers=data.centers)
    elapsed_sec = time.perf_counter() - started
    metrics = model.score(data.beta)

    record = make_run_record(variant, repeat, result, metrics, elapsed_sec, settings)
    if save_plots:
        plot_dir = settings.output_dir / "plots" / variant / f"repeat_{repeat}"
        model.save_diagnostics(plot_dir, beta_true=data.beta, prefix=variant)
    return record


def make_run_record(
    variant: str,  # Имя варианта ADP.
    repeat: int,  # Номер повтора.
    result: ADPResult,  # Результат обучения.
    metrics: dict[str, float],  # Метрики восстановления beta.
    elapsed_sec: float,  # Время обучения.
    settings: DemoSettings,  # Пороги приемки.
) -> RunRecord:
    """Преобразует результат обучения в запись проверки.

    Вход:
        variant: имя варианта ADP.
        repeat: номер повтора.
        result: результат обучения.
        metrics: словарь метрик.
        elapsed_sec: время обучения.
        settings: пороги приемки.
    Выход:
        RunRecord с локальным вердиктом.
    """

    cosine_abs = float(metrics["cosine_abs"])
    angle_deg = float(metrics["angle_deg"])
    objective = float(result.objective)
    passed = bool(
        np.isfinite(cosine_abs)
        and np.isfinite(angle_deg)
        and np.isfinite(objective)
        and cosine_abs >= settings.min_cosine_abs
        and len(result.history) == settings.outer_steps * settings.inner_steps
    )
    return RunRecord(
        variant=variant,
        repeat=repeat,
        cosine_abs=cosine_abs,
        angle_deg=angle_deg,
        objective=objective,
        elapsed_sec=float(elapsed_sec),
        history_steps=len(result.history),
        weights_mean=float(result.statistics.weights_mean),
        passed=passed,
    )


def summarize_variant(
    variant: str,  # Имя варианта ADP.
    records: list[RunRecord],  # Записи всех или одного варианта.
    settings: DemoSettings,  # Пороги приемки.
) -> VariantSummary:
    """Строит итог по варианту и применяет общие пороги.

    Вход:
        variant: имя варианта ADP.
        records: записи запусков.
        settings: пороги приемки.
    Выход:
        VariantSummary с итоговым вердиктом.
    """

    selected = [record for record in records if record.variant == variant]
    if not selected:
        return VariantSummary(variant, 0, 0, np.nan, np.nan, np.nan, np.nan, False)

    cosine_values = np.array([record.cosine_abs for record in selected], dtype=float)
    angle_values = np.array([record.angle_deg for record in selected], dtype=float)
    passed_count = sum(record.passed for record in selected)
    min_cosine_abs = float(np.min(cosine_values))
    mean_cosine_abs = float(np.mean(cosine_values))
    max_angle_deg = float(np.max(angle_values))
    mean_angle_deg = float(np.mean(angle_values))
    passed = bool(
        passed_count == len(selected)
        and min_cosine_abs >= settings.min_cosine_abs
        and mean_cosine_abs >= settings.min_mean_cosine_abs
        and mean_angle_deg <= settings.max_mean_angle_deg
    )
    return VariantSummary(
        variant=variant,
        repeat_count=len(selected),
        passed_count=passed_count,
        min_cosine_abs=min_cosine_abs,
        mean_cosine_abs=mean_cosine_abs,
        max_angle_deg=max_angle_deg,
        mean_angle_deg=mean_angle_deg,
        passed=passed,
    )


def overall_verdict(
    summaries: list[VariantSummary],  # Сводки по вариантам ADP.
) -> bool:
    """Проверяет общий вердикт по всем вариантам.

    Вход:
        summaries: сводки по вариантам.
    Выход:
        Логическое значение: да, если каждый вариант прошел проверку.
    """

    return bool(summaries) and all(summary.passed for summary in summaries)


def save_tables(
    records: list[RunRecord],  # Записи отдельных запусков.
    summaries: list[VariantSummary],  # Сводки по вариантам.
    output_dir: Path,  # Каталог для файлов.
) -> dict[str, Path]:
    """Сохраняет таблицы с полными результатами проверки.

    Вход:
        records: записи отдельных запусков.
        summaries: сводки по вариантам.
        output_dir: каталог сохранения.
    Выход:
        Словарь с путями к сохраненным таблицам.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "verdict_records.csv"
    summary_path = output_dir / "verdict_summary.csv"

    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "repeat",
                "cosine_abs",
                "angle_deg",
                "objective",
                "elapsed_sec",
                "history_steps",
                "weights_mean",
                "passed",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.variant,
                    record.repeat,
                    f"{record.cosine_abs:.8f}",
                    f"{record.angle_deg:.8f}",
                    f"{record.objective:.8f}",
                    f"{record.elapsed_sec:.8f}",
                    record.history_steps,
                    f"{record.weights_mean:.8f}",
                    int(record.passed),
                ]
            )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "repeat_count",
                "passed_count",
                "min_cosine_abs",
                "mean_cosine_abs",
                "max_angle_deg",
                "mean_angle_deg",
                "passed",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.variant,
                    summary.repeat_count,
                    summary.passed_count,
                    f"{summary.min_cosine_abs:.8f}",
                    f"{summary.mean_cosine_abs:.8f}",
                    f"{summary.max_angle_deg:.8f}",
                    f"{summary.mean_angle_deg:.8f}",
                    int(summary.passed),
                ]
            )

    return {"records": records_path, "summary": summary_path}


def print_header(
    settings: DemoSettings,  # Настройки демонстрационной проверки.
) -> None:
    """Печатает размеры данных и правила приемки.

    Вход:
        settings: настройки проверки.
    Выход:
        None; текст выводится в терминал.
    """

    print("Строгая проверка ADP")
    print(f"  данные: n={settings.n}, d={settings.d}, повторы={settings.repeats}, связь={settings.link}")
    print(f"  обучение: J={settings.n_centers}, P={settings.n_directions}, внешние шаги={settings.outer_steps}, внутренние шаги={settings.inner_steps}")
    print(f"  правило запуска: |cos(beta, beta_hat)| >= {settings.min_cosine_abs:.2f}")
    print(f"  правило варианта: средний |cos| >= {settings.min_mean_cosine_abs:.2f}, средний угол <= {settings.max_mean_angle_deg:.1f}")
    print("  истинный beta используется только для оценки качества, не как стартовая точка")


def print_record(
    record: RunRecord,  # Запись одного запуска.
) -> None:
    """Печатает строку результата одного запуска.

    Вход:
        record: запись одного запуска.
    Выход:
        None; текст выводится в терминал.
    """

    mark = "прошел" if record.passed else "провал"
    print(
        f"  {record.variant:>3} повтор {record.repeat}: "
        f"|cos|={record.cosine_abs:.4f}, угол={record.angle_deg:5.2f}, "
        f"цель={record.objective:.6f}, время={record.elapsed_sec:.2f} c, {mark}"
    )


def print_summary(
    summaries: list[VariantSummary],  # Сводки по вариантам.
    saved_paths: dict[str, Path],  # Пути к сохраненным таблицам.
) -> None:
    """Печатает итоговые сводки и расположение файлов.

    Вход:
        summaries: сводки по вариантам.
        saved_paths: пути к таблицам.
    Выход:
        None; текст выводится в терминал.
    """

    print("\nСводка")
    for summary in summaries:
        mark = "прошел" if summary.passed else "провал"
        print(
            f"  {summary.variant:>3}: "
            f"запусков {summary.passed_count}/{summary.repeat_count}, "
            f"минимальный |cos|={summary.min_cosine_abs:.4f}, "
            f"средний |cos|={summary.mean_cosine_abs:.4f}, "
            f"средний угол={summary.mean_angle_deg:.2f}, {mark}"
        )
    print(f"\nТаблица запусков: {saved_paths['records']}")
    print(f"Таблица сводки: {saved_paths['summary']}")
    print("Графики первого повтора: demo_outputs/plots/")


def main() -> int:
    """Запускает строгую демонстрационную проверку.

    Вход:
        Нет явных аргументов.
    Выход:
        0, если оба варианта прошли проверку, иначе 1.
    """

    settings = DemoSettings()
    variants: tuple[VariantName, ...] = ("new", "old")
    records: list[RunRecord] = []

    print_header(settings)
    for repeat in range(settings.repeats):
        data = generate_demo_data(settings, repeat)
        print(f"\nПовтор {repeat}: X={data.X.shape}, y={data.y.shape}, centers={data.centers.shape}")
        for variant_index, variant in enumerate(variants):
            record = fit_variant(
                variant=variant,
                data=data,
                settings=settings,
                repeat=repeat,
                random_state=settings.model_seed + repeat * 10 + variant_index,
                save_plots=repeat == 0,
            )
            records.append(record)
            print_record(record)

    summaries = [summarize_variant(variant, records, settings) for variant in variants]
    saved_paths = save_tables(records, summaries, settings.output_dir)
    print_summary(summaries, saved_paths)

    if overall_verdict(summaries):
        print("\nИТОГ: МЕТОД РАБОТАЕТ ПО ЗАДАННЫМ ПРАВИЛАМ.")
        return 0
    print("\nИТОГ: МЕТОД НЕ ПРОШЕЛ ПРОВЕРКУ ПО ЗАДАННЫМ ПРАВИЛАМ.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
