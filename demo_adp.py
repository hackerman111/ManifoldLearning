from __future__ import annotations

from pathlib import Path
from typing import Literal

from adp import ADP, ADPConfig, ADPData, ADPResult


def make_demo_config(
    random_state: int,  # Начальное число для воспроизводимости.
) -> ADPConfig:
    """Создает легкую конфигурацию для демонстрации.

    Вход:
        random_state: начальное число генератора.
    Выход:
        ADPConfig с малым числом шагов, чтобы пример запускался быстро.
    """

    return ADPConfig(
        n_centers=28,
        n_directions=6,
        min_neighbors=7,
        outer_steps=2,
        inner_steps=5,
        show_progress=False,
        random_state=random_state,
    )


def fit_variant(
    variant: Literal["new", "old"],  # Вариант ADP: new или old.
    data: ADPData,  # Общие данные для сравнения вариантов.
    random_state: int,  # Начальное число модели.
) -> tuple[ADPResult, dict[str, float]]:
    """Обучает один вариант ADP и считает качество восстановления beta.

    Вход:
        variant: имя варианта ADP.
        data: сгенерированные данные.
        random_state: начальное число модели.
    Выход:
        Кортеж из результата обучения и словаря метрик.
    """

    model = ADP.create(variant, make_demo_config(random_state))
    result = model.fit(data.X, data.y, centers=data.centers, beta0=data.beta)
    metrics = model.score(data.beta)
    model.save_diagnostics(Path("demo_outputs") / variant, beta_true=data.beta, prefix=variant)
    return result, metrics


def print_variant_report(
    variant: str,  # Имя варианта ADP.
    result: ADPResult,  # Результат обучения.
    metrics: dict[str, float],  # Метрики восстановления beta.
) -> None:
    """Печатает короткий отчет по одному варианту.

    Вход:
        variant: имя варианта ADP.
        result: результат обучения.
        metrics: словарь метрик.
    Выход:
        None; отчет выводится в терминал.
    """

    print(f"\nВариант {variant}")
    print(f"  вычислитель: {result.backend}")
    print(f"  цель: {result.objective:.6f}")
    print(f"  |cos(beta, beta_hat)|: {metrics['cosine_abs']:.4f}")
    print(f"  угол, градусы: {metrics['angle_deg']:.2f}")
    print(f"  шагов в истории: {len(result.history)}")
    print(f"  средняя локальная масса: {result.statistics.weights_mean:.2f}")
    print(f"  графики: demo_outputs/{variant}/")


def main() -> None:
    """Запускает полный демонстрационный пример.

    Вход:
        Нет явных аргументов.
    Выход:
        None; печатает отчет и сохраняет графики в demo_outputs.
    """

    generator = ADP.create("new", make_demo_config(random_state=1))
    data = generator.generate_data(n=160, d=6, noise=0.03, link="linear")

    print("Демонстрация ADP")
    print(f"  X: {data.X.shape}")
    print(f"  y: {data.y.shape}")
    print(f"  centers: {data.centers.shape}")
    print(f"  функция связи: {data.link_name}")

    for variant, random_state in (("new", 2), ("old", 3)):
        result, metrics = fit_variant(variant, data, random_state)
        print_variant_report(variant, result, metrics)

    print("\nПроверка завершена: оба варианта обучились и сохранили диагностику.")


if __name__ == "__main__":
    main()
