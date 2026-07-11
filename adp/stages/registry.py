from __future__ import annotations

from collections.abc import Iterable

from .contracts import StageContext, StageFactory


STAGE_METHODS: dict[str, tuple[str, ...]] = {
    "beta_initializer": ("initialize",),
    "center_selector": ("select",),
    "bandwidth_selector": ("select_initial", "select_anisotropy"),
    "direction_sampler": ("prepare",),
    "statistics_builder": ("compute",),
    "local_solver": ("solve",),
    "beta_solver": ("solve",),
    "stop_rule": ("should_stop",),
}

DEFAULT_STAGE_NAMES: dict[str, str] = {
    "beta_initializer": "default",
    "center_selector": "random_sample",
    "bandwidth_selector": "adaptive_mass",
    "direction_sampler": "random_sphere",
    "statistics_builder": "random_projection",
    "local_solver": "least_squares",
    "beta_solver": "cg",
    "stop_rule": "convergence",
}


def _deferred_builtin_factory(category: str) -> StageFactory:
    def factory(context: StageContext):
        from .builtins import build_builtin_stage

        return build_builtin_stage(category, context)

    return factory


class StageRegistry:
    """Изолированный реестр именованных фабрик этапов ADP."""

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, StageFactory]] = {
            category: {} for category in STAGE_METHODS
        }

    @classmethod
    def with_defaults(cls) -> "StageRegistry":
        registry = cls()
        for category, name in DEFAULT_STAGE_NAMES.items():
            registry.register(category, name, _deferred_builtin_factory(category))
        return registry

    def copy(self) -> "StageRegistry":
        copied = StageRegistry()
        copied._factories = {
            category: dict(factories)
            for category, factories in self._factories.items()
        }
        return copied

    def register(
        self,
        category: str,
        name: str,
        factory: StageFactory,
        *,
        replace: bool = False,
    ) -> None:
        self._validate_category(category)
        if not name:
            raise ValueError("Имя реализации этапа не должно быть пустым")
        if not callable(factory):
            raise TypeError("Фабрика этапа должна быть callable")
        if name in self._factories[category] and not replace:
            raise ValueError(
                f"Реализация {name!r} уже зарегистрирована для {category!r}"
            )
        self._factories[category][name] = factory

    def resolve(self, category: str, name: str) -> StageFactory:
        self._validate_category(category)
        try:
            return self._factories[category][name]
        except KeyError as exc:
            available = ", ".join(self.available(category)) or "нет"
            raise ValueError(
                f"Неизвестная реализация {name!r} для этапа {category!r}; "
                f"доступны: {available}"
            ) from exc

    def available(self, category: str) -> tuple[str, ...]:
        self._validate_category(category)
        return tuple(sorted(self._factories[category]))

    @property
    def categories(self) -> Iterable[str]:
        return tuple(STAGE_METHODS)

    def _validate_category(self, category: str) -> None:
        if category not in STAGE_METHODS:
            available = ", ".join(STAGE_METHODS)
            raise ValueError(
                f"Неизвестный этап {category!r}; доступны: {available}"
            )

