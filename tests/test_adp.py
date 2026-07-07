from pathlib import Path

import numpy as np
import pytest

import adp.core as adp_core
from adp import ADP, ADPConfig

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    matplotlib = None


def test_factory_generates_single_index_data():
    model = ADP.create("new", random_state=1)

    data = model.generate_data(n=80, d=5, n_centers=12, n_directions=4, noise=0.01)

    assert data.X.shape == (80, 5)
    assert data.y.shape == (80,)
    assert data.beta.shape == (5,)
    assert data.centers.shape == (12, 5)
    assert data.directions is not None
    assert data.directions.shape == (12, 4, 5)
    assert np.isclose(np.linalg.norm(data.beta), 1.0)


def test_new_variant_fits_beta_with_random_projection_statistics():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=36,
            n_directions=8,
            min_neighbors=8,
            outer_steps=2,
            inner_steps=6,
            backend="numpy",
            show_progress=False,
            random_state=2,
        ),
    )
    data = model.generate_data(n=180, d=6, noise=0.03, link="linear")

    result = model.fit(data.X, data.y, beta0=data.beta)
    metrics = model.score(data.beta)

    assert result.beta.shape == (6,)
    assert result.history[-1].objective <= result.history[0].objective
    assert metrics["cosine_abs"] > 0.85
    assert result.statistics.directions is not None


def test_factory_rejects_removed_old_variant():
    with pytest.raises(ValueError, match="только 'new'"):
        ADP.create("old", ADPConfig(show_progress=False))


def test_backend_is_numpy_only():
    with pytest.raises(ValueError, match="Only numpy"):
        ADP.create("new", ADPConfig(backend="gpu"))

    package_text = "\n".join(path.read_text() for path in Path("adp").rglob("*.py"))
    assert "torch" not in package_text.lower()
    assert "cupy" not in package_text.lower()


def test_numpy_backend_smoke_uses_public_api():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=12,
            n_directions=4,
            min_neighbors=5,
            outer_steps=1,
            inner_steps=3,
            show_progress=False,
            random_state=4,
        ),
    )
    data = model.generate_data(n=60, d=4, noise=0.01, link="linear")

    result = model.fit(data.X, data.y, beta0=data.beta)

    assert result.backend == "numpy"
    assert np.isfinite(result.objective)
    assert np.isclose(np.linalg.norm(result.beta), 1.0)


def test_progress_output_exposes_informative_training_state(monkeypatch):
    captured = []

    class FakeTqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.kwargs = kwargs
            captured.append(("init", kwargs))

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, values, refresh=True):
            captured.append(("postfix", dict(values), refresh))

        def close(self):
            captured.append(("close",))

    monkeypatch.setattr(adp_core, "tqdm", FakeTqdm)
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=14,
            n_directions=4,
            min_neighbors=5,
            outer_steps=2,
            inner_steps=3,
            backend="numpy",
            show_progress=True,
            random_state=5,
        ),
    )
    data = model.generate_data(n=70, d=4, noise=0.01, link="linear")

    result = model.fit(data.X, data.y, beta0=data.beta)

    postfix_values = [item[1] for item in captured if item[0] == "postfix"]
    assert postfix_values
    assert result.progress
    assert result.progress[-1]["objective"] == result.objective
    for key in ("variant", "backend", "outer", "inner", "h", "weights", "objective", "delta", "elapsed"):
        assert key in postfix_values[-1]


def test_information_methods_save_plots_to_files(tmp_path):
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=16,
            n_directions=4,
            min_neighbors=5,
            outer_steps=2,
            inner_steps=4,
            backend="numpy",
            show_progress=False,
            random_state=6,
        ),
    )
    data = model.generate_data(n=90, d=4, noise=0.01, link="linear")
    model.fit(data.X, data.y, beta0=data.beta)

    history_path = tmp_path / "history.png"
    ax = model.plot_history(save_path=history_path)
    assert ax is not None
    assert history_path.exists()
    assert history_path.stat().st_size > 0

    saved = model.save_diagnostics(tmp_path / "diagnostics", beta_true=data.beta)
    expected_names = {"objective", "delta", "bandwidth", "weights", "beta_compare"}
    assert expected_names.issubset(saved)
    for path in saved.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    summary = model.summary()
    expected_summary = {"history": str(history_path)}
    expected_summary.update({name: str(path) for name, path in saved.items()})
    assert summary["diagnostic_plots"] == expected_summary
