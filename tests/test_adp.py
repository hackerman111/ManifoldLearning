from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_data_generator_corr_controls_pairwise_feature_correlation():
    corr = 0.65
    sigma_x = 1.7
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=1,
            n_directions=1,
            show_progress=False,
            random_state=101,
        ),
    )

    X = np.asarray(
        model.generate_data(n=20_000, d=6, corr=corr, sigma_x=sigma_x).X
    )
    correlation = np.corrcoef(X, rowvar=False)
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]

    assert np.allclose(off_diagonal, corr, atol=0.03)
    assert np.allclose(X.var(axis=0), sigma_x**2, rtol=0.05)
    assert np.all(np.abs(X.mean(axis=0)) < 0.05 * sigma_x)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_centers", 0),
        ("n_centers", -1),
        ("n_directions", 0),
        ("n_directions", -1),
    ],
)
def test_data_generator_rejects_nonpositive_count_overrides(name, value):
    model = ADP.create(
        "new",
        ADPConfig(show_progress=False, random_state=102),
    )

    with pytest.raises(ValueError, match=rf"{name} должен быть положительным"):
        model.generate_data(n=20, d=3, **{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_centers", 0),
        ("n_centers", -1),
        ("n_directions", 0),
        ("n_directions", -1),
    ],
)
def test_config_rejects_nonpositive_data_generation_counts(name, value):
    with pytest.raises(ValueError, match=rf"{name} должен быть положительным"):
        ADPConfig(**{name: value})


@pytest.mark.parametrize("kernel", ("gausian", None, []))
def test_config_rejects_unknown_kernel(kernel):
    with pytest.raises(ValueError, match="kernel"):
        ADPConfig(kernel=kernel)


@pytest.mark.parametrize("chunk_size", (0, -1, True, 1.5))
def test_config_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size должен быть положительным целым"):
        ADPConfig(chunk_size=chunk_size)


@pytest.mark.parametrize(
    ("converter", "value", "name"),
    (
        (adp_core.as_2d_float, [[0.0, np.nan]], "X"),
        (adp_core.as_2d_float, [[0.0, np.inf]], "centers"),
        (adp_core.as_1d_float, [0.0, -np.inf], "y"),
    ),
)
def test_public_array_converters_reject_nonfinite_values(converter, value, name):
    with pytest.raises(ValueError, match=rf"{name}.*конечные"):
        converter(np.asarray(value), name)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("noise", np.nan, "noise должен быть конечным и неотрицательным"),
        ("noise", np.inf, "noise должен быть конечным и неотрицательным"),
        ("noise", -0.1, "noise должен быть конечным и неотрицательным"),
        ("sigma_x", np.nan, "sigma_x должен быть конечным и положительным"),
        ("sigma_x", np.inf, "sigma_x должен быть конечным и положительным"),
        ("sigma_x", 0.0, "sigma_x должен быть конечным и положительным"),
        ("sigma_x", -1.0, "sigma_x должен быть конечным и положительным"),
    ],
)
def test_data_generator_rejects_invalid_scales(name, value, message):
    model = ADP.create(
        "new",
        ADPConfig(show_progress=False, random_state=103),
    )

    with pytest.raises(ValueError, match=message):
        model.generate_data(n=20, d=3, **{name: value})


def test_data_generator_rejects_beta_with_wrong_dimension():
    model = ADP.create(
        "new",
        ADPConfig(show_progress=False, random_state=104),
    )

    with pytest.raises(ValueError, match="beta должен быть одномерным вектором длины 3"):
        model.generate_data(n=20, d=3, beta=np.ones(4))


def test_data_generator_rejects_nonfinite_beta():
    model = ADP.create(
        "new",
        ADPConfig(show_progress=False, random_state=105),
    )

    with pytest.raises(ValueError, match="beta должен содержать только конечные значения"):
        model.generate_data(n=20, d=3, beta=np.array([1.0, np.nan, 0.0]))


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
    assert result.statistics.directions is None
    assert result.statistics.n_directions == 8
    assert model.directions_ is None


def test_save_directions_flag_keeps_directions_in_result_and_model():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=10,
            n_directions=3,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=2,
            save_directions=True,
            show_progress=False,
            random_state=13,
        ),
    )
    data = model.generate_data(n=50, d=4, noise=0.01, link="linear")

    result = model.fit(data.X, data.y, centers=data.centers, directions=data.directions)

    assert result.statistics.directions is not None
    assert result.statistics.directions.shape == (10, 3, 4)
    assert model.directions_ is not None
    assert model.directions_.shape == (10, 3, 4)


def test_factory_rejects_removed_old_variant():
    with pytest.raises(ValueError, match="только 'new'"):
        ADP.create("old", ADPConfig(show_progress=False))


def test_cupy_backend_requires_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)

    with pytest.raises(ImportError, match="CuPy backend"):
        ADP.create("new", ADPConfig(backend="cupy"))

    package_text = "\n".join(path.read_text() for path in Path("adp").rglob("*.py"))
    assert "torch" not in package_text.lower()


def install_fake_cupy(monkeypatch):
    calls = []

    def record(name, fn):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return fn(*args, **kwargs)

        return wrapped

    fake = SimpleNamespace(
        asarray=record("asarray", np.asarray),
        asnumpy=record("asnumpy", np.asarray),
        exp=record("exp", np.exp),
        maximum=record("maximum", np.maximum),
        einsum=record("einsum", np.einsum),
        matmul=record("matmul", np.matmul),
        swapaxes=record("swapaxes", np.swapaxes),
        finfo=np.finfo,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return calls


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
