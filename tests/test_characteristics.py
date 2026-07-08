import json

import numpy as np

from adp.characteristics import ADP_Characteristics


def test_characteristics_records_steps_and_scalar_values():
    characteristics = ADP_Characteristics()

    beta = np.array([1.0, 0.0])
    beta_hat_step_0 = np.array([0.0, 1.0])
    beta_hat_step_k = np.array([1.0, 1.0])

    characteristics.H0_Save(1.5)
    characteristics.H_k_Save(0.75)
    characteristics.Rho_k_Save(0.25)
    characteristics.Step_0_Characteristics_Save(beta, beta_hat_step_0, 1.5)
    characteristics.Step_k_Characteristics_Save(beta, beta_hat_step_k, 0.25, 1.5, 0.75)

    data = characteristics.Characteristics_Get()

    assert data["h_0_values"] == [1.5]
    assert data["h_k_values"] == [0.75]
    assert data["rho_k_values"] == [0.25]
    assert data["step_records"][0]["step_name"] == "step_0"
    assert data["step_records"][0]["cosine"] == 0.0
    assert data["step_records"][1]["step_name"] == "step_k"
    assert data["step_records"][1]["step"] == 1
    assert np.isclose(data["step_records"][1]["cosine"], np.sqrt(0.5))


def test_cosine_returns_none_for_missing_or_degenerate_beta():
    characteristics = ADP_Characteristics()

    assert characteristics.Cosine_Calculate(None, np.array([1.0, 0.0])) is None
    assert characteristics.Cosine_Calculate(np.array([1.0, 0.0]), None) is None
    assert characteristics.Cosine_Calculate(np.zeros(2), np.array([1.0, 0.0])) is None
    assert characteristics.Cosine_Calculate(np.array([1.0, 0.0]), np.zeros(2)) is None


def test_run_save_creates_folder_tables_and_graphics(tmp_path):
    characteristics = ADP_Characteristics()

    beta = np.array([1.0, 0.0])
    characteristics.H0_Save(1.2)
    characteristics.Step_0_Characteristics_Save(beta, np.array([0.8, 0.6]), 1.2)
    characteristics.H_k_Save(0.6)
    characteristics.Rho_k_Save(0.35)
    characteristics.Step_k_Characteristics_Save(beta, np.array([0.9, 0.1]), 0.35, 1.2, 0.6)

    run_folder = characteristics.Run_Save(
        tmp_path,
        run_name="test_run",
        use_latex=False,
    )

    assert run_folder == tmp_path / "test_run"
    assert (run_folder / "characteristics.csv").exists()
    assert (run_folder / "characteristics.json").exists()
    assert (run_folder / "plots" / "cosine.png").exists()
    assert (run_folder / "plots" / "rho_k.png").exists()
    assert (run_folder / "plots" / "h_0.png").exists()
    assert (run_folder / "plots" / "h_k.png").exists()
    assert (run_folder / "plots" / "all_characteristics.png").exists()

    saved = json.loads((run_folder / "characteristics.json").read_text(encoding="utf-8"))

    assert saved["step_records"][0]["step_name"] == "step_0"
    assert saved["step_records"][1]["rho_k"] == 0.35


def test_plot_axis_labels_are_horizontal_and_latex_can_be_enabled(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.figure

    captured_labels = []
    original_savefig = matplotlib.figure.Figure.savefig

    def capture_savefig(self, *args, **kwargs):
        for axis in self.axes:
            captured_labels.append(axis.yaxis.label.get_rotation())

        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_savefig)

    characteristics = ADP_Characteristics()
    beta = np.array([1.0, 0.0])

    characteristics.Step_0_Characteristics_Save(beta, np.array([1.0, 0.0]), 1.0)
    characteristics.Step_k_Characteristics_Save(beta, np.array([0.8, 0.6]), 0.4, 1.0, 0.5)

    characteristics.Run_Save(
        tmp_path,
        run_name="latex_style",
        use_latex=True,
    )

    assert matplotlib.rcParams["text.usetex"] is True
    assert captured_labels
    assert all(rotation == 0.0 for rotation in captured_labels)
