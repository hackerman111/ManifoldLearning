import numpy as np

import ADP_Data_Gen as data
import ADP_step0 as step0
import ADP_stepk as stepk
import ADP_Runtime as runtime_tools
import ADP_Trace as trace_tools
import Main_ADP as main
import MyADP
from ADP_single_index import ADP_single_index

# Основные настройки теста. Для дальнейших экспериментов редактируй сначала их.
TEST_CONFIG = {
    "n": 1000,  # 220
    "d": 100,  # 5
    "seed": 12,  # 12
    "function": "sin",  # sin
    "data_type": "normal",  # normal
    "noise_std": 0.02,  # 0.02
    "n_J": 700,  # 100
    "n_directions": 100,  # 5
    "n_min": 100,  # 22
    "min_cosine": 0.75,  # 0.75
    "trace_output_dir": "adp_trace_test_outputs",
}


def assert_shape(name, value, expected_shape):
    actual_shape = value.shape
    assert (
        actual_shape == expected_shape
    ), f"{name}: ожидалась форма {expected_shape}, получена {actual_shape}"


def make_test_data(config):
    X, Y, beta, info = data.MakeData(
        n=config["n"],
        d=config["d"],
        function=config["function"],
        data_type=config["data_type"],
        noise_std=config["noise_std"],
        seed=config["seed"],
        return_info=True,
    )
    return X, Y, beta, info


def test_data_generation(config):
    X, Y, beta, info = make_test_data(config)

    assert_shape("X", X, (config["n"], config["d"]))
    assert_shape("Y", Y, (config["n"],))
    assert_shape("beta", beta, (config["d"],))
    assert np.isclose(np.linalg.norm(beta), 1.0), "beta должен быть нормирован"
    assert info["function"] == config["function"]
    assert info["data_type"] == config["data_type"]

    return X, Y, beta


def test_step0(config, X):
    state = step0.PrepareADPInitialState(
        X,
        n_J=config["n_J"],
        n_directions=config["n_directions"],
        n_min=config["n_min"],
        seed=config["seed"],
    )

    assert_shape("J", state["J"], (config["n_J"],))
    assert_shape("x_j", state["x_j"], (config["n_J"], config["d"]))
    assert_shape(
        "directions",
        state["directions"],
        (config["n_J"], config["n_directions"], config["d"]),
    )
    assert_shape("weights", state["weights"], (config["n_J"], config["n"]))
    assert state["h0"] > 0, "h0 должен быть положительным"
    assert state["weights"].sum(axis=1).mean() >= config["n_min"] - 1e-6

    return state


def test_stepk(config, X, Y, state):
    gradients = stepk.EstimateLocalGradients(
        X,
        Y,
        state["x_j"],
        state["h0"],
        weights=state["weights"],
    )

    assert_shape("local_gradients", gradients, (config["n_J"], config["d"]))
    assert np.all(np.isfinite(gradients)), "градиенты должны быть конечными"

    return gradients


def test_main_pipeline(config, X, Y, beta):
    trace = trace_tools.CreateTrace(store_arrays=False)
    runtime_monitor = runtime_tools.CreateRuntimeMonitor(
        enabled=True,
        use_tqdm=False,
        use_rich=False,
        log_runtime=False,
    )
    result = main.RunADP(
        X,
        Y,
        n_J=config["n_J"],
        n_min=config["n_min"],
        n_directions=config["n_directions"],
        seed=config["seed"],
        trace=trace,
        runtime_monitor=runtime_monitor,
    )

    assert_shape("result beta", result["beta"], (config["d"],))
    assert_shape("result weights", result["weights"], (config["n_J"], config["n"]))
    assert_shape(
        "result local_gradients",
        result["local_gradients"],
        (config["n_J"], config["d"]),
    )

    cosine = stepk.CosineSimilarity(result["beta"], beta)
    assert cosine >= config["min_cosine"], (
        f"ADP плохо восстановил направление: cosine={cosine:.3f}, "
        f"порог={config['min_cosine']:.3f}"
    )
    assert "trace" in result, "результат должен содержать trace"
    assert len(result["trace"]["steps"]) >= 5, "trace должен содержать шаги процесса"
    assert any(step["name"] == "final_result" for step in result["trace"]["steps"])

    saved = trace_tools.SaveADPDiagnostics(
        result,
        output_dir=config["trace_output_dir"],
    )
    assert saved["plots"], "должен быть создан хотя бы один график"
    assert "trace_summary" in saved, "должен быть создан CSV summary трассировки"
    assert "runtime" in result, "результат должен содержать runtime"
    assert "total_pipeline" in result["runtime"]["summary"]
    assert result["runtime"]["summary"]["total_pipeline"] >= 0

    return result, cosine


def test_compatibility_facade(config, X, Y):
    result = MyADP.RunADP(
        X,
        Y,
        n_J=min(config["n_J"], 60),
        n_min=min(config["n_min"], 18),
        seed=config["seed"] + 1,
    )

    assert_shape("MyADP beta", result["beta"], (config["d"],))


def test_single_index_class_api():
    config = {
        "n": 180,
        "d": 6,
        "seed": 21,
        "function": "sin",
        "data_type": "normal",
        "noise_std": 0.03,
        "n_J": 70,
        "n_directions": 6,
        "n_min": 18,
        "min_cosine": 0.75,
    }
    X, Y, beta, _ = data.MakeData(
        n=config["n"],
        d=config["d"],
        function=config["function"],
        data_type=config["data_type"],
        noise_std=config["noise_std"],
        seed=config["seed"],
        return_info=True,
    )

    model = ADP_single_index(
        n_J=config["n_J"],
        n_min=config["n_min"],
        n_directions=config["n_directions"],
        seed=config["seed"],
        trace_enabled=True,
    )
    fitted = model.fit(X, Y)

    assert fitted is model
    assert_shape("model beta_", model.beta_, (config["d"],))
    assert model.h0_ > 0
    assert model.result_["trace"]["steps"], "класс должен сохранять trace при trace_enabled=True"

    projected = model.transform(X[:5])
    assert_shape("projected", projected, (5, 1))
    assert np.allclose(projected[:, 0], model.predict_index(X[:5]))

    cosine = model.score_direction(beta)
    assert cosine >= config["min_cosine"], f"class API cosine={cosine:.3f}"

    generated_X, generated_Y, generated_beta = model.make_data(n=40, d=4, seed=3)
    assert_shape("generated_X", generated_X, (40, 4))
    assert_shape("generated_Y", generated_Y, (40,))
    assert_shape("generated_beta", generated_beta, (4,))


def run_all_tests(config=None):
    if config is None:
        config = TEST_CONFIG

    X, Y, beta = test_data_generation(config)
    state = test_step0(config, X)
    test_stepk(config, X, Y, state)
    result, cosine = test_main_pipeline(config, X, Y, beta)
    test_compatibility_facade(config, X, Y)
    test_single_index_class_api()

    return {
        "cosine": cosine,
        "h0": result["h0"],
        "beta_true": beta,
        "beta_hat": result["beta"],
    }


def format_vector(vector, max_items=8):
    vector = np.asarray(vector)

    if vector.size <= 2 * max_items:
        return str(np.round(vector, 4))

    head = np.round(vector[:max_items], 4)
    tail = np.round(vector[-max_items:], 4)
    return f"{head} ... {tail}  shape={vector.shape}"


def main_report():
    summary = run_all_tests(TEST_CONFIG)

    print("ADP pipeline test: OK")
    print(f"cosine(beta_hat, beta_true) = {summary['cosine']:.3f}")
    print(f"h0 = {summary['h0']:.6f}")
    print("beta_true =", format_vector(summary["beta_true"]))
    print("beta_hat  =", format_vector(summary["beta_hat"]))


if __name__ == "__main__":
    main_report()
