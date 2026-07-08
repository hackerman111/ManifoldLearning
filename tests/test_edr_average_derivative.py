import numpy as np
import pytest
from scipy import sparse
from scipy.spatial import cKDTree

from adp.edr import EDR
from adp.edr_adp_single_index import EDR_ADP_single_index


class SpyCharacteristics:
    def __init__(self):
        self.h_0_values = []
        self.h_k_values = []
        self.rho_k_values = []
        self.step_0_values = []
        self.step_k_values = []

    def H0_Save(self, h_0):
        self.h_0_values.append(h_0)

    def H_k_Save(self, h_k):
        self.h_k_values.append(h_k)

    def Rho_k_Save(self, rho_k):
        self.rho_k_values.append(rho_k)

    def Step_0_Characteristics_Save(self, beta, beta_hat, h_0):
        self.step_0_values.append((beta, beta_hat, h_0))

    def Step_k_Characteristics_Save(self, beta, beta_hat, rho_k, h_0, h_k):
        self.step_k_values.append((beta, beta_hat, rho_k, h_0, h_k))


def configure_small_model(**kwargs):
    model = EDR(
        n_centers=3,
        n_directions=4,
        min_neighbors=1,
        h_decay=2.0,
        h_min=0.05,
        max_outer_steps=3,
        random_starts=3,
        max_iter=40,
        random_state=7,
        **kwargs,
    )

    model.X = np.array(
        [
            [-1.0, -1.0],
            [-0.2, 0.0],
            [0.0, 0.3],
            [0.8, 0.2],
            [1.2, 1.0],
        ],
        dtype=float,
    )
    model.Y = model.X @ np.array([1.0, -0.5])
    model.centers = model.X[[0, 2, 4]].copy()
    model.tree = cKDTree(model.X)

    return model


def test_init_sets_algorithm_state():
    model = EDR(random_state=1, n_centers=5, n_directions=6, min_neighbors=3)

    assert isinstance(model.rng, np.random.Generator)
    assert model.n_centers == 5
    assert model.n_directions == 6
    assert model.min_neighbors == 3
    assert model.X is None
    assert model.Y is None
    assert model.tree is None
    assert model.Data is not None
    assert model.Characteristics is not None


def test_mean_kernel_beta_and_direction_helpers_are_stable():
    model = configure_small_model()
    model.X = np.array([[1.0, 2.0, 5.0], [3.0, 6.0, 5.0], [5.0, 10.0, 5.0]])

    feature_mean, feature_scale = model.Mean_Calculate()

    assert np.allclose(feature_mean, [3.0, 6.0, 5.0])
    assert np.allclose(feature_scale[:2], np.std(model.X[:, :2], axis=0))
    assert feature_scale[2] == 1.0
    assert model.mean is feature_mean
    assert model.scale is feature_scale

    kernel_values = model.Kernel_Calculate(np.array([0.0, 0.25, 1.0, 2.0]))

    assert np.allclose(kernel_values, [1.0, 0.5625, 0.0, 0.0])
    assert model.Kernel_Calculate(0.0) == pytest.approx(1.0)

    normalized_beta = model.Beta_Normalize(np.array([3.0, 4.0, 0.0]))
    random_beta = model.Beta_Normalize(np.zeros(3))
    sign_aligned_beta = model.Beta_Normalize(np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    assert np.linalg.norm(normalized_beta) == pytest.approx(1.0)
    assert np.linalg.norm(random_beta) == pytest.approx(1.0)
    assert np.allclose(sign_aligned_beta, [1.0, -0.0, -0.0])


def test_generate_data_populates_arrays_and_tree_from_existing_state():
    model = EDR(n_centers=2, random_state=2)

    raw_X = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]])
    raw_Y = np.array([1.0, 2.0, 4.0, 8.0])
    raw_beta = np.array([1.0, 1.0])

    model.X = raw_X
    model.Y = raw_Y
    model.beta = raw_beta
    model.centers = raw_X[:2].copy()

    model.Generate_Data()

    assert model.X.shape == raw_X.shape
    assert model.Y.shape == raw_Y.shape
    assert model.centers.shape == (2, 2)
    assert model.tree.query(model.centers[:1])[0].shape == (1,)
    assert np.allclose(np.mean(model.X, axis=0), [0.0, 0.0])
    assert np.linalg.norm(model.beta) == pytest.approx(1.0)


def test_generate_direction_returns_unit_vectors_and_anisotropy_aligns_with_beta():
    model_a = configure_small_model()
    model_b = configure_small_model()

    directions_a = model_a.Generate_Direction()
    directions_b = model_b.Generate_Direction()

    assert directions_a.shape == (3, 4, 2)
    assert np.allclose(np.linalg.norm(directions_a, axis=-1), 1.0)
    assert np.allclose(directions_a, directions_b)

    beta_previous = np.array([1.0, 0.0])
    anisotropic_small_rho = model_a.Generate_Anisotropic_Direction(
        beta_previous,
        h_k=0.5,
        rho_k=0.05,
    )
    anisotropic_large_rho = model_a.Generate_Anisotropic_Direction(
        beta_previous,
        h_k=0.5,
        rho_k=1.0,
    )

    small_rho_projection = np.mean(np.abs(anisotropic_small_rho @ beta_previous))
    large_rho_projection = np.mean(np.abs(anisotropic_large_rho @ beta_previous))

    assert anisotropic_small_rho.shape == (3, 4, 2)
    assert np.allclose(np.linalg.norm(anisotropic_small_rho, axis=-1), 1.0)
    assert small_rho_projection > large_rho_projection


def test_weight_h_and_rho_calculations_use_sparse_local_masses():
    model = configure_small_model()
    model.Characteristics = SpyCharacteristics()

    isotropic_weights = model.Weight_Calculate(h=1.2)
    anisotropic_weights = model.Weight_Calculate(h=1.2, rho=0.2, beta=np.array([1.0, 0.0]))

    assert sparse.isspmatrix_csr(isotropic_weights)
    assert isotropic_weights.shape == (3, 5)
    assert np.all(isotropic_weights.data > 0)
    assert np.allclose(np.asarray(isotropic_weights.sum(axis=1)).ravel(), 1.0)
    assert not np.allclose(isotropic_weights.toarray(), anisotropic_weights.toarray())

    h_0 = model.H0_Calculate()
    updated_bandwidth = model.H_Update(h_0)

    assert h_0 > 0
    assert model._Average_Kernel_Mass_Calculate(h_0) >= model._Target_Neighbor_Count() - 1e-8
    assert updated_bandwidth == pytest.approx(h_0 / model.h_decay)
    assert model.Characteristics.h_0_values == [h_0]
    assert model.Characteristics.h_k_values == [updated_bandwidth]

    rho_k = model.Rho_Calculate(np.array([1.0, 0.0]), updated_bandwidth)

    assert model.rho_min <= rho_k <= 1.0
    mass_at_rho_min = model._Average_Kernel_Mass_Calculate(
        updated_bandwidth,
        model.rho_min,
        np.array([1.0, 0.0]),
    )
    anisotropic_mass = model._Average_Kernel_Mass_Calculate(
        updated_bandwidth,
        rho_k,
        np.array([1.0, 0.0]),
    )

    if mass_at_rho_min >= model._Target_Neighbor_Count():
        assert anisotropic_mass >= model._Target_Neighbor_Count() - 1e-8
    else:
        assert rho_k == pytest.approx(model.rho_min)

    assert model.Characteristics.rho_k_values == [rho_k]


def test_statistics_recalculate_after_in_place_response_change():
    model = configure_small_model()
    weights = sparse.csr_matrix(
        [
            [0.25, 0.75, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    local_mean = model.Local_Mean_Calculate(weights)
    directions = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )

    I_before_change, U_before_change = model.Average_Derivative_Statistics_Calculate(
        weights,
        local_mean,
        directions,
    )

    model.Y[1] += 10.0

    I_after_change, U_after_change = model.Average_Derivative_Statistics_Calculate(
        weights,
        local_mean,
        directions,
    )

    assert not np.allclose(I_before_change, I_after_change)
    assert np.allclose(U_before_change, U_after_change)


def test_local_mean_and_statistics_match_manual_formula():
    model = configure_small_model()
    weights = sparse.csr_matrix(
        [
            [0.25, 0.75, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    directions = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )

    local_mean = model.Local_Mean_Calculate(weights)
    I, U = model.Average_Derivative_Statistics_Calculate(weights, local_mean, directions)

    expected_local_mean = weights @ model.X
    expected_I = np.zeros((3, 2))
    expected_U = np.zeros((3, 2, 2))

    # Считаем прямую формулу по центрам, чтобы сравнить с CSR-реализацией.
    for center_index in range(weights.shape[0]):
        row_start = weights.indptr[center_index]
        row_end = weights.indptr[center_index + 1]
        neighbor_indices = weights.indices[row_start:row_end]
        neighbor_weights = weights.data[row_start:row_end]
        centered_neighbors = model.X[neighbor_indices] - local_mean[center_index]
        projection_values = centered_neighbors @ directions[center_index].T

        weighted_response = neighbor_weights * model.Y[neighbor_indices]
        weighted_centered_neighbors = neighbor_weights[:, None] * centered_neighbors

        expected_I[center_index] = weighted_response @ projection_values
        expected_U[center_index] = projection_values.T @ weighted_centered_neighbors

    assert np.allclose(local_mean, expected_local_mean)
    assert np.allclose(I, expected_I)
    assert np.allclose(U, expected_U)
    assert np.allclose(model.I_Calculate(weights, local_mean, directions), I)
    assert np.allclose(model.U_Calculate(weights, local_mean, directions), U)


def test_l_beta_objective_and_alternating_minimization_are_consistent():
    model = EDR(random_state=11, ridge=1e-8, random_starts=6, max_iter=80, tol=1e-10)

    true_beta = np.array([1.0, 2.0])
    true_beta = true_beta / np.linalg.norm(true_beta)
    U = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[2.0, 1.0], [1.0, -1.0], [0.5, 1.5]],
            [[-1.0, 2.0], [1.0, 0.5], [2.0, -0.5]],
        ]
    )
    local_slopes = np.array([1.5, -0.7, 0.9])
    I = local_slopes[:, None] * np.einsum("mqd,d->mq", U, true_beta)

    calculated_slopes = model.L_Calculate(I, U, true_beta)
    solved_beta = model.Beta_Calculate(I, U, local_slopes)
    estimated_beta = model.Alternating_Minimization(I, U)

    initial_beta = np.array([1.0, 0.0])
    initial_slopes = model.L_Calculate(I, U, initial_beta)
    initial_objective = model.Objective_Calculate(I, U, initial_slopes, initial_beta)
    recorded_beta_starts = []
    original_l_calculate = model.L_Calculate

    def record_l_calculate(I_for_record, U_for_record, beta_for_record):
        recorded_beta_starts.append(np.array(beta_for_record, copy=True))

        return original_l_calculate(I_for_record, U_for_record, beta_for_record)

    model.L_Calculate = record_l_calculate
    estimated_from_initial = model.Alternating_Minimization(I, U, initial_beta)
    model.L_Calculate = original_l_calculate

    final_slopes = model.L_Calculate(I, U, estimated_from_initial)
    final_objective = model.Objective_Calculate(I, U, final_slopes, estimated_from_initial)

    normal_matrix = model.ridge * np.eye(2)
    right_hand_side = np.zeros(2)

    # Собираем нормальную систему явно, чтобы проверить закрытую формулу beta.
    for center_index in range(U.shape[0]):
        U_for_center = U[center_index]
        local_slope = local_slopes[center_index]

        normal_matrix += local_slope**2 * (U_for_center.T @ U_for_center)
        right_hand_side += local_slope * (U_for_center.T @ I[center_index])

    expected_beta = np.linalg.solve(normal_matrix, right_hand_side)

    assert np.allclose(calculated_slopes, local_slopes)
    assert np.allclose(solved_beta, expected_beta)
    assert model.Objective_Calculate(I, U, local_slopes, true_beta) == pytest.approx(0.0)
    assert np.linalg.norm(estimated_beta) == pytest.approx(1.0)
    assert abs(float(estimated_beta @ true_beta)) > 0.95
    assert np.allclose(recorded_beta_starts[0], model.Beta_Normalize(initial_beta))
    assert final_objective <= initial_objective + 1e-8


def test_step_condition_combines_step_limit_and_bandwidth_limit():
    model = EDR(h_decay=2.0, h_min=0.1, max_outer_steps=2)

    model.current_outer_step = 0
    assert model.Step_k_Condition(0.5) is True

    model.current_outer_step = 2
    assert model.Step_k_Condition(0.5) is False

    model.current_outer_step = 0
    assert model.Step_k_Condition(0.15) is False


def test_edr_single_index_runs_step_0_and_step_k_smoke():
    rng = np.random.default_rng(123)
    raw_X = rng.normal(size=(80, 5))
    true_beta = np.array([1.0, -0.5, 0.25, 0.0, 0.75])
    true_beta = true_beta / np.linalg.norm(true_beta)
    raw_Y = np.sin(raw_X @ true_beta)

    model = EDR_ADP_single_index(
        n_centers=12,
        n_directions=5,
        min_neighbors=3,
        h_decay=1.5,
        h_min=0.05,
        max_outer_steps=2,
        random_starts=3,
        max_iter=20,
        random_state=4,
    )
    model.Characteristics = SpyCharacteristics()
    model.X = raw_X
    model.Y = raw_Y
    model.beta = true_beta
    model.centers = raw_X[:12].copy()
    model.Generate_Data()

    estimated_beta_step_0, initial_bandwidth = model.run_step_0(model.beta)
    estimated_beta_step_k, updated_bandwidth, rho_k = model.run_step_k(
        estimated_beta_step_0,
        initial_bandwidth,
        model.beta,
        initial_bandwidth,
    )

    assert estimated_beta_step_0.shape == (5,)
    assert estimated_beta_step_k.shape == (5,)
    assert np.linalg.norm(estimated_beta_step_0) == pytest.approx(1.0)
    assert np.linalg.norm(estimated_beta_step_k) == pytest.approx(1.0)
    assert np.isfinite(initial_bandwidth)
    assert np.isfinite(updated_bandwidth)
    assert np.isfinite(rho_k)
    assert model.Characteristics.step_0_values
    assert model.Characteristics.step_k_values
