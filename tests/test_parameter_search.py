import numpy as np

from experiments import parameter_search as study


def test_search_grid_preserves_requested_parameters_and_covers_levels():
    for case in study.scenarios():
        arms = study.configurations(case)
        base = study.base_config(case)
        assert len({study.key(a) for a in arms}) == len(arms)
        for arm in arms:
            assert sum(arm[k] != base[k] for k in base) <= 1
            j = arm['N_J'] or case['n']
            p = arm['N_phi'] or min(arm['N_loc'], case['d'])
            assert case['n'] / arm['N_loc'] <= j <= case['n']
            assert p > case['m']
        for field in study.VARIABLES[:4]:
            values = study.levels(case)[field]
            assert len(values) == 10
            assert np.allclose(np.diff(values), np.diff(values)[0])
        assert len(study.SEEDS['explore']) >= 10
        assert base['N_J'] is None and base['N_phi'] is None
        assert base['N_loc'] == 10 and base['N_lin'] is None


def test_metric_is_rotation_invariant_and_measures_missing_directions():
    truth = np.eye(6)[:, :2]
    rotated = truth @ np.array([[0., -1.], [1., 0.]])
    assert study.geometry(rotated, truth)['edr'] < 1e-14
    orthogonal = np.eye(6)[:, 2:4]
    assert np.isclose(study.geometry(orthogonal, truth)['edr'], 1)
    partial = np.eye(6)[:, [0, 2]]
    assert np.isclose(study.geometry(partial, truth)['edr'], 1 / np.sqrt(2))
    assert np.isclose(study.geometry(partial, truth)['max_sine'], 1)


