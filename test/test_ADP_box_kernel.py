from functools import partial

import numpy as np
import pytest

from ADP.engine.box_kernel import (
    box_kernel,
    make_plateau_kernel,
    plateau_kernel,
    sparse_kernel_parameters,
)


def test_box_and_plateau_kernel_boundaries():
    q = np.array([0.0, 0.5, 0.75, 1.0, 2.0])
    np.testing.assert_array_equal(
        box_kernel(q),
        [1.0, 1.0, 1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        plateau_kernel(q, tau=0.5),
        [1.0, 1.0, 0.5, 0.0, 0.0],
        atol=1e-15,
    )


def test_plateau_tau_and_kernel_identity_are_validated():
    kernel = make_plateau_kernel(0.4)
    assert isinstance(kernel, partial)
    assert sparse_kernel_parameters(box_kernel) == ("box", None)
    assert sparse_kernel_parameters(kernel) == ("plateau", 0.4)
    with pytest.raises(ValueError, match="tau"):
        make_plateau_kernel(1.0)
    with pytest.raises(ValueError, match="finite"):
        plateau_kernel(np.array([np.nan]))
