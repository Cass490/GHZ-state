import numpy as np
import pytest
from src.states import w_state
from src.density import ket_to_density, validate_density


def test_pure_density():
    rho = ket_to_density(w_state())
    validate_density(rho)
    np.testing.assert_allclose(rho @ rho, rho, atol=1e-12)
    assert np.linalg.matrix_rank(rho) == 1


@pytest.mark.parametrize('rho', [np.eye(4), np.diag([1.1, -0.1]),
                               np.array([[1, 1], [0, 0]]), np.ones((2, 3))])
def test_invalid_density(rho):
    with pytest.raises(ValueError):
        validate_density(rho)


def test_reject_unnormalized_ket():
    with pytest.raises(ValueError):
        ket_to_density(np.ones(8))
