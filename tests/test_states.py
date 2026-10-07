import numpy as np
import pytest
from src.states import (ghz_state, w_state, product_state, biseparable_state,
                        random_pure_state, random_local_unitary, rotate_locally)


def test_known_amplitudes():
    np.testing.assert_allclose(ghz_state(), np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2))
    np.testing.assert_allclose(w_state(), np.array([0, 1, 1, 0, 1, 0, 0, 0]) / np.sqrt(3))


@pytest.mark.parametrize('pair', [(0, 1), (0, 2), (1, 2)])
def test_biseparable_normalization(pair):
    assert np.isclose(np.vdot(biseparable_state(pair), biseparable_state(pair)), 1)


def test_random_states_and_rotations():
    rng = np.random.default_rng(42)
    for psi in [ghz_state(), w_state(), product_state(rng), random_pure_state(rng, 8)]:
        assert np.isclose(np.vdot(psi, psi), 1)
        rotated = rotate_locally(psi, rng)
        assert np.isclose(np.vdot(rotated, rotated), 1)
    u = random_local_unitary(rng)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-12)


def test_seed_reproducibility():
    np.testing.assert_array_equal(product_state(np.random.default_rng(7)), product_state(np.random.default_rng(7)))
