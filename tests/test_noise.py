import numpy as np
import pytest
from src.states import w_state
from src.density import ket_to_density, validate_density
from src.qubit_loss import trace_out_qubit
from src.noise import add_state_noise, add_measurement_noise
from src.entanglement import negativity


def test_noise_commutes_with_loss_and_extremes():
    rho = ket_to_density(w_state())
    np.testing.assert_allclose(add_state_noise(rho, 0), rho)
    np.testing.assert_allclose(add_state_noise(rho, 1), np.eye(8) / 8)
    for p in [0, 0.1, 0.5, 1]:
        noisy = add_state_noise(rho, p)
        validate_density(noisy)
        np.testing.assert_allclose(trace_out_qubit(noisy, 2), add_state_noise(trace_out_qubit(rho, 2), p))


def test_w_noise_boundary_and_monotonicity():
    reduced = trace_out_qubit(ket_to_density(w_state()), 0)
    n0 = (np.sqrt(5) - 1) / 6
    values = [negativity(add_state_noise(reduced, p)) for p in np.linspace(0, 1, 21)]
    assert np.all(np.diff(values) <= 1e-12)
    critical = n0 / (n0 + 0.25)
    assert negativity(add_state_noise(reduced, critical)) < 1e-12
    assert negativity(add_state_noise(reduced, critical - 0.001)) > 0


def test_invalid_noise_and_measurement_bounds():
    with pytest.raises(ValueError):
        add_state_noise(ket_to_density(w_state()), -0.1)
    features = add_measurement_noise(np.array([1., -1.]), 0.5, np.random.default_rng(2))
    assert np.all(np.abs(features) <= 1)
