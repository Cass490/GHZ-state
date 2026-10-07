import numpy as np
import pytest
from src.states import ghz_state, w_state, random_pure_state
from src.density import ket_to_density
from src.qubit_loss import trace_out_qubit
from src.observables import ALL_OBSERVABLES, extract_pauli_features, pauli_observable


def test_known_correlations():
    names = ['XX', 'YY', 'ZZ']
    np.testing.assert_allclose(extract_pauli_features(trace_out_qubit(ket_to_density(ghz_state()), 0), names), [0, 0, 1])
    np.testing.assert_allclose(extract_pauli_features(trace_out_qubit(ket_to_density(w_state()), 0), names), [2/3, 2/3, -1/3])


def test_full_pauli_reconstruction_and_bounds():
    rho = trace_out_qubit(ket_to_density(random_pure_state(np.random.default_rng(4), 8)), 1)
    features = extract_pauli_features(rho)
    assert len(features) == 15 and np.all(np.abs(features) <= 1)
    reconstructed = (np.eye(4) + sum(x * pauli_observable(name) for x, name in zip(features, ALL_OBSERVABLES))) / 4
    np.testing.assert_allclose(reconstructed, rho, atol=1e-12)
    with pytest.raises(ValueError):
        pauli_observable('II')
