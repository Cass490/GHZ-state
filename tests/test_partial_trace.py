import numpy as np
import pytest
from src.states import ghz_state, w_state, product_state, random_pure_state, tensor
from src.density import ket_to_density, validate_density
from src.qubit_loss import trace_out_qubit, partial_trace


@pytest.mark.parametrize('lost', [0, 1, 2])
def test_analytic_reduced_states(lost):
    ghz_expected = np.diag([0.5, 0, 0, 0.5])
    w_expected = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]) / 3
    np.testing.assert_allclose(trace_out_qubit(ket_to_density(ghz_state()), lost), ghz_expected, atol=1e-12)
    np.testing.assert_allclose(trace_out_qubit(ket_to_density(w_state()), lost), w_expected, atol=1e-12)


@pytest.mark.parametrize('lost', [0, 1, 2])
def test_random_against_explicit_basis_sum(lost):
    rho = ket_to_density(random_pure_state(np.random.default_rng(5), 8))
    retained = [q for q in range(3) if q != lost]
    expected = np.zeros((4, 4), dtype=complex)
    def full_index(pair, bit):
        bits = [0, 0, 0]
        bits[lost] = bit
        bits[retained[0]], bits[retained[1]] = pair // 2, pair % 2
        return 4 * bits[0] + 2 * bits[1] + bits[2]
    for i in range(4):
        for j in range(4):
            expected[i, j] = sum(rho[full_index(i, b), full_index(j, b)] for b in [0, 1])
    actual = trace_out_qubit(rho, lost)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    validate_density(actual)


def test_multiple_traces_and_retained_order():
    a, b, c = np.array([1, 0]), np.array([0, 1]), np.array([1, 1]) / np.sqrt(2)
    rho = ket_to_density(tensor(a, b, c))
    np.testing.assert_allclose(partial_trace(rho, [0, 2]), ket_to_density(b))
    np.testing.assert_allclose(partial_trace(rho, [0, 1, 2]), [[1]])
    np.testing.assert_allclose(partial_trace(rho, []), rho)
    with pytest.raises(ValueError):
        partial_trace(rho, [1, 1])
