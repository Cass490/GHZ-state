import numpy as np
import pytest
from src.states import ghz_state, w_state, product_state, biseparable_state, rotate_locally
from src.density import ket_to_density
from src.qubit_loss import trace_out_qubit
from src.entanglement import negativity, partial_transpose, entanglement_label


@pytest.mark.parametrize('lost', [0, 1, 2])
def test_ghz_w_negativity(lost):
    assert negativity(trace_out_qubit(ket_to_density(ghz_state()), lost)) < 1e-12
    assert np.isclose(negativity(trace_out_qubit(ket_to_density(w_state()), lost)), (np.sqrt(5) - 1) / 6)


@pytest.mark.parametrize('pair', [(0, 1), (0, 2), (1, 2)])
def test_bell_pair_survives_only_spectator_loss(pair):
    rho = ket_to_density(biseparable_state(pair))
    for lost in range(3):
        n = negativity(trace_out_qubit(rho, lost))
        assert np.isclose(n, 0 if lost in pair else 0.5, atol=1e-12)


def test_product_and_local_invariance():
    rng = np.random.default_rng(2)
    for psi in [ghz_state(), w_state(), biseparable_state(), product_state(rng)]:
        rotated = rotate_locally(psi, rng)
        for lost in range(3):
            before = trace_out_qubit(ket_to_density(psi), lost)
            after = trace_out_qubit(ket_to_density(rotated), lost)
            assert np.isclose(negativity(before), negativity(after), atol=1e-12)
            np.testing.assert_allclose(partial_transpose(partial_transpose(before)), before)
    assert entanglement_label(trace_out_qubit(ket_to_density(product_state(rng)), 0)) == 0
