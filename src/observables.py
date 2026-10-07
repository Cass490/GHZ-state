import numpy as np
from .density import validate_density

PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.diag([1, -1]).astype(complex),
}
ALL_OBSERVABLES = ('XI', 'YI', 'ZI', 'IX', 'IY', 'IZ', 'XX', 'XY', 'XZ',
                   'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ')


def pauli_observable(name):
    if name not in ALL_OBSERVABLES:
        raise ValueError(f"unknown nontrivial two-qubit Pauli observable: {name}")
    return np.kron(PAULI[name[0]], PAULI[name[1]])


def extract_pauli_features(rho, names=ALL_OBSERVABLES):
    rho = validate_density(rho)
    if rho.shape != (4, 4):
        raise ValueError("expected a two-qubit density matrix")
    values = np.array([np.trace(rho @ pauli_observable(n)).real for n in names])
    if np.any(np.abs(values) > 1 + 1e-9):
        raise ValueError("expectation outside [-1, 1]")
    return np.clip(values, -1, 1)
