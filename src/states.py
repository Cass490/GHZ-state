"""Three-qubit states; qubit 0 is the leftmost computational-basis bit."""
from functools import reduce
import numpy as np


def tensor(*arrays):
    return reduce(np.kron, arrays)


def ghz_state():
    psi = np.zeros(8, dtype=complex)
    psi[[0, 7]] = 1 / np.sqrt(2)
    return psi


def w_state():
    psi = np.zeros(8, dtype=complex)
    psi[[1, 2, 4]] = 1 / np.sqrt(3)
    return psi


def random_pure_state(rng, dim=2):
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)


def product_state(rng):
    return tensor(*(random_pure_state(rng) for _ in range(3)))


def biseparable_state(pair=(0, 1)):
    if len(pair) != 2 or len(set(pair)) != 2 or any(q not in range(3) for q in pair):
        raise ValueError("pair must contain two distinct qubit indices")
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[sum(1 << (2 - q) for q in pair)] = 1 / np.sqrt(2)
    return psi


def random_local_unitary(rng):
    """Haar-distributed U(2); irrelevant global phase is retained."""
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q @ np.diag(d / np.abs(d))


def apply_local_unitaries(psi, unitaries):
    if len(unitaries) != 3:
        raise ValueError("three one-qubit unitaries required")
    for u in unitaries:
        if np.shape(u) != (2, 2) or not np.allclose(u.conj().T @ u, np.eye(2)):
            raise ValueError("local matrices must be unitary")
    return tensor(*unitaries) @ np.asarray(psi)


def rotate_locally(psi, rng):
    return apply_local_unitaries(psi, [random_local_unitary(rng) for _ in range(3)])
