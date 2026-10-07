import numpy as np


def ket_to_density(psi):
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 1 or not np.isclose(np.vdot(psi, psi), 1, atol=1e-10, rtol=0):
        raise ValueError("statevector must be one-dimensional and normalized")
    return np.outer(psi, psi.conj())


def validate_density(rho, atol=1e-10):
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density matrix must be square")
    if not np.all(np.isfinite(rho)) or not np.allclose(rho, rho.conj().T, atol=atol, rtol=0):
        raise ValueError("density matrix must be finite and Hermitian")
    if not np.isclose(np.trace(rho), 1, atol=atol, rtol=0):
        raise ValueError("density matrix must have unit trace")
    if np.linalg.eigvalsh(rho).min() < -atol:
        raise ValueError("density matrix must be positive semidefinite")
    return rho
