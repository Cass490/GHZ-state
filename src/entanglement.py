import numpy as np
from .density import validate_density


def partial_transpose(rho, subsystem=1):
    # The operation also accepts a previously transposed (non-positive) matrix.
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4) or subsystem not in (0, 1) or not np.all(np.isfinite(rho)):
        raise ValueError("expected two qubits and subsystem 0 or 1")
    axes = (2, 1, 0, 3) if subsystem == 0 else (0, 3, 2, 1)
    return rho.reshape(2, 2, 2, 2).transpose(axes).reshape(4, 4)


def ppt_eigenvalues(rho):
    rho = validate_density(rho)
    return np.linalg.eigvalsh(partial_transpose(rho))


def negativity(rho):
    return float(-np.minimum(ppt_eigenvalues(rho), 0).sum())


def entanglement_label(rho, threshold=1e-8):
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    return int(negativity(rho) > threshold)
