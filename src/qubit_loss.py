import numpy as np
from .density import validate_density


def partial_trace(rho, traced_qubits):
    """Trace selected qubits; preserve ascending order of retained qubits."""
    rho = validate_density(rho)
    n = int(np.log2(len(rho)))
    if 2**n != len(rho):
        raise ValueError("dimension must be a power of two")
    traced = list(traced_qubits)
    if len(set(traced)) != len(traced) or any(q not in range(n) for q in traced):
        raise ValueError("invalid or repeated traced qubit")
    view = rho.reshape([2] * (2 * n))
    for q in sorted(traced, reverse=True):
        view = np.trace(view, axis1=q, axis2=q + n)
        n -= 1
    return view.reshape(2**n, 2**n)


def trace_out_qubit(rho, qubit):
    if np.shape(rho) != (8, 8):
        raise ValueError("expected a three-qubit 8 by 8 matrix")
    return partial_trace(rho, [qubit])
