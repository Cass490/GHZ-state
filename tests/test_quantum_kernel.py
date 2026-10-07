"""Deferred quantum checks: these execute tiny circuits, never SVM training."""
import numpy as np
import pennylane as qml
from src.quantum_kernel import QuantumKernel


def test_fidelity_against_compute_uncompute(tmp_path):
    X = np.array([[0.1, 0.2, -0.3, 0.4], [-0.2, 0.3, 0.5, -0.1], [0.7, -0.2, 0.1, 0.3]])
    kernel = QuantumKernel(4, cache_dir=tmp_path)
    K = kernel.matrix(X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(K), 1, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() >= -1e-12
    device = qml.device('default.qubit', wires=4)
    def embedding(x):
        for wire in range(4):
            qml.Hadamard(wires=wire)
        for wire in range(4):
            qml.RY(np.pi * x[wire], wires=wire)
        for wire in range(3):
            qml.CNOT(wires=[wire, wire + 1])
    @qml.qnode(device)
    def overlap(x, y):
        embedding(x)
        qml.adjoint(embedding)(y)
        return qml.probs(wires=range(4))
    assert np.isclose(K[0, 1], overlap(X[0], X[1])[0])
    np.testing.assert_array_equal(kernel.matrix(X), K)
    assert len(list(tmp_path.glob('*.npy'))) == 1


def test_reuploading_and_rectangular_kernel():
    X = np.random.default_rng(9).uniform(-1, 1, (3, 15))
    kernel = QuantumKernel(15)
    assert kernel.wires == 4
    assert kernel.matrix(X[:1], X).shape == (1, 3)
    np.testing.assert_allclose(np.diag(kernel.matrix(X)), 1, atol=1e-12)
