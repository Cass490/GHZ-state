"""Exact simulator fidelity kernels, with content-addressed matrix caching."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import f1_score


class QuantumKernel:
    def __init__(self, num_features, max_wires=4, layers=1, cache_dir=None):
        if num_features < 1 or max_wires < 1 or layers < 1:
            raise ValueError("positive dimensions and layers required")
        self.num_features = num_features
        self.wires = min(num_features, max_wires)
        self.layers = layers
        self.cache_dir = Path(cache_dir) if cache_dir else None
        device = qml.device('default.qubit', wires=self.wires, shots=None)

        @qml.qnode(device, interface=None, diff_method=None)
        def feature_state(x):
            for wire in range(self.wires):
                qml.Hadamard(wires=wire)
            for _ in range(self.layers):
                for start in range(0, num_features, self.wires):
                    for wire, value in enumerate(x[start:start + self.wires]):
                        qml.RY(np.pi * value, wires=wire)
                    for wire in range(self.wires - 1):
                        qml.CNOT(wires=[wire, wire + 1])
            return qml.state()

        self.feature_state = feature_state

    def states(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.num_features or not np.all(np.isfinite(X)):
            raise ValueError("invalid feature matrix")
        if np.any(np.abs(X) > 1 + 1e-9):
            raise ValueError("quantum features must be in [-1, 1]")
        return np.array([self.feature_state(x) for x in X])

    def matrix(self, X, Y=None):
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        metadata = json.dumps({'version': 1, 'features': self.num_features, 'wires': self.wires,
                               'layers': self.layers, 'pennylane': qml.__version__,
                               'shapes': [X.shape, Y.shape]}, sort_keys=True)
        digest = hashlib.sha256(metadata.encode() + X.tobytes() + Y.tobytes()).hexdigest()
        path = self.cache_dir / f'{digest}.npy' if self.cache_dir else None
        if path and path.exists():
            return np.load(path, allow_pickle=False)
        a = self.states(X)
        b = a if np.array_equal(X, Y) else self.states(Y)
        K = np.clip(np.abs(a.conj() @ b.T)**2, 0, 1)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, K)
        return K


def quantum_kernel_matrix(X, Y=None, **kwargs):
    return QuantumKernel(np.shape(X)[1], **kwargs).matrix(X, Y)


def train_quantum(K_train, y_train, K_val, y_val, c_grid, seed=42):
    best = None
    score = -1
    for c in c_grid:
        model = SVC(C=c, kernel='precomputed', random_state=seed)
        model.fit(K_train, y_train)
        candidate = f1_score(y_val, model.predict(K_val), zero_division=0)
        if candidate > score:
            best, score = model, candidate
    return best, {'C': best.C, 'validation_f1': float(score)}
