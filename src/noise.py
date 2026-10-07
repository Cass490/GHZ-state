import numpy as np
from .density import validate_density


def add_state_noise(rho, p):
    rho = validate_density(rho)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("white-noise strength must be in [0, 1]")
    return (1 - p) * rho + p * np.eye(len(rho)) / len(rho)


def add_measurement_noise(features, std, rng):
    if not np.isfinite(std) or std < 0:
        raise ValueError("measurement-noise standard deviation must be nonnegative")
    return np.clip(features + rng.normal(0, std, np.shape(features)), -1, 1)
