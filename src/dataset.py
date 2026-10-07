"""Labels and metadata stay separate from the observable model inputs."""
import numpy as np
import pandas as pd
from .states import ghz_state, w_state, product_state, biseparable_state, random_pure_state, rotate_locally
from .density import ket_to_density
from .qubit_loss import trace_out_qubit
from .entanglement import negativity
from .observables import ALL_OBSERVABLES, extract_pauli_features
from .noise import add_state_noise, add_measurement_noise


def generate_dataset(config):
    rng = np.random.default_rng(config['seed'])
    levels = config['noise_levels']
    if not levels or len(set(levels)) != len(levels) or min(levels) < 0 or max(levels) > 1:
        raise ValueError("noise_levels must be distinct strengths in [0, 1]")
    if 0.0 not in levels:
        raise ValueError("include noise level 0.0 for the clean-training study")
    count = int(config['num_samples'])
    if count < 40:
        raise ValueError("at least 40 samples required for grouped experiments")
    families = ['ghz', 'w', 'product', 'biseparable']
    if config.get('include_generic', False):
        families.append('generic')
    rows = []
    bases = []
    group = 0
    while len(rows) < count:
        desired = group % 2
        for attempt in range(10000):
            family = str(rng.choice(families))
            pair = tuple(sorted(rng.choice(3, size=2, replace=False)))
            if family == 'ghz':
                psi = ghz_state()
            elif family == 'w':
                psi = w_state()
            elif family == 'product':
                psi = product_state(rng)
            elif family == 'biseparable':
                psi = biseparable_state(pair)
            else:
                psi = random_pure_state(rng, 8)
            psi = rotate_locally(psi, rng)
            lost = int(rng.integers(3))
            ideal = trace_out_qubit(ket_to_density(psi), lost)
            if int(negativity(ideal) > config['label_threshold']) == desired:
                break
        else:
            raise RuntimeError("unable to sample requested clean-label class")
        base_id = f"base_{group:05d}"
        bases.append(psi)
        for p in levels:
            reduced = add_state_noise(ideal, p)
            n = negativity(reduced)
            features = extract_pauli_features(reduced)
            std = config.get('measurement_noise_std', 0.0)
            if std:
                features = add_measurement_noise(features, std, rng)
            rows.append(dict(sample_id=f"sample_{len(rows):05d}", base_state_id=base_id,
                             family=family, lost_qubit=lost, noise_level=float(p),
                             bell_pair=''.join(map(str, pair)) if family == 'biseparable' else '',
                             **dict(zip(ALL_OBSERVABLES, features)), negativity=n,
                             label=int(n > config['label_threshold'])))
            if len(rows) == count:
                break
        group += 1
    return pd.DataFrame(rows), np.array(bases)
