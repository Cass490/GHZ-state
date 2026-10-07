"""Deferred integration checks: these generate small datasets when explicitly run."""
import numpy as np
import yaml
from pathlib import Path
from src.dataset import generate_dataset
from src.preprocessing import grouped_split


def test_dataset_reproducibility_and_labels():
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config.yaml').read_text())
    config['num_samples'] = 100
    first, bases = generate_dataset(config)
    second, other = generate_dataset(config)
    assert first.equals(second)
    np.testing.assert_array_equal(bases, other)
    np.testing.assert_array_equal(first.label, (first.negativity > config['label_threshold']).astype(int))
    assert first.label.value_counts().to_dict() == {0: 50, 1: 50}
    assert first.base_state_id.nunique() == 20
    grouped_split(first)
