import numpy as np
import pandas as pd
import pytest
from src.preprocessing import grouped_split, feature_matrix, validate_split


def fake_metadata():
    # Synthetic metadata only; no quantum state generation or training.
    return pd.DataFrame([{'base_state_id': f'b{i}', 'label': i % 2, 'XX': 0.1, 'ZZ': -0.2}
                         for i in range(40) for _ in range(5)])


def test_reproducible_leakage_free_partition():
    df = fake_metadata()
    first, second = grouped_split(df), grouped_split(df)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert {k: len(v) for k, v in first.items()} == {'train': 140, 'validation': 30, 'test': 30}
    validate_split(df, first)


def test_metadata_cannot_be_used_as_features():
    with pytest.raises(ValueError):
        feature_matrix(fake_metadata(), ['XX', 'negativity'])


def test_leakage_rejected():
    df = fake_metadata()
    split = grouped_split(df)
    a = split['train'][0]
    b = split['test'][0]
    split['train'][0], split['test'][0] = b, a
    with pytest.raises(ValueError, match='leakage'):
        validate_split(df, split)
