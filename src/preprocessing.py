import json
import numpy as np
from sklearn.model_selection import train_test_split
from .observables import ALL_OBSERVABLES


def grouped_split(df, seed=42, train_fraction=0.70, val_fraction=0.15):
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 - train_fraction:
        raise ValueError("train, validation and test fractions must be positive")
    groups = df.groupby('base_state_id', sort=True)['label'].max()
    # Integer counts avoid flooring 0.15/(1-0.70) to one group too few.
    n_train = round(len(groups) * train_fraction)
    n_val = round(len(groups) * val_fraction)
    if min(n_train, n_val, len(groups) - n_train - n_val) < 2:
        raise ValueError("too few base states for three stratified splits; increase num_samples")
    train, rest = train_test_split(groups.index.to_numpy(), train_size=n_train,
                                   stratify=groups.to_numpy(), random_state=seed)
    val, test = train_test_split(rest, train_size=n_val,
                                 stratify=groups.loc[rest].to_numpy(), random_state=seed + 1)
    split = {name: np.flatnonzero(df.base_state_id.isin(ids).to_numpy())
             for name, ids in [('train', train), ('validation', val), ('test', test)]}
    validate_split(df, split)
    return split


def validate_split(df, split):
    used = []
    group_sets = []
    for name in ('train', 'validation', 'test'):
        idx = np.asarray(split[name], dtype=int)
        if len(idx) == 0 or len(set(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= len(df)):
            raise ValueError("invalid split indices")
        used.extend(idx.tolist())
        group_sets.append(set(df.iloc[idx].base_state_id))
        if df.iloc[idx].label.nunique() != 2:
            raise ValueError("each split must contain both classes; use more base states")
    if sorted(used) != list(range(len(df))):
        raise ValueError("splits must partition all rows exactly once")
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("base-state leakage between splits")


def feature_matrix(df, names):
    if not names or len(set(names)) != len(names) or any(n not in ALL_OBSERVABLES for n in names):
        raise ValueError("model inputs must be distinct Pauli observables")
    return df.loc[:, list(names)].to_numpy(dtype=float)


def save_split(split, path):
    path.write_text(json.dumps({k: v.tolist() for k, v in split.items()}, indent=2))
