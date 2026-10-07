"""Explicit experiment orchestration. Importing this module runs no experiment."""
import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import joblib
from .states import ghz_state, w_state
from .density import ket_to_density, validate_density
from .qubit_loss import trace_out_qubit
from .entanglement import negativity
from .noise import add_state_noise
from .dataset import generate_dataset
from .preprocessing import grouped_split, validate_split, feature_matrix, save_split
from .classical_svm import train_classical
from .evaluation import evaluate, save_json


def paths(config):
    output = Path(config['output_dir']) / f"seed_{config['seed']}"
    data = Path(config['data_dir']) / f"seed_{config['seed']}"
    for directory in [data, output / 'metrics', output / 'figures', output / 'models', output / 'kernels']:
        directory.mkdir(parents=True, exist_ok=True)
    return data, output


def write_provenance(config, output):
    packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib', 'PennyLane', 'PyYAML', 'joblib']
    save_json({'config': config, 'versions': {p: importlib.metadata.version(p) for p in packages}},
              output / 'metrics' / 'provenance.json')


def validate_states(config):
    rows = []
    for family, psi in [('GHZ', ghz_state()), ('W', w_state())]:
        rho = validate_density(ket_to_density(psi))
        expected = 0.0 if family == 'GHZ' else (np.sqrt(5) - 1) / 6
        for lost in range(3):
            reduced = validate_density(trace_out_qubit(rho, lost))
            n = negativity(reduced)
            if reduced.shape != (4, 4) or not np.isclose(n, expected, atol=1e-10):
                raise RuntimeError(f"physics validation failed for {family}, lost qubit {lost}")
            rows.append({'family': family, 'lost_qubit': lost, 'negativity': n})
    _, output = paths(config)
    pd.DataFrame(rows).to_csv(output / 'metrics' / 'physics_validation.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return pd.DataFrame(rows)


def physics_demo(config):
    from .plotting import plt, save_figure
    rows = validate_states(config)
    _, output = paths(config)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, family in enumerate(['GHZ', 'W']):
        ax.bar(np.arange(3) + (i - 0.5) * 0.35, rows.loc[rows.family == family, 'negativity'],
               width=0.35, label=family)
    ax.set(xticks=range(3), xticklabels=['q0', 'q1', 'q2'], xlabel='Lost qubit', ylabel='Negativity',
           title='Surviving two-qubit entanglement')
    ax.legend()
    save_figure(fig, output / 'figures' / 'ghz_w_loss')
    noise_rows = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for family, psi in [('GHZ', ghz_state()), ('W', w_state())]:
        rho = ket_to_density(psi)
        values = []
        for p in np.linspace(0, 1, 101):
            n = negativity(trace_out_qubit(add_state_noise(rho, p), 0))
            values.append(n)
            noise_rows.append({'family': family, 'noise_level': p, 'negativity': n})
        ax.plot(np.linspace(0, 1, 101), values, label=family)
    ax.set(xlabel='White-noise strength p', ylabel='Negativity', title='Entanglement under state noise')
    ax.legend()
    save_figure(fig, output / 'figures' / 'negativity_noise')
    pd.DataFrame(noise_rows).to_csv(output / 'metrics' / 'physics_noise.csv', index=False)


def prepare_dataset(config):
    from .plotting import plot_dataset
    data, output = paths(config)
    df, bases = generate_dataset(config)
    split = grouped_split(df, config['seed'], config['train_fraction'], config['val_fraction'])
    df.to_csv(data / 'dataset.csv', index=False)
    np.savez_compressed(data / 'base_states.npz', states=bases,
                        base_state_ids=df.base_state_id.drop_duplicates().to_numpy(dtype=str))
    save_split(split, data / 'split.json')
    save_json(config, data / 'dataset_config.json')
    summary = {'samples': len(df), 'base_states': df.base_state_id.nunique(),
               'label_counts': {str(k): int(v) for k, v in df.label.value_counts().items()},
               'split_rows': {k: len(v) for k, v in split.items()},
               'split_groups': {k: df.iloc[v].base_state_id.nunique() for k, v in split.items()},
               'family_counts': {str(k): int(v) for k, v in df.family.value_counts().items()}}
    save_json(summary, output / 'metrics' / 'dataset_summary.json')
    write_provenance(config, output)
    plot_dataset(df, output / 'figures' / 'dataset_distribution')
    print(f"Saved {len(df)} samples from {df.base_state_id.nunique()} base states to {data}")
    return df, split


def load_dataset(config):
    import json
    data, _ = paths(config)
    if not (data / 'dataset.csv').exists():
        raise FileNotFoundError("Generate the dataset first: python run_experiments.py dataset")
    recorded = json.loads((data / 'dataset_config.json').read_text())
    for key in ['seed', 'num_samples', 'noise_levels', 'label_threshold', 'measurement_noise_std',
                'include_generic', 'train_fraction', 'val_fraction']:
        if recorded.get(key) != config.get(key):
            raise ValueError(f"dataset configuration mismatch for {key}; regenerate the dataset")
    df = pd.read_csv(data / 'dataset.csv')
    split = {k: np.asarray(v, dtype=int) for k, v in json.loads((data / 'split.json').read_text()).items()}
    validate_split(df, split)
    return df, split


def fit_evaluate(config, df, split, features, models=('classical', 'quantum'), tag=None, figures=False):
    """Tune only on validation groups; preserve training-only fits for inference."""
    from .plotting import plot_confusion
    _, output = paths(config)
    X = feature_matrix(df, features)
    y = df.label.to_numpy(dtype=int)
    train, val, test = (split[k] for k in ['train', 'validation', 'test'])
    for idx in [train, val]:
        if len(np.unique(y[idx])) < 2:
            raise ValueError("training and validation require both labels")
    tag = tag or f"budget_{len(features)}"
    rows, fitted = [], {}
    for model_name in models:
        if model_name == 'classical':
            model, params = train_classical(X[train], y[train], X[val], y[val],
                                            config['c_grid'], config['gamma_grid'], config['seed'])
            test_input = X[test]
        else:
            from .quantum_kernel import QuantumKernel, train_quantum
            kernel = QuantumKernel(len(features), **config['quantum'], cache_dir=output / 'kernels')
            K_train = kernel.matrix(X[train])
            symmetric = np.allclose(K_train, K_train.T, atol=1e-10)
            diagonal = np.allclose(np.diag(K_train), 1, atol=1e-10)
            eigen_min = float(np.linalg.eigvalsh(K_train).min())
            if not symmetric or not diagonal or eigen_min < -1e-8:
                raise RuntimeError("invalid training quantum kernel")
            save_json({'symmetric': symmetric, 'unit_diagonal': diagonal, 'min_eigenvalue': eigen_min,
                       'shape': list(K_train.shape), 'wires': kernel.wires, 'layers': kernel.layers},
                      output / 'metrics' / f'{tag}_kernel_validation.json')
            model, params = train_quantum(K_train, y[train], kernel.matrix(X[val], X[train]), y[val],
                                         config['c_grid'], config['seed'])
            test_input = kernel.matrix(X[test], X[train])
        predicted = model.predict(test_input)
        metrics = evaluate(y[test], predicted, model.decision_function(test_input))
        result = {'seed': config['seed'], 'model': model_name, 'num_observables': len(features),
                  'features': list(features), **metrics, 'hyperparameters': params}
        save_json(result, output / 'metrics' / f'{tag}_{model_name}.json')
        pd.DataFrame({'sample_id': df.iloc[test].sample_id.to_numpy(), 'true_label': y[test],
                      'predicted_label': predicted, 'decision_score': model.decision_function(test_input)}
                     ).to_csv(output / 'metrics' / f'{tag}_{model_name}_predictions.csv', index=False)
        bundle = {'model': model, 'features': list(features), 'model_type': model_name,
                  'quantum_config': config['quantum'], 'X_train': X[train] if model_name == 'quantum' else None}
        joblib.dump(bundle, output / 'models' / f'{tag}_{model_name}.joblib')
        if figures:
            plot_confusion(metrics['confusion_matrix'], f'{model_name.title()} SVM ({len(features)} observables)',
                           output / 'figures' / f'{tag}_{model_name}_confusion')
        fitted[model_name] = bundle
        rows.append(result)
        print(f"seed={config['seed']} {tag} {model_name}: accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
    return rows, fitted


def predict_bundle(bundle, X):
    """Predict from observable vectors in the bundle's saved feature order."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(bundle['features']):
        raise ValueError("feature shape differs from saved model")
    if bundle['model_type'] == 'quantum':
        from .quantum_kernel import QuantumKernel
        X = QuantumKernel(X.shape[1], **bundle['quantum_config']).matrix(X, bundle['X_train'])
    return bundle['model'].predict(X)


def measurement_budget(config, df, split):
    from .plotting import plot_study
    rows = []
    for features in config['feature_sets']:
        result, _ = fit_evaluate(config, df, split, features,
                                 figures=features == config['primary_features'])
        rows.extend(result)
    _, output = paths(config)
    table = pd.DataFrame(rows)
    table.drop(columns=['confusion_matrix', 'hyperparameters']).to_csv(output / 'metrics' / 'measurement_budget.csv', index=False)
    plot_study(table, 'num_observables', 'Measurement budget (fixed grouped split)',
               output / 'figures' / 'measurement_budget')
    return table


def noise_study(config, df, split):
    from .plotting import plot_study
    features = config['primary_features']
    _, output = paths(config)
    per_noise = {}
    for p in config['noise_levels']:
        mask = np.isclose(df.noise_level.to_numpy(), p)
        per_noise[p] = {name: idx[mask[idx]] for name, idx in split.items()}
    _, clean_models = fit_evaluate(config, df, per_noise[0.0], features, tag='noise_clean_train')
    rows = []
    X = feature_matrix(df, features)
    for p in config['noise_levels']:
        test = per_noise[p]['test']
        if len(test) == 0:
            raise ValueError(f"no test groups at noise level {p}")
        for model_name, bundle in clean_models.items():
            metrics = evaluate(df.iloc[test].label.to_numpy(), predict_bundle(bundle, X[test]))
            rows.append({'seed': config['seed'], 'model': model_name, 'protocol': 'clean_train',
                         'noise_level': p, **metrics})
        if all(df.iloc[per_noise[p][name]].label.nunique() == 2 for name in ['train', 'validation']):
            result, _ = fit_evaluate(config, df, per_noise[p], features, tag=f'noise_matched_{p:g}')
            for row in result:
                rows.append({**row, 'protocol': 'matched_noise', 'noise_level': p})
        else:
            save_json({'status': 'skipped', 'reason': 'Only one class remains in training or validation at this noise'},
                      output / 'metrics' / f'noise_matched_{p:g}_skipped.json')
    table = pd.DataFrame(rows)
    table.drop(columns=['confusion_matrix', 'hyperparameters'], errors='ignore').to_csv(output / 'metrics' / 'noise_study.csv', index=False)
    for protocol, subset in table.groupby('protocol'):
        plot_study(subset, 'noise_level', f'Classification noise robustness: {protocol}',
                   output / 'figures' / f'classification_noise_{protocol}')
    fraction = df.groupby('noise_level').agg(entangled_fraction=('label', 'mean'),
                                            mean_negativity=('negativity', 'mean'), samples=('label', 'size'))
    fraction.to_csv(output / 'metrics' / 'entanglement_noise_summary.csv')
    return table


def create_report(configs):
    """Create a report only from actual saved runs, never from expected scores."""
    from .plotting import plot_study
    root = Path(configs[0]['output_dir'])
    reports = []
    for name, x in [('measurement_budget', 'num_observables'), ('noise_study', 'noise_level')]:
        tables = [pd.read_csv(root / f"seed_{c['seed']}" / 'metrics' / f'{name}.csv') for c in configs]
        merged = pd.concat(tables, ignore_index=True)
        merged.to_csv(root / f'{name}_all_seeds.csv', index=False)
        grouping = ['model', x] + (['protocol'] if name == 'noise_study' else [])
        aggregate = merged.groupby(grouping)[['accuracy', 'precision', 'recall', 'f1']].agg(['mean', 'std']).fillna(0)
        aggregate.to_csv(root / f'{name}_aggregate.csv')
        reports.append(f'### {name.replace("_", " ").title()}\n\n```text\n{aggregate.round(4).to_string()}\n```')
        if name == 'measurement_budget':
            plot_study(merged, x, 'Across seeds: mean ± sample standard deviation', root / 'figures' / name)
        else:
            for protocol, subset in merged.groupby('protocol'):
                plot_study(subset, x, f'Across seeds: {protocol}', root / 'figures' / f'{name}_{protocol}')
    text = '# Entanglement robustness: experimental report\n\n'
    text += 'GHZ reduced-pair negativity is 0; ideal W negativity is (√5 − 1)/6 ≈ 0.206011. '
    text += 'Qubit 0 is the leftmost bit. Noise mixes with I/8 before loss, equivalently I/4 after loss.\n\n'
    text += f"Seeds: {[c['seed'] for c in configs]}; {configs[0]['num_samples']} samples per seed. "
    text += 'Noise variants share base-state groups across the 70/15/15 split. '
    text += 'Both models receive the same observable subset and split. Hyperparameters maximize validation F1; '
    text += 'test data are used only for the final evaluation. StandardScaler fits training inputs for the RBF model. '
    text += 'The quantum model uses θ = πx without learned scaling.\n\n'
    text += 'Quantum feature states are simulated in PennyLane default.qubit; exact squared overlaps form the kernel. '
    text += 'At most four wires are used, with higher dimensional vectors reuploaded in sequential chunks. '
    text += 'The one-chunk final CNOT chain cancels in the fidelity; its presence alone does not demonstrate a kernel advantage. '
    text += 'The full 15-observable budget is a tomography reference; smaller budgets are the limited-measurement task.\n\n'
    text += '\n\n'.join(reports)
    text += '\n\nLimitations: noiseless analytic measurement estimation by default; balanced clean-label family sampling; '
    text += 'small numbers of independent base states; structured simulated families; no hardware or quantum-advantage claim. '
    text += 'Rows at different noise levels are correlated within a base state. Seed dispersion is not a confidence interval.\n'
    (root / 'report.md').write_text(text)
    slides = '# Entanglement under qubit loss\n\nGHZ/W physics and limited-observable classification\n\n---\n\n'
    slides += '## Physical result\n\nGHZ: N = 0 after loss. W: N ≈ 0.206011 after loss. White noise destroys surviving entanglement.\n\n---\n\n'
    slides += '## Protocol\n\nPartial trace → PPT label + Pauli features → grouped splits → validation tuning → test metrics.\n\n---\n\n'
    slides += '## Model comparison\n\nClassical RBF SVM versus PennyLane fidelity-kernel SVM. See report.md and aggregate CSVs for measured scores.\n\n---\n\n'
    slides += '## Measurement and noise studies\n\nBudgets: 2, 3, 4, 6, 9, 15. Protocols: clean training and matched-noise training.\n\n---\n\n'
    slides += '## Interpretation\n\nCompare scores without assuming quantum superiority. Limited observables may not determine entanglement uniquely.\n'
    (root / 'presentation.md').write_text(slides)


def main(default_stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = ['validate', 'physics', 'dataset', 'classical', 'quantum', 'budget', 'noise', 'all']
    if default_stage is None:
        parser.add_argument('stage', choices=stages)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'config.yaml'))
    parser.add_argument('--seed', type=int, help='Run one seed instead of configured seeds')
    parser.add_argument('--samples', type=int, help='Override dataset size; regenerate before training')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    for key in ['output_dir', 'data_dir']:
        config[key] = str((config_path.parent / config[key]).resolve())
    if args.samples is not None:
        config['num_samples'] = args.samples
    stage = default_stage or args.stage
    seeds = [args.seed] if args.seed is not None else (config['seeds'] if stage == 'all' else [config['seed']])
    completed = []
    for seed in seeds:
        current = deepcopy(config)
        current['seed'] = seed
        if stage == 'validate':
            validate_states(current)
            continue
        if stage == 'physics':
            physics_demo(current)
            continue
        if stage == 'all':
            physics_demo(current)
        if stage in ['dataset', 'all']:
            df, split = prepare_dataset(current)
            if stage == 'dataset':
                continue
        else:
            df, split = load_dataset(current)
        if stage in ['classical', 'quantum']:
            fit_evaluate(current, df, split, current['primary_features'], models=(stage,),
                         tag='primary', figures=True)
        if stage in ['budget', 'all']:
            measurement_budget(current, df, split)
        if stage in ['noise', 'all']:
            noise_study(current, df, split)
        completed.append(current)
    if stage == 'all':
        create_report(completed)


if __name__ == '__main__':
    main()
