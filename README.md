# Entanglement robustness under qubit loss

Implementation of `Quantum_Entanglement_Robustness_Project_Documentation.docx`: simulate three-qubit state families, discard one qubit, label surviving two-qubit entanglement with PPT negativity, and compare RBF and quantum-kernel SVMs using Pauli expectations.

**Execution status:** implementation is provided; main experiments, dataset generation, SVM training, and quantum-kernel evaluation have not been run. Metrics and figures are produced only when you explicitly run the experiment commands below. The primary environment is **qisk**.

## Environment

```bash
conda activate qisk
python --version
```

The existing environment contains the dependencies. `requirements.txt` records its direct package versions. Install those only when recreating the environment:

```bash
python -m pip install -r requirements.txt
```

The local interpreter can also be invoked directly as `/home/iyanshi/anaconda3/envs/qisk/bin/python`. Python 3.10 was used for the small unit checks. No environment changes are required to use this project.

## Choose which qubit to remove and display PPT eigenvalues

Run the standalone interactive demonstration:

```bash
python qubit_loss_ppt/check_ppt.py
```

Choose GHZ, W, or both and removal position 1 (first), 2 (second), or 3
(third). It displays the PPT eigenvalues, whether a negative eigenvalue is
present, and whether the remaining pair is entangled. For no prompts, use
`python qubit_loss_ppt/check_ppt.py --state both --remove 2`.
See [the standalone script instructions](qubit_loss_ppt/README.md) for the
calculation, expected results, and source functions. Only small matrices are
evaluated; no dataset generation or model training is involved.

## Small unit checks

The following command checks physics and split logic without generating a dataset, training a classifier, or executing a quantum feature circuit:

```bash
python -m pytest tests/test_states.py tests/test_density.py tests/test_partial_trace.py tests/test_entanglement.py tests/test_observables.py tests/test_noise.py tests/test_preprocessing.py
```

`tests/test_dataset.py` and `tests/test_quantum_kernel.py` are deferred integration checks. A plain `python -m pytest` includes those and therefore generates small datasets and evaluates tiny quantum circuits. It still does not train either SVM.

## Experiments to run when ready

All commands below are **opt-in**. Importing modules does not execute experiments.

```bash
python experiments/01_validate_states.py
python experiments/02_qubit_loss_demo.py
python experiments/03_generate_dataset.py
python experiments/04_classical_baseline.py
python experiments/05_quantum_kernel.py
python experiments/06_measurement_budget.py
python experiments/07_noise_study.py
```

Run dataset generation before the ML stages. For a development dataset, pass `--samples 100` to generation and every subsequent stage, so saved configuration checks agree. Stage scripts default to seed 42; `--seed 7` selects another seed.

The unified runner supports `validate`, `physics`, `dataset`, `classical`, `quantum`, `budget`, `noise`, and `all`. A complete three-seed study and generated report can be requested explicitly:

```bash
python run_experiments.py all
# Or a complete single-seed development run:
python run_experiments.py all --seed 42 --samples 100
```

`--config path/to/config.yaml` selects another configuration. Output paths are resolved relative to the configuration file. Full runs use seeds 42, 7, and 123, each with 400 rows and 80 independent base states. The report aggregates per-seed scores using the mean and sample standard deviation.

## Scientific protocol

- Computational-basis order is `|q0 q1 q2⟩`, with q0 the leftmost bit. Partial traces preserve the order of retained qubits.
- GHZ, W, random product, and Bell-pair-plus-spectator families receive Haar-random local U(2) rotations. Biseparable Bell pairs and lost qubits are sampled across all positions. Labels derive from the reduced state, never family identity.
- Clean-label rejection sampling alternates separable and entangled base states. Each base generates five noise variants. At the default noise levels the dataset is balanced; wider noise ranges or optional generic states can change noisy-label balance. The balancing changes the family distribution and is documented in saved summaries.
- White noise is `(1-p)ρ + pI/d`. Applying it before loss is equivalent to mixing the reduced state with `I/4`. Full three-qubit density matrices are not stored as model inputs. Base kets are saved separately for audit.
- Negativity is the sum of absolute negative eigenvalues of the two-qubit partial transpose. Labels use `N > 1e-8`.
- Only explicitly selected Pauli columns enter either model. Budgets are 2, 3, 4, 6, 9, and 15. Fifteen independent nontrivial expectations provide a tomography reference; smaller subsets implement the limited-measurement study.
- All variants sharing a `base_state_id` remain in the same split. Group-level stratification produces 56 training, 12 validation, and 12 test base states at default settings. The resulting 60 test rows contain only 12 independent bases.
- RBF standardization fits training data only. C and gamma are chosen by validation F1. Quantum C uses the same validation criterion. Both models use the same split and features. Models remain fitted to training data; validation is reserved for tuning.
- The quantum map applies Hadamards, `RY(πx)` encodings, and a nearest-neighbor CNOT chain. The default has one layer and at most four wires. For larger budgets, features are reuploaded in chunks on those wires; `quantum.layers` and `quantum.max_wires` are configurable.
- PennyLane `default.qubit` returns exact feature statevectors, and NumPy computes `|⟨φ(x)|φ(y)⟩|²`. This simulator optimization avoids a separate circuit for every pair. It remains the same fidelity kernel; it is not hardware execution. Matrices are cached by input contents, shape, circuit settings, and PennyLane version.
- For a single encoding chunk, the common trailing CNOT chain cancels in the overlap. The compact circuit does not by itself imply a nonclassical similarity or quantum advantage. More layers can be investigated separately.
- Noise classification uses two protocols: tune on clean training/validation groups and test at every noise level, or train and tune at each matched noise level. Test base-state groups remain disjoint from training across the whole sweep. Degenerate matched-noise datasets with only one remaining class are explicitly recorded as skipped.
- Optional `measurement_noise_std` adds clipped Gaussian estimation noise; it is distinct from state noise. `include_generic: true` adds generic random pure states to the training distribution; it is not a held-out-family generalization experiment.

The noiseless sanity values are GHZ = 0 and W = `(sqrt(5)-1)/6 ≈ 0.206011`. Under white noise, W negativity is `max(0, (1-p)N_W - p/4)` and vanishes around p = 0.451768. The default classification sweep ends at p = 0.3; the separate physics sweep reaches p = 1.

## Outputs

| Location | Contents after explicit execution |
|---|---|
| `data/processed/seed_<seed>/` | Dataset CSV, base statevectors, split indices, generation configuration |
| `results/seed_<seed>/metrics/` | Physics tables, dataset summary, provenance, scores, predictions, kernel checks |
| `results/seed_<seed>/figures/` | PNG and PDF physics, class distribution, confusion, budget, and noise plots |
| `results/seed_<seed>/kernels/` | Content-addressed exact kernel matrices |
| `results/seed_<seed>/models/` | Joblib inference bundles, including quantum reference training vectors |
| `results/` | Cross-seed CSVs, aggregate figures, generated `report.md`, and `presentation.md` |

Load a model you have trained and provide Pauli values in the saved order:

```python
import joblib
from src.workflow import predict_bundle

bundle = joblib.load('results/seed_42/models/primary_classical.joblib')
print(bundle['features'])
labels = predict_bundle(bundle, [[0.2, 0.2, -0.1, 0.0]])
```

`primary_*` bundles come from individual classical/quantum stages. The budget stage writes `budget_4_*` for the default primary feature subset. Quantum bundles additionally contain the training feature reference required for precomputed-kernel predictions.

## Repository

`src/` contains the physics, data, kernels, evaluation, plots, and orchestration modules. `experiments/` has the seven requested executable stages. `tests/` separates small unit checks from deferred dataset/kernel integration checks. `docs/implementation_report.md` maps the specification to the implementation; `docs/presentation.md` provides a presentation outline while measured results are pending.

Limited Pauli information does not necessarily uniquely determine entanglement. Performance depends on the sampled distribution. A classical model can outperform the quantum kernel; this is a valid outcome. No measured scores or superiority claims are supplied before executing the experiments.

PennyLane implementation references: [state output](https://docs.pennylane.ai/en/stable/code/api/pennylane.state.html), [quantum circuit construction](https://docs.pennylane.ai/en/stable/introduction/circuits.html), and [fidelity kernels](https://www.pennylane.ai/demos/tutorial_kernels_module).
