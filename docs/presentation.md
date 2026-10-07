# Detecting entanglement survival under qubit loss

GHZ versus W with limited Pauli measurements and quantum-kernel SVM classification

---

## Research question

Can a model infer surviving bipartite entanglement from incomplete measurements of the pair remaining after one qubit is lost?

---

## Physical motivation

GHZ: `( |000⟩ + |111⟩ )/√2` loses pairwise entanglement after any qubit is discarded.

W: `( |001⟩ + |010⟩ + |100⟩ )/√3` retains two-qubit negativity `(√5−1)/6 ≈ 0.206011`.

---

## Simulation and labels

Generate state → local rotations → white noise → partial trace.

The reduced density matrix supplies a PPT negativity label internally. Models receive selected Pauli expectations only.

---

## Dataset and fair comparison

Four structured families; 400 rows per seed; five noise variants per base; 70/15/15 group split.

Train-only classical scaling. Validation-only hyperparameter selection. Identical inputs and partitions for both SVMs.

---

## Quantum kernel

Encode `πx` with Hadamard, RY and CNOT gates. Compute `K(x,y)=|⟨φ(x)|φ(y)⟩|²` with PennyLane simulation.

Classical SVM optimization uses the precomputed kernel. Larger budgets reupload features on at most four wires.

---

## Experiment plan

Loss validation; negativity versus noise; dataset; RBF baseline; quantum-kernel model; measurement budgets; clean and matched-noise classification.

Three configured seeds provide a measure of score variation.

---

## Results to insert after an authorized run

Use the generated loss/noise plots, both confusion matrices, cross-model metric table, budget curves, and noise curves. Empirical accuracy and F1 are pending; no scores are assumed.

---

## Interpretation and limits

Incomplete observables can be ambiguous. Structured simulated data and a small number of independent base states limit generalization. Exact expectation values exclude finite-shot effects by default.

A classical model may perform better. Quantum superiority is not required for a scientifically useful robustness study.
