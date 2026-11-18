# Choose a qubit and check PPT eigenvalues

From the project root, with your Python environment active, run:

```bash
python qubit_loss_ppt/check_ppt.py
```

Choose `ghz`, `w`, or `both`, then choose which position to remove:

- `1`: first (leftmost) qubit in `|q0 q1 q2>`; internal index 0.
- `2`: second (middle) qubit; internal index 1.
- `3`: third (rightmost) qubit; internal index 2.

There is no default removal position: the script asks unless `--remove` is supplied.
For a command without prompts:

```bash
python qubit_loss_ppt/check_ppt.py --state both --remove 1
python qubit_loss_ppt/check_ppt.py --state w --remove 2
python qubit_loss_ppt/check_ppt.py --state ghz --remove 3
```

The script prints the removed and retained positions, all four eigenvalues of
the partially transposed reduced density matrix, a True/False negative-eigenvalue
check, and a True/False entanglement result with a text explanation.

## Calculation and interpretation

The calculation constructs the ideal normalized statevector, forms
`rho = |psi><psi|`, discards the selected qubit by partial trace, and partially
transposes the second surviving qubit. It diagonalizes that matrix using
`numpy.linalg.eigvalsh`. Eigenvalues below `-1e-10` count as negative to avoid
classifying numerical roundoff as entanglement.

For two qubits, a negative partial-transpose eigenvalue establishes entanglement;
a positive semidefinite partial transpose establishes separability. The numerical
decision here uses the stated tolerance. This describes the **remaining pair**,
not the original three-qubit state's entanglement. Partial trace represents
discarding a qubit without conditioning on a measurement outcome.

Ideal GHZ and W are each symmetric under qubit permutations. Therefore choosing
different removal positions gives the same spectrum within each of these states:

```text
GHZ: [0.  0.  0.5 0.5]
Negative eigenvalue present: False
Remaining pair entangled: False
Result: Separable (not entangled)

W: [-0.20601133  0.33333333  0.33333333  0.53934466]
Negative eigenvalue present: True
Remaining pair entangled: True
Result: Entangled
```

The W eigenvalues are `(1-sqrt(5))/6`, `1/3`, `1/3`, and
`(1+sqrt(5))/6`. GHZ reduces to an equal mixture of `|00>` and `|11>`.

## Existing implementation reused

- `../src/states.py`: `ghz_state()` and `w_state()`.
- `../src/density.py`: `ket_to_density()`.
- `../src/qubit_loss.py`: `trace_out_qubit()` accepts internal indices 0, 1, 2.
- `../src/entanglement.py`: `ppt_eigenvalues()` calls `partial_transpose()`.

Only NumPy and the local source files are needed. This standalone demonstration
does not generate datasets or train models.
