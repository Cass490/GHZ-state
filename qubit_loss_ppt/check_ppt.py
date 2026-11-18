"""Choose a qubit to discard and inspect the surviving pair's PPT spectrum."""
import argparse
from pathlib import Path
import sys

import numpy as np

# Allow execution by file path, including from outside the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.density import ket_to_density
from src.entanglement import ppt_eigenvalues
from src.qubit_loss import trace_out_qubit
from src.states import ghz_state, w_state


def prompt_choice(prompt, choices):
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer
        print("Please choose one of: " + ", ".join(choices))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=str.lower, choices=("ghz", "w", "both"))
    parser.add_argument("--remove", type=int, choices=(1, 2, 3),
                        help="Position to remove: 1 = leftmost, 3 = rightmost")
    args = parser.parse_args()
    try:
        state = args.state or prompt_choice("State (ghz/w/both): ", ("ghz", "w", "both"))
        position = args.remove or int(prompt_choice(
            "Qubit to remove (1=first, 2=second, 3=third): ", ("1", "2", "3")))
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, "\nSelection cancelled. You can also supply --state and --remove.\n")

    constructors = {"ghz": ghz_state, "w": w_state}
    selected = constructors if state == "both" else {state: constructors[state]}
    retained = [str(p) for p in (1, 2, 3) if p != position]
    tolerance = 1e-10
    print(f"\nRemoved position: {position}; remaining positions: {', '.join(retained)}")
    print("PPT test on the remaining two-qubit state (no noise).")
    for name, constructor in selected.items():
        rho = ket_to_density(constructor())
        reduced_rho = trace_out_qubit(rho, position - 1)
        eigenvalues = ppt_eigenvalues(reduced_rho)
        negative = bool(np.any(eigenvalues < -tolerance))
        print(f"\nState: {name.upper()}")
        print("PPT eigenvalues:", np.array2string(eigenvalues, precision=8))
        print(f"Negative eigenvalue present: {negative}")
        print(f"Remaining pair entangled: {negative}")
        print("Result:", "Entangled" if negative else "Separable (not entangled)")
    print(f"\nNegative means eigenvalue < -{tolerance:g} (numerical tolerance).")


if __name__ == "__main__":
    main()
