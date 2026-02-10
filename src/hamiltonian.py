"""Hamiltonian construction for molecular systems.

Hardcoded H₂ Hamiltonian (STO-3G basis, Jordan-Wigner mapping).
PySCF doesn't build easily on Windows, so we use pre-computed Pauli coefficients.
These are the same values Qiskit Nature / PySCF would produce.

If you later install PySCF (e.g. via conda on Linux), you can swap this out
for the dynamic version using PySCFDriver + JordanWignerMapper.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def build_h2_hamiltonian(bond_length: float = 0.735):
    """Build the H₂ qubit Hamiltonian.

    Pre-computed from STO-3G basis via Jordan-Wigner mapping.
    Jordan-Wigner is the simplest mapping: one qubit per electron orbital.
    For H₂ with STO-3G this gives 4 qubits.

    Args:
        bond_length: Distance between H atoms in Angstroms.
                     Only 0.735 (equilibrium) is currently supported.
                     Other bond lengths need PySCF to compute.

    Returns:
        qubit_op: SparsePauliOp — the Hamiltonian as a sum of Pauli terms.
        exact_energy: float — exact ground state energy in Hartree.

    Notes:
        - Hartree is the energy unit used in quantum chemistry (~27.2 eV).
        - The exact ground state energy of H₂ at 0.735 Å is about -1.137 Ha.
        - STO-3G = "Slater Type Orbital with 3 Gaussians" — the simplest
          basis set approximation for electron orbitals.
    """
    if bond_length != 0.735:
        raise NotImplementedError(
            f"Bond length {bond_length} not supported without PySCF. "
            "Only 0.735 Å (equilibrium) is hardcoded."
        )

    # Each tuple is (Pauli string, coefficient).
    # Pauli string is read right-to-left: "IZII" means Z on qubit 2.
    # These encode the electron-electron and electron-nucleus interactions.
    qubit_op = SparsePauliOp.from_list([
        ("IIII", -0.8105479805373266),
        ("IIIZ",  0.17218393261915552),
        ("IIZI", -0.22575349222402472),
        ("IZII",  0.17218393261915552),
        ("ZIII", -0.22575349222402472),
        ("IIZZ",  0.12091263261776641),
        ("IZIZ",  0.16892753870087912),
        ("IZZI",  0.04523279994605785),
        ("ZIIZ",  0.04523279994605785),
        ("ZIZI",  0.16892753870087912),
        ("ZZII",  0.17464343068300453),
        ("XXYY", -0.04523279994605785),
        ("XYYX",  0.04523279994605785),
        ("YXXY",  0.04523279994605785),
        ("YYXX", -0.04523279994605785),
    ])

    # Exact ground state energy (classically computed eigenvalue)
    exact_energy = np.linalg.eigvalsh(qubit_op.to_matrix())[0] # weighted sum of above 16x16 matrices

    return qubit_op, exact_energy


if __name__ == "__main__":
    qubit_op, exact_energy = build_h2_hamiltonian()
    print(f"Number of qubits: {qubit_op.num_qubits}")
    print(f"Number of Pauli terms: {len(qubit_op)}")
    print(f"Exact ground state energy: {exact_energy:.6f} Ha")
    print(f"\nHamiltonian:\n{qubit_op}")