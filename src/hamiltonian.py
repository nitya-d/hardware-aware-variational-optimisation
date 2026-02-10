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

    Pre-computed from STO-3G basis via parity mapping with two-qubit reduction.
    This gives a 2-qubit Hamiltonian where the smallest eigenvalue directly
    equals the physical ground state energy (~-1.137 Ha).

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

    # 2-qubit H₂ Hamiltonian (parity mapping with two-qubit reduction).
    # This is the standard form used in VQE tutorials.
    # Nuclear repulsion energy (0.7200 Ha) is folded into the II term
    # so that the smallest eigenvalue directly equals the total ground state energy.
    #
    # With 4 qubits (Jordan-Wigner), some eigenvalues are unphysical
    # (wrong electron count). The 2-qubit reduced form avoids this —
    # the smallest eigenvalue IS the physical ground state.
    qubit_op = SparsePauliOp.from_list([
        ("II", -0.33240425132388),    # electronic energy + nuclear repulsion
        ("IZ",  0.39793742484318),
        ("ZI", -0.39793742484318),
        ("ZZ", -0.01128010425624),
        ("XX",  0.18093119978423),
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