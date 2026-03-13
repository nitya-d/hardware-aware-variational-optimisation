import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from src.hamiltonian import build_h2_hamiltonian


def test_returns_correct_types():
    # Confirms the function returns the expected (SparsePauliOp, float) pair
    op, energy = build_h2_hamiltonian()
    assert isinstance(op, SparsePauliOp)
    assert isinstance(energy, float)


def test_num_qubits(hamiltonian):
    # H₂ with parity mapping + two-qubit reduction must be a 2-qubit operator
    assert hamiltonian.num_qubits == 2


def test_num_pauli_terms(hamiltonian):
    # The hardcoded H₂ Hamiltonian has exactly 5 Pauli terms (II, IZ, ZI, ZZ, XX)
    assert len(hamiltonian) == 5


def test_exact_energy_value(exact_energy):
    # Ground state energy at equilibrium bond length should match the known STO-3G value
    assert abs(exact_energy - (-1.137306)) < 1e-6


def test_exact_energy_is_minimum(hamiltonian, exact_energy):
    # exact_energy should equal the smallest eigenvalue of the Hamiltonian matrix
    eigenvalues = np.linalg.eigvalsh(hamiltonian.to_matrix())
    assert abs(eigenvalues[0] - exact_energy) < 1e-10


def test_unsupported_bond_length_raises():
    # Only 0.735 Å is hardcoded — any other bond length needs PySCF and must raise
    with pytest.raises(NotImplementedError):
        build_h2_hamiltonian(bond_length=0.5)
