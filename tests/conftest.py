import pytest
import numpy as np
from qiskit.primitives import StatevectorEstimator
from src.hamiltonian import build_h2_hamiltonian
from src.ansatz import build_ansatz


@pytest.fixture
def hamiltonian():
    op, _ = build_h2_hamiltonian()
    return op


@pytest.fixture
def exact_energy():
    _, e = build_h2_hamiltonian()
    return e


@pytest.fixture
def ansatz():
    # reps=1 keeps the circuit shallow for fast integration tests
    return build_ansatz(num_qubits=2, reps=1)


@pytest.fixture
def ideal_estimator():
    return StatevectorEstimator()
