import pytest
from qiskit.circuit import QuantumCircuit
from src.ansatz import build_ansatz


def test_real_amplitudes_num_qubits():
    # Circuit must match the 2-qubit Hamiltonian or the estimator will reject it
    circuit = build_ansatz(num_qubits=2, reps=1)
    assert circuit.num_qubits == 2


def test_real_amplitudes_num_parameters_reps1():
    # RealAmplitudes parameter count is num_qubits * (reps + 1) = 2 * 2 = 4
    circuit = build_ansatz(num_qubits=2, reps=1)
    assert circuit.num_parameters == 4


def test_real_amplitudes_num_parameters_reps2():
    # Increasing reps adds another layer of Ry gates: 2 * (2 + 1) = 6
    circuit = build_ansatz(num_qubits=2, reps=2)
    assert circuit.num_parameters == 6


def test_efficient_su2_returns_circuit():
    # Checks the efficient_su2 branch of build_ansatz runs without error
    circuit = build_ansatz(num_qubits=2, reps=1, ansatz_type="efficient_su2")
    assert isinstance(circuit, QuantumCircuit)


def test_efficient_su2_has_more_params():
    # EfficientSU2 uses Ry + Rz per qubit vs Ry only, so must have more parameters
    ra = build_ansatz(num_qubits=2, reps=1, ansatz_type="real_amplitudes")
    esu2 = build_ansatz(num_qubits=2, reps=1, ansatz_type="efficient_su2")
    assert esu2.num_parameters > ra.num_parameters


def test_invalid_ansatz_type_raises():
    # Unknown ansatz types should fail loudly rather than silently use a default
    with pytest.raises(ValueError):
        build_ansatz(num_qubits=2, reps=1, ansatz_type="unknown")


def test_deeper_reps_increases_depth():
    # More reps means more gate layers — decompose first since blueprint circuits are lazy
    shallow = build_ansatz(num_qubits=2, reps=1).decompose()
    deep = build_ansatz(num_qubits=2, reps=2).decompose()
    assert deep.depth() > shallow.depth()
