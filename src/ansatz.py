"""Ansatz (parameterized circuit) construction for VQE.

The ansatz is the quantum circuit whose parameters θ we optimize.
Think of it as the model architecture — we're choosing the family
of quantum states VQE can search over.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, EfficientSU2


def build_ansatz(
    num_qubits: int = 4,
    reps: int = 2,
    ansatz_type: str = "real_amplitudes",
) -> QuantumCircuit:
    """Build a parameterized quantum circuit (ansatz) for VQE.

    Args:
        num_qubits: Number of qubits (must match Hamiltonian).
        reps: Number of repeated layers. More reps = more expressive
              but deeper circuit (more noise on real hardware).
        ansatz_type: Which ansatz template to use.
            - "real_amplitudes": Ry rotations + CNOTs. Simplest.
            - "efficient_su2": Ry + Rz rotations + CNOTs. More expressive.
            - Reasons:
            - Benchmark VQE with different ansatz types to see how it affects optimisation and results.
            - Pass ansatz as config option
            - Add more ansatz types later (hardware-efficient, problem-inspired, etc.)

    Returns:
        QuantumCircuit with unbound parameters.
    """
    if ansatz_type == "real_amplitudes":
        ansatz = RealAmplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement="linear",  # Each qubit entangled with neighbor (q0↔q1, q1↔q2, q2↔q3)
        )
    elif ansatz_type == "efficient_su2":
        ansatz = EfficientSU2(
            num_qubits=num_qubits,
            reps=reps,
            entanglement="linear",  # Real hardware doesn't have all-to-all connectivity so use linear entanglement.
        )
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")

    return ansatz


if __name__ == "__main__":
    # Build and inspect the default ansatz
    ansatz = build_ansatz()

    print(f"Ansatz type: RealAmplitudes")
    print(f"Number of qubits: {ansatz.num_qubits}")
    print(f"Number of parameters: {ansatz.num_parameters}")
    print(f"Circuit depth: {ansatz.depth()}")
    print(f"\nParameter names: {[p.name for p in ansatz.parameters]}")
    print(f"\nCircuit diagram:")
    print(ansatz.draw())

    # Compare with EfficientSU2
    ansatz2 = build_ansatz(ansatz_type="efficient_su2")
    print(f"\n--- EfficientSU2 ---")
    print(f"Number of parameters: {ansatz2.num_parameters}")
    print(f"Circuit depth: {ansatz2.depth()}")
    print(ansatz2.draw())