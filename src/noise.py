"""Noise model construction for simulating real quantum hardware.

Uses Qiskit Aer to add realistic noise to circuit simulation.
This lets us test optimizer robustness without burning real hardware time.
"""

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.primitives import StatevectorEstimator, BackendEstimatorV2


def build_noise_model(
    single_gate_error: float = 0.001,
    two_gate_error: float = 0.01,
    readout_error: float = 0.02,
):
    """Build a simple noise model mimicking real hardware.

    Args:
        single_gate_error: Error rate for single-qubit gates (Ry, Rz).
            Real IBM devices: ~0.03-0.1%.
        two_gate_error: Error rate for two-qubit gates (CNOT).
            Real IBM devices: ~0.5-2%. CNOTs are ~10x noisier than single gates.
        readout_error: Probability of reading 0 as 1 or vice versa.
            Real IBM devices: ~1-5%.

    Returns:
        NoiseModel that can be passed to AerSimulator.

    Notes:
        Depolarizing error = with probability p, replace the qubit state
        with a random state. Simplest noise model — not perfectly realistic
        but captures the key effect: gates are imperfect.
    """
    noise_model = NoiseModel()

    # Single-qubit gate errors (applied to Ry, Rz, etc.)
    single_error = depolarizing_error(single_gate_error, 1)
    noise_model.add_all_qubit_quantum_error(single_error, ["ry", "rz", "rx"])

    # Two-qubit gate errors (applied to CNOTs — the noisiest gates)
    two_error = depolarizing_error(two_gate_error, 2)
    noise_model.add_all_qubit_quantum_error(two_error, ["cx"])

    # Readout errors (measurement mistakes)
    # p(read 1 | true 0) = readout_error, p(read 0 | true 1) = readout_error
    read_err = ReadoutError(
        [[1 - readout_error, readout_error],
         [readout_error, 1 - readout_error]]
    )
    noise_model.add_all_qubit_readout_error(read_err)

    return noise_model


def get_noisy_estimator(noise_model=None, shots=1024, seed=None):
    """Create an estimator that simulates with noise.

    Args:
        noise_model: NoiseModel to use. If None, builds default.
        shots: Number of measurement repetitions. More shots = less
               statistical noise but more circuit evaluations.
               Real hardware typically uses 1024-8192 shots.
        seed: Random seed for the simulator. Different seeds give
              different noise realizations. None = random each time.

    Returns:
        Estimator backed by noisy AerSimulator.
    """
    if noise_model is None:
        noise_model = build_noise_model()

    # Build an AerSimulator backend with our noise model baked in.
    # Using BackendEstimatorV2 (from qiskit.primitives) instead of
    # Aer's own EstimatorV2, because Aer's version caches results
    # and ignores seed changes.
    backend = AerSimulator(noise_model=noise_model)
    if seed is not None:
        backend.set_options(seed_simulator=seed)

    # default_precision controls shots internally:
    # precision ≈ 1/sqrt(shots), so 1/sqrt(1024) ≈ 0.03
    precision = 1.0 / (shots ** 0.5)

    noisy_estimator = BackendEstimatorV2(
        backend=backend,
        options={
            "default_precision": precision,
            "seed_simulator": seed if seed is not None else 0,
        },
    )

    return noisy_estimator


if __name__ == "__main__":
    from hamiltonian import build_h2_hamiltonian
    from ansatz import build_ansatz
    from qiskit import transpile
    import numpy as np

    hamiltonian, exact_energy = build_h2_hamiltonian()
    ansatz = build_ansatz(num_qubits=hamiltonian.num_qubits)

    # Decompose ansatz into basic gates (Ry, CNOT, etc.)
    # StatevectorEstimator does this automatically, but Aer needs it explicit
    ansatz_decomposed = ansatz.decompose()

    # Compare ideal vs noisy energy for the same parameters
    params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # Ideal
    ideal_estimator = StatevectorEstimator()
    pub = (ansatz, [hamiltonian], [params])
    ideal_energy = ideal_estimator.run([pub]).result()[0].data.evs[0]

    # Noisy (run 5 times to show variance from shot noise + gate noise)
    # Each run uses a different seed → different noise realization
    print(f"Exact ground state:  {exact_energy:.6f} Ha")
    print(f"Ideal sim energy:    {ideal_energy:.6f} Ha")
    print(f"\nNoisy sim energies (same params, 5 runs):")
    for i in range(5):
        noisy_estimator = get_noisy_estimator(shots=1024, seed=42 + i)
        pub = (ansatz_decomposed, [hamiltonian], [params])
        noisy_energy = noisy_estimator.run([pub]).result()[0].data.evs[0]
        print(f"  Run {i}: {noisy_energy:.6f} Ha (diff from ideal: {abs(noisy_energy - ideal_energy):.6f})")