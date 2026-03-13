import sys
sys.path.insert(0, ".")
from qiskit.primitives import StatevectorEstimator
import numpy as np
from src.optimisers import get_optimizer
from src.hamiltonian import build_h2_hamiltonian
from src.ansatz import build_ansatz

def cost_function(params, ansatz, hamiltonian, estimator):
    """Compute ⟨ψ(θ)|H|ψ(θ)⟩ — the expected energy for given parameters.
    On an ideal simulator, this gives the exact expectation value. On real hardware / noisy sim,
    it gives a noisy estimate (that's where optimiser robustness matters)
    This is the 'loss function' of VQE.
    """
    # Bind the parameter values to the circuit
    pub = (ansatz, [hamiltonian], [params])

    # Run the circuit and compute expectation value
    result = estimator.run([pub]).result()
    energy = result[0].data.evs[0]

    return energy


def run_vqe(ansatz, hamiltonian, optimizer_name="COBYLA", maxiter=200, estimator=None):
    """Run VQE with a given optimiser.

    Args:
        ansatz: Parameterised circuit from build_ansatz().
        hamiltonian: SparsePauliOp from build_h2_hamiltonian().
        optimizer_name: Which classical optimiser ("COBYLA", "SPSA", "L-BFGS-B").
        maxiter: Max optimiser iterations.
        estimator: Qiskit Estimator to use. If None, uses ideal StatevectorEstimator.
                   Pass a noisy estimator from noise.py to simulate hardware.

    Returns:
        result: dict with final energy, optimal params, and convergence history.
    """
    if estimator is None:
        estimator = StatevectorEstimator()

    # History tracking is the optimiser's responsibility (see optimisers.py).
    # Pass a plain objective; the optimiser wraps it and records each evaluation.
    def objective(params):
        return cost_function(params, ansatz, hamiltonian, estimator)

    # Random initial parameters (like random weight init in ML)
    initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # Run the optimiser
    # Use the optimiser module instead of inline scipy
    optimizer = get_optimizer(optimizer_name)
    result = optimizer(objective, initial_params, maxiter)

    return result


if __name__ == "__main__":
    # Step 1: Build problem
    hamiltonian, exact_energy = build_h2_hamiltonian()
    ansatz = build_ansatz(num_qubits=hamiltonian.num_qubits, reps=2)

    # Step 2: Run VQE
    result = run_vqe(ansatz, hamiltonian, optimizer_name="COBYLA", maxiter=200)

    # Step 3: Report
    print(f"Exact energy:    {exact_energy:.6f} Ha")
    print(f"VQE energy:      {result['optimal_energy']:.6f} Ha")
    print(f"Error:           {abs(result['optimal_energy'] - exact_energy):.6f} Ha")
    print(f"Evaluations:     {result['num_evals']}")
    print(f"Optimiser:       {result['optimizer']}")
