from qiskit.primitives import StatevectorEstimator
import numpy as np
from optimisers import get_optimizer
from hamiltonian import build_h2_hamiltonian
from ansatz import build_ansatz

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
    """Run VQE with a given optimizer.
    
    Args:
        ansatz: Parameterized circuit from build_ansatz().
        hamiltonian: SparsePauliOp from build_h2_hamiltonian().
        optimizer_name: Which classical optimizer ("COBYLA", "SPSA", "L-BFGS-B").
        maxiter: Max optimizer iterations.
        estimator: Qiskit Estimator to use. If None, uses ideal StatevectorEstimator.
                   Pass a noisy estimator from noise.py to simulate hardware.
        
    Returns:
        result: dict with final energy, optimal params, and convergence history.
    """
    if estimator is None:
        estimator = StatevectorEstimator()
    
    # Track energy at each step (for plotting convergence later)
    history = []
    
    def objective(params):
        energy = cost_function(params, ansatz, hamiltonian, estimator)
        history.append(energy)
        return energy
    
    # Random initial parameters (like random weight init in ML)
    initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    
    # Run the optimizer
    # Use the optimizer module instead of inline scipy
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
    print(f"Optimizer:       {result['optimizer']}")