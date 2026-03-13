import numpy as np
import pytest
from src.vqe import cost_function, run_vqe

REQUIRED_KEYS = {"optimal_energy", "optimal_params", "history", "num_evals", "optimizer"}


def test_cost_function_returns_scalar(hamiltonian, ansatz, ideal_estimator):
    # Confirms cost_function produces a single number, not an array or Qiskit result object
    params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    energy = cost_function(params, ansatz, hamiltonian, ideal_estimator)
    assert np.isscalar(energy)


def test_cost_function_is_bounded(hamiltonian, ansatz, ideal_estimator):
    # Energy must lie within the physical eigenvalue range of the H₂ Hamiltonian
    params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    energy = cost_function(params, ansatz, hamiltonian, ideal_estimator)
    assert -2.0 <= energy <= 1.0


def test_run_vqe_result_keys(hamiltonian, ansatz, ideal_estimator):
    # All callers (benchmark, ML pipeline) rely on a consistent result dict shape
    result = run_vqe(ansatz, hamiltonian, optimizer_name="COBYLA", maxiter=5,
                     estimator=ideal_estimator)
    assert REQUIRED_KEYS.issubset(result.keys())


def test_run_vqe_history_populated(hamiltonian, ansatz, ideal_estimator):
    # History must be recorded so convergence plots have data to draw
    result = run_vqe(ansatz, hamiltonian, optimizer_name="COBYLA", maxiter=5,
                     estimator=ideal_estimator)
    assert len(result["history"]) > 0


def test_run_vqe_params_shape(hamiltonian, ansatz, ideal_estimator):
    # Optimal params must match the ansatz so they can be plugged back into the circuit
    result = run_vqe(ansatz, hamiltonian, optimizer_name="COBYLA", maxiter=5,
                     estimator=ideal_estimator)
    assert result["optimal_params"].shape == (ansatz.num_parameters,)


@pytest.mark.slow
def test_run_vqe_cobyla_ideal_converges(hamiltonian, exact_energy, ideal_estimator):
    # End-to-end check that VQE actually finds the ground state on a noise-free simulator
    from src.ansatz import build_ansatz
    deep_ansatz = build_ansatz(num_qubits=2, reps=2)  # reps=2 matches benchmark config
    result = run_vqe(deep_ansatz, hamiltonian, optimizer_name="COBYLA", maxiter=200,
                     estimator=ideal_estimator)
    assert abs(result["optimal_energy"] - exact_energy) < 1e-4
