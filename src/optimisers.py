"""Classical optimizer wrappers for VQE.

Provides a uniform interface so VQE can swap optimizers without
changing any other code. Same idea as using a common API for
SGD/Adam/AdaGrad in PyTorch.
"""

import numpy as np
from scipy.optimize import minimize


def run_scipy_optimizer(objective, initial_params, method="COBYLA", maxiter=200):
    """Wrap scipy optimizers (COBYLA, L-BFGS-B, Nelder-Mead, etc.)."""
    history = []

    def tracked_objective(params):
        energy = objective(params)
        history.append(energy)
        return energy

    result = minimize(
        tracked_objective,
        initial_params,
        method=method,
        options={"maxiter": maxiter},
    )

    return {
        "optimal_energy": result.fun,
        "optimal_params": result.x,
        "history": history,
        "num_evals": len(history),
        "optimizer": method,
    } 

def run_spsa(objective, initial_params, maxiter=200, lr=0.1, perturb=0.1):
    """Simultaneous Perturbation Stochastic Approximation.

    Why SPSA matters for quantum:
    - Estimates gradient with only 2 circuit evaluations regardless
      of parameter count (vs 2N for parameter-shift rule).
    - Designed for noisy cost functions — which is exactly what you
      get from quantum hardware.

    Args:
        objective: Cost function f(params) -> energy.
        initial_params: Starting parameter values.
        maxiter: Number of optimization steps.
        lr: Learning rate (step size). Decays over iterations.
        perturb: Perturbation size for gradient estimate. Also decays.
    """
    params = initial_params.copy()
    history = []

    for k in range(maxiter):
        # Decay learning rate and perturbation (standard SPSA schedule)
        ak = lr / (k + 1) ** 0.602
        ck = perturb / (k + 1) ** 0.101

        # Random perturbation direction (±1 for each parameter)
        delta = np.random.choice([-1, 1], size=len(params))

        # Evaluate cost at two perturbed points
        f_plus = objective(params + ck * delta)
        f_minus = objective(params - ck * delta)

        # Estimate gradient
        gradient = (f_plus - f_minus) / (2 * ck * delta)

        # Update parameters
        params = params - ak * gradient

        # Track the current energy
        current_energy = objective(params)
        history.append(current_energy)

    return {
        "optimal_energy": history[-1],
        "optimal_params": params,
        "history": history,
        "num_evals": maxiter * 3,  # 3 evaluations per step (plus, minus, current)
        "optimizer": "SPSA",
    }

def get_optimizer(name: str):
    """Get an optimizer function by name.

    Returns a function with signature:
        optimizer(objective, initial_params, maxiter) -> result dict

    This lets VQE swap optimizers without changing any other code.
    """
    optimizers = {
        "COBYLA": lambda obj, p0, m: run_scipy_optimizer(obj, p0, "COBYLA", m),
        "L-BFGS-B": lambda obj, p0, m: run_scipy_optimizer(obj, p0, "L-BFGS-B", m),
        "Nelder-Mead": lambda obj, p0, m: run_scipy_optimizer(obj, p0, "Nelder-Mead", m),
        "SPSA": lambda obj, p0, m: run_spsa(obj, p0, m),
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")

    return optimizers[name]

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.hamiltonian import build_h2_hamiltonian
    from src.ansatz import build_ansatz
    from qiskit.primitives import StatevectorEstimator

    # Setup
    hamiltonian, exact_energy = build_h2_hamiltonian()
    ansatz = build_ansatz(num_qubits=hamiltonian.num_qubits)
    estimator = StatevectorEstimator()

    def objective(params):  # input angles, output energy
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run([pub]).result()
        return result[0].data.evs[0]

    initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # Run each optimizer- tries different angles to find the lowest energy and tracks convergence
    print(f"Exact energy: {exact_energy:.6f} Ha\n")
    for name in ["COBYLA", "SPSA"]:
        opt = get_optimizer(name)
        result = opt(objective, initial_params.copy(), 200)
        error = abs(result["optimal_energy"] - exact_energy)
        print(f"{name:12s} | Energy: {result['optimal_energy']:.6f} | "
              f"Error: {error:.6f} | Evals: {result['num_evals']}")