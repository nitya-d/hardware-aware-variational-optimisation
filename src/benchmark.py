"""Benchmark script — run all optimizers × multiple seeds × noise environments.

This is the experiment runner. It produces the data for your
convergence plots and comparison tables.

Dimensions:
  - Optimizers (COBYLA, SPSA, etc.)
  - Seeds (random initial params)
  - Environments (ideal, noisy_low, noisy_high)
"""

import json
import numpy as np
import sys
sys.path.insert(0, ".")

from src.hamiltonian import build_h2_hamiltonian
from src.ansatz import build_ansatz
from src.vqe import run_vqe
from src.noise import build_noise_model, get_noisy_estimator


# Noise environment definitions
# Each maps to (single_gate_error, two_gate_error, readout_error) or None for ideal
ENVIRONMENTS = {
    "ideal": None,
    "noisy_low": (0.001, 0.01, 0.02),    # Mild noise — good hardware
    "noisy_high": (0.01, 0.05, 0.10),     # Harsh noise — old/cheap hardware
}


def run_benchmark(
    optimizer_names=("COBYLA", "SPSA"),
    seeds=(0, 1, 2, 3, 4),
    environments=("ideal", "noisy_low", "noisy_high"),
    maxiter=200,
    reps=2,
    shots=1024,
    save_path="results/benchmark.json",
):
    """Run VQE with each optimizer × each seed × each environment.

    Args:
        optimizer_names: Which optimizers to benchmark.
        seeds: Random seeds for initial parameter reproducibility.
        environments: Which noise environments to test. Keys from ENVIRONMENTS dict.
        maxiter: Max optimizer iterations per run.
        reps: Ansatz repetitions (depth).
        shots: Shot count for noisy estimators.
        save_path: Where to save JSON results.
    """
    hamiltonian, exact_energy = build_h2_hamiltonian()
    ansatz_ideal = build_ansatz(num_qubits=hamiltonian.num_qubits, reps=reps)
    # Aer needs decomposed circuits (explicit Ry/CNOT gates)
    ansatz_decomposed = ansatz_ideal.decompose()

    all_results = []
    total = len(environments) * len(optimizer_names) * len(seeds)
    run_num = 0

    for env_name in environments:
        noise_params = ENVIRONMENTS[env_name]
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")

        for opt_name in optimizer_names:
            for seed in seeds:
                run_num += 1
                np.random.seed(seed)
                print(f"  [{run_num}/{total}] {opt_name} | seed {seed}...", end=" ")

                # Build the right estimator for this environment
                if noise_params is None:
                    estimator = None  # run_vqe defaults to ideal
                    ansatz = ansatz_ideal
                else:
                    noise_model = build_noise_model(*noise_params)
                    estimator = get_noisy_estimator(
                        noise_model=noise_model, shots=shots, seed=seed
                    )
                    ansatz = ansatz_decomposed

                # SPSA needs more iterations than COBYLA to converge
                iters = maxiter * 3 if opt_name == "SPSA" else maxiter

                result = run_vqe(
                    ansatz, hamiltonian,
                    optimizer_name=opt_name,
                    maxiter=iters,
                    estimator=estimator,
                )
                error = abs(result["optimal_energy"] - exact_energy)

                record = {
                    "environment": env_name,
                    "optimizer": opt_name,
                    "seed": seed,
                    "final_energy": result["optimal_energy"],
                    "exact_energy": exact_energy,
                    "error": error,
                    "num_evals": result["num_evals"],
                    "history": result["history"],
                }
                all_results.append(record)
                print(f"Energy: {result['optimal_energy']:.6f} | Error: {error:.6f}")

    # Save results
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    serializable_results = json.loads(
        json.dumps(all_results, default=to_serializable)
    )

    with open(save_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Print summary table
    print(f"\n{'Environment':12s} | {'Optimizer':12s} | {'Mean Energy':>12s} | {'Mean Error':>10s} | {'Std Error':>10s} | {'Mean Evals':>10s}")
    print("-" * 85)
    for env_name in environments:
        for opt_name in optimizer_names:
            subset = [r for r in all_results
                      if r["environment"] == env_name and r["optimizer"] == opt_name]
            energies = [r["final_energy"] for r in subset]
            errors = [r["error"] for r in subset]
            evals = [r["num_evals"] for r in subset]
            print(f"{env_name:12s} | {opt_name:12s} | {np.mean(energies):12.6f} | {np.mean(errors):10.6f} | {np.std(errors):10.6f} | {np.mean(evals):10.1f}")

    return all_results


if __name__ == "__main__":
    run_benchmark()