"""Benchmark script — run all optimizers × multiple seeds.

This is the experiment runner. It produces the data for your
convergence plots and comparison tables.
"""

import json
import numpy as np
import sys
sys.path.insert(0, ".")

from src.hamiltonian import build_h2_hamiltonian
from src.ansatz import build_ansatz
from src.vqe import run_vqe


def run_benchmark(
    optimizer_names=("COBYLA", "SPSA"),
    seeds=(0, 1, 2, 3, 4),
    maxiter=200,
    reps=2,
    save_path="results/benchmark.json",
):
    """Run VQE with each optimizer × each seed, collect results.

    Multiple seeds because random initial params affect convergence.
    Same as running ML training with different random seeds to check stability.
    """
    hamiltonian, exact_energy = build_h2_hamiltonian()
    ansatz = build_ansatz(num_qubits=hamiltonian.num_qubits, reps=reps)

    all_results = []

    for opt_name in optimizer_names:
        for seed in seeds:
            np.random.seed(seed)
            print(f"Running {opt_name} | seed {seed}...", end=" ")

            result = run_vqe(ansatz, hamiltonian, optimizer_name=opt_name, maxiter=maxiter)
            error = abs(result["optimal_energy"] - exact_energy)

            record = {
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
    # Convert numpy types to Python types for JSON serialization
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
    print(f"\n{'Optimizer':12s} | {'Mean Energy':>12s} | {'Mean Error':>10s} | {'Std Error':>10s} | {'Mean Evals':>10s}")
    print("-" * 65)
    for opt_name in optimizer_names:
        opt_results = [r for r in all_results if r["optimizer"] == opt_name]
        energies = [r["final_energy"] for r in opt_results]
        errors = [r["error"] for r in opt_results]
        evals = [r["num_evals"] for r in opt_results]
        print(f"{opt_name:12s} | {np.mean(energies):12.6f} | {np.mean(errors):10.6f} | {np.std(errors):10.6f} | {np.mean(evals):10.1f}")


if __name__ == "__main__":
    run_benchmark()