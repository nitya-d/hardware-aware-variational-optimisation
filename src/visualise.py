"""Visualization module — load benchmark results and produce plots.

Generates:
  1. Convergence curves (energy vs iteration) per environment
  2. Final energy error bar chart (optimizer × environment)

Saves plots to results/ directory.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")


def load_results(path="results/benchmark.json"):
    """Load benchmark results from JSON."""
    with open(path) as f:
        return json.load(f)


def plot_convergence(results, save_dir="results"):
    """Plot convergence curves: energy vs iteration for each optimizer.

    One subplot per environment. Each optimizer gets its own colour.
    Multiple seeds are shown as thin lines with the mean as a thick line.
    """
    # Get unique environments and optimizers
    environments = sorted(set(r["environment"] for r in results))
    optimizers = sorted(set(r["optimizer"] for r in results))
    colours = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))

    fig, axes = plt.subplots(1, len(environments), figsize=(6 * len(environments), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, env in zip(axes, environments):
        ax.set_title(f"Environment: {env}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy (Ha)")

        # Plot exact energy as horizontal line
        exact_energy = results[0]["exact_energy"]
        ax.axhline(y=exact_energy, color="black", linestyle="--",
                    linewidth=1, label=f"Exact ({exact_energy:.4f})")

        for opt, colour in zip(optimizers, colours):
            # Get all runs for this optimizer + environment
            runs = [r for r in results
                    if r["optimizer"] == opt and r["environment"] == env]

            if not runs:
                continue

            # Plot individual seed runs as thin transparent lines
            for run in runs:
                ax.plot(run["history"], color=colour, alpha=0.2, linewidth=0.8)

            # Plot mean convergence as thick line
            # Pad shorter histories to the length of the longest
            max_len = max(len(r["history"]) for r in runs)
            padded = []
            for r in runs:
                h = r["history"]
                # Pad with final value if some runs are shorter
                padded.append(h + [h[-1]] * (max_len - len(h)))

            mean_history = np.mean(padded, axis=0)
            ax.plot(mean_history, color=colour, linewidth=2, label=f"{opt}")

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("VQE Convergence: Optimizer × Environment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = f"{save_dir}/convergence.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved convergence plot to {save_path}")
    plt.close()


def plot_final_energy(results, save_dir="results"):
    """Bar chart of final energy error by optimizer × environment.

    Each group of bars = one environment. Each bar = one optimizer.
    Error bars show std across seeds.
    """
    environments = sorted(set(r["environment"] for r in results))
    optimizers = sorted(set(r["optimizer"] for r in results))
    colours = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.8 / len(optimizers)
    x = np.arange(len(environments))

    for i, (opt, colour) in enumerate(zip(optimizers, colours)):
        means = []
        stds = []
        for env in environments:
            subset = [r for r in results
                      if r["optimizer"] == opt and r["environment"] == env]
            errors = [r["error"] for r in subset]
            means.append(np.mean(errors))
            stds.append(np.std(errors))

        offset = (i - len(optimizers) / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=opt, color=colour, alpha=0.85, capsize=3)

    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel("Energy Error |E_vqe - E_exact| (Ha)", fontsize=12)
    ax.set_title("Final Energy Error: Optimizer × Environment", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(environments)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = f"{save_dir}/final_energy_error.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved final energy plot to {save_path}")
    plt.close()


def plot_eval_count(results, save_dir="results"):
    """Bar chart of circuit evaluations by optimizer × environment.

    Shows computational cost — fewer evals = cheaper on real hardware.
    """
    environments = sorted(set(r["environment"] for r in results))
    optimizers = sorted(set(r["optimizer"] for r in results))
    colours = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.8 / len(optimizers)
    x = np.arange(len(environments))

    for i, (opt, colour) in enumerate(zip(optimizers, colours)):
        means = []
        stds = []
        for env in environments:
            subset = [r for r in results
                      if r["optimizer"] == opt and r["environment"] == env]
            evals = [r["num_evals"] for r in subset]
            means.append(np.mean(evals))
            stds.append(np.std(evals))

        offset = (i - len(optimizers) / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=opt, color=colour, alpha=0.85, capsize=3)

    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel("Circuit Evaluations", fontsize=12)
    ax.set_title("Computational Cost: Optimizer × Environment", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(environments)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = f"{save_dir}/eval_count.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved eval count plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    results = load_results()
    plot_convergence(results)
    plot_final_energy(results)
    plot_eval_count(results)
    print("\nAll plots saved to results/")
