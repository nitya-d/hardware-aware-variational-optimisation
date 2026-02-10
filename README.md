# Hardware-Aware Variational Optimisation

## Table of Contents

- [Hardware-Aware Variational Optimisation](#hardware-aware-variational-optimisation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Motivation](#motivation)
  - [Phases](#phases)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
  - [Usage](#usage)
  - [Noise Simulation](#noise-simulation)
  - [Phase 1 Results](#phase-1-results)
    - [Before SPSA tuning](#before-spsa-tuning)
    - [After SPSA tuning](#after-spsa-tuning)
    - [Key findings](#key-findings)
  - [Roadmap](#roadmap)

## Overview

Variational quantum algorithms like VQE (Variational Quantum Eigensolver) are among the most promising applications of near-term quantum computers. They work by iteratively tuning the parameters of a quantum circuit to minimise a cost function — conceptually identical to training a neural network, but with quantum circuits instead of layers of neurons and energy instead of loss.

The problem is that today's quantum hardware is noisy. Every gate operation introduces small errors, and measurements are imprecise. Standard classical optimizers (COBYLA, SPSA) were not designed to handle this kind of noise, and their performance degrades significantly on real devices. This creates a gap between the theoretical promise of variational algorithms and what they can actually achieve in practice.

This project investigates that gap. It benchmarks classical optimizers across ideal, mildly noisy, and heavily noisy simulation environments to quantify exactly how noise degrades optimizer performance. The ultimate goal is to build a **learned optimizer** — a small ML model (MLP or LSTM) that takes the history of energy evaluations and learns a parameter update rule that is more robust to hardware noise than hand-designed optimizers.

The benchmark problem is the ground state energy of the hydrogen molecule (H₂). This is the "MNIST of quantum computing" — the exact answer is known (-1.137306 Hartree), so optimizer quality can be measured precisely. The system is small enough (2 qubits) to simulate quickly, but the optimisation landscape has all the features that make real variational problems hard: local minima, noisy gradients, and sensitivity to circuit depth.

## Motivation

Classical optimizers treat the quantum circuit as a black-box function: put parameters in, get an energy value out. They have no awareness of *why* the energy estimate is noisy or *how* the noise is structured. A learned optimizer could potentially exploit patterns in the noise — for example, learning that high-noise environments require more conservative parameter updates, or that certain directions in parameter space are less affected by gate errors.

The key question this project aims to answer: **can a noise-aware learned optimizer outperform standard optimizers on noisy quantum hardware, either by achieving lower error or by converging with fewer circuit evaluations?**

## Phases

The project is structured in four phases, each building on the last:

**Phase 1 — VQE baseline with classical optimizers (complete).** Built the full VQE pipeline: Hamiltonian construction, parameterized quantum circuit (ansatz), optimisation loop, and noisy simulation. Benchmarked COBYLA and SPSA across three noise environments (ideal, low noise, high noise) with multiple random seeds for statistical robustness. Produced convergence plots and comparison tables. The key finding: both optimizers converge to the same noise-limited accuracy, but COBYLA is 15× more efficient in circuit evaluations.

**Phase 2 — ML optimizer.** Train a small neural network to predict VQE parameter updates from the optimisation trajectory (energy history, parameter history, gradient estimates). The model will learn an update rule end-to-end, replacing the hand-designed logic of COBYLA/SPSA.

**Phase 3 — Noise conditioning.** Extend the ML optimizer to take hardware noise metrics as additional inputs (gate error rates, readout error, circuit depth). This makes the optimizer "hardware-aware" — it adapts its strategy based on the noise profile of the device it's running on.

**Phase 4 — Validation.** Benchmark the learned optimizer against classical baselines across noise levels, and optionally on real IBM Quantum hardware. The goal is to demonstrate that noise-awareness translates to measurable improvements in convergence speed or final accuracy.

## Project Structure

```
src/
├── hamiltonian.py    # H₂ qubit Hamiltonian (2-qubit parity mapping)
├── ansatz.py         # Parameterized quantum circuits (RealAmplitudes, EfficientSU2)
├── vqe.py            # VQE optimization loop
├── optimisers.py     # Classical optimizers (COBYLA, SPSA, L-BFGS-B, Nelder-Mead)
├── noise.py          # Noise models and noisy estimator construction
├── benchmark.py      # Multi-optimizer × multi-seed × multi-environment experiment runner
└── visualise.py      # Convergence plots and comparison charts
results/
├── benchmark.json    # Raw benchmark data
├── convergence.png   # Energy vs iteration plots
├── final_energy_error.png
└── eval_count.png
notes.md              # Learning notes (Hamiltonians, VQE, QAOA, ansatz, optimizers)
```

## Getting Started

**Requirements:** Python 3.11+ (Anaconda recommended — Qiskit does not support 3.13)

```bash
pip install -r requirements.txt
```

**Dependencies:** qiskit, qiskit-aer, numpy, scipy, matplotlib

## Usage

**Run a single VQE optimization:**
```bash
python src/vqe.py
```

**Run the full benchmark** (2 optimizers × 5 seeds × 3 noise environments = 30 runs):
```bash
python src/benchmark.py
```

**Generate plots** from benchmark results:
```bash
python src/visualise.py
```

## Noise Simulation

Noise simulation is central to this project. The goal is to understand how optimizer performance degrades as hardware noise increases, and ultimately to build an optimizer that handles noise better.

The project uses two complementary noise strategies:

**Custom noise model** (`build_noise_model`) provides three tunable knobs: single-gate error rate, two-gate (CNOT) error rate, and readout error rate. This is the primary tool for controlled experiments — it answers questions like "how does COBYLA degrade as CNOT error increases from 1% to 5%?" and isolates individual noise sources. Results are reproducible and not tied to any specific IBM device.

**IBM fake backends** (e.g. `FakeManilaV2`) import real device calibration data including non-uniform gate errors, T1/T2 relaxation times, crosstalk, and qubit connectivity. These are useful for realistic validation but cannot easily be swept across noise levels since they represent a fixed snapshot of a real device.

| Environment | Purpose |
|---|---|
| Ideal (no noise) | Upper bound — best possible performance |
| Custom noise, low (0.1% / 1% / 2%) | Controlled experiment — mild noise |
| Custom noise, high (1% / 5% / 10%) | Controlled experiment — harsh noise |
| FakeBackend (optional) | Realistic validation against a real noise profile |

The custom model tells the story: "here's how each optimizer degrades across noise levels." The fake backend validates that story against reality.

## Phase 1 Results

Ran 2 optimizers × 5 seeds × 3 environments = 30 VQE runs on H₂ (2-qubit, RealAmplitudes ansatz, 6 parameters).

### Before SPSA tuning

| Environment | Optimizer | Mean Error (Ha) | Std Error | Mean Evals |
|---|---|---|---|---|
| ideal | COBYLA | 0.000000 | 0.000000 | 98.8 |
| ideal | SPSA | 0.482812 | 0.315214 | 600.0 |
| noisy_low | COBYLA | 0.041631 | 0.004655 | 70.2 |
| noisy_low | SPSA | 0.541160 | 0.304048 | 600.0 |
| noisy_high | COBYLA | 0.248935 | 0.010067 | 67.2 |
| noisy_high | SPSA | 0.676979 | 0.183904 | 600.0 |

SPSA initially underperformed everywhere. This was not because SPSA is a poor algorithm, but because of three configuration issues: too few iterations for its stochastic gradient estimates to converge, a learning rate schedule that decayed too aggressively, and the fact that SPSA's efficiency advantage (constant-cost gradient estimation) does not matter on a 6-parameter problem.

### After SPSA tuning

Tuned SPSA: `lr` 0.1→0.5, `perturb` 0.1→0.2, added stability constant=10, 600 iterations (3× COBYLA), track best-seen params.

| Environment | Optimizer | Mean Error (Ha) | Std Error | Mean Evals |
|---|---|---|---|---|
| ideal | COBYLA | 0.000000 | 0.000000 | 98.8 |
| ideal | SPSA | 0.001544 | 0.001047 | 1200.0 |
| noisy_low | COBYLA | 0.041631 | 0.004655 | 70.2 |
| noisy_low | SPSA | 0.044614 | 0.004719 | 1200.0 |
| noisy_high | COBYLA | 0.248935 | 0.010067 | 67.2 |
| noisy_high | SPSA | 0.251730 | 0.012600 | 1200.0 |

### Key findings

1. **Noise is the bottleneck, not the optimizer.** Once both optimizers converge, they hit the same noise floor — approximately 0.04 Ha error under low noise and 0.25 Ha under high noise. No classical optimizer can do better than the noise allows.
2. **COBYLA is 15× more efficient.** It achieves the same accuracy in ~70–100 circuit evaluations versus SPSA's 1200. On real hardware, where each evaluation costs time and money, this difference matters.
3. **Both degrade equally under noise.** Neither optimizer can see through the measurement noise distorting the energy landscape. They both converge to the same noise-limited floor.
4. **This motivates Phase 2.** The open question is whether a learned optimizer can either break through the noise floor by learning noise patterns, or match COBYLA's accuracy with fewer evaluations while being more noise-robust.

## Roadmap

- [x] Folder structure, dependencies
- [x] Build the Hamiltonian — H₂ molecule (2-qubit parity mapping)
- [x] Build the parameterized circuit (ansatz) — RealAmplitudes with tunable depth
- [x] Implement the VQE loop — circuit → measure → energy → update params → repeat
- [x] Run with classical optimizers — COBYLA, SPSA as baselines
- [x] Multi-optimizer benchmark — all optimizers × multiple seeds, save results
- [x] Add noisy simulation — custom noise model + noisy estimator
- [x] Benchmark & visualize — convergence plots, final energy comparisons
- [ ] Run on real IBM Quantum hardware
- [ ] Use real IBM device noise models (`NoiseModel.from_backend`)
- [ ] ML optimizer — train MLP/LSTM to predict parameter updates from optimization trajectory
- [ ] Condition on noise metrics (gate error rates, circuit depth, shot noise)
- [ ] Extend to other problems (LiH, MaxCut via QAOA, bond length sweeps)
- [ ] Production hardening (config files, CLI, logging, CI/CD, Docker)