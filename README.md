# Hardware-Aware Variational Optimisation

Exploring hardware-aware variational quantum algorithms for optimisation and machine learning on near-term quantum devices.

Build a hybrid optimization pipeline where an ML model learns to drive VQE/QAOA parameter updates, conditioned on circuit noise and structure — and benchmark it rigorously against standard optimizers on both simulators and real IBM Quantum hardware.

## Table of Contents

- [Hardware-Aware Variational Optimisation](#hardware-aware-variational-optimisation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
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

**Problem instance:** H₂ molecular ground state energy via VQE — the canonical benchmark. Results are easily validated against the known exact solution (-1.137306 Ha), and nobody asks "why quantum?" for molecular ground state energy.

**Phases:**

1. VQE baseline with standard optimizers (COBYLA, SPSA) — **done**
2. ML optimizer (MLP/LSTM) that learns from optimization trajectory
3. Condition on noise metrics (gate error, T1/T2, circuit depth)
4. Benchmark across ideal sim → noisy sim → real IBM hardware

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

We use **two** noise strategies that serve different purposes:

**1. Custom noise model** (`build_noise_model`) — 3 tunable knobs: single-gate error, two-gate error, readout error.
- Run **controlled experiments**: "how does COBYLA degrade as CNOT error increases from 1% to 5%?"
- **Isolate** individual noise sources (e.g. only readout error, zero gate error)
- **Reproducible** results not tied to any specific IBM device
- This is what makes the project "hardware-aware" — sweeping noise parameters is the whole point

**2. IBM fake backends** (e.g. `FakeManilaV2`) — real device calibration data (non-uniform gate errors, T1/T2 relaxation, crosstalk, qubit connectivity). Good for **realistic validation** but can't easily sweep noise levels since it's a fixed snapshot.

| Environment | Purpose |
|---|---|
| Ideal (no noise) | Upper bound — best possible performance |
| Custom noise, low (0.1% / 1% / 2%) | Controlled experiment — mild noise |
| Custom noise, high (1% / 5% / 10%) | Controlled experiment — harsh noise |
| FakeBackend (optional) | Realistic validation against real noise profile |

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

SPSA underperformed everywhere — not because it's a bad algorithm, but because:
1. **Too few iterations** — 200 steps wasn't enough for stochastic gradient estimates to converge.
2. **Problem too small** — SPSA's O(2) gradient cost advantage doesn't matter with only 6 parameters.
3. **Hyperparameters too aggressive** — learning rate decayed too fast.

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

1. **Noise is the bottleneck, not the optimizer.** Once both converge, they hit the same noise floor (~0.04 Ha for low noise, ~0.25 Ha for high noise). No classical optimizer can do better than the noise allows.
2. **COBYLA is 15× more efficient.** Same accuracy in ~70-100 evals vs SPSA's 1200. On real hardware (where each eval costs time and money), that matters.
3. **Both degrade equally under noise.** The 0.25 Ha error under high noise is the measurement noise distorting the energy landscape — neither optimizer can see through it.
4. **This motivates Phase 2.** Can a learned optimizer either (a) break through the noise floor by learning noise patterns, or (b) match COBYLA's accuracy with fewer evaluations while being more noise-robust?

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
- This is the core contribution that makes the project novel