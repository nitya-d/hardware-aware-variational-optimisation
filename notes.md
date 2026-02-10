# Learning Notes

Notes on quantum computing concepts encountered while building this project. Written from the perspective of an ML/SWE engineer — heavy on analogies to classical ML.

## Table of Contents

- [Hamiltonians](#hamiltonians)
- [VQE (Variational Quantum Eigensolver)](#vqe-variational-quantum-eigensolver)
- [QAOA](#qaoa-quantum-approximate-optimization-algorithm)
- [Optimizers](#optimizers)
- [Benchmarking Strategy](#benchmarking-strategy)
- [Ansatz](#ansatz)

---

## Hamiltonians

A Hamiltonian is a matrix that encodes the "energy" of a system. Its smallest eigenvalue is the system's ground state energy (lowest possible energy), and the corresponding eigenvector is the state that achieves it.

Think of it like a **loss function**, but expressed as a matrix. The "best solution" to whatever problem is being encoded is the eigenvector with the smallest eigenvalue.

**Scale:** For n qubits, the Hamiltonian is a 2ⁿ×2ⁿ matrix. 10 qubits → 1024×1024. 50 qubits → ~10¹⁵ entries. This is why `numpy.linalg.eigh()` is infeasible at scale — the matrix is exponentially large.

**The key insight:** Many useful problems can be rephrased as "find the smallest eigenvalue of this matrix." The matrix is constructed from the problem's rules and constraints.

### Where do Hamiltonians come from?

**Molecules (chemistry):** Physics dictates the rules. Electrons interact via Coulomb forces. These interactions are written down mathematically, then discretized into qubit operators using transformations (Jordan-Wigner, Bravyi-Kitaev). Libraries like Qiskit Nature handle this automatically — specify "H₂ with bond length 0.735 Å" and it produces the Hamiltonian matrix.

**Mental model:** A Hamiltonian is to quantum computing what a loss function is to ML — the thing being minimised. It happens to be expressed as a matrix because quantum mechanics operates on vectors and matrices (linear algebra). The word "energy" is physicist speak; in CS terms, it's just "cost."

---

## VQE (Variational Quantum Eigensolver)

VQE is gradient descent on a quantum circuit. Direct analogy to ML training:

| ML Training | VQE |
|---|---|
| Neural network with weights | Quantum circuit with rotation angles (parameters θ) |
| Forward pass → prediction | Run circuit → measure quantum state |
| Loss function | Hamiltonian expectation value ⟨ψ(θ)\|H\|ψ(θ)⟩ (energy) |
| Backprop → gradients | Parameter shift rule → gradients |
| Optimizer updates weights | Optimizer updates rotation angles |
| Converges to low loss | Converges to ground state energy |

**The loop:**
1. Build a **parameterized circuit** (ansatz) — rotation gates Ry(θ₁), Rz(θ₂), etc. Think: model architecture.
2. Run the circuit, measure output.
3. Compute expected energy ⟨ψ(θ)\|H\|ψ(θ)⟩ — this is the loss.
4. Classical optimizer updates θ to lower the energy.
5. Repeat until converged.

**Output:** minimum energy (≈ smallest eigenvalue) and the parameters that produce it.

### Expectation value ⟨ψ(θ)|H|ψ(θ)⟩

The expected energy of the state the circuit produces:
- |ψ(θ)⟩ = the quantum state the circuit outputs for angles θ
- H = the Hamiltonian matrix
- ⟨ψ(θ)| = conjugate transpose of |ψ(θ)⟩

In NumPy: `state.conj().T @ H_matrix @ state` → one scalar (the energy). It's a weighted average of H's eigenvalues. If the state matches the ground state eigenvector, the result is the smallest eigenvalue. VQE adjusts θ until this number is minimised.

### Why "variational"?

The **variational principle** states: for any state |ψ⟩, ⟨ψ|H|ψ⟩ ≥ true ground state energy. It is impossible to undershoot. So "lower = closer to truth" is mathematically guaranteed. VQE *varies* the parameters θ to find the lowest energy, approaching the ground state from above.

---

## QAOA (Quantum Approximate Optimization Algorithm)

QAOA is a special case of VQE for **combinatorial optimization** (MaxCut, scheduling, portfolio, etc.):
- Fixed ansatz structure: alternating "problem" and "mixer" layers
- Only 2p parameters (p = number of layers)
- Problem must already be expressed as a Hamiltonian (cost function → Pauli operators)

VQE = general purpose, the ansatz is a design choice. QAOA = structured ansatz designed specifically for optimization problems.

---

## Optimizers

### Parameter updates

Same as weight updates in ML: **θ(t+1) = θ(t) − η · g(θ(t))**

Different optimizers compute the update direction g differently:

| Optimizer | How it works | Quantum advantage |
|---|---|---|
| **COBYLA** | Gradient-free. Builds linear approximation of cost function from function values only. | Popular for VQE — no gradient circuits needed. Can be slow. |
| **SPSA** | Wiggles *all* parameters simultaneously (±Δ), estimates full gradient from 2 evaluations — regardless of parameter count. | The key quantum optimizer. Designed for noisy functions. |
| **Adam** | Gradient estimates + momentum + adaptive learning rates. | Standard ML optimizer, less common in quantum. |

**SPSA's efficiency advantage:**

| Method | Evaluations per step (12 params) |
|---|---|
| Finite differences / parameter shift | 24 |
| SPSA | 2 |

SPSA is rarely used in classical ML because backprop gives exact gradients for free. On quantum hardware, there is no backprop — gradients must be estimated by running circuits.

### Two levels of optimization in this project

1. **Inner loop (VQE itself):** Minimise energy of H₂ → find ground state. Standard task.
2. **Outer loop (the project's contribution):** Optimise the *optimizer*. Train an ML model that produces better parameter updates than COBYLA/SPSA/Adam, especially under hardware noise.

---

## Benchmarking Strategy

The comparison is between **optimizers**, not quantum vs classical computing. The question: *"Can a learned optimizer navigate the noisy VQE landscape better than hand-designed ones?"*

### Metrics

| Metric | Meaning |
|---|---|
| **Final energy** | How close to the true ground state? Lower = better |
| **Convergence speed** | How many circuit evaluations to get there? Fewer = better (circuits are expensive) |
| **Robustness to noise** | Does the optimizer still work under realistic hardware noise? |
| **Stability** | Consistent convergence or wild oscillation? |

### Environments

| Environment | Purpose |
|---|---|
| **Ideal simulator** (no noise) | Upper bound — best possible performance |
| **Noisy simulator** (calibration data) | Realistic test — does the optimizer handle noise? |
| **Real IBM hardware** | Ground truth — does it actually work? |

For each environment, all optimizers are run, convergence curves are plotted, and final energies are compared.

### The big picture

```
Known answer (exact eigenvalue of H₂)
         ↑
         |  How close does it get?
         |
    [VQE loop]
    circuit(θ) → measure → energy → optimizer → new θ → repeat
         |                              ↑
         |                   COBYLA? SPSA? Adam? ML MODEL?
         |
    [Run on what?]
    ideal sim │ noisy sim │ real hardware
```

---

## Ansatz

The ansatz is the **parameterized quantum circuit** — the model architecture of VQE.

| ML | VQE |
|---|---|
| Model architecture (e.g. 3-layer MLP) | Ansatz (e.g. RealAmplitudes circuit) |
| Number of layers / neurons | Number of circuit layers / rotation gates |
| Weights (learned) | Rotation angles θ (learned) |
| Too small → underfits | Too shallow → can't reach ground state |
| Too big → overfits / hard to train | Too deep → barren plateaus / noise kills it |

### Why is it needed?

The ansatz restricts the search to a family of states parameterised by a few angles θ. It's analogous to choosing an MLP architecture — the search is not over all possible functions, just those representable by the chosen model.

### Structure: Rotation + Entanglement

```
     ┌──────────┐     ┌──────────┐
q0:  │ Ry(θ₀)   │──■──│ Ry(θ₄)   │──■──
     ├──────────┤  │  ├──────────┤  │
q1:  │ Ry(θ₁)   │──⊕──│ Ry(θ₅)   │──⊕──
     ├──────────┤  │  ├──────────┤  │
q2:  │ Ry(θ₂)   │──■──│ Ry(θ₆)   │──■──
     ├──────────┤  │  ├──────────┤  │
q3:  │ Ry(θ₃)   │──⊕──│ Ry(θ₇)   │──⊕──
     └──────────┘     └──────────┘
      Layer 1           Layer 2
```

- **Ry(θ)** = rotation gate. The tunable knob.
- **CNOT (■──⊕)** = entanglement. Without it, qubits are independent and cannot represent correlated states (e.g. electron correlations in molecules).

### Why entanglement matters

Without CNOTs, each qubit is independent — like assuming all features are uncorrelated. Real molecules have electron correlations, so the quantum state must be entangled. Without entanglement, VQE literally cannot represent the H₂ ground state.

### Common ansatz templates

| Template | Gates | Notes |
|---|---|---|
| **RealAmplitudes** | Ry + CNOTs | Simple, good for chemistry. Used in this project. |
| **EfficientSU2** | Ry + Rz + CNOTs | More expressive. |
| **UCCSD** | Physics-inspired | Very accurate but very deep circuits. |
| **HardwareEfficient** | Device-native gates | Less noise on real hardware. |

### The depth tradeoff

This is the core tension of the project:

| | Few parameters | Many parameters |
|---|---|---|
| Optimization | Faster | Slower |
| Noise impact | Less (fewer gates) | More (more gates) |
| Expressiveness | May miss ground state | More likely to reach it |
| Gradients | Clear signal | Risk of barren plateaus |

On noisy hardware, fewer parameters can actually be *better*. The ML optimizer should learn to navigate this tradeoff.

### Why H₂?

H₂ serves as the MNIST of quantum chemistry — the answer is known, so optimizer quality can be measured precisely. For larger molecules, classical computation is exponentially hard and VQE is genuinely needed. Verification then relies on the variational principle (VQE ≥ true energy), classical approximations, and experimental data.
