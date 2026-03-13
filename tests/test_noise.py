import numpy as np
import pytest
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer.noise import NoiseModel
from src.noise import build_noise_model, get_noisy_estimator


def test_build_noise_model_default():
    # Checks the default noise model constructs without error using preset error rates
    model = build_noise_model()
    assert isinstance(model, NoiseModel)


def test_build_noise_model_custom_params():
    # Checks custom error rates are accepted, since benchmark uses non-default values
    model = build_noise_model(
        single_gate_error=0.005,
        two_gate_error=0.02,
        readout_error=0.01,
    )
    assert isinstance(model, NoiseModel)


def test_get_noisy_estimator_returns_estimator():
    # Confirms the factory returns the right estimator type that VQE expects
    estimator = get_noisy_estimator()
    assert isinstance(estimator, BackendEstimatorV2)


@pytest.mark.slow
def test_noisy_estimator_seeded_reproducible(hamiltonian, ansatz):
    # Same seed must give consistent results so benchmark runs are reproducible
    circuit = ansatz.decompose()  # Aer needs explicit Ry/CNOT gates, not high-level gates
    params = np.zeros(ansatz.num_parameters)

    def run_once(seed):
        est = get_noisy_estimator(shots=1024, seed=seed)
        pub = (circuit, [hamiltonian], [params])
        return est.run([pub]).result()[0].data.evs[0]

    e1 = run_once(42)
    e2 = run_once(42)
    assert abs(e1 - e2) < 0.01
