import numpy as np
import pytest
from src.optimisers import run_scipy_optimizer, run_spsa, get_optimizer

REQUIRED_KEYS = {"optimal_energy", "optimal_params", "history", "num_evals", "optimizer"}

# Simple quadratic with known minimum at x=2
def quadratic(x):
    return (x[0] - 2.0) ** 2

INITIAL = np.array([0.0])


@pytest.mark.parametrize("name", ["COBYLA", "SPSA", "L-BFGS-B"])
def test_get_optimizer_returns_callable(name):
    # Registry should return a callable for every supported optimiser name
    opt = get_optimizer(name)
    assert callable(opt)


def test_get_optimizer_unknown_raises():
    # An unrecognised name should fail loudly so typos are caught early
    with pytest.raises(ValueError):
        get_optimizer("UNKNOWN")


@pytest.mark.parametrize("name", ["COBYLA", "SPSA"])
def test_result_dict_has_required_keys(name):
    # All optimisers must return the same dict shape so callers can treat them uniformly
    opt = get_optimizer(name)
    result = opt(quadratic, INITIAL.copy(), 50)
    assert REQUIRED_KEYS.issubset(result.keys())


def test_cobyla_minimises_quadratic():
    # Sanity-checks COBYLA finds the correct minimum on a trivial problem
    result = run_scipy_optimizer(quadratic, INITIAL.copy(), method="COBYLA", maxiter=200)
    assert abs(result["optimal_params"][0] - 2.0) < 0.01


def test_spsa_minimises_quadratic():
    # Sanity-checks SPSA finds the correct minimum (looser tolerance — SPSA is stochastic)
    np.random.seed(42)
    result = run_spsa(quadratic, INITIAL.copy(), maxiter=500)
    assert abs(result["optimal_params"][0] - 2.0) < 0.1


def test_history_is_populated():
    # Convergence plots depend on history being recorded — verify it's non-empty
    for fn in [
        lambda: run_scipy_optimizer(quadratic, INITIAL.copy(), method="COBYLA", maxiter=50),
        lambda: run_spsa(quadratic, INITIAL.copy(), maxiter=50),
    ]:
        result = fn()
        assert len(result["history"]) > 0


def test_cobyla_num_evals_matches_history():
    # num_evals is derived from history length — checks the two stay in sync
    result = run_scipy_optimizer(quadratic, INITIAL.copy(), method="COBYLA", maxiter=50)
    assert result["num_evals"] == len(result["history"])


def test_spsa_num_evals_is_2x_maxiter():
    # SPSA evaluates the objective twice per step (plus and minus perturbation)
    maxiter = 30
    result = run_spsa(quadratic, INITIAL.copy(), maxiter=maxiter)
    assert result["num_evals"] == maxiter * 2
