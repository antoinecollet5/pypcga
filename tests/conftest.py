"""Shared fixtures for a small, fast synthetic inverse problem.

The forward model and sensitivity matrix are defined at module level (not as
closures) so that they remain picklable -- this is required to exercise the
``ProcessPoolExecutor``-based multiprocessing code path in
``PCGA.linear_iteration``.
"""

import covmats
import numpy as np
import pytest

S_DIM = 8
N_OBS = 6
N_PC = 4

_rng = np.random.default_rng(0)
# Mildly nonlinear (quadratic term) forward-model sensitivity matrices.
A_LIN = _rng.normal(size=(N_OBS, S_DIM))
A_QUAD = 0.01 * _rng.normal(size=(N_OBS, S_DIM))


def quadratic_forward_model(s_ensemble: np.ndarray) -> np.ndarray:
    """A mildly nonlinear forward model, module-level so it can be pickled."""
    return A_LIN @ s_ensemble + A_QUAD @ (s_ensemble**2)


def linear_forward_model(s_ensemble: np.ndarray) -> np.ndarray:
    """A purely linear forward model, module-level so it can be pickled."""
    return A_LIN @ s_ensemble


@pytest.fixture()
def small_problem():
    """Return kwargs for a small, fast PCGA problem (no solver constructed)."""
    prior_cov = covmats.CovViaDiagonal(np.ones(S_DIM))
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=N_PC, random_state=0)
    s_true = _rng.normal(loc=1.0, scale=1.0, size=(S_DIM, 1))
    obs = quadratic_forward_model(s_true)
    cov_obs = covmats.CovViaDiagonal(np.full(N_OBS, 0.05))
    s_init = np.zeros((S_DIM, 1))
    return {
        "s_init": s_init,
        "obs": obs,
        "cov_obs": cov_obs,
        "forward_model": quadratic_forward_model,
        "Q": Q,
        "s_true": s_true,
    }
