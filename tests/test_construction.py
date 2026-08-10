import logging

import covmats
import numpy as np
import pypcga
import pytest

from .conftest import N_PC, S_DIM, quadratic_forward_model


def _make_kwargs(n_obs: int, **overrides):
    prior_cov = covmats.CovViaDiagonal(np.ones(S_DIM))
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=N_PC, random_state=0)
    cov_obs = covmats.CovViaDiagonal(np.full(n_obs, 0.05))
    kwargs = dict(
        s_init=np.zeros((S_DIM, 1)),
        obs=np.zeros(n_obs),
        cov_obs=cov_obs,
        forward_model=quadratic_forward_model,
        Q=Q,
        maxiter=1,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "drift_factory",
    (
        lambda: None,
        lambda: covmats.ConstantDriftMatrix(S_DIM),
        lambda: covmats.LinearDriftMatrix(
            np.arange(S_DIM).reshape(-1, 1).astype(float)
        ),
    ),
)
def test_drift_options(drift_factory) -> None:
    solver = pypcga.PCGA(**_make_kwargs(n_obs=6, drift=drift_factory()))
    assert solver.drift.s_dim == S_DIM


def test_direct_solve_warning_for_many_observations() -> None:
    with pytest.warns(UserWarning, match="direct approach"):
        pypcga.PCGA(**_make_kwargs(n_obs=150, is_direct_solve=True))


def test_no_warning_for_direct_solve_with_few_observations(recwarn) -> None:
    pypcga.PCGA(**_make_kwargs(n_obs=6, is_direct_solve=True))
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "prior_s_var,expected_shape",
    (
        (2.0, (S_DIM,)),
        (np.full(S_DIM, 3.0), (S_DIM,)),
    ),
)
def test_prior_s_var_valid(prior_s_var, expected_shape) -> None:
    solver = pypcga.PCGA(**_make_kwargs(n_obs=6, prior_s_var=prior_s_var))
    assert solver.prior_s_var.shape == expected_shape
    np.testing.assert_allclose(
        solver.prior_s_var,
        np.ravel(prior_s_var)[0] * np.ones(S_DIM)
        if np.size(prior_s_var) == 1
        else prior_s_var,
    )


def test_prior_s_var_invalid_size() -> None:
    solver = pypcga.PCGA(**_make_kwargs(n_obs=6))
    with pytest.raises(ValueError, match="prior_s_var must be"):
        solver.prior_s_var = np.ones(S_DIM + 1)


def test_cov_obs_setter_invalid_type() -> None:
    solver = pypcga.PCGA(**_make_kwargs(n_obs=6))
    with pytest.raises(ValueError, match="covmats.CovarianceMatrix"):
        solver.cov_obs = np.eye(6)  # ty:ignore[invalid-assignment]


def test_display_init_parameters_with_line_search(caplog) -> None:
    logger = logging.getLogger("pcga-test-line-search")
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="pcga-test-line-search"):
        pypcga.PCGA(**_make_kwargs(n_obs=6, is_line_search=True, logger=logger))
    assert any("max_it_ls" in rec.message for rec in caplog.records)


def test_display_init_parameters_with_lm(caplog) -> None:
    logger = logging.getLogger("pcga-test-lm")
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="pcga-test-lm"):
        pypcga.PCGA(
            **_make_kwargs(
                n_obs=6, is_lm=True, lm_smin=-10.0, lm_smax=10.0, max_it_lm=2
            )
        )


def test_get_v0() -> None:
    solver = pypcga.PCGA(**_make_kwargs(n_obs=6, random_state=0))
    v0 = solver.get_v0(5)
    assert v0.shape == (5,)  # ty:ignore[unresolved-attribute]

    # get_v0's `random_state is None` branch is unreachable through normal
    # construction (check_random_state never returns None), but the method
    # itself still supports it defensively -- exercise it directly.
    solver.random_state = None
    assert solver.get_v0(5) is None
