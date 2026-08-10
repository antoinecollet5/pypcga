import covmats
import numpy as np
import pypcga
import pytest

from .conftest import N_PC, S_DIM, linear_forward_model, quadratic_forward_model


def _make_kwargs(n_obs=6, n_pc=N_PC, **overrides):
    prior_cov = covmats.CovViaDiagonal(np.ones(S_DIM))
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=n_pc, random_state=0)
    cov_obs = covmats.CovViaDiagonal(np.full(n_obs, 0.05))
    kwargs = dict(
        s_init=np.full((S_DIM, 1), 0.1),
        obs=np.zeros(n_obs),
        cov_obs=cov_obs,
        forward_model=quadratic_forward_model,
        Q=Q,
        maxiter=2,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("is_direct_solve", (True, False))
def test_lm_with_bounds_not_all_violated(is_direct_solve) -> None:
    solver = pypcga.PCGA(
        **_make_kwargs(
            is_lm=True,
            is_direct_solve=is_direct_solve,
            max_it_lm=3,
            lm_smin=-1000.0,
            lm_smax=1000.0,
        )
    )
    s_hat, simul_obs, post_diagv, iter_best = solver.run()
    assert s_hat.shape == (S_DIM, 1)


@pytest.mark.parametrize("is_direct_solve", (True, False))
def test_lm_all_candidates_violate_bounds_falls_back(is_direct_solve, caplog) -> None:
    import logging

    logger = logging.getLogger("pcga-fallback-test")
    logger.setLevel(logging.INFO)
    solver = pypcga.PCGA(
        **_make_kwargs(
            is_lm=True,
            is_direct_solve=is_direct_solve,
            max_it_lm=3,
            # Impossible bounds: every candidate necessarily violates this.
            lm_smin=1.0,
            lm_smax=-1.0,
            logger=logger,
        )
    )
    with caplog.at_level(logging.INFO, logger="pcga-fallback-test"):
        s_hat, simul_obs, post_diagv, iter_best = solver.run()
    assert any("All LM candidates violate" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("is_direct_solve", (True, False))
def test_lm_multiprocessing(is_direct_solve) -> None:
    # max_workers = max(max_workers_lm, max_it_lm) when is_lm is True; with
    # both > 1 and n_internal_loops > 1, linear_iteration dispatches through
    # ProcessPoolExecutor instead of the single-worker loop.
    solver = pypcga.PCGA(
        **_make_kwargs(
            is_lm=True,
            is_direct_solve=is_direct_solve,
            max_it_lm=2,
            max_workers_lm=2,
            forward_model=linear_forward_model,
        )
    )
    s_hat, simul_obs, post_diagv, iter_best = solver.run()
    assert s_hat.shape == (S_DIM, 1)


def test_heteroscedastic_cov_obs_logs_approximation(caplog) -> None:
    import logging

    logger = logging.getLogger("pcga-heteroscedastic-test")
    logger.setLevel(logging.INFO)
    n_obs = 6
    cov_obs = covmats.CovViaDiagonal(np.linspace(0.01, 0.5, n_obs))
    kwargs = _make_kwargs(n_obs=n_obs, cov_obs=cov_obs, logger=logger)
    solver = pypcga.PCGA(**kwargs)
    with caplog.at_level(logging.INFO, logger="pcga-heteroscedastic-test"):
        solver.run()
    assert any("not homoscedastic" in rec.message for rec in caplog.records)


def test_is_s_violate_lm_bounds_direct() -> None:
    solver = pypcga.PCGA(
        **_make_kwargs(is_lm=True, max_it_lm=3, lm_smin=-1.0, lm_smax=1.0)
    )
    S = np.array([[-2.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    violates = solver.is_s_violate_lm_bounds(S)
    # column 0 violates smin, column 2 violates smax, column 1 is fine
    np.testing.assert_array_equal(violates, [True, False, True])


@pytest.mark.parametrize(
    "n_obs,n_pc",
    (
        (6, 6),  # n_pc == n_obs
        (4, 6),  # n_pc > n_obs (n_pc still <= s_dim=8)
    ),
)
def test_get_invA_as_linop_rank_edge_cases(n_obs, n_pc) -> None:
    solver = pypcga.PCGA(**_make_kwargs(n_obs=n_obs, n_pc=n_pc, is_direct_solve=False))
    rng = np.random.default_rng(20)
    HZ = rng.normal(size=(n_obs, n_pc))
    HX = solver.drift.mat
    op = solver.get_invA_as_linop(HZ, HX, solver.cov_obs)
    assert op.shape == (n_obs + solver.drift.beta_dim, n_obs + solver.drift.beta_dim)


def test_linear_iteration_raises_on_bad_forward_model_shape() -> None:
    # Behave correctly for the first two forward_model calls (the initial
    # guess evaluation, then the Jacobian-perturbation ensemble inside
    # jac_mat), then return a mismatched number of columns -- this reliably
    # targets linear_iteration's later "evaluate the candidate solution(s)"
    # call regardless of exactly how many columns each call happens to use.
    n_obs = 6
    call_count = {"n": 0}

    def flaky_forward_model(s_ensemble):
        call_count["n"] += 1
        out = linear_forward_model(s_ensemble)
        if call_count["n"] > 2:
            return np.hstack([out, out])
        return out

    kwargs = _make_kwargs(n_obs=n_obs, is_lm=False, forward_model=flaky_forward_model)
    solver = pypcga.PCGA(**kwargs)
    with pytest.raises(ValueError, match="forward_model returned shape"):
        solver.run()
