"""Regression tests for bugs fixed in 0.3.1."""

from typing import Any

import covmats
import numpy as np
import pypcga
import pytest
from pypcga._utils import check_random_state

S_DIM, N_OBS, N_PC = 30, 12, 10
_rng = np.random.default_rng(0)
A_LIN = _rng.normal(size=(N_OBS, S_DIM))


def linear_fm(s: np.ndarray) -> np.ndarray:
    return A_LIN @ s


# Shared prior: the truncated eigen factorization of an identity prior is not
# unique, so every solver of a given test must use the very same Q.
Q_PRIOR = covmats.eigen_factorize_cov_mat(
    covmats.CovViaDiagonal(np.linspace(2.0, 0.5, S_DIM)), n_pc=N_PC, random_state=0
)


def _solver(**kwargs) -> pypcga.PCGA:
    Q = Q_PRIOR
    defaults: dict[str, Any] = dict(
        s_init=np.full(S_DIM, 0.1),
        obs=linear_fm(np.random.default_rng(1).normal(size=(S_DIM, 1))).ravel(),
        cov_obs=covmats.CovViaDiagonal(np.full(N_OBS, 0.05)),
        forward_model=linear_fm,
        Q=Q,
        random_state=1,
        maxiter=3,
    )
    defaults.update(kwargs)
    return pypcga.PCGA(**defaults)


def _linearization(solver):
    Z = np.sqrt(solver.Q.eig_vals).T * solver.Q.eig_vects
    s0 = np.full((S_DIM, 1), 0.3)
    y0 = linear_fm(s0)
    HX, HZ, Hs, _ = solver.jac_mat(s0, y0, Z)
    b = np.zeros((N_OBS + 1, 1))
    b[:N_OBS] = solver.obs.reshape(-1, 1) - y0 + Hs
    return Z, HX, HZ, b


@pytest.mark.parametrize("inflation", (1.0, 10.0, 300.0))
def test_direct_and_krylov_agree_with_inflation(inflation) -> None:
    """The LM inflation must apply to R in both solvers (HQH^T + a R)."""
    s_d, s_k = _solver(is_direct_solve=True), _solver(is_direct_solve=False)
    Z, HX, HZ, b = _linearization(s_d)
    inv = s_k.get_invA_as_linop(HZ, HX, s_k.cov_obs)
    xd = s_d.internal_iteration_direct(HZ, HX, Z, b, inflation)
    xk = s_k.internal_iteration_krylov_subspace(HZ, HX, Z, b, inv, inflation)
    for a, b_ in zip(xd, xk):
        np.testing.assert_allclose(a, b_, atol=1e-8)

    # get_psi must be HQH^T + inflation * R
    Psi = s_d.get_psi(HZ, s_d.cov_obs, inflation)
    np.testing.assert_allclose(Psi, HZ @ HZ.T + inflation * s_d.cov_obs.todense())

    # and the posterior variance too
    pv_d = s_d._compute_post_cov_diag(HZ, HX, s_d.cov_obs, inflation, True)
    pv_k = s_k._compute_post_cov_diag(HZ, HX, s_k.cov_obs, inflation, False)
    np.testing.assert_allclose(pv_d, pv_k, atol=1e-8)


def test_lowrank_posterior_matches_brute_force() -> None:
    solver = _solver(is_direct_solve=True)
    Z, HX, HZ, _ = _linearization(solver)
    Psi = solver.get_psi(HZ, solver.cov_obs, 2.0)
    A = solver.build_dense_A(Psi, HX)
    b_all = np.vstack([HZ @ Z.T, solver.drift.mat.T])
    expected = solver.prior_s_var - np.sum(b_all * np.linalg.solve(A, b_all), axis=0)
    got = solver._compute_post_cov_diag(HZ, HX, solver.cov_obs, 2.0, True)
    np.testing.assert_allclose(got.ravel(), expected, atol=1e-10)


@pytest.mark.parametrize("is_direct_solve", (True, False))
def test_dense_post_cov_consistent_with_post_diagv(is_direct_solve) -> None:
    solver = _solver(is_direct_solve=is_direct_solve, maxiter=4)
    _, _, post_diagv, _ = solver.run()
    np.testing.assert_allclose(
        np.diag(solver.get_dense_post_cov()), post_diagv.ravel(), atol=1e-8
    )


def test_is_save_jac_false() -> None:
    solver = _solver(is_save_jac=False, is_direct_solve=True)
    s_hat, *_ = solver.run()
    assert s_hat.shape == (S_DIM, 1)
    assert solver.istate.HZ.size == 0
    with pytest.raises(RuntimeError, match="Jacobian products"):
        solver.get_dense_post_cov()


def test_line_search_uses_max_it_ls() -> None:
    n_cols = []

    def fm(s):
        n_cols.append(s.shape[1])
        return linear_fm(s)

    solver = _solver(is_line_search=True, max_it_lm=3, max_it_ls=7, forward_model=fm)
    solver.line_search(np.full((S_DIM, 1), 0.3), np.zeros((S_DIM, 1)))
    assert n_cols[-1] == 7


def test_line_search_ignores_nan_objectives() -> None:
    def fm(s):
        out = linear_fm(s)
        out[:, 0] = np.nan
        return out

    solver = _solver(max_it_ls=5, forward_model=fm)
    s_hat, simul, obj = solver.line_search(
        np.full((S_DIM, 1), 0.3), np.zeros((S_DIM, 1))
    )
    assert np.isfinite(obj)
    assert np.all(np.isfinite(simul))


def test_max_workers_respects_user_limit() -> None:
    assert _solver(is_lm=True, max_it_lm=8, max_workers_lm=2).max_workers == 2
    assert _solver(is_lm=True, max_it_lm=2, max_workers_lm=8).max_workers == 2
    assert _solver(is_lm=False, max_it_lm=8, max_workers_lm=8).max_workers == 1


def test_lm_threads_with_unpicklable_forward_model() -> None:
    # lambdas cannot be pickled: this used to fail with ProcessPoolExecutor
    solver = _solver(
        is_lm=True, max_it_lm=3, max_workers_lm=3, forward_model=lambda s: A_LIN @ s
    )
    s_hat, *_ = solver.run()
    assert s_hat.shape == (S_DIM, 1)


def test_run_twice_resets_state() -> None:
    solver = _solver(is_direct_solve=True)
    solver.run()
    n = len(solver.istate.obj_seq)
    solver.run()
    assert len(solver.istate.obj_seq) == n


def test_rmse_normalized_is_per_column() -> None:
    solver = _solver()
    res = np.random.default_rng(3).normal(size=(N_OBS, 4))
    out = solver.rmse(res, True)
    assert out.shape == (4,)
    np.testing.assert_allclose(out[1], solver.rmse(res[:, 1:2], True).item())


def test_zero_s_init_no_warning() -> None:
    solver = _solver(s_init=np.zeros(S_DIM), maxiter=2)
    with np.errstate(all="raise"):
        solver.run()


def test_best_values_are_nan_without_improvement() -> None:
    istate = pypcga.InternalState(s_best=np.zeros(3))
    istate.Q2_seq.append(1.0)
    istate.cR_seq.append(1.0)
    istate.inflation_seq.append(1.0)
    assert np.isnan(istate.Q2_best)
    assert np.isnan(istate.best_cR)
    assert np.isnan(istate.best_inflation)
    assert istate.status == "IDLE."


def test_inv_a_linop_with_inflation_does_not_mutate() -> None:
    solver = _solver()
    _, HX, HZ, _ = _linearization(solver)
    op = solver.get_invA_as_linop(HZ, HX, solver.cov_obs)
    op2 = op.with_inflation(5.0)
    assert op.inflation == 1.0 and op2.inflation == 5.0
    assert op.with_inflation(1.0) is op
    x = np.random.default_rng(0).normal(size=op.shape[0])
    # op2 is the inverse of A(5.0)
    A5 = solver.get_A_as_linop(HX, HZ, 5.0)
    np.testing.assert_allclose(A5 @ (op2 @ x), x, atol=1e-8)


def test_check_random_state() -> None:
    assert check_random_state(None) is np.random.mtrand._rand
    assert isinstance(check_random_state(3), np.random.RandomState)
    g = np.random.default_rng(0)
    assert check_random_state(g) is g
    with pytest.raises(ValueError):
        check_random_state("foo")  # ty: ignore[invalid-argument-type]


def test_ghep_dense_symmetric() -> None:
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.normal(size=(40, 5)))[0]
    A = U @ np.diag([5.0, 4.0, 3.0, 2.0, 1.0]) @ U.T
    for single_pass in (True, False):
        V, s = pypcga.ghep(
            A,
            np.eye(40),
            np.eye(40),
            r=5,
            d=5,
            single_pass=single_pass,
            random_state=0,
        )
        np.testing.assert_allclose(s.ravel()[:5], [5, 4, 3, 2, 1], atol=1e-8)


def test_gmres_non_convergence_warns() -> None:
    solver = _solver()
    _, HX, HZ, b = _linearization(solver)
    A = solver.get_A_as_linop(HX, HZ, 1.0)
    with pytest.warns(RuntimeWarning, match="GMRES did not converge"):
        solver._solve_iterative_subspace_krylov(A, b, None, restart=1, maxiter=1)


def test_line_search_bad_forward_model_shape() -> None:
    solver = _solver(max_it_ls=4, forward_model=lambda s: linear_fm(s)[:, :1])
    with pytest.raises(ValueError, match="max_it_ls"):
        solver.line_search(np.full((S_DIM, 1), 0.3), np.zeros((S_DIM, 1)))


def test_inv_psi_other_inflation_and_eigen_post_cov_matmat() -> None:
    solver = _solver(is_direct_solve=True)
    _, HX, HZ, _ = _linearization(solver)
    op = solver.get_invA_as_linop(HZ, HX, solver.cov_obs)
    np.testing.assert_allclose(
        op.inv_psi(np.eye(N_OBS), 3.0), op.with_inflation(3.0).get_invPsi(3.0)
    )
    solver.run()
    post = solver.get_eigen_post_cov(n_pc=5, random_state=0)
    assert post.shape == (S_DIM, S_DIM)
