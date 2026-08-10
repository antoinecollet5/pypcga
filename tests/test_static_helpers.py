import covmats
import numpy as np
import pypcga
from pypcga._pcga import PCGA

from .conftest import N_PC, S_DIM, quadratic_forward_model


def _make_solver(n_obs=6, **overrides):
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
    return pypcga.PCGA(**kwargs)  # ty:ignore[invalid-argument-type]


def test_build_dense_A() -> None:
    rng = np.random.default_rng(10)
    n, p = 5, 2
    Psi = rng.normal(size=(n, n))
    Psi = Psi @ Psi.T + n * np.eye(n)
    HX = rng.normal(size=(n, p))
    A = PCGA.build_dense_A(Psi, HX)
    assert A.shape == (n + p, n + p)
    np.testing.assert_allclose(A[:n, :n], Psi)
    np.testing.assert_allclose(A[:n, n:], HX)
    np.testing.assert_allclose(A[n:, :n], HX.T)
    np.testing.assert_allclose(A[n:, n:], 0.0)


def test_build_dense_A_from_cholesky_matches_solve() -> None:
    rng = np.random.default_rng(11)
    n, p = 5, 2
    Psi = rng.normal(size=(n, n))
    Psi = Psi @ Psi.T + n * np.eye(n)
    HX = rng.normal(size=(n, p))

    A = PCGA.build_dense_A(Psi, HX)
    LA = PCGA.build_cholesky(Psi, HX)
    A_reconstructed = PCGA.build_dense_A_from_cholesky(LA, p)
    np.testing.assert_allclose(A, A_reconstructed, atol=1e-8)

    # solve_cholesky should agree with a direct dense solve of A x = b
    b = rng.normal(size=(n + p, 1))
    x_direct = np.linalg.solve(A, b)
    x_chol = PCGA.solve_cholesky(LA, b.copy(), p)
    np.testing.assert_allclose(x_direct, x_chol, atol=1e-6)


def test_get_A_as_linop_matvec_rmatvec_symmetric() -> None:
    solver = _make_solver()
    rng = np.random.default_rng(12)
    n_obs = solver.n_obs
    beta_dim = solver.drift.beta_dim
    HX = rng.normal(size=(n_obs, beta_dim))
    HZ = rng.normal(size=(n_obs, N_PC))

    op = solver.get_A_as_linop(HX, HZ, inflation=1.0)
    # Use a 1D vector: this matches how gmres/minres actually invoke this
    # operator internally (internal_iteration_krylov_subspace). A 2D column
    # vector would trip an unrelated covmats.CovViaDiagonal broadcasting
    # quirk on 2D input, which isn't what this test is about.
    x = rng.normal(size=(n_obs + beta_dim,))
    y_matvec = op.matvec(x)
    y_rmatvec = op.rmatvec(x)
    # A is symmetric, so matvec and rmatvec should agree.
    np.testing.assert_allclose(y_matvec, y_rmatvec)


def test_line_search() -> None:
    solver = _make_solver(max_it_lm=5)
    rng = np.random.default_rng(13)
    s_cur = rng.normal(size=(S_DIM, 1))
    s_past = rng.normal(size=(S_DIM, 1))
    s_hat, simul_obs_new, best_obj = solver.line_search(s_cur, s_past)
    assert s_hat.shape == (S_DIM, 1)
    assert simul_obs_new.shape == (solver.n_obs, 1)
    assert best_obj < 1.0e20
