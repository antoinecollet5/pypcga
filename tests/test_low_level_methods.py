import covmats
import numpy as np
import pypcga
import pytest

from .conftest import N_PC, S_DIM, linear_forward_model, quadratic_forward_model


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


@pytest.mark.parametrize("delta", (None, 1e-3))
def test_perturb_ensemble_adaptive_and_fixed_delta(delta) -> None:
    solver = _make_solver()
    rng = np.random.default_rng(4)
    s = rng.normal(size=(S_DIM, 1))
    x = rng.normal(size=(S_DIM, 3))
    x_out, deltas = solver._perturb_ensemble(x.copy(), s, eps=1e-8, delta=delta)
    assert x_out.shape == (S_DIM, 3)
    assert deltas.shape == (3, 1)
    if delta is not None:
        np.testing.assert_allclose(deltas, delta)


def test_perturb_ensemble_zero_direction_fallback() -> None:
    solver = _make_solver()
    s = np.zeros((S_DIM, 1))  # s = 0 forces the heuristic to yield exactly 0
    x = np.zeros((S_DIM, 2))
    x[:, 0] = 1.0  # non-zero direction, but s = 0 -> mag = absmag = 0
    x_out, deltas = solver._perturb_ensemble(x.copy(), s, eps=1e-8, delta=None)
    # Fallback assigns sqrt(eps) rather than leaving delta at 0.
    assert np.all(deltas > 0)


def test_jac_vect_raises_on_bad_forward_model_shape() -> None:
    def broken_forward_model(s_ensemble):
        # Always returns a single column regardless of the ensemble size,
        # which will not match `nruns` as soon as more than 1 column is
        # requested (as jac_vect does internally).
        return linear_forward_model(s_ensemble)[:, :1]

    solver = _make_solver(forward_model=broken_forward_model)
    rng = np.random.default_rng(5)
    s = rng.normal(size=(S_DIM, 1))
    x = rng.normal(size=(S_DIM, 3))
    with pytest.raises(ValueError, match="is not nruns"):
        solver.jac_vect(x, s, np.zeros((6, 1)), eps=1e-8)


def test_objective_function_no_beta_new() -> None:
    solver = _make_solver()
    rng = np.random.default_rng(6)
    s_cur = rng.normal(size=(S_DIM, 1))
    simul_obs = quadratic_forward_model(s_cur)
    obj = solver.objective_function_no_beta_new(s_cur, simul_obs)
    # Should be close to (and never lower than) the closed-form marginalized
    # objective computed by the cheaper `objective_function_no_beta`.
    obj_ref = solver.objective_function_no_beta(s_cur, simul_obs)
    assert obj == pytest.approx(obj_ref, rel=1e-4)


def test_jac_mat_point_prior_not_implemented() -> None:
    # A drift matrix with zero columns (beta_dim = 0) hits the "point prior"
    # branch of jac_mat, which is not implemented.
    point_drift = covmats.DriftMatrix(mat=np.empty((S_DIM, 0)))
    solver = _make_solver(drift=point_drift)
    s_cur = np.zeros((S_DIM, 1))
    simul_obs = quadratic_forward_model(s_cur)
    Z = np.sqrt(solver.Q.eig_vals).T * solver.Q.eig_vects
    with pytest.raises(NotImplementedError):
        solver.jac_mat(s_cur, simul_obs, Z)


def test_jac_mat_multi_column_drift() -> None:
    # LinearDriftMatrix in 1D gives beta_dim = 2 > 1, exercising jac_mat's
    # SVD-based U_data branch (as opposed to the p == 1 norm-based one).
    pts = np.arange(S_DIM).reshape(-1, 1).astype(float)
    drift = covmats.LinearDriftMatrix(pts)
    solver = _make_solver(drift=drift)
    s_cur = np.zeros((S_DIM, 1))
    simul_obs = quadratic_forward_model(s_cur)
    Z = np.sqrt(solver.Q.eig_vals).T * solver.Q.eig_vects
    HX, HZ, Hs, U_data = solver.jac_mat(s_cur, simul_obs, Z)
    assert HX.shape[1] == 2
    assert U_data.shape == (6, 2)
