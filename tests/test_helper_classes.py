import covmats
import numpy as np
from pypcga._pcga import InvALinOp, Residual


def test_residual() -> None:
    residual = Residual()
    assert residual.itercount() == 0
    residual(np.array([1.0]))
    residual(np.array([0.5]))
    assert residual.itercount() == 2
    residual.clear()
    assert residual.itercount() == 0


def test_inv_a_linop() -> None:
    rng = np.random.default_rng(3)
    n_obs = 6
    n_pc = 4
    beta_dim = 1

    HX = rng.normal(size=(n_obs, beta_dim))
    HZZTHT_eig_vects = rng.normal(size=(n_obs, n_pc))
    # orthonormalize columns for a well-posed test
    HZZTHT_eig_vects, _ = np.linalg.qr(HZZTHT_eig_vects)
    HZZTHT_eig_vals = np.abs(rng.normal(size=(n_pc, 1))) + 0.1
    cov_obs = covmats.CovViaDiagonal(np.full(n_obs, 0.1))

    op = InvALinOp(
        HX=HX,
        HZZTHT_eig_vects=HZZTHT_eig_vects,
        HZZTHT_eig_vals=HZZTHT_eig_vals,
        cov_obs=cov_obs,
        n_obs=n_obs,
        beta_dim=beta_dim,
    )
    assert op.inflation == 1.0
    op.update_inflation_factor(2.5)
    assert op.inflation == 2.5

    invPsi = op.get_invPsi(op.inflation)
    assert invPsi.shape == (n_obs, n_obs)

    x = rng.normal(size=(n_obs + beta_dim, 1))
    y_matvec = op.matvec(x)
    y_rmatvec = op.rmatvec(x)
    # InvALinOp is meant to represent a symmetric operator (A^{-1} is
    # symmetric since A is), so matvec and rmatvec should agree.
    np.testing.assert_allclose(y_matvec, y_rmatvec)

    X = np.hstack([x, x * 2.0])
    y_matmat = op.matmat(X)
    y_rmatmat = op.rmatmat(X)
    np.testing.assert_allclose(y_matmat, y_rmatmat)
    np.testing.assert_allclose(y_matmat[:, 0:1], y_matvec)
