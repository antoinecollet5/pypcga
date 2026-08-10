import logging

import numpy as np
import pytest
from pypcga._utils import ensemble_dot, ghep, mgs_stable


@pytest.mark.parametrize("with_logger", (False, True))
def test_mgs_stable(with_logger, caplog) -> None:
    rng = np.random.default_rng(0)
    A = np.diag(np.arange(1, 21).astype(float))
    Z = rng.normal(size=(20, 5))

    logger = logging.getLogger("pcga-mgs-stable-test")
    logger.setLevel(logging.INFO)

    with caplog.at_level(logging.INFO, logger="pcga-mgs-stable-test"):
        q, Aq, r = mgs_stable(A, Z, logger=logger if with_logger else None)

    assert q.shape == (20, 5)
    assert Aq.shape == (20, 5)
    assert r.shape == (5, 5)
    # Q^T A Q = I
    np.testing.assert_allclose(q.T @ Aq, np.eye(5), atol=1e-8)
    if with_logger:
        assert any(
            "Re-orthogonalization counts per column" in rec.message
            for rec in caplog.records
        )
        # The actual reorth counts must appear in the formatted message (not
        # just the literal prefix), i.e. eager f-string formatting rather
        # than a broken %-style logger.info(msg, value) call.
        assert any(
            "Re-orthogonalization counts per column: [" in rec.message
            for rec in caplog.records
        )
    else:
        assert len(caplog.records) == 0


def test_mgs_stable_singular_r(caplog) -> None:
    # Duplicate columns make Z rank-deficient, so after orthogonalization the
    # second copy collapses and r becomes singular -> exercises the
    # LinAlgError branch of the logged diagnostics.
    logger = logging.getLogger("pcga-mgs-stable-singular-test")
    logger.setLevel(logging.INFO)

    A = np.eye(6)
    Z = np.zeros((6, 2))
    Z[:, 0] = 1.0
    Z[:, 1] = 1.0
    with caplog.at_level(logging.INFO, logger="pcga-mgs-stable-singular-test"):
        mgs_stable(A, Z, logger=logger)
    assert any("YR^{-1}-Q is singular" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("single_pass", (True, False))
@pytest.mark.parametrize("keep_neg_eigvals", (True, False))
def test_ghep(single_pass, keep_neg_eigvals) -> None:
    rng = np.random.default_rng(1)
    n = 15
    B_sqrt = rng.normal(size=(n, n))
    B = B_sqrt @ B_sqrt.T + n * np.eye(n)  # SPD
    Binv = np.linalg.inv(B)
    M = rng.normal(size=(n, n))
    A = M @ M.T  # SPD -> only positive eigenvalues

    vects, vals = ghep(
        A,
        B,
        Binv,
        r=4,
        d=2,
        single_pass=single_pass,
        keep_neg_eigvals=keep_neg_eigvals,
        random_state=0,
    )
    assert vals.ndim == 2
    assert vals.shape[1] == 1
    assert vects.shape[0] == n
    assert vects.shape[1] == vals.shape[0]
    # Reproducible given the same random_state
    vects2, vals2 = ghep(
        A,
        B,
        Binv,
        r=4,
        d=2,
        single_pass=single_pass,
        keep_neg_eigvals=keep_neg_eigvals,
        random_state=0,
    )
    np.testing.assert_allclose(vals, vals2)


def test_ensemble_dot() -> None:
    rng = np.random.default_rng(2)
    X1 = rng.normal(size=(10, 4))
    X2 = rng.normal(size=(10, 4))
    result = ensemble_dot(X1, X2)
    assert result.shape == (4,)
    expected = np.sum(X1 * X2, axis=0)
    np.testing.assert_allclose(result, expected)
