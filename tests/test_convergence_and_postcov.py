import covmats
import numpy as np
import pypcga
import pytest

from .conftest import N_PC, S_DIM, quadratic_forward_model


def _make_kwargs(n_obs=6, **overrides):
    prior_cov = covmats.CovViaDiagonal(np.ones(S_DIM))
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=N_PC, random_state=0)
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


def test_callback_is_invoked() -> None:
    calls = []

    def callback(solver, s_hat, simul_obs, n_iter):
        calls.append(n_iter)

    solver = pypcga.PCGA(**_make_kwargs(callback=callback, maxiter=2))
    solver.run()
    # Called once for the initial state (n_iter=0) and once per GN iteration.
    assert 0 in calls
    assert len(calls) >= 2


def test_ftarget_triggers_early_stop() -> None:
    solver = pypcga.PCGA(**_make_kwargs(ftarget=1.0e10, maxiter=5))
    solver.run()
    assert solver.istate.status == "CONVERGENCE: F_<=_FACTR*EPSMCH"
    assert solver.istate.is_success is True


def test_maxiter_reached_without_other_convergence() -> None:
    solver = pypcga.PCGA(
        **_make_kwargs(
            restol=0.0,
            ftol=0.0,
            ftarget=None,
            is_line_search=False,
            maxiter=2,
        )
    )
    solver.run()
    assert solver.istate.status == "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"


def test_gauss_newton_line_search_never_improves(monkeypatch, caplog) -> None:
    import logging

    logger = logging.getLogger("pcga-no-progress-test")
    logger.setLevel(logging.INFO)
    solver = pypcga.PCGA(
        **_make_kwargs(
            is_line_search=True,
            maxiter=5,
            logger=logger,
            restol=-1.0,
            ftol=-1.0,
        )
    )

    s_fixed = solver.s_init.copy()
    simul_obs_fixed = solver.forward_model(s_fixed)
    beta_fixed = np.zeros((solver.drift.beta_dim, 1))
    inflation_fixed = 1.0
    # Always worse than any real, recomputed objective -- guarantees "no
    # progress" is detected on every single iteration, regardless of the
    # actual (moderate) objective value recomputed downstream from the fixed
    # state.
    huge_obj = 1.0e21

    def fake_step(self, s_past, simul_obs_cur, n_iter):
        self.istate.obj_seq.append(huge_obj)
        self.istate.inflation_seq.append(inflation_fixed)
        self.istate.Q2_seq.append(1.0)
        self.istate.cR_seq.append(1.0)
        return s_fixed, beta_fixed, simul_obs_fixed, inflation_fixed, huge_obj

    def fake_line_search(self, s_cur, s_past):
        return s_fixed, simul_obs_fixed, huge_obj

    # PCGA uses __slots__, so instance-level monkeypatching of methods isn't
    # possible -- patch the class instead (auto-restored after the test).
    monkeypatch.setattr(pypcga.PCGA, "_gauss_newton_step", fake_step)
    monkeypatch.setattr(pypcga.PCGA, "line_search", fake_line_search)

    with caplog.at_level(logging.INFO, logger="pcga-no-progress-test"):
        solver.run()

    # case 1 (progress) never triggers -> the initial guess is never beaten.
    assert solver.istate.iter_best == 0
    assert any("wait for one more iteration" in rec.message for rec in caplog.records)
    # The hard-break message is "no progress in obj value" *without* the
    # "...but wait for one more iteration..." suffix -- check for a record
    # that is exactly that (grace-pass messages share the same prefix).
    assert any(rec.message == "no progress in obj value" for rec in caplog.records)
    assert any("Did not found better solution" in rec.message for rec in caplog.records)
    assert any(
        "skip posterior variance computation" in rec.message for rec in caplog.records
    )


def test_gauss_newton_line_search_succeeds(monkeypatch) -> None:
    # Force case 2 (no progress) on the raw Gauss-Newton step, but let
    # line_search "succeed" (return an improved objective) -> exercises the
    # `if obj < obj_old:` success branch inside the line-search sub-case.
    solver = pypcga.PCGA(**_make_kwargs(is_line_search=True, maxiter=3))

    s_fixed = solver.s_init.copy()
    simul_obs_fixed = solver.forward_model(s_fixed)
    beta_fixed = np.zeros((solver.drift.beta_dim, 1))
    huge_obj = 1.0e21

    def fake_step(self, s_past, simul_obs_cur, n_iter):
        self.istate.obj_seq.append(huge_obj)
        self.istate.inflation_seq.append(1.0)
        self.istate.Q2_seq.append(1.0)
        self.istate.cR_seq.append(1.0)
        return s_fixed, beta_fixed, simul_obs_fixed, 1.0, huge_obj

    def fake_line_search(self, s_cur, s_past):
        # A tiny, but real and finite, objective -> always beats obj_old.
        return s_fixed, simul_obs_fixed, -1.0

    monkeypatch.setattr(pypcga.PCGA, "_gauss_newton_step", fake_step)
    monkeypatch.setattr(pypcga.PCGA, "line_search", fake_line_search)

    solver.run()

    # Line search "succeeded" on iteration 0 -> istate.s_best gets updated.
    assert solver.istate.iter_best == 1


def test_gauss_newton_no_line_search_breaks_immediately(monkeypatch) -> None:
    solver = pypcga.PCGA(**_make_kwargs(is_line_search=False, maxiter=5))

    s_fixed = solver.s_init.copy()
    simul_obs_fixed = solver.forward_model(s_fixed)
    beta_fixed = np.zeros((solver.drift.beta_dim, 1))
    huge_obj = 1.0e21

    def fake_step(self, s_past, simul_obs_cur, n_iter):
        self.istate.obj_seq.append(huge_obj)
        self.istate.inflation_seq.append(1.0)
        self.istate.Q2_seq.append(1.0)
        self.istate.cR_seq.append(1.0)
        return s_fixed, beta_fixed, simul_obs_fixed, 1.0, huge_obj

    monkeypatch.setattr(pypcga.PCGA, "_gauss_newton_step", fake_step)
    solver.run()

    assert solver.istate.status == "CONVERGENCE: NO PROGRESS IN OBJ VALUE"
    assert solver.istate.is_success is False
    assert solver.istate.iter_best == 0


@pytest.mark.parametrize("is_direct_solve", (True, False))
def test_get_dense_post_cov_explicit_args(is_direct_solve) -> None:
    solver = pypcga.PCGA(**_make_kwargs())
    solver.run()
    cov = solver.get_dense_post_cov(is_direct_solve=is_direct_solve, inflation=2.0)
    assert cov.shape == (S_DIM, S_DIM)


def test_get_eigen_post_cov_explicit_inflation_default_n_pc() -> None:
    solver = pypcga.PCGA(**_make_kwargs())
    solver.run()
    # n_pc left at its default (None) -> exercises the `_n_pc = self.Q.n_pc`
    # branch; inflation given explicitly -> exercises `_inflation = inflation`.
    cov = solver.get_eigen_post_cov(inflation=1.5)
    assert cov.shape == (S_DIM, S_DIM)
