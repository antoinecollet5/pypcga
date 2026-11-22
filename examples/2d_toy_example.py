import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - 2D toy example")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple 2D inversion test

    This toy example illustrates how to use `pypcga` and exposes its main features.
    The forward model is a simple static smoothing of the squared field (nonlinear but
    without time dependence). It is used both to produce the synthetic observations from
    a reference field and to perform the inversion.

    Run it interactively with `marimo edit 2d_toy_example.py` or as a plain script with
    `python 2d_toy_example.py`.
    """)
    return


@app.cell
def _():
    import logging

    import covmats
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pypcga
    import scipy as sp

    return covmats, logging, mo, np, plt, pypcga, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Solver messages go through a standard `logging.Logger`. Here they are printed in
    the output of the cell that runs the inversion.
    """)
    return


@app.cell
def _(logging):
    class PrintHandler(logging.Handler):
        """Print log records (marimo shows stdout below the running cell)."""

        def emit(self, record: logging.LogRecord) -> None:
            print(self.format(record))

    pcga_logger = logging.getLogger("PCGA")
    pcga_logger.setLevel(logging.INFO)
    pcga_logger.handlers.clear()  # avoid duplicated handlers when re-running
    pcga_logger.addHandler(PrintHandler())
    pcga_logger.propagate = False
    return (pcga_logger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior

    We use an example provided by `covmats`. The prior covariance matrix
    $\mathbf{C}_{\mathrm{prior}}$ is represented as a sparse factorization of its inverse,
    $\mathbf{LDL}^{\mathrm{T}} = \mathbf{P}\mathbf{C}_{\mathrm{prior}}^{-1}\mathbf{P}^{\mathrm{T}}$,
    wrapped in a `covmats.CovViaSparsePrecisionCholesky` instance. It has shape
    (4225, 4225), i.e., a (65, 65) square grid. The vectors are ordered with $x$ varying
    fastest, so `v.reshape(ny, nx)` gives an image with rows along $y$.
    """)
    return


@app.cell
def _(covmats, np):
    cov_prior = covmats.CovViaSparsePrecisionCholesky(
        covmats.load_precision_example_4225x_SCF()
    )
    nx = ny = int(np.sqrt(cov_prior.shape[0]))
    n_cells = nx * ny

    def to_img(v):
        """Reshape a parameter vector (x varying fastest) into a (ny, nx) image."""
        return np.asarray(v).reshape(ny, nx)

    return cov_prior, n_cells, nx, ny, to_img


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two non-conditional simulations from the prior give the reference field (shifted by
    a mean of 50) and the initial guess. Change the seeds to obtain other fields.
    """)
    return


@app.cell
def _(cov_prior, np):
    mean = 50.0
    s_ref = np.abs(
        cov_prior.sample_mvnormal(shape=(1,), random_state=2026).ravel() + mean
    )
    s_init = np.abs(cov_prior.sample_mvnormal(shape=(1,), random_state=15653).ravel())
    return s_init, s_ref


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward model

    The data transform is a Gaussian smoothing of $4 s^2$, followed by a sampling at a
    set of fixed cells (the observation locations). `forward_model` works on an ensemble
    of vectors with shape $(N_s, N_e)$ and returns $(N_{\mathrm{obs}}, N_e)$: in PCGA,
    most forward calls of an iteration are independent and can be run in parallel. It is
    the user's responsibility to decide how (a loop, a vectorized call as here, mpi,
    multiprocessing, joblib, ...).
    """)
    return


@app.cell
def _(n_cells, np, nx, ny, sp):
    # About 50 observation locations (1.2 % of the cells), drawn without replacement
    obs_idx = np.sort(np.random.default_rng(0).choice(n_cells, size=50, replace=False))

    def forward_model(s_ensemble: np.ndarray) -> np.ndarray:
        """Smooth 4 s^2 and sample it at the observation locations."""
        S = np.asarray(s_ensemble).reshape(n_cells, -1)  # (N_s, N_e)
        imgs = S.T.reshape(-1, ny, nx)  # (N_e, ny, nx)
        smoothed = sp.ndimage.gaussian_filter(4.0 * imgs**2, sigma=2.0, axes=(1, 2))
        return smoothed.reshape(S.shape[1], -1)[:, obs_idx].T  # (N_obs, N_e)

    return forward_model, obs_idx


@app.cell
def _(forward_model, np, s_init, s_ref):
    obs = forward_model(s_ref)[:, 0]

    # Sanity check: the ensemble version is consistent with single evaluations
    _d_pred = forward_model(np.column_stack([s_ref, s_init]))
    assert _d_pred.shape == (obs.size, 2)
    np.testing.assert_allclose(_d_pred[:, 0], obs)
    return (obs,)


@app.cell
def _(nx, obs_idx, plt, s_init, s_ref, to_img):
    _fig, _axes = plt.subplots(
        1, 2, figsize=(9, 4.3), sharey=True, layout="constrained"
    )
    for _ax, (_title, _s) in zip(
        _axes, {"Reference": s_ref, "Initial guess": s_init}.items()
    ):
        _im = _ax.imshow(to_img(_s), origin="lower", cmap="jet", vmin=0.0, vmax=120.0)
        _ax.set_title(_title)
        _ax.set_xlabel("X")
    _axes[0].plot(obs_idx % nx, obs_idx // nx, "k+", label="observations")
    _axes[0].legend(loc="upper right", fontsize=8)
    _axes[0].set_ylabel("Y")
    _fig.colorbar(_im, ax=_axes, shrink=0.8, label="s")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observation errors

    To illustrate a more complex case, the covariance matrix of observation errors
    ($\mathbf{R}$, `cov_obs`) is not diagonal: it is defined through a lower triangular
    (Cholesky) factor with a few random off-diagonal terms. For the simplest case
    (independent errors), use `covmats.CovViaDiagonal` instead.
    """)
    return


@app.cell
def _(covmats, np, obs, plt):
    _n = obs.size
    _amplitude = np.ptp(obs)
    _rng = np.random.default_rng(42)

    # CASE 1 - diagonal covariance matrix (simplest case): 10 % of the amplitude
    # cov_obs = covmats.CovViaDiagonal(np.full(_n, (0.1 * _amplitude) ** 2))

    # CASE 2 - non diagonal covariance matrix through its Cholesky factor: ~10 % of
    # the lower off-diagonal terms are non zero.
    _L = np.zeros((_n, _n))
    _i, _j = np.tril_indices(_n, k=-1)
    _mask = _rng.random(_i.size) < 0.1
    _L[_i[_mask], _j[_mask]] = _rng.uniform(-0.05, 0.05, _mask.sum()) * _amplitude
    _L[np.diag_indices(_n)] = 0.1 * _amplitude
    cov_obs = covmats.CovViaCholesky(_L)

    _fig, _ax = plt.subplots(figsize=(4.5, 3.8), layout="constrained")
    _fig.colorbar(_ax.imshow(cov_obs.todense(), cmap="RdBu_r"), ax=_ax)
    _ax.set_title(r"$\mathbf{R}$ (cov_obs)")
    _fig
    return (cov_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The observations are perturbed with a sample of the noise to avoid the *inverse
    crime* (using the same forward model to generate the synthetic data and to invert
    them makes the problem artificially simple). Only the perturbed values are used in
    the inversion.
    """)
    return


@app.cell
def _(cov_obs, np, obs, plt):
    obs_perturb = (
        obs + cov_obs.sample_mvnormal([1], random_state=np.random.default_rng(2151))[0]
    )

    _fig, _ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
    _lims = np.array([obs.min(), obs.max()]) + np.array([-0.2, 0.2]) * np.ptp(obs)
    _ax.plot(_lims, _lims, color="r")
    _ax.scatter(obs_perturb, obs)
    _ax.set(
        xlabel="Perturbed values (observations)",
        ylabel="Noise-free values",
        xlim=_lims,
        ylim=_lims,
        aspect="equal",
    )
    _fig
    return (obs_perturb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inversion settings

    The prior covariance matrix is compressed with a truncated eigen decomposition (with
    `covmats`). Fewer principal components (PC) means fewer forward calls per iteration
    (critical for expensive models) but also more smoothing (loss of details).
    Modify the settings below and press *Run*: the inversion and all the following cells
    are updated.
    """)
    return


@app.cell
def _(mo):
    settings = mo.ui.dictionary(
        {
            "n_pc": mo.ui.slider(
                10, 200, step=10, value=50, label="Number of PC (n_pc)"
            ),
            "maxiter": mo.ui.slider(
                1, 15, value=5, label="Max Gauss-Newton iterations"
            ),
            "is_lm": mo.ui.checkbox(value=True, label="Levenberg-Marquardt"),
            "max_it_lm": mo.ui.slider(2, 10, value=5, label="LM inflation factors"),
            "is_direct_solve": mo.ui.checkbox(
                value=False, label="Direct (Cholesky) solve instead of Krylov"
            ),
        }
    ).form(submit_button_label="Run")
    settings
    return (settings,)


@app.cell
def _(settings):
    # The form has no value until it is first submitted -> use the defaults
    params = settings.value or {
        "n_pc": 50,
        "maxiter": 5,
        "is_lm": True,
        "max_it_lm": 5,
        "is_direct_solve": False,
    }
    return (params,)


@app.cell
def _(cov_prior, covmats, params):
    eig_mat = covmats.eigen_factorize_cov_mat(
        cov_prior, n_pc=params["n_pc"], random_state=0
    )
    return (eig_mat,)


@app.cell
def _(cov_prior, covmats, eig_mat, np, plt):
    _fig, _ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    _ax.plot(
        np.arange(1, eig_mat.n_pc + 1),
        np.cumsum(covmats.get_explained_var(eig_mat.eig_vals, cov_prior)) * 100.0,
    )
    _ax.set(
        xlabel="Number of principal components",
        ylabel="Cumulative explained variance (%)",
    )
    _ax.grid(alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the inversion

    Two lines are enough: create the `PCGA` instance and call `run`.
    """)
    return


@app.cell
def _(
    cov_obs,
    eig_mat,
    forward_model,
    obs_perturb,
    params,
    pcga_logger,
    pypcga,
    s_init,
):
    solver = pypcga.PCGA(
        s_init=s_init,
        obs=obs_perturb,
        cov_obs=cov_obs,
        forward_model=forward_model,
        Q=eig_mat,
        maxiter=params["maxiter"],
        is_lm=params["is_lm"],
        max_it_lm=params["max_it_lm"],
        is_direct_solve=params["is_direct_solve"],
        random_state=2026,
        logger=pcga_logger,
    )
    s_hat, simul_obs, post_diagv, iter_best = solver.run()
    return iter_best, post_diagv, s_hat, simul_obs, solver


@app.cell
def _(iter_best, mo, solver):
    mo.md(f"""
    **Status**: {solver.istate.status} - best solution at iteration {iter_best}
    (objective function = {solver.istate.best_obj:.3e}).
    Model-checking statistics (should be close to 1):
    $Q_2$ = {solver.istate.Q2_best:.3f}, $c_R$ = {solver.istate.best_cR:.3f}.
    """)
    return


@app.cell
def _(np, obs_perturb, plt, s_hat, s_ref, simul_obs, solver, to_img):
    _fig, _axes = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained")
    for _ax, (_title, _s) in zip(
        _axes[:2], {"Reference": s_ref, "Estimate": s_hat}.items()
    ):
        _im = _ax.imshow(to_img(_s), origin="lower", cmap="jet", vmin=0.0, vmax=120.0)
        _ax.set_title(_title)
    _fig.colorbar(_im, ax=_axes[:2], shrink=0.8)

    _lims = [
        min(obs_perturb.min(), simul_obs.min()),
        max(obs_perturb.max(), simul_obs.max()),
    ]
    _axes[2].plot(_lims, _lims, "k-")
    _axes[2].plot(obs_perturb, simul_obs.ravel(), ".")
    _axes[2].set(
        xlabel="observed", ylabel="simulated", title="Data fit", aspect="equal"
    )
    _axes[2].ticklabel_format(style="sci", scilimits=(0, 0))

    _axes[3].semilogy(
        np.arange(len(solver.istate.obj_seq)), solver.istate.obj_seq, "o-"
    )
    _axes[3].set(xlabel="iteration", ylabel="objective function", title="Convergence")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior covariance

    The solver returns the posterior variance as a vector of size $N_s$
    (`post_diagv`). A low-rank approximation of the full posterior covariance matrix can
    also be obtained, with a number of principal components (and an inflation factor)
    that can differ from the inversion. Here we build it with 25 and 50 PC, and also
    build the dense matrix for comparison (not possible for large-scale problems).
    """)
    return


@app.cell
def _(solver):
    post_cov_dense = solver.get_dense_post_cov()
    post_cov_25_pc = solver.get_eigen_post_cov(n_pc=25)
    post_cov_50_pc = solver.get_eigen_post_cov(n_pc=50)
    return post_cov_25_pc, post_cov_50_pc, post_cov_dense


@app.cell
def _(np, plt, post_cov_25_pc, post_cov_50_pc, post_cov_dense, post_diagv, to_img):
    _data = {
        "post_diagv": post_diagv,
        "Diag of dense post cov": np.diagonal(post_cov_dense),
        "Diag of eigen post cov (25 PC)": post_cov_25_pc.get_diagonal(),
        "Diag of eigen post cov (50 PC)": post_cov_50_pc.get_diagonal(),
    }
    _vmax = max(np.max(_v) for _v in _data.values())
    _fig, _axes = plt.subplots(
        2, 2, figsize=(9, 8.5), layout="constrained", sharex=True, sharey=True
    )
    for _ax, (_title, _v) in zip(_axes.ravel(), _data.items()):
        _im = _ax.imshow(
            to_img(_v), origin="lower", cmap="viridis", vmin=0.0, vmax=_vmax
        )
        _ax.set_title(_title)
    _fig.colorbar(_im, ax=_axes, shrink=0.6, label="Posterior variance")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The posterior covariance matrix can be used to draw realizations (posterior
    samples) and thus to quantify the uncertainty on predictions (at the cost of one
    forward call per sample). Below, 200 realizations are drawn from the 50 PC
    approximation and added to the estimate. The first 25 are displayed.
    """)
    return


@app.cell
def _(post_cov_50_pc, s_hat, solver):
    post_samples = (
        s_hat.T
        + post_cov_50_pc.sample_mvnormal(shape=(200,), random_state=solver.random_state)
    ).T
    post_samples.shape
    return (post_samples,)


@app.cell
def _(plt, post_samples, to_img):
    _fig, _axes = plt.subplots(
        5, 5, figsize=(10, 9.3), layout="constrained", sharex=True, sharey=True
    )
    for _i, _ax in enumerate(_axes.ravel()):
        _im = _ax.imshow(
            to_img(post_samples[:, _i]),
            origin="lower",
            cmap="jet",
            vmin=0.0,
            vmax=120.0,
        )
        _ax.set_title(f"r#{_i}", fontsize=9)
    _fig.colorbar(_im, ax=_axes, shrink=0.5, label="Parameter value")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, the posterior variance can be estimated back from the ensemble of
    realizations:
    """)
    return


@app.cell
def _(covmats, np, plt, post_cov_50_pc, post_diagv, post_samples, to_img):
    post_cov_ens = covmats.CovViaEnsemble(post_samples.T)
    _data = {
        "post_diagv": post_diagv,
        "Diag of eigen post cov (50 PC)": post_cov_50_pc.get_diagonal(),
        "Diag of ensemble post cov (200 R)": post_cov_ens.get_diagonal(),
    }
    _vmax = max(np.max(_v) for _v in _data.values())
    _fig, _axes = plt.subplots(
        1, 3, figsize=(12, 3.8), layout="constrained", sharey=True
    )
    for _ax, (_title, _v) in zip(_axes, _data.items()):
        _im = _ax.imshow(
            to_img(_v), origin="lower", cmap="viridis", vmin=0.0, vmax=_vmax
        )
        _ax.set_title(_title, fontsize=10)
    _fig.colorbar(_im, ax=_axes, shrink=0.8, label="Posterior variance")
    _fig
    return


if __name__ == "__main__":
    app.run()
