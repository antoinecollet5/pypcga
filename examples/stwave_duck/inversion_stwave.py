import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - STWAVE bathymetry")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nearshore bathymetry estimation with STWAVE (Duck, NC)

    The bathymetry (water depth) of a 550 m x 415 m nearshore area at the USACE Field
    Research Facility in Duck, NC, is estimated from wave speed observations simulated
    with the spectral wave model STWAVE (forcing from the 8 m array, October 7, 2015).
    The prior follows Hojat Ghorbanidehno's CSKF paper.

    **Requirements**: the STWAVE executable (`input_files/stwave`, an old build needing
    `libgfortran.so.3`) and ERDC's `stwave_utils` python package, which provides
    `STWaveProblem` and `read_speed` used by `stwave.py`.
    """)
    return


@app.cell
def _():
    import datetime as dt
    import logging
    import os
    import sys

    import covmats
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pypcga

    here = mo.notebook_dir()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    return covmats, dt, here, logging, mo, np, os, plt, pypcga


@app.cell
def _(logging):
    class PrintHandler(logging.Handler):
        """Print log records (marimo shows stdout below the running cell)."""

        def emit(self, record: logging.LogRecord) -> None:
            print(self.format(record))

    pcga_logger = logging.getLogger("PCGA")
    pcga_logger.setLevel(logging.INFO)
    pcga_logger.handlers.clear()
    pcga_logger.addHandler(PrintHandler())
    pcga_logger.propagate = False
    return (pcga_logger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem setup

    | Description | Value |
    |---|---|
    | Grid ($n_x \times n_y$, cell size) | 110 x 83, 5 m |
    | Prior kernel (Gaussian) | $q(x, x') = 1.5^2 \exp\left(-\lvert x - x' \rvert^2 / 90^2\right)$ |
    | Number of principal components | 50 |
    | Observation error std (m/s) | 0.1 |
    """)
    return


@app.cell
def _(here, np):
    nx, ny = 110, 83
    dx = 5.0
    Lx, Ly = nx * dx, ny * dx  # 550 m, 415 m
    obs = np.loadtxt(here / "obs.txt")
    s_true = np.loadtxt(here / "true_depth.txt")  # for comparison only

    def to_img(v):
        """(ny, nx) image of a vector, flipped to show offshore distance left to right."""
        return np.flipud(np.fliplr(np.asarray(v).reshape(ny, nx)))

    return Lx, Ly, dx, nx, ny, obs, s_true, to_img


@app.cell
def _(covmats, dx, np, nx, ny, obs):
    prior_std = 1.5
    prior_cov_scale = np.array([18.0 * dx, 18.0 * dx])

    def kernel(r):
        return prior_std**2 * np.exp(-(r**2))

    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel, mesh_dim=[dx, dx], domain_shape=[nx, ny], len_scale=prior_cov_scale
    )
    n_pc = 50
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=n_pc, random_state=0)
    cov_obs = covmats.CovViaDiagonal(np.full(obs.size, 0.1**2))
    return Q, cov_obs, n_pc, prior_cov_scale, prior_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Leading eigenvectors (every other one) and eigenvalues of the prior covariance:
    """)
    return


@app.cell
def _(Q, np, nx, ny, plt):
    _fig, _axes = plt.subplots(
        4, 5, figsize=(12, 7), layout="constrained", sharex=True, sharey=True
    )
    for _k, _ax in enumerate(_axes[:, :4].ravel()):
        _ax.imshow(Q.eig_vects[:, 2 * _k].reshape(ny, nx), extent=[0, nx, 0, ny])
        _ax.set_title(f"eigv #{2 * _k}", fontsize=9)
    _gs = _axes[0, 4].get_gridspec()
    for _ax in _axes[:, 4]:
        _ax.remove()
    _axe = _fig.add_subplot(_gs[:, 4])
    _axe.semilogy(np.ravel(Q.eig_vals), "o")
    _axe.set(title="eigenvalues", xlabel="index")
    _fig
    return


@app.cell
def _(dt, here, os):
    stwave_params = {
        "homedir": str(here),
        "nx": 110,
        "ny": 83,
        "Lx": 550,
        "Ly": 415,
        "x0": 62.0,
        "y0": 568.0,
        "t1": dt.datetime(2015, 10, 7, 20, 00),
        "t2": dt.datetime(2015, 10, 7, 21, 00),
        "offline_dataloc": str(
            here / "input_files" / "8m-array_2015100718_2015100722.nc"
        ),
    }

    try:
        import stwave as st  # local wrapper (stwave.py), requires stwave_utils

        wrapper_error = None
    except ImportError as _e:
        st = None
        wrapper_error = str(_e)

    _exe = here / "input_files" / "stwave"
    missing = []
    if wrapper_error is not None:
        missing.append(
            f"the forward model wrapper cannot be imported ({wrapper_error})"
        )
    if not (_exe.is_file() and os.access(_exe, os.X_OK)):
        missing.append(f"STWAVE executable not found or not executable: `{_exe}`")
    return missing, st, stwave_params


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run the inversion")
    run_button
    return (run_button,)


@app.cell
def _(missing, mo, os, run_button, st, stwave_params):
    mo.stop(
        bool(missing),
        mo.callout(
            mo.md("Cannot run the inversion:\n\n- " + "\n- ".join(missing)), kind="warn"
        ),
    )
    mo.stop(
        not (run_button.value or mo.app_meta().mode == "script"),
        mo.md("Press the button above to run the inversion."),
    )
    st_model = st.Model(stwave_params)
    ncores = os.cpu_count() or 1

    def forward_model(s_ensemble):
        """(N_s, N_e) depth ensemble -> (N_obs, N_e) wave speeds, run in parallel."""
        return st_model.run(s_ensemble, ncores > 1, ncores)

    return (forward_model,)


@app.cell
def _(Q, cov_obs, forward_model, np, obs, pcga_logger, pypcga, s_true):
    solver = pypcga.PCGA(
        s_init=np.full(s_true.size, np.mean(s_true)),
        obs=obs,
        cov_obs=cov_obs,
        forward_model=forward_model,
        Q=Q,
        maxiter=10,
        restol=0.01,
        is_lm=True,
        is_line_search=True,
        random_state=0,
        logger=pcga_logger,
    )
    s_hat, simul_obs, post_diagv, iter_best = solver.run()
    post_std = np.sqrt(np.maximum(post_diagv.ravel(), 0.0))
    return post_std, s_hat, simul_obs, solver


@app.cell
def _(mo, solver):
    mo.md(f"""
    **{solver.istate.status}** - best solution at iteration {solver.istate.iter_best}.
    $Q_2$ = {solver.istate.Q2_best:.3f}, $c_R$ = {solver.istate.best_cR:.3f}
    (should be close to 1).
    """)
    return


@app.cell
def _(n_pc, nx, ny, plt, post_std, prior_std, s_hat, s_true, to_img):
    _fig, _axes = plt.subplots(
        1, 3, figsize=(15, 4.2), layout="constrained", sharey=True
    )
    _fig.suptitle(f"prior var.: ({prior_std:g})$^2$, n_pc: {n_pc}")
    for _ax, (_title, _v) in zip(
        _axes[:2], {"(a) True": s_true, "(b) Estimate": s_hat}.items()
    ):
        _im = _ax.imshow(
            to_img(-_v), extent=[0, nx, 0, ny], vmin=-7.0, vmax=0.0, cmap="jet"
        )
        _ax.set(title=_title, xlabel="Offshore distance (px)")
    _fig.colorbar(_im, ax=_axes[:2], shrink=0.8, label="elevation (m)")
    _axes[0].set_ylabel("Alongshore distance (px)")
    _fig.colorbar(
        _axes[2].imshow(to_img(post_std), extent=[0, nx, 0, ny], cmap="jet"),
        ax=_axes[2],
        shrink=0.8,
    )
    _axes[2].set(title="(c) Uncertainty (std)", xlabel="Offshore distance (px)")
    _fig
    return


@app.cell
def _(dx, np, nx, ny, plt, post_std, prior_cov_scale, s_hat, s_true):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained", sharey=True)
    _fig.suptitle(
        f"Transects (lx = {prior_cov_scale[0]:g} m, ly = {prior_cov_scale[1]:g} m)"
    )
    _linex = np.arange(1, nx + 1) * dx
    for _ax, _px in zip(_axes, (25, 45)):
        _row = ny - _px + 1
        _true = s_true.reshape(ny, nx)[_row]
        _est = s_hat.reshape(ny, nx)[_row]
        _std = post_std.reshape(ny, nx)[_row]
        _ax.fill_between(
            _linex,
            np.flipud(-(_est + 1.96 * _std)),
            np.flipud(-(_est - 1.96 * _std)),
            color="0.8",
            label="95% credible interval",
        )
        _ax.plot(_linex, np.flipud(-_true), "r-", label="True")
        _ax.plot(_linex, np.flipud(-_est), "k-", label="Estimated")
        _ax.set(title=f"{_px * dx:g} m alongshore", xlabel="offshore distance (m)")
    _axes[0].set_ylabel("elevation (m)")
    _axes[0].legend()
    _fig
    return


@app.cell
def _(np, obs, plt, simul_obs, solver):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    _lims = [min(obs.min(), simul_obs.min()), max(obs.max(), simul_obs.max())]
    _axes[0].plot(_lims, _lims, "k-")
    _axes[0].plot(obs, simul_obs.ravel(), ".")
    _axes[0].set(
        title="obs. vs simul.",
        xlabel="observation",
        ylabel="simulation",
        aspect="equal",
    )
    _axes[1].semilogy(
        np.arange(len(solver.istate.obj_seq)), solver.istate.obj_seq, "o-"
    )
    _axes[1].set(title="Objective function", xlabel="iteration")
    _fig
    return


if __name__ == "__main__":
    app.run()
