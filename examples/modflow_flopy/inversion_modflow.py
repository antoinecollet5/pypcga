import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - MODFLOW hydraulic tomography")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hydraulic tomography with MODFLOW (FloPy)

    *Adapted from Application 1 of [Lee and Kitanidis, WRR 2014](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014WR015483)
    (not exactly the same setting).*

    870 steady-state head observations were generated from a true log-conductivity
    field with crosswell pumping tests: groundwater is extracted at one well and the
    hydraulic head is observed at the 29 other wells; this is repeated for each of the
    30 wells. Gaussian noise (std = 0.5 m) was added. **Using these observations and
    MODFLOW, can we reconstruct the true field?**

    The number of MODFLOW runs is larger than reported in the paper because the
    Levenberg-Marquardt and line search options are on. They increase the inversion
    time, but usually give a very accurate solution even for highly nonlinear problems.

    **Requirements**: `flopy` and a working MODFLOW-2005 executable. The executables
    shipped in `input_files/` are old builds (the Linux one needs `libgfortran.so.3`,
    absent from recent distributions). If it does not run on your machine, install a
    recent one (e.g. with `get-modflow :python` from FloPy) and set the `MF2005_EXE`
    environment variable to its path, or put `mf2005` on your `PATH`.
    """)
    return


@app.cell
def _():
    import logging
    import os
    import shutil
    import sys

    import covmats
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pypcga

    here = mo.notebook_dir()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    return covmats, here, logging, mo, np, os, plt, pypcga, shutil


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
    | Domain length and width $L_x$, $L_y$ (m) | 1000, 750 |
    | Grid size $dx$, $dy$ (m) | 10 |
    | Pumping rate $Q$ (m³/d) / recharge (m/d) | 25 / 0.001 |
    | True field | mean(log K) = 2.5 (ln m/d) |
    | Prior kernel used in the inversion | $q(x, x') = \exp(-\lvert x - x' \rvert / 250)$ |
    | Number of principal components | 50 |
    | Number of observations / error std (m) | 870 / 0.5 |
    """)
    return


@app.cell
def _(here, np):
    Lx, Ly = 1000.0, 750.0
    nlay, nrow, ncol = 1, 75, 100
    dx = dy = 10.0

    # cell-center coordinates
    x = np.linspace(dx / 2.0, Lx - dx / 2.0, ncol)
    y = np.linspace(dy / 2.0, Ly - dy / 2.0, nrow)
    XX, YY = np.meshgrid(x, y)

    # 30 wells: each one is pumped in turn, heads are observed at the other ones
    obs_locmat = np.zeros((nlay, nrow, ncol), dtype=bool)
    obs_locmat[0, 5:71:16, 9:96:16] = True
    Q_locs = list(zip(*np.nonzero(obs_locmat)))

    s_true = np.loadtxt(here / "true_logK.txt")  # for comparison only
    obs = np.loadtxt(here / "obs.txt")
    assert obs.size == len(Q_locs) * (len(Q_locs) - 1)
    return (
        Lx,
        Ly,
        Q_locs,
        XX,
        YY,
        dx,
        dy,
        nlay,
        ncol,
        nrow,
        obs,
        obs_locmat,
        s_true,
        x,
        y,
    )


@app.cell
def _(Lx, Ly, XX, YY, ncol, nrow, obs_locmat, plt, s_true, x, y):
    _fig, _ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    _im = _ax.pcolormesh(XX, YY, s_true.reshape(nrow, ncol), cmap="jet")
    _rows, _cols = obs_locmat[0].nonzero()
    _ax.plot(x[_cols], y[_rows], "ko", label="wells")
    _ax.set(
        xlim=(0, Lx),
        ylim=(0, Ly),
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
        title="True log K and wells",
    )
    _ax.legend(loc="upper right")
    _fig.colorbar(_im, ax=_ax, label="ln K (ln m/d)")
    _fig
    return


@app.cell
def _(covmats, dx, dy, ncol, np, nrow, obs):
    prior_std = 1.0
    prior_cov_scale = np.array([250.0, 250.0])

    def kernel(r):
        return prior_std**2 * np.exp(-r)

    # Stationary kernel on a regular grid -> FFT-based matrix-vector products.
    # Vectors are ordered with x (columns) varying fastest, hence (ncol, nrow).
    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel, mesh_dim=[dx, dy], domain_shape=[ncol, nrow], len_scale=prior_cov_scale
    )
    n_pc = 50
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=n_pc, random_state=0)
    cov_obs = covmats.CovViaDiagonal(np.full(obs.size, 0.5**2))
    return Q, cov_obs, n_pc, prior_std


@app.cell
def _(here, os, shutil):
    # MODFLOW executable: MF2005_EXE env variable > mf2005 on the PATH > bundled one
    _bundled = here / "input_files" / ("mf2005.exe" if os.name == "nt" else "mf2005")
    mf_exec = os.environ.get("MF2005_EXE") or shutil.which("mf2005") or str(_bundled)

    try:
        import mf  # local wrapper (mf.py), requires flopy

        wrapper_error = None
    except ImportError as _e:
        mf = None
        wrapper_error = str(_e)

    missing = []
    if wrapper_error is not None:
        missing.append(
            f"the forward model wrapper cannot be imported ({wrapper_error})"
        )
    if not (os.path.isfile(mf_exec) and os.access(mf_exec, os.X_OK)):
        missing.append(f"MODFLOW executable not found or not executable: `{mf_exec}`")
    return mf, mf_exec, missing


@app.cell
def _(mo):
    run_button = mo.ui.run_button(
        label="Run the inversion (about 1,500 MODFLOW runs per iteration)"
    )
    run_button
    return (run_button,)


@app.cell
def _(
    Lx,
    Ly,
    Q_locs,
    here,
    mf,
    mf_exec,
    missing,
    mo,
    ncol,
    nlay,
    nrow,
    obs_locmat,
    os,
    run_button,
):
    mo.stop(
        bool(missing),
        mo.callout(
            mo.md("Cannot run the inversion:\n\n- " + "\n- ".join(missing)), kind="warn"
        ),
    )
    # Run on click, or directly when executed as a script (python inversion_modflow.py)
    mo.stop(
        not (run_button.value or mo.app_meta().mode == "script"),
        mo.md("Press the button above to run the inversion."),
    )

    mf_params = {
        "mf_exec": mf_exec,
        "homedir": str(here),
        "input_dir": "input_files",
        "sim_dir": "simul",
        "Lx": Lx,
        "Ly": Ly,
        "Q": 25.0,
        "Rch": 0.001,
        "nlay": nlay,
        "nrow": nrow,
        "ncol": ncol,
        "zbot": -1.0,
        "ztop": 0.0,
        "obs_locmat": obs_locmat,
        "Q_locs": Q_locs,
    }
    mf_model = mf.Model(mf_params)
    ncores = os.cpu_count() or 1

    def forward_model(s_ensemble):
        """(N_s, N_e) log K ensemble -> (N_obs, N_e) heads, run in parallel."""
        return mf_model.run(s_ensemble, ncores > 1, ncores)

    return (forward_model,)


@app.cell
def _(Q, cov_obs, forward_model, np, obs, pcga_logger, pypcga, s_true):
    solver = pypcga.PCGA(
        s_init=np.ones(s_true.size),  # in most cases the initial guess does not matter
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
    return post_diagv, s_hat, simul_obs, solver


@app.cell
def _(mo, solver):
    mo.md(f"""
    **{solver.istate.status}** - best solution at iteration {solver.istate.iter_best}.
    $Q_2$ = {solver.istate.Q2_best:.3f}, $c_R$ = {solver.istate.best_cR:.3f}
    (should be close to 1).
    """)
    return


@app.cell
def _(XX, YY, n_pc, ncol, np, nrow, plt, post_diagv, prior_std, s_hat, s_true):
    _vmin, _vmax = s_true.min(), s_true.max()
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained", sharey=True)
    _fig.suptitle(f"prior var.: ({prior_std:g})$^2$, n_pc: {n_pc}")
    for _ax, (_title, _v) in zip(
        _axes[:2], {"(a) True": s_true, "(b) Estimate": s_hat}.items()
    ):
        _im = _ax.pcolormesh(
            XX, YY, _v.reshape(nrow, ncol), vmin=_vmin, vmax=_vmax, cmap="jet"
        )
        _ax.set(title=_title, xlabel="x (m)", aspect="equal")
    _fig.colorbar(_im, ax=_axes[:2], shrink=0.8, label="ln K")
    _axes[0].set_ylabel("y (m)")
    _std = np.sqrt(np.maximum(post_diagv, 0.0)).reshape(nrow, ncol)
    _fig.colorbar(
        _axes[2].pcolormesh(XX, YY, _std, cmap="jet"), ax=_axes[2], shrink=0.8
    )
    _axes[2].set(
        title="(c) Uncertainty (posterior std)", xlabel="x (m)", aspect="equal"
    )
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
        xlabel="observation (m)",
        ylabel="simulation (m)",
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
