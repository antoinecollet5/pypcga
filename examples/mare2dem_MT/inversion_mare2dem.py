import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - MARE2DEM magnetotellurics")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Resistivity estimation from a magnetotelluric (MT) survey with MARE2DEM

    - Domain: 2 km x 2 km.
    - Unknowns: 10,000 (100 x 100) log-normal resistivities, Gaussian covariance
      kernel with scales of (500, 200) m.
    - Measurements: 5,248 (21 receivers at the ground surface x 32 frequency bands, EM
      amplitude and phase) with a noise level of 0.1.

    While MARE2DEM supports unstructured grids, the resistivities are defined on
    equally spaced points (interpolated on the finite element mesh) to use FFT-based
    covariance products.

    **Requirements**: an MPI launcher (`mpirun`) and a working `MARE2DEM` executable.
    The one shipped in `input_files/` is linked against Intel MKL and Intel MPI
    runtime libraries.
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


@app.cell
def _(here, np):
    nx = nz = 100
    L = 2000.0  # m
    s_true = np.loadtxt(here / "true_100x100.txt")  # ln(rho), for comparison only
    obs = np.loadtxt(here / "obs.txt")
    extent = (0.0, L, L, 0.0)  # depth increases downwards
    return L, extent, nx, nz, obs, s_true


@app.cell
def _(L, extent, np, nx, nz):
    def plot_log_rho(ax, v, title):
        """Plot ln(rho) with a resistivity colorbar in Ohm.m."""
        im = ax.imshow(
            np.asarray(v).reshape(nz, nx),
            vmin=0.0,
            vmax=np.log(1000.0),
            cmap="jet",
            extent=extent,
        )
        ax.set(
            title=title,
            xlabel="x [m]",
            xticks=np.linspace(0, L, 3),
            yticks=np.linspace(0, L, 3),
        )
        return im

    def add_rho_colorbar(fig, im, ax):
        _ticks = np.log([1.0, 10.0, 100.0, 1000.0])
        cbar = fig.colorbar(im, ax=ax, ticks=_ticks, shrink=0.8)
        cbar.ax.set_yticklabels(["1", "10", "100", "1000"])
        cbar.set_label(r"$\rho$ [$\Omega \cdot m$]")
        return cbar

    return add_rho_colorbar, plot_log_rho


@app.cell
def _(add_rho_colorbar, plot_log_rho, plt, s_true):
    _fig, _ax = plt.subplots(figsize=(5, 4), layout="constrained")
    add_rho_colorbar(_fig, plot_log_rho(_ax, s_true, "True resistivity"), _ax)
    _ax.set_ylabel("depth [m]")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior and inversion settings

    | Description | Value |
    |---|---|
    | Prior kernel (Gaussian) | $q(x, x') = 1.0^2 \exp(-r^2)$, scales (500, 200) m |
    | Number of principal components | 50 |
    | Observation error std | 0.1 |
    | Max. iterations / `restol` | 4 / 0.1 |
    | Finite-difference perturbation (`eps`) | 1e-4 |
    """)
    return


@app.cell
def _(L, covmats, np, nx, nz, obs):
    prior_std = 1.0
    prior_cov_scale = np.array([500.0, 200.0])

    def kernel(r):
        return prior_std**2 * np.exp(-(r**2))

    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel,
        mesh_dim=[L / (nx - 1), L / (nz - 1)],
        domain_shape=[nx, nz],
        len_scale=prior_cov_scale,
    )
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=50, random_state=0)
    cov_obs = covmats.CovViaDiagonal(np.full(obs.size, 0.1**2))
    return Q, cov_obs


@app.cell
def _(here, os, shutil):
    try:
        import mare2dem  # local wrapper (mare2dem.py)

        wrapper_error = None
    except ImportError as _e:
        mare2dem = None
        wrapper_error = str(_e)

    missing = []
    if wrapper_error is not None:
        missing.append(
            f"the forward model wrapper cannot be imported ({wrapper_error})"
        )
    if shutil.which("mpirun") is None:
        missing.append("`mpirun` not found on the PATH")
    _exe = shutil.which("MARE2DEM") or str(here / "input_files" / "MARE2DEM")
    if not (os.path.isfile(_exe) and os.access(_exe, os.X_OK)):
        missing.append(f"MARE2DEM executable not found or not executable: `{_exe}`")
    return mare2dem, missing


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run the inversion")
    run_button
    return (run_button,)


@app.cell
def _(here, mare2dem, missing, mo, nx, nz, os, run_button):
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
    m2d_model = mare2dem.Model({"nx": nx, "ny": nz, "homedir": str(here)})
    # each MARE2DEM run itself uses 6 MPI processes (see mare2dem.py)
    ncores = max(1, (os.cpu_count() or 1) // 6)

    def forward_model(s_ensemble):
        """(N_s, N_e) ln(rho) ensemble -> (N_obs, N_e) MT responses."""
        return m2d_model.run(s_ensemble, ncores > 1, ncores)

    return (forward_model,)


@app.cell
def _(Q, cov_obs, forward_model, np, obs, pcga_logger, pypcga, s_true):
    solver = pypcga.PCGA(
        s_init=np.full(s_true.size, np.mean(s_true)),
        obs=obs,
        cov_obs=cov_obs,
        forward_model=forward_model,
        Q=Q,
        maxiter=4,
        restol=0.1,
        eps=1e-4,
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
def _(
    L,
    add_rho_colorbar,
    extent,
    np,
    nx,
    nz,
    plot_log_rho,
    plt,
    post_diagv,
    s_hat,
    s_true,
):
    _fig, _axes = plt.subplots(
        1, 3, figsize=(15, 4.2), layout="constrained", sharey=True
    )
    plot_log_rho(_axes[0], s_true, "true")
    add_rho_colorbar(_fig, plot_log_rho(_axes[1], s_hat, "estimated"), _axes[:2])
    _axes[0].set_ylabel("depth [m]")
    _std = np.sqrt(np.maximum(post_diagv, 0.0)).reshape(nz, nx)
    _fig.colorbar(
        _axes[2].imshow(_std, cmap="jet", extent=extent), ax=_axes[2], shrink=0.8
    )
    _axes[2].set(
        title=r"Uncertainty - posterior std of $\ln \rho$",
        xlabel="x [m]",
        xticks=np.linspace(0, L, 3),
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
        title="obs. vs simul.", xlabel="observed", ylabel="simulated", aspect="equal"
    )
    _axes[1].semilogy(
        np.arange(len(solver.istate.obj_seq)), solver.istate.obj_seq, "o-"
    )
    _axes[1].set(title="Objective function", xlabel="iteration")
    _fig
    return


if __name__ == "__main__":
    app.run()
