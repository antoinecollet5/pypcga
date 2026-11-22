import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - tracer tomography (CrunchTope)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Permeability estimation from tracer travel times (CrunchTope)

    - An isotope tracer test is performed in a 2D (depth-integrated) confined aquifer,
      and a surface electrical resistivity survey tells when the tracer reaches every
      location of the aquifer.
    - Because of the non-unique petrophysical relationship, we assume that only the mean
      tracer arrival times (first normalized moment of the breakthrough curves,
      $m_1 / m_0$) are reliable.
    - Tracer transport (advection-diffusion) is simulated with
      [CrunchTope](https://bitbucket.org/crunchflow/crunchtope/downloads/). The forward
      problem was prepared by Mahta Ansari (UIUC, Druhan Lab).
    - The domain is **5 m x 5 m** (50 x 50 cells) with a regional groundwater flow from
      left to right.

    **Assuming that we know all the other aquifer conditions, can we reconstruct the
    permeability field from tracer travel times only?**

    **Requirements**: a CrunchTope executable in `input_files/` (the shipped one is linked
    against Intel Fortran runtime libraries). For more information on reactive transport
    modeling, see the [PSU PNG550/CE574 lecture page](https://www.e-education.psu.edu/png550/node/829).
    """)
    return


@app.cell
def _():
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
    return covmats, here, logging, mo, np, os, plt, pypcga


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
    nx = ny = 50
    dx = 5.0 / nx  # 0.1 m cells
    # breakthrough-curve record times (days) used to compute the travel times
    t = np.array([1.1574e-05, 0.01, *np.round(np.arange(0.1, 2.01, 0.1), 1)])

    s_true = np.loadtxt(here / "true.txt")  # true ln k, for comparison only
    cobs = np.loadtxt(here / "cobs.txt")  # concentrations simulated with s_true
    # Noisy mean travel times (days), generated as
    # obs = simul_obs + 0.01 * noise (error std = 0.01 day), clipped at 0.
    obs = np.loadtxt(here / "obs.txt")
    return cobs, dx, nx, ny, obs, s_true, t


@app.cell
def _(cobs, nx, ny, plt, s_true, t):
    _fig, _axes = plt.subplots(
        1, 5, figsize=(15, 3.3), layout="constrained", sharey=True
    )
    _fig.colorbar(
        _axes[0].pcolormesh(s_true.reshape(ny, nx), cmap="viridis"),
        ax=_axes[0],
        shrink=0.7,
    )
    _axes[0].set(title="true ln k", aspect="equal")
    for _ax, _it in zip(_axes[1:], (2, 4, 6, 11)):
        _im = _ax.pcolormesh(cobs[:, _it].reshape(ny, nx), vmin=0.0, vmax=1.0e-7)
        _ax.set(title=f"tracer at {t[_it]:3.1f} day", aspect="equal")
    _fig.colorbar(_im, ax=_axes[1:], shrink=0.7, label="concentration")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior and observation errors

    | Description | Value |
    |---|---|
    | Prior kernel | $q(x, x') = 0.2^2 \exp\left(-\lvert x - x' \rvert / 2\right)$ |
    | Number of principal components | 50 |
    | Observation error std (day) | 0.01 |
    """)
    return


@app.cell
def _(covmats, dx, np, nx, ny, obs):
    prior_std = 0.2
    prior_cov_scale = np.array([2.0, 2.0])

    def kernel(r):
        return prior_std**2 * np.exp(-r)

    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel, mesh_dim=[dx, dx], domain_shape=[nx, ny], len_scale=prior_cov_scale
    )
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=50, random_state=0)
    cov_obs = covmats.CovViaDiagonal(np.full(obs.size, 0.01**2))
    return Q, cov_obs


@app.cell
def _(here, os):
    try:
        import crunch  # local wrapper (crunch.py)

        wrapper_error = None
    except ImportError as _e:
        crunch = None
        wrapper_error = str(_e)

    _exe = here / "input_files" / "CrunchTope"
    missing = []
    if wrapper_error is not None:
        missing.append(
            f"the forward model wrapper cannot be imported ({wrapper_error})"
        )
    if not (_exe.is_file() and os.access(_exe, os.X_OK)):
        missing.append(f"CrunchTope executable not found or not executable: `{_exe}`")
    return crunch, missing


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run the inversion")
    run_button
    return (run_button,)


@app.cell
def _(crunch, here, missing, mo, nx, ny, os, run_button, t):
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
    crunch_model = crunch.Model({"nx": nx, "ny": ny, "t": t, "homedir": str(here)})
    ncores = os.cpu_count() or 1

    def forward_model(s_ensemble):
        """(N_s, N_e) ln k ensemble -> (N_obs, N_e) travel times, run in parallel."""
        return crunch_model.run(s_ensemble, ncores > 1, ncores)

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
    return post_diagv, s_hat, simul_obs, solver


@app.cell(hide_code=True)
def _(mo, solver):
    mo.md(f"""
    **{solver.istate.status}** - best solution at iteration {solver.istate.iter_best}.
    $Q_2$ = {solver.istate.Q2_best:.3f}, $c_R$ = {solver.istate.best_cR:.3f}
    (should be close to 1).

    It has been shown that sparse tracer data alone are not informative enough to
    estimate the underlying permeability field [Cirpka and Kitanidis, 2000; Lee and
    Kitanidis, 2014]. However, with **lots** of tracer information, the field can be
    reconstructed quite well without any pressure measurement (see Lee et al. [2016]).
    With the original setup, the 50 x 50 field was estimated with less than 200
    simulations (6 minutes on a 36-core machine).
    """)
    return


@app.cell
def _(np, nx, ny, plt, post_diagv, s_hat, s_true):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained", sharey=True)
    for _ax, (_title, _v) in zip(
        _axes[:2], {"true ln k": s_true, "estimated ln k": s_hat}.items()
    ):
        _im = _ax.pcolormesh(_v.reshape(ny, nx), vmin=-33.5, vmax=-28.0)
        _ax.set(title=_title, aspect="equal")
    _fig.colorbar(_im, ax=_axes[:2], shrink=0.8)
    _std = np.sqrt(np.maximum(post_diagv, 0.0)).reshape(ny, nx)
    _fig.colorbar(_axes[2].pcolormesh(_std), ax=_axes[2], shrink=0.8)
    _axes[2].set(title="Uncertainty (posterior std)", aspect="equal")
    _fig
    return


@app.cell
def _(np, obs, plt, simul_obs, solver):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    _lim = max(obs.max(), simul_obs.max())
    _axes[0].plot([0.0, _lim], [0.0, _lim], "k-")
    _axes[0].plot(obs, simul_obs.ravel(), ".")
    _axes[0].set(
        title="obs. vs simul.",
        xlabel="observed travel time (day)",
        ylabel="simulated (day)",
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
