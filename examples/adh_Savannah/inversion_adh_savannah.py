import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - AdH Savannah River bathymetry")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Riverine bathymetry imaging with AdH (Savannah River)

    ![true bathymetry](public/bathymetry_true.png)

    - A 1 mile reach of the Savannah River near Augusta, GA.
    - 20,541 unknown bathymetry points, estimated from velocity measurements along 5
      simulated drifter tracks (408 locations, see the figure above).
    - Reference: J. Lee, H. Ghorbanidehno, M. Farthing, T. Hesser, E. F. Darve and
      P. K. Kitanidis, *Riverine bathymetry imaging with indirect observations*, Water
      Resources Research, 2018.

    **Requirements**

    - The AdH input directories `mean_files`, `mesh_files`, `observation_files`,
      `sim_files` and `true_files`, available from
      <https://github.com/jonghyunharrylee/adh_lee_et_al_wrr_2018>, copied next to this
      notebook.
    - The `adh` and `pre_adh` executables in `./bin/v4/` (contact the
      [ERDC team](https://www.erdc.usace.army.mil/Media/Fact-Sheets/Fact-Sheet-Article-View/Article/476708/adaptive-hydraulics-model-system/)
      for access).
    - ERDC's `enkf_tools` package and `tables` (PyTables), used by `setup_savannah.py`.

    The original example was run on a 48-core workstation.
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
def _(here):
    sim_params = {
        "homedir": str(here),
        "sim_dir": str(here / "simul"),
        "adh_exec": str(here / "bin" / "v4" / "adh"),
        "pre_adh_exec": str(here / "bin" / "v4" / "pre_adh"),
        "adh_version": 4.5,
        "adh_ntsim": 1,
        "Q_b": 6873.5,
        "z_f": 97.14,
        "adh_grid": str(here / "mesh_files" / "grid_savannah_river_nx501_ny41"),
        "adh_rect": str(here / "mesh_files" / "rect_savannah_river_nx501_ny41"),
        "adh_mesh": str(here / "sim_files" / "savannah_gridgen_new_nx501_ny41.3dm"),
        "adh_bc": str(here / "sim_files" / "savannah_gridgen_new_nx501_ny41.bc"),
        "velocity_obs_file": str(
            here / "observation_files" / "observation_loc_drogue12345_50ft.dat"
        ),
        "elevation_obs_file": str(
            here / "observation_files" / "observation_loc_none.dat"
        ),
        "true_soln_file_h5": str(
            here / "true_files" / "savannah_gridgen_true_nx501_ny41_p0.h5"
        ),
        "true_soln_meshbase": str(
            here / "true_files" / "savannah_gridgen_true_nx501_ny41"
        ),
    }
    return (sim_params,)


@app.cell
def _(here, np):
    s_true = np.loadtxt(here / "true.txt")  # for comparison only
    obs = np.loadtxt(here / "obs.txt")
    # mesh info for visualization
    triangles = np.loadtxt(here / "triangles.txt")
    meshnode = np.loadtxt(here / "meshnode.txt")

    # transformed (curvilinear) domain used for the prior
    nx, ny = 501, 41
    return meshnode, nx, ny, obs, s_true, triangles


@app.cell
def _(meshnode, np, os, plt, sim_params, triangles):
    def plot_on_mesh(ax, values, title, label, **kwargs):
        im = ax.tripcolor(
            meshnode[:, 0],
            meshnode[:, 1],
            triangles,
            np.ravel(values),
            cmap="jet",
            **kwargs,
        )
        ax.set(
            xlabel="Easting [ft]", ylabel="Northing [ft]", title=title, aspect="equal"
        )
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.05, label=label)
        return im

    # velocity observation locations, if the AdH input files are available
    _f = sim_params["velocity_obs_file"]
    velocity_obs_loc = np.loadtxt(_f) if os.path.isfile(_f) else None
    return plot_on_mesh, velocity_obs_loc


@app.cell
def _(plot_on_mesh, plt, s_true, velocity_obs_loc):
    _fig, _ax = plt.subplots(figsize=(9, 9), layout="constrained")
    plot_on_mesh(_ax, s_true, "True Savannah bathymetry", "Elevation [ft]")
    if velocity_obs_loc is not None:
        _ax.scatter(
            velocity_obs_loc[:, 0],
            velocity_obs_loc[:, 1],
            c="k",
            s=0.5,
            alpha=0.7,
            label="obs",
        )
        _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior and inversion settings

    The prior is defined in the transformed (curvilinear, 501 x 41 cells) domain.

    | Description | Value |
    |---|---|
    | Prior kernel (Gaussian) | $q(x, x') = 3.3^2 \exp(-r^2)$ (ft²), scales (48, 12) cells |
    | Number of principal components | 50 (use e.g. 100 for a more accurate uncertainty map) |
    | Observation error std (ft/s) | 0.17 |
    | Max. iterations / `restol` | 5 / 0.01 |
    """)
    return


@app.cell
def _(covmats, np, nx, ny, obs):
    prior_std = 3.3  # ft
    prior_cov_scale = np.array([48.0, 12.0])  # cells in the transformed domain

    def kernel(r):
        return prior_std**2 * np.exp(-(r**2))

    # same grid as the original example: linspace(0, 501, 501) x linspace(0, 41, 41)
    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel,
        mesh_dim=[501.0 / (nx - 1), 41.0 / (ny - 1)],
        domain_shape=[nx, ny],
        len_scale=prior_cov_scale,
    )
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=50, random_state=0)
    cov_obs = covmats.CovViaDiagonal(np.full(obs.size, 0.17**2))
    return Q, cov_obs


@app.cell
def _(here, os, sim_params):
    try:
        import adh  # local wrapper (adh.py), requires enkf_tools and tables

        wrapper_error = None
    except ImportError as _e:
        adh = None
        wrapper_error = str(_e)

    missing = []
    if wrapper_error is not None:
        missing.append(
            f"the forward model wrapper cannot be imported ({wrapper_error})"
        )
    for _key in ("adh_exec", "pre_adh_exec"):
        if not (
            os.path.isfile(sim_params[_key]) and os.access(sim_params[_key], os.X_OK)
        ):
            missing.append(
                f"executable not found or not executable: `{sim_params[_key]}`"
            )
    for _d in (
        "mean_files",
        "mesh_files",
        "observation_files",
        "sim_files",
        "true_files",
    ):
        if not (here / _d).is_dir():
            missing.append(f"AdH input directory not found: `{here / _d}`")
    return adh, missing


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run the inversion")
    run_button
    return (run_button,)


@app.cell
def _(adh, missing, mo, os, run_button, sim_params):
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
    adh_model = adh.Model(sim_params)
    ncores = os.cpu_count() or 1

    def forward_model(s_ensemble):
        """(N_s, N_e) bathymetry ensemble -> (N_obs, N_e) velocities, in parallel."""
        return adh_model.run(s_ensemble, ncores > 1, ncores).reshape(
            -1, s_ensemble.shape[1]
        )

    return (forward_model,)


@app.cell
def _(Q, cov_obs, forward_model, here, np, obs, pcga_logger, pypcga, s_true):
    solver = pypcga.PCGA(
        s_init=np.full(s_true.size, np.mean(s_true)),
        obs=obs,
        cov_obs=cov_obs,
        forward_model=forward_model,
        Q=Q,
        maxiter=5,
        restol=1e-2,
        is_lm=True,
        is_line_search=True,
        random_state=0,
        logger=pcga_logger,
    )
    s_hat, simul_obs, post_diagv, iter_best = solver.run()

    # Save the results, e.g. for model validation and posterior sampling
    np.savez(
        here / "results.npz",
        s_hat=s_hat,
        simul_obs=simul_obs,
        post_diagv=post_diagv,
        iter_best=iter_best,
        obj_seq=np.array(solver.istate.obj_seq),
        Q2=solver.istate.Q2_best,
        cR=solver.istate.best_cR,
    )
    return post_diagv, s_hat, simul_obs, solver


@app.cell
def _(mo, solver):
    mo.md(f"""
    **{solver.istate.status}** - best solution at iteration {solver.istate.iter_best}.
    $Q_2$ = {solver.istate.Q2_best:.3f}, $c_R$ = {solver.istate.best_cR:.3f}
    (should be close to 1). Results saved in `results.npz`.
    """)
    return


@app.cell
def _(np, plot_on_mesh, plt, post_diagv, s_hat, s_true):
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
    plot_on_mesh(
        _axes[0],
        s_hat,
        "Estimated bathymetry",
        "Elevation [ft]",
        vmin=s_true.min(),
        vmax=s_true.max(),
    )
    plot_on_mesh(
        _axes[1],
        np.sqrt(np.maximum(post_diagv, 0.0)),
        "Uncertainty (posterior std)",
        "std [ft]",
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
        xlabel="observation [ft/s]",
        ylabel="simulation [ft/s]",
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
