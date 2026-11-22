import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - 3D ERT with E4D")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3D electrical resistivity tomography (ERT) with E4D

    Example from the AGU 2014 presentation by Peter Kitanidis, rewritten in python to
    illustrate an ERT application of PCGA. A detailed explanation is given in the
    arXiv write-up [*Accelerated 3D Electrical Resistivity Tomography with a Scalable
    Jacobian-free Approach*](http://arxiv.org/abs/2202.00059).

    - Unknowns: 256,711 log-conductivities on a regular 91 x 91 x 31 grid
      (90 m x 90 m x 30 m).
    - Measurements: 60,235 ERT data (error std = 15).

    **Requirements**: install [E4D](https://github.com/pnnl/E4D) (see its
    [installation guide](https://github.com/pnnl/E4D/blob/master/Installation.txt)) and
    copy the `e4d` and `mpirun` executables in `input_files/`. Each E4D run uses 6 MPI
    processes (see `ert.py`); adjust `ncores` (the number of concurrent E4D runs) below
    to your machine.
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
    nx = ny = 91
    nz = 31
    dx = 1.0  # m, i.e. a 90 m x 90 m x 30 m domain
    obs = np.loadtxt(here / "obs.txt")

    forward_params = {
        "homedir": str(here),
        "deletedir": True,
        "xyz": np.load(here / "xyz.npy"),
        "xyzout": np.load(here / "xyzout.npy"),
        "elexyz": np.load(here / "elexyz.npy"),
        "xyzoutval": -4.6052,
    }
    return dx, forward_params, nx, ny, nz, obs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior and inversion settings

    | Description | Value |
    |---|---|
    | Prior kernel | $q(x, x') = 0.5^2 \exp(-r)$, scales (30, 30, 15) m |
    | Number of principal components | 96 |
    | Observation error std | 15 |
    | Initial guess | $\ln(0.015)$ everywhere |
    | Max. iterations / `restol` | 7 / 0.01 |
    """)
    return


@app.cell
def _(covmats, dx, np, nx, ny, nz, obs):
    prior_std = 0.5
    prior_cov_scale = np.array([30.0, 30.0, 15.0])

    def kernel(r):
        return prior_std**2 * np.exp(-r)

    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel,
        mesh_dim=[dx, dx, dx],
        domain_shape=[nx, ny, nz],
        len_scale=prior_cov_scale,
    )
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=96, random_state=0)
    cov_obs = covmats.CovViaDiagonal(np.full(obs.size, 15.0**2))
    return Q, cov_obs


@app.cell
def _(here, os):
    try:
        import ert  # local wrapper (ert.py)

        wrapper_error = None
    except ImportError as _e:
        ert = None
        wrapper_error = str(_e)

    missing = []
    if wrapper_error is not None:
        missing.append(
            f"the forward model wrapper cannot be imported ({wrapper_error})"
        )
    for _exe in ("e4d", "mpirun"):
        _path = here / "input_files" / _exe
        if not (_path.is_file() and os.access(_path, os.X_OK)):
            missing.append(f"executable not found or not executable: `{_path}`")
    return ert, missing


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run the inversion")
    run_button
    return (run_button,)


@app.cell
def _(ert, forward_params, missing, mo, run_button):
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
    ert_model = ert.Model(forward_params)
    # number of concurrent E4D runs (e.g. 8 on a 48-core machine, each run using 6 cores)
    ncores = 8

    def forward_model(s_ensemble):
        """(N_s, N_e) log-conductivity ensemble -> (N_obs, N_e) ERT data."""
        return ert_model.run(s_ensemble, True, ncores)

    return (forward_model,)


@app.cell
def _(Q, cov_obs, forward_model, np, nx, ny, nz, obs, pcga_logger, pypcga):
    solver = pypcga.PCGA(
        s_init=np.full(nx * ny * nz, np.log(0.015)),
        obs=obs,
        cov_obs=cov_obs,
        forward_model=forward_model,
        Q=Q,
        maxiter=7,
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
def _(np, nx, ny, nz, plt, post_diagv, s_hat):
    _s3d = s_hat.reshape(nz, ny, nx)
    _std3d = np.sqrt(np.maximum(post_diagv, 0.0)).reshape(nz, ny, nx)
    _fig, _axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for _j, _k in enumerate((nz // 6, nz // 2, 5 * nz // 6)):
        _im0 = _axes[0, _j].imshow(_s3d[_k], origin="lower", cmap="jet")
        _axes[0, _j].set_title(f"estimate, layer {_k}")
        _fig.colorbar(_im0, ax=_axes[0, _j], shrink=0.8)
        _im1 = _axes[1, _j].imshow(_std3d[_k], origin="lower", cmap="viridis")
        _axes[1, _j].set_title(f"posterior std, layer {_k}")
        _fig.colorbar(_im1, ax=_axes[1, _j], shrink=0.8)
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
