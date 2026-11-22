import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - TOUGH joint inversion")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Joint inversion of pressure and temperature data with TOUGH

    pypcga application using [LBNL's TOUGH simulator](https://tough.lbl.gov/) and
    [PyTOUGH](https://github.com/acroucher/PyTOUGH) (TOUGH Symposium 2018,
    [slides](https://www2.hawaii.edu/~jonghyun/files/TOUGH2018_Kokkinaki.pdf); the latest
    version is maintained in the [pyPCGA-TOUGH repository](https://github.com/amaliak/pyPCGA-TOUGH)).

    - Domain: 300 m x 100 m x 20 m, discretized with 30 x 10 x 10 cells.
    - Unknowns: 3,000 log-permeabilities (`pmx` values), log-normal with a Gaussian
      covariance kernel (scales 100, 100 and 10 m).
    - Boundary conditions: injection-extraction system, warm water injected.
    - Simulation time: 374 days.
    - Measurements: pressure and temperature every 5 days at 128 monitoring locations
      between the injection and extraction wells.

    **Requirements**: PyTOUGH (`mulgrids`, `t2data`, ...), `pandas`, an MPI launcher
    (`mpirun`) and the `tough2-mp-eos1.debug` executable on the `PATH`, and the TOUGH
    input files in `input_files/` (not shipped, see the pyPCGA-TOUGH repository).
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
    nx = [30, 10, 10]
    dx = [10.0, 10.0, 2.0]
    forward_model_params = {
        "homedir": str(here),
        "nx": nx,
        "dx": dx,
        "deletedir": False,
        "xlocs": np.arange(5, 21, 2),
        "ylocs": np.array([1, 3, 5, 7]),
        "zlocs": np.array([3, 5, 7, 9]),
        "obs_type": ["Gas Pressure", "Temperature"],
        "t_obs_interval": 86400.0 * 5.0,
        "max_timesteps": 9000,
        "tstop": 0.32342126e08,
        "const_timestep": 6,
        "max_timestep": 86400,
        "absolute_error": 1,
        "relative_error": 5.0e-6,
        "print_interval": 9000,
        "timestep_reduction": 3.0,
        "gravity": 9.81,
        "default_incons": [100.0e4, 10],
        "multi_params": {
            "num_components": 1,
            "num_equations": 2,
            "num_phases": 2,
            "num_secondary_parameters": 6,
        },
        "solver_params": {
            "type": 5,
            "z_precond": 1,
            "o_precond": 0,
            "relative_max_iterations": 0.8,
            "closure": 1.0e-7,
        },
        "output_times_params": {
            "num_times_specified": 2,
            "time": [0.8640e04, 0.32342126e08],
        },
    }

    s_true = np.loadtxt(here / "true_30_10_10_gau.txt")  # for comparison only

    # Noisy measurements: 7400 pressures (Pa) followed by temperatures (C)
    obs = np.loadtxt(here / "obs.txt")
    n_pres = 7400
    std_obs = np.where(np.arange(obs.size) < n_pres, 1000.0, 0.5)
    return dx, forward_model_params, n_pres, nx, obs, s_true, std_obs


@app.cell
def _(nx, plt, s_true):
    _fig, _axes = plt.subplots(
        1, 3, figsize=(13, 2.6), layout="constrained", sharey=True
    )
    _s3d = s_true.reshape(nx[2], nx[1], nx[0])
    for _ax, _k in zip(_axes, range(1, 10, 3)):
        _im = _ax.pcolormesh(_s3d[_k], vmin=-2.0, vmax=1.5, cmap="jet")
        _ax.set(title=f"true ln(pmx), layer {_k}", aspect="equal", xticks=[], yticks=[])
    _fig.colorbar(_im, ax=_axes, shrink=0.9)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior and inversion settings

    | Description | Value |
    |---|---|
    | Prior kernel (Gaussian) | $q(x, x') = 0.7^2 \exp(-r^2)$, scales (100, 100, 10) m |
    | Number of principal components | 30 |
    | Observation error std | 1000 Pa (pressure), 0.5 °C (temperature) |
    | Max. iterations / `restol` | 3 / 0.6 |
    | Finite-difference perturbation (`eps`) | 2e-3 |
    """)
    return


@app.cell
def _(covmats, dx, np, nx, std_obs):
    prior_std = 0.7
    prior_cov_scale = np.array([100.0, 100.0, 10.0])

    def kernel(r):
        return prior_std**2 * np.exp(-(r**2))

    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel, mesh_dim=dx, domain_shape=nx, len_scale=prior_cov_scale
    )
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=30, random_state=0)
    cov_obs = covmats.CovViaDiagonal(std_obs**2)
    return Q, cov_obs


@app.cell
def _(here, shutil):
    try:
        import tough  # local wrapper (tough.py), requires PyTOUGH and pandas

        wrapper_error = None
    except ImportError as _e:
        tough = None
        wrapper_error = str(_e)

    missing = []
    if wrapper_error is not None:
        missing.append(
            f"the forward model wrapper cannot be imported ({wrapper_error})"
        )
    for _exe in ("mpirun", "tough2-mp-eos1.debug"):
        if shutil.which(_exe) is None:
            missing.append(f"`{_exe}` not found on the PATH")
    if not (here / "input_files").is_dir():
        missing.append(f"TOUGH input files not found in `{here / 'input_files'}`")
    return missing, tough


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run the inversion")
    run_button
    return (run_button,)


@app.cell
def _(forward_model_params, missing, mo, run_button, tough):
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
    tough_model = tough.Model(forward_model_params)
    # each TOUGH run itself uses 6 MPI processes (see tough.py)
    ncores = 6

    def forward_model(s_ensemble):
        """(N_s, N_e) ln(pmx) ensemble -> (N_obs, N_e) pressures and temperatures."""
        return tough_model.run(s_ensemble, True, ncores)

    return (forward_model,)


@app.cell
def _(Q, cov_obs, forward_model, np, obs, pcga_logger, pypcga, s_true):
    solver = pypcga.PCGA(
        s_init=np.full(s_true.size, np.mean(s_true)),
        obs=obs,
        cov_obs=cov_obs,
        forward_model=forward_model,
        Q=Q,
        maxiter=3,
        restol=0.6,
        eps=2e-3,
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
def _(np, nx, plt, post_diagv, s_hat, s_true):
    _shape = (nx[2], nx[1], nx[0])
    _data = {
        "true": s_true.reshape(_shape),
        "estimated": s_hat.reshape(_shape),
        "posterior std": np.sqrt(np.maximum(post_diagv, 0.0)).reshape(_shape),
    }
    _fig, _axes = plt.subplots(
        3, 3, figsize=(13, 7), layout="constrained", sharex=True, sharey=True
    )
    for _j, _k in enumerate(range(1, 10, 3)):
        for _i, (_name, _v) in enumerate(_data.items()):
            _kw = dict(vmin=-2.0, vmax=1.5) if _i < 2 else {}
            _im = _axes[_i, _j].pcolormesh(_v[_k], cmap="jet", **_kw)
            _axes[_i, _j].set(
                title=f"{_name} ln(pmx), layer {_k}",
                aspect="equal",
                xticks=[],
                yticks=[],
            )
            if _j == 2:
                _fig.colorbar(_im, ax=_axes[_i, :], shrink=0.9)
    _fig
    return


@app.cell
def _(n_pres, np, obs, plt, simul_obs, solver):
    _sim = simul_obs.ravel()
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for _ax, (_title, _sl) in zip(
        _axes[:2],
        {
            "pressure (Pa)": slice(None, n_pres),
            "temperature (C)": slice(n_pres, None),
        }.items(),
    ):
        _lims = [
            min(obs[_sl].min(), _sim[_sl].min()),
            max(obs[_sl].max(), _sim[_sl].max()),
        ]
        _ax.plot(_lims, _lims, "k-")
        _ax.plot(obs[_sl], _sim[_sl], ".")
        _ax.set(title=_title, xlabel="observation", ylabel="simulation", aspect="equal")
    _axes[0].ticklabel_format(style="sci", scilimits=(0, 0))
    _axes[2].semilogy(
        np.arange(len(solver.istate.obj_seq)), solver.istate.obj_seq, "o-"
    )
    _axes[2].set(title="Objective function", xlabel="iteration")
    _fig
    return


if __name__ == "__main__":
    app.run()
