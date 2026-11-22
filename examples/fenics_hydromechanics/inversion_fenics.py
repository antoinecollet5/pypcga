import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - FEniCS hydromechanics")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Permeability estimation in a poroelastic medium (FEniCS)

    pypcga application for permeability estimation using
    [FEniCS](https://fenicsproject.org/) and
    [multiphenics](https://github.com/multiphenics/multiphenics) (Biot
    hydromechanical model, see `poro.py`). Details can be found in Kadeethum et al.,
    [*A framework for data-driven solution and parameter estimation of PDEs using
    conditional generative adversarial networks*](https://arxiv.org/abs/2105.13136).

    - Unknowns: 16,384 log-permeabilities on a regular 128 x 128 grid over the unit
      square, mapped onto the finite element (DG0) mesh with `logk_idx.txt` (see
      `gen_logk_idx.py`). The vectors are ordered with **y varying fastest**.
    - Measurements: 356 noisy observations (error std = 50).

    **Requirements**: legacy FEniCS (`dolfin`, `ufl`) and `multiphenics`.
    """)
    return


@app.cell
def _():
    import logging
    import sys

    import covmats
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pypcga

    here = mo.notebook_dir()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    return covmats, here, logging, mo, np, plt, pypcga


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
    nx = ny = 128
    ptx = np.linspace(0.0, 1.0, nx)
    pty = np.linspace(0.0, 1.0, ny)

    forward_params = {
        "ptx": ptx,
        "pty": pty,
        "pts_fem": np.loadtxt(here / "dof_perm_dg0.csv", delimiter=","),
        "logk_idx": np.loadtxt(here / "logk_idx.txt").astype(int),
    }

    s_true = np.loadtxt(here / "s_true.txt")  # for comparison only
    obs = np.loadtxt(here / "obs.txt")  # noisy observations generated with poro.py

    def to_img(v):
        """(ny, nx) image of a vector ordered with y varying fastest."""
        return np.asarray(v).reshape(nx, ny).T

    return forward_params, nx, ny, obs, ptx, s_true, to_img


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior and inversion settings

    | Description | Value |
    |---|---|
    | Prior kernel | $q(x, x') = 2^2 \exp(-r)$, scale 0.1 |
    | Number of principal components | 96 |
    | Observation error std | 50 |
    | Initial guess | -20 everywhere |
    | Max. iterations / `restol` | 10 / 0.1 |
    """)
    return


@app.cell
def _(covmats, np, nx, ny, obs, ptx):
    prior_std = 2.0
    prior_cov_scale = np.array([0.1, 0.1])

    def kernel(r):
        return prior_std**2 * np.exp(-r)

    # isotropic kernel on a square grid: the (x or y fastest) ordering does not matter
    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel,
        mesh_dim=[ptx[1] - ptx[0]] * 2,
        domain_shape=[ny, nx],
        len_scale=prior_cov_scale,
    )
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=96, random_state=0)
    cov_obs = covmats.CovViaDiagonal(np.full(obs.size, 50.0**2))
    return Q, cov_obs


@app.cell
def _():
    try:
        import poro  # local wrapper (poro.py), requires dolfin and multiphenics

        missing = []
    except ImportError as _e:
        poro = None
        missing = [f"the forward model wrapper cannot be imported ({_e})"]
    return missing, poro


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run the inversion")
    run_button
    return (run_button,)


@app.cell
def _(forward_params, missing, mo, poro, run_button):
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
    poro_model = poro.Model(forward_params)

    def forward_model(s_ensemble):
        """(N_s, N_e) log-permeability ensemble -> (N_obs, N_e), in parallel."""
        return poro_model.run(s_ensemble, True)

    return (forward_model,)


@app.cell
def _(Q, cov_obs, forward_model, np, obs, pcga_logger, pypcga, s_true):
    solver = pypcga.PCGA(
        s_init=np.full(s_true.size, -20.0),
        obs=obs,
        cov_obs=cov_obs,
        forward_model=forward_model,
        Q=Q,
        maxiter=10,
        restol=0.1,
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
def _(np, plt, post_diagv, s_hat, s_true, to_img):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained", sharey=True)
    _data = {
        "true ln k": s_true,
        "estimated ln k": s_hat,
        "posterior std": np.sqrt(np.maximum(post_diagv, 0.0)),
    }
    for _ax, (_title, _v) in zip(_axes, _data.items()):
        _fig.colorbar(
            _ax.imshow(to_img(_v), origin="lower", extent=(0, 1, 0, 1), cmap="jet"),
            ax=_ax,
            shrink=0.8,
        )
        _ax.set(title=_title, xlabel="x")
    _axes[0].set_ylabel("y")
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
