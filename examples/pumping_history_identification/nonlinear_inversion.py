import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - pumping history (nonlinear)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pumping history identification (2/2): nonlinear inversion

    This is the follow-up of `linear_inversion.py` (same aquifer, wells and data).

    - We now know that the unknown pumping rate cannot be **negative**: the owner of
      the well at (0, 0) extracts groundwater when needed, without any injection.
    - This prior information (constraint) can be incorporated in the inversion in
      various ways. Here, we enforce non-negativity with a log-transformation: instead
      of the pumping rate $q$, we estimate $s = \ln(q)$.

    The forward model becomes nonlinear,

    $$y = H \exp(s)$$

    where $y$ holds the $n$ drawdowns at the monitoring well and $s$ the $m$
    log-transformed pumping rates.
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

    # Local forward model (drawdown.py), located next to this notebook
    _here = str(mo.notebook_dir())
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import drawdown as dd

    return covmats, dd, logging, mo, np, plt, pypcga


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
    | Number of unknowns (log pumping rates every 0.1 min) | $m$ = 10,001 |
    | Prior kernel | $q(t, t') = 1.0^2 \exp\left(-\lvert t - t' \rvert / 100\right)$ |
    | Number of principal components | 100 |
    | Number of observations (every 10 min) | 100 |
    | Observation error variance $R$ | $0.02^2$ m² |
    | Max. Gauss-Newton iterations / `restol` | 10 / 0.01 |
    | Levenberg-Marquardt / line search | yes / yes |

    The prior parameters and the observation error were estimated with a model
    validation approach [Kitanidis, 2007]. The stationary kernel on a regular 1D grid
    allows fast FFT-based covariance matrix-vector products.
    """)
    return


@app.cell
def _(mo, np):
    data_dir = mo.notebook_dir()
    m = 10001  # number of unknowns
    t = np.linspace(0.0, 1000.0, m)  # time (min)

    s_true = np.loadtxt(data_dir / "true.txt")  # for comparison only
    # 100 noisy observations recorded every 10 min, generated as
    # obs = H @ s_true + noise
    obs = np.loadtxt(data_dir / "obs.txt")
    return m, obs, s_true, t


@app.cell
def _(plt, s_true, t):
    _fig, _ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    _ax.plot(t, s_true, "r-")
    _ax.set(
        title="True pumping history at (0, 0)",
        xlabel="time (min)",
        ylabel=r"s ($m^3$/min)",
    )
    _fig
    return


@app.cell
def _(covmats, dd, m, np, obs, t):
    prior_std = 1.0
    prior_cov_scale = np.array([100.0])

    def kernel(r):
        return prior_std**2 * np.exp(-r)

    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel, mesh_dim=t[1] - t[0], domain_shape=m, len_scale=prior_cov_scale
    )
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=100, random_state=0)

    cov_obs = covmats.CovViaDiagonal(np.full(obs.size, 0.02**2))

    # The forward model takes an ensemble (m, N_e) and returns (n_obs, N_e).
    # y = H @ exp(s): see drawdown.py
    forward_model = dd.Model({"log": True})
    return Q, cov_obs, forward_model


@app.cell
def _(Q, cov_obs, forward_model, np, obs, pcga_logger, pypcga, s_true):
    solver = pypcga.PCGA(
        s_init=np.full(s_true.size, -1.0),  # any initial guess works
        obs=obs,
        cov_obs=cov_obs,
        forward_model=forward_model,
        Q=Q,
        maxiter=10,
        restol=0.01,
        is_lm=True,
        max_it_lm=5,
        is_line_search=True,
        random_state=0,
        logger=pcga_logger,
    )
    s_hat, simul_obs, post_diagv, iter_best = solver.run()
    return post_diagv, s_hat, simul_obs, solver


@app.cell(hide_code=True)
def _(mo, solver):
    mo.md(f"""
    The log-transformation makes the problem nonlinear, which requires a few more
    iterations ({solver.istate.status}, best solution at iteration
    {solver.istate.iter_best}). But there is no more negative extraction (injection) in
    the solution, and the non-negativity constraint (a confident prior knowledge)
    improves the observation matching compared to the linear inversion.

    Model-checking statistics (should be close to 1):
    $Q_2$ = {solver.istate.Q2_best:.3f}, $c_R$ = {solver.istate.best_cR:.3f}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The transformation leads to a non-symmetric posterior distribution in the original
    (untransformed) space: the 95% credible interval, computed in log space and
    back-transformed, is narrow for low pumping rates and wide for high rates (note the
    log scale).
    """)
    return


@app.cell
def _(np, obs, plt, post_diagv, s_hat, s_true, simul_obs, solver, t):
    post_std = np.sqrt(np.maximum(post_diagv.ravel(), 0.0))
    _s = s_hat.ravel()

    _fig, _axes = plt.subplot_mosaic(
        [["hist", "hist"], ["fit", "conv"]], figsize=(10, 7.5), layout="constrained"
    )
    _ax = _axes["hist"]
    _ax.fill_between(
        t,
        np.exp(_s - 1.96 * post_std),
        np.exp(_s + 1.96 * post_std),
        color="0.8",
        label="95% credible interval",
    )
    _ax.plot(t, np.exp(_s), "k-", label="estimated")
    _ax.plot(t, s_true, "r-", label="true")
    _ax.set_yscale("log")
    _ax.set_ylim(bottom=1e-4)
    _ax.set(
        title="Estimated pumping history", xlabel="time (min)", ylabel=r"q ($m^3$/min)"
    )
    _ax.legend()

    _ax = _axes["fit"]
    _lims = [min(obs.min(), simul_obs.min()), max(obs.max(), simul_obs.max())]
    _ax.plot(_lims, _lims, "k-")
    _ax.plot(obs, simul_obs.ravel(), ".")
    _ax.set(
        title="Observed vs simulated drawdown",
        xlabel="observation (m)",
        ylabel="simulation (m)",
        aspect="equal",
    )

    _ax = _axes["conv"]
    _ax.semilogy(np.arange(len(solver.istate.obj_seq)), solver.istate.obj_seq, "o-")
    _ax.set(title="Convergence", xlabel="iteration", ylabel="objective function")
    _fig
    return


if __name__ == "__main__":
    app.run()
