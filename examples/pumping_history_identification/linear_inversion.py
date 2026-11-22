import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="pypcga - pumping history (linear)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pumping history identification (1/2): linear inversion

    *Example adapted from the Stanford CEE362G course.*

    - Flow takes place in a homogeneous and isotropic confined aquifer, and is nearly
      horizontal.
    - An extraction well is located at $(x_1, x_2) = (0, 0)$ m, but its pumping rate
      $s(t)$ is not measured.
    - Luckily, a neighboring monitoring well at $(6, 8)$ m records the drawdown caused by
      the pumping.

    **Assuming that all the aquifer properties are known, can we reconstruct the pumping
    history $s(t)$ from the drawdown time series measured from $t = 0$ to 1000 min?**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Governing equations (skip if you are only interested in pypcga)": mo.md(
                r"""
    The hydraulic head $\phi$ satisfies

    $$S \frac{\partial \phi}{\partial t} - T \left( \frac{\partial^2 \phi}{\partial x_1^2}
    + \frac{\partial^2 \phi}{\partial x_2^2} \right) = -s(t)\, \delta(x_1)\, \delta(x_2)$$

    with $t$ the time [T], $(x_1, x_2)$ the spatial coordinates [L], $S$ the storativity
    [-], $T$ the transmissivity [L²/T], $s(t)$ the extraction rate [L³/T] and $\delta$ the
    Dirac delta function. With the hydraulic diffusivity $D = T / S$, the solution for an
    unbounded domain with level $\phi_0$ away from the well (neglecting well storage) is

    $$\phi(t, x_1, x_2) = \phi_0 - \frac{1}{4 \pi T} \int_0^t \frac{1}{t - \tau}
    \exp\left(-\frac{x_1^2 + x_2^2}{4 D (t - \tau)}\right) s(\tau)\, d\tau$$

    The drawdown is $y = \phi_0 - \phi$ and, with any appropriate quadrature rule,

    $$y = H s$$

    where $y$ holds the $n$ drawdowns at the monitoring well, $s$ the $m$ extraction
    rates at the pumping well and $H$ is an $n \times m$ matrix (stored in `H.mat`, see
    `drawdown.py`).
    """
            )
        }
    )
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
    | Number of unknowns (pumping rates every 0.1 min) | $m$ = 10,001 |
    | Prior kernel | $q(t, t') = 0.04^2 \exp\left(-\lvert t - t' \rvert / 200\right)$ |
    | Number of principal components | 50 |
    | Number of observations (every 10 min) | 100 |
    | Observation error variance $R$ | $0.04^2$ m² |
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
    prior_std = 0.04
    prior_cov_scale = np.array([200.0])

    def kernel(r):
        return prior_std**2 * np.exp(-r)

    prior_cov = covmats.CovKernelAsLinopViaFFT(
        kernel, mesh_dim=t[1] - t[0], domain_shape=m, len_scale=prior_cov_scale
    )
    Q = covmats.eigen_factorize_cov_mat(prior_cov, n_pc=50, random_state=0)

    cov_obs = covmats.CovViaDiagonal(np.full(obs.size, 0.04**2))

    # The forward model takes an ensemble (m, N_e) and returns (n_obs, N_e).
    forward_model = dd.Model()
    return Q, cov_obs, forward_model


@app.cell
def _(Q, cov_obs, forward_model, np, obs, pcga_logger, pypcga, s_true):
    solver = pypcga.PCGA(
        s_init=np.full(s_true.size, np.mean(s_true)),  # any initial guess works
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
    Only two lines of code (initialization, run) are needed for the inversion. Because
    the problem is linear, convergence is reached after a couple of linearizations
    ({solver.istate.status}). PCGA also works for large-scale linear inverse problems
    with a black-box forward model.

    Model-checking statistics (should be close to 1):
    $Q_2$ = {solver.istate.Q2_best:.3f}, $c_R$ = {solver.istate.best_cR:.3f}.
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
        _s - 2.0 * post_std,
        _s + 2.0 * post_std,
        color="0.8",
        label="95% credible interval",
    )
    _ax.plot(t, _s, "k-", label="estimated")
    _ax.plot(t, s_true, "r-", label="true")
    _ax.set(
        title="Estimated pumping history", xlabel="time (min)", ylabel=r"s ($m^3$/min)"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The estimate goes below zero at times, i.e., it suggests some injection. What if we
    know that the pumping rate is non-negative (the owner of the well only extracts
    groundwater)? See the second part: `nonlinear_inversion.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
