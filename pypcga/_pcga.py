"""
Implement the Principal Component Geostatistical Approach for large-scale inversion.

The original code has been written by Jonghyun Harry Lee.

See: https://github.com/jonghyunharrylee/pyPCGA
"""

from __future__ import annotations

import copy
import logging
import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import isnan, sqrt
from time import time
from typing import Callable, List, Optional, Tuple, Union

import covmats
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator, gmres

from pypcga._utils import (
    NDArrayBool,
    NDArrayFloat,
    check_random_state,
    ensemble_dot,
    ghep,
)

VERY_LARGE_NUMBER = 1.0e20


class Residual:
    def __init__(self) -> None:
        self.res: List[NDArrayFloat] = []

    def __call__(self, rk: NDArrayFloat) -> None:
        self.res.append(rk)

    def itercount(self) -> int:
        return len(self.res)

    def clear(self) -> None:
        self.res = []


@dataclass
class InternalState:
    """Class to keep track of internal state."""

    # keep track of some values (best, init)
    s_best: NDArrayFloat
    beta_best: NDArrayFloat = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    simul_obs_best: NDArrayFloat = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    simul_obs_init: NDArrayFloat = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    # Jacobian-related products from the last completed linearization
    # (see PCGA's `is_save_jac` parameter), required to compute the
    # posterior covariance after the inversion completes.
    HX: NDArrayFloat = field(default_factory=lambda: np.array([]))
    HZ: NDArrayFloat = field(default_factory=lambda: np.array([]))
    Hs: NDArrayFloat = field(default_factory=lambda: np.array([]))
    # Approximate inverse of A used as a preconditioner to solve Ax = b with
    # the Krylov subspace approach. None until the first iterative solve.
    invA_as_linop: Optional["InvALinOp"] = None
    obj_seq: List[float] = field(default_factory=lambda: [])
    inflation_seq: List[float] = field(default_factory=lambda: [])
    Q2_seq: List[float] = field(default_factory=lambda: [])
    cR_seq: List[float] = field(default_factory=lambda: [])
    iter_best: int = 0
    status: str = "IDLE."
    is_success: bool = False
    # Jacobian-related products (HX, HZ, Hs) of the most recent linearization.
    # Promoted to HX/HZ/Hs above only when that linearization produced a new
    # best solution, so that the posterior covariance is always consistent
    # with `s_best` and `best_inflation`.
    last_jac: Optional[Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]] = field(
        default=None, repr=False
    )

    @property
    def best_obj(self) -> float:
        """
        Return the best objective function obtained in the optimization.

        The first objective function is ignored because beta is minimized.
        """
        return self.obj_seq[self.iter_best]

    def _get_best(self, seq: List[float]) -> float:
        """Return the value of `seq` at the best iteration (NaN if none)."""
        # iter_best == 0 means that no iteration improved on the initial
        # guess: there is no associated Q2/cR/inflation (seq[-1] would
        # silently return the value of an unrelated, rejected iteration).
        if self.iter_best == 0 or len(seq) < self.iter_best:
            return float("nan")
        return seq[self.iter_best - 1]

    @property
    def Q2_best(self) -> float:
        return self._get_best(self.Q2_seq)

    @property
    def best_cR(self) -> float:
        """
        Return the cR predictive model-checking statistic at the best iteration.

        See :meth:`PCGA._compute_cR`. Like `Q2_best`, should be close to 1.0
        for a well-specified covariance model.
        """
        return self._get_best(self.cR_seq)

    @property
    def best_inflation(self) -> float:
        return self._get_best(self.inflation_seq)

    @property
    def Q2_cur(self) -> float:
        return self.Q2_seq[-1]

    @property
    def cR_cur(self) -> float:
        """
        Return the cR predictive model-checking statistic at the current iteration.

        See :meth:`PCGA._compute_cR`. Like `Q2_cur`, should be close to 1.0
        for a well-specified covariance model.
        """
        return self.cR_seq[-1]


class InvALinOp(LinearOperator):
    def __init__(
        self,
        HX: NDArrayFloat,
        HZZTHT_eig_vects: NDArrayFloat,
        HZZTHT_eig_vals: NDArrayFloat,
        cov_obs: covmats.CovarianceMatrix,
        n_obs: int,
        beta_dim: int,
        HZ: Optional[NDArrayFloat] = None,
        inflation: float = 1.0,
    ) -> None:
        """Initialize the instance."""
        super().__init__(dtype=np.float64, shape=(n_obs + beta_dim, n_obs + beta_dim))
        self.n_obs: int = n_obs
        self.beta_dim: int = beta_dim
        self.HX: NDArrayFloat = HX
        # Only kept as a reference to know which linearization this
        # approximation was built from (see PCGA._get_invA_for).
        self.HZ: Optional[NDArrayFloat] = HZ
        self.HZZTHT_eig_vects: NDArrayFloat = HZZTHT_eig_vects
        self.HZZTHT_eig_vals: NDArrayFloat = HZZTHT_eig_vals
        self.cov_obs = cov_obs
        self.update_inflation_factor(inflation)

    def update_inflation_factor(self, value: float) -> None:
        """
        Update the inflation factor.

        The inflation-dependent quantities (the diagonal correction of
        Psi^{-1} and the inverse of the small (p x p) Schur complement S) are
        precomputed here once, instead of at every matrix-vector product.
        """
        self.inflation: float = float(value)
        self._dvec = self._get_dvec(self.inflation)
        S = self.HX.T @ self.inv_psi(self.HX, self.inflation)
        # p is usually between 1 and 3 so inverting S directly is fine.
        self._invS = np.linalg.inv(np.atleast_2d(S))

    def with_inflation(self, value: float) -> "InvALinOp":
        """
        Return a shallow copy of the operator for another inflation factor.

        The (potentially large) eigen-decomposition arrays are shared, only
        the small inflation-dependent caches are recomputed. Unlike
        :meth:`update_inflation_factor`, the current instance is not
        mutated, which makes it safe to use from several threads.
        """
        if float(value) == self.inflation:
            return self
        new = copy.copy(self)
        new.update_inflation_factor(value)
        return new

    def _get_dvec(self, inflation: float) -> NDArrayFloat:
        # lambda / (lambda + alpha) == (lambda / alpha) / (lambda / alpha + 1)
        lam = self.HZZTHT_eig_vals.ravel()
        return lam / (lam + inflation)

    def inv_psi(self, v: NDArrayFloat, inflation: float) -> NDArrayFloat:
        """Apply (HQH^T + inflation * R)^{-1} (low rank approximation) to v."""
        if inflation == getattr(self, "inflation", None):
            dvec = self._dvec
        else:
            dvec = self._get_dvec(inflation)
        UTv = self.HZZTHT_eig_vects.T @ v
        if UTv.ndim == 1:
            DUTv = dvec * UTv
        else:
            DUTv = dvec[:, np.newaxis] * UTv
        return (1.0 / inflation) * (
            self.cov_obs.solve(v) - self.HZZTHT_eig_vects @ DUTv
        )

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        # See Eq. 14 in leeScalableSubsurfaceInverse2016.
        # Warning, there is typo for the block 22 of A^{-1}.
        # i.e., it is -S^{-1} instead of -S.
        # With y2 = S^{-1} (HX^T Psi^{-1} x1 - x2), the product reads
        #   [ Psi^{-1} x1 - Psi^{-1} HX y2 ]
        #   [ y2                           ]
        # which only requires two applications of Psi^{-1} (instead of three).
        x1 = x[: self.n_obs]
        x2 = x[self.n_obs :]
        invPsi_x1 = self.inv_psi(x1, self.inflation)
        y2 = self._invS @ (self.HX.T @ invPsi_x1 - x2)
        y1 = invPsi_x1 - self.inv_psi(self.HX @ y2, self.inflation)
        return np.concatenate((y1, y2), axis=0)

    def _matmat(self, X: NDArrayFloat) -> NDArrayFloat:
        # _matvec supports matrix multiplication.
        return self._matvec(X)

    def _rmatvec(self, x: NDArrayFloat) -> NDArrayFloat:
        return self._matvec(x)

    def _rmatmat(self, X: NDArrayFloat) -> NDArrayFloat:
        return self._rmatvec(X)

    def get_invPsi(self, inflation) -> NDArrayFloat:
        return self.inv_psi(np.identity(self.n_obs), inflation)


class PCGA:
    """
    Solve inverse problem with PCGA (approx to quasi-linear method)

    Every values are represented as 2D np.array.
    """

    __slots__: List[str] = [
        "s_init",
        "obs",
        "_cov_obs",
        "forward_model",
        "Q",
        "callback",
        "is_line_search",
        "is_lm",
        "is_direct_solve",
        "is_objfun_exact",
        "max_it_lm",
        "alphamax_lm",
        "lm_smin",
        "lm_smax",
        "max_it_ls",
        "maxiter",
        "ftol",
        "restol",
        "ftarget",
        "eps",
        "max_workers",
        "istate",
        "random_state",
        "_prior_s_var",
        "post_diagv",
        "drift",
        "is_save_jac",
        "cov_obs_inflation_factors",
        "logger",
    ]

    def __init__(
        self,
        *,
        s_init: NDArrayFloat,
        obs: NDArrayFloat,
        cov_obs: covmats.CovarianceMatrix,
        forward_model: Callable,
        Q: covmats.CovViaEigenFactorization,
        drift: Optional[covmats.DriftMatrix] = None,
        prior_s_var: Optional[Union[float, NDArrayFloat]] = None,
        callback: Optional[Callable] = None,
        is_line_search: bool = False,
        is_lm: bool = False,
        is_direct_solve: bool = False,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None,
        is_objfun_exact: bool = False,  # former objeval
        max_it_lm: int = multiprocessing.cpu_count(),
        alphamax_lm: float = 10.0**3.0,  # does it sound ok?
        lm_smin: Optional[float] = None,
        lm_smax: Optional[float] = None,
        max_it_ls: int = 20,
        max_workers_lm: int = multiprocessing.cpu_count(),
        maxiter: int = 10,
        ftol: float = 1e-5,
        ftarget: Optional[float] = None,
        restol: float = 1e-2,
        logger: Optional[logging.Logger] = None,
        is_save_jac: bool = True,
        eps: float = 1.0e-8,
    ) -> None:
        r"""
        Initialize the instance.

        Parameters
        ----------
        s_init : NDArrayFloat
            1D array of initial control parameters, i.e., initial solution for
            Gauss-Newton method. In theory, the choice of
            s_init does not affect the estimation while total number of
            iterations/number of forward model runs depend on `s_init`. Expected
            shape is (:math:`N_{s}`) or (:math:`N_{s}, 1`).
        obs : numpy.ndarray, optional
            1D array of (noisy) measurements used for inversion.
        cov_obs : covmats.CovarianceMatrix
            Covariance matrix of observed data measurement errors with dimensions
            (:math:`N_{\mathrm{obs}}`, :math:`N_{\mathrm{obs}}`).
            Also denoted :math:`R` in the literature.
            All covariance representations provided by :py:mod:`covmats` are supported.
        forward_model : Callable
            Wrapper for forward model obs = f(s). See a template python file in each
            example for more information. Return shape (:math:`N_{\mathrm{obs}}`,
            :math:`N_{e}`).
        Q : covmats.CovViaEigenFactorization
            Eigen factorization of the Covariance matrix of the inverted parameters
            with shape (:math:`N_{s}`, :math:`N_{s}`).
        drift : Optional[DriftMatrix], optional
            Drift (prior mean trend) matrix ``X`` such that the prior mean of
            ``s`` is modeled as ``X @ beta`` for some unknown drift
            coefficients ``beta``. If None, a constant drift
            (``covmats.ConstantDriftMatrix``) is used, i.e. the prior mean is
            an unknown constant. By default None.
        prior_s_var : Optional[Union[float, NDArrayFloat]], optional
            A priori variance of the control variables (the diagonal of the
            prior covariance, used e.g. to compute the posterior variance).
            Can be a single float (applied to all parameters) or a 1D array
            with one value per parameter. If None, it is taken from
            ``Q.get_diagonal()``. By default None.
        callback : Optional[Callable], optional
            Optional user callback invoked as
            ``callback(self, s_hat=..., simul_obs=..., n_iter=...)`` once
            after the initial state and again at the end of every
            Gauss-Newton iteration. Useful to save/plot intermediate solver
            states. By default None.
        is_line_search : bool, optional
            Whether to use line search (add ref) if the Gauss-Newton iteration fails
            to lower the cost function value. It comes at the cost of
            extra forward model runs (not parallelized). By default False.
        is_lm : bool, optional
            Whether to use Levenberg Marquard inner iterations in each Gauss-Newton
            iterations. It consists in inflating the covariance matrix of observed data
            measurement errors (`cov_obs`), by adding weights on its diagonal.
            It acts as a regularization and can help to make the objective function
            more convex. This comes at the price of extra system inversions and
            `max_it_lm` forward model runs. The runs can be performed in parallel
            since it relies on `forward_model`. But it is the user's responsibility
            to parallelize  `forward_model`. By default False.
        is_direct_solve : bool, optional
            Whether to solve the saddle point system (Ax = b), see eq 4.53 in
            (add ref), with:

            - the direct approach (Cholesky factorization of A), see eq (4.57) in
              :cite:`colletAssistedHistoryMatching2024`,
            - or the alternative iterative Krylov subspace approach as described
              in :cite:`saibabaEfficientMethodsLargescale2012,
              saibabaFastAlgorithmsGeostatistical2013`.

            Using direct solve is practical if the number of observations is
            around 100 or less. Beyond, it is advised to rely on the iterative
            Krylov subspace approach. The default is False (iterative Krylov
            subspace).
        random_state : Optional[ Union[int, np.random.Generator, np.random.RandomState]]
            Pseudorandom number generator state used to generate resamples.
            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.
        is_objfun_exact : bool, optional
            Reserved flag (formerly ``objeval`` in the original MATLAB/Python
            pyPCGA implementation). Currently stored on the instance but not
            yet consumed anywhere else in this implementation -- kept for API
            compatibility while the feature is ported. By default False.
        max_it_lm: int
            Maximum number iterations when using Levenberg Marquard regularization.
            Only applies if `is_lm` is True. By default use all available CPUs.
        alphamax_lm : float, optional
            Maximum inflation weight used for the observation-error covariance
            during Levenberg-Marquardt regularization. The weights used across
            the `max_it_lm` inner iterations are logarithmically spaced between
            1.0 and `alphamax_lm` (see :meth:`get_cov_obs_inflation_factors`,
            ``10 ** linspace(0, log10(alphamax_lm), max_it_lm)``).
            By default 10.0**3.
        lm_smax : Optional[float], optional
            Maximum LM solution, by default None
        max_it_ls : int, optional
            Number of candidate points evaluated (in a single call to
            `forward_model`) by the line search, by default 20.
        max_workers_lm : int, optional
            Maximum number of threads used to solve the `max_it_lm` LM
            linear systems concurrently. Only applies if `is_lm` is True.
            Threads are used rather than processes: the work is dominated by
            BLAS/LAPACK calls which release the GIL, and nothing (in
            particular `forward_model`) needs to be picklable.
            By default use all available CPUs.
        maxiter : int, optional
            Maximum Gauss-Newton iterations, by default 10.
        ftarget: Optional[Union[float, Callable]] = None, optional
            Target objective function (stop criterion) .
            The iteration stops when ``f^{k+1} <= fmin``. If None, the stop criterion
            is ignored. The default is None.
        ftol : float, optional
            Objective function minimum change (stop criterion). The iteration stops
            when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
            Typical values for `ftol` on a computer with 15 digits of accuracy in double
            precision are as follows: `ftol` = 5e-3 for low accuracy; `ftol` = 5e-8
            for moderate accuracy; `ftol` = 5e-14 for extremely high accuracy.
            If `ftol` = 0, the test will stop the algorithm only if the objective
            function remains unchanged after one iteration. The default is 1e-5.
        restol : float, optional
            Mininmum change in the update value. Below this threshold, changes are
            considered unisgnificant and inversion is stopped.
            The change is computed as the euclidean norm of the difference between the
            current updated values and the values at the previous iteration scaled by
            the euclidean norm of the values at the previous iteration:

            .. math::

                \mathrm{change} = \dfrac{\left\lVert \mathbf{s}_{\ell+1} -
                \mathbf{s}_{\ell} \right\rVert}{\left\lVert
                \mathbf{s}_{\ell} \right\rVert}

            by default 1e-2.
        logger: Optional[Logger], optional
            Logger instance. If no logger is passed, there will be no output.
            By default None.
        is_save_jac : bool, optional
            Whether to keep the last computed Jacobian-related products
            (`self.istate.HX`, `self.istate.HZ`, `self.istate.Hs`) after each
            Gauss-Newton iteration. These are required by
            :meth:`get_dense_post_cov` and :meth:`get_eigen_post_cov` to
            compute the posterior covariance after the inversion completes,
            so this should stay True unless memory is a concern and the
            posterior covariance will not be needed. By default True.
        eps : float, optional
            PCGA perturbation scalar (see eq. in ...), By default 1.0e-8.
        """
        ##### Forward Model
        # Make sure the array has a second dimension of length 1.
        self.s_init = np.array(s_init).reshape(-1, 1)
        # Observations
        self.obs = np.array(obs).ravel()  # 1D vector
        self.cov_obs = cov_obs
        # forward solver setting should be done externally as a blackbox
        # including the parallelization
        self.forward_model = forward_model
        self.Q: covmats.CovViaEigenFactorization = Q
        self.callback: Optional[Callable] = callback
        self.is_line_search: bool = is_line_search
        self.is_lm: bool = is_lm
        self.is_direct_solve: bool = is_direct_solve

        # Define Drift (or Prior) functions
        # NOTE: this must be set before the direct-solve warning below, which
        # relies on self.drift.beta_dim.
        if drift is not None:
            assert drift.s_dim == self.s_dim
            self.drift: covmats.DriftMatrix = drift
        else:
            self.drift = covmats.ConstantDriftMatrix(self.s_dim)

        # Add a warning depending on the number of observations
        if self.is_direct_solve and self.n_obs > 100:
            mat_shape = self.n_obs + self.drift.beta_dim
            warnings.warn(
                "You have chosen to solve the saddle point system (Ax = b) with the "
                "direct approach (Cholesky factorization of A) while A will be of shape"
                f" ({mat_shape}, {mat_shape}).\n"
                f"This is because n_obs = {self.n_obs} and "
                f"n_beta = {self.drift.beta_dim}\n"
                "This is practical up to A of shape (100, 100). "
                "If you are way above this dimension"
                " consider using iterative Krylov subspace approahces by setting "
                "is_direct_solve to False!"
            )

        self.is_objfun_exact: bool = is_objfun_exact
        self.max_it_lm = max_it_lm
        self.alphamax_lm: float = alphamax_lm
        self.lm_smin: Optional[float] = lm_smin
        self.lm_smax: Optional[float] = lm_smax
        self.max_it_ls: int = max_it_ls
        self.maxiter: int = maxiter
        self.ftol: float = ftol
        self.restol: float = restol
        self.ftarget: Optional[float] = ftarget

        # PCGA parameters (purturbation size)
        self.eps: float = eps

        # NOTE: max_workers is derived from is_lm/max_it_lm/max_workers_lm rather
        # than being independently configurable. Multi-processing is only
        # worthwhile when the number of observations (and thus the per-worker
        # forward model cost) is large enough to outweigh the process-startup
        # overhead; for small problems it is not interesting.
        # There is no point in having more workers than LM candidates, and the
        # user-provided `max_workers_lm` is an upper bound (it used to be
        # max(max_workers_lm, max_it_lm), which ignored the user limit).
        self.max_workers = max(1, min(max_workers_lm, max_it_lm)) if is_lm else 1

        # keep track of the internal state
        self.istate = InternalState(s_best=self.s_init)
        # Random state for v0 vector used by eigsh and svds
        self.random_state = check_random_state(random_state)

        if prior_s_var is not None:
            self.prior_s_var = prior_s_var
        else:
            self.prior_s_var = self.Q.get_diagonal()

        # Initialized as the diagonal of the covariance matrix
        self.post_diagv = self.prior_s_var

        # Internal state
        self.is_save_jac = is_save_jac

        self.cov_obs_inflation_factors = self.get_cov_obs_inflation_factors()
        self.logger: Optional[logging.Logger] = logger

        # NOTE: HX, HZ, Hs, invA_as_linop and simul_obs_init are tracked on
        # `self.istate` (InternalState) rather than as separate slots here,
        # since they are part of the mutable solver-run state.

        ##### Optimization
        self.display_init_parameters()

    @property
    def n_internal_loops(self) -> int:
        """
        Return the number of internal optimization loops.

        This is only relevant when using Levenberg Marquard, otherwise, returns 1.
        """
        if self.is_lm:
            return self.max_it_lm
        return 1

    def loginfo(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)

    def display_init_parameters(self) -> None:
        self.loginfo("##### PCGA Inversion #####")
        self.loginfo("##### 1. Initialize forward and inversion parameters")
        self.loginfo("------------ Inversion Parameters -------------------------")
        _dict = {
            "Number of unknowns": self.s_dim,
            "Number of observations": self.d_dim,
            "Number of principal components (n_pc)": self.Q.n_pc,
            "Maximum Gauss-Newton iterations": self.maxiter,
            "Machine eps (delta = sqrt(eps))": self.eps,
            "Minimum model change (restol)": f"{self.restol:.3e}",
            "Minimum obj fun change (ftol)": f"{self.ftol:.3e}",
            "Target obj fun (ftarget)": (
                f"{self.ftarget:.3e}" if self.ftarget is not None else None
            ),
            "Levenberg-Marquardt (is_lm)": self.is_lm,
        }
        if self.is_lm:
            _dict["Minimum LM solution (lm_smin)"] = self.lm_smin
            _dict["Maximum LM solution (lm_smax)"] = self.lm_smax
            _dict["Maximum LM iterations (max_it_lm)"] = self.max_it_lm

        _dict["Line search"] = self.is_line_search

        if self.is_line_search:
            _dict["Maximum line-search iterations (max_it_ls)"] = self.max_it_ls

        # dipslay the dict content
        # first get the max length
        max_length: int = int(np.max([len(_str) for _str in _dict.keys()]))
        for k, v in _dict.items():
            self.loginfo(f"  {k: <{max_length}} : {v}")

        self.loginfo("-----------------------------------------------------------")

    @property
    def s_dim(self) -> int:
        """Return the length of the parameters vector."""
        return self.s_init.size

    @property
    def n_obs(self) -> int:
        """Return the number of observations/forecast data."""
        return self.obs.size

    @property
    def d_dim(self) -> int:
        """Return the number of forecast data. Alias for n_obs"""
        return self.n_obs

    @property
    def cov_obs(self) -> covmats.CovarianceMatrix:
        """Get the observation errors covariance matrix."""
        return self._cov_obs

    @cov_obs.setter
    def cov_obs(self, value: covmats.CovarianceMatrix) -> None:
        """
        Set the observation errors covariance matrix.

        It must be a 2D array, or a 1D array if the covariance matrix is diagonal.
        """
        if not isinstance(value, covmats.CovarianceMatrix):
            raise ValueError(
                "`cov_obs` must be an implementation of `covmats.CovarianceMatrix`"
            )
        self._cov_obs = value

    @property
    def prior_s_var(self) -> NDArrayFloat:
        """Get the a priori variance of the control variables."""
        return self._prior_s_var

    @prior_s_var.setter
    def prior_s_var(self, values: Union[float, NDArrayFloat]) -> None:
        """Set the a priori variance of the control variables."""
        _values = np.asarray(values)
        if _values.size == 1:
            self._prior_s_var: NDArrayFloat = np.ones(self.s_dim) * _values.ravel()[0]
        elif _values.size == self.s_dim:
            self._prior_s_var = _values.ravel()
        else:
            raise ValueError(
                "prior_s_var must be either a float value, either a 1D "
                "array with the same number of elements as in s_init!"
            )

    def get_v0(self, size) -> Optional[NDArrayFloat]:
        if self.random_state is not None:
            return self.random_state.uniform(size=(size,))
        else:
            return None

    def get_cov_obs_inflation_factors(self) -> NDArrayFloat:
        """
        Inflation factors used in each internal loop.

        It is a sequence of `max_it_lm` values, logarithmically spaced between
        1.0 and `alphamax_lm`, only if Levenberg-Marquardt is on (`is_lm`).
        Otherwise it is a single value of 1.0 (no inflation).

        Returns
        -------
        NDArrayFloat
            1D array of inflation factors, one per internal (LM) loop. Has
            length `max_it_lm` if `is_lm` is True, otherwise length 1
            containing only ``[1.0]``.
        """
        if self.is_lm:
            return 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), self.max_it_lm))
        return np.array([1.0])

    def _perturb_ensemble(
        self,
        x: NDArrayFloat,
        s: NDArrayFloat,
        eps: float,
        delta: Optional[float] = None,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """
        Build the ensemble of perturbed points used for finite-difference Jacobians.

        For each column (direction) ``x[:, i]``, compute a perturbation step
        size ``deltas[i]`` and overwrite ``x[:, i]`` in place with the
        perturbed point ``s + deltas[i] * x[:, i]``.

        If `delta` is None, NaN, or 0, the step size is chosen per-direction
        following the adaptive heuristic of Brown and Saad [1990]; a floor of
        ``sqrt(eps)`` is used whenever the heuristic would otherwise yield
        exactly zero (which happens when `s` and/or the perturbation
        direction are zero). Otherwise, the fixed step size `delta` is used
        for every direction.

        Parameters
        ----------
        x : NDArrayFloat
            Ensemble of perturbation directions, shape (m, nruns). Modified
            in place: on return, each column holds the perturbed point.
        s : NDArrayFloat
            Current point around which to perturb, shape (m, 1).
        eps : float
            Relative perturbation size used by the Brown & Saad heuristic.
        delta : Optional[float], optional
            Fixed step size to use for every direction instead of the
            adaptive heuristic. By default None (use the adaptive heuristic).

        Returns
        -------
        Tuple[NDArrayFloat, NDArrayFloat]
            x : the perturbed ensemble, shape (m, nruns) (same array object
                as the input `x`, returned for convenience).
            deltas : the per-direction step sizes used, shape (nruns, 1).
        """
        nruns = np.size(x, 1)
        deltas = np.zeros((nruns, 1), dtype="d")

        if delta is None or isnan(delta) or delta == 0:
            # Brown and Saad [1990] adaptive step size, vectorized over
            # columns instead of looping in Python.
            mag = s.T @ x  # (1, nruns)
            absmag = np.abs(s).T @ np.abs(x)  # (1, nruns)
            signmag = np.where(mag >= 0, 1.0, -1.0)  # (1, nruns)
            norms = np.linalg.norm(x, axis=0, keepdims=True)  # (1, nruns)

            deltas[:, 0] = (
                signmag
                * sqrt(eps)
                * np.maximum(np.abs(mag), absmag)
                / ((norms + np.finfo(float).eps) ** 2)
            ).ravel()

            # s = 0 or x[:, i] = 0 -> the heuristic yields exactly zero;
            # fall back to sqrt(eps) for those directions only.
            for i in np.flatnonzero(deltas[:, 0] == 0):
                self.loginfo(
                    "%d-th delta: signmag %g, eps %g, max abs %g, norm %g"
                    % (
                        i,
                        signmag[0, i],
                        eps,
                        max(abs(mag[0, i]), absmag[0, i]),
                        norms[0, i] ** 2,
                    )
                )
                deltas[i, 0] = sqrt(eps)
                self.loginfo(
                    f"{i}-th delta: assigned as sqrt(eps) - {deltas[i, 0]:.2e}",
                )
                # raise ValueError('delta is zero? - plz check your
                # s_init is within a reasonable range')
        else:
            deltas[:, 0] = delta

        # reuse storage x by updating x, in one vectorized assignment
        x[:, :] = s + deltas.T * x

        return x, deltas

    def jac_vect(self, x, s, simul_obs, eps, delta=None):
        """
        Jacobian times Matrix (Vectors) in Parallel
        perturbation interval delta determined following Brown and Saad [1990]
        """
        nruns = np.size(x, 1)

        x, deltas = self._perturb_ensemble(x, s, eps, delta)

        simul_obs_purturbation = self.forward_model(x)

        if np.size(simul_obs_purturbation, 1) != nruns:
            raise ValueError(
                "size of simul_obs_purturbation (%d,%d) is not nruns %d"
                % (
                    simul_obs_purturbation.shape[0],
                    simul_obs_purturbation.shape[1],
                    nruns,
                )
            )

        Jxs = np.true_divide(simul_obs_purturbation - simul_obs, deltas.T)

        return Jxs

    def objective_function_ls(self, simul_obs) -> NDArrayFloat:
        r"""
        Compute the data-misfit (least-squares) part of the objective function.

        Computed for an ensemble of realizations at once (vectorized via
        :func:`ensemble_dot`), as
        :math:`0.5 (y - h(s))^T R^{-1} (y - h(s))`.

        Parameters
        ----------
        simul_obs : NDArrayFloat
            Simulated observations, shape (n_obs, Ne).

        Returns
        -------
        NDArrayFloat
            Data-misfit value for each realization, shape (Ne,).
        """
        ymhs = simul_obs.T - self.obs
        return 0.5 * ensemble_dot(ymhs.T, self.cov_obs.solve(ymhs.T))

    def objective_function_reg(
        self, s_cur: NDArrayFloat, beta_cur: NDArrayFloat
    ) -> NDArrayFloat:
        """
        s_cur with shape (N_s, Ne)
        beta_cur with shape (N_b, Ne)
        0.5(s-Xb)^TC^{-1}(s-Xb)

        return size = Ne
        """
        smxb = s_cur - np.dot(self.drift.mat, beta_cur)
        return 0.5 * ensemble_dot(smxb, self.Q.solve(smxb))

    def objective_function(self, s_cur, beta_cur, simul_obs) -> NDArrayFloat:
        """
        0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)

        Return size = Ne
        """
        return self.objective_function_ls(simul_obs) + self.objective_function_reg(
            s_cur, beta_cur
        )

    def objective_function_no_beta_new(self, s_cur, simul_obs) -> float:
        """
        marginalized objective w.r.t. beta
        0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)

        Return size = Ne
        Note: this is an alternative way, more expensive.
        """
        X = self.drift.mat

        def fun(beta: NDArrayFloat) -> float:
            """-X^TC^{-1}(s-Xb)"""
            smxb = (s_cur - np.dot(X, beta.reshape(-1, 1))).ravel()
            return float(0.5 * smxb.T.dot(self.Q.solve(smxb)).item())

        # We solve with a newton to find the optimal beta
        def jac_wrt_beta(beta: NDArrayFloat) -> NDArrayFloat:
            """-X^TC^{-1}(s-Xb)"""
            smxb = (s_cur - np.dot(X, beta.reshape(-1, 1))).ravel()
            return (-X.T.dot(self.Q.solve(smxb))).ravel()

        hess = X.T.dot(self.Q.solve(X))

        def hess_wrt_beta(beta: NDArrayFloat) -> NDArrayFloat:
            """X^TC^{-1}X"""
            return hess

        res = sp.optimize.minimize(
            x0=sp.linalg.lstsq(X, s_cur)[0].ravel(),
            fun=fun,
            method="trust-exact",
            jac=jac_wrt_beta,
            hess=hess_wrt_beta,
        )
        self.loginfo(f"new reg part = {res.fun}")
        return self.objective_function_ls(simul_obs) + res.fun

    def objective_function_no_beta(self, s_cur, simul_obs) -> float:
        """
        marginalized objective w.r.t. beta
        0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)

        return size = Ne
        """
        X = self.drift.mat

        invZs = np.multiply(
            1.0 / np.sqrt(self.Q.eig_vals), np.dot(self.Q.eig_vects.T, s_cur)
        )
        invZX = np.multiply(
            1.0 / np.sqrt(self.Q.eig_vals), np.dot(self.Q.eig_vects.T, X)
        )
        XTinvQs = np.dot(invZX.T, invZs)
        XTinvQX = np.dot(invZX.T, invZX)
        tmp = np.linalg.solve(
            XTinvQX, XTinvQs
        )  # inexpensive solve p by p where p <= 3, usually p = 1 (scalar division)
        return float(
            (
                self.objective_function_ls(simul_obs)
                + 0.5 * (np.dot(invZs.T, invZs) - np.dot(XTinvQs.T, tmp))
            ).item()
        )

    def rmse(self, residuals: NDArrayFloat, is_normalized: bool) -> NDArrayFloat:
        """Return the root mean square error.

        residuals size = N_s x N_e

        return size = Ne
        """
        if is_normalized:
            # ensemble_dot -> one value per column (the full Gram matrix used
            # to be returned for Ne > 1).
            return np.sqrt(
                ensemble_dot(residuals, self.cov_obs.solve(residuals)) / self.d_dim
            )
        return np.linalg.norm(residuals, axis=0) / np.sqrt(self.d_dim)

    def _get_projected_model_singular_values(
        self, HZ: NDArrayFloat, U_data: NDArrayFloat
    ) -> NDArrayFloat:
        r"""
        Singular values of the model sensitivity HZ, projected orthogonal to
        the drift sensitivity HX.

        These singular values are the building block for the cR
        model-checking statistic (see :meth:`_compute_cR`): they are the
        (approximate) square roots of the nonzero eigenvalues of
        :math:`P (H Q H^T) P`, where :math:`P = I - U_{\mathrm{data}}
        U_{\mathrm{data}}^T` projects onto the orthogonal complement of the
        column space of `HX` (i.e. removes the directions explained by the
        unknown drift coefficients `beta`), and :math:`H Q H^T \approx (HZ)
        (HZ)^T` is the low-rank approximation of the simulated-data
        covariance implied by the `n_pc`-truncated prior.

        Parameters
        ----------
        HZ : NDArrayFloat
            Jacobian-vector products with the prior sqrt-covariance factor
            `Z`, shape (n_obs, n_pc).
        U_data : NDArrayFloat
            Orthonormal basis for the column space of `HX` (the "pre-
            posterior data space"), shape (n_obs, p).

        Returns
        -------
        NDArrayFloat
            The largest ``min(n_pc, n_obs - p)`` singular values of ``P @
            HZ``, in decreasing order.
        """
        p_hz = HZ - U_data @ (U_data.T @ HZ)
        sigma = np.linalg.svd(p_hz, compute_uv=False)
        n_keep = min(self.Q.n_pc, self.d_dim - self.drift.beta_dim)
        return sigma[:n_keep]

    def _compute_cR(
        self,
        Q2_all: NDArrayFloat,
        sigma_cR: NDArrayFloat,
        inflations: NDArrayFloat,
    ) -> NDArrayFloat:
        r"""
        Compute the cR predictive model-checking statistic.

        Follows Kitanidis, P.K. (1991), "Orthonormal residuals in
        geostatistics: Model criticism and parameter estimation",
        Mathematical Geology, 23(5), 741-758 (see also
        :cite:`kitanidisIntroductionGeostatisticsApplications1997`, of which
        the Q2 statistic already computed elsewhere in this module is
        Eq. 6.67). Like Q2, cR should be close to 1.0 for a well-specified
        covariance model. While Q2 is an average of the squared orthonormal
        residuals, cR additionally weighs in the *spread* of the data
        covariance's eigenvalues: it is the product of Q2 with the geometric
        mean of the eigenvalues of :math:`\Xi = H Q H^T + \mathrm{inflation}
        \cdot R`, restricted (projected) to the ``n_obs - p``-dimensional
        space orthogonal to the drift sensitivity `HX` (since `p` degrees of
        freedom are used up estimating the unknown drift coefficients
        `beta`):

        .. math::

            cR = Q2 \cdot \exp\left(
                \frac{1}{n_{\mathrm{obs}} - p}
                \sum_i \log(\lambda_i)
            \right)

        where :math:`\lambda_i` are the eigenvalues of the projected
        :math:`\Xi`. Since :math:`H Q H^T` is only rank `n_pc` (the number of
        principal components retained from the prior), only the first
        ``min(n_pc, n_obs - p)`` of these eigenvalues get a contribution from
        the model term (``sigma_cR ** 2``, see
        :meth:`_get_projected_model_singular_values`); the remaining
        ``n_obs - p - n_pc`` eigenvalues are pure measurement noise,
        ``inflation * R``.

        Note
        ----
        Unlike Q2, this raw formula is **not invariant to the physical units**
        of `obs`/`R`/`Q`: the eigenvalues :math:`\lambda_i` carry variance
        units, so rescaling the problem (e.g. Pa to kPa) rescales cR by the
        square of that factor even for a perfectly well-specified model.
        Since `dof = n_obs - p` is normally much larger than `n_pc`, most of
        the `dof` eigenvalues averaged above are just ``inflation * R``, so
        in that common regime ``cR ~ Q2 * inflation * mean(R)`` -- i.e. cR's
        magnitude mostly just reflects the numeric scale of `R`, not model
        fit. To keep the "close to 1.0" interpretation meaningful regardless
        of units, this implementation reports ``cR / mean(R)`` rather than
        the raw literal formula above. This only rescales every candidate's
        cR by the same constant, so it preserves their relative ordering (a
        larger raw cR stays larger); what it changes -- and the reason for
        doing it -- is that "how close to 1.0" specifically is not
        meaningful under the raw, unit-dependent formula, since a change of
        units alone can move cR arbitrarily far from 1 even for a correct
        model. Dividing by `mean(R)` removes that unit-dependence.

        This also assumes (or approximates, if not) that `R` is homoscedastic
        (constant measurement-error variance): the noise-only eigenvalues
        use the mean of `R`'s diagonal as a representative value, since
        diagonalizing `R` exactly in the same rotated (projected) basis as
        the model term would require an additional, more expensive
        eigendecomposition. This matches Kitanidis's original derivation
        exactly when `R` is homoscedastic, and is an approximation otherwise.

        Parameters
        ----------
        Q2_all : NDArrayFloat
            Q2 statistic for each candidate inflation factor, shape (k,).
        sigma_cR : NDArrayFloat
            Singular values from :meth:`_get_projected_model_singular_values`,
            shape (min(n_pc, n_obs - p),).
        inflations : NDArrayFloat
            Observation-error covariance inflation factor for each candidate,
            shape (k,).

        Returns
        -------
        NDArrayFloat
            The (unit-normalized) cR statistic for each candidate, shape (k,).
        """
        n = self.d_dim
        p = self.drift.beta_dim
        dof = n - p

        r_diag = self.cov_obs.get_diagonal()
        r_mean = float(np.mean(r_diag))
        if not np.allclose(r_diag, r_diag[0]):
            self.loginfo(
                "cR: observation-error covariance is not homoscedastic; "
                "using the mean measurement-error variance as a "
                "representative eigenvalue for the noise-only directions "
                "(approximation)."
            )

        n_pc_eff = sigma_cR.size

        # eigenvalues of the projected Xi = HQH^T + inflation*R for each
        # candidate inflation factor, shape (k, dof)
        eigvals_reduced = np.outer(inflations, np.full(dof, r_mean))
        eigvals_reduced[:, :n_pc_eff] += sigma_cR**2

        # temporary numerical safety floor (mirrors the original
        # implementation): a near-singular Xi should not blow up log().
        eigvals_reduced[eigvals_reduced <= 0] = 1.0e-16

        cR_raw = Q2_all * np.exp(np.mean(np.log(eigvals_reduced), axis=1))

        # Normalize by the mean observation-error variance so cR is
        # dimensionless and comparable to 1.0 regardless of the physical
        # units of obs/R/Q (see Note above).
        return cR_raw / r_mean

    def jac_mat(
        self, s_cur: NDArrayFloat, simul_obs: NDArrayFloat, Z: NDArrayFloat
    ) -> Tuple[
        NDArrayFloat,
        NDArrayFloat,
        NDArrayFloat,
        NDArrayFloat,
    ]:
        m: int = self.s_dim
        p: int = self.drift.beta_dim
        n_pc: int = self.Q.n_pc
        eps: float = self.eps

        temp = np.zeros((m, p + n_pc + 1), dtype="d")  # [HX, HZ, Hs]

        temp[:, 0:p] = np.copy(self.drift.mat)
        temp[:, p : p + n_pc] = np.copy(Z)
        temp[:, p + n_pc : p + n_pc + 1] = np.copy(s_cur)

        Htemp = self.jac_vect(temp, s_cur, simul_obs, eps)

        HX = Htemp[:, 0:p]
        HZ = Htemp[:, p : p + n_pc]
        Hs = Htemp[:, p + n_pc : p + n_pc + 1]

        # Keep track of the last linearization. It is promoted to
        # istate.HX/HZ/Hs by `gauss_newton` only if it yields a new best
        # solution (see InternalState.last_jac).
        self.istate.last_jac = (HX, HZ, Hs)

        # compute the pre-posterior data space (orthonormal basis of span(HX))
        if p == 1:
            U_data = HX / np.linalg.norm(HX)
        elif p > 1:
            # A thin QR is cheaper than an SVD and gives the same column space
            U_data = np.linalg.qr(HX, mode="reduced")[0]
        else:  # point prior
            raise NotImplementedError
        return HX, HZ, Hs, U_data

    def is_s_violate_lm_bounds(self, S: NDArrayFloat) -> NDArrayBool:
        # check prescribed solution range for LM evaluations
        out: NDArrayBool = np.zeros(self.n_internal_loops, dtype=bool)
        if self.is_lm is False:
            return out
        if self.lm_smin is not None:
            out = S.min(axis=0) <= self.lm_smin
        if self.lm_smax is not None:
            out = np.logical_or(out, S.max(axis=0) >= self.lm_smax)
        return out

    def linear_iteration(
        self,
        s_cur: NDArrayFloat,
        simul_obs: NDArrayFloat,
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, float, float, float, float]:
        """
        Solve the geostatistical system using a direct solver.
        Not to be used unless the number of measurements are small O(100)
        """
        m = self.s_dim
        n = self.d_dim
        p = self.drift.beta_dim
        # n_pc = self.Q.n_pc

        Z = np.sqrt(self.Q.eig_vals).T * self.Q.eig_vects
        # Compute Jacobian-Matrix products
        start1: float = time()
        HX, HZ, Hs, U_data = self.jac_mat(s_cur, simul_obs, Z)

        # Compute eig(P*HQHT*P) approximately by svd(P*HZ). This only depends
        # on the current linearization (HZ, U_data), not on the observation-
        # error inflation factor, so it is computed once and reused below for
        # every LM candidate's cR statistic (see _compute_cR).
        sigma_cR = self._get_projected_model_singular_values(HZ, U_data)

        start2 = time()

        self.loginfo(
            f"computed Jacobian-Matrix products in : {(start2 - start1):.3e} s"
        )

        b = np.zeros((n + p, 1), dtype="d")
        # Ax = b, b = obs - h(s) + Hs
        b[:n] = self.obs.reshape(-1, 1) - simul_obs + Hs[:]

        # LM parameters
        xi_all = np.zeros((n, self.n_internal_loops), dtype=np.float64)
        beta_all = np.zeros((p, self.n_internal_loops), dtype=np.float64)
        s_hat_all = np.zeros((m, self.n_internal_loops), dtype=np.float64)
        Q2_all = np.zeros((self.n_internal_loops), dtype=np.float64)

        # Call a different internal iteration routine to solve the linear system.
        if self.is_direct_solve:
            self.loginfo("Use direct solver for saddle-point (cokrigging) system")

            def internal_iteration(
                _inflation: float,
            ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
                return self.internal_iteration_direct(HZ, HX, Z, b, _inflation)

        else:
            self.loginfo(
                "Use Krylov subspace iterative solver "
                "for saddle-point (cokrigging) system"
            )
            invA_as_linop = self.get_invA_as_linop(HZ, HX, self.cov_obs)
            self.istate.invA_as_linop = invA_as_linop

            def internal_iteration(
                _inflation: float,
            ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
                return self.internal_iteration_krylov_subspace(
                    HZ, HX, Z, b, invA_as_linop, _inflation
                )

        # Single worker
        if self.max_workers == 1 or self.n_internal_loops == 1:
            results = map(internal_iteration, self.cov_obs_inflation_factors)
            # unpack the results
            for lm_it, res in enumerate(results):
                xi_all[:, lm_it], s_hat_all[:, lm_it], beta_all[:, lm_it] = (
                    _r.ravel() for _r in res
                )
        # Multi-threading -> only relevant if LM is ON. Threads are enough
        # since the work is dominated by BLAS/LAPACK calls (GIL released), and
        # they avoid pickling `self` (incl. Q and forward_model) for each task.
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(
                    executor.map(internal_iteration, self.cov_obs_inflation_factors)
                )
            # unpack the results
            for lm_it, res in enumerate(results):
                xi_all[:, lm_it], s_hat_all[:, lm_it], beta_all[:, lm_it] = (
                    _r.ravel() for _r in res
                )

        # 6.67 in kitanidisIntroductionGeostatisticsApplications1997.
        Q2_all[:] = np.dot(b[:n].T, xi_all) / (n - p)
        cR_all = self._compute_cR(Q2_all, sigma_cR, self.cov_obs_inflation_factors)

        # if lm is off, this will always be True
        is_valid_s_hat = np.invert(self.is_s_violate_lm_bounds(s_hat_all))

        if not np.any(is_valid_s_hat):
            # Every LM candidate violates the prescribed (lm_smin, lm_smax)
            # bounds. Rather than silently proceeding with empty arrays
            # (which would crash forward_model/argmin downstream with a
            # confusing error), fall back to considering all candidates and
            # let the objective-function comparison below pick the best one.
            self.loginfo(
                "All LM candidates violate the prescribed lm_smin/lm_smax "
                "bounds; falling back to evaluating all candidates and "
                "selecting the one with the lowest objective function."
            )
            is_valid_s_hat = np.ones_like(is_valid_s_hat, dtype=bool)

        # keep only valid s vectors and associated inflation factors
        # note that this has no effects if LM is off
        beta_hat_all = beta_all[:, is_valid_s_hat]
        s_hat_all = s_hat_all[:, is_valid_s_hat]
        valid_inflations = self.cov_obs_inflation_factors[is_valid_s_hat]
        # Q2_all/cR_all must be filtered the same way: best_obj_idx below is
        # computed in this filtered index space, and indexing the original,
        # unfiltered arrays with it would silently select the wrong
        # candidate's Q2/cR whenever any candidate is dropped above.
        Q2_all = Q2_all[is_valid_s_hat]
        cR_all = cR_all[is_valid_s_hat]

        # evaluate solutions
        if self.is_lm:
            self.loginfo(
                f"Evaluate the {np.count_nonzero(is_valid_s_hat)} LM solutions"
            )
        else:
            self.loginfo("Evaluate the best solution (no LM)")

        # Get the predictions. If LM is off, it is a single vector, otherwise it is
        # and ensemble with as many vectors as there are valid inflation factors
        # i.e., respecting the imposed bounds.
        simul_obs_all = self.forward_model(s_hat_all)

        n_valid = int(np.count_nonzero(is_valid_s_hat))
        if np.shape(simul_obs_all) != (self.obs.size, n_valid):
            raise ValueError(
                f"forward_model returned shape {np.shape(simul_obs_all)}, expected "
                f"({self.obs.size}, {n_valid}) i.e. (n_obs, number of valid "
                "LM candidates)"
            )

        self.loginfo(f"{n_valid} objective value evaluations")

        # objective function for all vectors
        objs: NDArrayFloat = self.objective_function(
            s_hat_all,
            beta_hat_all,
            simul_obs_all,
        )

        # get the index of the smallest objective function among the potential solutions
        best_obj_idx: int = np.argmin(objs).item()

        self.loginfo(
            f"solution obj={objs[best_obj_idx]:.2e} "
            f"(cov_obs inflation factor={valid_inflations[best_obj_idx]:.2e})"
        )

        # return the best solution
        return (
            s_hat_all[:, best_obj_idx].reshape(-1, 1),  # (n x 1)
            beta_hat_all[:, best_obj_idx].reshape(-1, 1),  # (p x 1)
            simul_obs_all[:, best_obj_idx].reshape(-1, 1),  # (N_obs x 1)
            float(valid_inflations[best_obj_idx]),
            float(objs[best_obj_idx]),
            float(Q2_all[best_obj_idx]),
            float(cR_all[best_obj_idx]),
        )

    def internal_iteration_direct(
        self,
        HZ: NDArrayFloat,
        HX: NDArrayFloat,
        Z: NDArrayFloat,
        b: NDArrayFloat,
        inflation: float,
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        # HQH^{t} + inflation * R
        Psi = self.get_psi(HZ, self.cov_obs, inflation)

        # use cholesky factorization to solve the system
        # this is much more efficient than direct solve
        LA = self.build_cholesky(Psi, HX)
        x = self.solve_cholesky(LA, b, self.drift.beta_dim)

        # Extract components and return final solution
        # x dimension (n+p,1)
        xi, beta = x[: self.n_obs, :], x[self.n_obs :, :]
        # HQ^T xi = Z (HZ^T xi): avoids building the dense (n_obs, s_dim) HQ.
        s_hat = (np.dot(self.drift.mat, beta) + Z @ (HZ.T @ xi)).reshape(-1, 1)

        # Return three column vectors
        return xi, s_hat, beta

    def internal_iteration_krylov_subspace(  # iterative approaches
        self,
        HZ: NDArrayFloat,
        HX: NDArrayFloat,
        Z: NDArrayFloat,
        b: NDArrayFloat,
        invA_as_linop: Optional[InvALinOp],
        inflation: float,
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        # The preconditioner must use the same inflation factor as A. Use a
        # (cheap, shallow) copy rather than mutating the shared instance, which
        # may be used concurrently by other threads.
        precond = (
            invA_as_linop.with_inflation(inflation)
            if invA_as_linop is not None
            else None
        )
        x = self._solve_iterative_subspace_krylov(
            self.get_A_as_linop(HX, HZ, inflation),  # linear op for matvec and matmat
            b,
            precond,  # preconditioner
            restart=50,
            atol=1e-10,
            rtol=1e-10,
            # number of restart cycles, bounded by the system size (n_obs + p)
            maxiter=self.n_obs + self.drift.beta_dim,
        )

        # Extract components and return final solution
        # x dimension (n+p,1)
        xi, beta = x[: self.n_obs].reshape(-1, 1), x[self.n_obs :].reshape(-1, 1)
        s_hat = (np.dot(self.drift.mat, beta) + np.dot(Z, np.dot(HZ.T, xi))).reshape(
            -1, 1
        )

        # Return three column vectors
        return xi, s_hat, beta

    def _solve_iterative_subspace_krylov(
        self,
        A: LinearOperator,
        b: NDArrayFloat,
        invA: Optional[LinearOperator],
        restart: int = 50,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        maxiter: int = 50,
    ) -> NDArrayFloat:
        callback = Residual()
        x, info = gmres(
            A,
            b,
            restart=restart,
            maxiter=maxiter,
            callback=callback,
            M=invA,
            atol=atol,
            rtol=rtol,
            callback_type="legacy",
        )
        if info > 0:
            msg = (
                f"GMRES did not converge to the requested tolerance after {info} "
                f"iterations (last residual: "
                f"{callback.res[-1] if callback.res else float('nan'):.2e})."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            self.loginfo(msg)
        return x

    def get_invA_as_linop(
        self, HZ: NDArrayFloat, HX: NDArrayFloat, cov_obs: covmats.CovarianceMatrix
    ) -> InvALinOp:
        """
        Return the low rank inverse of A as a linear operator.

        The output is used as a preconditioner for Krylov subspace iterative approaches
        when solving Ax = b.

        See section 2.3 in :cite:t:`leeScalableSubsurfaceInverse2016`.
        """
        self.loginfo(
            "Preconditioner construction using Generalized Eigen-decomposition"
        )
        t_start_precond = time()

        # generalized Hermitian eigenvalue problem (GHEP):
        # GHEP : HQHT u = lamdba R u
        # We can write R = LLT (cholesky)=> L^{-1}HQHTL^{-T}L^T u = lambda L^{T} u
        # y = L^{T} u
        # So this is a HEP:
        # L^{-1}HQHTL^{-T}y = lambda y

        # But this is not practical for large-scale R -> cholesky factorization is
        # needed.

        # Alternative from saibabaRandomizedAlgorithmsGeneralized2016:
        # Randomized eigenvalue decomposition (alg2). Still we must be able to
        # compute R^{-1} x (matvects)

        n_obs = self.n_obs
        beta_dim = self.drift.beta_dim

        if self.Q.n_pc < self.n_obs:
            k = self.Q.n_pc
        elif self.Q.n_pc == self.n_obs:
            k = self.Q.n_pc - 1
        else:
            k = self.n_obs - 1

        class InvRLinOp(LinearOperator):
            def __init__(self) -> None:
                super().__init__(shape=(n_obs, n_obs), dtype=np.float64)

            def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
                return cov_obs.solve(x)

        class HZZTHTLinOp(LinearOperator):
            def __init__(self) -> None:
                super().__init__(shape=(n_obs, n_obs), dtype=np.float64)

            def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
                return HZ @ (HZ.T @ x)

        HZZTHT_eig_vects, HZZTHT_eig_vals = ghep(
            HZZTHTLinOp(),
            cov_obs,
            InvRLinOp(),
            r=k,
            d=min(20, max(n_obs - k, 0)),
            single_pass=False,
            keep_neg_eigvals=False,
            random_state=self.random_state,
        )

        self.loginfo(
            f"- 1st eigv : {HZZTHT_eig_vals[0].item():.2e},"
            f"{HZZTHT_eig_vals.size}-th eigv : {HZZTHT_eig_vals[-1].item():.2e}, "
            f"ratio: {(HZZTHT_eig_vals[-1] / HZZTHT_eig_vals[0]).item():.2e}"
        )

        self.loginfo(
            "Time for data covarance construction :"
            f"{(time() - t_start_precond):.2e} sec"
        )
        if HZZTHT_eig_vects.shape[1] != self.Q.n_pc:
            self.loginfo(
                f"- rank of data covariance :{HZZTHT_eig_vects.shape[1]} "
                "for preconditioner construction."
            )

        # The inflation factor can be changed later with `with_inflation`.
        return InvALinOp(
            HX, HZZTHT_eig_vects, HZZTHT_eig_vals, cov_obs, n_obs, beta_dim, HZ=HZ
        )

    def line_search(
        self, s_cur: NDArrayFloat, s_past: NDArrayFloat
    ) -> Tuple[NDArrayFloat, NDArrayFloat, float]:
        """
        Simple line search between the previous and the current solutions.

        ``max_it_ls`` candidates ``delta * s_past + (1 - delta) * s_cur`` with
        ``delta`` linearly spaced in [-0.1, 1.1] are evaluated with a single
        call to `forward_model`, and the one with the lowest (beta-marginalized)
        objective function is returned.
        """
        n_pts = self.max_it_ls
        delta = np.linspace(-0.1, 1.1, n_pts)
        # (m, n_pts) ensemble built in one vectorized operation
        s_hat_all = s_past * delta + s_cur * (1.0 - delta)

        self.loginfo("evaluate linesearch solutions")
        simul_obs_all = self.forward_model(s_hat_all)

        if np.shape(simul_obs_all) != (self.n_obs, n_pts):
            raise ValueError(
                f"forward_model returned shape {np.shape(simul_obs_all)}, expected "
                f"({self.n_obs}, {n_pts}) i.e. (n_obs, max_it_ls)"
            )

        objs = np.array(
            [
                self.objective_function_no_beta(
                    s_hat_all[:, i : i + 1], simul_obs_all[:, i : i + 1]
                )
                for i in range(n_pts)
            ]
        )
        # NaN objectives (e.g. failed forward runs) are never selected
        objs[~np.isfinite(objs)] = np.inf
        i_best = int(np.argmin(objs))
        self.loginfo(
            f"{i_best}-th solution obj {objs[i_best]:e} (delta {delta[i_best]:f})"
        )
        return (
            s_hat_all[:, i_best : i_best + 1],
            simul_obs_all[:, i_best : i_best + 1],
            float(objs[i_best]),
        )

    def get_optimal_beta(self, s_cur: NDArrayFloat) -> NDArrayFloat:
        """
        Return the drift coefficients minimizing the regularization term.

        beta = (X^T Q^{-1} X)^{-1} X^T Q^{-1} s, i.e., the beta implicitly
        used by :meth:`objective_function_no_beta`.
        """
        X = self.drift.mat
        inv_sqrt_l = 1.0 / np.sqrt(np.ravel(self.Q.eig_vals))[:, np.newaxis]
        invZs = inv_sqrt_l * (self.Q.eig_vects.T @ s_cur)
        invZX = inv_sqrt_l * (self.Q.eig_vects.T @ X)
        return np.linalg.solve(invZX.T @ invZX, invZX.T @ invZs).reshape(-1, 1)

    def display_objfun(
        self,
        loss_ls: float,
        n_obs: int,
        rmse: float,
        n_rmse: float,
        n_iter: Optional[int] = None,
        obj: Optional[float] = None,
        res: Optional[float] = None,
        is_beta: bool = True,
        Q2: Optional[float] = None,
        cR: Optional[float] = None,
    ) -> None:
        # NOTE: n_iter=0 is the first Gauss-Newton iteration (its header used
        # to be skipped); None means "not an iteration" (initial/final state).
        if n_iter is not None:
            self.loginfo(f"== iteration {n_iter + 1:d} summary ==")

        dat = {
            "LS objfun 0.5 (obs. diff.)^T R^{-1}(obs. diff.)": loss_ls,
            "norm LS objfun 0.5 / nobs (obs. diff.)^T R^{-1}(obs. diff.)": loss_ls
            / n_obs,
            "RMSE (norm(obs. diff.)/sqrt(nobs))": rmse,
            "norm RMSE (norm(obs. diff./sqrtR)/sqrt(nobs))": n_rmse,
        }
        if obj is not None:
            if is_beta:
                dat["objective function"] = obj
            else:
                dat["objective function (no beta)"] = obj
        if Q2 is not None:
            dat["predictive model checking Q2 (should be close to 1.0)"] = Q2
        if cR is not None:
            dat["predictive model checking cR (should be close to 1.0)"] = cR
        if res is not None:
            _n = 0 if n_iter is None else n_iter
            dat[f"relative L2-norm diff btw sol {_n:d} and sol {_n + 1:d}"] = res

        maxlen = max([len(k) for k in dat]) + 1
        for k, v in dat.items():
            self.loginfo(f"** {k:<{maxlen}} : {v:.3e}")

    def _invoke_callback(
        self, s_hat: NDArrayFloat, simul_obs: NDArrayFloat, n_iter: int
    ) -> None:
        """
        Call the optional user callback, if any, with the current solver state.

        Used both to save the initial state (`n_iter=0`, before any
        Gauss-Newton iteration) and at the end of every subsequent internal
        iteration, so some intermediate solver states can be saved/plotted.
        """
        if self.callback is not None:
            self.callback(self, s_hat=s_hat, simul_obs=simul_obs, n_iter=n_iter)

    def _gauss_newton_step(
        self, s_past: NDArrayFloat, simul_obs_cur: NDArrayFloat, n_iter: int
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, float, float]:
        """
        Perform a single Gauss-Newton (+ optional LM) internal iteration.

        Runs one `linear_iteration`, invokes the optional user callback, and
        records the resulting objective/inflation/Q2/cR values in
        `self.istate`. Convergence checks and the best-solution bookkeeping
        stay in the caller (`gauss_newton`), since they drive the loop's
        control flow (`break`/line search).

        Returns
        -------
        Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, float, float]
            (s_cur, beta_cur, simul_obs_cur, inflation_cur, obj) for this step.
        """
        start = time()
        self.loginfo("")
        self.loginfo(f"***** Iteration {n_iter + 1} ******")

        s_cur, beta_cur, simul_obs_cur, inflation_cur, obj, Q2, cR = (
            self.linear_iteration(s_past, simul_obs_cur)
        )

        self._invoke_callback(s_hat=s_cur, simul_obs=simul_obs_cur, n_iter=n_iter)

        # save the objective function, inflation factors etc.
        self.istate.obj_seq.append(obj)
        self.istate.inflation_seq.append(inflation_cur)
        self.istate.Q2_seq.append(Q2)
        self.istate.cR_seq.append(cR)

        self.loginfo(
            "- Geostat. inversion at iteration %d is %g sec"
            % ((n_iter + 1), round(time() - start))
        )
        return s_cur, beta_cur, simul_obs_cur, inflation_cur, obj

    def gauss_newton(self) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, int]:
        """
        Gauss-newton iteration
        """

        s_init = self.s_init

        res = 1.0

        # Reset the internal state so that calling run() twice does not mix
        # (and index into) the sequences of two distinct inversions.
        self.istate = InternalState(s_best=np.copy(s_init))
        self.post_diagv = self.prior_s_var

        self.loginfo("##### 2. Start PCGA Inversion #####")
        self.loginfo("-- evaluate initial solution")

        # s_init has shape (s_dim, 1)
        simul_obs_init = self.forward_model(s_init)
        self.istate.simul_obs_best = simul_obs_init

        self.istate.simul_obs_init = simul_obs_init
        residuals = (simul_obs_init.T - self.obs).T
        # item() to convert to scalar
        rmse_init: float = self.rmse(residuals, False).item()
        n_rmse_init: float = self.rmse(residuals, True).item()
        loss_ls_init: float = self.objective_function_ls(simul_obs_init).item()

        simul_obs_cur = np.copy(simul_obs_init)
        s_cur = np.copy(s_init)
        s_past = np.copy(s_init)

        obj = self.objective_function_no_beta(s_cur, simul_obs_cur)
        # initial objective function -> very high to avoid early termination
        # because objective function no beta might be too high
        obj_old = VERY_LARGE_NUMBER

        self.display_objfun(
            loss_ls_init,
            simul_obs_init.size,
            rmse_init,
            n_rmse_init,
            obj=obj,
            is_beta=False,
        )

        # save the initial objective function
        self.istate.obj_seq.append(float(obj))

        # Save the initial state
        self._invoke_callback(s_hat=s_cur, simul_obs=simul_obs_cur, n_iter=0)

        for n_iter in range(self.maxiter):
            s_cur, beta_cur, simul_obs_cur, inflation_cur, obj = (
                self._gauss_newton_step(s_past, simul_obs_cur, n_iter)
            )

            # case 1: progress in objective function
            if obj < obj_old:
                self.istate.s_best = s_cur
                self.istate.beta_best = beta_cur
                self.istate.simul_obs_best = simul_obs_cur
                self.istate.iter_best = n_iter + 1
            # case 2: no progress in objective function
            else:
                if self.is_line_search:
                    self.loginfo(
                        "perform simple linesearch due to no progress in obj value"
                    )
                    s_cur, simul_obs_cur, obj = self.line_search(s_cur, s_past)
                    # beta consistent with the line-search solution (the one
                    # from the Gauss-Newton step belongs to another s).
                    beta_cur = self.get_optimal_beta(s_cur)
                    if obj < obj_old:
                        self.istate.s_best = s_cur
                        self.istate.beta_best = beta_cur
                        self.istate.simul_obs_best = simul_obs_cur
                        self.istate.iter_best = n_iter + 1
                    else:
                        if n_iter > 1:
                            self.loginfo("no progress in obj value")
                            n_iter += 1
                            break
                        else:
                            self.loginfo(
                                "no progress in obj value but wait for one "
                                "more iteration..."
                            )
                            # allow first few iterations
                            pass  # allow for
                else:
                    self.istate.status = "CONVERGENCE: NO PROGRESS IN OBJ VALUE"
                    self.istate.is_success = False
                    n_iter += 1
                    break

            norm_s_past = float(np.linalg.norm(s_past))
            res = (
                float(np.linalg.norm(s_past - s_cur)) / norm_s_past
                if norm_s_past > 0.0
                else float("inf")  # e.g. s_init = 0
            )
            residuals = (simul_obs_cur.T - self.obs).T
            loss_ls = self.objective_function_ls(simul_obs_cur).item()
            rmse = self.rmse(residuals, False).item()
            n_rmse = self.rmse(residuals, True).item()
            obj = self.objective_function(s_cur, beta_cur, simul_obs_cur).item()

            self.display_objfun(
                loss_ls,
                simul_obs_init.size,
                rmse,
                n_rmse,
                n_iter,
                obj=obj,
                res=res,
                Q2=self.istate.Q2_cur,
                cR=self.istate.cR_cur,
            )

            if res < self.restol:
                self.istate.status = "CONVERGENCE: MODEL_CHANGE_<=_RES_TOL"
                self.istate.is_success = True
                n_iter += 1
                break
            elif np.abs((obj - obj_old)) / max(abs(obj_old), abs(obj), 1) < self.ftol:
                self.istate.status = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL"
                self.istate.is_success = True
                n_iter += 1
                break
            elif self.ftarget is not None:
                if self.ftarget > obj:
                    self.istate.status = "CONVERGENCE: F_<=_FTARGET"
                    self.istate.is_success = True
                    n_iter += 1
                    break

            # To add before the previous check, otherwise
            # obj == self.istate.obj_seq[-1] and the elif condition is always True
            # which cause an early break
            s_past = np.copy(s_cur)

            if n_iter + 1 >= self.maxiter:
                self.istate.status = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
                self.istate.is_success = True

            obj_old = copy.copy(obj)

            # Only recompute the posterior variance when this iteration
            # actually produced a new best estimate (case 1 above, or case 2
            # with a successful line search). self.istate.HZ/HX (set by the
            # linear_iteration call inside _gauss_newton_step) and
            # inflation_cur are the Jacobian/inflation that produced s_cur;
            # if s_cur was not accepted as the new best -- e.g. the "wait for
            # one more iteration" grace pass below, where line search failed
            # but we haven't broken out yet -- then self.istate.s_best is
            # unchanged from a previous iteration, and recomputing post_diagv
            # here would attribute posterior uncertainty to the wrong
            # solution. The posterior variance already stored from that
            # earlier best iteration remains the correct one to report.
            if self.istate.iter_best == n_iter + 1:
                # need to use HZ and HX here !
                # assume linesearch result close to the current solution
                assert self.istate.last_jac is not None
                HX_cur, HZ_cur, Hs_cur = self.istate.last_jac
                # Jacobian products consistent with s_best/best_inflation, required
                # for the posterior covariance (also after the inversion).
                self.istate.HX, self.istate.HZ, self.istate.Hs = (
                    HX_cur,
                    HZ_cur,
                    Hs_cur,
                )
                start = time()
                if self.is_direct_solve:
                    self.loginfo(
                        "start direct posterior variance computation "
                        "- this option works for O(nobs) ~ 100"
                    )
                else:
                    self.loginfo("start posterior variance computation")
                self.post_diagv = self._compute_post_cov_diag(
                    HZ_cur,
                    HX_cur,
                    self.cov_obs,
                    inflation_cur,
                    is_direct_solve=self.is_direct_solve,
                )
                self.loginfo(f"posterior diag. computed in {(time() - start):.3e} s")
            else:
                self.loginfo(
                    "skip posterior variance computation "
                    "(no new best solution at this iteration)"
                )
            # if self.iter_save:
            #     np.savetxt("./postv.txt", self.post_diagv)

        # Free memory
        self.istate.last_jac = None
        if not self.is_save_jac:
            self.istate.HX = np.array([])
            self.istate.HZ = np.array([])
            self.istate.Hs = np.array([])
            self.istate.invA_as_linop = None

        self.loginfo("------------ Inversion Summary ---------------------------")
        self.loginfo(f"** Success = {self.istate.is_success}")
        self.loginfo(f"** Status  = {self.istate.status}")

        if self.istate.iter_best == 0:
            self.loginfo("** Did not found better solution than initial guess")
        else:
            self.loginfo(f"** Found solution at iteration {self.istate.iter_best}")
        residuals = (self.istate.simul_obs_best.T - self.obs).T
        loss_ls_best = self.objective_function_ls(self.istate.simul_obs_best).item()
        rmse_best: float = self.rmse(residuals, False).item()
        n_rmse_best: float = self.rmse(residuals, True).item()

        self.display_objfun(
            loss_ls_best,
            simul_obs_init.size,
            rmse_best,
            n_rmse_best,
            obj=self.istate.best_obj,
        )

        self.loginfo(
            f"- Final predictive model checking Q2 = {self.istate.Q2_best:.3e}"
            " (should be as close to 1.0 as possible.)"
        )
        self.loginfo(
            f"- Final predictive model checking cR = {self.istate.best_cR:.3e}"
            " (should be as close to 1.0 as possible.)"
        )

        return (
            self.istate.s_best,
            self.istate.simul_obs_best,
            self.post_diagv,
            self.istate.iter_best,
        )

    def run(self) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, int]:
        start = time()
        s_hat, simul_obs, post_diagv, iter_best = self.gauss_newton()
        self.loginfo(f"** Total elapsed time is {(time() - start):.3e} s")
        self.loginfo("----------------------------------------------------------")
        return s_hat, simul_obs, post_diagv, iter_best

    def get_psi(
        self, HZ: NDArrayFloat, cov_obs: covmats.CovarianceMatrix, inflation: float
    ) -> NDArrayFloat:
        """Get the matrix HQH^{T} + \alpha R."""
        # NOTE: the inflation factor applies to R (Levenberg-Marquardt
        # inflation of the observation-error covariance), consistently with
        # the Krylov operator (get_A_as_linop), the preconditioner (InvALinOp)
        # and the cR statistic. It used to multiply HQH^{T} instead.
        return np.dot(HZ, HZ.T) + inflation * cov_obs.todense()

    @staticmethod
    def build_cholesky(Psi: NDArrayFloat, HX: NDArrayFloat) -> NDArrayFloat:
        # HQH^{T} and R are positive semi-definite
        # Then we just need to factorize L11
        # Cholesky:
        L11 = sp.linalg.cholesky(Psi, lower=True, overwrite_a=False)
        L12 = sp.linalg.solve_triangular(L11, HX, lower=True, trans="N")
        L22 = sp.linalg.cholesky(L12.T @ L12, lower=True)
        return np.hstack(
            [np.vstack([L11, L12.T]), np.vstack([np.zeros(L12.shape), L22])]
        )

    @staticmethod
    def build_dense_A_from_cholesky(LA: NDArrayFloat, p: int) -> NDArrayFloat:
        # We don't build explicitly E, we just build the diagonal values
        E = np.identity(n=LA.shape[0])
        E[-p:, -p:] *= -1
        return (LA @ E) @ LA.T

    @staticmethod
    def solve_cholesky(LA: NDArrayFloat, v: NDArrayFloat, p: int) -> NDArrayFloat:
        # LA is the lowest triangle of the cholesky factorization LA @ E @ LA.T.
        v = sp.linalg.solve_triangular(LA, v, lower=True)
        v[-p:] *= -1
        return sp.linalg.solve_triangular(LA.T, v, lower=False)

    @staticmethod
    def build_dense_A(Psi: NDArrayFloat, HX: NDArrayFloat) -> NDArrayFloat:
        n: int = Psi.shape[0]
        p: int = HX.shape[1]
        A = np.zeros((n + p, n + p), dtype=np.float64)
        A[0:n, 0:n] = np.copy(Psi)
        A[0:n, n : n + p] = np.copy(HX)
        A[n : n + p, 0:n] = np.copy(HX.T)
        return A

    def get_A_as_linop(
        self, HX: NDArrayFloat, HZ: NDArrayFloat, inflation: float
    ) -> LinearOperator:
        n_obs = self.n_obs
        beta_dim = self.drift.beta_dim

        def mv(v: NDArrayFloat) -> NDArrayFloat:
            return np.concatenate(
                (
                    (
                        np.dot(HZ, np.dot(HZ.T, v[: self.n_obs]))
                        + inflation * self.cov_obs @ v[: self.n_obs]
                        + np.dot(HX, v[self.n_obs :])
                    ),
                    (np.dot(HX.T, v[: self.n_obs])),
                ),
                axis=0,
            )

        # Matrix handle
        class ALinOp(LinearOperator):
            def __init__(self) -> None:
                super().__init__(
                    shape=(n_obs + beta_dim, n_obs + beta_dim), dtype=np.float64
                )

            def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
                return mv(x)

            def _rmatvec(self, x: NDArrayFloat) -> NDArrayFloat:
                return mv(x)

        return ALinOp()

    def _get_invA_for(
        self, HZ: NDArrayFloat, HX: NDArrayFloat, cov_obs: covmats.CovarianceMatrix
    ) -> InvALinOp:
        """
        Return the low rank approximation of A^{-1} for the given linearization.

        The operator built during the last linear iteration is reused when it
        was built from the very same (HZ, HX) arrays, which avoids a second,
        expensive randomized generalized eigen-decomposition.
        """
        op = self.istate.invA_as_linop
        if op is not None and op.HZ is HZ and op.HX is HX and op.cov_obs is cov_obs:
            return op
        return self.get_invA_as_linop(HZ, HX, cov_obs)

    def _get_post_cov_lowrank(
        self,
        HZ: NDArrayFloat,
        HX: NDArrayFloat,
        cov_obs: covmats.CovarianceMatrix,
        inflation: float,
        is_direct_solve: bool,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        r"""
        Return (W, K) such that the posterior covariance is Q - W^T K W.

        The classical formulation is Q - b^T A^{-1} b with
        b = [HQ; X^T] = [HZ Z^T; X^T], of shape (n_obs + p, s_dim), which
        requires solving A with s_dim right-hand sides. Writing
        b = G W with G = blockdiag(HZ, I_p), shape (n_obs + p, n_pc + p), and
        W = [Z^T; X^T], shape (n_pc + p, s_dim), gives
        b^T A^{-1} b = W^T (G^T A^{-1} G) W = W^T K W, so that only n_pc + p
        right-hand sides are needed, and K is a small (n_pc + p) square matrix.
        """
        Z = np.sqrt(self.Q.eig_vals).T * self.Q.eig_vects
        p = self.drift.beta_dim
        n_pc = HZ.shape[1]

        # W = [Z^T; X^T], shape (n_pc + p, s_dim)
        W = np.vstack([Z.T, self.drift.mat.T])
        # G = blockdiag(HZ, I_p), shape (n_obs + p, n_pc + p)
        G = np.zeros((self.n_obs + p, n_pc + p), dtype=np.float64)
        G[: self.n_obs, :n_pc] = HZ
        G[self.n_obs :, n_pc:] = np.identity(p)

        if is_direct_solve:
            # Use cholesky factorization to solve the system
            LA = self.build_cholesky(self.get_psi(HZ, cov_obs, inflation), HX)
            invAG = PCGA.solve_cholesky(LA, G, p)
        else:
            # Solving Ax = b with a Krylov solver for each column would be too
            # long, so we use the (low rank) preconditioner, which works fine.
            invAG = self._get_invA_for(HZ, HX, cov_obs).with_inflation(inflation) @ G
        K = G.T @ invAG
        # Symmetric in exact arithmetic
        return W, 0.5 * (K + K.T)

    def _check_jac_available(self) -> None:
        if np.size(self.istate.HZ) == 0:
            raise RuntimeError(
                "The Jacobian products (HX, HZ) at the best solution are not "
                "available: either `run` has not been called, no iteration "
                "improved on the initial guess, or `is_save_jac` is False."
            )

    def _compute_post_cov_diag(
        self,
        HZ: NDArrayFloat,
        HX: NDArrayFloat,
        cov_obs: covmats.CovarianceMatrix,
        inflation: float,
        is_direct_solve: bool,
    ) -> NDArrayFloat:
        """Computing posterior diagonal entries using cholesky/it krylov subspace."""
        W, K = self._get_post_cov_lowrank(
            HZ=HZ,
            HX=HX,
            cov_obs=cov_obs,
            inflation=inflation,
            is_direct_solve=is_direct_solve,
        )
        # diag(W^T K W) without forming the (s_dim, s_dim) matrix
        return (self.prior_s_var - ensemble_dot(W, K @ W)).reshape(-1, 1)

    def get_dense_post_cov(
        self,
        is_direct_solve: Optional[bool] = None,
        inflation: Optional[float] = None,
    ) -> NDArrayFloat:
        """
        Return the dense posterior covariance matrix..

        Notes
        -----
        This is not practical for large scale models.

        Parameters
        ----------
        is_direct_solve : Optional[bool], optional
            Whether to solve the saddle-point system with the direct Cholesky
            approach rather than the iterative Krylov subspace approach used
            when building the posterior covariance.
            If None, `self.is_direct_solve` (the value used during the
            inversion) is taken. By default None.
        inflation : Optional[float], optional
            Inflation factor used to build the posterior covariance matrix.
            If None, the random_state used by PCGA is taken. By default None.

        Notes
        -----
        This is practical for posterior sampling.
        """
        if is_direct_solve is None:
            _is_direct_solve: bool = self.is_direct_solve
        else:
            _is_direct_solve = is_direct_solve
        if inflation is None:
            _inflation: float = self.istate.best_inflation
        else:
            _inflation = inflation
        self._check_jac_available()

        W, K = self._get_post_cov_lowrank(
            HZ=self.istate.HZ,
            HX=self.istate.HX,
            cov_obs=self.cov_obs,
            inflation=_inflation,
            is_direct_solve=_is_direct_solve,
        )
        return self.Q.todense() - W.T @ (K @ W)

    def get_eigen_post_cov(
        self,
        is_direct_solve: Optional[bool] = None,
        inflation: Optional[float] = None,
        n_pc: Optional[int] = None,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None,
    ) -> covmats.CovViaEigenFactorization:
        """
        Return the posterior covariance matrix through an Eigen factorization.

        Notes
        -----
        This is practical for posterior sampling.

        Parameters
        ----------
        is_direct_solve : Optional[bool], optional
            Whether to solve the saddle-point system with the direct Cholesky
            approach rather than the iterative Krylov subspace approach used
            when building the posterior covariance.
            If None, `self.is_direct_solve` (the value used during the
            inversion) is taken. By default None.
        inflation : Optional[float], optional
            Inflation factor used to build the posterior covariance matrix.
            If None, the random_state used by PCGA is taken. By default None.
        n_pc : Optional[int], optional
            Number of principal component used for the eigen factorization of the
            posterior covariance matrix.
            It can differ from the number of PC used by PCGA.
            If None, the number of PC used by PCGA is taken. By default None.
        random_state : Optional[ Union[int, np.random.Generator, np.random.RandomState]]
            If None, the random_state used by PCGA is taken. By default None.

        Returns
        -------
        covmats.CovViaEigenFactorization
            Low rank approximation of the posterior covariance matrix.
        """
        if is_direct_solve is None:
            _is_direct_solve: bool = self.is_direct_solve
        else:
            _is_direct_solve = is_direct_solve
        if inflation is None:
            _inflation: float = self.istate.best_inflation
        else:
            _inflation = inflation
        if n_pc is None:
            _n_pc: int = self.Q.n_pc
        else:
            _n_pc = n_pc
        if random_state is None:
            _random_state = self.random_state
        else:
            _random_state = check_random_state(random_state)
        self._check_jac_available()

        W, K = self._get_post_cov_lowrank(
            HZ=self.istate.HZ,
            HX=self.istate.HX,
            cov_obs=self.cov_obs,
            inflation=_inflation,
            is_direct_solve=_is_direct_solve,
        )

        Q = self.Q

        class _op(LinearOperator):
            def __init__(self):
                super().__init__(shape=Q.shape, dtype="d")

            def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
                """Return the covariance matrix times the vector x."""
                return Q @ x - W.T @ (K @ (W @ x))

            def _matmat(self, X: NDArrayFloat) -> NDArrayFloat:
                return Q @ X - W.T @ (K @ (W @ X))

        return covmats.CovViaEigenFactorization(
            covmats.get_linop_eigen_factorization(
                _op(), size=self.s_dim, n_pc=_n_pc, random_state=_random_state
            )
        )
