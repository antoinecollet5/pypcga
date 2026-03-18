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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from math import isnan, sqrt
from time import time
from typing import Any, Callable, Generator, Iterator, List, Optional, Tuple, Union

import covmats
import numpy as np
import scipy as sp
from scipy._lib._util import check_random_state
from scipy.sparse.linalg import LinearOperator, gmres

from pypcga._utils import (
    NDArrayBool,
    NDArrayFloat,
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
    obj_seq: List[float] = field(default_factory=lambda: [])
    inflation_seq: List[float] = field(default_factory=lambda: [])
    Q2_seq: List[float] = field(default_factory=lambda: [])
    cR_seq: List[float] = field(default_factory=lambda: [])
    iter_best: int = 0
    status: str = "ILDE."
    is_success: bool = False

    @property
    def best_obj(self) -> float:
        """
        Return the best objective function obtained in the optimization.

        The first objective function is ignored because beta is minimized.
        """
        return self.obj_seq[self.iter_best]

    @property
    def Q2_best(self) -> float:
        return self.Q2_seq[self.iter_best - 1]

    @property
    def best_cR(self) -> float:
        return self.cR_seq[self.iter_best - 1]

    @property
    def best_inflation(self) -> float:
        return self.inflation_seq[self.iter_best - 1]

    @property
    def Q2_cur(self) -> float:
        return self.Q2_seq[-1]

    @property
    def cR_cur(self) -> float:
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
    ) -> None:
        """Initialize the instance."""
        super().__init__(dtype=np.float64, shape=(n_obs + beta_dim, n_obs + beta_dim))
        self.inflation: float = 1.0
        self.n_obs: int = n_obs
        self.beta_dim: int = beta_dim
        self.HX: NDArrayFloat = HX
        self.HZZTHT_eig_vects: NDArrayFloat = HZZTHT_eig_vects
        self.HZZTHT_eig_vals: NDArrayFloat = HZZTHT_eig_vals
        self.cov_obs = cov_obs

    def update_inflation_factor(self, value: float) -> None:
        """Update the inflation factor."""
        self.inflation = value

    def inv_psi(self, v: NDArrayFloat, inflation: float) -> NDArrayFloat:
        Dvec = sp.sparse.diags_array(
            np.divide(
                (1.0 / inflation * self.HZZTHT_eig_vals.ravel()),
                ((1.0 / inflation) * self.HZZTHT_eig_vals.ravel() + 1.0),
            )
        )  # (n_pc,)

        return (1.0 / inflation) * (
            self.cov_obs.solve(v)
            - (self.HZZTHT_eig_vects @ (Dvec @ (self.HZZTHT_eig_vects.T @ v)))
        )

    def _matvec(self, x: NDArrayFloat) -> NDArrayFloat:
        # See Eq. 14 in leeScalableSubsurfaceInverse2016.
        # Warning, there is typo for the block 22 of A^{-1}.
        # i.e., it is -S^{-1} instead of -S.

        # (n_obs, 1) vector  -> First part of the multiplication with
        # block (1, 1) of A^{-1}
        invPsiv = self.inv_psi(x[: self.n_obs], self.inflation)

        # S is a p by p matrix
        S = np.dot(self.HX.T, self.inv_psi(self.HX, self.inflation))
        # So we can invert it directly because p is usually between 1 and 3.
        invS = np.linalg.inv(S)

        # multiplication with block (2,1) of A^{-1} => S^{-1} \Phi^T \Psi^{-1}
        invSHXTinvPsiv = invS @ np.dot(self.HX.T, invPsiv)

        # Second part of the multiplication with block (1, 1) of A^{-1}
        invPsiHXinvSHXTinvPsiv = self.inv_psi(
            np.dot(self.HX, invSHXTinvPsiv), self.inflation
        )

        # multiplication with block (1,2) of A^{-1} =>  \Psi^{-1} \Phi^T S^{-1}
        invPsiHXinvSv1 = self.inv_psi(
            np.dot(self.HX, invS @ x[self.n_obs :]), self.inflation
        )

        # multiplication with block (2, 2)
        invSv1 = invS @ x[self.n_obs :]

        # Gathering the resulting vector.
        return np.concatenate(
            (
                (invPsiv - invPsiHXinvSHXTinvPsiv + invPsiHXinvSv1),
                (invSHXTinvPsiv - invSv1),
            ),
            axis=0,
        )

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
        "HX",
        "HZ",
        "Hs",
        "invA_as_linop",
        "simul_obs_init",
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
        ] = np.random.default_rng(),
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
            _description_, by default None
        prior_s_var : Optional[Union[float, NDArrayFloat]], optional
            _description_, by default None
        callback : Optional[Callable], optional
            _description_, by default None
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
            Whether to solve the saddle point system (Ax = b), see eq 4.53 in (add ref),
            with:
                - the direct approach (Cholesky factorization of A), see eq (4.57) in
                :cite:`colletAssistedHistoryMatching2024a`
                - or to use the alternative iterative Krylov subspace approach as
                described in :cite`saibabaEfficientMethodsLargescale2012,
                saibabaFastAlgorithmsGeostatistical2013`.
            Using direct solve is practical if the number of observations is around 100
            or less. Beyond, it is advise to rely on the The default is False
            (using iterative Kryloc subspace by default).
        random_state : Optional[ Union[int, np.random.Generator, np.random.RandomState]]
            Pseudorandom number generator state used to generate resamples.
            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.
        is_objfun_exact : bool, optional
            _description_, by default False
        max_it_lm: int
            Maximum number iterations when using Levenberg Marquard regularization.
            Only applies if `is_lm` is True. By default use all available CPUs.
        alphamax_lm : float, optional
            Maximum weight for LM. TODO: add the formula. By default 10.0**3.
        lm_smax : Optional[float], optional
            Maximum LM solution, by default None
        max_it_ls : int, optional
            Maximum number of iterations when using line search, by default 20.
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
                \mathbf{s}_{\ell} \right\rVert^{2}}{\left\lVert
                \mathbf{s}_{\ell} \right\rVert^{2}}
            by default 1e-2.
        logger: Optional[Logger], optional
            Logger instance. If no logger is passed, there will be no output.
            By default None.
        is_save_jac : bool, optional
            _description_, by default False
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

        # TODO: make configurable + explain that this is not intesreting for small
        # size problem because the time to start the processes is higher than the
        # calculation time
        # this is interesting when the number of observations is large.
        self.max_workers = max(max_workers_lm, max_it_lm) if is_lm else 1

        # keep track of the internal state
        self.istate = InternalState(s_best=s_init)
        # Random state for v0 vector used by eigsh and svds
        self.random_state = check_random_state(random_state)

        if prior_s_var is not None:
            self.prior_s_var = prior_s_var
        else:
            self.prior_s_var = self.Q.get_diagonal()

        # Initialized as the diagonal of the covariance matrix
        self.post_diagv = self.prior_s_var

        # Define Drift (or Prior) functions
        if drift is not None:
            assert drift.s_dim == self.s_dim
            self.drift: covmats.DriftMatrix = drift
        else:
            self.drift = covmats.ConstantDriftMatrix(self.s_dim)

        # Internal state
        self.is_save_jac = is_save_jac

        self.cov_obs_inflation_factors = self.get_cov_obs_inflation_factors()
        self.logger: Optional[logging.Logger] = logger

        # TODO: see if we move these internal states
        self.HX: NDArrayFloat = np.array([])
        self.HZ: NDArrayFloat = np.array([])
        self.Hs: NDArrayFloat = np.array([])

        # approximate inverse of A used a preconditioner to solve Ax = b
        # None by default -> updated later
        self.invA_as_linop: Optional[InvALinOp] = None

        self.simul_obs_init: NDArrayFloat = np.array([], dtype=np.float64)

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
            "Minimum model change (restol)": np.round(self.restol, 3),
            "Minimum obj fun change (ftol)": np.round(self.ftol, 3),
            "Target obj fun (ftarget)": (
                np.round(self.ftarget, 3) if self.ftarget is not None else None
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

        It is a sequence only if Levenberg Marqard is on. that depends on both the
        max number of Levenberg

        Returns
        -------
        NDArrayFloat
            _description_
        """
        if self.is_lm:
            return 10 ** (np.linspace(0.0, np.log10(self.alphamax_lm), self.max_it_lm))
        return np.array([1.0])

    def jac_vect(self, x, s, simul_obs, eps, delta=None):
        """
        Jacobian times Matrix (Vectors) in Parallel
        perturbation interval delta determined following Brown and Saad [1990]
        """
        nruns = np.size(x, 1)

        # TODO: create a function perturb x (make an ensemble of perturbed values)
        # And test the function outside the loop
        deltas = np.zeros((nruns, 1), "d")

        if delta is None or isnan(delta) or delta == 0:
            for i in range(nruns):
                mag = np.dot(s.T, x[:, i : i + 1])
                absmag = np.dot(abs(s.T), abs(x[:, i : i + 1]))
                if mag >= 0:
                    signmag = 1.0
                else:
                    signmag = -1.0

                deltas[i] = (
                    signmag
                    * sqrt(eps)
                    * (max(abs(mag), absmag))
                    / ((np.linalg.norm(x[:, i : i + 1]) + np.finfo(float).eps) ** 2)
                )

                if deltas[i] == 0:  # s = 0 or x = 0
                    self.loginfo(
                        "%d-th delta: signmag %g, eps %g, max abs %g, norm %g"
                        % (
                            i,
                            signmag,
                            eps,
                            (max(abs(mag), absmag)),
                            (np.linalg.norm(x) ** 2),
                        )
                    )

                    deltas[i] = sqrt(eps)

                    self.loginfo(
                        f"{i}-th delta: assigned as sqrt(eps) - {deltas[i]:.2e}",
                    )
                    # raise ValueError('delta is zero? - plz check your
                    # s_init is within a reasonable range')

                # reuse storage x by updating x
                x[:, i : i + 1] = s + deltas[i] * x[:, i : i + 1]

        else:
            for i in range(nruns):
                deltas[i] = delta
                # reuse storage x by updating x
                x[:, i : i + 1] = s + deltas[i] * x[:, i : i + 1]

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

        Jxs = np.zeros_like(simul_obs_purturbation)

        # solve Hx HZ HQT
        for i in range(nruns):
            Jxs[:, i : i + 1] = np.true_divide(
                (simul_obs_purturbation[:, i : i + 1] - simul_obs), deltas[i]
            )

        return Jxs

    def objective_function_ls(self, simul_obs) -> NDArrayFloat:
        """

        simul_obs with shape (n_obs, ne)

        0.5(y-h(s))^TR^{-1}(y-h(s))

        TODO: as vectors.

        return size = Ne
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
            smxb = (s_cur - np.dot(X, np.atleast_2d(beta))).ravel()
            return float(0.5 * smxb.T.dot(self.Q.solve(smxb)).item())

        # We solve with a newton to find the optimal beta
        def jac_wrt_beta(beta: NDArrayFloat) -> NDArrayFloat:
            """-X^TC^{-1}(s-Xb)"""
            smxb = (s_cur - np.dot(X, np.atleast_2d(beta))).ravel()
            return -X.T.dot(self.Q.solve(smxb))

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
            return np.sqrt(residuals.T.dot(self.cov_obs.solve(residuals)) / self.d_dim)
        return np.linalg.norm(residuals, axis=0) / np.sqrt(self.d_dim)

    def jac_mat(
        self, s_cur: NDArrayFloat, simul_obs: NDArrayFloat, Z: NDArrayFloat
    ) -> Tuple[
        NDArrayFloat,
        NDArrayFloat,
        NDArrayFloat,
        Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat],
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

        if self.is_save_jac:
            self.HX = HX
            self.HZ = HZ
            self.Hs = Hs

        # compute the pre-posterior data space
        if p == 1:
            U_data = HX / np.linalg.norm(HX)
        elif p > 1:
            from scipy.linalg import svd

            U_data: Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat] = svd(
                HX, full_matrices=False, compute_uv=True, lapack_driver="gesdd"
            )[0]
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

        # Compute eig(P*HQHT*P) approximately by svd(P*HZ)
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
        # TODO
        cR_all = np.zeros((self.n_internal_loops), dtype=np.float64)

        # Call a different internal iteration routine to solve the linear system.
        if self.is_direct_solve:
            self.loginfo("Use direct solver for saddle-point (cokrigging) system")
            # Construct HQ directly
            HQ: NDArrayFloat = np.dot(HZ, Z.T)

            def internal_iteration(
                _inflation,
            ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
                return self.internal_iteration_direct(HZ, HX, HQ, b, _inflation)

        else:
            self.loginfo(
                "Use Krylov subspace iterative solver "
                "for saddle-point (cokrigging) system"
            )
            self.invA_as_linop = self.get_invA_as_linop(HZ, HX, self.cov_obs)

            def internal_iteration(
                _inflation,
            ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
                return self.internal_iteration_krylov_subspace(
                    HZ, HX, Z, b, self.invA_as_linop, _inflation
                )

        # if LM_smax, LM_smin defined and solution violates them, LM_eval[i] "
        # "becomes True

        # Single worker (no multi-processing)
        if self.max_workers == 1 or self.n_internal_loops == 1:
            for lm_it, inflation in enumerate(self.cov_obs_inflation_factors):
                (
                    xi_all[:, lm_it, None],
                    s_hat_all[:, lm_it, None],
                    beta_all[:, lm_it, None],
                ) = internal_iteration(inflation)
        # Multi-processing enabled -> only relevant if LM is ON.
        else:
            # Function that returns a generator so we can repeat the parameters
            # used by
            def get_internal_loop_params(p: Any) -> Callable:
                def _f() -> Generator[Any, None, None]:
                    while True:
                        yield (p)

                return _f

            # perform the calculations in parallel with multiprocessing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                if self.is_direct_solve:
                    results: Iterator[
                        Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
                    ] = executor.map(
                        self.internal_iteration_direct,
                        get_internal_loop_params(HZ)(),
                        get_internal_loop_params(HX)(),
                        get_internal_loop_params(HQ)(),
                        get_internal_loop_params(b)(),
                        self.cov_obs_inflation_factors,
                    )
                else:
                    results = executor.map(
                        self.internal_iteration_krylov_subspace,
                        get_internal_loop_params(HZ)(),
                        get_internal_loop_params(HX)(),
                        get_internal_loop_params(Z)(),
                        get_internal_loop_params(b)(),
                        get_internal_loop_params(self.invA_as_linop)(),
                        self.cov_obs_inflation_factors,
                    )

            # unpack the results
            for lm_it, res in enumerate(results):
                (
                    xi_all[:, lm_it, None],
                    s_hat_all[:, lm_it, None],
                    beta_all[:, lm_it, None],
                ) = res

        # 6.67 in kitanidisIntroductionGeostatisticsApplications1997.
        Q2_all[:] = np.dot(b[:n].T, xi_all) / (n - p)

        # if lm is off, this will always be True
        is_valid_s_hat = np.invert(self.is_s_violate_lm_bounds(s_hat_all))

        # TODO: what if no valid s_hat ??? -> consider the best obj-fun
        # or maybe change the valid and use clip instead ???

        # keep only valid s vectors and associated inflation factors
        # note that this has no effects if LM is off
        beta_hat_all = beta_all[:, is_valid_s_hat]
        s_hat_all = s_hat_all[:, is_valid_s_hat]
        valid_inflations = self.cov_obs_inflation_factors[is_valid_s_hat]

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

        if np.shape(simul_obs_all) != (self.obs.size, self.n_internal_loops):
            raise ValueError("np.size(simul_obs_all,1) != n_internal_loops")

        self.loginfo("%d objective value evaluations" % self.n_internal_loops)

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
        HQ: NDArrayFloat,
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
        s_hat = (np.dot(self.drift.mat, beta) + np.dot(HQ.T, xi)).reshape(-1, 1)

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
        x = self._solve_iterative_subspace_krylov(
            self.get_A_as_linop(HX, HZ, inflation),  # linear op for matvec and matmat
            b,
            invA_as_linop,  # preconditioner
            restart=50,
            atol=1e-10,
            rtol=1e-10,
            maxiter=self.s_dim,
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
        self.loginfo("-- Number of iterations for gmres %g" % (callback.itercount()))
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
            d=20,
            single_pass=False,
            keep_neg_eigvals=False,
        )

        logging.info(
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

        # This inflation factor will be updated later
        return InvALinOp(
            HX, HZZTHT_eig_vects, HZZTHT_eig_vals, cov_obs, n_obs, beta_dim
        )

    def line_search(self, s_cur, s_past):
        n_internal_loops = self.max_it_lm
        m = self.s_dim

        s_hat_all = np.zeros((m, n_internal_loops), "d")
        # need to remove delta = 0 and 1
        delta = np.linspace(-0.1, 1.1, n_internal_loops)

        for i in range(n_internal_loops):
            s_hat_all[:, i : i + 1] = delta[i] * s_past + (1.0 - delta[i]) * s_cur

        self.loginfo("evaluate linesearch solutions")
        simul_obs_all = self.forward_model(s_hat_all)

        # will change assert to valueerror
        assert np.size(simul_obs_all, 1) == n_internal_loops
        best_obj = 1.0e20

        for i in range(n_internal_loops):
            obj = self.objective_function_no_beta(
                s_hat_all[:, i : i + 1], simul_obs_all[:, i : i + 1]
            )

            if obj < best_obj:
                self.loginfo("%d-th solution obj %e (delta %f)" % (i, obj, delta[i]))
                s_hat = s_hat_all[:, i : i + 1]
                simul_obs_new = simul_obs_all[:, i : i + 1]
                best_obj = obj

        return s_hat, simul_obs_new, best_obj

    def display_objfun(
        self,
        loss_ls: float,
        n_obs: int,
        rmse: float,
        n_rmse: float,
        n_iter: int = 0,
        obj: Optional[float] = None,
        res: Optional[float] = None,
        is_beta: bool = True,
    ) -> None:
        if n_iter != 0:
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
        if res is not None:
            dat[f"relative L2-norm diff btw sol {n_iter:d} and sol {n_iter + 1:d}"] = (
                res
            )

        maxlen = max([len(k) for k in dat]) + 1
        for k, v in dat.items():
            self.loginfo(f"** {k:<{maxlen}} : {v:.3e}")

    def gauss_newton(self) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, int]:
        """
        Gauss-newton iteration
        """

        s_init = self.s_init
        self.maxiter

        res = 1.0

        self.loginfo("##### 2. Start PCGA Inversion #####")
        self.loginfo("-- evaluate initial solution")

        # s_init has shape (s_dim, 1)
        simul_obs_init = self.forward_model(s_init)
        self.istate.simul_obs_best = simul_obs_init

        self.simul_obs_init = simul_obs_init
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
        if self.callback is not None:
            self.callback(self, s_hat=s_cur, simul_obs=simul_obs_cur, n_iter=0)

        for n_iter in range(self.maxiter):
            start = time()

            # TODO: make a loop for that
            self.loginfo("")
            self.loginfo(f"***** Iteration {n_iter + 1} ******")
            s_cur, beta_cur, simul_obs_cur, inflation_cur, obj, Q2, cR = (
                self.linear_iteration(s_past, simul_obs_cur)
            )

            # Call the optional callback at the end of each linear iteration so some
            # intermediate solver states could be saved
            # TODO: move somewhere else ???
            if self.callback is not None:
                self.callback(self, s_hat=s_cur, simul_obs=simul_obs_cur, n_iter=n_iter)

            # save the objective function, inflation factors etc.
            self.istate.obj_seq.append(obj)
            self.istate.inflation_seq.append(inflation_cur)
            self.istate.Q2_seq.append(Q2)
            self.istate.cR_seq.append(cR)

            self.loginfo(
                "- Geostat. inversion at iteration %d is %g sec"
                % ((n_iter + 1), round(time() - start))
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
                    if obj < obj_old:
                        self.istate.s_best = s_cur
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

            res = float(np.linalg.norm(s_past - s_cur) / np.linalg.norm(s_past))
            residuals = (simul_obs_cur.T - self.obs).T
            loss_ls = self.objective_function_ls(simul_obs_cur).item()
            rmse = self.rmse(residuals, False).item()
            n_rmse = self.rmse(residuals, True).item()
            obj = self.objective_function(s_cur, beta_cur, simul_obs_cur).item()

            self.display_objfun(
                loss_ls, simul_obs_init.size, rmse, n_rmse, n_iter, obj=obj, res=res
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
                    self.istate.status = "CONVERGENCE: F_<=_FACTR*EPSMCH"
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

            # TODO: A posteriori estimation should match the best estimation

            # need to use HZ and HX here !
            # assume linesearch result close to the current solution
            start = time()
            if self.is_direct_solve:
                self.loginfo(
                    "start direct posterior variance computation "
                    "- this option works for O(nobs) ~ 100"
                )
                self.post_diagv = self._compute_post_cov_diag(
                    self.HZ, self.HX, self.cov_obs, inflation_cur, is_direct_solve=True
                )
            else:
                self.loginfo("start posterior variance computation")
                self.post_diagv = self._compute_post_cov_diag(
                    self.HZ, self.HX, self.cov_obs, inflation_cur, is_direct_solve=False
                )
            self.loginfo(f"posterior diag. computed in {(time() - start):.3e} s")
            # if self.iter_save:
            #     np.savetxt("./postv.txt", self.post_diagv)

        # return s_cur, beta_cur, simul_obs, iter_cur
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
            f"- Final cR = {self.istate.best_cR:.3e} (should be as small as possible.)"
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
        return cov_obs.todense() + inflation * np.dot(HZ, HZ.T)

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

    def _get_post_cov_build_inputs(
        self,
        HZ: NDArrayFloat,
        HX: NDArrayFloat,
        cov_obs: covmats.CovarianceMatrix,
        inflation: float,
        is_direct_solve: bool,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:

        Z = np.sqrt(self.Q.eig_vals).T * self.Q.eig_vects

        # [HQ, X^{T}]  shape (N_obs + N_p, N_s)
        b_all = np.vstack([np.dot(HZ, Z.T), self.drift.mat.T])
        p = self.drift.beta_dim

        if is_direct_solve:
            # Use cholesky factorization to solve the system
            LA = self.build_cholesky(self.get_psi(HZ, cov_obs, inflation), HX)
            # shape (n_pc + p, ns) => solve only once
            invAb_all: NDArrayFloat = PCGA.solve_cholesky(LA, b_all, p)
        else:
            # Use iterative Krylov subspace to solve the system
            # (n_pc + p, ns) => solve only once
            # Solving ns Ax = b is too long so we simply use the preconditioner and it
            # works very fine.
            invAb_all = self.get_invA_as_linop(HZ, HX, cov_obs) @ b_all

        return b_all, invAb_all

    def _compute_post_cov_diag(
        self,
        HZ: NDArrayFloat,
        HX: NDArrayFloat,
        cov_obs: covmats.CovarianceMatrix,
        inflation: float,
        is_direct_solve: bool,
    ) -> NDArrayFloat:
        """Computing posterior diagonal entries using cholesky/it krylov subspace."""
        b_all, invAb_all = self._get_post_cov_build_inputs(
            HZ=HZ,
            HX=HX,
            cov_obs=cov_obs,
            inflation=inflation,
            is_direct_solve=is_direct_solve,
        )
        return (self.prior_s_var - np.sum(b_all * invAb_all, axis=0)).reshape(-1, 1)

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
            _description_, by default None
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
        if is_direct_solve is None:
            _is_direct_solve: bool = self.is_direct_solve
        else:
            _is_direct_solve = is_direct_solve

        b_all, invAb_all = self._get_post_cov_build_inputs(
            HZ=self.HZ,
            HX=self.HX,
            cov_obs=self.cov_obs,
            inflation=_inflation,
            is_direct_solve=_is_direct_solve,
        )
        return self.Q.todense() - b_all.T @ invAb_all

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
            _description_, by default None
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
        if is_direct_solve is None:
            _is_direct_solve: bool = self.is_direct_solve
        else:
            _is_direct_solve = is_direct_solve

        b_all, invAb_all = self._get_post_cov_build_inputs(
            HZ=self.HZ,
            HX=self.HX,
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
                return Q @ x - b_all.T @ (invAb_all @ x)

        return covmats.CovViaEigenFactorization(
            covmats.get_linop_eigen_factorization(
                _op(), size=self.s_dim, n_pc=_n_pc, random_state=random_state
            )
        )
