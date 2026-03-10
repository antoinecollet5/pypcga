import logging

import covmats
import numpy as np
import pypcga
import pytest
import scipy as sp
from pyrtid.utils.types import NDArrayFloat


def forward_model(x) -> NDArrayFloat:
    return sp.ndimage.gaussian_filter(4.0 * x**2, sigma=2.0)


def sample_d(d, percent_of_values: float) -> NDArrayFloat:
    return d.ravel("F")[:: int(d.size / (percent_of_values * 1000))]


@pytest.mark.parametrize(
    "is_direct_solve,is_lm",
    (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ),
)
def test_integration(
    is_direct_solve: bool,
    is_lm: bool,
) -> None:

    # Create loggers
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    pcga_logger = logging.getLogger("PCGA")
    pcga_logger.setLevel(logging.INFO)
    main_logger.info("This is the main logger")
    pcga_logger.info("This is the PCGA logger")

    # Create a prior
    cov_prior = covmats.CovViaSparsePrecisionCholesky(
        covmats.load_precision_example_4225x_SCF()
    )
    cov_prior

    # Square domain (65, 65)
    nx = ny = int(np.sqrt(cov_prior.shape[0]))

    # Non conditional simulation -> change the random state to obtain a different field
    simu_ = cov_prior.sample_mvnormal(shape=(1,), random_state=2026).reshape(ny, nx).T
    mean = 50.0
    s_ref = np.abs(simu_ + mean)

    # Define a very simple non linear forward model (just a simple static smoohting, no
    # time dependence here) and produce a reference field from which observations will
    # be sampled.
    d_ref = forward_model(s_ref)

    s_init = np.abs(
        cov_prior.sample_mvnormal(shape=(1,), random_state=15653).reshape(ny, nx).T
    )

    # Create a sampling function that uses 5% of the available data
    _percent_of_values = 0.05

    obs = sample_d(d_ref, _percent_of_values)

    # 10% error on the observations (diagonal covariance matrix = independent errors)
    cov_obs = covmats.CovViaDiagonal(
        (np.ones(obs.shape) * (np.max(obs) - np.min(obs)) * 0.1) ** 2
    )

    # Factorize the priori covariance matri
    eig_mat = covmats.eigen_factorize_cov_mat(cov_prior, n_pc=50)

    assert eig_mat.n_pts == 4225

    # Create a forward model
    def forward_model_wrapper(s_ensemble, *args, **kargs) -> NDArrayFloat:
        d_pred = np.zeros((obs.size, s_ensemble.shape[1]))
        for i in range(s_ensemble.shape[1]):
            # use preconditionning
            res = forward_model(s_ensemble[:, i].reshape(nx, ny, order="F"))
            d_pred[:, i] = sample_d(res, _percent_of_values)
        return d_pred

    # test
    s_ens = np.vstack([s_ref.ravel("F"), s_init.ravel("F")]).T
    assert s_ens.shape == (nx * ny, 2)

    d_pred = forward_model_wrapper(s_ens)
    assert d_pred.shape == (obs.size, 2)

    np.testing.assert_almost_equal(d_pred[:, 0], obs)

    # Perturb observations
    obs_perturb = (
        obs + cov_obs.sample_mvnormal([1], random_state=np.random.default_rng(2151))[0]
    )

    # Create the PCGA instance
    solver = pypcga.PCGA(
        s_init=s_init.ravel("F"),
        obs=obs_perturb,
        cov_obs=cov_obs,
        forward_model=forward_model_wrapper,
        Q=eig_mat,
        maxiter=5,
        is_lm=is_lm,
        is_direct_solve=is_direct_solve,
        prior_s_var=None,
        max_it_lm=1,
        random_state=2026,
        logger=pcga_logger,
    )

    assert solver.s_dim == nx * ny
    assert solver.d_dim == obs.size

    # Solve
    s_hat, simul_obs, post_diagv, iter_best = solver.run()

    _ = solver.get_dense_post_cov()
    _ = solver.get_eigen_post_cov(n_pc=25)
    post_cov_50_pc = solver.get_eigen_post_cov(n_pc=50)
    # Second run with parameters
    post_cov_50_pc = solver.get_eigen_post_cov(
        n_pc=50, is_direct_solve=is_direct_solve, random_state=4678
    )

    # make 200 posterior realizations => we sample from post_cov_50_pc and add the
    # results to s_hat, our inversed parameter vector
    post_samples = (
        s_hat.T
        + post_cov_50_pc.sample_mvnormal(shape=(200,), random_state=solver.random_state)
    ).T
    post_samples.shape

    # Create a covariance matrix from the ensemble
    _ = covmats.CovViaEnsemble(post_samples.T)
