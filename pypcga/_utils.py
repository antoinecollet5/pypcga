"""Provide utilities."""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy as sp
from scipy._lib._util import check_random_state
from scipy.sparse.linalg import LinearOperator

NDArrayFloat = npt.NDArray[np.floating]
NDArrayBool = npt.NDArray[np.bool_]


def mgs_stable(
    A: Union[NDArrayFloat, LinearOperator],
    Z: NDArrayFloat,
    verbose=False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """
    Returns QR decomposition of Z with Q*AQ = I.

    Q and R satisfy the following relations in exact arithmetic:

        1. QR    	= Z
        2. Q^*AQ 	= I
        3. Q^*AZ	= R
        4. ZR^{-1}	= Q

    Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
    for computing the A-orthogonal QR factorization

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
            An array, sparse matrix, or LinearOperator representing
            the operation ``A * x``, where A is a real or complex square matrix.
    Z : ndarray
        Matrix of vectors to be A-orthogonalized, shape (n, k) with k <= n.
    logger: Optional[logging.Logger]
        Optional logger to display info. The default is None.

    Returns
    -------
    q : ndarray
            The A-orthogonal vectors
    Aq : ndarray
            The A^{-1}-orthogonal vectors
    r : ndarray
            The r of the QR decomposition


    See Also
    --------
    mgs : Modified Gram-Schmidt without re-orthogonalization
    precholqr  : Based on CholQR


    References
    ----------
    .. [1] A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for
            Generalized Hermitian Eigenvalue Problems with application to computing
            Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885

    .. [2] W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980

    Examples
    --------

    >>> import numpy as np
    >>> A = np.diag(np.arange(1,101))
    >>> Z = np.random.randn(100,10)
    >>> q, Aq, r = mgs_stable(A, Z, verbose = True)

    """

    # Get sizes
    n = np.size(Z, 1)

    # Convert into linear operator
    Aop = sp.sparse.linalg.aslinearoperator(A)

    # Initialize
    Aq = np.zeros_like(Z, dtype="d")
    r = np.zeros((n, n), dtype="d")

    reorth = np.zeros((n,), dtype="d")
    eps = np.finfo(np.float64).eps

    q = np.copy(Z)

    for k in np.arange(n):
        Aq[:, k] = Aop.matvec(q[:, k])
        t = np.sqrt(np.dot(q[:, k].T, Aq[:, k]))

        nach = 1
        u = 0
        while nach:
            u += 1
            for i in np.arange(k):
                s = np.dot(Aq[:, i].T, q[:, k])
                r[i, k] += s
                q[:, k] -= s * q[:, i]

            Aq[:, k] = Aop.matvec(q[:, k])
            tt = np.sqrt(np.dot(q[:, k].T, Aq[:, k]))
            if tt > t * 10.0 * eps and tt < t / 10.0:
                nach = 1
                t = tt
            else:
                nach = 0
                if tt < 10.0 * eps * t:
                    tt = 0.0

        reorth[k] = u
        r[k, k] = tt
        tt = 1.0 / tt if np.abs(tt * eps) > 0.0 else 0.0
        q[:, k] *= tt
        Aq[:, k] *= tt

    if logger is not None:
        # Number of re-orthogonalization passes performed for each column
        logger.info(f"Re-orthogonalization counts per column: {reorth}")

        # Verify Q*R = Y
        logger.info(f"||QR-Y|| is {np.linalg.norm(np.dot(q, r) - Z, 2)}")

        # Verify Q'*A*Q = I
        T = np.dot(q.T, Aq)
        logger.info(f"||Q^TAQ-I|| is {np.linalg.norm(T - np.eye(n, dtype='d'), ord=2)}")

        # verify Q'AY = R
        logger.info(f"||R - Q^TAY|| is {np.linalg.norm(r - np.dot(Aq.T, Z), 2)}")

        # Verify YR^{-1} = Q
        val = np.inf
        try:
            val = np.linalg.norm(np.linalg.solve(r.T, Z.T).T - q, 2)
        except sp.linalg.LinAlgError:
            logger.info("YR^{-1}-Q is singular")
        logger.info(f"||YR^{{-1}}-Q|| is {val}")

    return q, Aq, r


def ghep(
    A: Union[NDArrayFloat, LinearOperator],
    B: Union[NDArrayFloat, LinearOperator],
    Binv: Union[NDArrayFloat, LinearOperator],
    r: int,
    d: int,
    single_pass: bool = True,
    keep_neg_eigvals: bool = False,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Randomized generalized Hermitian eigenvalue problem (EVD) solver.

    Computes a rank-``r`` approximate eigen-decomposition of the (generalized)
    Hermitian eigenvalue problem ``A v = s B v`` using a randomized range-finder,
    following the randomized SVD/EVD approach of Halko, Martinsson and Tropp.

    Parameters
    ----------
    A : Union[NDArrayFloat, LinearOperator]
        Symmetric operator to decompose, A ∈ R^{N x N}.
    B : Union[NDArrayFloat, LinearOperator]
        Symmetric positive-definite operator defining the generalized inner
        product used for the A-orthogonal projection (see :func:`mgs_stable`).
    Binv : Union[NDArrayFloat, LinearOperator]
        Operator (or matrix) applying the inverse of ``B``. Used to map the
        random test matrix into the correct space before sampling.
    r : int
        Desired rank of the approximation.
    d : int
        Oversampling parameter. Typically, ``d`` is chosen to be less than 20
        following the arguments in [3]_. The improvement in the approximation
        error with increasing oversampling is verified in both theory and
        experiment.
    single_pass : bool, optional
        If True, use a single-pass (one matrix application of ``A``) algorithm,
        trading some accuracy for fewer operator applications. If False, a
        second pass through ``A`` is used to improve accuracy. By default True.
    keep_neg_eigvals : bool, optional
        If True, keep negative eigenvalues in the returned decomposition.
        If False (default), only strictly positive eigenvalues (and their
        associated eigenvectors) are returned.
    random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to draw the random test
        matrix ``Omega``. If None, a new default generator is used (results
        will not be reproducible across calls). If an int, a new generator is
        seeded with it. If already a ``Generator``/``RandomState`` instance,
        that instance is used directly. By default None.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        eig_vects : NDArrayFloat
            Approximate eigenvectors, shape (N, k) where k <= r + d depending
            on ``keep_neg_eigvals``.
        eig_vals : NDArrayFloat
            Corresponding eigenvalues as a column vector, shape (k, 1), sorted
            in decreasing order.

    References
    ----------
    .. [1] Halko N, Martinsson PG, Tropp JA. Finding structure with randomness:
       probabilistic algorithms for constructing approximate matrix
       decompositions. SIAM Review 2011; 53(2):217-288.
    .. [2] Bui-Thanh T, Burstedde C, Ghattas O, Martin J, Stadler G, Wilcox LC.
       Extreme-scale UQ for Bayesian inverse problems governed by PDEs. In
       Proceedings of the International Conference on High Performance
       Computing, Networking, Storage and Analysis. IEEE Computer Society
       Press: Portland, OR, 2012.
    .. [3] Liberty E, Woolfe F, Martinsson PG, Rokhlin V, Tygert M. Randomized
       algorithms for the low-rank approximation of matrices. Proceedings of
       the National Academy of Sciences 2007; 104(51):20167-20172.
    """
    # Initiate random matrix
    _random_state = check_random_state(random_state)
    Omega = _random_state.normal(loc=0.0, scale=1.0, size=(A.shape[1], r + d))
    # Sample column space
    Y = Binv @ A @ Omega
    # Orthogonalize column samples alternatively msg_stable
    Qy = mgs_stable(B, Y, verbose=False)[0]
    # SVD of k × k compressed row sample matrix
    if single_pass:
        s, Z = sp.linalg.eigh(Qy.T @ Y @ sp.linalg.pinv(Qy.T @ Omega), lower=True)
    else:
        s, Z = sp.linalg.eigh(Qy.T @ A.T @ Qy, lower=True)

    _sort = np.argsort(s)[::-1]

    if not keep_neg_eigvals:
        _sort = _sort[s[_sort] > 0.0]

    # V, s, U
    return (Qy @ Z)[:, _sort], s[_sort].reshape(-1, 1)


def ensemble_dot(X1: NDArrayFloat, X2: NDArrayFloat) -> NDArrayFloat:
    r"""
    Return the dot products for multiple vectors.

    Parameters
    ----------
    X1 : NDArrayFloat
        First ensemble of vectors with shape $(N_{\mathrm{s}}, N_{\mathrm{e}})$.
    X2 : NDArrayFloat
        First ensemble of vectors with shape $(N_{\mathrm{s}}, N_{\mathrm{e}})$.

    Returns
    -------
    NDArrayFloat
        $N_{\mathrm{e}}$ dot products as a 1D vector.
    """
    # same as np.sum(X1 * X2, axis=0,keepdims=False) but faster
    return np.einsum("ij,ij->j", X1, X2)
