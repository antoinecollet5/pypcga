==============
Changelog
==============

0.3.1 (2026-11-22)
------------------

Bug fixes
^^^^^^^^^

* **Levenberg-Marquardt with** ``is_direct_solve=True``: ``get_psi`` applied the
  inflation factor to ``HQH^T`` instead of ``R`` (``alpha HQH^T + R`` instead of
  ``HQH^T + alpha R``), so the direct and Krylov solvers returned different LM
  candidates. Both now solve the same system.
* The Krylov preconditioner and the Krylov posterior variance ignored the LM
  inflation factor (``InvALinOp.inflation`` was never updated), giving a wrong
  posterior variance whenever ``best_inflation != 1``.
* ``is_save_jac=False`` crashed at the first posterior-variance computation.
* ``get_dense_post_cov``/``get_eigen_post_cov`` used the Jacobian of the *last*
  linearization, which is not the one of ``s_best`` when the last iteration was
  rejected: ``diag(get_dense_post_cov())`` did not match ``post_diagv``.
* ``line_search`` evaluated ``max_it_lm`` points instead of ``max_it_ls``, could
  raise ``UnboundLocalError`` when all objectives were NaN, and left a stale
  ``beta`` (now recomputed with the new ``get_optimal_beta``).
* ``max_workers`` was ``max(max_workers_lm, max_it_lm)`` (user limit ignored);
  it is now ``min(...)``.
* Calling ``run()`` twice accumulated the objective/Q2/cR sequences of both runs
  and mis-indexed the "best" values. The internal state is now reset.
* ``rmse(..., is_normalized=True)`` returned an (Ne, Ne) matrix instead of Ne
  values.
* ``Q2_best``/``best_cR``/``best_inflation`` silently returned the values of the
  last (rejected) iteration when no iteration improved on the initial guess;
  they now return NaN.
* Division by zero in the relative model-change criterion when ``s_init = 0``.
* Wrong ``ftarget`` status message (``F_<=_FACTR*EPSMCH`` -> ``F_<=_FTARGET``),
  missing log header for the first iteration, ``ILDE`` typo, log through the root
  logger in ``get_invA_as_linop``, ``ftol``/``restol`` displayed as ``0.0``.
* ``ghep``: the compressed matrix is symmetrized before ``eigh`` (its upper
  triangle used to be silently discarded), and ``Binv @ A @ Omega`` no longer forms
  the N x N product ``Binv @ A`` when dense matrices are given.
* GMRES non-convergence is now reported (warning + log) instead of being ignored.

Improvements
^^^^^^^^^^^^

* Posterior (co)variance computed through a low-rank formulation
  ``Q - W^T K W`` with ``K`` of size ``(n_pc + p)``: only ``n_pc + p`` right-hand
  sides instead of ``s_dim`` (about 10x faster on a 40 000-cell test case), and the
  preconditioner built during the iteration is reused instead of recomputed.
* The dense ``HQ`` matrix (``n_obs x s_dim``) is no longer built in the direct
  solver.
* ``InvALinOp`` caches the inflation-dependent quantities and uses 2 instead of 3
  applications of ``Psi^{-1}`` per product; ``with_inflation`` gives thread-safe
  copies.
* LM candidates are solved with a ``ThreadPoolExecutor`` instead of a
  ``ProcessPoolExecutor``: nothing needs to be picklable anymore (lambdas and
  closures are accepted as ``forward_model``) and ``self`` is not pickled per task.
* Vectorized Gram-Schmidt projections in ``mgs_stable`` and line-search ensemble.
* ``scipy._lib._util.check_random_state`` (private SciPy API) replaced by
  ``pypcga._utils.check_random_state``.
* Examples rewritten as `marimo <https://marimo.io>`_ notebooks (plain ``.py``
  files) instead of Jupyter notebooks.

API changes
^^^^^^^^^^^

* ``internal_iteration_direct(HZ, HX, HQ, b, inflation)`` now takes ``Z`` instead
  of ``HQ`` (same signature as ``internal_iteration_krylov_subspace``).
* ``jac_mat`` no longer writes ``istate.HX/HZ/Hs`` directly; they now hold the
  products at the best solution.
* ``_get_post_cov_build_inputs`` replaced by ``_get_post_cov_lowrank``.

0.3.0 (2026-08-10)
------------------

New features
^^^^^^^^^^^^

* Implemented the ``cR`` predictive model-checking statistic (Kitanidis, 1991),
  which was previously a placeholder always returning ``NaN``. The statistic is
  normalized by the mean observation-error variance so that, like ``Q2``, it
  remains close to 1.0 for a well-specified covariance model regardless of the
  physical units of the observations/covariances.
* ``Q2`` and ``cR`` are now displayed at the end of every Gauss-Newton
  iteration (previously only in the final summary), making it easier to watch
  covariance-model diagnostics evolve during an inversion.

Bug fixes
^^^^^^^^^

* Fixed a shared, non-reproducible default random state: ``random_state`` was
  previously instantiated once at function-definition time and shared/mutated
  across every ``PCGA`` instance created without an explicit seed.
* Fixed an ``AttributeError`` raised at construction time whenever
  ``is_direct_solve=True`` and ``n_obs > 100`` (the drift matrix was read
  before being assigned).
* Fixed ``get_eigen_post_cov`` silently ignoring its resolved random state and
  falling back to an unseeded generator instead.
* Fixed ``ghep``'s randomized eigensolver using a hardcoded seed instead of a
  configurable ``random_state``.
* Fixed an incorrect shape check in the internal Levenberg-Marquardt loop that
  could raise a spurious ``ValueError`` whenever some (but not all) LM
  candidates were filtered out by the prescribed solution bounds -- the
  normal, intended case.
* Added a fallback for the edge case where *every* Levenberg-Marquardt
  candidate violates the prescribed bounds, instead of silently proceeding
  with empty arrays and crashing downstream.
* Fixed a silent index mismatch between ``Q2``/``cR`` and the filtered LM
  candidate arrays, which could report the wrong candidate's diagnostics
  whenever bound-filtering dropped at least one candidate.
* Fixed ``objective_function_no_beta_new``, which crashed for drift dimension
  greater than 1 (``np.atleast_2d`` returned a row vector instead of the
  required column vector) and had a related shape issue for drift dimension 1.
* Fixed broken ``logger.info(...)`` calls in ``mgs_stable`` that used
  ``%``-style lazy formatting without the required placeholders, silently
  dropping every logged diagnostic value.

Improvements
^^^^^^^^^^^^

* Consolidated mutable solver-run state (``HX``, ``HZ``, ``Hs``,
  ``invA_as_linop``, ``simul_obs_init``) into ``PCGA.istate``
  (``InternalState``) instead of scattering it across separate instance
  attributes.
* Extracted the duplicated callback-invocation logic and the per-iteration
  Gauss-Newton step body into dedicated, independently testable methods.
* Extracted and vectorized the finite-difference perturbation-ensemble
  construction used for Jacobian estimation, removing a per-column Python
  loop.
* Expanded and corrected docstrings throughout ``_pcga.py`` and ``_utils.py``.
* Added a full test suite reaching 100% statement coverage.

0.2.0 (2026-02-19)
------------------
* First release on PyPI.
