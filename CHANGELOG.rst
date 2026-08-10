==============
Changelog
==============

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
