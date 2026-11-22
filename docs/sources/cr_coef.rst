Principal Component Geostatistical Approach (PCGA): :math:`c_R` coefficient
===========================================================================

In the Principal Component Geostatistical Approach (PCGA) used in inverse
problems (e.g., by Peter Kitanidis and collaborators), the :math:`c_R`
coefficient appears in the :math:`c_R/Q_2` diagnostic criterion used to
evaluate and tune covariance or variogram parameters such as correlation
length, sill, nugget, and measurement error.

This diagnostic originates from cross–validation methods described in
Kitanidis (1997), *Introduction to Geostatistics*.

Definition of :math:`c_R`
-------------------------

The :math:`c_R` coefficient measures the normalized magnitude of the
cross-validation residuals.

For each observation :math:`i`:

.. math::

   \epsilon_i =
   \frac{z_i - \hat{z}_{-i}}{\sigma_{-i}}

where

- :math:`z_i` is the observed value
- :math:`\hat{z}_{-i}` is the estimate at location :math:`i`
  obtained **without using observation** :math:`i`
  (leave-one-out kriging)
- :math:`\sigma_{-i}` is the kriging standard deviation
  at location :math:`i`

The :math:`c_R` statistic is defined as

.. math::

   c_R = \frac{1}{n} \sum_{i=1}^{n} |\epsilon_i|

or equivalently

.. math::

   c_R =
   \frac{1}{n}\sum_{i=1}^{n}
   \frac{|z_i-\hat z_{-i}|}{\sigma_{-i}}

Interpretation
--------------

The coefficient :math:`c_R` measures how large the prediction errors are
relative to their expected uncertainty.

- :math:`c_R \approx 1` → model is statistically consistent
- :math:`c_R > 1` → errors are larger than predicted
  (covariance underestimates variability)
- :math:`c_R < 1` → errors are smaller than predicted
  (covariance overestimates variability)

Thus it is a goodness-of-fit diagnostic for the covariance model.

Relation with :math:`Q_2`
-------------------------

The pair :math:`c_R / Q_2` is commonly used to assess covariance models.

The statistic :math:`Q_2` is defined as

.. math::

   Q_2 = \frac{1}{n}\sum_{i=1}^{n}\epsilon_i^2

Interpretation:

- :math:`Q_2 \approx 1` → predicted variance is correct
- :math:`Q_2 > 1` → predicted variance is too small
- :math:`Q_2 < 1` → predicted variance is too large

Role in PCGA
------------

In PCGA inverse problems, the user must specify prior covariance
parameters such as

- covariance model
- correlation length
- variance
- measurement noise

The :math:`c_R` and :math:`Q_2` diagnostics are used to select these
hyperparameters by

1. performing leave-one-out kriging
2. computing normalized residuals
3. adjusting covariance parameters until

.. math::

   c_R \approx 1, \qquad Q_2 \approx 1

This ensures that the prior covariance model is statistically consistent
with the observed data before running the inversion.

Algorithm to compute :math:`c_R`
--------------------------------

Given observations :math:`z_i` at locations :math:`x_i`:

**Step 1 — Leave-one-out estimation**

For each :math:`i`:

- remove data point :math:`i`
- perform kriging using the remaining :math:`n-1` points
- obtain :math:`\hat z_{-i}` and :math:`\sigma_{-i}`

**Step 2 — Compute normalized residual**

.. math::

   \epsilon_i =
   \frac{z_i-\hat z_{-i}}{\sigma_{-i}}

**Step 3 — Compute statistic**

.. math::

   c_R=\frac{1}{n}\sum_{i=1}^{n}|\epsilon_i|

Typical target values
---------------------

+-------------+-------------------+
| Statistic   | Desired value     |
+=============+===================+
| :math:`Q_1` | :math:`\approx 0` |
+-------------+-------------------+
| :math:`Q_2` | :math:`\approx 1` |
+-------------+-------------------+
| :math:`c_R` | :math:`\approx 1` |
+-------------+-------------------+

Summary
-------

- :math:`c_R` is the **mean absolute normalized cross-validation error**
- it is used to validate covariance models in geostatistics
- in PCGA it helps tune prior covariance parameters.
