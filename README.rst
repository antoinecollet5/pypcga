=======
pyPCGA
=======

|License| |Stars| |Python| |PyPI| |Downloads| |Build Status| |Documentation Status| |Coverage| |Codacy| |Precommit: enabled| |Ruff| |ty| |DOI|

🐍 A python implementation of the Principal Component Geostatistical Approach (PCGA) for large scale inversion.

**The complete and up to date documentation can be found here**: https://pypcga.readthedocs.io.

======================
📖 Courses and theory
======================

- Please check out the `UH CEE696 course on data assimilation <https://www2.hawaii.edu/~jonghyun/classes/S21/CEE696/>`_
- As well as the theory description (add link).

===============
🚀 Quick start
===============

To install `pypcga`, the easiest way is through `pip`:

.. code-block::

    pip install pypcga

Or alternatively using `conda`

.. code-block::

    conda install pypcga

You might also clone the repository and install from source

.. code-block::

    pip install -e .

🏗️ Complete example with supporting paper coming Q1 2026.

Implemented features:

- Direct inversion with Cholesky (practical up to 100 obs. TODO).
- Exact preconditioner construction (inverse of cokriging/saddle-point matrix) using generalized eigendecomposition [Lee et al., WRR 2016, Saibaba et al, NLAA 2016]

- Fast hyperparameter tuning and predictive model validation using cR/Q2 criteria [Kitanidis, Math Geol 1991] ([Lee et al., 2021 in preparation])

- Fast posterior variance/std computation using exact preconditioner

🏗️ Complete example with supporting paper coming Q1 2026.


# Example Notebooks

1D linear inversion example below will be helpful to understand how pyPCGA can be implemented. Please check Google Colab examples.

- `1D linear inversion example <https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/pumping_history_identification/linear_inverse_problem_pumping_history_identification.ipynb>`_ (from Stanford 362G course) `Google Colab example <https://colab.research.google.com/drive/13lpxTYgNxOc1gYm2bMvIaTpGFejTVX0r?usp=sharing>`_

- `1D nonlinear inversion example <https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/pumping_history_identification/nonlinear_inverse_problem_pumping_history_identification.ipynb>`_ (from Stanford 362G course) `Google Colab example <https://colab.research.google.com/drive/1NPX-q_os5_kVAyFBDOhX_BJJMWUti0u_?usp=sharing>`_

- `Hydraulic conductivity estimation example using USGS-FloPy (MODFLOW) <https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/modflow_flopy/inversion_modflow.ipynb>`_ [Lee and Kitanidis, 2014] `Google Colab example <https://colab.research.google.com/drive/1djVDZNjh390czXlzBbu7FvRQne9mf8SP?usp=sharing>`_

- `Tracer tomography example using Crunch <https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/tracer_tomography_ade_crunch/inversion_example_advection_diffusion_crunchtope.ipynb>`_ (with Mahta Ansari from UIUC Druhan Lab)

- `Bathymetry estimation example using STWAVE <https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/stwave_duck/inversion_stwave.ipynb>`_ (with USACE-ERDC-CHL)

- `Permeability estimation example using TOUGH2 <https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/tough_heat/joint_inversion_example_tough.ipynb>`_ (with Amalia Kokkianki, USFCA)

- `Electrical conductivity estimation example using magnetotelluric (MT) survey with MARE2DEM <https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/mare2dem_MT/inversion_mare2dem.ipynb>`_ (with Niels Grobbe, UHM)

- `DNAPL plume estimation using hydraulic head, self-potential (SP) and partitioning tracer data <https://github.com/XueyuanK/DNAPL_Inv>`_ (with Xueyuan Kang et al.)

- `ERT example using E4D <https://github.com/jonghyunharrylee/pyPCGA/tree/master/examples/ERT_E4D>`_ will be completed soon.

==================
📚 Main References
==================

- J Lee, H Yoon, PK Kitanidis, CJ Werth, AJ Valocchi, "Scalable subsurface inverse modeling of huge data sets with an application to tracer concentration breakthrough data from magnetic resonance imaging", Water Resources Research 52 (7), 5213-5231

- AK Saibaba, J Lee, PK Kitanidis, Randomized algorithms for generalized Hermitian eigenvalue problems with application to computing Karhunen–Loève expansion, Numerical Linear Algebra with Applications 23 (2), 314-339

- J Lee, PK Kitanidis, "Large‐scale hydraulic tomography and joint inversion of head and tracer data using the Principal Component Geostatistical Approach (PCGA)", WRR 50 (7), 5410-5427

- PK Kitanidis, J Lee, Principal Component Geostatistical Approach for large‐dimensional inverse problems, WRR 50 (7), 5428-5443

================
💻 Applications
================

- T. Kadeethum, D. O'Malley, JN Fuhg, Y. Choi, J. Lee, HS Viswanathan and N. Bouklas, A framework for data-driven solution and parameter estimation of PDEs using conditional generative adversarial networks, Nature Computational Science, 819–829, 2021

- J Lee, H Ghorbanidehno, M Farthing, T. Hesser, EF Darve, and PK Kitanidis, Riverine bathymetry imaging with indirect observations, Water Resources Research, 54(5): 3704-3727, 2018

- J Lee, A Kokkinaki, PK Kitanidis, Fast large-scale joint inversion for deep aquifer characterization using pressure and heat tracer measurements, Transport in Porous Media, 123(3): 533-543, 2018

- PK Kang, J Lee, X Fu, S Lee, PK Kitanidis, J Ruben, Improved Characterization of Heterogeneous Permeability in Saline Aquifers from Transient Pressure Data during Freshwater Injection, Water Resources Research, 53(5): 4444-458, 2017

- S. Fakhreddine, J Lee, PK Kitanidis, S Fendorf, M Rolle, Imaging Geochemical Heterogeneities Using Inverse Reactive Transport Modeling: an Example Relevant for Characterizing Arsenic Mobilization and Distribution, Advances in Water Resources, 88: 186-197, 2016

===========
🔑 Credits
===========

**pypcga** is based on Lee et al. [2016] and currently used for Stanford-USACE ERDC project led by EF Darve and PK Kitanidis and NSF EPSCoR `Ike Wai project.

Code contributors include:

- `Jonghyun Harry Lee <https://github.com/jonghyunharrylee>`_
- `Matthew Farthing <https://github.com/mfarthin>`_
- `Ty Hesser <https://github.com/Mahtaw>`_
- `Antoine COLLET <https://github.com/antoinecollet5>`_

===========
🔑 License
===========

This project is released under the **BSD 3-Clause License**.

Copyright (c) 2026, Antoine COLLET. All rights reserved.

For more details, see the `LICENSE <https://github.com/antoinecollet5/pypcga/blob/master/LICENSE>`_ file included in this repository.

==============
⚠️ Disclaimer
==============

This software is provided "as is", without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a particular purpose,
or non-infringement. In no event shall the authors or copyright holders be liable for
any claim, damages, or other liability, whether in an action of contract, tort,
or otherwise, arising from, out of, or in connection with the software or the use
or other dealings in the software.

By using this software, you agree to accept full responsibility for any consequences,
and you waive any claims against the authors or contributors.

==========
📧 Contact
==========

For questions, suggestions, or contributions, you can reach out via:

- Email: antoinecollet5@gmail.com
- GitHub: https://github.com/antoinecollet5/pypcga

We welcome contributions!

* Free software: SPDX-License-Identifier: BSD-3-Clause


.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/antoinecollet5/pypcga/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/antoinecollet5/pypcga.svg?style=social&label=Star&maxAge=2592000
    :target: https://github.com/antoinecollet5/pypcga/stargazers
    :alt: Stars

.. |Python| image:: https://img.shields.io/pypi/pyversions/pypcga.svg
    :target: https://pypi.org/pypi/pypcga
    :alt: Python

.. |PyPI| image:: https://img.shields.io/pypi/v/pypcga.svg
    :target: https://pypi.org/pypi/pypcga
    :alt: PyPI

.. |Downloads| image:: https://static.pepy.tech/badge/pypcga
    :target: https://pepy.tech/project/pypcga
    :alt: Downoads

.. |Build Status| image:: https://github.com/antoinecollet5/pypcga/actions/workflows/main.yml/badge.svg
    :target: https://github.com/antoinecollet5/pypcga/actions/workflows/main.yml
    :alt: Build Status

.. |Documentation Status| image:: https://readthedocs.org/projects/pypcga/badge/?version=latest
    :target: https://pypcga.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Coverage| image:: https://codecov.io/gh/antoinecollet5/pypcga/graph/badge.svg?token=WY1765AKTB
    :target: https://codecov.io/gh/antoinecollet5/pypcga
    :alt: Coverage

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/66f245ab3cb043d7bb8987cf5989d469
    :target: https://app.codacy.com/gh/antoinecollet5/pypcga/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
    :alt: codacy

.. |Precommit: enabled| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |ty| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json
    :target: https://github.com/astral-sh/ty
    :alt: Checked with ty

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11384588.svg
   :target: https://doi.org/10.5281/zenodo.11384588
