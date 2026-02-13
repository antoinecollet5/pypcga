"""
Purpose
=======

**pypcga** is an open-source, and object-oriented library that provides
a user friendly implementation of the Principal Component Geostatistical Approach
for large-scale inversion.

The original code has been written by Jonghyun Harry Lee.

See: https://github.com/jonghyunharrylee/pyPCGA

The following functionalities are directly provided on module-level.

Classes
=======

.. autosummary::
   :toctree: _autosummary

   PCGA

Utilitary functions
===================

.. autosummary::
   :toctree: _autosummary

    ensemble_dot
    ghep
    mgs_stable

"""

from pypcga.__about__ import __version__
from pypcga._pcga import PCGA, InternalState
from pypcga._utils import ensemble_dot, ghep, mgs_stable

__all__ = ["__version__", "PCGA", "InternalState", "ensemble_dot", "ghep", "mgs_stable"]
