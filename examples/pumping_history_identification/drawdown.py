"""
Groundwater drawdown model for pumping history reconstruction.

Linear forward model of the drawdown observed at a monitoring well located at
(x1, x2) = (6, 8) m, caused by an unknown pumping history s(t) at (0, 0), in a
homogeneous, isotropic and infinite confined aquifer (Theis convolution):

    y = H s          (or y = H exp(s) when working with log-transformed rates)

The (100 x 10001) matrix H, obtained by quadrature of the Theis convolution
integral, is precomputed and stored in ``H.mat``. Aquifer properties (Stanford
CEE362g course): transmissivity T = 0.02 m2/min, storativity S = 0.001.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat

NDArrayFloat = npt.NDArray[np.floating]

#: Default location of the precomputed sensitivity matrix (next to this file).
DEFAULT_H_PATH = Path(__file__).resolve().parent / "H.mat"


@lru_cache(maxsize=4)
def load_H(path: Union[str, Path] = DEFAULT_H_PATH) -> NDArrayFloat:
    """
    Load (once) the forward operator matrix H from a ``.mat`` file.

    Parameters
    ----------
    path : Union[str, Path], optional
        Path to the ``.mat`` file holding the ``H`` variable. By default the
        ``H.mat`` file shipped next to this module.

    Returns
    -------
    NDArrayFloat
        The forward operator H with shape (n_obs, n_params). The returned array
        is cached and read-only.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    H = np.asarray(loadmat(path)["H"], dtype=np.float64)
    H.setflags(write=False)
    return H


class Model:
    """
    Drawdown forward model ``simul_obs = H @ s`` (or ``H @ exp(s)``).

    Parameters
    ----------
    params : dict, optional
        Optional configuration with keys:

        - ``"log"`` (bool): if True, the parameters are log-transformed pumping
          rates and the model is ``H @ exp(s)``. By default False.
        - ``"H_path"`` (str or Path): location of ``H.mat``. By default the file
          shipped next to this module.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        params = params or {}
        self.log: bool = bool(params.get("log", False))
        self.H: NDArrayFloat = load_H(params.get("H_path", DEFAULT_H_PATH))

    @property
    def n_obs(self) -> int:
        return self.H.shape[0]

    @property
    def n_params(self) -> int:
        return self.H.shape[1]

    def run(self, s_ensemble: NDArrayFloat, *args, **kwargs) -> NDArrayFloat:
        """
        Run the forward model for an ensemble of parameter vectors.

        The whole ensemble is evaluated with a single (BLAS) matrix product,
        which is much faster than looping or spawning processes. Extra
        positional/keyword arguments (e.g. the former ``parallel`` and ``ncores``)
        are accepted for backward compatibility and ignored.

        Parameters
        ----------
        s_ensemble : NDArrayFloat
            Parameter vectors with shape (n_params, n_ensemble) or (n_params,).

        Returns
        -------
        NDArrayFloat
            Simulated drawdowns with shape (n_obs, n_ensemble).
        """
        S = np.asarray(s_ensemble, dtype=np.float64).reshape(self.n_params, -1)
        return self.H @ (np.exp(S) if self.log else S)

    def __call__(self, s_ensemble: NDArrayFloat) -> NDArrayFloat:
        """Alias for :meth:`run`, so that a Model can be used as `forward_model`."""
        return self.run(s_ensemble)


if __name__ == "__main__":
    s_true = np.zeros((10001, 1))
    s_true[1001:3502] = 0.12
    s_true[3502:4002] = 0.06

    model = Model()
    print("single run:", model.run(s_true).shape)
    s_ens = s_true + 0.1 * np.random.default_rng(0).normal(size=(10001, 3))
    print("ensemble run:", model.run(s_ens).shape)
    print("log model:", Model({"log": True}).run(np.log(s_true + 1e-6)).shape)
