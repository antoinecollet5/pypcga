# pypcga examples

All the examples are [marimo](https://marimo.io) notebooks, i.e. plain python files:

```bash
pip install pypcga[examples]        # matplotlib + marimo (python >= 3.10)

marimo edit 2d_toy_example.py       # open as a reactive notebook
python 2d_toy_example.py            # or run it as a script
marimo export html 2d_toy_example.py -o 2d_toy_example.html  # or export it
```

Data files are located relative to each notebook, so they can be run from any
working directory.

## Self-contained examples

| Notebook | Description | Runtime |
|---|---|---|
| [`2d_toy_example.py`](2d_toy_example.py) | Main features: prior factorization, inversion (with interactive settings), posterior covariance, posterior sampling | ~5 s |
| [`pumping_history_identification/linear_inversion.py`](pumping_history_identification/linear_inversion.py) | 1D linear inversion: pumping history from drawdown data (Stanford CEE362G) | ~2 s |
| [`pumping_history_identification/nonlinear_inversion.py`](pumping_history_identification/nonlinear_inversion.py) | Same problem with a log-transform (non-negative rates) | ~3 s |

## Examples relying on external simulators

These notebooks load the data and build the prior without the simulator, then check
that the simulator is available. If it is not, they stop before the inversion and list
what is missing. In the interactive editor, the inversion only starts when you press
the *Run* button; as a script, it starts directly.

| Notebook | Simulator / requirements |
|---|---|
| [`modflow_flopy/inversion_modflow.py`](modflow_flopy/inversion_modflow.py) | `flopy` + MODFLOW-2005. The bundled binaries are old builds (the Linux one needs `libgfortran.so.3`): install a recent one (e.g. `get-modflow :python`) and set `MF2005_EXE` or put `mf2005` on the `PATH`. |
| [`stwave_duck/inversion_stwave.py`](stwave_duck/inversion_stwave.py) | STWAVE (bundled Linux binary needs `libgfortran.so.3`) + ERDC's `stwave_utils` package |
| [`tracer_tomography_ade_crunch/inversion_crunchtope.py`](tracer_tomography_ade_crunch/inversion_crunchtope.py) | [CrunchTope](https://bitbucket.org/crunchflow/crunchtope/downloads/) (bundled binary needs the Intel Fortran runtime) |
| [`tough_heat/inversion_tough.py`](tough_heat/inversion_tough.py) | TOUGH2 (`tough2-mp-eos1.debug`), `mpirun`, [PyTOUGH](https://github.com/acroucher/PyTOUGH), `pandas`, and the input files from [pyPCGA-TOUGH](https://github.com/amaliak/pyPCGA-TOUGH) |
| [`mare2dem_MT/inversion_mare2dem.py`](mare2dem_MT/inversion_mare2dem.py) | MARE2DEM + `mpirun` (bundled binary needs Intel MKL / Intel MPI) |
| [`adh_Savannah/inversion_adh_savannah.py`](adh_Savannah/inversion_adh_savannah.py) | AdH executables, ERDC's `enkf_tools`, `tables`, and the input directories from [adh_lee_et_al_wrr_2018](https://github.com/jonghyunharrylee/adh_lee_et_al_wrr_2018) |
| [`ERT_E4D/inversion_ert.py`](ERT_E4D/inversion_ert.py) | [E4D](https://github.com/pnnl/E4D) (`e4d` and `mpirun` copied in `input_files/`) |
| [`fenics_hydromechanics/inversion_fenics.py`](fenics_hydromechanics/inversion_fenics.py) | Legacy FEniCS (`dolfin`) + [multiphenics](https://github.com/multiphenics/multiphenics) |

Each folder also contains the python wrapper of the simulator (`mf.py`, `stwave.py`,
...). They expose a `Model(params).run(s_ensemble, parallel, ncores)` method returning
the simulated observations with shape (N_obs, N_e), which the notebooks pass to
`pypcga.PCGA` as `forward_model`.

## Writing your own forward model

`forward_model` receives an ensemble of parameter vectors with shape (N_s, N_e) and
must return the simulated observations with shape (N_obs, N_e). Most of the forward
runs of a Gauss-Newton iteration are independent: run them in parallel if they are
expensive (multiprocessing, mpi, joblib, a job scheduler, ...). Since pypcga 0.3.1,
the forward model does not need to be picklable (lambdas and closures are fine).
