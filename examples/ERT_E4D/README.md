#### pypcga E4D example

This is the code example from the AGU 2014 presentation by Peter Kitanidis, rewritten in
python to illustrate an ERT application of PCGA.

A detailed explanation is given in the short arXiv write-up
["Accelerated 3D Electrical Resistivity Tomography with a Scalable Jacobian-free Approach"](http://arxiv.org/abs/2202.00059).

To run the example (`inversion_ert.py`, a marimo notebook):

1. install pypcga with the examples dependencies: `pip install pypcga[examples]`
2. install E4D (https://github.com/pnnl/E4D), see the installation guide
   (https://github.com/pnnl/E4D/blob/master/Installation.txt)
3. copy E4D's executables (`e4d` and `mpirun`) in the `input_files` folder
4. each E4D run uses 6 MPI processes (see `ert.py`): adjust `ncores` (the number of
   concurrent E4D runs) in `inversion_ert.py` to your machine
5. run `marimo edit inversion_ert.py` (or `python inversion_ert.py`)
