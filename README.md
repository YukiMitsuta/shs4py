# SHS4py

SHS4py is a Python-base script to calculate Scaled Hypersphere Searching method[1][2].

[1] Ohno, K.; Maeda, S. Chem. Phys. Lett. 2004, 384 (4–6), 277–282.

[2] Maeda, S.; Ohno, K. Chem. Phys. Lett. 2005, 404 (1–3), 95–99.

# Requirement

* Python 3
* fasteners
* numpy
* scipy

In addition, next packages are required to accelerate the calculations.

* Cython

In calculations of free energy surfaces of metadynamics, further accerelation can be applied.
If your workstation use MPI multiprocessing, you can use multisred calculation by using next package.

* mpi4py

Furthermore, you can use GPGPU acceleration by using next package.

* cupy


# Note

I don't test environments under Windows.
If you use this script, please refer the next article.

Mitsuta, Yuki, and Yasuteru Shigeta. "Analytical Method Using a Scaled Hypersphere Search for High-Dimensional Metadynamics Simulations." Journal of chemical theory and computation 16.6 (2020): 3869-3878.
(https://pubs.acs.org/doi/10.1021/acs.jctc.0c00010)

# Author

* Yuki Mitsuta (満田祐樹）
* Center for Computational Sciences, University of Tsukuba
* E-mail: mitsutay[at]ccs.tsukuba.ac.jp

# License

"SHS4py" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
