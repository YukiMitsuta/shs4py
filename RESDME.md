# shs2py

SHS2py is a Python-base script to calculate Scaled Hypersphere Searching method[1][2].

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

Y. Mitsuta et al. (now in place...)
 
# Author
 
* Yuki Mitsuta
* Center for Computational Sciences, University of Tsukuba
* E-mail: mitsutay[at]ccs.tsukuba.ac.jp

# License
 
"Physics_Sim_Py" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
 
