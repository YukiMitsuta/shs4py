#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

import numpy as np
import SHS2py
class constClass():
    pass
def main():

    constC = constClass()
    constC.threshold          = 0.1
    constC.sameEQthreshold    = 0.05
    constC.IOEsphereA_initial = 0.2
    constC.IOEsphereA_dist    = 0.10
    constC.deltas0            = 0.01
    constC.deltas             = 0.005

    constC.periodicQ = False

    constC.calc_mpiQ  = False
    constC.calc_cupyQ = False
    constC.cythonQ    = False

    constC.exportADDpointsQ = True

    initialpointlist = [np.random.rand(2) * 8.0 - 4.0 for _ in range(10)]

    SHS2py.SHSearch(f, grad, hessian,
            importinitialpointQ = False, initialpoints = initialpointlist,
            const = constC)

def f(x):
    """
    In this test calculation, STYBLINSKI-TANG FUNCTION is used.
    https://www.sfu.ca/~ssurjano/stybtang.html
    f(x) = \frac{1}{2}\sum^d_{i=1}(x^4_i-16x_i^2+5x_i)
    """
    t1 = 4.0 - 2.1 * x[0] * x[0] + x[0] * x[0] * x[0] * x[0]  / 3.0
    t1 *= x[0] * x[0]
    t2 = x[0] * x[1]
    t3 = 4.0 * (x[1] * x[1] - 1.0) * x[1] * x[1]
    return t1 + t2 + t3
def grad(x):
    """
    greadient of f(x)
    """
    returngrad    = np.zeros(2)
    returngrad[0] = 8.0 * x[0] - 8.4 * x[0] * x[0] * x[0] + 2.0 * x[0] * x[0] * x[0] * x[0] * x[0] + x[1]
    returngrad[1] = x[0] + 16.0 * x[1] * x[1] * x[1] - 8.0 * x[1]
    return returngrad
def hessian(x):
    """
    hessian of f(x)
    """
    dim = len(x)
    returnhess = np.zeros((dim,dim))
    returnhess[0,0] = 8.0 - 25.2 * x[0] * x[0] + 10.9 * x[0] * x[0] * x[0] * x[0]
    returnhess[1,0] = 1.0
    returnhess[0,1] = 1.0
    returnhess[1,1] = 48.0 * x[1] * x[1] - 8.0
    
    return returnhess
if __name__ == "__main__":
    main()
