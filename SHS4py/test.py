#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

import numpy as np
import SHS4py
def main():
    SHS4py.SHSearch(f, grad, hessian)

def f(x):
    """
    in this test calculation, STYBLINSKI-TANG FUNCTION is used.
    https://www.sfu.ca/~ssurjano/stybtang.html
    f(x) = \frac{1}{2}\sum^d_{i=1}(x^4_i-16x_i^2+5x_i)
    """
    returnf = 0.0
    for i in range(len(x)):
        returnf += x[i]**4 - 16.0 * x[i]**2 + 5.0 * x[i]
    returnf *= 0.5
    return returnf
def grad(x):
    """
    greadient of f(x)
    """
    returngrad = np.zeros(len(x))
    for i in range(len(x)):
        returngrad[i] = 4.0 * x[i]**3 - 32.0 * x[i] + 5.0 
    returngrad *= 0.5
    return returngrad
def hessian(x):
    """
    hessian of f(x)
    """
    dim = len(x)
    returnhess = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if i == j:
                returnhess[i,j] = 6.0 * x[i]**2 - 16.0
    
    return returnhess
if __name__ == "__main__":
    main()
