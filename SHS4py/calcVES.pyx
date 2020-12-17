# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.
import  numpy as np
cimport numpy as np
import copy

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cpdef DTYPE_t fourier_f(DTYPE_t x_i, int k, DTYPE_t piPinv):
    cdef:
        DTYPE_t a
    if k == 0:
        return 1.0
    else:
        a = (((k-1) // 2)+1) * piPinv 
        if k % 2 == 0:
            return np.sin(a * x_i)
        else:
            return np.cos(a * x_i)
cpdef DTYPE_t fourier_grad(DTYPE_t x_i, int k, DTYPE_t piPinv):
    cdef:
        DTYPE_t a
    if k == 0:
        return 0.0
    else:
        a = (((k-1) // 2)+1) * piPinv 
        if k % 2 == 0:
            return   a * np.cos(a * x_i)
        else:
            return - a * np.sin(a * x_i)
cpdef DTYPE_t fourier_gradgrad(DTYPE_t x_i, int k, DTYPE_t piPinv):
    cdef:
        DTYPE_t a
    if k == 0:
        return 0.0
    else:
        a = (((k-1) // 2)+1) * piPinv 
        if k % 2 == 0:
            return - a * a * np.sin(a * x_i)
        else:
            return - a * a * np.cos(a * x_i)

cpdef DTYPE_t chebyshev_f(DTYPE_t t, int k):
    cdef:
        DTYPE_t returnf
        DTYPE_t series_f0, series_f1
        int i
    series_f0 = t
    series_f1 = 1.0
    for i in range(k-1):
        returnf = 2.0 * t * series_f0 - series_f1
        series_f0, series_f1 = returnf, series_f0
    return returnf
cpdef DTYPE_t chebyshev_grad(DTYPE_t t, int k, DTYPE_t tconst):
    cdef:
        DTYPE_t a, returnf, returngrad
        DTYPE_t series_f0, series_f1
        DTYPE_t series_grad0, series_grad1
        int i
    series_f0    = t
    series_f1    = 1.0
    series_grad0 = 1.0
    series_grad1 = 0.0
    for i in range(k-1):
        returnf = 2.0 * t * series_f0 - series_f1
        returngrad = 2.0 *     series_f0 \
                   + 2.0 * t * series_grad0 \
                             - series_grad1
        series_f0,    series_f1    = returnf, series_f0
        series_grad0, series_grad1 = returngrad, series_grad0
    return returngrad * tconst
cpdef DTYPE_t chebyshev_gradgrad(DTYPE_t t, int k, DTYPE_t tconst):
    cdef:
        DTYPE_t a, returnf, returngrad
        DTYPE_t series_f0, series_f1
        DTYPE_t series_grad0, series_grad1
        DTYPE_t series_gradgrad0, series_gradgrad1
        int i
    series_f0    = t
    series_f1    = 1.0
    series_grad0 = 1.0
    series_grad1 = 0.0
    series_gradgrad0 = 0.0
    series_gradgrad1 = 0.0
    for i in range(k-1):
        returnf = 2.0 * t * series_f0 - series_f1
        returngrad = 2.0 *     series_f0 \
                   + 2.0 * t * series_grad0 \
                             - series_grad1
        returngradgrad = 4.0 *     series_grad0 \
                       + 2.0 * t * series_gradgrad0 \
                                 - series_gradgrad1
        series_f0,    series_f1    = returnf, series_f0
        series_grad0, series_grad1 = returngrad, series_grad0
        series_gradgrad0, series_gradgrad1 = returngradgrad, series_grad0
    return returngradgrad * tconst * tconst

cpdef DTYPE_t legendre_f(DTYPE_t t, int k):
    cdef:
        DTYPE_t a, b, returnf
        DTYPE_t series_f0, series_f1
        int i
    series_f0 = t
    series_f1 = 1.0
    for i in range(k-1):
        a = (2.0 * (i - 1.0) + 1.0) / i 
        b = (i - 1.0) / i 
        returnf = a * t * series_f0 - b * series_f1 
        series_f0, series_f1 = returnf, series_f0
    return returnf
cpdef DTYPE_t legendre_grad(DTYPE_t t, int k, DTYPE_t tconst):
    cdef:
        DTYPE_t a, b, returnf, returngrad
        DTYPE_t series_f0, series_f1
        DTYPE_t series_grad0, series_grad1
        int i
    series_f0    = t
    series_f1    = 1.0
    series_grad0 = 1.0
    series_grad1 = 0.0
    for i in range(k-1):
        a = (2.0 * (i - 1.0) + 1.0) / i 
        b = (i - 1.0) / i 
        returnf = a * t * series_f0 \
                        - b * series_f1 
        returngrad = a     * series_f0 \
                       + a * t * series_grad0 \
                           - b * series_grad1 
        series_f0,    series_f1    = returnf, series_f0
        series_grad0, series_grad1 = returngrad, series_grad0
    return returngrad * tconst
cpdef DTYPE_t legendre_gradgrad(DTYPE_t t, int k, DTYPE_t tconst):
    cdef:
        DTYPE_t a, b, returnf, returngrad
        DTYPE_t series_f0, series_f1
        DTYPE_t series_grad0, series_grad1
        DTYPE_t series_gradgrad0, series_gradgrad1
        int i
    series_f0    = t
    series_f1    = 1.0
    series_grad0 = 1.0
    series_grad1 = 0.0
    series_gradgrad0 = 0.0
    series_gradgrad1 = 0.0
    for i in range(k-1):
        a = (2.0 * (i - 1.0) + 1.0) / i 
        b = (i - 1.0) / i 
        returnf = a * t * series_f0 \
                        - b * series_f1 
        returngrad = a     * series_f0 \
                       + a * t * series_grad0 \
                           - b * series_grad1 
        returngradgrad = 2.0 * a * series_grad0 \
                             + a * t * series_gradgrad0 \
                                 - b * series_gradgrad1 
        series_f0,    series_f1    = returnf, series_f0
        series_grad0, series_grad1 = returngrad, series_grad0
        series_gradgrad0, series_gradgrad1 = returngradgrad, series_grad0


        returnf = 2.0 * t * series_f0 - series_f1
        returngrad = 2.0 *     series_f0 \
                   + 2.0 * t * series_grad0 \
                             - series_grad1
        returngradgrad = 4.0 *     series_grad0 \
                       + 2.0 * t * series_gradgrad0 \
                                 - series_gradgrad1
    return returngradgrad * tconst * tconst
