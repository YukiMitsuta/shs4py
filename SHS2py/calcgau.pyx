# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.
import  numpy as np
cimport numpy as np
import copy

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cpdef DTYPE_t periodic_dist(
        DTYPE_t a, DTYPE_t b, DTYPE_t periodicmax, DTYPE_t periodicmin):
    cdef:
        DTYPE_t bdamp
    if b < periodicmin + a or periodicmax + a < b:
        bdamp  = (b - periodicmax - a) % (periodicmin - periodicmax)
        bdamp += periodicmax + a
    else:
        bdamp = b
    return a - bdamp
cpdef calcdist_periodic(np.ndarray x, np.ndarray s, np.ndarray sigmainv, 
        DTYPE_t periodicmax, DTYPE_t periodicmin, int dim):
    cdef:
        np.ndarray d
        DTYPE_t dpoint, tot
        int i
    d   = np.zeros(dim)
    tot = 0.0
    for i in range(dim):
        dpoint = periodic_dist(x[i], s[i], periodicmax, periodicmin) * sigmainv[i]
        tot    += dpoint * dpoint
        d[i] += dpoint
    return True, d, tot
cpdef calcdist(np.ndarray x, np.ndarray s, np.ndarray sigmainv, 
        DTYPE_t periodicmax, DTYPE_t periodicmin, int dim):
    cdef:
        np.ndarray d
        DTYPE_t dpoint, tot
        int i
    d   = np.zeros(dim)
    tot = 0.0
    for i in range(dim):
        dpoint = (x[i] - s[i]) * sigmainv[i]
        tot    += dpoint * dpoint
        d[i] += dpoint
    return True, d, tot
cpdef DTYPE_t f(np.ndarray x, np.ndarray s, np.ndarray sigmainv, 
        DTYPE_t periodicmax, DTYPE_t periodicmin, int dim, DTYPE_t h, periodicQ):
    cdef:
        np.ndarray d
        DTYPE_t tot
    if periodicQ:
        getdistQ, d, tot = calcdist_periodic(x, s, sigmainv, periodicmax, periodicmin, dim)
    else:
        getdistQ, d, tot = calcdist(x, s, sigmainv, periodicmax, periodicmin, dim)
    if getdistQ is False:
        return 0.0
    return - h * np.exp(- 0.5 * tot)
cpdef np.ndarray grad(np.ndarray x, np.ndarray s, np.ndarray sigmainv, 
        DTYPE_t periodicmax, DTYPE_t periodicmin, int dim, DTYPE_t h):
    cdef:
        np.ndarray d, returngrad
        DTYPE_t f_float, p, tot
        int i
    returngrad = np.zeros(dim)
    getdistQ, d, tot = calcdist(x, s, sigmainv, periodicmax, periodicmin, dim)
    if getdistQ is False:
        return returngrad
    f_float = - h * np.exp(- 0.5 * tot)
    for i in range(dim):
        returngrad[i] = - d[i] * f_float * sigmainv[i]
    return returngrad
cpdef np.ndarray hessian(np.ndarray x, np.ndarray s, np.ndarray sigmainv,
        DTYPE_t periodicmax, DTYPE_t periodicmin, int dim, DTYPE_t h):
    cdef:
        np.ndarray d, returnhess
        DTYPE_t f_float, p, tot
        int i,j
    returnhess = np.zeros((dim, dim))
    getdistQ, d, tot = calcdist(x, s, sigmainv, periodicmax, periodicmin, dim)
    if getdistQ is False:
        return returnhess
    f_float = - h * np.exp(- 0.5 * tot)
    for i in range(dim):
        for j in range(i, dim):
            if  i == j:
                returnhess[i,i] = f_float * (d[i] * d[i] - 1.0) * sigmainv[i] * sigmainv[i]
            else:
                returnhess[i,j] = f_float * d[i] * d[j] * sigmainv[i] *sigmainv[j]
                returnhess[j,i] += returnhess[i,j]
    return returnhess
cpdef np.ndarray calcgrad_theta(np.ndarray thetalistdamp, np.ndarray grad_x, DTYPE_t r):
    cdef:
        int thetaDim, thetaIndex, xIndex
        DTYPE_t grad_p
    thetaDim = len(thetalistdamp)
    grad_theta = [0.0 for _ in range(thetaDim)]
    for thetaIndex in range(thetaDim):
        for xIndex in range(thetaIndex, thetaDim + 1):
            if xIndex == 0:
                grad_p = - r * np.sin(thetalistdamp[0])
            else:
                grad_p = copy.copy(r)
                for i in range(xIndex):
                    if i == thetaIndex:
                        grad_p *= np.cos(thetalistdamp[i])
                    else:
                        grad_p *= np.sin(thetalistdamp[i])
                if xIndex != thetaDim:
                    if xIndex == thetaIndex:
                        grad_p *= - np.sin(thetalistdamp[xIndex])
                    else:
                        grad_p *= np.cos(thetalistdamp[xIndex])
            grad_theta[thetaIndex] += grad_p * grad_x[xIndex]
    grad_theta.reverse()
    return np.array(grad_theta)
cpdef DTYPE_t IOE(np.ndarray nADD, np.ndarray nADDneibor, np.ndarray SQ, DTYPE_t ADDfeM):
    cdef:
        DTYPE_t deltaTH, cosdamp
    deltaTH = angle_SHS(nADD, nADDneibor, SQ)
    if deltaTH <= np.pi * 0.5:
        cosdamp = np.cos(deltaTH)
        return ADDfeM * cosdamp * cosdamp * cosdamp
    else:
        return 0.0
cpdef DTYPE_t angle_SHS(np.ndarray nADD, np.ndarray nADDneibor, np.ndarray SQ_inv):
    cdef:
        np.ndarray q_x, q_y
        int i, j
    q_x = np.dot(SQ_inv, nADD)
    q_y = np.dot(SQ_inv, nADDneibor)
    return angle(q_x, q_y)
cpdef DTYPE_t angle(np.ndarray x, np.ndarray y):
    cdef:
        int i
        DTYPE_t dot_xy, norm_x, norm_y, _cos
    dot_xy = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(len(x)):
        dot_xy += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]
    _cos = dot_xy / np.sqrt(norm_x * norm_y)
    if _cos > 1:
        return 0.0
    elif _cos < -1:
        return np.pi
    return np.arccos(_cos)
cpdef np.ndarray SuperSphere_cartesian( DTYPE_t A, np.ndarray thetalist, np.ndarray SQ, int dim):
    cdef:
        int i
        DTYPE_t theta, a_k
        np.ndarray qlist
    qlist = np.array([np.sqrt(2.0 * A) for i in range(dim)])
    a_k = 1.0
    for i, theta in enumerate(thetalist):
        qlist[i] *= a_k * np.cos(theta)
        a_k *= np.sin(theta)
    qlist[-1] *= a_k
    qlist =  np.dot(SQ, qlist)
    return qlist
