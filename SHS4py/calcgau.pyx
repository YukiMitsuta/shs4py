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
        #cosdamp = np.cos(deltaTH)
        cosdamp = cosADD_SHS(nADD, nADDneibor, SQ)
        return ADDfeM * cosdamp * cosdamp * cosdamp
    else:
        return 0.0
cpdef np.ndarray IOE_grad(np.ndarray nADD, np.ndarray nADDneibor, np.ndarray SQ_inv, DTYPE_t ADDfeM, DTYPE_t r):
    cdef:
        DTYPE_t deltaTH, deltaTH_eps, cosdamp, xydot, IOE_center
        np.ndarray q_x, q_y, returngrad, qx_i
    #eps     = np.sqrt(np.finfo(float).eps)
    eps     = 1.0e-4
    q_x     = np.dot(SQ_inv, nADD)
    q_y     = np.dot(SQ_inv, nADDneibor)
    deltaTH = angle(q_x, q_y)
    returngrad = np.zeros(len(nADD))
    if deltaTH <= np.pi * 0.5:
        #cosdamp    = np.cos(deltaTH)
        cosdamp    = cosADD(q_x, q_y)
        IOE_center = - 3.0 * ADDfeM * cosdamp * cosdamp * np.sqrt(1.0 - cosdamp * cosdamp)
        for i in range(len(nADD)):
            qx_i          = copy.copy(q_x)
            qx_i[i]      += eps
            deltaTH_eps   = angle(qx_i, q_y)
            returngrad[i] = IOE_center * (deltaTH_eps - deltaTH) / eps


        #xydot      = np.dot(q_x, q_y)
        #IOE_center = cosdamp * cosdamp * cosdamp
        #for i in range(len(nADD)):
            #qx_i     = copy.copy(q_x)
            #qx_i[i] += eps
            #deltaTH  = angle(qx_i, q_y)
            #cosdamp  = np.cos(deltaTH)
            #cosdamp  = cosADD(qx_i,q_y)
            #IOE_eps  = cosdamp * cosdamp * cosdamp
            #returngrad[i] = ADDfeM * (IOE_eps - IOE_center) / eps
#        cosdamp = np.cos(deltaTH)
#        xydot   = np.dot(q_x, q_y)
#        for i in range(len(nADD)):
#            returngrad[i]  = q_y[i] * r * r
#            #returngrad[i] -= 2.0 * q_x[i] * xydot
#            returngrad[i] -= q_x[i] * xydot
#        returngrad *= ADDfeM * cosdamp * cosdamp
    return returngrad
cpdef np.ndarray delx_deltheta(DTYPE_t r, thetalist, int theta_i):
    cdef:
        np.ndarray Dqlist
        DTYPE_t a_k, theta
        int i
    Dqlist = np.array([r for _ in range(len(thetalist) + 1)])
    a_k = 1.0
    for i, theta in enumerate(thetalist):
        if i == theta_i:
            Dqlist[i] *= - a_k * np.sin(theta)
            a_k *= np.cos(theta)
        else:
            Dqlist[i] *= a_k * np.cos(theta)
            a_k *= np.sin(theta)
        if i < theta_i:
            Dqlist[i] *= 0.0
    Dqlist[-1] *= a_k
    return Dqlist
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
cpdef DTYPE_t cosADD_SHS(np.ndarray nADD, np.ndarray nADDneibor, np.ndarray SQ_inv):
    cdef:
        np.ndarray q_x, q_y
        int i, j
    q_x = np.dot(SQ_inv, nADD)
    q_y = np.dot(SQ_inv, nADDneibor)
    return cosADD(q_x, q_y)
cpdef DTYPE_t cosADD(np.ndarray x, np.ndarray y):
    cdef:
        int i
        DTYPE_t dot_xy, norm_x, norm_y
    dot_xy = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(len(x)):
        dot_xy += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]
    return dot_xy / np.sqrt(norm_x * norm_y)
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
