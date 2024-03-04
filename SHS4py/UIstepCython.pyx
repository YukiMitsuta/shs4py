import  numpy as np
cimport numpy as np
import  copy
import  itertools
import  collections
import  random
from    scipy.integrate import quad

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cpdef np.ndarray periodic(np.ndarray xi, np.ndarray periodicmax, np.ndarray periodicmin,
        np.ndarray beforepoint, int dim):
    cdef:
        np.ndarray xidamp
        DTYPE_t beforepointdamp
        int i
    xidamp = copy.copy(xi)
#    while True:
#        periodicoutQ = False
#        for i in range(dim):
#            if xidamp[i] < periodicmin[i] + beforepoint[i]:
#                xidamp[i] += (periodicmax[i] - periodicmin[i])
#                periodicoutQ = True
#            elif periodicmax[i] + beforepoint[i] < xidamp[i]:
#                xidamp[i] -= (periodicmax[i] - periodicmin[i])
#                periodicoutQ = True
#        if not periodicoutQ:
#            break
    for i in range(dim):
        beforepointdamp = copy.copy(beforepoint[i])
        #if beforepointdamp < periodicmin[i] or periodicmax[i] < beforepointdamp:
            #beforepointdamp  = (beforepointdamp - periodicmax[i]) % (periodicmin[i] - periodicmax[i])
            #beforepointdamp += periodicmax[i]

        if xidamp[i] < periodicmin[i] + beforepointdamp or periodicmax[i] + beforepointdamp < xidamp[i]:
            xidamp[i]  = (xidamp[i] - periodicmax[i] - beforepointdamp) % (periodicmin[i] - periodicmax[i])
            xidamp[i] += periodicmax[i] + beforepointdamp
    return xidamp

#def periodic(np.ndarray xi, np.ndarray periodicmax, np.ndarray periodicmin,
#        np.ndarray beforepoint, int dim):
#    cdef np.ndarray periodicplus
#    cdef np.ndarray periodicminus
#    cdef np.ndarray xidamp
#    cdef np.ndarray periodicplusbf
#    cdef np.ndarray periodicminusbf
#    cdef np.ndarray xidampbf
#    cdef int _i
#
#    periodicplus  = xi + (periodicmax - periodicmin)
#    periodicminus = xi - (periodicmax - periodicmin)
#    xidamp        = copy.copy(xi)
#
#    periodicplusbf  = periodicplus  - beforepoint
#    periodicminusbf = periodicminus - beforepoint
#    xidampbf        = xidamp        - beforepoint
#
#    periodicplusbf  = periodicplusbf * periodicplusbf
#    periodicminusbf = periodicminusbf * periodicminusbf
#    xidampbf        = xidampbf * xidampbf
#
#    for _i in range(dim):
#        if periodicminusbf[_i] < xidampbf[_i]:
#                xidamp[_i] = periodicminus[_i]
#        elif periodicplusbf[_i] < xidampbf[_i]:
#                xidamp[_i] = periodicplus[_i]
#    return xidamp
cpdef np.ndarray cgrad(np.ndarray covinv, np.ndarray xi, 
          np.ndarray ave,    np.ndarray K,
          np.ndarray ref,
          int dim, DTYPE_t betainv):
    cdef np.ndarray grad
    cdef DTYPE_t damp
    cdef int x, y
    grad = np.zeros(dim)
    for x in range(dim):
        for y in range(dim):
            grad[x] += betainv * covinv[x,y] * (xi[y] - ave[y])
            grad[x] -= K[x,y]  * (xi[y] - ref[y])
    return grad
cpdef DTYPE_t cgrad_on_vec(np.ndarray covinv, np.ndarray xi, 
                np.ndarray ave,    np.ndarray K,
                np.ndarray ref,    np.ndarray Evec,
                int dim, DTYPE_t betainv):
    cdef:
        DTYPE_t damp
        np.ndarray grad
    grad = cgrad(covinv, xi, ave, K, ref, dim, betainv)
    damp = 0.0
    for x in range(dim):
        damp += grad[x] * Evec[x]
    return damp
cpdef np.ndarray cdelta1(np.ndarray ave, np.ndarray K,
            np.ndarray ref, np.ndarray covinv,
            np.ndarray _Cnextref,
            np.ndarray Deltatarget,
            int dim, DTYPE_t betainv):
    cdef np.ndarray _delta1
    cdef int x,y
    _delta1 = np.zeros(dim)
    for x in range(dim):
        for y in range(dim):
            _delta1[x] += K[x,y] * (_Cnextref[y] - ref[y] - Deltatarget[y])
            _delta1[x] -= betainv * covinv[x,y] * (_Cnextref[y]- ave[y] - Deltatarget[y])
    return _delta1
#def cdelta(np.ndarray _delta1, np.ndarray K,
cpdef np.ndarray cdelta(np.ndarray _delta1, np.ndarray K,
           np.ndarray _nextK, np.ndarray covinv,
           np.ndarray tergetpointdamp, np.ndarray nextref,
           int dim, DTYPE_t betainv):
    cdef np.ndarray _delta
    cdef np.ndarray _delta2
    cdef int x,y
    _delta2 = np.linalg.inv(betainv * covinv - K + _nextK)
    _delta  = np.zeros(dim)
    for x in range(dim):
        for y in range(dim):
            _delta[x] += _delta2[x,y] * _delta1[x]
    _delta = _delta + tergetpointdamp - nextref
    return _delta
cpdef DTYPE_t cdelta_Norm(np.ndarray _delta1, np.ndarray K,
           np.ndarray _nextK, np.ndarray covinv,
           int dim, DTYPE_t betainv, np.ndarray tergetpointdamp, np.ndarray nextref):
    cdef np.ndarray _delta, movevec
    #cdef DTYPE_t movevec_Norm
    cdef DTYPE_t delta_Norm
    cdef int i
    _delta  = cdelta(_delta1, K, _nextK, covinv, tergetpointdamp, nextref, dim, betainv)
    #movevec = tergetpointdamp - _delta - nextref
    #movevec_Norm = 0.0
    delta_Norm = 0.0
    for i in range(dim):
        delta_Norm += _delta[i] * _delta[i]
    delta_Norm = np.sqrt(delta_Norm)
    return delta_Norm
cpdef np.ndarray calcK_theta(DTYPE_t krad, np.ndarray x, np.ndarray eigV, KoffdiagonalQ, int Kmin, int dim):
    #cdef np.ndarray Karray, _thetalist, K, P
    cdef np.ndarray Karray, K, P, Pt, Pinv
    cdef int  i, j, n
    cdef DTYPE_t a_k, theta
    K = Kmin * np.identity(dim)
    if krad != 0.0:
        _xlist     = [1.0 for i in range(dim)]
        a_k = 1.0
        for i, theta in enumerate(x):
            _xlist[i] = a_k * np.cos(theta)
            a_k *= np.sin(theta)
        _xlist[-1] = a_k 
        _xlist.reverse()
        Karray = np.array(_xlist)

        for i in range(dim):
            if Karray[i] < 0.0:
                Karray[i] *= -1.0
        Karray = krad * Karray
        for i in range(dim):
            K[i, i] += Karray[i]
    #if KoffdiagonalQ:
        #K = np.zeros((dim,dim))
        #K = Kmin * np.identity(dim)
    #else:
        #K = Kmin * np.identity(dim)
    #K = Kmin * np.identity(dim)
    #for i in range(dim):
        #K[i, i] += Karray[i]
    if KoffdiagonalQ:
        #if krad == 0.0:
            #print(K)
        P  = eigV
        Pt = copy.copy(P)
        Pt = Pt.transpose()
        K = np.dot(P , K)
        K = np.dot(K,  Pt)
        #if krad == 0.0:
            #print(K)
            #print("=")
    return K
cpdef DTYPE_t cPbias(np.ndarray xidamp, np.ndarray ave, np.ndarray covinv, DTYPE_t Pbiasave, int dim):
    #cdef np.ndarray xidamp
    cdef np.ndarray deltaxi
    cdef np.ndarray damp
    cdef DTYPE_t _Pbias
    cdef int _i, _j
    deltaxi = xidamp - ave
    damp = np.zeros(dim)
    for _i in range(dim):
        for _j in range(dim):
            damp[_i] += deltaxi[ _j] * covinv[_j, _i]
    _Pbias = 0.0
    for _i in range(dim):
        _Pbias += damp[_i] * deltaxi[_i]
    _Pbias = np.exp(-0.5 * _Pbias) * Pbiasave
    return _Pbias
cpdef DTYPE_t cPbias_periodic(np.ndarray xi, UI, np.ndarray periodicmax, np.ndarray periodicmin, int dim):
    cdef np.ndarray xidamp
    cdef np.ndarray deltaxi
    cdef np.ndarray damp
    cdef DTYPE_t _Pbias
    cdef int _i, _j
    xidamp = periodic(xi, periodicmax, periodicmin, UI.ref, dim)
    deltaxi = xidamp - UI.ave
    damp = np.zeros(dim)
    for _i in range(dim):
        for _j in range(dim):
            damp[_i] += deltaxi[ _j] * UI.covinv[_j, _i]
    _Pbias = 0.0
    for _i in range(dim):
        _Pbias += damp[_i] * deltaxi[_i]
    _Pbias = np.exp(-0.5 * _Pbias) * UI.Pbiasave
    return _Pbias

cpdef np.ndarray calcthlist(int dim, np.ndarray Mlist):
    cdef int i, j, x, denominator
    cdef DTYPE_t newth
    cdef np.ndarray thlist
    thlist = np.zeros(dim - 1)
    if Mlist[0] == 0   and Mlist[1] == 0:
        thlist[0] = 0.0
    elif Mlist[0] >= 0   and Mlist[1] >= 0:
        thlist[0] =  np.pi * 0.5 * Mlist[1]/(Mlist[0] + Mlist[1])
    elif Mlist[0] >= 0 and Mlist[1] < 0:
        thlist[0] = -np.pi * 0.5 * (- Mlist[1])/(Mlist[0] - Mlist[1])
    elif Mlist[0] < 0  and Mlist[1] >= 0:
        thlist[0] =  np.pi - np.pi * 0.5 * Mlist[1]/(- Mlist[0] + Mlist[1])
    elif Mlist[0] < 0  and Mlist[1] < 0:
        thlist[0] = -np.pi + np.pi * 0.5 * (- Mlist[1])/(- Mlist[0] - Mlist[1])
    thlist[0] += np.pi
    for i in range(1, dim - 1):
        denominator = sum([abs(Mlist[j]) for j in range(i + 2)]) - (sum([1 for x in Mlist[:i + 2] if int(x) != 0] ) - 2)
        if Mlist[0] == 0   and Mlist[1] == 0:
            denominator -= 1
        newth = np.pi * 0.5 * Mlist[i + 1] / denominator
        thlist[i] = newth + np.pi * 0.5
    for i in range(len(thlist) - 1):
        if thlist[i] > np.pi / 2:
            thlist[i] -= np.pi
    if thlist[-1] > np.pi:
        thlist[-1] -= 2.0 * np.pi
    return thlist
cpdef getpointvector(int dim, int M):
    cdef:
        int n, nmax, x, iterdim
        int key, value
        int i, j
    returnlist = []

    for n in range(1, dim + 1):
        nmax = M + n
        for MlistC in itertools.combinations_with_replacement(range(1, M + dim - 1), n):
            if sum([x for x in MlistC]) == nmax:
                MlistC = collections.Counter(list(MlistC) + [0] * (dim - n))
                iterlist = []
                keylist = []
                iterdim = copy.copy(dim)
                for key, value in MlistC.most_common():
                    iterlist.append([l for l in itertools.combinations(range(iterdim), value)])
                    keylist.append(key)
                    iterdim -= value
                for iter in itertools.product(*iterlist):
                    Mlist = np.zeros(dim)
                    pernums = list(range(dim))
                    for i, iternums in enumerate(iter):
                        for j, iternum in enumerate(iternums):
                            pernum        = pernums.pop(iternum - j)
                            Mlist[pernum] = keylist[i]
                    for signiter in itertools.product([-1, 1], repeat=n):
                        signiter = list(signiter)
                        Mlistdamp = copy.copy(Mlist)
                        for i in range(len(Mlist)):
                            if Mlist[i] != 0:
                                Mlistdamp[i] *= signiter.pop()
                        #yield Mlistdamp
                        thlist = calcthlist(dim, Mlistdamp)
                        for returnth in returnlist:
                            print(abs(thlist - returnth))
                            if sum(abs(thlist - returnth)) < 0.001:
                                break
                        else:
                            returnlist.append(calcthlist(dim, Mlistdamp))
    return returnlist
#def cPmax(UIlist, np.ndarray xi, int dim):
#    cdef DTYPE_t maxp
#    maxp = 0.0
#    for UI in UIlist:
#        #UIp = UI.Pbias(xi) / UI.Pbiasave
#        UIp = cPbias(xi, UI, dim) / UI.Pbiasave
#        if UIp > maxp:
#            maxp = UIp
#    return maxp
#def cPmax_periodic(UIlist, np.ndarray xi, int dim):
#    cdef DTYPE_t maxp
#    maxp = 0.0
#    for UI in UIlist:
#        #UIp = UI.Pbias(xi) / UI.Pbiasave
#        UIp = UI.Pbias(xi) / UI.Pbiasave
#        if UIp > maxp:
#            maxp = UIp
#    return maxp
cpdef np.ndarray cgradUIall(UIlist, np.ndarray xi, int dim, DTYPE_t betainv, DTYPE_t eps):
    cdef np.ndarray gradall, gradinUI
    cdef DTYPE_t allpinv
    cdef int i
    if len(UIlist) == 1:
        return UIlist[0].grad(xi)
    gradall = np.zeros(dim)
    if cDmin(UIlist, xi) < 10.0:
        allpinv  = 1.0 / cPUIall(UIlist, xi)
        for UI in UIlist:
            gradinUI = cgrad(UI.covinv, xi, UI.ave, 
                         UI.K, UI.ref, dim, betainv)
            gradall += UI.N * cPbias(xi, UI.ave, UI.covinv, UI.Pbiasave, dim) * allpinv * gradinUI
    else:
        for i in range(dim):
            gradall[i] = 10e10
    return gradall
cpdef np.ndarray cgradUIall_periodic(UIlist, np.ndarray xi, 
        np.ndarray periodicmax, np.ndarray periodicmin, int dim, DTYPE_t betainv, DTYPE_t eps):
    cdef np.ndarray gradall, gradinUI, xidamp
    cdef np.ndarray ref_periodic, ave_periodic
    cdef DTYPE_t allpinv
    cdef int i
    if len(UIlist) == 1:
        return UIlist[0].grad(xi)
    gradall = np.zeros(dim)
    #if cDmin(UIlist, xi) < 10.0:
    if True:
        allpinv  = cPUIall_periodic(UIlist, xi, periodicmax, periodicmin, dim)
        if allpinv < eps:
            for i in range(dim):
                gradall[i] = 10e10
            return gradall
        allpinv = 1.0 / allpinv
        xidamp = periodic(xi, periodicmax, periodicmin, np.zeros(dim), dim)
        for UI in UIlist:
            ave_periodic = periodic(UI.ave,       periodicmax, periodicmin, np.zeros(dim), dim)
            ref_periodic = periodic(UI.ref,       periodicmax, periodicmin, np.zeros(dim), dim)
            ave_periodic = periodic(ave_periodic, periodicmax, periodicmin, xidamp, dim)
            ref_periodic = periodic(ref_periodic, periodicmax, periodicmin, xidamp, dim)
            gradinUI = cgrad(UI.covinv, xidamp, ave_periodic, 
                         UI.K, ref_periodic, dim, betainv)
            gradall += UI.N * cPbias(xidamp, ave_periodic, UI.covinv, UI.Pbiasave, dim) * allpinv * gradinUI
    #else:
        #for i in range(dim):
            #gradall[i] = 10e10
    return gradall
cpdef DTYPE_t cPUIall(UIlist, np.ndarray xi):
    cdef DTYPE_t allp
    allp = 0.0
    for UI in UIlist:
        allp += UI.N * UI.Pbias(xi)
    return allp
cpdef DTYPE_t cPUIall_periodic(UIlist, np.ndarray xi, np.ndarray periodicmax, np.ndarray periodicmin, int dim):
    cdef DTYPE_t allp
    cdef np.ndarray xidamp, ave_periodic
    allp = 0.0
    xidamp       = periodic(xi, periodicmax, periodicmin, np.zeros(dim), dim)
    for UI in UIlist:
        ave_periodic = periodic(UI.ave,       periodicmax, periodicmin, np.zeros(dim), dim)
        ave_periodic = periodic(ave_periodic, periodicmax, periodicmin, xidamp,        dim)
        allp        += UI.N * cPbias(xidamp, ave_periodic, UI.covinv, UI.Pbiasave, dim)
    return allp
#cpdef trapzd(UIlist, np.ndarray initialpoint, np.ndarray Evec, DTYPE_t a, DTYPE_t b, int n, int dim, DTYPE_t olds):
#    cdef DTYPE_t x, tnm, sum, dell, s
#    cdef int it, j
#    #global s
#    if n == 1:
#        s = 0.5 * (b - a) * (
#                func(UIlist, initialpoint, Evec, a, dim)
#             + func(UIlist, initialpoint, Evec, b, dim))
#    else:
#        it = 1
#        for j in range(n):
#            it <<= 1
#        tnm = copy.copy(it)
#        dell = (b-a) /tnm
#        x = a + 0.5 * dell
#        sum = 0.0
#        for j in range(it):
#            x += dell
#            sum += func(UIlist, initialpoint, Evec, x, dim)
#        s = 0.5 * (olds + (b - a) * sum/tnm)
#    return s
#def calcUIallC(UIlist, np.ndarray initialpoint, np.ndarray finishpoint, allperiodicQ,
def calcUIallC(np.ndarray avelist, np.ndarray reflist, np.ndarray covinvlist, 
        np.ndarray Pbiasavelist, np.ndarray Klist, np.ndarray Nlist, 
        np.ndarray initialpoint, np.ndarray finishpoint, allperiodicQ,
        np.ndarray periodicmin, np.ndarray periodicmax, DTYPE_t betainv, int dim):
    cdef:
        np.ndarray fpointdamp, vec, Evec
        DTYPE_t    vecnorm, x, deltaA, d, t

    if allperiodicQ:
        fpointdamp = periodic(finishpoint, 
                        periodicmax, periodicmin, initialpoint, dim)
    else:
        fpointdamp = finishpoint
    vec     = fpointdamp - initialpoint
    vecnorm = 0.0
    for x in vec:
        vecnorm += x * x
    vecnorm = np.sqrt(vecnorm)
    if vecnorm == 0.0:
        return 0.0
    #if pUIall(UIlist, finishpoint) == 0.0:
        #return 1.0e10, 0.0
    Evec = vec / vecnorm
    if allperiodicQ:
        deltaA, d= quad(
            #lambda t:func_periodic(UIlist, initialpoint,
            lambda t:func_periodic(avelist, reflist, covinvlist, Pbiasavelist, Klist, Nlist, initialpoint,
                            Evec, t, periodicmin, periodicmax, dim, betainv),
            0, vecnorm)
    else:
        deltaA, d= quad(
            lambda t:func(avelist, reflist, covinvlist, Pbiasavelist, Klist, Nlist, initialpoint, Evec, t, dim, betainv),
            0, vecnorm)
    return deltaA
#def calcVarA(UIlist, initialpoint, finishpoint, Evec, vecnorm):
    #vecinv = calcvecinv(Evec)
    #sigma_rlist  = []
#cpdef DTYPE_t func_periodic(UIlist, np.ndarray initialpoint, np.ndarray Evec, DTYPE_t x, 
cpdef DTYPE_t func_periodic(np.ndarray avelist, np.ndarray reflist, np.ndarray covinvlist, 
        np.ndarray Pbiasavelist, np.ndarray Klist, np.ndarray Nlist,
        np.ndarray initialpoint, np.ndarray Evec, DTYPE_t x,
        np.ndarray periodicmin, np.ndarray periodicmax, int dim, DTYPE_t betainv):
    cdef np.ndarray grad, gradinUI, xi, xidamp
    cdef np.ndarray ave_periodic, ref_periodic
    cdef int i
    cdef DTYPE_t f, allP, _Pbias#, N
    #grad = cgradUIall(UIlist, initialpoint + x * Evec, dim)
    xi   = initialpoint + x * Evec
    grad = np.zeros(dim)
    #allpinv = 1.0 / cPUIall_periodic(UIlist, xi, periodicmax, periodicmin, dim)
    allP = 0.0
    xidamp = periodic(xi, periodicmax, periodicmin, np.zeros(dim), dim)
    #for UI in UIlist:
    for i in range(len(avelist)):
        ave_periodic = periodic(avelist[i],   periodicmax, periodicmin, np.zeros(dim), dim)
        ref_periodic = periodic(reflist[i],   periodicmax, periodicmin, np.zeros(dim), dim)
        ave_periodic = periodic(ave_periodic, periodicmax, periodicmin, xidamp,        dim)
        ref_periodic = periodic(ref_periodic, periodicmax, periodicmin, xidamp,        dim)
        gradinUI     = cgrad(covinvlist[i], xidamp, ave_periodic, 
                         Klist[i], ref_periodic, dim, betainv)
        _Pbias = Nlist[i] * cPbias(xidamp, ave_periodic, covinvlist[i], Pbiasavelist[i], dim)
        allP = allP + _Pbias
        grad = grad + _Pbias * gradinUI
    #if allP == 0.0:
        #return False
    grad = grad / allP
    f = 0.0
    for i in range(dim):
        f += grad[i] * Evec[i]
    return f
cpdef DTYPE_t func(np.ndarray avelist, np.ndarray reflist, np.ndarray covinvlist, 
        np.ndarray Pbiasavelist, np.ndarray Klist, np.ndarray Nlist, 
        np.ndarray initialpoint, np.ndarray Evec, DTYPE_t x, int dim, DTYPE_t betainv):
    cdef np.ndarray grad, gradinUI, xi
    cdef int i
    cdef DTYPE_t f, allP, _Pbias
    #grad = cgradUIall(UIlist, initialpoint + x * Evec, dim)
    xi   = initialpoint + x * Evec
    grad = np.zeros(dim)
    #allpinv = 1.0 / cPUIall(UIlist, xi)
    allP = 0.0
    #for UI in UIlist:
    for i in range(len(avelist)):
        gradinUI = cgrad(covinvlist[i], xi, avelist[i],
                         Klist[i], reflist[i], dim, betainv)
        #grad += UI.N * cPbias(xi, UI, dim) * allpinv * gradinUI
        _Pbias = Nlist[i] * cPbias(xi, avelist[i], covinvlist[i], Pbiasavelist[i], dim)
        allP = allP + _Pbias
        grad = grad + _Pbias * gradinUI
    grad = grad / allP
    f = 0.0
    for i in range(dim):
        f += grad[i] * Evec[i]
    return f
#cpdef qtrap(UIlist, np.ndarray initialpoint, np.ndarray Evec, DTYPE_t a, DTYPE_t b, int dim):
#    cdef int j, Jmax
#    cdef DTYPE_t s, st, olds, EPS, ost, os
#    Jmax = 20
#    EPS  = 1.0e-3
#    #olds = -1.0e30
#    ost  = -1.0e30
#    os   = -1.0e30
#    #s = trapzd(UIlist, initialpoint, Evec, a, b, 1, dim, 0.0)
#    for j in range(Jmax):
#        st = trapzd(UIlist, initialpoint, Evec, a, b, j + 1, dim, ost)
#        s  = (4.0 * st - ost)/3.0
#        print(s - os)
#        #if j > 5 and abs(s-os) < EPS * abs(os):
#        if abs(s-olds) < EPS * abs(olds):
#            return s
#        os = copy.copy(s)
#        ost = copy.copy(st)
#    print("ERROR : Too many steps in rutine qtrap\n***EXIT***")
#    exit()

cpdef DTYPE_t deltaA_UI(np.ndarray covinv, np.ndarray K, np.ndarray ave, np.ndarray ref,
        np.ndarray initialp, np.ndarray finalp, DTYPE_t betainv, allperiodicQ,
        np.ndarray periodicmax, np.ndarray periodicmin):
    cdef np.ndarray vec, Evec, Evecnorm2, initialp_ave, initialp_ref
    cdef int i, j, k, dim
    cdef DTYPE_t vecnorm, int2, int3, damp
    dim       = len (ave)
    vec       = finalp - initialp
    vecnorm   = 0.0
    for i in range(dim):
        vecnorm += vec[i] * vec[i]
    vecnorm = np.sqrt(vecnorm)
    if vecnorm == 0.0: 
        return 0.0
    Evec      = vec / vecnorm
    if allperiodicQ:
        initialp_ave = periodic(initialp, periodicmax, periodicmin, ave, dim)
        initialp_ref = periodic(initialp, periodicmax, periodicmin, ref, dim)
    else:
        initialp_ave = initialp
        initialp_ref = initialp
    Evecnorm2 = (vec * 0.5 + initialp_ave - ave) * vecnorm
    int2      = 0.0
    for j in range(dim):
        damp = 0.0
        for k in range(dim):
            damp += covinv[k,j] * Evec[k]
        int2 += damp * Evecnorm2[j]
    int2 *= betainv
    Evecnorm2 = (vec * 0.5 + initialp_ref - ref) * vecnorm
    int3 = 0.0
    for j in range(dim):
        damp = 0.0
        for k in range(dim):
            damp += K[k,j] * Evec[k]
        int3 += damp * Evecnorm2[j]
    return int2 - int3
cpdef DTYPE_t Dbias(np.ndarray xi, allperiodicQ, np.ndarray periodicmax, np.ndarray periodicmin,
        np.ndarray ave, np.ndarray covinv, int dim, partOPTQ, int partdim):
    cdef:
        #np.ndarray xidamp, _V, vec
        #DTYPE_t vecnorm, D, _N, th, a
        np.ndarray xidamp, vec
        #np.ndarray sortedsigma
        DTYPE_t D
        int i
    if allperiodicQ:
        xidamp = periodic(xi, periodicmax, periodicmin, ave, dim)
    else:
        xidamp = xi
    vec = xidamp - ave
    if partOPTQ:
        sortedsigma = sorted([covinv[i, i] for i in range(dim)])[:partdim]

    D = 0.0
    for i in range(dim):
        if partOPTQ:
            if not covinv[i,i] in sortedsigma:
                continue
        for j in range(dim):
            if partOPTQ:
                if not covinv[j, j] in sortedsigma:
                    continue
            D += vec[i] * covinv[i, j] * vec[j]
    D = np.sqrt(D)
    return D
cpdef DTYPE_t cDmin(UIlist, np.ndarray xi):
    cdef:
        DTYPE_t minD, UID

    minD = 1.0e30
    for UI in UIlist:
        if UI.stepN < 0:
            continue
        #UIp = UI.Pbias(xi) / UI.Pbiasave
        UID = UI.Dbias(xi)
        if UID < minD:
            minD = copy.copy(UID)
    return minD
#cpdef DTYPE_t Dbias(np.ndarray xi, np.ndarray ave, np.ndarray cov_eigN, np.ndarray cov_eigV, int dim, partOPTQ, int partdim):
#    cdef:
#        np.ndarray xidamp, _V, vec
#        np.ndarray sortedsigma
#        DTYPE_t vecnorm, D, _N, th
#    vec = xi - ave
#    vecnorm = 0.0
#    for i in range(dim):
#        vecnorm += vec[i] * vec[i]
#    if vecnorm == 0.0: 
#        return 0.0
#    vecnorm = np.sqrt(vecnorm)
#    D = 0.0
#    for _N, _V in zip(cov_eigN, cov_eigV):
#        th = angle(_V, vec)
#        a  = vecnorm * np.cos(th)
#        D +=  a * a / _N
#    D = np.sqrt(D)
#    return D
cpdef DTYPE_t angle(np.ndarray x, np.ndarray y):
    """
    angle between x vector and y vector
    0 < angle < 2 * pi in this calculation
    """
    cdef:
        DTYPE_t dot_xy, norm_x, norm_y, _cos
        int i
    #dot_xy = np.dot(x, y)
    #norm_x = np.linalg.norm(x)
    #norm_y = np.linalg.norm(y)
    dot_xy, norm_x, norm_y = 0.0, 0.0, 0.0
    for i in range(len(x)):
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]
        dot_xy += x[i] * y[i]
    norm_x = np.sqrt(norm_x)
    norm_y = np.sqrt(norm_y)

    _cos = dot_xy / (norm_x * norm_y)
    if _cos > 1.0:
        return 0.0
    elif _cos < -1.0:
        return np.pi
    return np.arccos(_cos)
#cpdef np.ndarray calc_nADD(UIeq, DTYPE_t minimumA, np.ndarray nADDnext, DTYPE_t betainv, int dim):
#    cdef:
#        DTYPE_t    _Adamp, _d,
#        np.ndarray nADDnextdamp
#
#    _Adamp       = deltaA_UI(UIeq.covinv, UIeq.K, UIeq.ave, UIeq.ref,
#                UIeq.ave, UIeq.ave + nADDnext, betainv)
#    _d           = 1.0
#    while _Adamp < minimumA:
#        _d      *= 2.0
#        nADDnextdamp = nADDnext * _d
#        #_Adamp       = UIeq.deltaA_UI(UIeq.ave, UIeq.ave + nADDnextdamp)
#        _Adamp       = deltaA_UI(UIeq.covinv, UIeq.K, UIeq.ave, UIeq.ref,
#                        UIeq.ave, UIeq.ave + nADDnextdamp, betainv)
#    _delta       = copy.copy(_d)
#    nADDnextdamp = copy.copy(nADDnext)
#    while abs(_Adamp - minimumA) > 0.0001:
#        #print("abs(_Adamp - minimumA) = %s"%abs(_Adamp - minimumA))
#        #nADDnext_before = copy.copy(nADDnext)
#        _delta *= 0.5
#        if _Adamp > minimumA:
#            _d -= _delta
#        else:
#            _d += _delta
#        nADDnextdamp = nADDnext * _d
#        _Adamp       = deltaA_UI(UIeq.covinv, UIeq.K, UIeq.ave, UIeq.ref,
#                            UIeq.ave, UIeq.ave + nADDnextdamp, betainv)
#    return nADDnextdamp
#cpdef np.ndarray SperSphere_cartesian(np.ndarray eigN, np.ndarray eigV, UIeq,
cpdef np.ndarray SperSphere_cartesian(np.ndarray eigN, np.ndarray eigV,
                np.ndarray covinv, np.ndarray K, np.ndarray ave, np.ndarray ref,
                DTYPE_t A, thetalist, DTYPE_t betainv, int dim,
                allperiodicQ, np.ndarray periodicmax, np.ndarray periodicmin, DTYPE_t eps):
    cdef:
        np.ndarray xlist, SSvec, SSvecbefore
        DTYPE_t    theta
        DTYPE_t    cosS, a_k
        int        i, n
    _xlist     = [1.0 for i in range(dim)]
    #_thetalist = list(thetalist)
    #_thetalist.reverse()
    #for i, theta in enumerate(_thetalist):
        #cosS = np.cos(theta)
        #for n in range(dim - i - 1):
            #_xlist[n] *= cosS
        #_xlist[dim - i - 1] *= np.sin(theta)
    a_k = 1.0
    for i, theta in enumerate(thetalist):
        _xlist[i] = a_k * np.cos(theta)
        a_k *= np.sin(theta)
    _xlist[-1] = a_k 
    _xlist.reverse()
    xlist  = np.array(_xlist)
    #SSvec = SperSphere_cartesian_n(eigN, eigV, UIeq, A, xlist, betainv, allperiodicQ, periodicmax, periodicmin, eps)
    SSvec = SperSphere_cartesian_n(eigN, eigV, covinv, K, ave, ref, A, xlist, betainv, allperiodicQ, periodicmax, periodicmin, eps)
    return SSvec
cpdef np.ndarray SperSphere_polar(np.ndarray ave, np.ndarray eigN, np.ndarray eigV, np.ndarray xlist, DTYPE_t A, int dim):
    cdef:
        np.ndarray SQaxesinv, SSvec
        DTYPE_t    theta, _cos
        int        i, j
    SQaxesinv = np.array([
                np.sqrt(1.0 / eigN[i]) * eigV[i]
                for i in range(dim)
                ])
    SQaxesinv  = np.linalg.inv(SQaxesinv)
    SSvec      = xlist - ave
    SSvec      = SSvec / np.linalg.norm(SSvec) / np.sqrt(A * 2.0)
    #_xlist = np.zeros(dim)
    _xlist = [0.0 for i in range(dim)]
    for i in range(dim):
        for j in range(dim):
            _xlist[i] += SSvec[i] * SQaxesinv[j, i]
    _xlist.reverse()
    #_thetalist = np.array([1.0 for i in range(dim - 1)])
    _thetalist = [1.0 for i in range(dim - 1)]
    for i in range(dim - 2):
        r = 0.0
        for j in range(i, dim):
            r += _xlist[j] * _xlist[j]
        r = np.sqrt(r)
        if r == 0.0:
            print("Caution!: r = 0.0")
            _thetalist[i] = 0.0
        else:
            _cos = _xlist[i] / r 
            _thetalist[i] = np.arccos(_cos)
    r = np.sqrt(_xlist[-1] * _xlist[-1] + _xlist[-2] * _xlist[-2])
    if r == 0.0:
        print("Caution!: r = 0.0")
        _thetalist[-1] = 0.0
    else:
        _thetalist[-1] = np.arccos(_xlist[-2] / r)
    if _xlist[-1] < 0:
        _thetalist[-1] = np.pi * 2.0 - _thetalist[-1]
    return np.array(_thetalist)
#cpdef np.ndarray SperSphere_cartesian_n(np.ndarray eigN, np.ndarray eigV, UIeq
cpdef np.ndarray SperSphere_cartesian_n(np.ndarray eigN, np.ndarray eigV,
                np.ndarray covinv, np.ndarray K, np.ndarray ave, np.ndarray ref,
                DTYPE_t A, np.ndarray _xlist, DTYPE_t betainv,
                allperiodicQ, np.ndarray periodicmax, np.ndarray periodicmin, DTYPE_t eps):
    cdef:
        np.ndarray SSvec, SQaxes
        np.ndarray SS, SSdamp, SSdelta, SSdampbefore
        DTYPE_t    theta
        DTYPE_t    _Adamp, _Abeforedamp, _d
        DTYPE_t    cosS
        DTYPE_t    D_maha, sigma
        DTYPE_t    sigmadamp, sigmadelta, sigmadampbefore
        int        i, j, dim, whileN1, whileN2
    dim  = len(_xlist)
    D_maha = np.sqrt(A * 2.0)
    #sigma = calcsigma(_xlist, D_maha, UIeq.covinv)
    sigma = calcsigma(_xlist, D_maha, covinv)

    SSvec = _xlist * sigma
    sigmadamp = copy.copy(sigma)
    #SSvecbefore = np.zeros(dim)

    #_Adamp       = deltaA_UI(UIeq.covinv, UIeq.K, UIeq.ave, UIeq.ref,
                        #UIeq.ave, UIeq.ave + SSvec, betainv, 
                        #allperiodicQ, periodicmax, periodicmin)
    _Adamp       = deltaA_UI(covinv, K, ave, ref,
                        ave, ave + SSvec, betainv, 
                        allperiodicQ, periodicmax, periodicmin)
    _d           = 0.1
    whileN1 = 1
    #while abs(_Adamp - A) > eps:
    #while np.linalg.norm(SSvec - SSvecbefore) > eps:
    while True:
        whileN1 += 1
        if 10000 < whileN1:
            print("whileN1 over 10000")
            return False
        _d         *= 0.5
        if _d < eps:
            break
        sigmadelta     = sigma * _d
        #SSvecbefore = copy.copy(SSvec)
        if _Adamp > A:
            sigmadamp  -= sigmadelta
        else:
            sigmadamp  += sigmadelta
        SSvec        = _xlist * sigmadamp 
        _Abeforedamp = copy.copy(_Adamp)
        #_Adamp      += UIeq.deltaA_UI(UIeq.ave + SSvecbefore, UIeq.ave + SSvec)
        #_Adamp       = UIeq.deltaA_UI(UIeq.ave + SSvecbefore, UIeq.ave + SSvec)
        #_Adamp       = deltaA_UI(UIeq.covinv, UIeq.K, UIeq.ave, UIeq.ref,
                        #UIeq.ave, UIeq.ave + SSvec, betainv, 
                        #allperiodicQ, periodicmax, periodicmin)
        _Adamp       = deltaA_UI(covinv, K, ave, ref,
                        ave, ave + SSvec, betainv, 
                        allperiodicQ, periodicmax, periodicmin)
        _Adamplist   = sorted([_Adamp, _Abeforedamp])
        whileN2 = 1
        while A < _Adamplist[0] or _Adamplist[1] < A:
            whileN2 += 1
            if 100000 < whileN2:
                print("whileN2 over 100000")
                return False

            
            #SSvecbefore = copy.copy(SSvec)
            if _Adamp > A:
                sigmadamp  -= sigmadelta
            else:
                sigmadamp  += sigmadelta
            SSvec        = _xlist * sigmadamp
            _Abeforedamp = copy.copy(_Adamp)
            #_Adamp      += deltaA_UI(UIeq.covinv, UIeq.K, UIeq.ave, UIeq.ref,
                              #UIeq.ave + SSvecbefore, UIeq.ave + SSvec, betainv,
                              #allperiodicQ, periodicmax, periodicmin)
            #_Adamp       = deltaA_UI(UIeq.covinv, UIeq.K, UIeq.ave, UIeq.ref,
                        #UIeq.ave, UIeq.ave + SSvec, betainv, 
                        #allperiodicQ, periodicmax, periodicmin)
            _Adamp       = deltaA_UI(covinv, K, ave, ref,
                        ave, ave + SSvec, betainv, 
                        allperiodicQ, periodicmax, periodicmin)
            _Adamplist   = sorted([_Adamp, _Abeforedamp])
    return SSvec
cpdef DTYPE_t calcsigma(np.ndarray _vec, DTYPE_t D_maha, np.ndarray covinv):
    cdef:
        #DTYPE_t vecnorm, sigma, D, _delta, _Ddamp
        #np.ndarray Evec, sigmapoint
        #int i
        DTYPE_t vecnorm, a, sigma
        int dim
    dim = len(_vec)
    vecnorm = 0.0
    for i in range(dim):
        vecnorm += _vec[i] * _vec[i]
    if vecnorm == 0.0: 
        return 0.0
    vecnorm = np.sqrt(vecnorm)
    Evec      = _vec / vecnorm

    a = 0.0
    for i in range(dim):
        for j in range(dim):
            a += Evec[i] * covinv[i,j] * Evec[j]
    sigma = D_maha * D_maha / a
    sigma = np.sqrt(sigma)

    return sigma
#cpdef histoC(np.ndarray data, np.ndarray pointmin, 
#                np.ndarray histodelta, Nbin, int dim):
#    cdef:
#        np.ndarray point
#        int i, j
#    #Mmax = int(1.0/_bin)
#    histo = []
#    for point in data:
#        for histolist in histo:
#            if all(histolist[i] <= point[i] < histolist[i] + histodelta[i] for i in range(dim)):
#                histolist[-1] += 1
#                break
#        else:
#            hispoint = list(pointmin)
#            for i in range(dim):
#                for j in range(Nbin[i]):
#                    if hispoint[i] <= point[i] < hispoint[i] + histodelta[i]:
#                        #hispoint.append(Mpoint)
#                        break
#                    hispoint[i] += histodelta[i]
#            histo.append(hispoint + [1])
#    return histo
#cpdef A_WHAM(np.ndarray xi, np.ndarray ref, np.ndarray ave, np.ndarray K,
#                histo, np.ndarray histodelta,
#                np.ndarray periodicmax, np.ndarray periodicmin, DTYPE_t betainv,  int dim, allperiodicQ):
#    cdef:
#        np.ndarray xidamp, vec_ave, vec_xi
#        DTYPE_t avehisto, xihisto, w_ave, w_xi
#        int i, j
#    avehisto = findhispointC(ave, ref, histo, histodelta,
#                    periodicmax, periodicmin, dim, allperiodicQ)
#    if avehisto == 0.0:
#        print("in %s avehisto = 0"%ave)
#    if allperiodicQ:
#        xidamp = periodic(xi, periodicmax, periodicmin, ref, dim)
#    else:
#        xidamp = xi
#    xihisto  = findhispointC(xidamp, ref, histo, histodelta,
#                    periodicmax, periodicmin, dim, allperiodicQ)
#    if xihisto == 0.0:
#        return 1.0e30
#    vec_ave = ref - ave
#    vec_xi  = ref - xidamp
#    w_ave   = 0.0
#    w_xi    = 0.0
#    for i in range(dim):
#        for j in range(dim):
#            w_ave += vec_ave[i] * K[i,j] * vec_ave[j]
#            w_xi  += vec_xi[i]  * K[i,j] * vec_xi[j]
#    return -betainv * np.log(xihisto / avehisto) - 0.5 * (w_xi - w_ave)
#cpdef findhispointC(np.ndarray xi, np.ndarray ref, np.ndarray histo, np.ndarray histodelta,
#                np.ndarray periodicmax, np.ndarray periodicmin, int dim, allperiodicQ):
#    cdef:
#        np.ndarray xidamp
#        int i
#        DTYPE_t returnH
#    if allperiodicQ:
#        xidamp = periodic(xi, periodicmax, periodicmin, ref, dim)
#    else:
#        xidamp = xi
#    for histolist in histo:
#        if all(histolist[i] <= xidamp[i] < histolist[i]  + histodelta[i] for i in range(dim)):
#            break
#    else:
#        print("In findhispointC: There is not %s point"%xidamp)
#        return 0.0
#    histopointpars = []
#    for i in range(dim):
#        if histolist[i] + histodelta[i] * 0.5 < xidamp[i]:
#            histopointpars.append([histolist[i], histolist[i] + histodelta[i]])
#        else:
#            histopointpars.append([histolist[i], histolist[i] - histodelta[i]])
#    histolistlist = []
#    for prod in itertools.product(range(2), repeat=dim):
#        xiprod = []
#        for i in range(dim):
#            xiprod.append(histopointpars[i][prod[i]])
#        for histolist in histo:
#            if all(histolist[i] == xiprod[i] for i in range(dim)):
#                histolistlist.append(histolist)
#                break
#        else:
#            histolistlist.append(xiprod + [0])
#    returnH = LShisto(histolistlist, histodelta, xidamp, dim)
#    if returnH < 0.0:
#        print("returnH = %s < 0.0"%returnH)
#        return 0.0
#    else:
#        return returnH
#cpdef LShisto(histolistlist, np.ndarray histodelta, np.ndarray xi, int dim):
#    cdef:
#        np.ndarray LSmat, Zvec, histopoint, xvec
#        int i, j
#    LSmat = np.zeros((dim + 1, dim + 1))
#    Zvec  = np.zeros(dim + 1)
#    for histolist in histolistlist:
#        histopoint = np.zeros(dim)
#        for i in range(dim):
#            histopoint[i] = histolist[i] + 0.5 * histodelta[i]
#        LSmat[0, 0] += 1.0
#        Zvec[0]     += histolist[-1]
#        for i in range(dim):
#            LSmat[0,     i + 1] += histopoint[i]
#            LSmat[i + 1, 0]     += histopoint[i]
#            Zvec[i + 1]         += histolist[-1] * histopoint[i]
#            for j in range(dim):
#                LSmat[i + 1, j + 1] += histopoint[i] * histopoint[j]
#    
#    #xvec = np.dot(np.linalg.inv(LSmat), Zvec)
#    LSmat = np.linalg.inv(LSmat)
#    xvec = np.zeros(dim + 1)
#    for i in range(dim):
#        for j in range(dim):
#            xvec[i] += LSmat[i, j] * Zvec[j]
#    returnH = xvec[0]
#    for i in range(dim):
#        returnH += xi[i] * xvec[i + 1]
#    return returnH
cpdef DTYPE_t Box_Muller():
    cdef:
        DTYPE_t r1, r2
    r1 = random.random()
    r2 = random.random()
    return np.sqrt(-2.0 * np.log(r1)) * np.cos( 2.0 * np.pi * r2)
cpdef np.ndarray random_hyper_sphereC(int dim):
    cdef:
        np.ndarray point
        DTYPE_t x, r
    point = np.array([Box_Muller() for _ in range(dim)]) 
    #r     = np.sqrt(sum(x * x for x in point))
    r = 0.0
    for x in point:
        r += x * x
    r = np.sqrt(r)
    point = point / r
    return point
#def getneedwindowC(UIlist, UIlist_initial, 
#                np.ndarray initialpoint, np.ndarray finishpoint, np.ndarray nADD,
#                DTYPE_t ADDstepsigmaTH, DTYPE_t minimizeTH, DTYPE_t nextstepsigmamaxTH, 
#                allperiodicQ, np.ndarray periodicmin, np.ndarray periodicmax, int dim):
#    cdef:
#        np.ndarray initialP, finalP, nDeltabefore, nADDchk, needwindowpoint
#        np.ndarray n
#        DTYPE_t sigma, Dmindamp, Dmin_minimum, _Dbias, S
#        int whileN, i, partdim
#    partOPTQ = False
#    partdim  = dim
#    initialP        = copy.copy(initialpoint)
#    needwindowpoint = copy.copy(initialpoint)
#    nDeltabefore    = np.zeros(dim)
#    nADDchk         = np.zeros(dim)
#    if cDmin(UIlist_initial, finishpoint) < ADDstepsigmaTH:
#        acceptQ         = True
#        return acceptQ, needwindowpoint
#    acceptQ = False
#    for i in range(11):
#        needwindowpoint = initialpoint + i * 0.10 * nADD
#        #periodicpoint(needwindowpoint, {})
#        if allperiodicQ:
#            needwindowpoint = periodic(needwindowpoint,
#                periodicmax, periodicmin, np.zeros(dim), dim)
#        if ADDstepsigmaTH < cDmin(UIlist_initial, needwindowpoint):
#            #with open("./UIlistdata.txt", "a")  as wf:
#                #wf.write("cDmin(needwindowpoint) = %s\n"%cDmin(UIlist, needwindowpoint))
#            #acceptQ = False
#            #break
#            UIlist_initial = []
#            for UI in UIlist:
#                _Dbias = Dbias(needwindowpoint, allperiodicQ, periodicmax, periodicmin,
#                                UI.ave, UI.covinv, dim, partOPTQ, partdim)
#                if _Dbias < ADDstepsigmaTH:
#                    UIlist_initial.append(UI)
#            if not UIlist_initial:
#                acceptQ         = False
#                return acceptQ, needwindowpoint
#            if cDmin(UIlist_initial, finishpoint) < ADDstepsigmaTH:
#                acceptQ         = True
#                return acceptQ, needwindowpoint
#    return acceptQ, needwindowpoint
def getneedwindowC(UIlist, UIlist_initial, 
                np.ndarray initialpoint, np.ndarray finishpoint, np.ndarray nADD,
                DTYPE_t ADDstepsigmaTH, DTYPE_t minimizeTH, DTYPE_t nextstepsigmaTH, 
                allperiodicQ, np.ndarray periodicmin, np.ndarray periodicmax, int dim):
    cdef:
        np.ndarray initialP, finalP, nDeltabefore, nADDchk, needwindowpoint, returnpoint
        np.ndarray n
        DTYPE_t    sigma, Dmindamp, Dmin_max, _Dbias, S
        int        whileN, i, partdim
    #Dmindamp = cDmin(UIlist, finishpoint)
    #if Dmindamp < ADDstepsigmaTH:
        #acceptQ = True
        #return acceptQ, finishpoint
    acceptQ = False
    i = 0
    returnpoint = copy.copy(finishpoint)
    #Dmin_max = copy.copy(Dmindamp)
    Dmin_max = 0.0
    #nADD = nADD / np.linalg.norm(nADD)
    while True:
        i += 1
        if 1000 < i:
            print("i over 1000: Dmin_max = %s"%Dmin_max)
            acceptQ = False
            return acceptQ, returnpoint
            #if ADDstepsigmaTH < Dmin_max:
                #acceptQ         = False
                #return acceptQ, returnpoint
            #else:
        needwindowpoint = initialpoint + (i * 0.1) * nADD
        if allperiodicQ:
            needwindowpoint = periodic(needwindowpoint,
                periodicmax, periodicmin, np.zeros(dim), dim)
        needwindowD = cDmin(UIlist, needwindowpoint)
        if nextstepsigmaTH < needwindowD:
            acceptQ         = False
            #print("i = %s: Dmin_max = %s"%(i,Dmin_max))
            #return acceptQ, returnpoint
            return acceptQ, needwindowpoint
        if Dmin_max < needwindowD: 
            #print("needwindowD = %s"%needwindowD)
            returnpoint = copy.copy(needwindowpoint)
            Dmin_max    = copy.copy(needwindowD)
        if 10 < i:
            if needwindowD < ADDstepsigmaTH:
                acceptQ = False
                return acceptQ, returnpoint
        if i == 10:
            #print("i is 100: Dmin_max = %s"%Dmin_max)
            if Dmin_max < ADDstepsigmaTH:
                #print("accept")
                acceptQ = True
                return acceptQ, finishpoint
            #elif needwindowD < nextstepsigmaTH:
            elif needwindowD < Dmin_max:
                acceptQ = False
                return acceptQ, returnpoint
        #else:
            #print("i = %s"%i)
