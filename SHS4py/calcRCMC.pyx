# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.
import  numpy as np
cimport numpy as np
import copy

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

#cpdef np.ndarray calcdash(int dim, int maxind, DTYPE_t sig, np.ndarray Kmat_n,
cpdef calcdash(int dim, int maxind, DTYPE_t sig, np.ndarray Kmat_n,
                int size, int rank):
    cdef:
        #np.ndarray Kmatdash
        int mpicounter, k, l
    #Kmatdash = np.zeros((dim, dim))
    Kmatdash = []
    mpicounter = -1
    for k in range(dim):
        for l in range(dim):
            if k == l:
                continue
            mpicounter += 1
            if mpicounter % size != rank:
                continue
            if k < maxind:
                beforek = k
            else:
                beforek = k + 1
            if l < maxind:
                beforel = l
            else:
                beforel = l + 1
            #Kmatdash[k,l] = Kmat_n[beforek,beforel] + Kmat_n[beforek, maxind] * Kmat_n[maxind, beforel] * sig
            if Kmat_n[beforek, beforel] != 0.0:
                #mpicounter += 1
                #if mpicounter % size != rank:
                    #continue
                if Kmat_n[beforek, maxind] != 0.0 and  Kmat_n[maxind, beforel] != 0.0:
                    Kmatdash.append([k,l, Kmat_n[beforek,beforel] + Kmat_n[beforek, maxind] * Kmat_n[maxind, beforel] * sig])
                else:
                    Kmatdash.append([k,l, Kmat_n[beforek,beforel] ])
            elif Kmat_n[beforek, maxind] != 0.0 and  Kmat_n[maxind, beforel] != 0.0:
                #mpicounter += 1
                #if mpicounter % size != rank:
                    #continue
                Kmatdash.append([k,l, Kmat_n[beforek, maxind] * Kmat_n[maxind, beforel] * sig])
    return Kmatdash
#cpdef np.ndarray calcmatrix(int dim, int maxind, DTYPE_t sig, np.ndarray Kmat_n, np.ndarray Kmatdash,
cpdef calcmatrix(int dim, int maxind, DTYPE_t sig, np.ndarray Kmat_n, np.ndarray Kmatdash,
                int size, int rank):
    cdef:
        #np.ndarray Kmat
        int mpicounter, k, l

    #Kmat = np.zeros((dim, dim))
    Kmat = []
    mpicounter = -1
    for k in range(dim):
        for l in range(dim):
            if k == l:
                continue
            mpicounter += 1
            if mpicounter % size != rank:
                continue
            if k < maxind:
                beforek = k
            else:
                beforek = k + 1
            if l < maxind:
                beforel = l
            else:
                beforel = l + 1
            #Kmat[k, l] = Kmatdash[k,l] / (1.0 + sig * Kmat_n[beforek, maxind])
            if Kmatdash[k,l] != 0.0:
                #mpicounter += 1
                #if mpicounter % size != rank:
                    #continue
                if Kmat_n[beforek, maxind] != 0.0:
                    Kmat.append([k,l, Kmatdash[k,l] / (1.0 + sig * Kmat_n[beforek, maxind])])
                else:
                    Kmat.append([k,l, Kmatdash[k,l]])
    return Kmat
