# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.
import  numpy as np
cimport numpy as np
import copy

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from libcpp.vector cimport vector
from libcpp.pair cimport pair

cpdef find_maxK(np.ndarray Kmat_row, np.ndarray Kmat_col, np.ndarray Kmat_data, int Pmaxindex1, int Pmaxindex2):
    cdef:
        vector[int]     Kmat_rowV  = Kmat_row
        vector[int]     Kmat_colV  = Kmat_col
        vector[DTYPE_t] Kmat_dataV = Kmat_data
        int i_row = Kmat_rowV.size()
        int i, maxind
        int k, l
        DTYPE_t maxK, sig
        vector[int] maxind_indexs
    maxK = 0.0
    for i in range(i_row):
        k = Kmat_rowV[i]
        l = Kmat_colV[i]
        if k == l:
            continue
        if k == Pmaxindex1 or k == Pmaxindex2:
            continue
        #if l == Pmaxindex1 or l == Pmaxindex2:
            #continue
        K_kl = Kmat_dataV[i]
        if maxK < K_kl:
            maxK   = K_kl
            maxind = k
    sig = 0.0
    if maxK != 0.0:
        for i in range(i_row):
            k = Kmat_rowV[i]
            l = Kmat_colV[i]
            if k == l:
                continue
            elif k == maxind:
                sig += Kmat_dataV[i]
                maxind_indexs.push_back(i)
            elif l == maxind:
                maxind_indexs.push_back(i)
        sig = 1.0 / sig

    return maxK, maxind, list(maxind_indexs), sig

cpdef calcdash(int maxind, DTYPE_t sig,
        np.ndarray Kmat_row, np.ndarray Kmat_col, np.ndarray Kmat_data,
        #np.ndarray Kmat_kl_row, np.ndarray Kmat_kl_col, np.ndarray Kmat_kl_data, 
        klist, llist,
                int size, int rank, DTYPE_t k_min):
    cdef:
        vector[int]     Kmat_rowV  = Kmat_row
        vector[int]     Kmat_colV  = Kmat_col
        vector[DTYPE_t] Kmat_dataV = Kmat_data
        vector[int]     klistV  = klist
        vector[int]     llistV  = llist
        int i_row = Kmat_rowV.size()
        int mpicounter, k, l, k_anker
        int next_k, next_l, i, j, index_Kmat, m
        DTYPE_t K_k_l, K_k_max, K_max_l
        pair[int, int] indexpair
        vector[pair[int, int]] Kmatdash_index
        vector[DTYPE_t] Kmatdash_data
    mpicounter = -1
    for k in klistV:
        if k == maxind:
            continue
        K_k_max = 0.0
        for i in range(i_row):
            if Kmat_rowV[i] == k and Kmat_colV[i] == maxind:
                K_k_max = Kmat_dataV[i]
                break
        for l in llistV:
            if k == l:
                continue
            if l == maxind:
                continue
            mpicounter += 1
            if mpicounter % size != rank:
                continue
            K_max_l = 0.0
            K_k_l   = 0.0
            for i in range(i_row):
                if Kmat_rowV[i] == maxind and Kmat_colV[i] == l:
                    K_max_l = Kmat_dataV[i]
                    break
            #for i in range(len(Kmat_kl_data)):
                #if Kmat_kl_row[i] == k and Kmat_kl_col[i] == l:
                    #K_k_l = Kmat_kl_data[i]
                    #break
            indexpair.first  = k
            indexpair.second = l
            if K_max_l != 0.0:
                K_k_l += K_k_max * K_max_l * sig
            #if k_min < K_k_l:
                #Kmatdash_index.push_back(indexpair)
                #Kmatdash_data.push_back(K_k_l)
            Kmatdash_index.push_back(indexpair)
            Kmatdash_data.push_back(K_k_l)
        #print("%s; len(Kmatdash_data) = %s"%(k, Kmatdash_data.size()))


#    mpicounter = -1
#    for i in range(i_row):
#        k = Kmat_rowV[i]
#        l = Kmat_colV[i]
#        if k == maxind:
#            continue
#        if l != maxind:
#            continue
#        mpicounter += 1
#        if mpicounter % size != rank:
#            continue
#        indexpair.first = k
#        k_anker = k
#        K_k_max = Kmat_dataV[i]
#        for j in range(i_row):
#            k = Kmat_rowV[j]
#            l = Kmat_colV[j]
#            if l == maxind:
#                continue
#            if k == maxind:
#                K_max_l = Kmat_dataV[j]
#                indexpair.second = l
#                if indexpair.first == indexpair.second:
#                    continue
#                K_k_l = K_k_max * K_max_l * sig
#                for m in range(i_row):
#                    k = Kmat_rowV[m]
#                    l = Kmat_colV[m]
#                    if indexpair.first == k and indexpair.second == l:
#                        K_k_l += Kmat_dataV[m]
#                        break
#                if k_min < K_k_l:
#                    Kmatdash_index.push_back(indexpair)
#                    Kmatdash_data.push_back(K_k_l)
#        for j in range(i_row):
#            k = Kmat_rowV[j]
#            if k != k_anker:
#                continue
#            l = Kmat_colV[j]
#            kdash_needQ = True
#            for indexpair in Kmatdash_index:
#                if indexpair.first == k and indexpair.second == l:
#                    kdash_needQ  = False
#                    break
#            if kdash_needQ:
#                indexpair.first  = k
#                indexpair.second = l
#                Kmatdash_index.push_back(indexpair)
#                K_k_l = 0.0
#                for m in range(i_row):
#                    k = Kmat_rowV[m]
#                    l = Kmat_colV[m]
#                    if indexpair.first == k and indexpair.second == l:
#                        K_k_l += Kmat_dataV[m]
#                        break
#                Kmatdash_data.push_back(K_k_l)

    return [[Kmatdash_index[i].first,
             Kmatdash_index[i].second,
             Kmatdash_data[i]] for i in range(Kmatdash_data.size())]
             #Kmatdash_data[i]] for i in range(Kmatdash_data.size())], list(maxind_indexs)
cpdef calcmatrix(DTYPE_t sig,
        np.ndarray Kmatbefore_row,  np.ndarray Kmatbefore_data,
        np.ndarray Kmatdash_row, np.ndarray Kmatdash_col, np.ndarray Kmatdash_data,
                int size, int rank, DTYPE_t k_min):
    cdef:
        vector[int]     Kmatdash_rowV  = Kmatdash_row
        vector[int]     Kmatdash_colV  = Kmatdash_col
        vector[DTYPE_t] Kmatdash_dataV = Kmatdash_data
        int i_Kmatdash = Kmatdash_rowV.size()
        DTYPE_t K_k_l
        int mpicounter, k,l
        unsigned int i, j
        pair[int, int] indexpair
        vector[pair[int, int]] Kmat_index
        vector[DTYPE_t] Kmat_data
    #mpicounter = -1

    for i in range(i_Kmatdash):
        #mpicounter += 1
        #if mpicounter % size != rank:
            #continue
        indexpair.first = Kmatdash_rowV[i]
        indexpair.second = Kmatdash_colV[i]
        for j in range(len(Kmatbefore_row)):
            if indexpair.first == Kmatbefore_row[j]:
                K_k_l = Kmatdash_dataV[i] / (1.0 + sig * Kmatbefore_data[j])
                if k_min < K_k_l:
                    Kmat_index.push_back(indexpair)
                    Kmat_data.push_back(K_k_l)
                else:
                    Kmat_index.push_back(indexpair)
                    Kmat_data.push_back(0.0)
                break
    return [[Kmat_index[i].first,
             Kmat_index[i].second,
             Kmat_data[i]] for i in range(Kmat_data.size())]
cpdef periodicpoint(np.ndarray x, np.ndarray periodicmax, np.ndarray periodicmin, np.ndarray beforepoint):
    cdef:
        int i
        np.ndarray bdamp
        DTYPE_t dis, point
    bdamp = np.zeros(len(x))
    if True:
        if True:
            for i in range(len(x)):
                if beforepoint is False:
                    if x[i] < periodicmin[i] or periodicmax[i] < x[i]:
                        x[i]  = (x[i] - periodicmax[i]) % (periodicmin[i] - periodicmax[i])
                        x[i] += periodicmax[i]
                    else:
                        bdamp[i] = x[i]
                else:
                    if bdamp[i] < periodicmin[i] + beforepoint[i] or periodicmax[i] + beforepoint[i] < bdamp[i]:
                        x[i]  = (x[i] - periodicmax[i] - beforepoint[i]) % (periodicmin[i] - periodicmax[i])
                        x[i] += periodicmax[i] + beforepoint[i]
                    else:
                        bdamp[i] = x[i]
    bdamp = bdamp - beforepoint
    dis = 0.0
    for point in bdamp:
        dis += point * point
    dis = np.sqrt(dis)
    return dis, bdamp
