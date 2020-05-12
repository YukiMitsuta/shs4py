#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020/01/10 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, sys, re
import copy
import subprocess as sp
import numpy as np


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

use_sparseQ = False

if use_sparseQ:
    from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
else:
    import pyximport  # for cython
    pyximport.install()
    try:
        from . import calcRCMC
    except ImportError:
        import calcRCMC

def main():
    """
    
    """
    k_B  = 1.38065e-26
    Temp = 298.0
    N_A  = 6.002214e23
    betainv     = Temp * k_B * N_A # 1/beta (kJ / mol)

    jobfilepath = "./jobfiles_meta/"

    FEmax = 40.0
    stateQN = 100


    if rank == root:
        eqlist     = []
        i          = 0
        nodenumdic = {}
        FE_dic     = {}
        for line in open(jobfilepath + "eqlist.csv"):
            if line[0] == "#":
                continue
            line   = line.split(",")
            eqname = line[0]
            eqlist.append(eqname)
            nodenumdic[eqname] = copy.copy(i)
            FE_dic[eqname]     = float(line[-1].replace("\n", ""))
            i += 1
        FEmin = min(FE_dic.values())
        tslist = []
        fricfactorlist = []
        for line in open(jobfilepath + "tslist.csv"):
            if line[0] == "#":
                continue
            line = line.split(",")
            tsname = line[0]
            #tslist.append(tsname)
            FE_dic[tsname]      = float(line[-1].replace("\n", ""))
            if FE_dic[tsname] - FEmin <= FEmax:
                tslist.append(tsname)
                fricpath = jobfilepath + "/%s/frac.csv"%tsname
                if os.path.exists(fricpath):
                    for line in open(fricpath):
                        fricfactorlist.append([tsname, float(line)])
        connectionlist = []
        for line in open(jobfilepath + "connections.csv"):
            if line[0] == "#":
                continue
            line = line.replace("\n","").split(", ")
            if not line[0] in FE_dic.keys():
                continue
            if FE_dic[line[0]] - FEmin <= FEmax:
                connectionlist.append(line)
        #for fricpath in glob.glob(jobfilepath + "/*/fric.csv"):
        #for fricpath in glob.glob(jobfilepath + "/*/frac.csv"):
            #TSname = fricpath.split("/")[-2]
            #for line in open(velcsv):
            #for line in open(fricpath):
                #fricfactorlist.append([TSname, float(line)])

                #line = line.split(", ")
                #line[-1] = float(line[-1])
                #if "EQ" in line[0]:
                    #fricfactorlist.append(line)
                #else:
                    #fricfactorlist.append([line[1], line[0], line[-1]])
        dim  = len(eqlist)
        if use_sparseQ:
            Kmat = lil_matrix((dim,dim), dtype=float)
        else:
            Kmat = np.zeros((dim, dim))
        fricfactorABS = np.mean([x[-1] for x in fricfactorlist])
        for tsname in tslist:
            tsconnection = []
            klist = []
            for connection in connectionlist:
               if connection[0] == tsname:
                    tsconnection.append(connection[1])
            if len(tsconnection) != 2:
                print("tsname = %s"%tsname)
                continue
            if not tsconnection[0] in FE_dic.keys() or not tsconnection[1] in FE_dic.keys():
                print("eqname = %s"%tsconnection)
                continue
            tsconnection =[(tsconnection[0],tsconnection[1]),
                           (tsconnection[1],tsconnection[0])]
            #for fricfactor in fricfactorlist:
            for fricTSname, fricfactor in fricfactorlist:
                    #if fricfactor[0] == initial_EQ and fricfactor[1] == tsname:
                    #break
                if tsname is fricTSname:
                    break
            else:
                print("ERROR: there is not fric factor in  %s"%(tsname), flush = True)
                print("replace to mean of factor (% 5.4f)"%fricfactorABS)
                #exit()
                #continue
                fricfactor = copy.copy(fricfactorABS)
            for initial_EQ, final_EQ in tsconnection:
                Delta_fe = np.abs(FE_dic[initial_EQ] - FE_dic[tsname])
                k = fricfactor * np.exp( - Delta_fe / betainv)
                #klist.append([tsconnection[0], k])
                #Kmat[nodenumdic[tsconnection[0]], nodenumdic[tsconnection[1]]] += k 
                klist.append([initial_EQ, k])
                Kmat[nodenumdic[initial_EQ], nodenumdic[final_EQ]] += k 
        SSlistlist = [eqlist]
        #stateNlist = [[np.exp(- float(x[-1]) / betainv) for x in eqlist]]
        stateNlist = [np.exp(- FE_dic[x] / betainv) for x in eqlist]
        totalN     = sum(stateNlist)
        stateNlist = [[x / totalN for x in stateNlist]]
        Kmatlist   = [Kmat]
        stateQlistlist = [list(np.identity(len(eqlist)))[:stateQN]]
    
        n = 0
        if not os.path.exists("RCMCresult"):
            os.mkdir("RCMCresult")
        writeline = "%3d"%int(n)
        for SSname in SSlistlist[0]:
            writeline += ", %s"%SSname
        writeline += "\n"
        with open("./RCMCresult/SSlist.csv", "w") as wf:
            wf.write(writeline)
        writeline = "%3d"%int(n)
        for stateN in stateNlist[-1]:
            writeline += ", %s"%stateN
        writeline += "\n"
        with open("./RCMCresult/stateN.csv", "w") as wf:
            wf.write(writeline)
        if not os.path.exists("RCMCresult/stateQ"):
            os.mkdir("RCMCresult/stateQ")
        for ip_index in range(len(stateQlistlist[-1])):
            writeline = "%3d"%int(n + 1)
            for steteQn, stateQ in enumerate(stateQlistlist[-1][ip_index]):
                if stateQ != 0.0:
                    writeline += ", %s, %s"%(steteQn, stateQ)
            writeline += "\n"
            with open("./RCMCresult/stateQ/%04d.csv"%ip_index, "a") as wf:
                wf.write(writeline)
        #for ip_index in range(len(stateQlistlist[0])):
            #with open("./RCMCresult/stateQ/%04d.csv"%ip_index, "w") as wf:
                #wf.write("%3d"%n)
                #print(stateQlistlist[0][ip_index])
                #for stateQ in stateQlistlist[0][ip_index]:
                    #wf.write(", %s"%stateQ)
                #wf.write("\n")
        #with open("./RCMCresult/maxK.csv", "w") as wf:
            #wf.write("%3d, 0.0\n"%(n))
    else:
        #eqlist         = None
        #nodenumdic     = None
        #FE_dic         = None
        #tslist         = None
        #connectionlist = None
        #fricfactorlist = None
        SSlistlist     = None
        stateNlist     = None
        stateQlistlist = None
        Kmatlist       = None
    if size != 1:
        #eqlist         = comm.bcast(eqlist,         root = 0)
        #nodenumdic     = comm.bcast(nodenumdic,     root = 0)
        #FE_dic         = comm.bcast(FE_dic,         root = 0)
        #tslist         = comm.bcast(tslist,         root = 0)
        #connectionlist = comm.bcast(connectionlist, root = 0)
        #fricfactorlist = comm.bcast(fricfactorlist, root = 0)
        SSlistlist     = comm.bcast(SSlistlist,     root = 0)
        stateNlist     = comm.bcast(stateNlist,     root = 0)
        stateQlistlist = comm.bcast(stateQlistlist, root = 0)
        Kmatlist       = comm.bcast(Kmatlist,       root = 0)

    n = 0

    while True:
        dim  = len(stateNlist[-1]) - 1
        if dim <= 1:
            break
        if 1 <= n:
            Kmatlist[n - 1]       = None
            SSlistlist[n - 1]     = None
            stateNlist[n - 1]     = None
            stateQlistlist[n - 1] = None
        #maxK = max([max(x) for x in Kmatlist[n][10:,10:]])
        #maxK = max([max(x) for x in Kmatlist[n]])
        #maxK = max([max(x) for x in Kmatlist[n]])
        if use_sparseQ:
            maxK = 0.0
            Kmat_coo = Kmatlist[n].tocoo()
            for ind in range(len(Kmat_coo.row)):
                k = Kmat_coo.row[ind]
                l = Kmat_coo.col[ind]
                K_kl = Kmatlist[n][k,l]
                if maxK < K_kl:
                    maxK = copy.copy(K_kl)
                    maxind = copy.copy(k)
        else:
            maxK = max([max(x) for x in Kmatlist[n]])
            maxind = np.where(Kmatlist[n] == maxK)
            maxind = maxind[0][0]
        if maxK == 0.0:
            print("maxK = 0.0")
            break
        if rank == root:
            with open("./RCMCresult/maxK.csv", "a") as wf:
                wf.write("%3d, %s\n"%(int(n + 1), maxK))
            #print("maxK = %4.3f ps "%(maxK), flush = True)
            print("1/maxK = %4.3f ps "%(1.0/maxK), flush = True)
        if rank == root:
            with open("./RCMCresult/maxKindex.csv", "a") as wf:
                wf.write("%3d, %s\n"%(int(n + 1), maxind))
        #sig    = 1.0 / sum(x for x in Kmatlist[n][maxind])
        #print(sum(Kmatlist[n][:,maxind]))
        #print(sum(Kmatlist[n][maxind]))
        #sig    = 1.0 / sum(x for x in Kmatlist[n][:,maxind])
        #sig    = 1.0 / sum(Kmatlist[n][maxind,i] for i in range(len(Kmatlist[n][maxind])))
        if use_sparseQ:
            sig    = 1.0 / sum(Kmatlist[n][maxind,i] for i in range(len(Kmat_coo.row)))
        else:
            sig    = 1.0 / sum(Kmatlist[n][maxind,i] for i in range(len(Kmatlist[n][maxind])))
        newSSlist = copy.copy(SSlistlist[-1])
        newSSlist.pop(maxind)
        SSlistlist.append(newSSlist)
        Kmatdash = calcdash(n, dim, maxind, sig, Kmatlist)
        stateNlist.append(calcstateN(n, dim, maxind, sig, stateNlist, Kmatlist))
        stateQlistlist.append(calcstateQ(n, dim, maxind, sig, stateQlistlist[n], Kmatlist))
        if rank == root:
            #writeline = ''
            writeline = "%3d"%int(n + 1)
            for SSname in newSSlist:
                writeline += ", %s"%SSname
            writeline += "\n"
            with open("./RCMCresult/SSlist.csv", "a") as wf:
                wf.write(writeline)
                #wf.write("%3d"%int(n + 1))
            writeline = "%3d"%int(n + 1)
            for stateN in stateNlist[-1]:
                writeline += ", %s"%stateN
            writeline += "\n"
            with open("./RCMCresult/stateN.csv", "a") as wf:
                wf.write(writeline)
                #wf.write("%3d"%int(n + 1))
                #for stateN in stateNlist[-1]:
                    #wf.write(", %s"%stateN)
                #wf.write("\n")
            for ip_index in range(len(stateQlistlist[-1])):
                writeline = "%3d"%int(n + 1)
                for steteQn, stateQ in enumerate(stateQlistlist[-1][ip_index]):
                    if stateQ != 0.0:
                        writeline += ", %s, %s"%(steteQn, stateQ)
                writeline += "\n"
                with open("./RCMCresult/stateQ/%04d.csv"%ip_index, "a") as wf:
                    wf.write(writeline)
                    #wf.write("%3d"%n)
                    #for steteQn, stateQ in enumerate(stateQlistlist[-1][ip_index]):
                        #wf.write(", %s"%stateQ)
                    #wf.write("\n")
        Kmatlist.append(calcmatrix(n, dim, maxind, sig, Kmatlist, Kmatdash))
#        if rank == root:
            #writeline = ""
#            for Kmatraw in Kmatlist[-1]:
#                writeline += "%s"%Kmatraw[0]
#                for KmatN in Kmatraw[1:]:
#                    writeline += ", %s"%KmatN
#                writeline += "\n"
            #for k in range(dim):
                #for l in range(dim):
                    #if Kmatlist[-1][k,l] != 0.0:
                        #writeline += "%s, %s, %s\n"%(k,l,Kmatlist[-1][k,l])
            #if not os.path.exists("RCMCresult/Kmatrix"):
                #os.mkdir("RCMCresult/Kmatrix")
            #with open("./RCMCresult/Kmatrix/%04d.csv"%int(n+1), "w") as wf:
                #wf.write(writeline)
        n += 1

def calcdash(n, dim, maxind, sig, Kmatlist):
    #if True:
    if not use_sparseQ:
        if rank == root:
            print("start calcRCMC.calcdash", flush = True)
        Kmatdash = calcRCMC.calcdash(dim, maxind, sig, Kmatlist[n], size, rank)
        if rank == root:
            print("end calcRCMC.calcdash", flush = True)
    else:
        #Kmatdash = np.zeros((dim, dim))
        #Kmatdash = lil_matrix((dim,dim))
        Kmatdash = lil_matrix((dim,dim), dtype = float)
        Kmat_coo = Kmatlist[n].tocoo()
        mpicounter = -1
        for k in range(dim):
            mpicounter += 1
            if mpicounter % size != rank:
                continue
            for l in range(dim):
                if k == l:
                    continue
                if k < maxind:
                    beforek = k
                else:
                    beforek = k + 1
                if l < maxind:
                    beforel = l
                else:
                    beforel = l + 1
                if beforek in Kmat_coo.row and beforel in Kmat_coo.col:
                    if maxind in Kmat_coo.row and maxind in Kmat_coo.col:
                        Kmatdash[k,l] = Kmatlist[n][beforek,beforel] + Kmatlist[n][beforek, maxind] * Kmatlist[n][maxind, beforel] * sig
                    else:
                        Kmatdash[k,l] = Kmatlist[n][beforek,beforel] 
                elif maxind in Kmat_coo.row and maxind in Kmat_coo.col:
                    Kmatdash[k,l] = Kmatlist[n][beforek, maxind] * Kmatlist[n][maxind, beforel] * sig
    if size == 1:
        Kmatdashall = [Kmatdash]
    else:
        Kmatdashall = comm.gather(Kmatdash, root = 0)
        Kmatdashall = comm.bcast(Kmatdashall, root = 0)
    if rank == root:
        if use_sparseQ:
            Kmatdash_return = lil_matrix((dim, dim), dtype = float)
            for Kmatdash_damp in Kmatdashall:
                Kmatdash_return += Kmatdash_damp
            print("space %s"%len(Kmatdash_return.row), flush = True)
        else:
            i = 0
            Kmatdash_return = np.zeros([dim, dim])
            for Kmatdash_damp in Kmatdashall:
                for k, l, Kmatdash_kl in Kmatdash_damp:
                    Kmatdash_return[k,l] += Kmatdash_kl
                    i += 1
            print("space %s"%i, flush = True)
    else:
        #Kmatdash_return = None
        Kmatdash_return = np.zeros([dim, dim])
        for Kmatdash_damp in Kmatdashall:
            for k, l, Kmatdash_kl in Kmatdash_damp:
                Kmatdash_return[k,l] += Kmatdash_kl
    #if size != 1:
        #Kmatdash_return = comm.bcast(Kmatdash_return, root = 0)
    return Kmatdash_return
def calcstateN(n,dim, maxind, sig, stateNlist, Kmatlist):
    #newStateN = np.zeros(dim)
    newStateNindex = []
    for k in range(dim):
        if k % size != rank:
            continue
        if k < maxind:
            beforek = k
        else:
            beforek = k + 1
        #newStateN[k] = stateNlist[n][beforek] + Kmatlist[n][maxind, beforek] * sig * stateNlist[n][maxind]
        newStateNindex.append([k, stateNlist[n][beforek] + Kmatlist[n][maxind, beforek] * sig * stateNlist[n][maxind]])
    newStateN = np.zeros(dim)
    if size == 1:
        newStateNall = [newStateNindex]
    else:
        newStateNall = comm.gather(newStateNindex, root = 0)
    if rank == root:
        for newStateNdamp in newStateNall:
            for k, newStateN_k in newStateNdamp:
                newStateN[k] += newStateN_k
    if size != 1:
        newStateN = comm.bcast(newStateN, root = 0)
    return newStateN
def calcstateQ(n, dim, maxind, sig, stateQlist, Kmatlist):
    newStateQlist = []
    for ip_index in range(len(stateQlist)):
        newStateQ = np.zeros(dim)
        for k in range(dim):
            if k < maxind:
                beforek = k
            else:
                beforek = k + 1
            newStateQ[k] = stateQlist[ip_index][beforek] + Kmatlist[n][maxind, beforek] * sig * stateQlist[ip_index][maxind]
        newStateQlist.append(newStateQ)
    return newStateQlist
def calcmatrix(n, dim, maxind, sig, Kmatlist, Kmatdash):
    #if True:
    #if False:
    if not use_sparseQ:
        if rank == root:
            print("start calcRCMC.calcmatrix", flush = True)
        Kmat= calcRCMC.calcmatrix(dim, maxind, sig, Kmatlist[n], Kmatdash, size, rank)
        if rank == root:
            print("end calcRCMC.calcmatrix", flush = True)
    else:
        #Kmat = np.zeros((dim, dim))
        Kmat = lil_matrix((dim,dim), dtype=float)
        Kmatdash_coo = Kmatdash.tocoo()
        Kmat_coo = Kmat.tocoo()
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
                if k in Kmatdash_coo.row and l in Kmatdash_coo.col:
                    if beforek in Kmat_coo.row and maxind in Kmat_coo.col:
                        Kmat[k, l] = Kmatdash[k,l] / (1.0 + sig * Kmatlist[n][beforek, maxind])
                    else:
                        Kmat[k, l] += Kmatdash[k,l]
                #Kmat[k, l] = Kmatdash[k,l] / (1.0 + sig * Kmatlist[n][beforek, maxind])
    if size == 1:
        Kmatall = [Kmat]
    else:
        Kmatall = comm.gather(Kmat,   root = 0)
        Kmatall = comm.bcast(Kmatall, root = 0)
        #Kmat_return = Kmat
    if rank == root:
        writeline = ""
        if use_sparseQ:
            Kmat_return = lil_matrix([dim, dim])
            for Kmat_damp in Kmatall:
                Kmat_return += Kmat_damp
            Kmat_coo = Kmat_return.tocoo()
            for ind in range(len(Kmat_coo.row)):
                k = Kmat_coo.row[ind]
                l = Kmat_coo.col[ind]
                writeline += "%6d, %6d, %s\n"%(k, l, Kmat_return[k,l])
            print("space %s"%len(Kmat_return), flush = True)
        else:
            i = 0
            Kmat_return = np.zeros([dim, dim])
            for Kmat_damp in Kmatall:
                for k, l, Kmat_kl in Kmat_damp:
                    Kmat_return[k,l] += Kmat_kl
                    writeline += "%6d, %6d, %s\n"%(k,l,Kmat_kl)
                    i += 1
            print("space %s"%i, flush = True)
        if not os.path.exists("RCMCresult/Kmatrix"):
            os.mkdir("RCMCresult/Kmatrix")
        with open("./RCMCresult/Kmatrix/%04d.csv"%int(n+1), "w") as wf:
            wf.write(writeline)
    else:
        Kmat_return = np.zeros([dim, dim])
        for Kmat_damp in Kmatall:
            for k, l, Kmat_kl in Kmat_damp:
                Kmat_return[k,l] += Kmat_kl
        #Kmat_return = None
        #for k in range(dim):
            #Kmatall = comm.gather(Kmat[k], root = 0)
            #if rank == root:
                #for Kmat_damp in Kmatall:
                    #Kmat_return[k] += Kmat_damp
    #if size != 1:
        #Kmat_return = comm.bcast(Kmat_return, root = 0)
    return Kmat_return
if __name__ == "__main__":
    main()

