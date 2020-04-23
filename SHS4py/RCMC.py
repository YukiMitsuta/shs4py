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

def main():
    """
    
    """
    k_B  = 1.38065e-26
    Temp = 298.0
    N_A  = 6.002214e23
    betainv     = Temp * k_B * N_A # 1/beta (kJ / mol)

    jobfilepath = "../cpu_7D400nsSHS3/jobfiles_meta/"


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
        tslist = []
        for line in open(jobfilepath + "tslist.csv"):
            if line[0] == "#":
                continue
            line = line.split(",")
            tsname = line[0]
            tslist.append(tsname)
            FE_dic[tsname]      = float(line[-1].replace("\n", ""))
        connectionlist = []
        for line in open(jobfilepath + "connections.csv"):
            if line[0] == "#":
                continue
            line = line.replace("\n","").split(", ")
            connectionlist.append(line)
        fricfactorlist = []
        for velcsv in glob.glob(jobfilepath + "../velocitydir/*.csv"):
            for line in open(velcsv):
                line = line.split(", ")
                line[-1] = float(line[-1])
                if "EQ" in line[0]:
                    fricfactorlist.append(line)
                else:
                    fricfactorlist.append([line[1], line[0], line[-1]])
        dim  = len(eqlist)
        Kmat = np.zeros((dim, dim))
        for tsname in tslist:
            tsconnection = []
            klist = []
            for connection in connectionlist:
               if connection[0] == tsname:
                    tsconnection.append(connection[1])
            if len(tsconnection) != 2:
                print("tsname = %s"%tsname)
                continue
            tsconnection =[(tsconnection[0],tsconnection[1]),
                           (tsconnection[1],tsconnection[0])]
            for initial_EQ, final_EQ in tsconnection:
                Delta_fe = np.abs(FE_dic[initial_EQ] - FE_dic[tsname])
                for fricfactor in fricfactorlist:
                    if fricfactor[0] == initial_EQ and fricfactor[1] == tsname:
                        break
                else:
                    print("ERROR: there is not fric factor between %s -> %s"%(initial_EQ, tsname), flush = True)
                    continue
                k = fricfactor[-1] * np.exp( - Delta_fe / betainv)
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
        stateQlistlist = [list(np.identity(len(eqlist)))[:200]]
    
        n = 0
        if not os.path.exists("RCMCresult"):
            os.mkdir("RCMCresult")
        with open("./RCMCresult/SSlist.csv", "w") as wf:
            wf.write("%3d"%n)
            for SSname in SSlistlist[0]:
                wf.write(", %s"%SSname)
            wf.write("\n")
        with open("./RCMCresult/stateN.csv", "w") as wf:
            wf.write("%3d"%n)
            for stateN in stateNlist[-1]:
                wf.write(", %s"%stateN)
            wf.write("\n")
        if not os.path.exists("RCMCresult/stateQ"):
            os.mkdir("RCMCresult/stateQ")
        for ip_index in range(len(stateQlistlist[0])):
            with open("./RCMCresult/stateQ/%04d.csv"%ip_index, "w") as wf:
                wf.write("%3d"%n)
                #print(stateQlistlist[0][ip_index])
                for stateQ in stateQlistlist[0][ip_index]:
                    wf.write(", %s"%stateQ)
                wf.write("\n")
        #with open("./RCMCresult/maxK.csv", "w") as wf:
            #wf.write("%3d, 0.0\n"%(n))
    else:
        eqlist         = None
        nodenumdic     = None
        FE_dic         = None
        tslist         = None
        connectionlist = None
        fricfactorlist = None
        SSlistlist     = None
        stateNlist     = None
        stateQlistlist = None
        Kmatlist       = None
    eqlist         = comm.bcast(eqlist,         root = 0)
    nodenumdic     = comm.bcast(nodenumdic,     root = 0)
    FE_dic         = comm.bcast(FE_dic,         root = 0)
    tslist         = comm.bcast(tslist,         root = 0)
    connectionlist = comm.bcast(connectionlist, root = 0)
    fricfactorlist = comm.bcast(fricfactorlist, root = 0)
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
        maxK = max([max(x) for x in Kmatlist[n]])
        if maxK == 0.0:
            print("maxK = 0.0")
            break
        if rank == root:
            with open("./RCMCresult/maxK.csv", "a") as wf:
                wf.write("%3d, %s\n"%(int(n + 1), maxK))
            #print("maxK = %4.3f ps "%(maxK), flush = True)
            print("1/maxK = %4.3f ps "%(1.0/maxK), flush = True)
        maxind = np.where(Kmatlist[n] == maxK)
        #print(maxind)
        maxind = maxind[0][0]
        if rank == root:
            with open("./RCMCresult/maxKindex.csv", "a") as wf:
                wf.write("%3d, %s\n"%(int(n + 1), maxind))
        #sig    = 1.0 / sum(x for x in Kmatlist[n][maxind])
        #print(sum(Kmatlist[n][:,maxind]))
        #print(sum(Kmatlist[n][maxind]))
        #sig    = 1.0 / sum(x for x in Kmatlist[n][:,maxind])
        sig    = 1.0 / sum(Kmatlist[n][maxind,i] for i in range(len(Kmatlist[n][maxind])))
        newSSlist = copy.copy(SSlistlist[-1])
        newSSlist.pop(maxind)
        SSlistlist.append(newSSlist)
        Kmatdash = calcdash(n, dim, maxind, sig, Kmatlist, stateNlist)
        stateNlist.append(calcstateN(n, dim, maxind, sig, stateNlist, Kmatlist))
        stateQlistlist.append(calcstateQ(n, dim, maxind, sig, stateQlistlist[n], Kmatlist))
        if rank == root:
            with open("./RCMCresult/SSlist.csv", "a") as wf:
                wf.write("%3d"%int(n + 1))
                for SSname in newSSlist:
                    wf.write(", %s"%SSname)
                wf.write("\n")
            with open("./RCMCresult/stateN.csv", "a") as wf:
                wf.write("%3d"%int(n + 1))
                for stateN in stateNlist[-1]:
                    wf.write(", %s"%stateN)
                wf.write("\n")
            for ip_index in range(len(stateQlistlist[-1])):
                with open("./RCMCresult/stateQ/%04d.csv"%ip_index, "a") as wf:
                    wf.write("%3d"%n)
                    for stateQ in stateQlistlist[-1][ip_index]:
                        wf.write(", %s"%stateQ)
                    wf.write("\n")
        Kmatlist.append(calcmatrix(n, dim, maxind, sig, Kmatlist, stateNlist, Kmatdash))
        if rank == root:
            writeline = ""
            for Kmatraw in Kmatlist[-1]:
                writeline += "%s"%Kmatraw[0]
                for KmatN in Kmatraw[1:]:
                    writeline += ", %s"%KmatN
                writeline += "\n"
            if not os.path.exists("RCMCresult/Kmatrix"):
                os.mkdir("RCMCresult/Kmatrix")
            with open("./RCMCresult/Kmatrix/%04d.csv"%int(n+1), "w") as wf:
                wf.write(writeline)
        n += 1

def calcdash(n, dim, maxind, sig, Kmatlist, stateNlist):
    Kmatdash = np.zeros((dim, dim))
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
            Kmatdash[k,l] = Kmatlist[n][beforek,beforel] + Kmatlist[n][beforek, maxind] * Kmatlist[n][maxind, beforel] * sig
    if size == 1:
        Kmatdash_return = Kmatdash
    else:
        Kmatdashall = comm.gather(Kmatdash, root = 0)
        if rank == root:
            Kmatdash_return = np.zeros([dim, dim])
            for Kmatdash_damp in Kmatdashall:
                Kmatdash_return += Kmatdash_damp
        else:
            Kmatdash_return = None
        Kmatdash_return = comm.bcast(Kmatdash_return, root = 0)
    return Kmatdash_return
def calcstateN(n,dim, maxind, sig, stateNlist, Kmatlist):
    newStateN = np.zeros(dim)
    for k in range(dim):
        if k < maxind:
            beforek = k
        else:
            beforek = k + 1
        newStateN[k] = stateNlist[n][beforek] + Kmatlist[n][maxind, beforek] * sig * stateNlist[n][maxind]
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
def calcmatrix(n, dim, maxind, sig, Kmatlist, stateNlist, Kmatdash):
    Kmat = np.zeros((dim, dim))
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
            Kmat[k, l] = Kmatdash[k,l] / (1.0 + sig * Kmatlist[n][beforek, maxind])
    if size == 1:
        Kmat_return = Kmat
    else:
        Kmatall = comm.gather(Kmat, root = 0)
        if rank == root:
            Kmat_return = np.zeros([dim, dim])
            for Kmat_damp in Kmatall:
                Kmat_return += Kmat_damp
        else:
            Kmat_return = None
        Kmat_return = comm.bcast(Kmat_return, root = 0)
    return Kmat_return
if __name__ == "__main__":
    main()

