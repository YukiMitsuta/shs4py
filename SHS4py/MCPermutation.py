#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020/08/28 MitsutaYuki
#
# Distributed under terms of the MIT license.
import os, glob, shutil, sys, re
import copy, time
import numpy as np

import networkx as nx

from scipy.sparse import lil_matrix, csr_matrix, coo_matrix

class StationalyPoint():
    def __init__(self,line):
        line    = line.split(",")
        self.name  = line[0]
        self.point = np.array(line[1:-1], dtype=float)
        self.z = int(self.point[-1])
        self.fe = float(line[-1])
def main(const):
    """

    """
    if const.calc_mpiQ:
        from mpi4py import MPI
        const.comm = MPI.COMM_WORLD
        const.rank = const.comm.Get_rank()
        const.size = const.comm.Get_size()
        const.root = 0
    else:
        const.rank = 0
        const.size = 1
        const.root = 0
    print("start Markov Chain")
    if not os.path.exists("./RCMCresult"):
        print("Error; there is not RCMCresult")
        exit()
    maxKindex = False
    n = 0
    for line in open("./RCMCresult/maxK.csv"):
        line = line.split(", ")
        k = float(line[1])
        if k < const.k_markov:
            n = int(line[0]) - 1
            break
    SSlist = []
    for line in open("./RCMCresult/SSlist.csv"):
        line = line.replace("\n","").split(", ")
        SSlist.append(line[-1])
    dim = len(SSlist)
    print("dim = %s"%dim)
#    chkpointname = "./RCMCresult/stateN/stateN_chkpoint%05d.csv"%int(n)
#    stateN = np.zeros(dim)
#    if os.path.exists(chkpointname):
#        for line in open(chkpointname):
#            line = line.split(",")
#            k    = int(line[0])
#            N_k  = float(line[1])
#            if const.stateNmin < N_k:
#                stateN[k] = N_k
#    else:
#        print("Error; there is not ./RCMCresult/Kmatrix/stateN_chkpoint%05d.csv"%int(n))
#        exit()
    #Kmat = lil_matrix((dim, dim), dtype=float)
    #Kmat = np.zeros((dim, dim), dtype=float)
    Kmat_lil = []
    chkpointname = "./RCMCresult/Kmatrix/Kmat_chkpoint%05d.csv"%int(n)
    indexset = set()
    if os.path.exists(chkpointname):
        for line in open("./RCMCresult/Kmatrix/Kmat_chkpoint%05d.csv"%int(n)):
            line = line.split(",")
            k    = int(line[0])
            l    = int(line[1])
            K_kl = float(line[2])
            if const.k_min < K_kl:
                #Kmat[k,l] = K_kl * const.t_delta
                #Kmat[k][l] = K_kl * const.t_delta
                Kmat_lil.append([k,l,K_kl*const.t_delta])
                indexset.add(k)
                indexset.add(l)
    else:
        print("Error; there is not ./RCMCresult/Kmatrix/Kmat_chkpoint%05d.csv"%int(n))
        exit()
    eqpointlist = []
    Z_dic = {}
    FE_dic = {}
    FE_dic_zmin = {z:1.0e30 for z in range(36)}
    for line in open(const.jobfilepath + "eqlist.csv"):
        if line[0] == "#":
            continue
        eqpoint = StationalyPoint(line)
        eqpointlist.append(eqpoint)
        Z_dic[eqpoint.name] = eqpoint.z
        if eqpoint.fe < FE_dic_zmin[eqpoint.z]:
            FE_dic_zmin[eqpoint.z] = eqpoint.fe
        if eqpoint.z == 0.0:
            eqpointname = eqpoint.name
            Z_dic[eqpointname] = eqpoint.z
            FE_dic[eqpointname] = eqpoint.fe
        else:
            eqpointname = eqpoint.name + "+"
            Z_dic[eqpointname] = eqpoint.z
            FE_dic[eqpointname] = eqpoint.fe
            eqpointname = eqpoint.name + "-"
            Z_dic[eqpointname] = - eqpoint.z
            FE_dic[eqpointname] = eqpoint.fe
    eqpointlist.sort(key=lambda x:x.fe)
    femin = eqpointlist[0].fe
    removepoint = []
    newSSlist = []
    for i, SSname in enumerate(SSlist):
        z = Z_dic[SSname]
        if 30 < z:
            removepoint.append(i)
        elif z < -30:
            removepoint.append(i)
        elif not i in indexset:
            removepoint.append(i)
        elif const.EQFEmax < FE_dic[SSname] - FE_dic_zmin[np.abs(z)]:
            removepoint.append(i)

        else:
            newSSlist.append(SSname)
    #removepoint = []
    print("len(Kmat_lil) = ",len(Kmat_lil))
    print("len(removepoint) = ",len(removepoint))
    newKmat_lil = copy.deepcopy(Kmat_lil)
    #for i,(k,l,K_kl) in enumerate(Kmat_lil):
    for i in range(len(Kmat_lil)):
        k = Kmat_lil[i][0]
        l = Kmat_lil[i][1]
        #print(K_kl)
        for remove_i in removepoint:
            if k == remove_i or l == remove_i:
                newKmat_lil[i][2] = 0.0
            if remove_i<k:
                newKmat_lil[i][0] -= 1
            if remove_i<l:
                newKmat_lil[i][1] -= 1
    Kmat_lil = newKmat_lil
    print(Kmat_lil[-1])
    print("len(SSlist) = ",len(SSlist))
    print("len(newSSlist) = ",len(newSSlist))
    #exit()
    SSlist = newSSlist
    dim = len(SSlist)
    Kmat = np.zeros((dim, dim), dtype=np.float)
    for k,l,K_kl in Kmat_lil:
        if K_kl != 0.0:
            Kmat[k,l] = K_kl
            #Kmat[l,l] = -K_kl

    sN_outsideP = - 1.0e30
    sN_outsideM = - 1.0e30
    outsidePlist = []
    outsideMlist = []
    for i, SSname in enumerate(SSlist):
        z = Z_dic[SSname]
        #fe = FE_dic[SSname]
        #sN = stateN[i]
        #if sN == 0:
            #continue
        #if 25.0 == z:
        if 30 == z:
        #if 24.5 < z:
        #if 0.0 < z:
        #if 30.0 < z:
            outsidePlist.append(i)
            print("plus;%s, %s"%(z,SSname))
            #if sN_outsideP < sN:
                #sN_outsideP = sN
                #outsideP = SSname
                #outsideP_i = i
                #outsidePlist = [i]
        elif z == -30:
        #elif z < - 24.5:
        #elif z < - 30.0:
        #elif z < -10.0:
        #elif z < 0.0:
            outsideMlist.append(i)
            print("minus;%s, %s"%(z,SSname))
            #if sN_outsideM < sN:
                #sN_outsideM = sN
                #outsideM = SSname
                #outsideM_i = i
                #outsideMlist = [i]
        #elif 30.0 < z:
            #for j in range(dim):
                #Kmat[i, j] = 0.0
                #Kmat[j, i] = 0.0
        #elif z < -30.0:
            #for j in range(dim):
                #Kmat[i, j] = 0.0
                #Kmat[j, i] = 0.0
    stateQ = np.zeros(dim, dtype=float)
    totalP = 0.0
    for outsideP_i in outsidePlist:
        stateQ[outsideP_i] = np.exp(-const.beta * FE_dic[SSlist[outsideP_i]])
        totalP += stateQ[outsideP_i]
    for outsideP_i in outsidePlist:
        stateQ[outsideP_i] /= totalP

    stateQ_total = sum(stateQ)
    stateQ = stateQ/stateQ_total
    print("stateQ = %s"%sum(stateQ))
    for outsideM_i in outsideMlist:
        for i in range(dim):
            Kmat[outsideM_i, i] = 0.0

    #Kmat_coo = Kmat.tocoo()
    diagnorm = np.zeros(dim)
    for k in range(dim):
        for l in range(dim):
            if k != l:
                diagnorm[k] -= Kmat[k,l]

#    for i in range(len(Kmat_coo.data)):
#        k = Kmat_coo.row[i]
#        diagnorm[k] -= Kmat_coo.data[i]
    
    for i in range(dim):
        if abs(diagnorm[i])>1.0:
            raise ValueError("the diagnorm is larger than 1.0")
        Kmat[i, i] = 1.0 + diagnorm[i]
        #Kmat[i, i] += 1.0
    Kmat = Kmat.transpose()
    Kmat = np.array(Kmat, dtype=np.cdouble)
    #kSimdic = "kSim_%s"%const.k_RCMC
    kSimdic = "kSimulation_MC"
    if not os.path.exists(kSimdic):
        os.mkdir(kSimdic)
    os.chdir(kSimdic)
    if os.path.exists("./eigenVal.npy") and os.path.exists("./eigenVec.npy"):
        vals = np.load("./eigenVal.npy")
        eigvs = np.load("./eigenVec.npy")
    else:
        vals, eigvs = np.linalg.eig(Kmat)
        np.save("./eigenVal.npy", vals)
        np.save("./eigenVec.npy", eigvs)
    print("eigen vectors are obtained")

    eiglist = []
    for i, val in enumerate(vals):
        eigdic = {}
        eigdic["val"] = val
        eigdic["vec"] = eigvs[:,i]
        eiglist.append(eigdic)
    calct = 0.0
    writeline = ""
    for SSname in SSlist:
        writeline += "%s, %s, %s\n"%(SSname,Z_dic[SSname],FE_dic[SSname])
    with open("./SSpoints.csv", "w") as wf:
        wf.write(writeline)
    writeline = ""
    for k in range(dim):
        for l in range(dim):
            if Kmat[k,l].real != 0.0:
                writeline += "%s, %s, %s\n"%(k,l,Kmat[k,l].real)
    with open("./Kmat.csv", "w") as wf:
        wf.write(writeline)

    print("t_delta = ", const.t_delta)
    #for totaltime in const.totaltimes:
    if os.path.exists("./eigenVec_inv.npy"):
        eigvsinv = np.load("./eigenVec_inv.npy")
    else:
        eigvsinv = np.linalg.inv(eigvs)
        np.save("./eigenVec_inv.npy", eigvsinv)
    while calct < const.totaltime:
        calct += const.time_stride
        stepN = np.double(calct/const.t_delta)
        print("calct = ", calct)
        #print("stepN = ", stepN)
        #os.makedirs("MCain_%sps"%const.totaltime, exist_ok=True)
        eiglistdamp = []
        for i in range(dim):
            #print(eiglist[i]["val"])
            #eiglist[i]["vec"] = eiglist[i]["vec"]*(eiglist[i]["val"]**stepN)
            eiglistdamp.append(eiglist[i]["vec"]*(eiglist[i]["val"]**stepN))
        #exit()
        c = eigvsinv@stateQ
        p_t = np.zeros(dim, dtype=np.cdouble)
        for i in range(dim):
            #p_t += eiglist[i]["vec"]*c[i]
            p_t += eiglistdamp[i]*c[i]
        p_t = np.array([np.abs(x) for x in p_t], dtype=np.double)
        p = p_t
        ptot = np.sum(np.abs(a_p) for a_p in p_t)
        print("ptot = % 16.15f"%ptot,flush=True)
        p /= ptot
        totalP_plus  = 0.0
        totalP_minus = 0.0
        for outsideP_i in outsidePlist:
            totalP_plus += p_t[outsideP_i]
        for outsideM_i in outsideMlist:
            totalP_minus += p_t[outsideM_i]
        with open("./throughstate.csv", "a") as wf:
            wf.write("%10.7f, %s, %s\n"%(
                calct/10**9, totalP_plus, totalP_minus))
        writeline = "%s,"%(calct/10**9)
        for i in range(dim):
            writeline += "%s,"%p_t[i]
        writeline =writeline.rstrip(",") + "\n"
        with open("./allQplot.csv", "a") as wf:
            wf.write(writeline)
        if 0.990 < totalP_minus:
            break


def calc_ksim(ssN,stepN, Kmat, dim):
    p = np.zeros(dim, dtype = float)
    p[ssN] = 1.0
    whileN = 1
    Kmatdamp = np.array(Kmat, dtype=float)
    while whileN < stepN:
        whileN += 1
        p = Kmatdamp@p
        #Kmat= Kmat@Kmatdamp
        #for i in range(10):
            #Kmatdamp = Kmatdamp@Kmat
            #print(i)
        #Kmat = Kmatdamp
        #Kmatdamp = Kmat
        #whileN *= 10
        if whileN % 1000 == 0:
        #if True:
            ptot = np.abs(np.sum(p))
            print("ptot(%s) = % 16.15f"%(whileN, ptot),flush=True)
        #print(whileN)
    return p
