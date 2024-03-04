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
#from scipy.sparse import linalg
#from scipy import linalg

class StationalyPoint():
    def __init__(self,line):
        line    = line.split(",")
        self.name  = line[0]
        self.point = np.array(line[1:-1], dtype=float)
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
    chkpointname = "./RCMCresult/stateN/stateN_chkpoint%05d.csv"%int(n)
    stateN = np.zeros(dim)
    if os.path.exists(chkpointname):
        for line in open(chkpointname):
            line = line.split(",")
            k    = int(line[0])
            N_k  = float(line[1])
            if const.stateNmin < N_k:
                stateN[k] = N_k
    else:
        print("Error; there is not ./RCMCresult/Kmatrix/stateN_chkpoint%05d.csv"%int(n))
        exit()
    Kmat = lil_matrix((dim, dim), dtype=float)
    chkpointname = "./RCMCresult/Kmatrix/Kmat_chkpoint%05d.csv"%int(n)
    if os.path.exists(chkpointname):
        for line in open("./RCMCresult/Kmatrix/Kmat_chkpoint%05d.csv"%int(n)):
            line = line.split(",")
            k    = int(line[0])
            l    = int(line[1])
            K_kl = float(line[2])
            if const.k_min < K_kl:
            #if True:
                #Kmat[k,l] = K_kl 
                Kmat[k,l] = K_kl * const.t_delta
                #Kmat[k,l] = K_kl #* const.t_delta
    else:
        print("Error; there is not ./RCMCresult/Kmatrix/Kmat_chkpoint%05d.csv"%int(n))
        exit()
    eqpointlist  = []
    for line in open(const.jobfilepath + "eqlist.csv"):
        if line[0] == "#":
            continue
        eqpoint = StationalyPoint(line)
        eqpointlist.append(eqpoint)
    eqpointlist.sort(key=lambda x:x.fe)
    femin = eqpointlist[0].fe

    Kmat_coo = Kmat.tocoo()
    diagnorm = np.zeros(dim)
    for i in range(len(Kmat_coo.data)):
        k = Kmat_coo.row[i]
        diagnorm[k] -= Kmat_coo.data[i]
    for i in range(dim):
        if abs(diagnorm[i])>1.0:
            raise ValueError("the diagnorm is larger than 1.0")
        Kmat[i, i] = 1.0 + diagnorm[i]
    Kmat = Kmat.transpose()
    Kmat = np.array(Kmat.toarray(), dtype=np.cdouble)
    #Kmat = np.array(Kmat.toarray(), dtype=np.clongdouble)
    if os.path.exists("./eigenVal.npy") and os.path.exists("./eigenVec.npy"):
        vals = np.load("./eigenVal.npy")
        eigvs = np.load("./eigenVec.npy")
    else:
        #vals, eigvs = linalg.eig(Kmat.toarray())
        #vals, eigvs = np.linalg.eig(Kmat.toarray())
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
#    eiglist.sort(key=lambda x: x["val"])
#    #vals, eigvs = linalg.eigs(Kmat,k = 100)
#    os.makedirs("eigenVector", exist_ok=True)
#    valstr = ""
#    for i, eigdic in enumerate(eiglist):
#        eigvstr = ""
#        val = eigdic["val"]
#        eigvec = eigdic["vec"]
#        #print(val)
#        valstr += "% 16.14f\n"%val
#        for a_eig in eigvec:
#            eigvstr += "% 16.14f\n"%a_eig
#        #eigvstr.rstrip(", ")
#        #eigvstr += "\n"
#        with open("./eigenVector/%s.csv"%i, "w") as wf:
#            wf.write(eigvstr)
#    with open("./eigenValue.csv", "w") as wf:
#        wf.write(valstr)
    #totaltime = 1000
    stepN = np.double(const.totaltime/const.t_delta)
    print("t_delta = ", const.t_delta)
    print("stepN = ", stepN)
    os.makedirs("MCain_%sps"%const.totaltime, exist_ok=True)
    if os.path.exists("./eigenVec_inv.npy"):
        eigvsinv = np.load("./eigenVec_inv.npy")
    else:
        eigvsinv = np.linalg.inv(eigvs)
        np.save("./eigenVec_inv.npy", eigvsinv)
    #stepNdelta = 100
    #stepNdelta = stepN * 0.1
    for i in range(dim):
        #eiglist[i]["vec"] = eiglist[i]["vec"]*(eiglist[i]["val"]**stepNdelta)
        #a = eiglist[i]["val"]**stepN
        #if np.linalg.norm(a) != 0.0:
            #print(a)
        eiglist[i]["vec"] = eiglist[i]["vec"]*(eiglist[i]["val"]**stepN)
        #eiglist[i]["vec"] = eiglist[i]["vec"] * a
    for eqN in range(dim):
        if eqN%const.size != const.rank:
            continue
        fe = eqpointlist[eqN].fe
        if const.EQFEmax < fe-femin:
            break
        eqname = eqpointlist[eqN].name
        exportpath = "./MCain_%sps/%s.csv"%(const.totaltime,eqname)
        if os.path.exists(exportpath):
            continue
        if not eqname in SSlist:
            continue
        ssN = SSlist.index(eqname)
        p = np.zeros(dim, dtype=np.cdouble)
        p[ssN] = 1.0

        #stepNwhile = 0
        #for i in range(dim):
           #eiglist[i]["vec"] = eiglist[i]["vec"]*(eiglist[i]["val"]**stepNdelta)
        #while stepNwhile < stepN:
        if True:
            c = eigvsinv@p
            p_t = np.zeros(dim, dtype=np.cdouble)
            for i in range(dim):
                #if np.abs(c[i]) < 1.0e-20:
                    #continue
                #p_t += eiglist[i]["vec"]*c[i]*(eiglist[i]["val"]**stepN)
                #p_t += eiglist[i]["vec"]*c[i]*(eiglist[i]["val"]**stepNdelta)
                p_t += eiglist[i]["vec"]*c[i]
            p_t = np.array([np.abs(x) for x in p_t], dtype=np.double)
            #ptot = np.abs(np.sum(p_t))
            #p_t /= ptot
            #stepNwhile += stepNdelta
            p = p_t
        print("p(%s) vector is obtained"%eqname,flush=True)
        ptot = np.sum(np.abs(a_p) for a_p in p_t)
        print("ptot = % 16.15f"%ptot,flush=True)
        #if 1.1 < ptot:
            #continue
        #if ptot< 0.9:
            #continue
        if 0.1 < np.abs(ptot - 1.0):
        #if 1.0e-5 < np.abs(ptot - 1.0):
        #if True:
            continue
            #print(writeline)
            #exit()
            print("Try knetics simulation",flush=True)
            p_ksim = calc_ksim(ssN, stepN, Kmat, dim)
            ptot = np.abs(np.sum(p_ksim))
            print("p_ksim = % 10.9f"%ptot)
            pdelta = p_t - p_ksim
            for i, p in enumerate(pdelta):
                if 0.0 < np.abs(p):
                    print("%s: % 16.12f"%(i,p))
            print("pdelta = % 16.15f"%np.linalg.norm(pdelta),flush=True)
            p_t = p_ksim
        writeline = ""
        for a_p in p_t:
            #writeline += "% 16.14f\n"%a_p.real
            writeline += "% 16.14f\n"%(np.abs(a_p)/ptot)
        with open(exportpath, "w") as wf:
            wf.write(writeline)
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
