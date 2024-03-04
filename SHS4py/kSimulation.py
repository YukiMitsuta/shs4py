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

def main(const):
    """

    """
    print("start kSimulation")
    if not os.path.exists("./RCMCresult"):
        print("Error; there is not RCMCresult")
        exit()
    maxKindex = False
    for line in open("./RCMCresult/maxK.csv"):
        line = line.split(", ")
        k = float(line[1])
        if k < const.k_RCMC:
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
                Kmat[k,l] = K_kl * const.t_delta
                #Kmat[k,l] = K_kl #* const.t_delta
    else:
        print("Error; there is not ./RCMCresult/Kmatrix/Kmat_chkpoint%05d.csv"%int(n))
        exit()
    eqpointlist  = []
    eqnamelist   = []
    FE_dic       = {}
    eqpointindex = {}
    SSlist       = []
    Z_dic = {}
    i = 0
    for line in open(const.jobfilepath + "eqlist.csv"):
        if line[0] == "#":
            continue
        line    = line.split(", ")
        #line    = line.split(",")
        eqname  = line[0]
        eqpoint = np.array(line[1:-1], dtype=float)
        eqFE = float(line[-1])
        eqpointlist.append(eqpoint)
        eqnamelist.append(eqname)
        if const.is_bilayer:
            Z_dic[eqname] = eqpoint[-1]
            z = eqpoint[-1]
            if z == 0.0:
                eqpointname = eqname
                SSlist.append(eqpointname)
                eqpointindex[eqpointname] = i
                FE_dic[eqpointname] = eqFE
                if const.is_bilayer:
                    Z_dic[eqpointname] = eqpoint[-1]
                i += 1
            else:
                eqpointname = eqname + "+"
                SSlist.append(eqpointname)
                eqpointindex[eqpointname] = i
                FE_dic[eqpointname] = eqFE
                if const.is_bilayer:
                    Z_dic[eqpointname] = eqpoint[-1]
                i += 1
                eqpointname = eqname + "-"
                SSlist.append(eqpointname)
                eqpointindex[eqpointname] = i
                FE_dic[eqpointname] = eqFE
                if const.is_bilayer:
                    Z_dic[eqpointname] = - eqpoint[-1]
                i += 1
        else:
            eqpointname = eqname
            SSlist.append(eqpointname)
            eqpointindex[eqpointname] = i
            FE_dic[eqpointname] = eqFE
            if const.is_bilayer:
                Z_dic[eqpointname] = eqpoint[-1]
            i += 1

    #stateQ = lil_matrix((dim, 1), dtype=float)
    stateQ = np.zeros(dim, dtype=float)
    sN_outsideP = - 1.0e30
    sN_outsideM = - 1.0e30
    outsidePlist = []
    outsideMlist = []
    for i, SSname in enumerate(SSlist):
        z = Z_dic[SSname]
        #fe = FE_dic[SSname]
        sN = stateN[i]
        if sN == 0:
            continue
        #if 25.0 == z:
        #if 30.0 == z:
        if 24.5 < z:
        #if 0.0 < z:
        #if 30.0 < z:
            outsidePlist.append(i)
            print("plus;%s, %s"%(z,SSname))
            #if sN_outsideP < sN:
                #sN_outsideP = sN
                #outsideP = SSname
                #outsideP_i = i
                #outsidePlist = [i]
        elif z < - 24.5:
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

    #stateQ[outsideP_i] = 1.0

    #totalP = 0.0
    #for outsideP_i in outsidePlist:
        #stateQ[outsideP_i] = np.exp(-const.beta * FE_dic[SSlist[outsideP_i]])
        #totalP += stateQ[outsideP_i]
    #for outsideP_i in outsidePlist:
        #stateQ[outsideP_i] /= totalP

    for line in open("./RCMCresult/stateQ/00000.csv"):
        line = line.split(",")
        n_stateQ = int(line[0])
        if n_stateQ != n:
            continue
        for i in range(1,len(line),2):
            stateindex = int(line[i])
            stateQdamp = float(line[i+1])
            #print("%s -> % 10.9f"%(stateindex, stateQdamp))
            stateQ[stateindex] += stateQdamp

    for outsideM_i in outsideMlist:
        for i in range(dim):
            Kmat[outsideM_i, i] = 0.0

    kSimdic = "kSim_%s"%const.k_RCMC
    if not os.path.exists(kSimdic):
        os.mkdir(kSimdic)
    os.chdir(kSimdic)

    Kmat_coo = Kmat.tocoo()
    #diaglist = []
    #for i in range(len(Kmat_coo.data)):
        #k = Kmat_coo.row[i]
        #diaglist.append(k)
    #print(diaglist)
    diagnorm = np.zeros(dim)
    #for i in range(dim):
        #diagnorm[i] = 0.0
    for i in range(len(Kmat_coo.data)):
        k = Kmat_coo.row[i]
        diagnorm[k] -= Kmat_coo.data[i]
    for i in range(dim):
        Kmat[i, i] = diagnorm[i]
        #print(diagnorm[i])
    #print(Kmat[outsideP_i, outsideM_i])
    print(Kmat[outsideM_i, outsideM_i])
    #exit()
    #print(stateQ[outsideP_i])
    writeline  = ""
    #writeline += "outsideP   , %s\n"%outsideP
    #writeline += "outsideP_i , %s\n"%outsideP_i
    #writeline += "Z_p        , %s\n"%Z_dic[outsideP]
    #writeline += "sN_outsideP, %s\n"%sN_outsideP
    writeline += "outsidePlist = %s\n"%outsidePlist
    writeline += "len(outsidePlist) = %s\n"%len(outsidePlist)
    #writeline += "outsideM   , %s\n"%outsideM 
    #writeline += "outsideM_i , %s\n"%outsideM_i
    #writeline += "Z_m        , %s\n"%Z_dic[outsideM]
    #writeline += "sN_outsideM, %s\n"%sN_outsideM
    writeline += "outsideMlist = %s\n"%outsideMlist
    writeline += "len(outsideMlist) = %s\n"%len(outsideMlist)
    with open("./info.csv", "w") as wf:
        wf.write(writeline)
    #print("stateQ[outsideP] = %s"%stateQ[outsideP])
    stepN = 0
    #exporttime = 1.0 / const.k_RCMC
    Kmat = Kmat.transpose()
    #expectation_k = 0.0
    expectation_tau = 0.0
    exportQdelta = 1.0/100.0
    exportQ = 0.0
    exportQ += exportQdelta
    totalP_minus = 0.0
    Kmat = csr_matrix(Kmat)
    while True:
        stepN += 1
        t = stepN * const.t_delta
        #if 1.0/const.k_min < t:
            #break
        #Qdelta = Kmat.dot(stateQ)
        #Qdelta = Kmat@stateQ
        Qdelta = Kmatdot(Kmat,stateQ)
        deltaP_minus = calcP_minus(Qdelta, outsideMlist)
        totalP_minus += deltaP_minus
        stateQ += Qdelta
        expectation_tau += deltaP_minus * t /10**9


        #if stepN % 100 == 0:
        #if exportQ < totalP_minus or stepN % 10000 == 0:
        if exportQ < totalP_minus or stepN % 1000000 == 0:
            #if 0.1 < exportQ:
                #exit()
            totalP_plus = 0.0
            for outsideP_i in outsidePlist:
                totalP_plus += stateQ[outsideP_i]
            with open("./throughstate.csv", "a") as wf:
                #wf.write("%8.4f, %s, %s\n"%(t/10**9, stateQ[outsideP_i], stateQ[outsideM_i]))
                wf.write("%10.7f, %s, %s\n"%(t/10**9, totalP_plus, totalP_minus))
            with open("./expectation.csv", "a") as wf:
                wf.write("%8.4f, %s\n"%(t/10**9, expectation_tau))
        #if exporttime < t:
        #if stepN % 10000 == 0:
            writeline = ""
            for i, stateQnum in enumerate(stateQ):
                if stateQnum == 0.0:
                    continue
                writeline += "%10d, %s\n"%(i, stateQnum)
            with open("./stateQ_chkpoint%05d.csv"%stepN, "w") as wf:
                wf.write(writeline)
            #while exporttime < 1.0 / const.k_min:
                #if t < exporttime:
                    #break
                #exporttime *= 10
            print("t = %8.4f mus"%(t/10**6), flush = True)
            #print("exporttime = %s"%exporttime)
            #if 1.0 - stateQ[outsideM_i] < 10e-6:
            if 1.0 - totalP_minus < 10e-4:
                break
            if 1.0 - exportQ <= exportQdelta *1.5:
                exportQdelta *= 0.1
            print("%s, %s"%(exportQ, exportQdelta))
            exportQ += exportQdelta
            exportQ = round(exportQ, 5)
def Kmatdot(Kmat, stateQ):
    return Kmat@stateQ
def calcP_minus(Qdelta, outsideMlist):
    deltaP_minus = 0.0
    for outsideM_i in outsideMlist:
        deltaP_minus += Qdelta[outsideM_i]
    return deltaP_minus


