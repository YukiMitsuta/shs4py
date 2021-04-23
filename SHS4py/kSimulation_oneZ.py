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
        #print(k)
        if k < const.k_RCMC:
            n = int(line[0]) - 1
            break
    #n = 0
    print("n = %s"%n, flush = True)
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
            #if const.k_min < K_kl:
            if True:
                #Kmat[k,l] = K_kl * const.t_delta
                Kmat[k,l] = K_kl
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
    eqFEmin = 1.0e30
    for line in open(const.jobfilepath + "eqlist.csv"):
        if line[0] == "#":
            continue
        line = line.split(", ")
        eqFE = float(line[-1])
        if eqFE < eqFEmin:
            eqFEmin = eqFE
    for line in open(const.jobfilepath + "eqlist.csv"):
        if line[0] == "#":
            continue
        line    = line.split(", ")
        #line    = line.split(",")
        eqname  = line[0]
        eqpoint = np.array(line[1:-1], dtype=float)
        eqFE = float(line[-1]) - eqFEmin
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
    #sN_initialpoint = - 1.0e30
    initialpoints = []
    for i, SSname in enumerate(SSlist):
        z = Z_dic[SSname]
        #fe = FE_dic[SSname]
        sN = stateN[i]
        if sN == 0:
            continue
        if z == const.ksim_Z:
            initialpoints.append(i)
    kSimdic = "kSim_%s"%const.ksim_Z
    if not os.path.exists(kSimdic):
        os.mkdir(kSimdic)
    os.chdir(kSimdic)
    #stateQ[outsideP_i] = 1.0
    totalP = 0.0
    for initialpoint_i in initialpoints:
        stateQ[initialpoint_i] = np.exp(-const.beta * FE_dic[SSlist[initialpoint_i]])
        totalP += stateQ[initialpoint_i]
    for initialpoint_i in initialpoints:
        stateQ[initialpoint_i] /= totalP
    writeline = ""
    for i, stateQnum in enumerate(stateQ):
        if stateQnum == 0.0:
            continue
        writeline += "%10d, %s\n"%(i, stateQnum)
    with open("./stateQ_chkpoint0.csv", "w") as wf:
        wf.write(writeline)

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
    #print(Kmat[outsideM_i, outsideM_i])
    #exit()
    #print(stateQ[outsideP_i])
    writeline  = ""
    #writeline += "outsideP   , %s\n"%outsideP
    #writeline += "outsideP_i , %s\n"%outsideP_i
    #writeline += "Z_p        , %s\n"%Z_dic[outsideP]
    #writeline += "sN_outsideP, %s\n"%sN_outsideP
    #writeline += "len(outsidePlist) = %s\n"%len(outsidePlist)
    #writeline += "outsideM   , %s\n"%outsideM 
    #writeline += "outsideM_i , %s\n"%outsideM_i
    #writeline += "Z_m        , %s\n"%Z_dic[outsideM]
    #writeline += "sN_outsideM, %s\n"%sN_outsideM
    #writeline += "len(outsideMlist) = %s\n"%len(outsideMlist)
    writeline += "len(initialpoints) = %s\n"%len(initialpoints)
    with open("./info.csv", "w") as wf:
        wf.write(writeline)
    #print("stateQ[outsideP] = %s"%stateQ[outsideP])
    stepN = 0
    #exporttime = 1.0 / const.k_RCMC
    exporttime = 0.1
    Kmat = Kmat.transpose()
    #expectation_k = 0.0
    expectation_tau = 0.0
    t = 0.0
    t_delta = gettdelta(stateQ, Kmat)
    with open("./diffusionplot.csv", "w") as wf:
        wf.write("0.0, 0.0\n")
    with open("./diffusionplotPlus.csv", "w") as wf:
        wf.write("0.0, 0.0\n")
    with open("./diffusionplotMinus.csv", "w") as wf:
        wf.write("0.0, 0.0\n")
    while True:
        stepN += 1
        #t_delta = gettdelta(stateQ, Kmat)
        #t_delta = const.t_delta
        t += t_delta
        #print("t, t_delta = %s, %s"%(t,t_delta))
        if 1.0/const.k_min < t:
            break
        Kmat_delta = Kmat * t_delta
        Qdelta = Kmat_delta.dot(stateQ)
        totalP_minus = 0.0
        deltaP_minus = 0.0
        #for outsideM_i in outsideMlist:
            #totalP_minus += stateQ[outsideM_i]
            #deltaP_minus += Qdelta[outsideM_i]
        #expectation_k += Qdelta[outsideM_i] * Qdelta[outsideM_i] / const.t_delta
        #expectation_tau += Qdelta[outsideM_i] * t /10**9
        #expectation_tau += deltaP_minus * t /10**9
        stateQ += Qdelta
        #print("stateQ[outsideP] = %s"%stateQ[outsideP_i])
        #print("stateQ[outsideM] = %s"%stateQ[outsideM_i])
        #exit()

        if exporttime < t:
            #print("t, t_delta = %s, %s"%(t,t_delta))
        #if int(t) % 10 == 0:
        #if stepN % 10 == 0:
        #if True:
            totalPdic = {}
            for i in range(len(stateQ)):
                if stateQ[i] != 0:
                    try:
                        totalPdic[str(Z_dic[SSlist[i]])] += stateQ[i]
                    except:
                        totalPdic[str(Z_dic[SSlist[i]])] = stateQ[i]
            writeline = ""
            for p in totalPdic.keys():
                writeline += "%4.2f, %s, %10.9f\n"%(t, p, totalPdic[p])
            with open("./throughstate_%s.csv"%stepN, "w") as wf:
                wf.write(writeline)
                #wf.write("%8.4f, %s, %s\n"%(t/10**9, stateQ[outsideP_i], stateQ[outsideM_i]))
            z2_ave = 0.0
            z2_avePlus = 0.0
            P_Plus = 0.0
            z2_aveMinus = 0.0
            P_Minus = 0.0
            for z, p in totalPdic.items():
                z = float(z)
                dist = z - const.ksim_Z
                dist2 = dist * dist
                #if dist == 0.0:
                    #dist2 = 0.0
                #else:
                    #dist2 = 1.0
                z2_ave += dist2 * p
                if dist < 0.0:
                    z2_aveMinus += dist2 * p
                    P_Minus += p
                elif 0.0 < dist:
                    z2_avePlus += dist2 * p
                    P_Plus += p
                #else:
                    #z2_aveMinus += dist2 * p
                    #P_Minus += p
                    #z2_avePlus += dist2 * p
                    #P_Plus += p
            #z2_aveMinus /= P_Minus
            #z2_avePlus  /= P_Plus
            with open("./diffusionplot.csv", "a") as wf:
                wf.write("%4.2f, %10.9f\n"%(t, z2_ave))
            with open("./diffusionplotPlus.csv", "a") as wf:
                wf.write("%4.2f, %10.9f\n"%(t, z2_avePlus))
            with open("./diffusionplotMinus.csv", "a") as wf:
                wf.write("%4.2f, %10.9f\n"%(t, z2_aveMinus))

            #for outsideP_i in outsidePlist:
                #totalP_plus += stateQ[outsideP_i]
            #with open("./throughstate.csv", "a") as wf:
                #wf.write("%8.4f, %s, %s\n"%(t/10**9, stateQ[outsideP_i], stateQ[outsideM_i]))
                #wf.write("%8.4f, %s, %s\n"%(t/10**9, totalP_plus, totalP_minus))
            #with open("./expectation.csv", "a") as wf:
                #wf.write("%8.4f, %s\n"%(t/10**9, expectation_tau))
        #if exporttime < t:
        #if stepN % 100 == 0:
        #if True:
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
            #print("t = %8.4f ps"%(t), flush = True)
            #print("exporttime = %s"%exporttime)
            #if 1.0 - stateQ[outsideM_i] < 10e-6:
            #if 1.0 - totalP_minus < 10e-6:
                #break
            exporttime += 0.1
        if 10.0 < t:
            break
    os.chdir("../")
def gettdelta(stateQ, Kmat):
    k_max = 0.0
    #for i in range(len(stateQ)):
        #if stateQ[i] == 0.0:
            #continue
    if True:
        Kmat_coo = Kmat.tocoo()
        for Kmat_i in range(len(Kmat_coo.data)):
            k = Kmat_coo.row[Kmat_i]
            #if i == k:
            if True:
                if k_max < Kmat_coo.data[Kmat_i]:
                    k_max = Kmat_coo.data[Kmat_i]
    #t_delta = 0.1 / k_max 
    t_delta = 0.001
    print(t_delta)
    return t_delta

