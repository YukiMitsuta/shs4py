#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020/01/10 MitsutaYuki
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
    if os.path.exists("./RCMCresult"):
        if const.rank == const.root:
            n, SSlistlist, stateNlist, stateQlistlist, Kmatlist = importNetwork(const)
        else:
            n              = None
            SSlistlist     = None
            stateNlist     = None
            stateQlistlist = None
            Kmatlist       = None
        if const.size != 1:
            n     = const.comm.bcast(n,     root = 0)
            SSlistlist     = const.comm.bcast(SSlistlist,     root = 0)
            stateNlist     = const.comm.bcast(stateNlist,     root = 0)
            stateQlistlist = const.comm.bcast(stateQlistlist, root = 0)
            Kmatlist       = const.comm.bcast(Kmatlist,       root = 0)
    else:
        n, SSlistlist, stateNlist, stateQlistlist, Kmatlist = makeNetwork(const)

    dim  = len(stateNlist[-1]) 
    start = time.time()
    while True:
        findexitQ = False
        if const.rank == const.root:
            if os.path.exists("./exit.txt"):
                print("exit.txt is Found; this job is suspended")
                os.remove("./exit.txt")
                findexitQ = True
            t = time.time() - start
            t /= 3600
            if const.maxtime < t:
                print("calculartiontime(%s hour) over maxtime (%s hour); exit"%(const.maxtime,t))
                findexitQ = True
        findexitQ = const.comm.bcast(findexitQ,     root = 0)
        if findexitQ:
            Kmat_coo        = Kmatlist[-1].tocoo()
            writeline = ""
            for i in range(len(Kmat_coo.row)):
                if i % const.size != const.rank:
                    continue
                k = Kmat_coo.row[i]
                l = Kmat_coo.col[i]
                Kmat_kl = Kmat_coo.data[i]
                if Kmat_kl == 0.0:
                    continue
                writeline += "%6d, %6d, %s\n"%(k,l, Kmat_kl)
            writelineall = const.comm.gather(writeline, root = 0)
            if const.rank == const.root:
                writeline_root = ""
                for writeline_g in writelineall:
                    writeline_root += writeline_g
                with open("./RCMCresult/Kmatrix/Kmat_chkpoint%s.csv"%int(n), "w") as wf:
                    wf.write(writeline_root)
                writeline = ""
                for i, stateN in enumerate(stateNlist[-1]):
                    writeline += "%10d, %s\n"%(i, stateN)
                with open("./RCMCresult/stateN/stateN_chkpoint%05d.csv"%int(n), "w") as wf:
                    wf.write(writeline)

            exit()

        if 1 <= n:
            Kmatlist       = [Kmatlist[-1]]
            SSlistlist     = [SSlistlist[-1]]
            stateNlist     = [stateNlist[-1]]
            stateQlistlist = [stateQlistlist[-1]]
        maxK = 0.0
        Kmat_coo = Kmatlist[-1].tocoo()
        for ind in range(len(Kmat_coo.row)):
            k    = Kmat_coo.row[ind]
            l    = Kmat_coo.col[ind]
            if k == l:
                continue
            K_kl = Kmat_coo.data[ind]
            if maxK < K_kl:
                maxK   = copy.copy(K_kl)
                maxind = copy.copy(k)
        if maxK == 0.0:
            if const.rank == const.root:
                print("maxK = 0.0; all connections are restrected.")
            break
        if const.rank == const.root:
            with open("./RCMCresult/maxK.csv", "a") as wf:
                wf.write("%3d, %s\n"%(int(n + 1), maxK))
            #print("maxK = %4.3f ps "%(maxK), flush = True)
            print("1/maxK = %4.3f ps "%(1.0/maxK), flush = True)
        if const.rank == const.root:
            with open("./RCMCresult/maxKindex.csv", "a") as wf:
                wf.write("%3d, %s\n"%(int(n + 1), maxind))
        sig = 0.0
        for i in range(len(Kmat_coo.row)):
            if Kmat_coo.row[i] == maxind:
                sig += Kmat_coo.data[i]
        sig = 1.0 / sig
        #newSSlist = copy.copy(SSlistlist[-1])
        #newSSlist.pop(maxind)
        #SSlistlist.append(newSSlist)
        Kmatdash = calcdash(n, dim, maxind, sig, Kmatlist, const)
        #if const.rank == const.root:
            #print("calcstateN", flush = True)
        stateNlist.append(calcstateN(n, dim, maxind, sig, stateNlist, Kmatlist, const))
        #if const.rank == const.root:
            #print("calcstateQ", flush = True)
        stateQlistlist.append(calcstateQ(n, dim, maxind, sig, stateQlistlist[-1], Kmatlist[-1], const))
        if const.rank == const.root:
            print("export %s"%int(n+1), flush = True)
            #writeline = "%6d"%int(n + 1)
            #for SSname in newSSlist:
                #writeline += ", %s"%SSname
            #writeline += "\n"
            #with open("./RCMCresult/SSlist/%05d.csv"%int(n+1), "w") as wf:
                #wf.write(writeline)
            #writeline = "%6d"%int(n + 1)
            #for stateN in stateNlist[-1]:
                #writeline += ", %s"%stateN
            #writeline += "\n"
            writeline = ""
            for i, stateN in enumerate(stateNlist[-1]):
                #if stateN != 0.0:
                if stateN != stateNlist[0][i]:
                    writeline += "%10d, %s\n"%(i, stateN)
            with open("./RCMCresult/stateN/%05d.csv"%int(n+1), "w") as wf:
                wf.write(writeline)
            for ip_index in range(len(stateQlistlist[-1])):
                writeline = "%6d"%int(n + 1)
                #writeline = ""
                for steteQn, stateQ in stateQlistlist[-1][ip_index]:
                    #writeline += "%s, %s\n"%(steteQn, stateQ)
                    writeline += ", %s, %s"%(steteQn, stateQ)
                writeline += "\n"
                with open("./RCMCresult/stateQ/%05d.csv"%ip_index, "a") as wf:
                    wf.write(writeline)
        Kmatlist.append(calcmatrix(n, dim, maxind, sig, Kmatlist, Kmatdash, const))
        n += 1
def makeNetwork(const):
    if const.rank == const.root:
        print("start makeNetwork", flush = True)
    FE_dic      = {}
    sameEQdic   = {}
    eqpointlist = []
    eqnamelist  = []
    if const.is_bilayer:
        P_dic       = {}
        #eqPlist     = []
    for line in open(const.jobfilepath + "eqlist.csv"):
        if line[0] == "#":
            continue
        line    = line.split(",")
        eqname  = line[0]
        #print(eqname)
        eqpoint = np.array(line[1:-1], dtype=float)
        if const.is_bilayer:
            eqpoint[-1] = np.abs(eqpoint[-1])
            if const.Pmax < abs(eqpoint[-1]):
                continue
            beforepointQ = False
            j = 0
            for j, beforepoint in enumerate(eqpointlist):
                if j % const.size != const.rank:
                    continue
                for i in range(len(beforepoint)):
                    if const.sameEQthreshold[i] < abs(beforepoint[i] - eqpoint[i]):
                        break
                else:
                    beforepointQ = True
                    break
            beforepointj_g = const.comm.gather(j, root=0)
            beforepointQ_g = const.comm.gather(beforepointQ, root=0)
            if const.rank == const.root:
                jpoint = False
                for i, beforepointQ in enumerate(beforepointQ_g):
                    if beforepointQ:
                        jpoint = beforepointj_g[i]
                        break
            else:
                jpoint = None
            jpoint = const.comm.bcast(jpoint, root=0)

            #if beforepointQ:
            if not jpoint is False:
                #if const.rank == const.root:
                    #print(len(eqpointlist), flush = True)
                    #print("len(eqpointlist) = %s"%len(eqpointlist), flush = True)
                #sameEQdic[eqname] = copy.copy(eqnamelist[j])
                sameEQdic[eqname] = copy.copy(eqnamelist[jpoint])
                if const.is_bilayer:
                    P_dic[eqname] = float(line[-2])
                FE_dic[eqname]     = FE_dic[sameEQdic[eqname]]
                continue

        eqpointlist.append(eqpoint)
        eqnamelist.append(eqname)
        sameEQdic[eqname] = copy.copy(eqname)
        FE_dic[eqname]     = float(line[-1])
        if const.is_bilayer:
            P_dic[eqname] = float(line[-2])
    FEmin = min(FE_dic.values())
    if const.rank == const.root:
        print("len(eqpointlist) = %s"%len(eqpointlist), flush = True)
        print("FEmin = % 5.3f"%FEmin, flush = True)
    tslist = []
    fricfactorlist = []
    tspointlist    = []
    for line in open(const.jobfilepath + "tslist.csv"):
        if line[0] == "#":
            continue
        line = line.split(",")
        tsname = line[0]
        tspoint = np.array(line[1:-1], dtype = float)
        if const.is_bilayer:
            tspoint[-1] = abs(tspoint[-1])
            if const.Pmax < abs(tspoint[-1]):
                continue
            beforepointQ = False
            j = 0
            for j, beforepoint in enumerate(tspointlist):
                if j % const.size != const.rank:
                    continue
                for i in range(len(beforepoint)):
                    if const.sameEQthreshold[i] < abs(beforepoint[i] - tspoint[i]):
                        break
                else:
                    beforepointQ = True
                    break
            beforepointj_g = const.comm.gather(j, root=0)
            beforepointQ_g = const.comm.gather(beforepointQ, root=0)
            if const.rank == const.root:
                jpoint = False
                for i, beforepointQ in enumerate(beforepointQ_g):
                    if beforepointQ:
                        jpoint = beforepointj_g[i]
                        break
            else:
                jpoint = None
            jpoint = const.comm.bcast(jpoint, root=0)
            if not jpoint is False:
                #if const.rank == const.root:
                    #print("len(tslist) = %s"%len(tslist), flush = True)
                continue
        tsname = line[0]
        #tslist.append(tsname)
        FE_dic[tsname] = float(line[-1].replace("\n", ""))
        if const.is_bilayer:
            P_dic[tsname]  = float(line[-2])
        if FE_dic[tsname] - FEmin <= const.TSFEmax:
            tslist.append(tsname)
            fricpath = const.jobfilepath + "/%s/frac.csv"%tsname
            if os.path.exists(fricpath):
                for line in open(fricpath):
                    fricfactorlist.append([tsname, float(line)])
            tspointlist.append(tspoint)
    if const.rank == const.root:
        print("len(tslist) = %s"%len(tslist), flush = True)
        connectionlist     = []
        eqlist_connections = []
        G = nx.Graph()
        for line in open(const.jobfilepath + "connections.csv"):
            if line[0] == "#":
                continue
            line = line.replace("\n","").split(", ")
            if not line[0] in FE_dic.keys():
                continue
            #if not line[1] in FE_dic.keys():
            if line[1] in sameEQdic.keys():
                if not sameEQdic[line[1]] in FE_dic.keys():
                    continue
            else:
                continue
            eqname_replace = sameEQdic[line[1]]
            if const.EQFEmax < FE_dic[eqname_replace] - FEmin:
                #print("FE(%s) = % 3.2f > EQFEmax"%(eqname_replace, FE_dic[eqname_replace] - FEmin), flush = True)
                continue

            if FE_dic[line[0]] - FEmin <= const.TSFEmax:
                #connectionlist.append([line[0], sameEQdic[line[1]]])
                connectionlist.append([line[0], line[1]])
                G.add_edge(line[0], sameEQdic[line[1]])
                eqlist_connections.append(sameEQdic[line[1]])

        eqlist_connections = list(set(eqlist_connections))
        eqlist_connections = sorted(eqlist_connections, key = lambda x: FE_dic[x])
        eqpoint_minimum    = eqlist_connections[0]
        #eqlist_defo = []
        #if True:
        if False:
            while True:
                rmnodes = []
                for node in G.nodes():
                    if G.degree(node) <= 1:
                        rmnodes.append(node)
                #print("len(rmnodes)   = %s"%len(rmnodes))
                #print("len(G.nodes()) = %s"%len(G.nodes()))
                if len(rmnodes) == 0:
                    break
                for removeNode in rmnodes:
                    G.remove_node(removeNode)
        print("len(G.nodes()) = %s"%len(G.nodes()))
        #if const.is_bilayer:
        if False:
            rmnodes = []
            for EQnode in G.nodes():
                if "TS" in EQnode:
                    continue
                Pcenter   = P_dic[EQnode]
                FEcenter = FE_dic[EQnode]
                for EQnode_min in G.nodes():
                    if "TS" in EQnode:
                        continue
                    P_min = P_dic[EQnode_min]
                    if abs(P_min - Pcenter) < const.samePrange:
                        FEmin = FE_dic[EQnode_min]
                        if const.EQFEmax_Prange < FEcenter - FEmin:
                            rmnodes.append(EQnode)
                            break
            for removeEQ in rmnodes:
                G.remove_node(removeEQ)
            while True:
                rmnodes = []
                for node in G.nodes():
                    if G.degree(node) <= 1:
                        rmnodes.append(node)
                #print("len(rmnodes) = %s"%len(rmnodes))
                #print("len(G.nodes()) = %s"%len(G.nodes()))
                if len(rmnodes) == 0:
                    break
                for removeNode in rmnodes:
                    G.remove_node(removeNode)
            print("len(G.nodes()) = %s"%len(G.nodes()))
        #print("len(G.nodes()) = %s"%len(G.nodes()))
        #if const.rm_disconnections:
        if False:
            rmnodes = []
            for EQnode in G.nodes():
                if not nx.has_path(G, source=eqpoint_minimum, target=EQnode):
                    rmnodes.append(EQnode)
            for removeEQ in rmnodes:
                G.remove_node(removeEQ)
            print("len(G.nodes()) = %s"%len(G.nodes()))
        #if not nx.is_connected(G):
            #print("Error: G is not connected")
        eqlist_defo = [ a for a in G.nodes() if "EQ" in a]
        eqlist_defo = sorted(eqlist_defo, key = lambda x: FE_dic[x])
        eqpoint_minimum    = eqlist_connections[0]
        print("eqpoint_minimum = %s"%eqpoint_minimum)

        if const.is_bilayer:
            eqlist = [[a+"+", a+"-"] for a in eqlist_defo]
            eqlist = sum(eqlist, [])
        else:
            eqlist = list(eqlist_defo)

        nodenumdic = {}
        for i, eqname in enumerate(eqlist):
            nodenumdic[eqname] = copy.copy(i)
        dim  = len(eqlist)
        print("dim = %s"%dim, flush = True)
        Kmat = lil_matrix((dim,dim), dtype=float)
        fricfactorABS = np.mean([x[-1] for x in fricfactorlist])
        G_ps = nx.Graph()
        for tsname in tslist:
            tsconnection = []
            for connection in connectionlist:
               if connection[0] == tsname:
                    tsconnection.append(connection[1])
            if len(tsconnection) != 2:
                #print("tsname = %s is removed."%tsname)
                continue
            if tsconnection[0] == tsconnection[1]:
                continue
            if not tsconnection[0] in FE_dic.keys() or not tsconnection[1] in FE_dic.keys():
                #print("eqname = %s is removed."%tsconnection)
            #if not tsconnection[0] in G.nodes() or not tsconnection[1] in G.nodes():
                #print("eqname = %s is removed."%tsconnection)
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
                fricfactor = copy.copy(fricfactorABS)
            if const.is_bilayer:
                for initial_EQ, final_EQ in tsconnection:
                    if 20.0 < abs(P_dic[tsname]):
                    #if 30.0 < abs(P_dic[initial_EQ]) and 30.0 < abs(P_dic[final_EQ]):
                        if P_dic[initial_EQ] * P_dic[final_EQ] < 0.0:
                            print("in %s"%tsname)
                            print("P[initial], P[final] = %s,%s; this TS connect the outside of bilayer\n removed"%(P_dic[initial_EQ], P_dic[final_EQ]))
                            break
                    if P_dic[initial_EQ] < 0:
                        initialEQ_ps = sameEQdic[initial_EQ] + "-"
                    else:
                        initialEQ_ps = sameEQdic[initial_EQ] + "+"
                    if P_dic[final_EQ] < 0:
                        finalEQ_ps = sameEQdic[final_EQ] + "-"
                    else:
                        finalEQ_ps = sameEQdic[final_EQ] + "+"
                    if not initialEQ_ps in nodenumdic.keys() or not finalEQ_ps in nodenumdic.keys():
                        continue
                    Delta_fe = np.abs(FE_dic[sameEQdic[initial_EQ]] - FE_dic[tsname])
                    k = fricfactor * np.exp( - Delta_fe / const.betainv)
                    Kmat[nodenumdic[initialEQ_ps], nodenumdic[finalEQ_ps]] += k
                    #Delta_fe = np.abs(FE_dic[final_EQ] - FE_dic[tsname])
                    #k = fricfactor * np.exp( - Delta_fe / const.betainv)
                    #Kmat[nodenumdic[finalEQ_ps], nodenumdic[initialEQ_ps]] += k
                    G_ps.add_edge(initialEQ_ps, finalEQ_ps)
                    if P_dic[initial_EQ] > 0:
                        initialEQ_ps = sameEQdic[initial_EQ] + "-"
                    else:
                        initialEQ_ps = sameEQdic[initial_EQ] + "+"
                    if P_dic[final_EQ] > 0:
                        finalEQ_ps = sameEQdic[final_EQ] + "-"
                    else:
                        finalEQ_ps = sameEQdic[final_EQ] + "+"
                    Delta_fe = np.abs(FE_dic[sameEQdic[initial_EQ]] - FE_dic[tsname])
                    k = fricfactor * np.exp( - Delta_fe / const.betainv)
                    Kmat[nodenumdic[initialEQ_ps], nodenumdic[finalEQ_ps]] += k
                    #Delta_fe = np.abs(FE_dic[final_EQ] - FE_dic[tsname])
                    #k = fricfactor * np.exp( - Delta_fe / const.betainv)
                    #Kmat[nodenumdic[finalEQ_ps], nodenumdic[initialEQ_ps]] += k
                    G_ps.add_edge(initialEQ_ps, finalEQ_ps)
            else:
                for initial_EQ, final_EQ in tsconnection:
                    if not initial_EQ in nodenumdic.keys() or not final_EQ in nodenumdic.keys():
                        continue
                    Delta_fe = np.abs(FE_dic[initial_EQ] - FE_dic[tsname])
                    k = fricfactor * np.exp( - Delta_fe / const.betainv)
                    Kmat[nodenumdic[initial_EQ], nodenumdic[final_EQ]] += k
                    #Delta_fe = np.abs(FE_dic[final_EQ] - FE_dic[tsname])
                    #k = fricfactor * np.exp( - Delta_fe / const.betainv)
                    #Kmat[nodenumdic[final_EQ], nodenumdic[initial_EQ]] += k

        #stateQlistlist = [list(np.identity(dim))[:stateQN]]
        stateQlist = [[[i,1.0]] for i in range(const.stateQN)]
        stateQlistlist = [stateQlist]

        if const.diffusionQ:
#            eqPmaxFE = 1.0e30
#            eqpoint_Pmax = False
#            for eqname in G_ps.nodes():
#                if not "EQ" in eqname:
#                    continue
#                if not "+" in eqname:
#                    continue
#                if not nx.has_path(G_ps, source=eqlist[0] , target=eqname):
#                    continue
#                eqname = eqname.replace("+","")
#                p = abs(P_dic[eqname])
#                #print("%s, %s"%(eqname, p))
#                if 30.0 < p < const.Pmax:
#                    FE = FE_dic[eqname]
#                    if FE < eqPmaxFE:
#                        eqPmaxFE = copy.copy(FE)
#                        eqpoint_Pmax = copy.copy(eqname)
#            if eqpoint_Pmax is False:
#                print("Error; we cannot found eqpoint_Pmax")
#                Kmat = False
#            else:
#                print("EQpoint of Pmax is %s"%eqpoint_Pmax, flush = True)
#                print("eqPmaxFE = %s"%eqPmaxFE, flush = True)
#                eqpoint_p = eqpoint_Pmax + "+"
#                eqpoint_m = eqpoint_Pmax + "-"
#                if not nx.has_path(G_ps, source=eqpoint_p, target=eqpoint_m):
#                    print("Error: There is not path between %s <-> %s"%(eqpoint_p, eqpoint_m))
#                    print("Error: %s <-> %s ; %s"%(eqpoint_p, eqlist[0], nx.has_path(G_ps, eqpoint_p, eqlist[0])))
#                    print("Error: %s <-> %s ; %s"%(eqpoint_m, eqlist[0], nx.has_path(G_ps, eqpoint_m, eqlist[0])))
#                    print("Error: %s <-> %s ; %s"%(eqpoint_p, eqlist[1], nx.has_path(G_ps, eqpoint_p, eqlist[1])))
#                    print("Error: %s <-> %s ; %s"%(eqpoint_m, eqlist[1], nx.has_path(G_ps, eqpoint_m, eqlist[1])))
#                    Kmat = False
#                else:
#                #if True:
#                    stateQlistlist[0].append([[nodenumdic[eqpoint_p],1.0]])
#                    for i in range(dim):
#                        Kmat[nodenumdic[eqpoint_p], i] = 0.0
#                    for i in range(dim):
#                        Kmat[nodenumdic[eqpoint_m], i] = 0.0
#                    stateQlistlist[0].append([[nodenumdic[eqpoint_m],1.0]])

            eqpoint_PmaxQ = False
            for eqname in G_ps.nodes():
                if not "EQ" in eqname:
                    continue
                if not "+" in eqname:
                    continue
                if not nx.has_path(G_ps, source=eqlist[0] , target=eqname):
                    continue
                eqname = eqname.replace("+","")
                p = abs(P_dic[eqname])
                if 30.0 < p < const.Pmax:
                    #stateQlistlist[0].append([[nodenumdic[eqpoint_p],1.0]])
                    eqpoint_p = eqname + "+"
                    eqpoint_m = eqname + "-"
                    for i in range(dim):
                        afterpoint = eqlist[i][:-1]
                        if 30.0 < abs(P_dic[afterpoint]):
                            continue
                        eqpoint_PmaxQ = True
                        Kmat[nodenumdic[eqpoint_p], i] = 0.0
                        Kmat[nodenumdic[eqpoint_m], i] = 0.0
                    #stateQlistlist[0].append([[nodenumdic[eqpoint_m],1.0]])
            if eqpoint_PmaxQ is False:
                print("Error; we cannot found eqpoint_Pmax")
                Kmat = False


        SSlistlist = [eqlist]
        if const.is_bilayer:
            stateNlist = [np.exp(- FE_dic[x[:-1]] / const.betainv) for x in eqlist]
        else:
            stateNlist = [np.exp(- FE_dic[x] / const.betainv) for x in eqlist]
        totalN     = sum(stateNlist)
        stateNlist = [[x / totalN for x in stateNlist]]
        if Kmat is False:
            Kmatlist = False
        else:
            Kmatlist   = [Kmat]


        n = 0
        if not os.path.exists("RCMCresult"):
            os.mkdir("RCMCresult")
        #writeline = "%3d"%int(n)
        writeline = ""
        for i, SSname in enumerate(SSlistlist[0]):
            writeline += "%10d, %s\n"%(i, SSname)
        #writeline += "\n"
        #if not os.path.exists("./RCMCresult/SSlist"):
            #os.mkdir("./RCMCresult/SSlist")
        #with open("./RCMCresult/SSlist/%05d.csv"%n, "w") as wf:
            #wf.write(writeline)
        with open("./RCMCresult/SSlist.csv", "w") as wf:
            wf.write(writeline)
        #writeline = "%3d"%int(n)
        writeline = ""
        for i, stateN in enumerate(stateNlist[-1]):
            if stateN != 0.0:
                writeline += "%10d, %s\n"%(i, stateN)
        writeline += "\n"
        if not os.path.exists("./RCMCresult/stateN"):
            os.mkdir("./RCMCresult/stateN")
        with open("./RCMCresult/stateN/%05d.csv"%n, "w") as wf:
            wf.write(writeline)
        if not os.path.exists("RCMCresult/stateQ"):
            os.mkdir("RCMCresult/stateQ")
        for ip_index in range(len(stateQlistlist[-1])):
            writeline = "%6d"%int(n)
            #for steteQn, stateQ in enumerate(stateQlistlist[-1][ip_index]):
            for steteQn, stateQ in stateQlistlist[-1][ip_index]:
                writeline += ", %s, %s"%(steteQn, stateQ)
            writeline += "\n"
            with open("./RCMCresult/stateQ/%05d.csv"%ip_index, "a") as wf:
                wf.write(writeline)
        if not os.path.exists("RCMCresult/Kmatrix"):
            os.mkdir("RCMCresult/Kmatrix")
        writeline = ""
        if not Kmat is False:
            Kmat_coo = Kmat.tocoo()
            for i in range(len(Kmat_coo.row)):
                k = Kmat_coo.row[i]
                l = Kmat_coo.col[i]
                writeline += "%6d, %6d, %s\n"%(k, l, Kmat[k,l])
            with open("./RCMCresult/Kmatrix/%05d.csv"%int(0), "w") as wf:
                wf.write(writeline)
    else:
        SSlistlist     = None
        stateNlist     = None
        stateQlistlist = None
        Kmatlist       = None

    if const.size != 1:
        SSlistlist     = const.comm.bcast(SSlistlist,     root = 0)
        stateNlist     = const.comm.bcast(stateNlist,     root = 0)
        stateQlistlist = const.comm.bcast(stateQlistlist, root = 0)
        Kmatlist       = const.comm.bcast(Kmatlist,       root = 0)

    if Kmatlist is False:
        exit()

    n = 0
    return n, SSlistlist, stateNlist, stateQlistlist, Kmatlist
def importNetwork(const):
    Kmatfiles = glob.glob("./RCMCresult/Kmatrix/*.csv")
    nlist = [int(os.path.splitext(os.path.basename(filepath))[0])
                for filepath in Kmatfiles if not "chkpoint" in filepath]
    nlist.sort()
    nlist.reverse()
    SSlist = []
    for line in open("./RCMCresult/SSlist.csv"):
        line = line.replace("\n","").split(",")
        SSlist.append(line[-1])
    dim = len(SSlist)
    SSlistlist = [SSlist]
    for n in nlist:
#        stateN = False
#        stateNpath = "./RCMCresult/stateN/%05d.csv"%int(n)
#        if os.path.exists(stateNpath):
#            stateN = np.zeros(dim)
#            for line in open("./RCMCresult/stateN/%05d.csv"%int(n)):
#                if line == "\n":
#                    break
#                line = line.split(",")
#                i = int(line[0])
#                stateNpoint = float(line[1])
#                stateN[i] = stateNpoint
#                stateNlist = [stateN]
#        if stateN is False:
#            continue
        stateQlist = []
        for ip_index in range(len(glob.glob("./RCMCresult/stateQ/*.csv"))):
            stateQ = False
            for line in open("./RCMCresult/stateQ/%05d.csv"%ip_index):
                line = line.split(",")
                if int(line[0]) == n:
                    stateQ = []
                    for i in range((len(line)-1)//2):
                        stateQ.append([int(line[2*i+1]), float(line[2*i+2])])
                    break
            if stateQ is False:
                print("stateQ/%05d.csv cannot found"%ip_index,flush = True)
                break
            stateQlist.append(stateQ)
        if stateQlist is False:
            continue
        stateQlistlist = [stateQlist]

        if os.path.exists("./RCMCresult/Kmatrix/%05d.csv"%int(n)):
            break
    chkpointname = "./RCMCresult/stateN/stateN_chkpoint%05d.csv"%int(n)
    stateN = np.zeros(dim)
    if os.path.exists(chkpointname):
        for line in open(chkpointname):
            line = line.split(",")
            k    = int(line[0])
            N_k  = float(line[1])
            stateN[k] = N_k
    else:
        for stateN_n in range(n+1):
            if not os.path.exists("./RCMCresult/stateN/%05d.csv"%int(steteN_n)):
                print("ERROR; there is not stateN/%05d.csv"%int(stateN_n))
                exit()
            for line in open("./RCMCresult/stateN/%05d.csv"%int(stateN_n)):
                line = line.split(",")
                k    = int(line[0])
                N_k  = float(line[1])
                stateN[k] = N_k
    stateNlist = [stateN]

    Kmat = lil_matrix((dim, dim), dtype=float)
    chkpointname = "./RCMCresult/Kmatrix/Kmat_chkpoint%s.csv"%int(n)
    if os.path.exists(chkpointname):
        for line in open("./RCMCresult/Kmatrix/Kmat_chkpoint%s.csv"%int(n)):
            line = line.split(",")
            k = int(line[0])
            l = int(line[1])
            K_kl = float(line[2])
            Kmat[k,l] = K_kl
    else:
        for KmatN in range(n+1):
            if not os.path.exists("./RCMCresult/Kmatrix/%05d.csv"%int(KmatN)):
                print("ERROR; there is not Kmatrix/%05d.csv"%int(KmatN))
                exit()
            for line in open("./RCMCresult/Kmatrix/%05d.csv"%int(KmatN)):
                line = line.split(",")
                k = int(line[0])
                l = int(line[1])
                K_kl = float(line[2])
                Kmat[k,l] = K_kl
    Kmatlist = [Kmat]
    print("import n = %s"%n)
    return n, SSlistlist, stateNlist, stateQlistlist, Kmatlist
def calcdash(n, dim, maxind, sig, Kmatlist, const):
    if const.cythonQ:
        #if const.rank == const.root:
            #print("start const.calcRCMC.calcdash", flush = True)
        Kmat_coo = Kmatlist[-1].tocoo()
        Kmatdash = const.calcRCMC.calcdash(maxind, sig,
                Kmat_coo.row, Kmat_coo.col, Kmat_coo.data,
                const.size, const.rank, const.k_min)
        #if const.rank == const.root:
            #print("end const.calcRCMC.calcdash", flush = True)
    else:
        Kmat_coo   = Kmatlist[-1].tocoo()
        Kmatdash   = []
        mpicounter = -1
        for i in range(len(Kmat_coo.row)):
            mpicounter += 1
            if mpicounter % const.size != const.rank:
                continue
            k = Kmat_coo.row[i]
            if k != maxind:
                continue
            l = Kmat_coo.col[i]
            if l == maxind:
                continue
            next_l  = l
            K_max_l = Kmat_coo.data[i]
            for j in range(len(Kmat_coo.row)):
                k = Kmat_coo.row[j]
                l = Kmat_coo.col[j]
                if l != maxind:
                    continue
                if k == maxind:
                    continue
                K_k_max = Kmat_coo.data[j]
                next_k = k
                if next_k == next_l:
                    continue
                K_k_l = K_k_max * K_max_l * sig
                if const.k_min < K_k_l:
                    Kmatdash.append([next_k, next_l, K_k_l])

    #if const.rank == const.root:
        #print("reconstruct Kmatdash_return", flush = True)

    Kmatdash_lil = lil_matrix((dim, dim), dtype = float)
    for k, l, Kmatdash_kl in Kmatdash:
        Kmatdash_lil[k,l] += Kmatdash_kl
    Kmatdashall = const.comm.gather(Kmatdash_lil, root = 0)
    #Kmatdashall = const.comm.bcast(Kmatdashall,   const.root = 0)
    if const.rank == const.root:
        Kmatdash_return  = sum(Kmatdashall)
        #Kmatdash_return = lil_matrix((dim, dim), dtype = float)
        #for Kmatdash_damp in Kmatdashall:
            #Kmatdash_return += Kmatdash_damp
    else:
        Kmatdash_return  = None
    Kmatdash_return = const.comm.bcast(Kmatdash_return,   root = 0)
    #Kmatdash_return = lil_matrix(Kmatdash_return, dtype = float)
    if True:
        #Kmatdash_return[maxind,:]  = 0.0
        #Kmatdash_return[:, maxind] = 0.0
        Kmatdash_return += Kmatlist[-1]
        Kmatdash_return_coo = Kmatdash_return.tocoo()
        for i in range(len(Kmatdash_return_coo.row)):
            k = Kmatdash_return_coo.row[i]
            l = Kmatdash_return_coo.col[i]
            if k == maxind or l == maxind:
                Kmatdash_return[k,l] = 0.0
                #print("*%s, %s ->0.0"%(k,l), flush = True)
    #else:
        #Kmatdash_return = None
    #Kmatdash_return = const.comm.bcast(Kmatdash_return, const.root = 0)
    
    #Kmatdash_return_coo = Kmatdash_return.tocoo()
    #for i in range(len(Kmatdash_return_coo.row)):
        #if Kmatdash_return_coo.data[i] == 0.0:
            #k = Kmatdash_return_coo.row[i]
            #l = Kmatdash_return_coo.col[i]
            #print("%s, %s ->0.0"%(k,l), flush = True)

    #if const.rank == const.root:
        #print("end reconstruct Kmatdash_return", flush = True)
    
    return Kmatdash_return
def calcstateN(n,dim, maxind, sig, stateNlist, Kmatlist, const):
    #newStateN = np.zeros(dim)
    newStateNindex = []
    for k in range(dim):
        if k % const.size != const.rank:
            continue
        if k == maxind:
            continue
        #if k < maxind:
            #beforek = k
        #else:
            #beforek = k + 1
        #newStateN[k] = stateNlist[n][beforek] + Kmatlist[n][maxind, beforek] * sig * stateNlist[n][maxind]
        #newStateNindex.append([k, stateNlist[n][beforek] + Kmatlist[n][maxind, beforek] * sig * stateNlist[n][maxind]])

        #stateN =  stateNlist[n][beforek] + Kmatlist[n][maxind, beforek] * sig * stateNlist[n][maxind]
        #stateN =  stateNlist[-1][beforek] + Kmatlist[-1][maxind, beforek] * sig * stateNlist[-1][maxind]
        stateN =  stateNlist[-1][k] + Kmatlist[-1][maxind, k] * sig * stateNlist[-1][maxind]
        if const.stateNmin < stateN:
            newStateNindex.append([k, stateN])
    newStateN = np.zeros(dim)
    if const.size == 1:
        newStateNall = [newStateNindex]
    else:
        newStateNall = const.comm.gather(newStateNindex, root = 0)
    if const.rank == const.root:
        for newStateNdamp in newStateNall:
            for k, newStateN_k in newStateNdamp:
                newStateN[k] += newStateN_k
    if const.size != 1:
        newStateN = const.comm.bcast(newStateN, root = 0)
    return newStateN
def calcstateQ(n, dim, maxind, sig, stateQlist, Kmat, const):
    newStateQlist = []
    Kmat_coo = Kmat.tocoo()
    for ip_index in range(len(stateQlist)):
        #if ip_index % const.size != const.rank:
            #continue
        #newStateQ = np.zeros(dim)
        newStateQ = []
        findmaxstate = False
        for k, stateQ in stateQlist[ip_index]:
            if k == maxind:
                findmaxstate = True
                max_stateQ = copy.copy(stateQ)
                continue
            #if k < maxind:
            if True:
                nextk = k
            else:
                nextk = k - 1
            newStateQ.append([nextk, stateQ])
        if findmaxstate:
            #tot = 0.0
            for i in range(len(Kmat_coo.row)):
                if Kmat_coo.row[i] == maxind:
                    l = Kmat_coo.col[i]
                    #if l < maxind:
                    if True:
                        nextl = l
                    else:
                        nextl = l - 1
                    for j, (k, nextstateQ) in enumerate(newStateQ):
                        if k == nextl:
                            #nextstateQ += Kmat_coo.data[i] * sig * max_stateQ
                            newStateQ[j][-1] += Kmat_coo.data[i] * sig * max_stateQ
                            break
                    else:
                        newStateQ.append([nextl, Kmat_coo.data[i] * sig * max_stateQ])
        newStateQ_return = []
        for k, nextstateQ in newStateQ:
            if const.stateNmin < nextstateQ:
                newStateQ_return.append([k,nextstateQ])
        newStateQlist.append(newStateQ_return)
    return newStateQlist
    #return returnlist
def calcmatrix(n, dim, maxind, sig, Kmatlist, Kmatdash, const):
    Kmat_coo        = Kmatlist[-1].tocoo()
    Kmatdash_coo    = Kmatdash.tocoo()
    Kmatbefore_row  = []
    Kmatbefore_data = []
    for j in range(len(Kmat_coo.col)):
        if maxind == Kmat_coo.col[j]:
            Kmatbefore_row.append(Kmat_coo.row[j])
            Kmatbefore_data.append(Kmat_coo.data[j])
    Kmatbefore_row  = np.array(Kmatbefore_row, dtype=int)
    Kmatbefore_data = np.array(Kmatbefore_data)
    if const.cythonQ:
        #if const.rank == const.root:
            #print("start const.calcRCMC.calcmatrix", flush = True)
        if const.rank == const.root:
            print("len(Kmatbefore_data) = %s"%len(Kmatbefore_data), flush = True)
        Kmat= const.calcRCMC.calcmatrix(sig,
                Kmatbefore_row,                       Kmatbefore_data,
                Kmatdash_coo.row, Kmatdash_coo.col, Kmatdash_coo.data,
                const.size, const.rank, const.k_min)
        #if const.rank == const.root:
            #print("end const.calcRCMC.calcmatrix", flush = True)
    else:
        Kmat = []
        mpicounter = -1
        for i in range(len(Kmatdash_coo.row)):
            mpicounter += 1
            if mpicounter % const.size != const.rank:
                continue
            k = Kmatdash_coo.row[i]
            l = Kmatdash_coo.col[i]
            for j in range(len(Kmatbefore_row)):
                if k == Kmatbefore_row[j]:
                    K_k_l = Kmatdash_coo.data[i] / (1.0 + sig * Kmatbefore_data[j])
                    if k_min < K_k_l:
                        Kmat.append([k,l, K_k_l])
                    break
    #if const.rank == const.root:
        #print("reconstruct Kmat_return", flush = True)
    Kmatall = None
    Kmatall = const.comm.gather(Kmat,   root = 0)
    Kmatall = const.comm.bcast(Kmatall, root = 0)
    #if const.rank == const.root:
    if True:
        #Kmat_return = Kmatdash
        #Kmat_return += sum(Kmatall)
        #Kmat_return = lil_matrix((dim, dim), dtype = float)
        Kmat_return = Kmatdash
        i = 0
        for Kmat in Kmatall:
            for k,l, sigindex in Kmat:
                Kmat_return[k,l] = sigindex
                i += 1
        #Kmat_return = Kmat_return[k_min <= Kmat_return]

        if const.rank == const.root:
            print("len(Kmatdelta) = %s"%i,flush = True)
        writeline = ""
        for i in range(len(Kmat_coo.row)):
            if i % const.size != const.rank:
                continue
            k = Kmat_coo.row[i]
            l = Kmat_coo.col[i]
            if Kmat_return[k,l] == 0.0:
                writeline += "%6d, %6d, 0.0\n"%(k,l)
        Kmat_r_coo = Kmat_return.tocoo()
        for i in range(len(Kmat_r_coo.row)):
            if i % const.size != const.rank:
                continue
            k = Kmat_r_coo.row[i]
            l = Kmat_r_coo.col[i]
            Kmat_kl = Kmat_r_coo.data[i]
            if Kmat_kl == 0.0:
                #print("%s, %s ->0.0"%(k,l), flush = True)
                #Kmat_return[k,l] = 0
                continue

            if Kmat_r_coo.data[i] != Kmatlist[-1][k,l]:
                if const.deltak_th * Kmatlist[-1][k,l] < Kmatlist[-1][k,l] - Kmat_r_coo.data[i]:
                    writeline += "%6d, %6d, %s\n"%(k,l, Kmat_r_coo.data[i])
    writelineall = const.comm.gather(writeline, root = 0)
    if const.rank == const.root:
        writeline_root = ""
        for writeline_g in writelineall:
            writeline_root += writeline_g
        with open("./RCMCresult/Kmatrix/%05d.csv"%int(n+1), "w") as wf:
            wf.write(writeline_root)
        #print("end reconstruct Kmat_return", flush = True)
    #else:
        #Kmat_return = Kmatlist[-1]
        #for k,l, Kmat_kl in Kmat:
            #Kmat_return[k,l] = Kmat_kl
    #Kmat_return = const.comm.bcast(Kmat_return, const.root = 0)
    return Kmat_return
if __name__ == "__main__":
    const = ConstantsClass()
    const.k_B     = 1.38065e-26
    #const.Temp    = 298
    const.Temp    = 303.15
    const.N_A     = 6.002214e23
    const.betainv = const.Temp * const.k_B * const.N_A # 1/beta (kJ / mol)

    const.jobfilepath = "./jobfiles_meta/"

    const.TSFEmax    = 60.0
    const.EQFEmax    = 40.0
    const.stateQN    = 100
    #const.Pmax       = 1.0e30
    const.Pmax       = 35.0
    const.samePrange = 5.0
    const.k_min      = 1.0e-12 # cut off in order of second
    #const.k_min      = 1.0e-16 # cut off in order of hour
    #const.stateNmin  = 1.0e-20 # 
    const.deltak_th  = 1.0e-3
    const.stateNmin  = 1.0e-5 # 
    const.is_bilayer = True
    const.diffusionQ = True
    #const.sameEQthreshold = [0.1 for _ in range(100)]
    const.sameEQthreshold = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0]
    const.maxtime = 1.0e30
    main(const)
