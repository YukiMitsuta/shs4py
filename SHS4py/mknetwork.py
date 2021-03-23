#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020/08/03 MitsutaYuki 
#
# Distributed under terms of the MIT license.

import os, glob, shutil, sys, re
import copy
import subprocess as sp

import networkx as nx
import numpy as np

from SHS4py import VESanalyzer, mkconst

if True:
    import pyximport  # for cython
    pyximport.install()
    try:
        from . import calcRCMC
    except ImportError:
        import calcRCMC


class pointC():
    def __init__(self, line, z):
        line       = line.split(",")
        self.name  = line[0]
        self.point = np.array(line[1:-1], dtype = float)
        self.z     = z
        self.fe    = float(line[-1])


class constClass():
    def __init__(self):
        self.periodicmax = np.array([ np.pi for _ in range(100)])
        self.periodicmin = np.array([-np.pi for _ in range(100)])
def main():
    """
        
    """
    constC = constClass()
    constC = mkconst.main(constC)
    
    constC.Temp    = 300.0           # tempeture (K)

    
    k_B            = 1.38065e-26      # Boltzmann constant (kJ / K)
    N_A            = 6.002214e23      # Avogadro constant (mol^-1)
    constC.betainv = constC.Temp * k_B * N_A # 1/beta (kJ / mol)
    constC.beta    = 1.0 / constC.betainv

    comm = None
    rank = 0
    root = 0
    size = 1
    plumedpath = './plumed.dat'
    VESclass = VESanalyzer.VESpotential(plumedpath, constC, rank, root, size, comm)
    
    felimit = 99999
    zlist = range(36)
    eqlist_all = []
    tslist_all = []
    connections_all = {}
    exporteqstr = ""
    eqpointindex = 0
    exporttsstr = ""
    tspointindex = 0
    for line in open("./diff_inner.csv"):
        angdiff_inner = float(line)
        break
    for line in open("./diff_outer.csv"):
        angdiff_outer = float(line)
        break

    for z in zlist:
        eqlist, tslist, connections, fricdic = importpoints(z)
        #eqpoint_femin = eqlist[0]
        G = nx.Graph()
        #print(len(set([x[0] for x in connections])))
        for name1, name2 in connections:
            if "EQ" in name1:
                eqname = name1
                tsname = name2
            else:
                eqname = name2
                tsname = name1
            eqpoint = [point for point in eqlist if point.name == eqname]
            if len(eqpoint) == 0:
                print("ERROR; there is not %s in eqlist.csv"%eqname)
                exit()
            eqpoint = eqpoint[0]
            #if felimit < eqpoint.fe - eqpoint_femin.fe:
                #continue
            G.add_edge(name1, name2)
            G.add_edge(name2, name1)
        for eqpoint in eqlist:
            if not eqpoint.name in G.nodes():
                continue
            eqpoint_femin = eqpoint
            break
        #eqlist_connected = []
        #for eqpoint in eqlist:
            #if not eqpoint.name in G.nodes():
                #continue
            #if nx.has_path(G, eqpoint_femin.name, eqpoint.name):
                #eqlist_connected.append(eqpoint)
        #eqlist = eqlist_connected
        tslist_connected = []
        for tspoint in tslist:
            if not tspoint.name in G.nodes():
                continue
            if nx.has_path(G, eqpoint_femin.name, tspoint.name):
                tslist_connected.append(tspoint)
        tslist = tslist_connected
        #exit()
        for eqpoint in eqlist:
            eqpointindex += 1
            eqpoint.networkname = "EQ%05d"%eqpointindex
            exporteqstr += eqpoint.networkname
            for x in eqpoint.point:
                exporteqstr += ", %s"%x
            exporteqstr += ", %s"%eqpoint.z
            exporteqstr += ", %s\n"%eqpoint.fe
        for tspoint in tslist:
            eqpoints = [x for x in G.neighbors(tspoint.name)]
            if len(eqpoints) == 1:
                continue
            if len(eqpoints) != 2:
                print("ERROR; eqpoints = %s"%eqpoints)
                exit()
            eqpoint1 = [point for point in eqlist if point.name == eqpoints[0]][0]
            eqpoint2 = [point for point in eqlist if point.name == eqpoints[1]][0]
            deltafe = tspoint.fe - eqpoint1.fe
            #print(tspoint)
            #print(fricdic.keys())
            k = fricdic[(eqpoint1.name, tspoint.name)] * np.exp(- constC.beta * deltafe)
            #eqedge = (eqpoint1.name, eqpoint2.name)
            eqedge = (eqpoint1.networkname, eqpoint2.networkname)
            if eqedge in connections_all.keys():
                connections_all[eqedge] += k
            else:
                connections_all[eqedge] = k

            deltafe = tspoint.fe - eqpoint2.fe
            k = fricdic[(eqpoint2.name, tspoint.name)] * np.exp(- constC.beta * deltafe)
            eqedge = (eqpoint2.networkname, eqpoint1.networkname)
            if eqedge in connections_all.keys():
                connections_all[eqedge] += k
            else:
                connections_all[eqedge] = k

            tspointindex += 1
            tspoint.networkname = "TS%05d"%tspointindex
            exporttsstr +=  tspoint.networkname
            for x in tspoint.point:
                exporttsstr += ", %s"%x
            exporttsstr += ", %s"%tspoint.z
            exporttsstr += ", %s\n"%tspoint.fe
        zfric = float(open("./jobfiles_%s/fric_z.csv"%z).readline())
        z_before = z - 1.0
        if z < 25.0:
            angfric = angdiff_inner
        else:
            angfric = angdiff_outer
        if z_before < 25.0:
            angfric_before = angdiff_inner
        else:
            angfric_before = angdiff_outer
        #if 0 <= z_before < 35:
        if z != 0:
            connections_all = calcMarcofFric(eqlist, z, zfric,
                    eqlist_before, zfric_before, angfric, angfric_before,
                    connections_all, constC, VESclass)

        eqlist_before = eqlist
        tslist_before = tslist
        connections_before = connections
        zfric_before = zfric
        print("z = %s"%z)
        print("connections_all//Len = %s"%len(connections_all.keys()))
    if not os.path.exists("./jobfiles_all"):
        os.mkdir("./jobfiles_all")
    with open("./jobfiles_all/eqlist.csv", "w") as wf:
        wf.write(exporteqstr)
    with open("./jobfiles_all/tslist.csv", "w") as wf:
        wf.write(exporttsstr)
    Kmatstr = ""
    for edge in connections_all.keys():
        Kmatstr += "%s, %s, %s\n"%(edge[0], edge[1], connections_all[edge])
    with open("./jobfiles_all/Kmatrix.csv", "w") as wf:
        wf.write(Kmatstr)
def importpoints(z):
    eqlist = []
    for line in open("./jobfiles_%s/eqlist.csv"%z):
        if "#" in line:
            continue
        eqpoint = pointC(line, z)
        eqlist.append(eqpoint)
    eqlist.sort(key = lambda point: point.fe)
    tslist = []
    for line in open("./jobfiles_%s/tslist.csv"%z):
        if "#" in line:
            continue
        tspoint = pointC(line, z)
        tslist.append(tspoint)
    tslist.sort(key = lambda point: point.fe)

    connections = []
    for line in open("./jobfiles_%s/connections.csv"%z):
        if "#" in line:
            continue
        line = line.replace("\n","").split(", ")
        connections.append(line)
    fricdic = {}
    for line in open("./jobfiles_%s/Frequencylist.csv"%z):
        if "#" in line:
            continue
        line = line.replace("\n","").split(", ")
        fricdic[(line[0], line[1])] = float(line[2])

    return eqlist, tslist, connections, fricdic
def calcMarcofFric(eqlist, z, zfric, 
        eqlist_before, zfric_before, angfric, angfric_before,
        connections_all, constC, VESclass):
    zfricabs = (zfric + zfric_before) * 0.5
    connectedpointN = 0
    for eqpoint in eqlist:
        #beforeEQ = False
        #beforeEQlen = 1.0e30
        for eqpoint_before in eqlist_before:
            beforepoint = periodicpoint(eqpoint_before.point, constC, eqpoint.point)
            EQlen = np.linalg.norm(beforepoint - eqpoint.point)
            #if EQlen < beforeEQlen:
                #beforeEQlen = EQlen
                #beforeEQ = eqpoint_before
            #if beforeEQlen < 0.5:
            if 0.5 < EQlen:
                continue
            print("%s -> %s; EQlen = %s"%(eqpoint.name, eqpoint_before.name, EQlen))
            connectedpointN += 1
            deltafe = eqpoint_before.fe - eqpoint.fe
            k_z = zfricabs * np.exp(- constC.beta * deltafe * 0.5)
            beforeFE = VESclass.f(np.array(list(eqpoint.point) + [z-1]))
            deltafe =  beforeFE - eqpoint.fe
            k_ang = angfric_before/EQlen/EQlen * np.exp(- constC.beta * deltafe * 0.5)
            #k = (k_z*k_ang)/np.sqrt(k_z*k_z+k_ang*k_ang)
            k = (k_z*k_ang)/(k_z+k_ang)
            eqedge = (eqpoint.networkname, eqpoint_before.networkname)
            if eqedge in connections_all.keys():
                connections_all[eqedge] += k
            else:
                connections_all[eqedge] = k

#    for eqpoint_before in eqlist_before:
#        for eqpoint in eqlist:
#            beforepoint = periodicpoint(eqpoint_before.point, constC, eqpoint.point)
#            EQlen = np.linalg.norm(beforepoint - eqpoint.point)
#            if 0.5 < EQlen:
#                continue
#            print("%s; EQlen = %s"%(eqpoint.name, EQlen))
            deltafe = eqpoint.fe - eqpoint_before.fe
            k_z = zfricabs * np.exp(- constC.beta * deltafe * 0.5)
            pointFE = VESclass.f(np.array(list(eqpoint_before.point) + [z]))
            deltafe = pointFE - eqpoint_before.fe
            k_ang = angfric/EQlen/EQlen * np.exp(- constC.beta * deltafe * 0.5)
            #k = (k_z*k_ang)/np.sqrt(k_z*k_z+k_ang*k_ang)
            k = (k_z*k_ang)/(k_z+k_ang)
            eqedge = (eqpoint_before.networkname, eqpoint.networkname)
            if eqedge in connections_all.keys():
                connections_all[eqedge] += k
            else:
                connections_all[eqedge] = k
    print("connectedpointN = %s"%connectedpointN)
    return connections_all
def periodicpoint(x, const, beforepoint = False):
    """
    periodicpoint: periodic calculation of x
    """
    bdamp = copy.copy(x)
    #if const.periodicQ:
        #if type(const.periodicmax) is float:
    if True:
        #if True:
        if False:
            for i in range(len(x)):
                if beforepoint is False:
                    if x[i] < const.periodicmin or const.periodicmax < x[i]:
                        bdamp[i]  = (x[i] - const.periodicmax) 
                        bdamp[i]  = bdamp[i] % (const.periodicmin - const.periodicmax)
                        bdamp[i] += const.periodicmax
                else:
                    if bdamp[i] < const.periodicmin + beforepoint[i] \
                            or const.periodicmax + beforepoint[i] < bdamp[i]:
                        bdamp[i]  = (x[i] - const.periodicmax - beforepoint[i]) 
                        bdamp[i]  = bdamp[i] % (const.periodicmin - const.periodicmax)
                        bdamp[i] += const.periodicmax + beforepoint[i]
        else:
            for i in range(len(x)):
                if beforepoint is False:
                    if x[i] < const.periodicmin[i] or const.periodicmax[i] < x[i]:
                        bdamp[i]  = (x[i] - const.periodicmax[i]) 
                        bdamp[i]  = bdamp[i] % (const.periodicmin[i] - const.periodicmax[i])
                        bdamp[i] += const.periodicmax[i]
                else:
                    if bdamp[i] < const.periodicmin[i] + beforepoint[i] \
                            or const.periodicmax[i] + beforepoint[i] < bdamp[i]:
                        bdamp[i]  = (x[i] - const.periodicmax[i] - beforepoint[i]) 
                        bdamp[i]  = bdamp[i] % (const.periodicmin[i] - const.periodicmax[i])
                        bdamp[i] += const.periodicmax[i] + beforepoint[i]
    return bdamp

if __name__ == "__main__":
    main()
