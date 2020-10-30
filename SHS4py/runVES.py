#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

import os, glob
import SHS4py
from SHS4py import VESanalyzer, mkconst
import numpy as np
import random
import sys

class constClass():
    pass
def main():
    constC = constClass()
    
    constC.Temp    = 300.0           # tempeture (K)

    
    k_B            = 1.38065e-26      # Boltzmann constant (kJ / K)
    N_A            = 6.002214e23      # Avogadro constant (mol^-1)
    constC.betainv = constC.Temp * k_B * N_A # 1/beta (kJ / mol)
    constC.beta    = 1.0 / constC.betainv
    constC.initialpointN      = 1000
    #constC.initialpointN       = 0
    constC.cythonQ             = True
    #constC.calc_mpiQ           = True
    constC.calc_mpiQ           = False
    constC.sameEQthreshold = [ 0.2 for _ in range(4)] + [ 2.0]
    constC.IOEsphereA_initial = 0.10
    constC.IOEsphereA_dist     = 0.02
    constC.deltas0 = 0.20
    constC.deltas  = 0.01

    constC.coeffPickNlist                = []
#! SET iteration  400000
    #constC.coeffabsmin  = 1.0e-3

    constC.periodicQ = True
    constC.periodicmax = [ np.pi for _ in range(4)] + [ 50.0]
    constC.periodicmin = [-np.pi for _ in range(4)] + [-50.0]

    constC.wallmax = [ 1.0e30 for _ in range(4)] + [ 50.0]
    constC.wallmin = [-1.0e30 for _ in range(4)] + [-50.0]

    constC.EQwallmax = [ 1.0e30 for _ in range(4)] + [35.0]
    constC.EQwallmin = [-1.0e30 for _ in range(4)] + [-35.0]

    constC.abslist = [False for _ in range(4)] + [True]

    constC.x0randomQ = True

    constC.gridQ = True
    constC.grid_importQ = True
    constC.grid_min = [-np.pi for _ in range(4)] + [-50.0]
    constC.grid_max = [ np.pi for _ in range(4)] + [50.0]
    constC.grid_bin = [   100 for _ in range(4)] + [500]

    args = sys.argv
    currentpoint = sys.argv[2]
    #currentpoint = 35

    constC.CVfixQ = True
    constC.fixlist = [(False, None)] * 4 + [(True, float(currentpoint))]

    constC.jobfilepath = "./jobfiles_%s/"%currentpoint
    if constC.calc_mpiQ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0
    else:
        comm = None
        rank = 0
        root = 0
        size = 1

    if rank == root:
        if not os.path.exists(constC.jobfilepath):
            os.mkdir(constC.jobfilepath)
    constC = mkconst.main(constC)
    if rank == root:
        with open("./%s/constants.txt"%constC.jobfilepath, "w") as wf:
            wf.write(constC.writestr)

    plumedpath = './plumed.dat'
    VESclass = SHS4py.VESanalyzer.VESpotential(plumedpath, constC, rank, root, size, comm)

    initialpointlist = []
    initialpointTSlist = []
    initialpointlist, initialpointTSlist = importinitialpoint(constC)
    initialpointlist += importinitialpoint_random(constC)

    SHS4py.SHSearch(VESclass.f, VESclass.grad, VESclass.hessian,
            importinitialpointQ = False, initialpoints = initialpointlist, initialTSpoints = initialpointTSlist,
            SHSrank = rank, SHSroot = root, SHSsize = size, SHScomm = comm,
            const = constC)
def importinitialpoint_random(constC):
    initialpointlist = []
    for collname in glob.glob("../VESMW_*/run*/COLVAR.*"):
        for line in open(collname):
            if '#' in line:
                continue
            #x = line.replace('\n','').split(' ')
            #initialpointlist.append(np.array(x[2:], dtype = float))
            x = line.replace('\n','').split(' ')[2:]
            if constC.CVfixQ:
                initialpoint = []
                x_index = 0
                for fixQ, fixX in constC.fixlist:
                    if fixQ:
                        continue
                    initialpoint.append(x[x_index])
                    x_index += 1
            else:
                initialpoint = x
            initialpointlist.append(np.array(initialpoint, dtype = float))
    initialpointlist = random.sample(initialpointlist, k = constC.initialpointN)
    return initialpointlist
def importinitialpoint(constC):
    args = sys.argv
    beforepoint = int(args[2]) + 1
    initialpointlist = []
    #eqpath = "/home/mitsutay/gromacsdir/VES/1plx_POPC/VES12A_MW2/run0/jobfiles_meta"
    #eqpath = "./jobfiles_meta.back3"
    #eqpath = "/home/mitsutay/gromacsdir/VES/1plx_POPC/VES13A_MW200ns/run0/jobfiles_meta"
    eqpath = "./jobfiles_%s"%beforepoint
    connectionEQ = []
    connectionTS = []
    for line in open("%s/connections.csv"%eqpath):
        line = line.split(", ")
        connectionTS.append(line[0])
        connectionEQ.append(line[1].replace("\n",""))
    connectionEQ = set(connectionEQ)
    connectionTS = set(connectionTS)
    initialfelist = []
    for line in open("%s/eqlist.csv"%eqpath):
        if "#" in line:
            continue
        line = line.split(",")
        if not line[0] in connectionEQ:
            continue
        initialpoint = np.array(line[1:-1], dtype = float)
        for i in range(len(initialpoint)):
            if initialpoint[i] < constC.EQwallmin[i] or constC.EQwallmax[i] < initialpoint[i]:
                break
        else:
            initialfelist.append(float(line[-1]))
    counter = -1
    minimumFE = min(initialfelist)
    for line in open("%s/eqlist.csv"%eqpath):
        if "#" in line:
            continue
        counter += 1
        if counter % 9 != int(args[1]):
            continue
        line = line.split(",")
        if not line[0] in connectionEQ:
            continue
        initialpoint = np.array(line[1:-1], dtype = float)
        for i in range(len(initialpoint)):
            if initialpoint[i] < constC.EQwallmin[i] or constC.EQwallmax[i] < initialpoint[i]:
                break
        else:
            #if float(line[-1]) - minimumFE < 30.0:
            if True:
                initialpointlist.append(initialpoint)
    initialpointTSlist = []
    for line in open("%s/tslist.csv"%eqpath):
        if "#" in line:
            continue
        counter += 1
        if counter % 9 != int(args[1]):
            continue
        line = line.split(",")
        if not line[0] in connectionTS:
            continue
        initialpoint = np.array(line[1:-1], dtype = float)
        for i in range(len(initialpoint)):
            if initialpoint[i] < constC.EQwallmin[i] or constC.EQwallmax[i] < initialpoint[i]:
                break
        else:
            #if float(line[-1]) - minimumFE < 30.0:
            if True:
                initialpointTSlist.append(np.array(line[1:-1], dtype = float))
    return initialpointlist, initialpointTSlist
if __name__ == "__main__":
    main()
