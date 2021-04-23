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
class EQclass():
    def __init__(self,line):
        line = line.split(",")
        self.name = line[0]
        self.point = np.array(line[1:5], dtype=float)
        self.cv = np.array(line[1:5], dtype=float)
        self.z = int(line[-2])
        self.fe = float(line[-1])

def main():
    constC = constClass()

    constC.Temp    = 300.0           # tempeture (K)

    k_B            = 1.38065e-26      # Boltzmann constant (kJ / K)
    N_A            = 6.002214e23      # Avogadro constant (mol^-1)
    constC.betainv = constC.Temp * k_B * N_A # 1/beta (kJ / mol)
    constC.beta    = 1.0 / constC.betainv
    #constC.initialpointN      = 100
    constC.initialpointN       = 0
    #constC.calc_cupyQ          = True
    constC.cythonQ             = True
    constC.use_jacQ            = True
    constC.threshold = 1.0
    constC.sameEQthreshold = [ 0.2 for _ in range(4)] + [ 2.0]
    constC.IOEsphereA_initial = 0.05
    constC.IOEsphereA_dist    = 0.01
    constC.deltas0 = 0.05
    constC.deltas  = 0.01

    constC.coeffPickNlist                = []
#! SET iteration  400000
    #constC.coeffabsmin  = 1.0e-3

    constC.periodicQ = True
    constC.periodicmax = [ np.pi for _ in range(4)] + [ 40.0]
    constC.periodicmin = [-np.pi for _ in range(4)] + [-40.0]

    constC.wallmax = [ 1.0e30 for _ in range(4)] + [ 40.0]
    constC.wallmin = [-1.0e30 for _ in range(4)] + [-40.0]

    constC.EQwallmax = [ 1.0e30 for _ in range(4)] + [35.0]
    constC.EQwallmin = [-1.0e30 for _ in range(4)] + [-35.0]

    constC.abslist = False

    constC.x0randomQ = False
    constC.chkBifurcationQ = False
    constC.chkinitialTSQ = True


    constC.gridQ = False
    constC.grid_importQ = False

    constC.CVfixQ = False

    constC.calc_mpiQ = False
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

    #if rank == root:
        #if not os.path.exists(constC.jobfilepath):
            #os.mkdir(constC.jobfilepath)
    constC = mkconst.main(constC)
    plumedpath = './plumed.dat'
    VESclass = SHS4py.VESanalyzer.VESpotential(plumedpath, constC, rank, root, size, comm)
    eqlist_all = []
    for line in open("./jobfiles_all/eqlist.csv"):
        if  "#" in line:
            continue
        eqpoint = EQclass(line)
        eqlist_all.append(eqpoint)
    for z in range(36):
        eqlist_z = [eqpoint for eqpoint in eqlist_all if eqpoint.z == z]
        femin = 1.0e30
        for eqpoint in eqlist_z:
            if eqpoint.fe < femin:
                femin = eqpoint.fe
                targetpoint = eqpoint.cv
        writeline = ""
        for z_pmf in range(36):
            p = np.array(list(targetpoint)+[z_pmf])
            print("p = %s"%p)
            fe =VESclass.f(p)
            writeline += "%s, %s\n"%(z_pmf, fe)
        #print(writeline)
        #exit()
        csvpath = "./jobfiles_all/PMF_%s.csv"%z
        with open(csvpath,"w") as wf:
            wf.write(writeline)
        print("%s is writen"%csvpath)

        

if __name__ == "__main__":
    main()
    
