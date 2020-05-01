#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

import os
import SHS4py
from SHS4py import metaDanalyzer, mkconst
import numpy as np
import random

class constClass():
    pass
def main():
    constC = constClass()
    constC.initialpointN      = 100
    #constC.initialpointN       = 0
    constC.calc_cupyQ          = False # Will you use cupy or not?
    constC.cythonQ             = False # Will you use cython or not?
    constC.calc_mpiQ           = False # Will you use mpi4py or not?
    constC. IOEsphereA_initial = 0.02
    constC.IOEsphereA_dist     = 0.01
    constC.deltas0 = 0.10
    constC.deltas  = 0.05
    #constC.lADDnQ              = True # Will you use l-ADD-n method?
    #constC.IOEl                = 8 # the number of l-ADD-n to find TS point.

    constC.WellTempairedQ = False     # well-tempaired or not?
    Temp           = 298.0            # tempeture (K)
    k_B            = 1.38065e-26      # Boltzmann constant (kJ / K)
    N_A            = 6.002214e23      # Avogadro constant (mol^-1)
    constC.betainv = Temp * k_B * N_A # 1/beta (kJ / mol)
    constC.beta    = 1.0 / constC.betainv

    constC.periodicQ = True # Is the CVs are periodic?
    constC.periodicmax = [ np.pi for _ in range(20)]
    constC.periodicmin = [-np.pi for _ in range(20)]

    constC.x0randomQ = True # Please set True to find ADDs on each SH.
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
        if not os.path.exists("./jobfiles_meta"):
            os.mkdir("./jobfiles_meta")
    constC = mkconst.main(constC)
    if rank == root:
        with open("./jobfiles_meta/constants.txt", "w") as wf:
            wf.write(constC.writestr)

    hillpath = './HILLS'
    metaD  = SHS2py.metaDanalyzer.Metad_result(hillpath, constC)
    initialpointlist = []
    #for line in open('./COLVAR'):
        #if '#' in line:
            #continue
        #x = line.replace('\n','').split(' ')
        #initialpointlist.append(np.array(x[2:], dtype = float))
    #initialpointlist = random.sample(initialpointlist, k = constC.initialpointN)

    if constC.initialpointN == 0:
        initialpointlist_gather = []
    else:
        if size != 1:
            hillD = constC.initialpointN // size
            if hillD != 0:
                hillD = len(metaD.hillCs) // (constC.initialpointN // size)
            #hillD = len(metaD.hillCs) // size
        else:
            hillD = len(metaD.hillCs) // (constC.initialpointN)
            #hillD = len(metaD.hillCs) // size
        initialpointlist = []
        if hillD != 0:
            for hillC in metaD.hillCs[::hillD]:
                initialpointlist.append(hillC.s)
        if size != 1:
            initialpointlist= comm.gather(initialpointlist, root=0)
            if rank == root:
                initialpointlist_gather = []
                for initialpointlist_chunk in initialpointlist:
                    initialpointlist_gather.extend(initialpointlist_chunk)
            else:
                initialpointlist_gather = None
            initialpointlist_gather = comm.bcast(initialpointlist_gather, root = 0)
        else:
            initialpointlist_gather = initialpointlist
    for line in open("./plumed.dat"):
        if "HEIGHT" in line:
            line = line.split("=")
            height = line[-1].replace("\n","")
    SHS2py.SHSearch(metaD.f, metaD.grad, metaD.hessian,
            importinitialpointQ = False, initialpoints = initialpointlist_gather,
            SHSrank = rank, SHSroot = root, SHScomm = comm,
            optdigTH = - constC.digThreshold * float(height), const = constC)
if __name__ == "__main__":
    main()
