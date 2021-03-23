#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

import os
import SHS4py
from SHS4py import VESanalyzer, mkconst
import numpy as np
import random

class constClass():
    pass
def main():
    constC = constClass()
    #constC.initialpointN      = 100
    constC.initialpointN       = 0
    #constC.calc_cupyQ          = True
    constC.cythonQ             = True
    constC.calc_mpiQ           = True
    constC.use_jacQ            = True
    constC. IOEsphereA_initial = 0.02
    constC.IOEsphereA_dist     = 0.01
    constC.deltas0 = 0.10
    constC.deltas  = 0.05
    constC.lADDnQ              = True
    constC.IOEl                = 8

    constC.coeffPickNlist                = [300000]
    #constC.coeffabsmin  = 1.0e-2

    constC.periodicQ = True
    constC.periodicmax = [ np.pi for _ in range(20)]
    constC.periodicmin = [-np.pi for _ in range(20)]

    constC.x0randomQ = True
    #constC.x0randomQ = False
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

    plumedpath = './plumedVES.dat'
    VESclass = SHS4py.VESanalyzer.VESpotential(plumedpath, constC, rank, root, size, comm)
    initialpointlist = []
#    for line in open('./COLVAR'):
#        if '#' in line:
#            continue
#        x = line.replace('\n','').split(' ')
#        #print(x)
#        initialpointlist.append(np.array(x[2:], dtype = float))
#    initialpointlist = random.sample(initialpointlist, k = constC.initialpointN)
    #print(initialpointlist)

    #for line in open("../VES2D_200ns/jobfiles_meta/eqlist.csv"):
    #for line in open("./jobfiles_meta.back1/eqlist.csv"):
        #if "#" in line:
            #continue
        #line = line.split(",")
        #initialpointlist.append(np.array(line[1:-1], dtype = float))
    #print(initialpointlist[0])


    SHS4py.SHSearch(VESclass.f, VESclass.grad, VESclass.hessian,
            importinitialpointQ = False, initialpoints = initialpointlist, 
            SHSrank = rank, SHSroot = root, SHSsize = size, SHScomm = comm,
            const = constC)
if __name__ == "__main__":
    main()
