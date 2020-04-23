#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright  2019.12.19 Yuki Mitsuta
# Distributed under terms of the MIT license.

import SHS4py
from SHS4py import metaDanalyzer, mkconst

class constClass():
    pass
def main():
    """
    template of the calculation of Metadynamics
    """
    constC = constClass()

    Temp           = 298.0            # tempeture (K) : must same with that of MD simulation
    k_B            = 1.38065e-26      # Boltzmann constant (kJ / K)
    N_A            = 6.002214e23      # Avogadro constant (mol^-1)
    constC.betainv = Temp * k_B * N_A # 1/beta (kJ / mol)
    constC.beta    = 1.0 / constC.betainv

    constC.threshold          = 0.1
    constC.sameEQthreshold    = 0.05
    constC.IOEsphereA_initial = 0.02
    constC.IOEsphereA_dist    = 0.005
    constC.deltas0            = 0.01
    constC.deltas             = 0.005

    constC.initialpointN      = 100

    constC.periodicQ = False
    #If CVs have periodic(like as angles or dihedrals), periodicQ must be True and define periodicmin and periodicmax.
    #For example, if you use dihedrals as CVs:
    #constC.periodicQ   =  True
    #constC.periodicmin = -np.pi
    #constC.periodicmax =  np.pi

    constC.calc_mpiQ  = False # package mpi4py is necessary
    constC.calc_cupyQ = False # package cupy   is necessary
    constC.cythonQ    = False # package cython and previous compile are necessary
    #If the trajectory is long, I recommend you to use mpi4py & cupy & cython with GPGPU machine.

    constC.WellTempairedQ = True
    constC.WT_Biasfactor  = 6 #BIASFACTOR of the PLUMED setting.

    constC.digThreshold = 5
    height = 0.5 #HEIGHT of the PLUMED setting.
    digTH  = - height * constC.digThreshold

    constC = mkconst.main(constC)
    
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

    metaD  = SHS4py.metaDanalyzer.Metad_result("./HILLS", constC)

    if size != 1:
        hillD = len(metaD.hillCs) // (constC.initialpointN // size)
    else:
        hillD = len(metaD.hillCs) // (constC.initialpointN)
    initialpointlist = []
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

    SHS4py.SHSearch(metaD.f, metaD.grad, metaD.hessian,
            importinitialpointQ = False, initialpoints = initialpointlist_gather, 
            SHSrank = rank, SHSroot = root, SHScomm = comm, optdigTH = digTH, const = constC)

if __name__ == "__main__":
    main()
