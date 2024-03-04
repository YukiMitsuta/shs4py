#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020/01/10 MitsutaYuki # # Distributed under terms of the MIT license.

from SHS4py import mkconst, MCPermutation
import constants
class ConstantsClass(): pass
def main():
    const = constants.constClass()
    #const.k_B     = 1.38065e-26 #const.Temp    = 298
    #const.Temp    = 303.15
    #const.Temp    = 300.0
    #const.N_A     = 6.002214e23
    #const.betainv = const.Temp * const.k_B * const.N_A # 1/beta (kJ / mol)
    #const.beta = 1.0/const.betainv

    const.jobfilepath = "./jobfiles_all/"
    const.MarkovNetworkQ = True

    const.TSFEmax    = 1.0e30
    const.EQFEmax    = 30.0
    #const.EQFEmax    = 10.0
    const.stateQN    = 400
    #const.Pmax       = 1.0e30
    const.Pmax       = 36.0
    #const.samePrange = 2.0
    #const.EQFEmax_Prange = 20.0
    const.samePrange = 0.0
    const.EQFEmax_Prange = 1.0e30
    #const.k_min      = 1.0
    #const.k_min      = 1.0e-6 
    #const.k_min      = 1.0e-12 # cut off in order of second
    #const.k_min      = 1.0e-16 # cut off in order of hour
    #const.k_min      = 1.0e-20
    const.k_min      = 0.0
    #const.stateNmin  = 1.0e-20 #
    #const.deltak_th  = 1.0e-16
    const.deltak_th  = 0.0
    #const.stateNmin  = 1.0e-10
    const.stateNmin  = 0.0
    const.is_bilayer = False
    const.diffusionQ = False
    const.oneoutsideEQ = False
    const.asymmetricQ = False
    const.rm_disconnections = False
    const.sameEQthreshold = [0.2 for _ in range(999)]
    #const.sameEQthreshold = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0]
    const.maxtime = 47.0
    const.calc_mpiQ = True
    const.cythonQ  = False
    const = mkconst.main(const)
    from mpi4py import MPI
    const.comm = MPI.COMM_WORLD
    const.rank = const.comm.Get_rank()
    const.size = const.comm.Get_size()
    const.root = 0

    const.k_markov = 100.0
    #const.k_RCMC = 1.0e-5
    #const.k_RCMC = 10.0
    #const.t_delta = 1.0 / const.k_RCMC / 100
    #const.t_delta = 1.0 / const.k_RCMC  * 0.1
    #const.t_delta = 1.0 / const.k_RCMC  * 0.01
    const.t_delta = 0.01
    #const.totaltime =  1000000 #100 mus
    #const.time_stride = 100000
    #MCPermutation.main(const)
    const.totaltime   = 10000000000 #10 ms
    const.time_stride = 1000000
    MCPermutation.main(const)
    #const.totaltime   = 100000000000 #1000 ms
    #const.time_stride = 1000000000
    #MCPermutation.main(const)

if __name__ == "__main__":
    main()
