#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# # Copyright 2020/01/10 MitsutaYuki # # Distributed under terms of the MIT license.  

import sys
import numpy as np
from SHS4py import mkconst, RCMC, COV2Diff
class ConstantsClass():
    pass
def main():
    const = ConstantsClass()
    const.k_B     = 1.38065e-26
    #const.Temp    = 298
    #const.Temp    = 303.15
    const.Temp    = 300.0
    const.N_A     = 6.002214e23
    const.betainv = const.Temp * const.k_B * const.N_A # 1/beta (kJ / mol)

    const.MarkovNetworkQ = True

    const.TSFEmax    = 1.0e30
    const.EQFEmax    = 1.0e30
    const.stateQN    = 400
    #const.Pmax       = 1.0e30
    const.Pmax       = 36.0
    #const.samePrange = 2.0
    #const.EQFEmax_Prange = 20.0
    const.samePrange = 0.0
    const.EQFEmax_Prange = 1.0e30
    #const.k_min      = 1.0e-12 # cut off in order of second
    #const.k_min      = 1.0e-16 # cut off in order of hour
    const.k_min      = 1.0e-20 
    #const.stateNmin  = 1.0e-20 # 
    #const.deltak_th  = 1.0e-16
    const.deltak_th  = 0.0
    #const.stateNmin  = 1.0e-10
    const.stateNmin  = 0.0
    const.is_bilayer = True
    const.diffusionQ = True
    const.oneoutsideEQ = True
    const.asymmetricQ = True
    const.rm_disconnections = False
    const.sameEQthreshold = [0.2 for _ in range(4)] + [2.0]
    #const.sameEQthreshold = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0]
    const.maxtime = 47.0
    const.calc_mpiQ = True
    const.cythonQ  = True
    const = mkconst.main(const)
    from mpi4py import MPI
    const.comm = MPI.COMM_WORLD
    const.rank = const.comm.Get_rank()
    const.size = const.comm.Get_size()
    const.root = 0

    const.periodicQ = True
    const.periodicmax = np.array([ np.pi for _ in range(4)] + [ 50.0])
    const.periodicmin = np.array([-np.pi for _ in range(4)] + [-50.0])

    const.useZpozitionQ = True
    const.colvarpath = [
#"../VESMW_100ns/run*",
#"../VESMW_200ns/run*",
#"../VESMW_300ns/run*",
#"../VESMW_400ns/run*",
"../VESMW_500ns/run*"]

    #for z in range(36):
        #const.zpozition = z
    args = sys.argv
    const.zpozition = int(sys.argv[1])
    const.jobfilepath = "./jobfiles_%s/"%const.zpozition
    COV2Diff.main(const)

    #const.jobfilepath = "./jobfiles_all/"
    #RCMC.main(const)
if __name__ == "__main__":
    main()
